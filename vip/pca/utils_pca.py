#! /usr/bin/env python

"""
Module with helping functions.
"""

from __future__ import division
from __future__ import print_function

__author__ = 'C. Gomez @ ULg'
__all__ = ['matrix_scaling',
           'prepare_matrix',
           'reshape_matrix',
           'svd_wrapper',
           'pca_annulus',
           'scale_cube_for_pca']

import numpy as np
from numpy import linalg
from matplotlib import pyplot as plt 
from scipy.sparse.linalg import svds
from sklearn.decomposition import randomized_svd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import scale #,minmax_scale
from ..var import mask_circle, get_annulus, get_square_robust, frame_center
from ..preproc import cube_derotate, cube_collapse, cube_rescaling


def scale_cube_for_pca(cube,scal_list, full_output=True, inverse=False, y_in=1,
                       x_in=1):
    """
    Wrapper to scale or descale a cube by factors given in scal_list, without 
    any loss of information (zero-padding if scaling > 1).
    Important: in case of ifs data, the scaling factors in var_list should be 
    >= 1 (ie. provide the scaling factors as for scaling to the longest 
    wavelength channel).

    Parameters:
    -----------
    cube: 3D-array
       Datacube that whose frames have to be rescaled.
    scal_list: 1D-array
       Vector of same dimension as the first dimension of datacube, containing 
       the scaling factor for each frame.
    full_output: bool, {True,False}, optional
       Whether to output just the rescaled cube (False) or also its median, 
       the new y and x shapes of the cube, and the new centers cy and cx of the 
       frames (True).
    inverse: bool, {True,False}, optional
       Whether to inverse the scaling factors in scal_list before applying them 
       or not; i.e. True is to descale the cube (typically after a first scaling
       has already been done)
    y_in, x-in:
       Initial y and x sizes.
       In case the cube is descaled, these values will be used to crop back the
       cubes/frames to their original size.

    Returns:
    --------
    frame: 2D-array
        The median of the rescaled cube.
    If full_output is set to True, the function returns:
    cube,frame,y,x,cy,cx: 3D-array,2D-array,int,int,int,int
        The rescaled cube, its median, the new y and x shapes of the cube, and 
        the new centers cy and cx of the frames
    """
    #First pad the cube with zeros appropriately to not loose info when scaling
    # the cube.
    # TBD next: pad with random gaussian noise instead of zeros. Padding with 
    # only zeros can make the svd not converge in a pca per zone.

    n, y, x = cube.shape

    max_sc = np.amax(scal_list)

    if not inverse and max_sc > 1:
        new_y = int(np.ceil(max_sc*y))
        new_x = int(np.ceil(max_sc*x))
        if (new_y - y)%2 != 0: new_y = new_y+1
        if (new_x - x)%2 != 0: new_x = new_x+1
        pad_len_y = (new_y - y)//2
        pad_len_x = (new_x - x)//2
        big_cube = np.pad(cube, ((0,0), (pad_len_y, pad_len_y), 
                                 (pad_len_x, pad_len_x)), 'constant', 
                          constant_values=(0,))
    else: 
        big_cube = cube.copy()

    n, y, x = big_cube.shape
    cy,cx = frame_center(big_cube[0])
    var_list = scal_list

    if inverse:
        var_list = 1./scal_list[:]
        cy,cx = frame_center(cube[0])

    # (de)scale the cube, so that a planet would now move radially
    cube,frame = cube_rescaling(big_cube,var_list,ref_y=cy, ref_x=cx)

    if inverse:
        if max_sc > 1:
            frame = get_square_robust(frame,max(y_in,x_in), cy,cx,strict=False)
            if full_output:
                n_z = cube.shape[0]
                array_old = cube.copy()
                cube = np.zeros([n_z,max(y_in,x_in),max(y_in,x_in)])
                for zz in range(n_z):
                    cube[zz]=get_square_robust(array_old[zz],max(y_in,x_in), 
                                               cy,cx,strict=False)


    if full_output: 
        return cube,frame,y,x,cy,cx
    else: 
        return frame


def pca_annulus(cube, angs, ncomp, annulus_width, r_guess, cube_ref=None,
                svd_mode='lapack', scaling=None, collapse='median'):
    """
    PCA process the cube only for an annulus of a given width and at a given
    radial distance to the frame center. It returns a PCA processed frame with 
    only non-zero values at the positions of the annulus.
    
    Parameters
    ----------
    cube : array_like
        The cube of fits images expressed as a numpy.array.
    angs : array_like
        The parallactic angles expressed as a numpy.array.
    ncomp : int
        The number of principal component.
    annulus_width : float
        The annulus width in pixel on which the PCA is performed.
    r_guess : float
        Radius of the annulus in pixels.
    cube_ref : array_like, 3d, optional
        Reference library cube. For Reference Star Differential Imaging.
    svd_mode : {'lapack', 'randsvd', 'eigen', 'arpack'}, str optional
        Switch for different ways of computing the SVD and selected PCs.
    scaling : {None, 'temp-mean', 'spat-mean', 'temp-standard', 'spat-standard'}
        With None, no scaling is performed on the input data before SVD. With
        "temp-mean" then temporal px-wise mean subtraction is done, with
        "spat-mean" then the spatial mean is subtracted, with "temp-standard"
        temporal mean centering plus scaling to unit variance is done and with
        "spat-standard" spatial mean centering plus scaling to unit variance is
        performed.
    collapse : {'median', 'mean', 'sum', 'trimmean', None}, str or None, optional
        Sets the way of collapsing the frames for producing a final image. If
        None then the cube of residuals is returned.
    
    Returns
    -------
    Depending on ``collapse`` parameter a final collapsed frame or the cube of
    residuals is returned.
    """
    data, ind = prepare_matrix(cube, scaling, mode='annular', annulus_radius=r_guess,
                               annulus_width=annulus_width, verbose=False)
    yy, xx = ind

    if cube_ref is not None:
        data_svd, _  = prepare_matrix(cube_ref, scaling, mode='annular',
                                      annulus_radius=r_guess,
                                      annulus_width=annulus_width, verbose=False)
    else:
        data_svd = data
        
    V = svd_wrapper(data_svd, svd_mode, ncomp, debug=False, verbose=False)
        
    transformed = np.dot(data, V.T)
    reconstructed = np.dot(transformed, V)                           
    residuals = data - reconstructed
    cube_zeros = np.zeros_like(cube)
    cube_zeros[:, yy, xx] = residuals
    cube_res_der = cube_derotate(cube_zeros, angs)
    if collapse is not None:
        pca_frame = cube_collapse(cube_res_der, mode=collapse)
        return pca_frame
    else:
        return cube_res_der



def matrix_scaling(matrix, scaling):
    """ Scales a matrix using sklearn.preprocessing.scale function.

    scaling : {None, 'temp-mean', 'spat-mean', 'temp-standard', 'spat-standard'}
    With None, no scaling is performed on the input data before SVD. With
    "temp-mean" then temporal px-wise mean subtraction is done, with
    "spat-mean" then the spatial mean is subtracted, with "temp-standard"
    temporal mean centering plus scaling to unit variance is done and with
    "spat-standard" spatial mean centering plus scaling to unit variance is
    performed.
    """
    if scaling==None:
        pass
    elif scaling=='temp-mean':
        matrix = scale(matrix, with_mean=True, with_std=False)
    elif scaling=='spat-mean':
        matrix = scale(matrix, with_mean=True, with_std=False, axis=1)
    elif scaling=='temp-standard':
        matrix = scale(matrix, with_mean=True, with_std=True)
    elif scaling=='spat-standard':
        matrix = scale(matrix, with_mean=True, with_std=True, axis=1)
    else:
        raise ValueError('Scaling mode not recognized')
    
    return matrix 


def prepare_matrix(array, scaling=None, mask_center_px=None, mode='fullfr',
                   annulus_radius=None, annulus_width=None, verbose=True):
    """ Builds the matrix for the SVD/PCA and other matrix decompositions, 
    centers the data and masks the frames central area if needed.
    
    Parameters
    ----------
    array : array_like
        Input cube, 3d array.
    scaling : {None, 'temp-mean', 'spat-mean', 'temp-standard', 'spat-standard'}
        With None, no scaling is performed on the input data before SVD. With 
        "temp-mean" then temporal px-wise mean subtraction is done, with 
        "spat-mean" then the spatial mean is subtracted, with "temp-standard" 
        temporal mean centering plus scaling to unit variance is done and with
        "spat-standard" spatial mean centering plus scaling to unit variance is
        performed.  
    mask_center_px : None or Int, optional
        Whether to mask the center of the frames or not.
    mode : {'fullfr', 'annular'}
        Whether to use the whole frames or a single annulus.
    annulus_radius : float
        Distance in pixels from the center of the frame to the center of the
        annulus.
    annulus_width : float
        Width of the annulus in pixels.
    verbose : {True, False}, bool optional
        If True prints intermediate info and timing.
    
    Returns
    -------
    If mode is `annular` then the indices of the annulus (yy, xx) are returned
    along with the matrix.

    matrix : array_like
        Out matrix whose rows are vectorized frames from the input cube.
    
    """
    if mode == 'annular':
        if annulus_radius is None or annulus_width is None:
            msgerr = 'Annulus_radius and/or annulus_width can be None in annular '
            msgerr += 'mode'
            raise ValueError(msgerr)

        ind = get_annulus(array[0], annulus_radius - annulus_width / 2.,
                          annulus_width, output_indices=True)
        yy, xx = ind
        matrix = array[:, yy, xx]

        matrix = matrix_scaling(matrix, scaling)

        if verbose:
            msg = 'Done vectorizing the cube annulus. Matrix shape [{:},{:}]'
            print(msg.format(matrix.shape[0], matrix.shape[1]))
        return matrix, ind

    elif mode == 'fullfr':
        if mask_center_px:
            array = mask_circle(array, mask_center_px)

        nfr = array.shape[0]
        matrix = np.reshape(array, (nfr, -1))  # == for i: array[i].flatten()

        matrix = matrix_scaling(matrix, scaling)

        if verbose:
            msg = 'Done vectorizing the frames. Matrix shape [{:},{:}]'
            print(msg.format(matrix.shape[0], matrix.shape[1]))
        return matrix
     

def reshape_matrix(array, y, x):
    """ Converts a matrix whose rows are vectorized frames to a cube with 
    reshaped frames.
    """
    return array.reshape(array.shape[0], y, x)


def svd_wrapper(matrix, mode, ncomp, debug, verbose, usv=False):
    """ Wrapper for different SVD libraries with the option of showing the 
    cumulative explained variance ratio.
    
    Note:
    ----
    Sklearn.PCA deprecated as it uses linalg.svd(X, full_matrices=False) under 
    the hood, which is already included.
    Sklearn.RandomizedPCA deprecated as it uses sklearn.randomized_svd which is
    already included.
    
    """
    if not matrix.ndim==2:
        raise TypeError('Input matrix is not a 2d array')
    
    def reconstruction(ncomp, U, S, V, var=1): 
        if mode=='lapack':
            rec_matrix = np.dot(U[:,:ncomp], np.dot(np.diag(S[:ncomp]), V[:ncomp]))
            rec_matrix = rec_matrix.T
            print('  Matrix reconstruction with {:} PCs:'.format(ncomp))
            print('  Mean Absolute Error =', mean_absolute_error(matrix, 
                                                                 rec_matrix))
            print('  Mean Squared Error =', mean_squared_error(matrix,rec_matrix))
            
            exp_var = S**2
            full_var = np.sum(S**2)
            explained_variance_ratio = exp_var / full_var        # % of variance explained by each PC
            ratio_cumsum = np.cumsum(explained_variance_ratio)
        elif mode=='eigen':
            exp_var = S**2                                       # squared because we previously took the sqrt of the EVals
            full_var = np.sum(S**2)
            explained_variance_ratio = exp_var / full_var        # % of variance explained by each PC
            ratio_cumsum = np.cumsum(explained_variance_ratio)
        else:
            rec_matrix = np.dot(U, np.dot(np.diag(S), V))
            print('  Matrix reconstruction MAE =', mean_absolute_error(matrix, 
                                                                       rec_matrix))
            exp_var = (S**2) / matrix.shape[0]
            full_var = np.var(matrix, axis=0).sum()
            explained_variance_ratio = exp_var / full_var        # % of variance explained by each PC       
            if var==1:  pass    
            else:  explained_variance_ratio = explained_variance_ratio[::-1]
            ratio_cumsum = np.cumsum(explained_variance_ratio)
            msg = '  This info makes sense when the matrix is mean centered '
            msg += '(temp-mean scaling)'
            print (msg)
        
        lw = 2
        alpha = 0.4
        fig = plt.figure(figsize=(6,3))
        fig.subplots_adjust(wspace=0.4)
        ax1 = plt.subplot2grid((1,3), (0,0), colspan=2)
        ax1.step(range(explained_variance_ratio.shape[0]), 
                 explained_variance_ratio, alpha=alpha, where='mid', 
                 label='Individual EVR', lw=lw)
        ax1.plot(ratio_cumsum, '.-', alpha=alpha, 
                 label='Cumulative EVR', lw=lw)
        ax1.legend(loc='best', frameon=False, fontsize='medium')
        ax1.set_ylabel('Explained variance ratio (EVR)')
        ax1.set_xlabel('Principal components')
        ax1.grid(linestyle='solid', alpha=0.2)
        ax1.set_xlim(-10, explained_variance_ratio.shape[0]+10)
        ax1.set_ylim(0, 1)
        
        trunc = 20
        ax2 = plt.subplot2grid((1,3), (0,2), colspan=1)
        #plt.setp(ax2.get_yticklabels(), visible=False)
        ax2.step(range(trunc), explained_variance_ratio[:trunc], alpha=alpha, 
                 where='mid', lw=lw)
        ax2.plot(ratio_cumsum[:trunc], '.-', alpha=alpha, lw=lw)
        ax2.set_xlabel('Principal components')
        ax2.grid(linestyle='solid', alpha=0.2)
        ax2.set_xlim(-2, trunc+2)
        ax2.set_ylim(0, 1)
        
        msg = '  Cumulative explained variance ratio for {:} PCs = {:.5f}'
        #plt.savefig('figure.pdf', dpi=300, bbox_inches='tight')
        print(msg.format(ncomp, ratio_cumsum[ncomp-1]))
        
    if ncomp>min(matrix.shape[0],matrix.shape[1]):
        msg = '{:} PCs can be obtained from a matrix with size [{:},{:}].'
        msg += ' Increase the size of the patches or decrease the number of'
        msg += ' principal components.'
        raise RuntimeError(msg.format(ncomp, matrix.shape[0], matrix.shape[1]))
        
    if mode=='eigen':
        # in our data n_frames is always smaller than n_pixels. In this setting
        # by taking the covariance as np.dot(matrix.T,matrix) we get all 
        # (n_pixels) eigenvectors but it is much slower and takes more memory 
        M = np.dot(matrix, matrix.T)                             # covariance matrix
        e, EV = linalg.eigh(M)                                   # eigenvalues and eigenvectors
        pc = np.dot(EV.T, matrix)                                # PCs using a compact trick when cov is MM'
        V = pc[::-1]                                             # reverse since last eigenvectors are the ones we want 
        S = np.sqrt(e)[::-1]                                     # reverse since eigenvalues are in increasing order 
        if debug: reconstruction(ncomp, None, S, None)
        for i in range(V.shape[1]):
            V[:,i] /= S                                          # scaling by the square root of eigenvalues
        V = V[:ncomp]
        if verbose: print('Done PCA with numpy linalg eigh functions')
        
    elif mode=='lapack':
        # in our data n_frames is always smaller than n_pixels. In this setting
        # taking the SVD of M' and keeping the left (transposed) SVs is faster
        # than taking the SVD of M and taking the right ones
        U, S, V = linalg.svd(matrix.T, full_matrices=False)         
        if debug: reconstruction(ncomp, U, S, V)
        V = V[:ncomp]                                           # we cut projection matrix according to the # of PCs               
        U = U[:,:ncomp]
        S = S[:ncomp]
        if verbose: print('Done SVD/PCA with numpy SVD (LAPACK)')
            
    elif mode=='arpack':
        U, S, V = svds(matrix, k=ncomp) 
        if debug: reconstruction(ncomp, U, S, V, -1)
        if verbose: print('Done SVD/PCA with scipy sparse SVD (ARPACK)')

    elif mode=='randsvd':
        U, S, V = randomized_svd(matrix, n_components=ncomp, n_iter=2, 
                                 transpose='auto', random_state=None)
        if debug: reconstruction(ncomp, U, S, V)
        if verbose: print('Done SVD/PCA with randomized SVD')

    else:
        raise TypeError('The SVD mode is not available')
            
    if usv:
        if mode=='lapack':
            return U.T, S, V.T
        else:
            return U, S, V
    else:
        if mode=='lapack':
            return U.T
        else:
            return V


