#! /usr/bin/env python

"""
Module with helping functions.
"""

from __future__ import division
from __future__ import print_function

__author__ = 'C. Gomez @ ULg'
__all__ = ['prepare_matrix',
           'reshape_matrix',
           'svd_wrapper',
           'pca_annulus',
           'scale_cube_for_pca']

import cv2
import numpy as np
from scipy import linalg
from scipy.sparse.linalg import svds
from sklearn.decomposition import randomized_svd
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import scale
from ..var import mask_circle, get_annulus, get_square_robust, frame_center
from ..calib import cube_derotate, cube_rescaling


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
        new_y = np.ceil(max_sc*y)
        new_x = np.ceil(max_sc*x)
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
                svd_mode='randsvd'):
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
    svd_mode : {'randsvd', 'eigen', 'lapack', 'arpack', 'opencv'}, str optional
        Switch for different ways of computing the SVD and selected PCs.
    
    Returns
    -------
    out: array_like
        The annulus PCA-ed frame.
        
    """
    # annulus_width is divided by 2 to be sure that the radial distance r_guess 
    # lies at the center of the annulus. If we don't divide, than r_guess 
    # corresponds to the outer radius of the annulus.
    indic = get_annulus(cube[0], r_guess-annulus_width/2., annulus_width,      
                        output_indices=True)                                   
    yy, xx = indic                                                             
    
    data = cube[:, yy, xx]
    #data = matrix_ann - matrix_ann.mean(axis=0) # centering?
    
    if cube_ref is not None:
        data_svd = cube_ref[:, yy, xx]
    else:
        data_svd = data
        
    V = svd_wrapper(data_svd, svd_mode, ncomp, debug=False, verbose=False)
        
    transformed = np.dot(data, V.T)
    reconstructed = np.dot(transformed, V)                           
    residuals = data - reconstructed
    cube_zeros = np.zeros_like(cube)
    cube_zeros[:, yy, xx] = residuals
    _, pca_frame = cube_derotate(cube_zeros, angs)
    return pca_frame  



def prepare_matrix(array, scaling='mean', mask_center_px=None, verbose=True):
    """ Builds the matrix for the SVD/PCA and other matrix decompositions, 
    centers the data and masks the frames central area if needed.
    
    Parameters
    ----------
    array : array_like
        Input cube, 3d array.
    scaling : {'mean', 'standard', None}, optional
        If "mean" then temporal px-wise mean subtraction is done, if "standard" 
        then mean centering plus scaling to unit variance is done. With None, no
        scaling is performed on the input data before SVD.
    mask_center_px : None or Int, optional
        Whether to mask the center of the frames or not.
    verbose : {True, False}, bool optional
        If True prints intermediate info and timing. 
    
    Returns
    -------
    matrix : array_like
        Out matrix whose rows are vectorized frames from the input cube.
    
    """
    if mask_center_px:
        array = mask_circle(array, mask_center_px)
    
    nfr = array.shape[0]
    matrix = np.reshape(array, (nfr, -1))  # == for i: array[i].flatten()                            
    
    if scaling==None:
        pass
    elif scaling=='mean':
        matrix = scale(matrix, with_mean=True, with_std=False)
    elif scaling=='standard':
        matrix = scale(matrix, with_mean=True, with_std=True)                
    else:
        raise ValueError('Scaling mode not recognized')
    
    if verbose:
        print('Done creating and centering the matrix')  
    return matrix
     

def reshape_matrix(array, y, x):
    """ Converts a matrix whose rows are vectorized frames to a cube with 
    reshaped frames.
    """
    return array.reshape(array.shape[0], y, x)


def svd_wrapper(matrix, mode, ncomp, debug, verbose, usv=False):
    """ Wrapper for different SVD libraries.
    
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
        rec_matrix = np.dot(U, np.dot(np.diag(S), V))
        print('  Matrix reconstruction MAE =', mean_absolute_error(matrix, 
                                                                   rec_matrix))
        exp_var = (S ** 2) / matrix.shape[0]
        full_var = np.var(matrix, axis=0).sum()
        explained_variance_ratio = exp_var / full_var           # Percentage of variance explained by each PC
        if var==1:
            pass    
        else:
            explained_variance_ratio = explained_variance_ratio[::-1]
        ratio_cumsum = explained_variance_ratio.cumsum()
        msg = '  Explained variance for {:} PCs = {:.5f}'
        print(msg.format(ncomp, ratio_cumsum[ncomp-1]))
        
    if mode=='eigen':
        M = np.dot(matrix, matrix.T)                             # covariance matrix
        e, EV = linalg.eigh(M)                                   # eigenvalues and eigenvectors
        pc = np.dot(EV.T, matrix)                                # PCs / compact trick
        V = pc[::-1]                                             # reverse since last eigenvectors are the ones we want 
        S = np.sqrt(e)[::-1]                                     # reverse since eigenvalues are in increasing order 
        for i in xrange(V.shape[1]): 
            V[:,i] /= S
        V = V[:ncomp]
        if verbose: print('Done SVD/PCA with scipy linalg eigh functions')
        
    # When num_px < num_frames (rare case) or we need all the PCs
    elif mode=='lapack':
        U, S, V = linalg.svd(matrix, full_matrices=False)         
        if debug: reconstruction(ncomp, U, S, V, 1)
        # we cut projection matrix according to the # of PCs
        V = V[:ncomp]                                             
        U = U[:,:ncomp]
        S = S[:ncomp]
        if verbose: print('Done SVD/PCA with scipy SVD (LAPACK)')
            
    elif mode=='arpack':
        U, S, V = svds(matrix, k=ncomp) 
        if debug: reconstruction(ncomp, U, S, V, -1)
        if verbose: print('Done SVD/PCA with scipy sparse SVD (ARPACK)')
        
    elif mode=='opencv':
        _, V = cv2.PCACompute(matrix, maxComponents=ncomp)          # PCs
        if verbose: print('Done SVD/PCA with opencv.')

    elif mode=='randsvd':
        U, S, V = randomized_svd(matrix, n_components=ncomp, n_iter=2, 
                                 transpose='auto', random_state=None)
        if debug: reconstruction(ncomp, U, S, V, 1)
        if verbose: print('Done SVD/PCA with randomized SVD')

    else:
        raise TypeError('The SVD mode is not available')
            
    if usv and mode!='opencv':
        return U, S, V
    else:
        return V


