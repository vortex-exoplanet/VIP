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
           'pca_annulus']

import cv2
import numpy as np
from scipy import linalg
from scipy.sparse.linalg import svds
from sklearn.decomposition import randomized_svd
from sklearn.metrics import mean_absolute_error
from ..var import mask_circle, get_annulus
from ..calib import cube_derotate


def pca_annulus(cube, angs, ncomp, annulus_width, r_guess, cube_ref=None, 
                display=False):
    """
    PCA process the cube only for an annulus of a given width and at a given
    radial distance to the frame center. It returns a PCA-ed frame with only 
    non-zero values at the positions of the annulus.
    
    Parameters
    ----------
    cube: numpy.array
        The cube of fits images expressed as a numpy.array.
    angs: numpy.array
        The parallactic angles expressed as a numpy.array.
    ncomp: int
        The number of principal component.
    annulus_width: float
        The annulus width in pixel on which the PCA is performed.
    r_guess: float
        Radius of the annulus in pixels.
    display: boolean (optional)
        If True, the PCA-ed frame is open into ds9.
    
    Returns
    -------
    out: numpy.array
        The annulus PCA-ed frame.
        
    """
    indic = get_annulus(cube[0], r_guess-annulus_width, annulus_width, 
                        output_indices=True)
    yy, xx = indic
    
    data = cube[:, yy, xx]
    #data = matrix_ann - matrix_ann.mean(axis=0) # centering?
    
    if cube_ref is not None:
        data_svd = cube_ref[:, yy, xx]
    else:
        data_svd = data
        
    V = svd_wrapper(data_svd, mode='randsvd', ncomp=ncomp, debug=False, 
                    verbose=False)
        
    transformed = np.dot(data, V.T)
    reconstructed = np.dot(transformed, V)                           
    residuals = data - reconstructed
    cube_zeros = np.zeros_like(cube)
    cube_zeros[:, yy, xx] = residuals
    cube_zeros_derot, pca_frame = cube_derotate(cube_zeros, angs)
    
    if display:
        display_array_ds9(pca_frame)

    return pca_frame  



def prepare_matrix(array, center=None, mask_center_px=None, verbose=True):
    """ Builds the matrix for the SVD/PCA and other matrix decompositions, 
    centers the data and masks the frames central area if needed.
    
    Parameters
    ----------
    array : array_like
        Input cube, 3d array.
    center : {None, 'spatial', 'temporal', 'global'}, optional
        None means no centering (no mean removal), 'spatial' means removing the
        average pixel value per frame, 'temporal' means removing the average 
        per pixel, 'global' means removing both means.
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
                
    matrix = np.reshape(array, (array.shape[0], -1))            # equivalent to a for loop: array[i].flatten()                            
    
    if center==None:
        pass
    elif center=='spatial':
        matrix = matrix - matrix.mean(axis=1).reshape(array.shape[0], -1)       
    elif center=='temporal':
        matrix = matrix - matrix.mean(axis=0)                                   
    elif center=='global':
        matrix = matrix - matrix.mean(axis=0)                                   
        matrix = matrix - matrix.mean(axis=1).reshape(array.shape[0], -1)       
    else:
        raise ValueError('Centering mode not recognized')
    
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
        U, S, V = linalg.svd(matrix, full_matrices=False)         # scipy SVD, S = variance(singular values)
        if debug: reconstruction(ncomp, U, S, V, 1)
        V = V[:ncomp]                                             # we cut projection matrix according to the # of PCs
        if verbose: print('Done SVD/PCA with scipy SVD (LAPACK)')
            
    elif mode=='arpack':
        U, S, V = svds(matrix, k=ncomp) 
        if debug: reconstruction(ncomp, U, S, V, -1)
        if verbose: print('Done SVD/PCA with scipy sparse SVD (ARPACK)')
        
        
    elif mode=='opencv':
        _, V = cv2.PCACompute(matrix, maxComponents=ncomp)          # eigenvectors, PCs
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


