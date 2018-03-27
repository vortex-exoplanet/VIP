#! /usr/bin/env python

"""
Module with NMF algorithm for ADI.
"""

from __future__ import division, print_function

__author__ = 'Carlos Alberto Gomez Gonzalez'
__all__ = ['nmf']

import numpy as np
from sklearn.decomposition import NMF
from ..preproc import cube_derotate, cube_collapse
from ..var import prepare_matrix
from ..conf import timing, time_ini


def nmf(cube, angle_list, cube_ref=None, ncomp=1, scaling=None, max_iter=100,
        random_state=None, mask_center_px=None, imlib='opencv',
        interpolation='lanczos4', collapse='median', full_output=False,
        verbose=True, **kwargs):
    """ Non Negative Matrix Factorization for ADI sequences. Alternative to the
    full-frame ADI-PCA processing that does not rely on SVD or ED for obtaining
    a low-rank approximation of the datacube. This function embeds the 
    scikit-learn NMF algorithm solved through coordinate descent method. 
    
    Parameters
    ----------
    cube : array_like, 3d
        Input cube.
    angle_list : array_like, 1d
        Corresponding parallactic angle for each frame.   
    cube_ref : array_like, 3d, optional
        Reference library cube. For Reference Star Differential Imaging.
    ncomp : int, optional
        How many components are used as for low-rank approximation of the 
        datacube.
    scaling : {None, 'temp-mean', 'spat-mean', 'temp-standard', 'spat-standard'}
        With None, no scaling is performed on the input data before SVD. With 
        "temp-mean" then temporal px-wise mean subtraction is done, with 
        "spat-mean" then the spatial mean is subtracted, with "temp-standard" 
        temporal mean centering plus scaling to unit variance is done and with
        "spat-standard" spatial mean centering plus scaling to unit variance is
        performed.  
    max_iter : int optional
        The number of iterations for the coordinate descent solver.
    random_state : int or None, optional
        Controls the seed for the Pseudo Random Number generator.
    mask_center_px : None or int
        If None, no masking is done. If an integer > 1 then this value is the
        radius of the circular mask.
    imlib : str, optional
        See the documentation of the ``vip_hci.preproc.frame_rotate`` function.
    interpolation : str, optional
        See the documentation of the ``vip_hci.preproc.frame_rotate`` function.
    collapse : {'median', 'mean', 'sum', 'trimmean'}, str optional
        Sets the way of collapsing the frames for producing a final image.
    full_output: boolean, optional
        Whether to return the final median combined image only or with other 
        intermediate arrays.  
    verbose : {True, False}, bool optional
        If True prints intermediate info and timing. 
    kwargs 
        Additional arguments for scikit-learn NMF algorithm.             
             
    Returns
    -------
    If full_output is False the final frame is returned. If True the algorithm
    returns the reshaped NMF components, the reconstructed cube, the residuals,
    the residuals derotated and the final frame.     
    
    """
    array = cube
    if verbose:  start_time = time_ini()
    n, y, x = array.shape
    
    matrix = prepare_matrix(cube, scaling, mask_center_px, mode='fullfr',
                            verbose=verbose)
    matrix += np.abs(matrix.min())
    if cube_ref is not None:
        matrix_ref = prepare_matrix(cube_ref, scaling, mask_center_px,
                                    mode='fullfr', verbose=verbose)
        matrix_ref += np.abs(matrix_ref.min())
           
    mod = NMF(n_components=ncomp, alpha=0, solver='cd', init='nndsvd', 
              max_iter=max_iter, random_state=random_state, **kwargs) 
    
    # H [ncomp, n_pixels]: Non-negative components of the data
    if cube_ref is not None:
        H = mod.fit(matrix_ref).components_
    else:
        H = mod.fit(matrix).components_          
    
    if verbose:  
        print('Done NMF with sklearn.NMF.')
        timing(start_time)
    # W: coefficients [n_frames, ncomp]
    W = mod.transform(matrix)
        
    reconstructed = np.dot(W, H)
    residuals = matrix - reconstructed
               
    array_out = np.zeros_like(array)
    for i in range(n):
        array_out[i] = residuals[i].reshape(y,x)
            
    array_der = cube_derotate(array_out, angle_list, imlib=imlib,
                              interpolation=interpolation)
    frame = cube_collapse(array_der, mode=collapse)
    
    if verbose:  
        print('Done derotating and combining.')
        timing(start_time)
    if full_output:
        return (H.reshape(ncomp,y,x), reconstructed.reshape(n,y,x), array_out,
                array_der, frame )
    else:
        return frame
    
    
