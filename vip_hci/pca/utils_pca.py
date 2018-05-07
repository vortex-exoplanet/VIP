#! /usr/bin/env python

"""
Module with helping functions.
"""

from __future__ import division
from __future__ import print_function

__author__ = 'Carlos Alberto Gomez Gonzalez'
__all__ = []

import numpy as np
from ..var import prepare_matrix
from ..preproc import cube_derotate, cube_collapse
from .svd import svd_wrapper


def pca_annulus(cube, angs, ncomp, annulus_width, r_guess, cube_ref=None,
                svd_mode='lapack', scaling=None, collapse='median',
                imlib='opencv', interpolation='lanczos4'):
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
    imlib : str, optional
        See the documentation of the ``vip_hci.preproc.frame_rotate`` function.
    interpolation : str, optional
        See the documentation of the ``vip_hci.preproc.frame_rotate`` function.
    
    Returns
    -------
    Depending on ``collapse`` parameter a final collapsed frame or the cube of
    residuals is returned.
    """
    data, ind = prepare_matrix(cube, scaling, mode='annular',
                               annulus_radius=r_guess, verbose=False,
                               annulus_width=annulus_width)
    yy, xx = ind

    if cube_ref is not None:
        data_svd, _ = prepare_matrix(cube_ref, scaling, mode='annular',
                                     annulus_radius=r_guess, verbose=False,
                                     annulus_width=annulus_width)
    else:
        data_svd = data
        
    V = svd_wrapper(data_svd, svd_mode, ncomp, debug=False, verbose=False)
        
    transformed = np.dot(data, V.T)
    reconstructed = np.dot(transformed, V)                           
    residuals = data - reconstructed
    cube_zeros = np.zeros_like(cube)
    cube_zeros[:, yy, xx] = residuals

    if angs is not None:
        cube_res_der = cube_derotate(cube_zeros, angs, imlib=imlib,
                                     interpolation=interpolation)
        if collapse is not None:
            pca_frame = cube_collapse(cube_res_der, mode=collapse)
            return pca_frame
        else:
            return cube_res_der

    else:
        if collapse is not None:
            pca_frame = cube_collapse(cube_zeros, mode=collapse)
            return pca_frame
        else:
            return cube_zeros




