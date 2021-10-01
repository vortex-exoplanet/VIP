#! /usr/bin/env python

"""
Implementation of the STIM map from [PAI18]
.. [PAI18]
   | Paret et al, 1018
   | **STIM map: detection map for exoplanets imaging beyond asymptotic Gaussian residual speckle noise**
   | *submitted in Monthly Notices of the Royal Astronomical Society*
"""
__author__ = 'Benoit Pairet'
__all__ = ['compute_stim_map',
           'compute_inverse_stim_map']

import numpy as np
from ..preproc import cube_derotate
from ..var import get_circle


def compute_stim_map(cube_der):
    """ Computes the STIM detection map.

    Parameters
    ----------
    cube_der : 3d numpy ndarray
        Input de-rotated cube, e.g. ``residuals_cube_`` output from
        ``vip_hci.pca.pca``.

    Returns
    -------
    detection_map : 2d ndarray
        STIM detection map.
    """
    t, n, _ = cube_der.shape
    mu = np.mean(cube_der, axis=0)
    sigma = np.sqrt(np.var(cube_der, axis=0))
    detection_map = np.divide(mu, sigma, out=np.zeros_like(mu),
                              where=sigma != 0)
    return get_circle(detection_map, int(np.round(n/2.)))


def compute_inverse_stim_map(cube, angle_list, imlib='opencv',
                             interpolation='lanczos4'):
    """ Computes the inverse STIM detection map.

    Parameters
    ----------
    cube : 3d numpy ndarray
        Non de-rotated residuals from reduction algorithm, eg. output residuals
        from ``vip_hci.pca.pca``.
    angle_list : numpy ndarray, 1d
        Corresponding parallactic angle for each frame.  
    imlib: str, opt
        See description of vip_hci.preproc.frame_derotate()
    interpolation: str, opt
        See description of vip_hci.preproc.frame_derotate()
        
    Returns
    -------
    inverse_stim_map : 2d ndarray
        Inverse STIM detection map.
    """
    t, n, _ = cube.shape
    cube_inv_der = cube_derotate(cube, -angle_list, imlib=imlib, 
                                 interpolation=interpolation)
    inverse_stim_map = compute_stim_map(cube_inv_der)
    return inverse_stim_map