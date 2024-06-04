#! /usr/bin/env python
"""
Implementation of the STIM map from [PAI19]

.. [PAI19]
   | Pairet et al. 2019
   | **STIM map: detection map for exoplanets imaging beyond asymptotic Gaussian
   residual speckle noise**
   | *MNRAS, 487, 2262*
   | `http://doi.org/10.1093/mnras/stz1350
     <http://doi.org/10.1093/mnras/stz1350>`_

"""
__author__ = 'Benoit Pairet'
__all__ = ['stim_map',
           'inverse_stim_map',
           'normalized_stim_map']

import numpy as np
from ..preproc import cube_derotate
from ..var import get_circle, mask_circle


def stim_map(cube_der):
    """Compute the STIM detection map as in [PAI19]_.

    Parameters
    ----------
    cube_der : 3d numpy ndarray
        Input de-rotated cube, e.g. ``residuals_cube_`` output from
        ``vip_hci.psfsub.pca``.

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


def inverse_stim_map(cube, angle_list, **rot_options):
    """Compute the inverse STIM detection map as in [PAI19]_.

    Parameters
    ----------
    cube : 3d numpy ndarray
        Non de-rotated residuals from reduction algorithm, eg.
        ``residuals_cube`` output from ``vip_hci.psfsub.pca``.
    angle_list : 1d numpy ndarray
        Vector of derotation angles to align North up in your cube images.
    rot_options: dictionary, optional
        Dictionary with optional keyword values for "nproc", "imlib",
        "interpolation, "border_mode", "mask_val",  "edge_blend",
        "interp_zeros", "ker" (see documentation of
        ``vip_hci.preproc.frame_rotate``)

    Returns
    -------
    inv_stim_map : 2d ndarray
        Inverse STIM detection map.

    """
    t, n, _ = cube.shape
    cube_inv_der = cube_derotate(cube, -angle_list, **rot_options)
    inv_stim_map = stim_map(cube_inv_der)
    return inv_stim_map


def normalized_stim_map(cube, angle_list, mask=None, **rot_options):
    """Compute the normalized STIM detection map as in [PAI19]_.

    Parameters
    ----------
    cube : 3d numpy ndarray
        Non de-rotated residuals from reduction algorithm, eg.
        ``residuals_cube`` output from ``vip_hci.psfsub.pca``.
    angle_list : 1d numpy ndarray
        Vector of derotation angles to align North up in your cube images.
    mask : int, float, numpy ndarray 2d or None
        Mask informing where the maximum value in the inverse STIM map should
        be calculated. If an integer or float, a circular mask with that radius
        masking the central part of the image will be used. If a 2D array, it
        should be a binary mask (ones in the areas that can be used).
    rot_options: dictionary, optional
        Dictionary with optional keyword values for "nproc", "imlib",
        "interpolation, "border_mode", "mask_val",  "edge_blend",
        "interp_zeros", "ker" (see documentation of
        ``vip_hci.preproc.frame_rotate``)

    Returns
    -------
    normalized STIM map : 2d ndarray
        STIM detection map.

    """
    inv_map = inverse_stim_map(cube, angle_list, **rot_options)

    if mask is not None:
        if np.isscalar(mask):
            inv_map = mask_circle(inv_map, mask)
        else:
            inv_map *= mask

    max_inv = np.nanmax(inv_map)
    if max_inv <= 0:
        msg = "The normalization value is found to be {}".format(max_inv)
        raise ValueError(msg)

    cube_der = cube_derotate(cube, angle_list, **rot_options)
    det_map = stim_map(cube_der)

    return det_map/max_inv
