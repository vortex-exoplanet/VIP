#! /usr/bin/env python

"""
Module with functions related to polarimetric data.
"""

__author__ = 'Valentin Christiaens'
__all__ = ['conv_QU_phi']

import numpy as np
from .shapes import frame_center, cart_to_pol


def conv_QU_phi(Q, U, delta_x=0, delta_y=0, scale_r2=False, 
                north_convention=False):
    """
    Returns Qphi and Uphi images, from input Q and U images.
    
    Parameters
    ----------
    Q: numpy ndarray
        2d numpy array containing the Q component of polarisation.
    U: numpy ndarray
        2d numpy array containing the U component of polarisation. Should have
        the same dimensions as Q.
    delta_x, delta_y: float, opt
        If the star is not at the center of the image, delta_x and delta_y 
        indicate by how much it is offset along the x and y dimensions, resp.
    scale_r2: bool, opt
        Whether to scale by r^2 during conversion.
    north_convention: bool, opt
        Whether to use angles measured from North up/East left (True), or
        measured from the positive x axis (False).
        
    Returns
    -------
    Qphi, Uphi: numpy ndarrays
        Qphi and Uphi images
    """
    
    cy,cx = frame_center(Q)
    Qphi = np.zeros_like(Q)
    Uphi = np.zeros_like(U)
    for ii in range(Q.shape[1]):
        for jj in range(Q.shape[0]):
            x = float(ii-cx-delta_x)
            y = float(jj-cy-delta_y)
            rho, phi = cart_to_pol(x, y, north_convention=north_convention)
            phi = np.deg2rad(phi)
            if scale_r2:
                Qphi[jj,ii] = (Q[jj,ii]*np.cos(2*phi) + 
                                U[jj,ii]*np.sin(2*phi))*rho**2
                Uphi[jj,ii] = (-Q[jj,ii]*np.sin(2*phi) + 
                                U[jj,ii]*np.cos(2*phi))*rho**2
            else:
                Qphi[jj,ii] = Q[jj,ii]*np.cos(2*phi) + U[jj,ii]*np.sin(2*phi)
                Uphi[jj,ii] = -Q[jj,ii]*np.sin(2*phi) + U[jj,ii]*np.cos(2*phi)
                
    return Qphi, Uphi