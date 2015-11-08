#! /usr/bin/env python

"""
Module with post-processing related functions called from within the NFC
algorithm.
"""

import numpy as np
from skimage.draw import circle
from ..var import frame_center
from ..pca import pca_annulus


def get_values_optimize(cube, angs, ncomp, annulus_width, aperture_radius, 
                        r_guess, theta_guess, cube_ref=None, svd_mode='lapack',
                        debug=False):
    """
    Extracts a PCA-ed annulus from the cube and returns the flux values of the 
    pixels included in a circular aperture centered at a given position.
    
    Parameters
    ----------
    cube: numpy.array
        The cube of fits images expressed as a numpy.array.
    angs: numpy.array
        The parallactic angle fits image expressed as a numpy.array.
    ncomp: int
        The number of principal component.
    annulus_width: float
        The width in pixel of the annulus on which the PCA is performed.
    aperture_radius: float
        The radius of the circular aperture.
    r_guess: float
        The radial position of the center of the circular aperture. This parameter 
        is NOT the radial position of the candidate associated to the Markov 
        chain, but should be the fixed initial guess.
    theta_guess: float
        The angular position of the center of the circular aperture. This parameter 
        is NOT the angular position of the candidate associated to the Markov 
        chain, but should be the fixed initial guess.  
    cube_ref : array_like, 3d, optional
        Reference library cube. For Reference Star Differential Imaging.
    svd_mode : {'lapack', 'eigen', 'randsvd', 'arpack', 'opencv'}, str optional
        Switch for different ways of computing the SVD and selected PCs.
    debug: boolean
        If True, the cube is returned along with the values.        
        
    Returns
    -------
    out: numpy.array
        The pixel values in the circular aperture after the PCA process.
        
    """

    pca_frame = pca_annulus(cube, angs, ncomp, annulus_width, r_guess, cube_ref,
                            svd_mode) 

    centy_fr, centx_fr = frame_center(cube[0])
    posy = r_guess * np.sin(np.deg2rad(theta_guess)) + centy_fr
    posx = r_guess * np.cos(np.deg2rad(theta_guess)) + centx_fr
    indices = circle(cy=posy, cx=posx, radius=aperture_radius)

    values = pca_frame[indices]
    
    if debug:
        return values, pca_frame
    else:
        return values

