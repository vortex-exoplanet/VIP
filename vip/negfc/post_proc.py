#! /usr/bin/env python

"""
Module with post-processing related functions called from within the NFC
algorithm.
"""

import numpy as np
from skimage.draw import circle
#from ..calib import cube_derotate
from ..var import frame_center #, get_annulus 
from ..pca import pca_annulus
from ..fits import display_array_ds9


def get_values_optimize(cube, angs, ncomp, annulus_width, aperture_radius, 
                        r_guess, theta_guess, cube_ref=None, display=False):
    """
    Extracts a PCA-ed annulus from the cube and returns the flux values of the 
    pixels included in a circular aperture centered at a given position.
    
    Paramters
    ---------
    cube: numpy.array
        The cube of fits images expressed as a numpy.array.
    angs: numpy.array
        The parallactic angle fits image expressed as a numpy.array.
    ncomp: int
        The number of principal component.
    annulus_width: float
        The width in pixel of the annulus on wich the PCA is performed.
    aperture_radius: float
        The radius of the circular aperture.
    r_guess: float
        The radial position of the center of the circular aperture. This parameter 
        is NOT the radial position of the candidat associated to the Markov 
        chain, but should be the fixed initial guess.
    theta_guess: float
        The angular position of the center of the circular aperture. This parameter 
        is NOT the angular position of the candidat associated to the Markov 
        chain, but should be the fixed initial guess.  
    cube_ref : array_like, 3d, optional
        Reference library cube. For Reference Star Differential Imaging.
    display: boolean
        If True, the cube is displayed with ds9.        
        
    Returns
    -------
    out: numpy.array
        The pixel values in the circular aperture after the PCA process.
        
    """

    pca_frame = pca_annulus(cube, angs, ncomp, annulus_width, r_guess, cube_ref)
    if display:
        display_array_ds9(pca_frame)   
    
    centy_fr, centx_fr = frame_center(cube[0])
    posy = r_guess * np.sin(np.deg2rad(theta_guess)) + centy_fr
    posx = r_guess * np.cos(np.deg2rad(theta_guess)) + centx_fr
    indices = circle(cy=posy, cx=posx, radius=aperture_radius)
    values = pca_frame[indices]
    
    return values