#! /usr/bin/env python

"""
2d fitting.
"""

__author__ = 'C. Gomez @ ULg'
__all__ = ['fit_2dgaussian']

import numpy as np
from scipy.optimize import leastsq
from photutils import morphology
from .shapes import get_square


def fit_2dgaussian(array, cy=None, cx=None, fwhm=4):
    """ Fitting a 2D Gaussian to the 2D distribution of the data with photutils.
    
    Parameters
    ----------
    array : array_like
        Input frame with a single point source, approximately at the center.
    cy : int
        Y integer position of source in the array for extracting a subframe. 
    cx : int
        X integer position of source in the array for extracting a subframe.
    fwhm : float    
        Expected FWHM of the gaussian.
    
    Returns
    -------
    x : float
        Source centroid x position on input array from fitting.
    y : float
        Source centroid y position on input array from fitting. 
    
    """
    if cy and cx and fwhm:
        subimage, suby, subx = get_square(array, size=6*int(fwhm), y=cy, x=cx, 
                                          position=True)
        x, y = morphology.centroid_2dg(subimage)
        x += subx
        y += suby
    else:
        x, y = morphology.centroid_2dg(array)
    return y, x


    
