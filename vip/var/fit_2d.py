#! /usr/bin/env python

"""
2d fitting.
"""

__author__ = 'C. Gomez @ ULg'
__all__ = ['fit_2dgaussian']

import numpy as np
from scipy.optimize import leastsq
from photutils import morphology
from .shapes import get_square_robust

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
        subimage, suby, subx = get_square_robust(array, size=6*int(fwhm), y=cy, x=cx, position=True)
        x, y = morphology.centroid_2dg(subimage)
        x += subx
        y += suby
    else:
        x, y = morphology.centroid_2dg(array)
    return y, x


    
def fit_2dmoffat(array, yy, xx, full_output=False):
    """Fits a star/planet with a 2D circular Moffat PSF.
    
    Parameters
    ----------
    array : array_like
        Subimage with a single point source, approximately at the center. 
    yy : int
        Y integer position of the first pixel (0,0) of the subimage in the 
        whole image.
    xx : int
        X integer position of the first pixel (0,0) of the subimage in the 
        whole image.
    
    Returns
    -------
    maxi : float
        Value of the source maximum signal (pixel).
    floor : float
        Level of the sky background (fit result).
    height : float
        PSF amplitude (fit result).
    mean_x : float
        Source centroid x position on the full image from fitting.
    mean_y : float
        Source centroid y position on the full image from fitting. 
    fwhm : float
        Gaussian PSF full width half maximum from fitting (in pixels).
    beta : float
        "beta" parameter of the moffat function.
    """
    maxi = array.max()                                                          # find starting values
    floor = np.ma.median(array.flatten())
    height = maxi - floor
    if height==0.0:                                                             # if star is saturated it could be that 
        floor = np.mean(array.flatten())                                        # median value is 32767 or 65535 --> height=0
        height = maxi - floor

    mean_y = (np.shape(array)[0]-1)/2
    mean_x = (np.shape(array)[1]-1)/2

    fwhm = np.sqrt(np.sum((array>floor+height/2.).flatten()))

    beta = 4
    
    p0 = floor, height, mean_y, mean_x, fwhm, beta

    def moffat(floor, height, mean_y, mean_x, fwhm, beta):                      # fitting
        alpha = 0.5*fwhm/np.sqrt(2.**(1./beta)-1.)    
        return lambda y,x: floor + height/((1.+(((x-mean_x)**2+(y-mean_y)**2)/alpha**2.))**beta)

    def err(p,data):
        return np.ravel(moffat(*p)(*np.indices(data.shape))-data)
    
    p = leastsq(err, p0, args=(array), maxfev=1000)
    p = p[0]
    
    floor = p[0]                                                                # results
    height = p[1]
    mean_y = p[2] + yy
    mean_x = p[3] + xx
    fwhm = np.abs(p[4])
    beta = p[5]
    
    if full_output:
        return maxi, floor, height, mean_y, mean_x, fwhm, beta
    else:
        return mean_y, mean_x
