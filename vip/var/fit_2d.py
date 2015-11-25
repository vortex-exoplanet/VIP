#! /usr/bin/env python

"""
2d fitting.
"""

__author__ = 'C. Gomez @ ULg'
__all__ = ['fit_2dgaussian',
           'fit_2dmoffat']

import numpy as np
from scipy.optimize import leastsq
from astropy.modeling import models
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.stats import gaussian_sigma_to_fwhm
from .shapes import get_square_robust, frame_center
from .utils import pp_subplots


def fit_2dgaussian(array, cent=None, fwhm=4, full_output=False, verbose=True):
    """ Fitting a 2D Gaussian to the 2D distribution of the data with photutils.
    
    Parameters
    ----------
    array : array_like
        Input frame with a single PSF.
    cent : tuple of int, optional
        X,Y integer position of source in the array for extracting a subframe. 
        If None the center of the frame is used for cropping the subframe (then
        the PSF is assumed to be ~ at the center of the frame). 
    fwhm : float, optional    
        Expected FWHM of the Gaussian.
    full_output : {False, True}, optional
        If False it returns just the centroid, if True also returns the 
        FWHM in X and Y (in pixels), the amplitude and the rotation angle.
    verbose : {True, False}, optional
        Whether to plot the arrays and print out fit results.
        
    Returns
    -------
    mean_y : float
        Source centroid y position on input array from fitting. 
    mean_x : float
        Source centroid x position on input array from fitting.
        
    If *full_output* is True it returns:
    mean_y, mean_x : floats
        Centroid. 
    fwhm_y : float
        FHWM in Y in pixels. 
    fwhm_x : float
        FHWM in X in pixels.
    amplitude : float
        Amplitude of the Gaussian.
    theta : float
        Rotation angle.
    
    """
    if cent is None:
        ceny, cenx = frame_center(array)
    else:
        cenx, ceny = cent
    
    # Cropping to 3*fwhm+1 
    psf_subimage,suby,subx = get_square_robust(array, min(3*fwhm, array.shape[0]), 
                                          ceny, cenx, position=True)    

    yme, xme = frame_center(psf_subimage)
    # Creating the 2D Gaussian model
    gauss = models.Gaussian2D(amplitude=psf_subimage.max(), x_mean=xme, 
                              y_mean=yme, x_stddev=fwhm, y_stddev=fwhm, theta=0)
    # Levenberg-Marquardt algorithm
    fitter = LevMarLSQFitter()                  
    y, x = np.indices(psf_subimage.shape)
    fit = fitter(gauss, x, y, psf_subimage)

    mean_y = fit.y_mean.value + suby
    mean_x = fit.x_mean.value + subx
    fwhm_y = fit.y_stddev.value*gaussian_sigma_to_fwhm
    fwhm_x = fit.x_stddev.value*gaussian_sigma_to_fwhm 
    amplitude = fit.amplitude.value
    theta = fit.theta.value
    
    if verbose:
        pp_subplots(array, psf_subimage, colorb=True, grid=True)
        print 'FWHM_y =', fwhm_y
        print 'FWHM_x =', fwhm_x
        print
        print 'centroid y =', mean_y
        print 'centroid x =', mean_x
        print 'centroid y subim =', fit.y_mean.value
        print 'centroid x subim =', fit.x_mean.value
        print 
        print 'peak =', amplitude
        print 'theta =', theta
    
    if full_output:
        return mean_y, mean_x, fwhm_y, fwhm_x, amplitude, theta
    else:
        return mean_y, mean_x


    
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
    maxi = array.max() # find starting values
    floor = np.ma.median(array.flatten())
    height = maxi - floor
    if height==0.0: # if star is saturated it could be that 
        floor = np.mean(array.flatten())  # median value is 32767 or 65535 --> height=0
        height = maxi - floor

    mean_y = (np.shape(array)[0]-1)/2
    mean_x = (np.shape(array)[1]-1)/2

    fwhm = np.sqrt(np.sum((array>floor+height/2.).flatten()))

    beta = 4
    
    p0 = floor, height, mean_y, mean_x, fwhm, beta

    def moffat(floor, height, mean_y, mean_x, fwhm, beta): # def Moffat function
        alpha = 0.5*fwhm/np.sqrt(2.**(1./beta)-1.)    
        return lambda y,x: floor + height/((1.+(((x-mean_x)**2+(y-mean_y)**2)/\
                                                alpha**2.))**beta)

    def err(p,data):
        return np.ravel(moffat(*p)(*np.indices(data.shape))-data)
    
    p = leastsq(err, p0, args=(array), maxfev=1000)
    p = p[0]
    
    # results
    floor = p[0]                                
    height = p[1]
    mean_y = p[2] + yy
    mean_x = p[3] + xx
    fwhm = np.abs(p[4])
    beta = p[5]
    
    if full_output:
        return floor, height, mean_y, mean_x, fwhm, beta
    else:
        return mean_y, mean_x

