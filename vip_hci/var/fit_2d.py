#! /usr/bin/env python

"""
2d fitting.
"""
from __future__ import print_function

__author__ = 'Carlos Alberto Gomez Gonzalez'
__all__ = ['fit_2dgaussian',
           'fit_2dmoffat']

import numpy as np
import pandas as pd
import photutils
from scipy.optimize import leastsq
from astropy.modeling import models
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.stats import gaussian_sigma_to_fwhm, gaussian_fwhm_to_sigma, sigma_clipped_stats
from .shapes import get_square, frame_center
from .utils_var import pp_subplots


def fit_2dgaussian(array, crop=False, cent=None, cropsize=15, fwhmx=4, fwhmy=4, 
                   theta=0, threshold=False, sigfactor=6, full_output=False, 
                   debug=False):
    """ Fitting a 2D Gaussian to the 2D distribution of the data.
    
    Parameters
    ----------
    array : array_like
        Input frame with a single PSF.
    crop : bool, optional
        If True an square sub image will be cropped.
    cent : tuple of int, optional
        X,Y integer position of source in the array for extracting the subimage. 
        If None the center of the frame is used for cropping the subframe (the 
        PSF is assumed to be ~ at the center of the frame). 
    cropsize : int, optional
        Size of the subimage.
    fwhmx, fwhmy : float, optional
        Initial values for the standard deviation of the fitted Gaussian, in px.
    theta : float, optional
        Angle of inclination of the 2d Gaussian counting from the positive X
        axis.
    threshold : bool, optional
        If True the background pixels (estimated using sigma clipped statistics)
        will be replaced by small random Gaussian noise.
    sigfactor : int, optional
        The background pixels will be thresholded before fitting a 2d Gaussian
        to the data using sigma clipped statistics. All values smaller than
        (MEDIAN + sigfactor*STDDEV) will be replaced by small random Gaussian 
        noise. 
    full_output : bool, optional
        If False it returns just the centroid, if True also returns the 
        FWHM in X and Y (in pixels), the amplitude and the rotation angle.
    debug : bool, optional
        If True, the function prints out parameters of the fit and plots the
        data, model and residuals.
        
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
    if not array.ndim == 2:
        raise TypeError('Input array is not a frame or 2d array')
    
    if crop:
        if cent is None:
            ceny, cenx = frame_center(array)
        else:
            cenx, ceny = cent
        
        imside = array.shape[0]
        psf_subimage, suby, subx = get_square(array, min(cropsize, imside), 
                                              ceny, cenx, position=True)  
    else:
        psf_subimage = array.copy()  
    
    if threshold:
        _, clipmed, clipstd = sigma_clipped_stats(psf_subimage, sigma=2)
        indi = np.where(psf_subimage <= clipmed + sigfactor * clipstd)
        subimnoise = np.random.randn(psf_subimage.shape[0],
                                     psf_subimage.shape[1]) * clipstd
        psf_subimage[indi] = subimnoise[indi]

    # Creating the 2D Gaussian model
    init_amplitude = np.ptp(psf_subimage)
    xcom, ycom = photutils.centroid_com(psf_subimage)
    gauss = models.Gaussian2D(amplitude=init_amplitude, theta=theta,
                              x_mean=xcom, y_mean=ycom,
                              x_stddev=fwhmx * gaussian_fwhm_to_sigma,
                              y_stddev=fwhmy * gaussian_fwhm_to_sigma)
    # Levenberg-Marquardt algorithm
    fitter = LevMarLSQFitter()                  
    y, x = np.indices(psf_subimage.shape)
    fit = fitter(gauss, x, y, psf_subimage)

    if crop:
        mean_y = fit.y_mean.value + suby
        mean_x = fit.x_mean.value + subx
    else:
        mean_y = fit.y_mean.value
        mean_x = fit.x_mean.value 
    fwhm_y = fit.y_stddev.value*gaussian_sigma_to_fwhm
    fwhm_x = fit.x_stddev.value*gaussian_sigma_to_fwhm 
    amplitude = fit.amplitude.value
    theta = fit.theta.value
    
    if debug:
        if threshold:  msg = 'Subimage thresholded / Model / Residuals'
        else: msg = 'Subimage (no threshold) / Model / Residuals'
        pp_subplots(psf_subimage, fit(x, y), psf_subimage-fit(x, y), 
                    colorb=True, grid=True, title=msg)
        print('FWHM_y =', fwhm_y)
        print('FWHM_x =', fwhm_x)
        print()
        print('centroid y =', mean_y)
        print('centroid x =', mean_x)
        print('centroid y subim =', fit.y_mean.value)
        print('centroid x subim =', fit.x_mean.value)
        print() 
        print('peak =', amplitude)
        print('theta =', theta)
    
    if full_output:
        return pd.DataFrame({'centroid_y': mean_y, 'centroid_x': mean_x,
                             'fwhm_y': fwhm_y, 'fwhm_x': fwhm_x,
                             'amplitude': amplitude, 'theta': theta}, index=[0])
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
    full_output: bool, opt
        Whether to return floor, height, mean_y, mean_x, fwhm, beta, or just 
        mean_y, mean_x
    
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
    def moffat(floor, height, mean_y, mean_x, fwhm, beta): # def Moffat function
        alpha = 0.5*fwhm/np.sqrt(2.**(1./beta)-1.)
        return lambda y,x: floor + height/((1.+(((x-mean_x)**2+(y-mean_y)**2)/\
                                                alpha**2.))**beta)

    def err(p,data):
        return np.ravel(moffat(*p)(*np.indices(data.shape))-data)
    ############################################################################

    maxi = array.max()  # find starting values
    floor = np.ma.median(array.flatten())
    height = maxi - floor
    if height == 0.0:   # if star is saturated it could be that
        floor = np.mean(array.flatten())
        height = maxi - floor

    mean_y = (np.shape(array)[0]-1)/2
    mean_x = (np.shape(array)[1]-1)/2

    fwhm = np.sqrt(np.sum((array > floor+height/2.).flatten()))

    beta = 4
    
    p0 = floor, height, mean_y, mean_x, fwhm, beta

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

