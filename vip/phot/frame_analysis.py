#! /usr/bin/env python

"""
Module
"""

from __future__ import division
from __future__ import print_function

__author__ = 'C. Gomez @ ULg'
__all__ = ['frame_quick_report']

import numpy as np
import photutils
from skimage import draw
from .snr import snr_ss
from ..var import fit_2dgaussian


def frame_quick_report(array, fwhm, y=None, x=None , verbose=True):
    """ Gets information from given frame: Integrated flux in aperture, SNR of
    central pixel (max or given coordinates), mean SNR in aperture, fitted 
    coordinates (2d gaussian).
    
    Parameters
    ----------
    array : array_like
        2d array or input frame.
    fwhm : float
        Size of the FWHM in pixels.
    y : int, optional, None by default
        Y coordinate of source center.
    x : int, optional, None by default
        X coordinate of source center.
    verbose : {True, False}, bool optional    
        If True prints to stdout the frame info.
        
    Returns
    -------
    obj_flux : float
        Integrated flux in aperture.
    snr_pixels : array_like
        SNR of pixels in 1FWHM aperture.
    fy, fx : float
        Fitted coordinates.
        
    """
    if not array.ndim == 2: raise TypeError('Array is not 2d.')
    
    # we get integrated flux on aperture with diameter=1FWHM
    if not y and not x:
        y, x = np.where(array == array.max())
        if verbose: 
            print
            print('Coordinates of Max px Y,X = {:},{:}'.format(y[0],x[0]))
    else:
        if verbose: 
            print
            print('Coordinates of chosen px Y,X = {:},{:}'.format(y,x))
    aper = photutils.CircularAperture((x, y), fwhm/2.)
    obj_flux = photutils.aperture_photometry(array, aper, method='exact')
    obj_flux = obj_flux['aperture_sum'][0]
    
    # we get the mean and stddev of SNRs on aperture
    yy, xx = draw.circle(y, x, fwhm/2.)
    snr_pixels = [snr_ss(array, y_, x_, fwhm, plot=False, verbose=False) for \
                  y_, x_ in zip(yy, xx)]
    meansnr = np.mean(snr_pixels)
    stdsnr = np.std(snr_pixels)
    if verbose: 
        print('Central pixel SNR: ')
        snr_ss(array, y, x, fwhm, plot=False, verbose=True)
        print('-----------------------------------------')
        print('In 1*FWHM circular aperture:')
        print('Integrated flux = {:.3f}'.format(obj_flux))
        print('Mean SNR = {:.3f}'.format(meansnr)) 
        print('Max SNR = {:.3f}, stddev SNRs = {:.3f}'.format(np.max(snr_pixels), 
                                                              stdsnr)) 
        print('-----------------------------------------')
        
    # we fit a 2d gaussian to the approx center px of the planet
    fy, fx = fit_2dgaussian(array, y, x, fwhm)
    if verbose: print('Fitted Y,X = {:.3f},{:.3f}'.format(fy, fx))
    
    return obj_flux, snr_pixels, fy, fx
    



