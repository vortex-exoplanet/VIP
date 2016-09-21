#! /usr/bin/env python

"""

"""

from __future__ import division
from __future__ import print_function

__author__ = 'C. Gomez @ ULg'
__all__ = ['frame_quick_report']

import numpy as np
import photutils
from skimage import draw
from .snr import snr_ss
from ..conf import sep


def frame_quick_report(array, fwhm, source_xy=None, verbose=True):
    """ Gets information from given frame: Integrated flux in aperture, SNR of
    central pixel (max or given coordinates), mean SNR in aperture.
    
    Parameters
    ----------
    array : array_like
        2d array or input frame.
    fwhm : float
        Size of the FWHM in pixels.
    source_xy : tuple of floats
        X and Y coordinates of the source center.
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

    if source_xy is not None: 
        x, y = source_xy
        if verbose:
            print(sep)
            print('Coordinates of chosen px (X,Y) = {:},{:}'.format(x,y))
    else:
        y, x = np.where(array == array.max())
        y = y[0]
        x = x[0]
        if verbose:
            print(sep)
            print('Coordinates of Max px (X,Y) = {:},{:}'.format(x,y))
    
    # we get integrated flux on aperture with diameter=1FWHM
    aper = photutils.CircularAperture((x, y), r=fwhm/2.)
    obj_flux = photutils.aperture_photometry(array, aper, method='exact')
    obj_flux = obj_flux['aperture_sum'][0]
    
    # we get the mean and stddev of SNRs on aperture
    yy, xx = draw.circle(y, x, fwhm/2.)
    snr_pixels = [snr_ss(array, (x_, y_), fwhm, plot=False, verbose=False) for \
                  y_, x_ in zip(yy, xx)]
    meansnr = np.mean(snr_pixels)
    stdsnr = np.std(snr_pixels)

    if verbose: 
        print('Central pixel S/N: ')
        snr_ss(array, (x, y), fwhm, plot=False, verbose=True)
        print(sep)
        print('Inside 1xFWHM circular aperture:')
        print('Mean S/N (shifting the aperture center) = {:.3f}'.format(meansnr))
        print('Max S/N (shifting the aperture center) = {:.3f}'.format(np.max(snr_pixels)))
        print('stddev S/NR (shifting the aperture center) = {:.3f}'.format(stdsnr))
        print(sep)
    
    return obj_flux, snr_pixels
    



