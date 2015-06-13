#! /usr/bin/env python

"""
Module with frame filtering funcions
"""

from __future__ import division

__author__ = 'C. Gomez @ ULg'
__all__ = ['wavelet_denoise',
           'convolution_gaussian2d',
           'gaussian_filter_sp']

import pywt
import numpy as np
import photutils
from scipy.ndimage import gaussian_filter        
from astropy.convolution import convolve_fft, Gaussian2DKernel
from .shapes import frame_center


SIGMA2FWHM = 2.35482004503          # fwhm = sigma2fwhm * sigma


def wavelet_denoise(array, wavelet, threshold, levels, thrmode='hard'):
    """ Wavelet filtering of a 2d array using Pywt library. First a 2d discrete
    wavelet transform is performed followed by a hard or soft thresholding of 
    the coefficients.
    
    Parameters
    ----------
    array : array_like
        Input 2d array or image.
    wavelet : Pywt wavelet object
        Pywt wavelet object. Example: pywt.Wavelet('bior2.2')
    threshold : int
        Threshold on the wavelet coefficients.
    levels : int
        Wavelet levels to be used.
    thrmode : {'hard','soft'}, optional
        Mode of thresholding of the wavelet coefficients.
    
    Returns
    -------
    array_filtered : array_like
        Filtered array with the same dimensions and size of the input one. 
    
    Notes
    -----
    Full documentation of the PyWavelets package here:
    http://www.pybytes.com/pywavelets/
    
    For information on the builtin wavelets and how to use them:
    http://www.pybytes.com/pywavelets/regression/wavelet.html
    http://wavelets.pybytes.com
    
    """
    if not array.ndim==2:
        raise TypeError('Input array is not a frame or 2d array')
    
    WC = pywt.wavedec2(array, wavelet, level=levels)
    if thrmode=='hard':
        NWC = map(lambda x: pywt.thresholding.hard(x, threshold), WC)
    elif thrmode=='soft':
        NWC = map(lambda x: pywt.thresholding.soft(x, threshold), WC)
    else:
        raise ValueError('Threshold mode not recognized')
    array_filtered = pywt.waverec2(NWC, wavelet)
    
    return array_filtered


def convolution_gaussian2d(array, size_fwhm):
    """ Convolution with a 2d gaussian kernel created with Astropy.
    
    Parameters
    ----------
    array : array_like
        Input array with the 2d frame.
    size_fwhm : float
        Size in pixels of the FWHM of the gaussian kernel.

    Returns
    -------
    array_out: array_like
        Convolved image.
        
    """
    if not array.ndim==2:
        raise TypeError('Input array is not a frame or 2d array')
    
    gaus = Gaussian2DKernel(stddev=size_fwhm/SIGMA2FWHM)
    return convolve_fft(array, gaus)


def gaussian_filter_sp(array, size_fwhm, order=0):
    """ Multidimensional gaussian filter from scipy.ndimage.
    
    Parameters
    ----------
    array : array_like
        Input 2d array.
    size_fwhm : float
        Size in pixels of fwhm.
    order : int
        0 corresponds to a convolution with a gaussian kernel. 1, 2 or 3 means 
        convolution with the first, second or third derivatives of the gaussian.
        
    Returns
    -------
    array_out : array_like
        Filtered image.
    
    """
    if not array.ndim==2:
        raise TypeError('Input array is not a frame or 2d array')
    return gaussian_filter(array, sigma=size_fwhm/SIGMA2FWHM, order=order)
    
    
def gaussian_kernel(size, size_y=None):
    """ Gaussian kernel.
    """
    size = int(size)
    if not size_y:
        size_y = size
    else:
        size_y = int(size_y)
    x, y = np.mgrid[-size:size+1, -size_y:size_y+1]
    g = np.exp(-(x**2/float(size)+y**2/float(size_y)))

    fwhm = size
    fwhm_aper = photutils.CircularAperture((frame_center(g)), fwhm/2.)
    fwhm_aper_phot = photutils.aperture_photometry(g, fwhm_aper)
    g_norm = g/np.array(fwhm_aper_phot['aperture_sum'])
     
    return g_norm/g_norm.max()



