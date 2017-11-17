#! /usr/bin/env python

"""
Module with frame/cube filtering functionalities
"""

from __future__ import division

__author__ = 'C. Gomez @ ULg'
__all__ = ['frame_filter_highpass',
           'frame_filter_lowpass',
           'cube_filter_highpass',
           'cube_filter_iuwt',
           'frame_filter_gaussian2d',
           'gaussian_kernel']

import numpy as np
import photutils
import pyprind
from scipy.ndimage import gaussian_filter, median_filter      
from astropy.convolution import convolve_fft, Gaussian2DKernel
from astropy.stats import gaussian_fwhm_to_sigma
from .shapes import frame_center
from ..exlib import iuwt


def cube_filter_iuwt(cube, coeff=5, rel_coeff=1, full_output=False):
    """
    
    Parameters
    ----------
    cube : array_like
        Input cube.
    coeff : int, optional
        Number of wavelet scales to be used in the decomposition.
    rel_coeff : int, optional
        Number of relevant coefficients. In other words how many wavelet scales
        will represent in a better way our data. One or two scales are enough
        for filtering our images.
    full_output : {False, True}, bool optional
        If True, an additional cube with the multiscale decomposition of each
        frame will be returned.
    
    Returns
    -------
    cubeout : array_like
        Output cube with the filtered frames.
    
    If full_output is True the filtered cube is returned together with the a 
    4d cube containing the multiscale decomposition of each frame.
    
    """
    cubeout = np.zeros_like(cube)
    cube_coeff = np.zeros((cube.shape[0], coeff, cube.shape[1], cube.shape[2]))
    n_frames = cube.shape[0]
    
    msg = 'Decomposing frames with the Isotropic Undecimated Wavelet Transform'
    bar = pyprind.ProgBar(n_frames, stream=1, title=msg)
    for i in range(n_frames):
        res = iuwt.iuwt_decomposition(cube[i], coeff, store_smoothed=False)
        cube_coeff[i] = res
        for j in range(rel_coeff):
            cubeout[i] += cube_coeff[i][j] 
        bar.update()
        
    if full_output:
        return cubeout, cube_coeff
    else:
        return cubeout


def cube_filter_highpass(array, mode, median_size=5, kernel_size=5, 
                          fwhm_size=5, btw_cutoff=0.2, btw_order=2):
    """ Wrapper of *frame_filter_highpass* for cubes or 3d arrays.

    Parameters
    ----------
    array : array_like
        Input 3d array.
    mode : {'kernel-conv', 'median-subt', 'gauss-subt', 'fourier-butter'}
        Type of High-pass filtering.
    median_size : int
        Size of the median box for filtering the low-pass median filter.
    kernel_size : 3, 5 or 7
        Size of the Laplacian kernel for convolution. 
    fwhm_size : int
        Size of the Gaussian kernel for the low-pass Gaussian filter.
    btw_cutoff : float
        Frequency cutoff for low-pass 2d Butterworth filter.
    btw_order : int
        Order of low-pass 2d Butterworth filter.
    
    Returns
    -------
    filtered : array_like
        High-pass filtered cube.
    """
    if not array.ndim==3:
        raise TypeError('Input array is not a cube or 3d array')
    
    n_frames = array.shape[0]
    array_out = np.zeros_like(array)
    msg = 'Applying the High-Pass filter on cube frames'
    bar = pyprind.ProgBar(n_frames, stream=1, title=msg)
    for i in range(n_frames):
        array_out[i] = frame_filter_highpass(array[i], mode, median_size, 
                                            kernel_size, fwhm_size, btw_cutoff, 
                                            btw_order)
        bar.update()
        
    return array_out
    

def fft(array):
    """ Performs the 2d discrete Fourier transform (using numpy's fft2 function) 
    on the data from the original image. This produces a new representation of 
    the image in which each pixel represents a spatial frequency and 
    orientation, rather than an xy coordinate. When Fourier-transformed images 
    are plotted graphically, the low frequencies are found at the centre; this 
    is not what fft2 actually produces, so we need to also apply numpy's 
    fftshift (centering low frequencies).
    """
    fft_array = np.fft.fftshift(np.fft.fft2(array))
    return fft_array
    
def ifft(array):
    """ Gets the inverse Fourier transform on the image. This produces an array 
    of complex numbers whose absolute values correspond to the image in the 
    original space (decentering).
    """
    new_array = np.abs(np.fft.ifft2(np.fft.ifftshift(array)))
    return new_array

def frame_filter_highpass(array, mode, median_size=5, kernel_size=5, 
                          fwhm_size=5, btw_cutoff=0.2, btw_order=2):
    """ High-pass filtering of input frame depending on parameter *mode*. The
    results are very different with different *mode* and varying the rest of
    parameters.
    
    Parameters
    ----------
    array : array_like
        Input array, 2d frame.
    mode : {'kernel-conv', 'median-subt', 'gauss-subt', 'fourier-butter'}
        Type of High-pass filtering.
    median_size : int
        Size of the median box for filtering the low-pass median filter.
    kernel_size : 3, 5 or 7
        Size of the Laplacian kernel for convolution. 
    fwhm_size : int
        Size of the Gaussian kernel for the low-pass Gaussian filter.
    btw_cutoff : float
        Frequency cutoff for low-pass 2d Butterworth filter.
    btw_order : int
        Order of low-pass 2d Butterworth filter.
    
    Returns
    -------
    filtered : array_like
        High-pass filtered image.
        
    """
    def butter2d_lp(size, cutoff, n=3):
        """ Create low-pass 2D Butterworth filter. 
        Function from PsychoPy library, credits to Jonathan Peirce, 2010

        Parameters
        ----------
        size : tuple
            size of the filter
        cutoff : float
            relative cutoff frequency of the filter (0 - 1.0)
        n : int, optional
            order of the filter, the higher n is the sharper
            the transition is.
        
        Returns
        -------
        numpy.ndarray
          filter kernel in 2D centered
        """
        if not 0 < cutoff <= 1.0:
            raise ValueError('Cutoff frequency must be between 0 and 1.0')
    
        if not isinstance(n, int):
            raise ValueError('n must be an integer >= 1')
    
        rows, cols = size
        x =  np.linspace(-0.5, 0.5, cols) * cols
        y =  np.linspace(-0.5, 0.5, rows) * rows
    
        # An array with every pixel = radius relative to center
        radius = np.sqrt((x**2)[np.newaxis] + (y**2)[:, np.newaxis])
        # The filter
        f = 1 / (1.0 + (radius / cutoff)**(2*n))   
        return f
    
    #---------------------------------------------------------------------------
    
    if not array.ndim==2:
        raise TypeError('Input array is not a frame or 2d array')
    
    if mode=='kernel-conv':
        # Performs convolution with Laplacian high-pass kernels. 
        # a simple and very narrow hp filter
        # Kernel "Laplacian" of size 3x3+1+1 with values from -1 to 8
        # Forming a output range from -8 to 8 (Zero-Summing)
        kernel3 = np.array([[-1, -1, -1], 
                            [-1,  8, -1],
                            [-1, -1, -1]])
        #kernel3 = np.array([[0,  -1,  0],
        #                    [-1, 4, -1],
        #                    [0, -1,  0]])
        #kernel3 = np.array([[-0.17, -0.67, -0.17],
        #                    [-0.67, 3.33, -0.67],
        #                    [-0.17, -0.67, -0.17]])
        #kernel5 = np.array([[-1, -1, -1, -1, -1],
        #                    [-1,  1,  2,  1, -1],
        #                    [-1,  2,  4,  2, -1],
        #                    [-1,  1,  2,  1, -1],
        #                    [-1, -1, -1, -1, -1]])
        # Kernel "Laplacian" of size 5x5+2+2 with values from -4 to 4
        # Forming a output range from -24 to 24 (Zero-Summing)
        kernel5 = np.array([[-4, -1,  0, -1, -4],
                            [-1,  2,  3,  2, -1],
                            [ 0,  3,  4,  3,  0],
                            [-1,  2,  3,  2, -1],
                            [-4, -1,  0, -1, -4]])
        # above /4. +1 in central px
        #kernel5 = np.array([[-0.25, -0.25, -0.25, -0.25, -0.25],
        #                    [-0.25,  0.25,  0.5 ,  0.25, -0.25],
        #                    [-0.25,  0.5 ,  2.  ,  0.5 , -0.25],
        #                    [-0.25,  0.25,  0.5 ,  0.25, -0.25],
        #                    [-0.25, -0.25, -0.25, -0.25, -0.25]])
        # Kernel "Laplacian" of size 7x7+3+3 with values from -10 to 8
        # Forming a output range from -1e+02 to 1e+02 (Zero-Summing)
        kernel7 = np.array([[-10, -5, -2, -1, -2, -5, -10],
                            [-5,   0,  3,  4,  3,  0,  -5],
                            [-2,   3,  6,  7,  6,  3,  -2],
                            [-1,   4,  7,  8,  7,  4,  -1],
                            [-2,   3,  6,  7,  6,  3,  -2],
                            [-5,   0,  3,  4,  3,  0,  -5],
                            [-10, -5, -2, -1, -2, -5, -10]])
        if kernel_size==3:  kernel = kernel3
        elif kernel_size==5:  kernel = kernel5
        elif kernel_size==7:  kernel = kernel7
        filtered = convolve_fft(array, kernel)
    
    elif mode=='median-subt':
        # Subtracting the low_pass filtered (median) image from the image itself  
        medianed = median_filter(array, median_size, mode='nearest')
        filtered = array - medianed
    
    elif mode=='gauss-subt':
        # Subtracting the low_pass filtered (median) image from the image itself  
        gaussed = frame_filter_gaussian2d(array, fwhm_size, mode='conv')
        filtered = array - gaussed
        
    elif mode=='fourier-butter':
        # Designs an n-th order high-pass 2D Butterworth filter
        filt = butter2d_lp(array.shape, cutoff=btw_cutoff, n=btw_order)
        filt = 1. - filt                        
        array_fft = fft(array)
        fft_new = array_fft * filt
        filtered = ifft(fft_new)        
        
    else:
        raise TypeError('Mode not recognized')    
        
    return filtered


def frame_filter_lowpass(array, mode, median_size=5, fwhm_size=5):
    """ Low-pass filtering of input frame depending on parameter *mode*. 
    
    Parameters
    ----------
    array : array_like
        Input array, 2d frame.
    mode : {'median', 'gauss'}
        Type of low-pass filtering.
    median_size : int
        Size of the median box for filtering the low-pass median filter.
    fwhm_size : int
        Size of the Gaussian kernel for the low-pass Gaussian filter.
    
    Returns
    -------
    filtered : array_like
        Low-pass filtered image.
        
    """
    if not array.ndim==2:
        raise TypeError('Input array is not a frame or 2d array')
       
    if mode=='median':
        # creating the low_pass filtered (median) image
        filtered = median_filter(array, int(median_size), mode='nearest')
    
    elif mode=='gauss':
        # creating the low_pass filtered (median) image 
        filtered = frame_filter_gaussian2d(array, fwhm_size, mode='conv')    
        
    else:
        raise TypeError('Mode not recognized')    
        
    return filtered


def frame_filter_gaussian2d(array, size_fwhm, mode='conv'):
    """ 2d Gaussian filter. 
    
    Parameters
    ----------
    array : array_like
        Input array, 2d frame.
    size_fwhm : float
        Size in pixels of the FWHM of the gaussian kernel.
    mode : {'conv', 'convfft'}
        'conv' uses the multidimensional gaussian filter from scipy.ndimage and
        'convfft' uses the fft convolution with a 2d Gaussian kernel.
        
    Returns
    -------
    filtered : array_like
        Convolved image.
        
    """
    if not array.ndim==2:
        raise TypeError('Input array is not a frame or 2d array')
    
    if mode=='conv':
        filtered = gaussian_filter(array, sigma=size_fwhm*gaussian_fwhm_to_sigma, 
                                   order=0, mode='nearest')
    elif mode=='convfft':
        # FFT Convolution with a 2d gaussian kernel created with Astropy.
        gaus = Gaussian2DKernel(stddev=size_fwhm*gaussian_fwhm_to_sigma)
        filtered = convolve_fft(array, gaus)
    else:
        raise TypeError('Mode not recognized')
    
    return filtered
   
    
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



