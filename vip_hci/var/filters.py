#! /usr/bin/env python

"""
Module with frame/cube filtering functionalities
"""

from __future__ import division

__author__ = 'Carlos Alberto Gomez Gonzalez'
__all__ = ['frame_filter_highpass',
           'frame_filter_lowpass',
           'cube_filter_highpass',
           'cube_filter_iuwt',
           'frame_filter_gaussian2d',
           'gaussian_kernel']

import warnings
try:
    import cv2
    no_opencv = False
except ImportError:
    msg = "Opencv python bindings are missing."
    warnings.warn(msg, ImportWarning)
    no_opencv = True
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
    
    msg = 'Decomposing frames with the Isotropic Undecimated Wavelet Transform.'
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


def cube_filter_highpass(array, mode='laplacian', median_size=5, kernel_size=5,
                         fwhm_size=5, btw_cutoff=0.2, btw_order=2,
                         verbose=True):
    """ Wrapper of ``frame_filter_highpass`` for cubes or 3d arrays.

    Parameters
    ----------
    array : array_like
        Input 3d array.
    mode : str, optional
        See the documentation of the ``frame_filter_highpass`` function.
    median_size : int, optional
        See the documentation of the ``frame_filter_highpass`` function.
    kernel_size : int, optional
        See the documentation of the ``frame_filter_highpass`` function.
    fwhm_size : int, optional
        See the documentation of the ``frame_filter_highpass`` function.
    btw_cutoff : float
        See the documentation of the ``frame_filter_highpass`` function.
    btw_order : int
        See the documentation of the ``frame_filter_highpass`` function.
    verbose : boolean, optional
        If True timing and progress bar are shown.
    
    Returns
    -------
    filtered : array_like
        High-pass filtered cube.
    """
    if not array.ndim == 3:
        raise TypeError('Input array is not a cube or 3d array')
    
    n_frames = array.shape[0]
    array_out = np.zeros_like(array)
    if verbose:
        msg = 'Applying the high-pass filter on cube frames:'
        bar = pyprind.ProgBar(n_frames, stream=1, title=msg, bar_char='.')
    for i in range(n_frames):
        array_out[i] = frame_filter_highpass(array[i], mode, median_size, 
                                            kernel_size, fwhm_size, btw_cutoff, 
                                            btw_order)
        if verbose:
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
    """ High-pass filtering of input frame depending on parameter ``mode``. The
    filtered image properties will depend on the ``mode`` and the relevant
    parameters.
    
    Parameters
    ----------
    array : array_like
        Input array, 2d frame.
    mode : {'laplacian', 'laplacian-conv', 'median-subt', 'gauss-subt', 'fourier-butter'}
        Type of High-pass filtering. ``laplacian`` applies a Laplacian fiter
        with kernel size defined by ``kernel_size`` using the Opencv library.
        ``laplacian-conv`` applies a Laplacian high-pass filter by defining a
        kernel (with ``kernel_size``) and using the ``convolve_fft`` Astropy
        function. ``median-subt`` subtracts a median low-pass filtered version
        of the image. ``gauss-subt`` subtracts a Gaussian low-pass filtered
        version of the image. ``fourier-butter`` applies a high-pass 2D
        Butterworth filter in Fourier domain.
    median_size : int, optional
        Size of the median box for filtering the low-pass median filter.
    kernel_size : int, optional
        Size of the Laplacian kernel used in ``laplacian`` mode. It must be an
        positive odd integer value.
    fwhm_size : int, optional
        Size of the Gaussian kernel used in ``gaus-subt`` mode.
    btw_cutoff : float
        Frequency cutoff for low-pass 2d Butterworth filter used in
        ``fourier-butter`` mode.
    btw_order : int
        Order of low-pass 2d Butterworth filter used in ``fourier-butter`` mode.
    
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
            raise ValueError('Cutoff frequency must be between 0 and 1.0.')
    
        if not isinstance(n, int):
            raise ValueError('n must be an integer >= 1.')
    
        rows, cols = size
        x = np.linspace(-0.5, 0.5, cols) * cols
        y = np.linspace(-0.5, 0.5, rows) * rows
    
        # An array with every pixel = radius relative to center
        radius = np.sqrt((x**2)[np.newaxis] + (y**2)[:, np.newaxis])
        # The filter
        f = 1 / (1.0 + (radius / cutoff)**(2*n))   
        return f
    
    #---------------------------------------------------------------------------
    
    if not array.ndim == 2:
        raise TypeError("Input array is not a frame or 2d array.")
    
    if mode == 'laplacian':
        # Applying a Laplacian high-pass kernel
        if kernel_size % 2 == 0 or kernel_size < 0:
            raise ValueError("Kernel size must be an odd and positive value.")
        if not no_opencv:
            msg = "Opencv bindings are missing. Trying a convolution with a "
            msg += "Laplacian kernel instead."

        filtered = cv2.Laplacian(-array, cv2.CV_32F, ksize=kernel_size)

    elif mode == 'laplacian-conv':
        # Applying a Laplacian high-pass kernel defining a kernel and using
        # the convolve_fft Astropy function
        kernel3 = np.array([[-1, -1, -1],
                            [-1, 8, -1],
                            [-1, -1, -1]])
        kernel5 = np.array([[-4, -1, 0, -1, -4],
                            [-1, 2, 3, 2, -1],
                            [0, 3, 4, 3, 0],
                            [-1, 2, 3, 2, -1],
                            [-4, -1, 0, -1, -4]])
        kernel7 = np.array([[-10, -5, -2, -1, -2, -5, -10],
                            [-5, 0, 3, 4, 3, 0, -5],
                            [-2, 3, 6, 7, 6, 3, -2],
                            [-1, 4, 7, 8, 7, 4, -1],
                            [-2, 3, 6, 7, 6, 3, -2],
                            [-5, 0, 3, 4, 3, 0, -5],
                            [-10, -5, -2, -1, -2, -5, -10]])
        if kernel_size == 3:
            kernel = kernel3
        elif kernel_size == 5:
            kernel = kernel5
        elif kernel_size == 7:
            kernel = kernel7
        else:
            raise ValueError('Kernel size must be either 3, 5 or 7.')
        filtered = convolve_fft(array, kernel, normalize_kernel=False,
                                nan_treatment='fill')
    
    elif mode == 'median-subt':
        # Subtracting the low_pass filtered (median) image from the image itself  
        medianed = median_filter(array, median_size, mode='nearest')
        filtered = array - medianed
    
    elif mode == 'gauss-subt':
        # Subtracting the low_pass filtered (median) image from the image itself  
        gaussed = frame_filter_gaussian2d(array, fwhm_size, mode='conv')
        filtered = array - gaussed
        
    elif mode == 'fourier-butter':
        # Designs an n-th order high-pass 2D Butterworth filter
        filt = butter2d_lp(array.shape, cutoff=btw_cutoff, n=btw_order)
        filt = 1. - filt                        
        array_fft = fft(array)
        fft_new = array_fft * filt
        filtered = ifft(fft_new)        
        
    else:
        raise TypeError('Mode not recognized.')
        
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
        raise TypeError('Input array is not a frame or 2d array.')
       
    if mode=='median':
        # creating the low_pass filtered (median) image
        filtered = median_filter(array, int(median_size), mode='nearest')
    
    elif mode=='gauss':
        # creating the low_pass filtered (median) image 
        filtered = frame_filter_gaussian2d(array, fwhm_size, mode='conv')    
        
    else:
        raise TypeError('Mode not recognized.')
        
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
        raise TypeError('Input array is not a frame or 2d array.')
    
    if mode=='conv':
        filtered = gaussian_filter(array, sigma=size_fwhm*gaussian_fwhm_to_sigma, 
                                   order=0, mode='nearest')
    elif mode=='convfft':
        # FFT Convolution with a 2d gaussian kernel created with Astropy.
        gaus = Gaussian2DKernel(stddev=size_fwhm*gaussian_fwhm_to_sigma)
        filtered = convolve_fft(array, gaus)
    else:
        raise TypeError('Mode not recognized.')
    
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



