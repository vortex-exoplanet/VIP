#! /usr/bin/env python

"""
Module with frame/cube filtering functionalities.
"""



__author__ = 'Carlos Alberto Gomez Gonzalez, Valentin Christiaens'
__all__ = ['frame_filter_highpass',
           'frame_filter_lowpass',
           'frame_deconvolution',
           'cube_filter_highpass',
           'cube_filter_lowpass',
           'cube_filter_iuwt']

import warnings
try:
    import cv2
    no_opencv = False
except ImportError:
    msg = "Opencv python bindings are missing."
    warnings.warn(msg, ImportWarning)
    no_opencv = True
import numpy as np
from scipy.ndimage import median_filter
from skimage.restoration import richardson_lucy
from astropy.convolution import (convolve_fft, convolve, Gaussian2DKernel)
from astropy.convolution import interpolate_replace_nans as interp_nan
from astropy.stats import gaussian_fwhm_to_sigma
from .iuwt import iuwt_decomposition
from ..config import Progressbar


def cube_filter_iuwt(cube, coeff=5, rel_coeff=1, full_output=False):
    """
    Isotropic Undecimated Wavelet Transform filtering.

    Parameters
    ----------
    cube : numpy ndarray
        Input cube.
    coeff : int, optional
        Number of wavelet scales to be used in the decomposition.
    rel_coeff : int, optional
        Number of relevant coefficients. In other words how many wavelet scales
        will represent in a better way our data. One or two scales are enough
        for filtering our images.
    full_output : bool, optional
        If True, an additional cube with the multiscale decomposition of each
        frame will be returned.

    Returns
    -------
    cubeout : numpy ndarray
        Output cube with the filtered frames.

    If full_output is True the filtered cube is returned together with the a
    4d cube containing the multiscale decomposition of each frame.

    """
    cubeout = np.zeros_like(cube)
    cube_coeff = np.zeros((cube.shape[0], coeff, cube.shape[1], cube.shape[2]))
    n_frames = cube.shape[0]

    print('Decomposing frames with the Isotropic Undecimated Wavelet Transform')
    for i in Progressbar(range(n_frames)):
        res = iuwt_decomposition(cube[i], coeff, store_smoothed=False)
        cube_coeff[i] = res
        for j in range(rel_coeff):
            cubeout[i] += cube_coeff[i][j]

    if full_output:
        return cubeout, cube_coeff
    else:
        return cubeout


def cube_filter_highpass(array, mode='laplacian', verbose=True, **kwargs):
    """
    Apply ``frame_filter_highpass`` to the frames of a 3d or 4d cube.

    Parameters
    ----------
    array : numpy ndarray
        Input cube, 3d or 4d.
    mode : str, optional
        ``mode`` parameter to the ``frame_filter_highpass`` function. Defaults
        to a Laplacian high-pass filter.
    verbose : bool, optional
        If ``True`` timing and progress bar are shown.
    **kwargs : dict
        Passed through to the ``frame_filter_highpass`` function.

    Returns
    -------
    filtered : numpy ndarray
        High-pass filtered cube.

    """
    array_out = np.empty_like(array)

    if array.ndim == 3:
        for i in Progressbar(range(array.shape[0]), verbose=verbose):
            array_out[i] = frame_filter_highpass(array[i], mode=mode, **kwargs)
    elif array.ndim == 4:
        for i in Progressbar(range(array.shape[1]), verbose=verbose):
            for lam in range(array.shape[0]):
                array_out[lam][i] = frame_filter_highpass(array[lam][i],
                                                          mode=mode, **kwargs)
    else:
        raise TypeError('Input array is not a 3d or 4d cube')

    return array_out


def fft(array):
    """
    Perform the 2d discrete Fourier transform, using numpy's fft2 function.

    This produces a new representation of
    the image in which each pixel represents a spatial frequency and
    orientation, rather than an xy coordinate. When Fourier-transformed images
    are plotted graphically, the low frequencies are found at the centre; this
    is not what fft2 actually produces, so we need to also apply numpy's
    fftshift (centering low frequencies).

    """
    fft_array = np.fft.fftshift(np.fft.fft2(array))
    return fft_array


def ifft(array):
    """
    Get the inverse Fourier transform on the image.

    This produces an array of complex numbers whose real values correspond to
    the image in the original space (decentering).

    Notes
    -----
    A real function corresponds to a symmetric function in fourier space. As
    long as the operations we apply in the fourier space do not break this
    symmetry, the data returned by ``ifft`` should not containy any imaginary
    part.

    """
    new_array = np.fft.ifft2(np.fft.ifftshift(array)).real
    return new_array


def frame_filter_highpass(array, mode, median_size=5, kernel_size=5,
                          fwhm_size=5, btw_cutoff=0.2, btw_order=2,
                          hann_cutoff=5, psf=None, conv_mode='conv', 
                          mask=None):
    """
    High-pass filtering of input frame depending on parameter ``mode``.

    The filtered image properties will depend on the ``mode`` and the relevant
    parameters.

    Parameters
    ----------
    array : numpy ndarray
        Input array, 2d frame.
    mode : str
        Type of High-pass filtering.

        ``laplacian``
            applies a Laplacian filter with kernel size defined by
            ``kernel_size`` using the Opencv library.
        ``laplacian-conv``
            applies a Laplacian high-pass filter by defining a kernel (with
            ``kernel_size``) and using the ``convolve_fft`` Astropy function.
        ``median-subt``
            subtracts a median low-pass filtered version of the image.
        ``gauss-subt``
            subtracts a Gaussian low-pass filtered version of the image.
        ``fourier-butter``
            applies a high-pass 2D Butterworth filter in Fourier domain.
        ``hann``
            uses a Hann window.

    median_size : int, optional
        Size of the median box for the ``median-subt`` filter.
    kernel_size : int, optional
        Size of the Laplacian kernel used in ``laplacian`` mode. It must be an
        positive odd integer value.
    fwhm_size : float, optional
        Size of the Gaussian kernel used in ``gaus-subt`` mode.
    btw_cutoff : float, optional
        Frequency cutoff for low-pass 2d Butterworth filter used in
        ``fourier-butter`` mode.
    btw_order : int, optional
        Order of low-pass 2d Butterworth filter used in ``fourier-butter`` mode.
    hann_cutoff : float
        Frequency cutoff for the ``hann`` mode.
    psf: numpy ndarray, optional
        Input normalised and centered psf, 2d frame. Should be provided if 
        mode is set to 'psf'.
    conv_mode : {'conv', 'convfft'}, str optional
        'conv' uses the multidimensional gaussian filter from scipy.ndimage and
        'convfft' uses the fft convolution with a 2d Gaussian kernel.
    mask: numpy ndarray, optional
        Binary mask indicating where the low-pass filtered image should be 
        interpolated with astropy.convolution. This otion can be useful if the 
        low-pass filtered image is aimed to capture low-spatial frequency sky 
        signal, while avoiding a stellar halo (set to one in the binary mask).
        Note: only works with Gaussian kernel or PSF convolution.

    Returns
    -------
    filtered : numpy ndarray
        High-pass filtered image.

    """
    def butter2d_lp(size, cutoff, n=3):
        """
        Create low-pass 2D Butterworth filter.

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
        numpy ndarray
          filter kernel in 2D centered

        """
        if not 0 < cutoff <= 1:
            raise ValueError('Cutoff frequency must be between 0 and 1.')

        if not isinstance(n, int):
            raise ValueError('n must be an integer >= 1.')

        rows, cols = size
        x = np.linspace(-0.5, 0.5, cols) * cols
        y = np.linspace(-0.5, 0.5, rows) * rows

        # An array with every pixel = radius relative to center
        radius = np.sqrt((x**2)[np.newaxis] + (y**2)[:, np.newaxis])
        # The filter
        f = 1 / (1 + (radius / cutoff)**(2*n))
        return f

    def round_away(x):
        """
        Round to the *nearest* integer, half-away-from-zero.

        Parameters
        ----------
        x : array-like

        Returns
        -------
        r_rounded : array-like (float)

        Notes
        -----
        IDL ``ROUND`` rounds to the *nearest* integer (commercial rounding),
        unlike numpy's round/rint, which round to the nearest *even*
        value (half-to-even, financial rounding) as defined in IEEE-754
        standard.

        """
        return np.trunc(x + np.copysign(0.5, x))

    # --------------------------------------------------------------------------

    if array.ndim != 2:
        raise TypeError("Input array is not a frame or 2d array.")

    if mask is not None and (mode!='psf-subt' and mode!='gauss-subt'):
        msg="Masking option only available for gauss-subt and psf-subt modes"
        raise TypeError(msg)

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
        medianed = frame_filter_lowpass(array, 'median',
                                        median_size=median_size)
        filtered = array - medianed

    elif mode == 'gauss-subt':
        # Subtracting the low_pass filtered (median) image from the image itself
        gaussed = frame_filter_lowpass(array, 'gauss', fwhm_size=fwhm_size,
                                       conv_mode=conv_mode, mask=mask)
        filtered = array - gaussed

    elif mode == 'psf-subt':
        if psf is None:
            raise TypeError("psf should be provided for psf-subt mode")
        # Subtracting the low_pass filtered (median) image from the image itself
        psfed = frame_filter_lowpass(array, 'psf', psf=psf, mask=mask)
        filtered = array - psfed

    elif mode == 'fourier-butter':
        # Designs an n-th order high-pass 2D Butterworth filter
        filt = 1 - butter2d_lp(array.shape, cutoff=btw_cutoff, n=btw_order)
        array_fft = fft(array)
        fft_new = array_fft * filt
        filtered = ifft(fft_new)
        
    elif mode == 'hann':
        # TODO: this code could be shortened using np.convolve
        # see http://scipy-cookbook.readthedocs.io/items/SignalSmooth.html

        # create a Hanning profile window cut at the chosen frequency:
        npix = array.shape[0]
        cutoff = npix/2 * hann_cutoff
        cutoff_inside = round_away(np.minimum(cutoff, (npix/2 - 1))).astype(int)
        winsize = 2*cutoff_inside + 1
        win1d = np.hanning(winsize)
        win = 1 - np.outer(win1d, win1d)

        array_fft = fft(array)
        # remove high spatial frequency along the Hann profile:
        array_fft[npix//2 - cutoff_inside: npix//2 + cutoff_inside + 1,
                  npix//2 - cutoff_inside: npix//2 + cutoff_inside + 1] *= win

        filtered = ifft(array_fft)

    else:
        raise TypeError('Mode not recognized.')

    return filtered


def frame_filter_lowpass(array, mode='gauss', median_size=5, fwhm_size=5,
                         conv_mode='convfft', kernel_sz=None, psf=None, 
                         mask=None, iterate=True, half_res_y=False, **kwargs):
    """
    Low-pass filtering of input frame depending on parameter ``mode``.

    Parameters
    ----------
    array : numpy ndarray
        Input array, 2d frame.
    mode : {'median', 'gauss', 'psf'}, str optional
        Type of low-pass filtering.
    median_size : int, optional
        Size of the median box for filtering the low-pass median filter.
    fwhm_size : float, optional
        Size of the Gaussian kernel for the low-pass Gaussian filter.
    conv_mode : {'conv', 'convfft'}, str optional
        'conv' uses the multidimensional gaussian filter from scipy.ndimage and
        'convfft' uses the fft convolution with a 2d Gaussian kernel.
    kernel_sz: int or None, optional
        Size of the kernel in pixels for 2D Gaussian and Moffat convolutions.
        If None, astropy.convolution will automatically consider 8*radius 
        kernel sizes.
    psf: numpy ndarray, optional
        Input normalised and centered psf, 2d frame. Should be provided if 
        mode is set to 'psf'.
    mask: numpy ndarray, optional
        Binary mask indicating where the low-pass filtered image should be 
        interpolated with astropy.convolution. This option can be useful if the 
        low-pass filtered image is aimed to capture low-spatial frequency sky 
        signal, while avoiding a stellar halo (set to one in the binary mask).
        Note: only works with Gaussian kernel or PSF convolution.
    iterate: bool, opt
        If the first convolution leaves nans, whether to continue replacing 
        nans by interpolation until they are all replaced.
    half_res_y: bool, {True,False}, optional
        Whether the input data has only half the angular resolution vertically 
        compared to horizontally (e.g. the case for some IFUs); in other words
        there are always 2 rows of pixels with exactly the same values.
        If so, the kernel will also be squashed vertically by a factor 2.
        Only used if mode is 'gauss'
    **kwargs : dict
        Passed through to the astropy.convolution.convolve or convolve_fft
        function.

    Returns
    -------
    filtered : numpy ndarray
        Low-pass filtered image.

    """
    if array.ndim != 2:
        raise TypeError('Input array is not a frame or 2d array.')
    if not isinstance(median_size, int):
        raise ValueError('`Median_size` must be integer')
        
    if mask is not None:
        if mode== 'median':
            msg="Masking not available for median filter"
        if mask.shape != array.shape:
            msg = "Mask dimensions should be the same as array"
            raise TypeError(msg)

    if mode == 'median':
        # creating the low_pass filtered (median) image
        filtered = median_filter(array, median_size, mode='nearest')
    elif mode == 'gauss':
        # 2d Gaussian filter
        sigma = fwhm_size * gaussian_fwhm_to_sigma
        kernel_sz_y = kernel_sz
        if half_res_y:
            sigma_y = max(1, sigma//2)
            if kernel_sz is not None:
                kernel_sz_y = kernel_sz//2
                if kernel_sz_y%2 != kernel_sz%2:
                    kernel_sz_y+=1
        else:
            sigma_y=sigma
            
        if conv_mode == 'conv':
            filtered = convolve(array, Gaussian2DKernel(x_stddev=sigma, 
                                                        y_stddev=sigma_y,
                                                        x_size=kernel_sz,
                                                        y_size=kernel_sz_y),
                                mask=mask, **kwargs)
            if iterate and np.sum(np.isnan(filtered))>0:
                filtered = interp_nan(filtered, 
                                      Gaussian2DKernel(x_stddev=sigma, 
                                                       y_stddev=sigma_y,
                                                       x_size=kernel_sz,
                                                       y_size=kernel_sz_y),
                                      convolve=convolve)
        elif conv_mode == 'convfft':
            # FFT Convolution with a 2d gaussian kernel created with Astropy.
            filtered = convolve_fft(array, Gaussian2DKernel(x_stddev=sigma,
                                                            y_stddev=sigma_y, 
                                                            x_size=kernel_sz,
                                                            y_size=kernel_sz_y),
                                    mask=mask, **kwargs)
            if iterate and np.sum(np.isnan(filtered))>0:
                filtered = interp_nan(filtered, 
                                      Gaussian2DKernel(x_stddev=sigma, 
                                                       y_stddev=sigma_y,
                                                       x_size=kernel_sz,
                                                       y_size=kernel_sz_y),
                                      convolve=convolve_fft, **kwargs)
        else:
            raise TypeError('2d Gaussian filter mode not recognized')
    elif mode == 'psf':
        if psf is None:
            raise TypeError('psf should be provided for convolution')
        elif psf.ndim != 2:
            raise TypeError('Input psf is not a frame or 2d array.')
        if psf.shape[-1] > array.shape[-1]:
            raise TypeError('Input psf is larger than input array. Crop.')
        # psf convolution
        if conv_mode == 'conv':
            filtered = convolve(array, psf, mask=mask, **kwargs)
            if iterate and np.sum(np.isnan(filtered))>0:
                filtered = interp_nan(filtered, psf, convolve=convolve, 
                                      **kwargs)
        elif conv_mode == 'convfft':
            filtered = convolve_fft(array, psf, mask=mask, **kwargs)
            if iterate and np.sum(np.isnan(filtered))>0:
                filtered = interp_nan(filtered, psf, convolve=convolve_fft, 
                                      **kwargs)
    else:
        raise TypeError('Low-pass filter mode not recognized')
        
    return filtered


def cube_filter_lowpass(array, mode='gauss', median_size=5, fwhm_size=5,
                        conv_mode='conv', kernel_sz=None, verbose=True, 
                        psf=None, mask=None, iterate=True, **kwargs):
    """
    Apply ``frame_filter_lowpass`` to the frames of a 3d or 4d cube.

    Parameters
    ----------
    array : numpy ndarray
        Input cube, 3d or 4d.
    mode : str, optional
        See the documentation of the ``frame_filter_lowpass`` function.
    median_size : int, optional
        See the documentation of the ``frame_filter_lowpass`` function.
    fwhm_size : float, optional
        See the documentation of the ``frame_filter_lowpass`` function.
    conv_mode : str, optional
        See the documentation of the ``frame_filter_lowpass`` function.
    kernel_sz: int, optional
        See the documentation of the ``frame_filter_lowpass`` function.
    verbose : bool, optional
        If True timing and progress bar are shown.
    psf: numpy ndarray, optional
        Input normalised and centered psf, 2d frame. Should be provided if 
        mode is set to 'psf'.
    mask: numpy ndarray, optional
        Binary mask indicating where the low-pass filtered image should be 
        interpolated with astropy.convolution. This otion can be useful if the 
        low-pass filtered image is aimed to capture low-spatial frequency sky 
        signal, while avoiding a stellar halo (set to one in the binary mask).
        Note: only works with Gaussian kernel or PSF convolution.
    iterate: bool, opt
        If the first convolution leaves nans, whether to continue replacing 
        nans by interpolation until they are all replaced.
    **kwargs : dict
        Passed through to the astropy.convolution.convolve or convolve_fft
        function.
        
    Returns
    -------
    filtered : numpy ndarray
        Low-pass filtered cube.

    """
    array_out = np.empty_like(array)

    if array.ndim == 3:
        for i in Progressbar(range(array.shape[0]), verbose=verbose):
            array_out[i] = frame_filter_lowpass(array[i], mode, median_size,
                                                fwhm_size, conv_mode, 
                                                kernel_sz, psf, mask, iterate, 
                                                **kwargs)
    elif array.ndim == 4:
        for i in Progressbar(range(array.shape[1]), verbose=verbose):
            for lam in range(array.shape[0]):
                array_out[lam][i] = frame_filter_lowpass(array[lam][i], mode,
                                                         median_size, fwhm_size,
                                                         conv_mode, kernel_sz, 
                                                         psf, mask, iterate, 
                                                         **kwargs)
    else:
        raise TypeError('Input array is not a 3d or 4d cube')

    return array_out


def frame_deconvolution(array, psf, n_it=30):
    """
    Iterative image deconvolution following the scikit-image implementation 
    of the Richardson-Lucy algorithm.
    
    Considering an image that has been convolved by the point spread function
    of an instrument, the algorithm will sharpen the blurred 
    image through a user-defined number of iterations, which changes the
    regularisation.

    Reference: William Hadley Richardson, “Bayesian-Based Iterative Method of 
    Image Restoration”, J. Opt. Soc. Am. A 27, 1593-1607 (1972), 
    DOI:10.1364/JOSA.62.000055

    See also description at:
    https://en.wikipedia.org/wiki/Richardson%E2%80%93Lucy_deconvolution
    

    Parameters
    ----------
    array : numpy ndarray
        Input image, 2d frame.
    psf : numpy ndarray
        Input psf, 2d frame.    
    n_it : int, optional
        Number of iterations.

    Returns
    -------
    deconv : numpy ndarray
        Deconvolved image.

    """
    if array.ndim != 2:
        raise TypeError('Input array is not a frame or 2d array.')
    if psf.ndim != 2:
        raise TypeError('Input psf is not a frame or 2d array.')

    max_I = np.amax(array)
    min_I = np.amin(array)
    drange = max_I-min_I

    deconv = richardson_lucy((array-min_I)/drange, psf, iterations=n_it)
    deconv*=drange
    deconv+=min_I

    return deconv