#! /usr/bin/env python

"""
2d fitting and creation of synthetic PSFs.
"""
from __future__ import division, print_function

__author__ = 'Carlos Alberto Gomez Gonzalez'
__all__ = ['create_synth_psf',
           'fit_2dgaussian',
           'fit_2dmoffat',
           'fit_2dairydisk']

import numpy as np
import pandas as pd
import photutils
from astropy.modeling import models, fitting
from astropy.stats import (gaussian_sigma_to_fwhm, gaussian_fwhm_to_sigma,
                           sigma_clipped_stats)
from .shapes import get_square, frame_center
from .plotting import pp_subplots


def create_synth_psf(model='gauss', shape=(9, 9), amplitude=1, x_mean=None,
                     y_mean=None, fwhm=4, theta=0, gamma=None, alpha=1,
                     radius=None, msdi=False):
    """ Creates a synthetic 2d or 3d PSF with a 2d model: Airy disk, Gaussian or
    Moffat, depending on ``model``.

    Parameters
    ----------
    model : {'gauss', 'moff', 'airy'}, str optional
        Model to be used to create the synthetic PSF.
    shape : tuple of ints, optional
        Shape of the output 2d array.
    amplitude : float, optional
        Value of the amplitude of the 2d distribution.
    x_mean : float or None, optional
        Value of the centroid in X of the distributions: the mean of the
        Gaussian or the location of the maximum of the Moffat or Airy disk
        models. If None, the centroid is placed at the center of the array.
    y_mean : float or None, optional
        Value of the centroid in Y of the distributions: the mean of the
        Gaussian or the location of the maximum of the Moffat or Airy disk
        models. If None, the centroid is placed at the center of the array.
    fwhm : float, tuple of floats, list or np.ndarray, optional
        FWHM of the model in pixels. For the Gaussian case, it controls the
        standard deviation of the Gaussian. If a tuple is given, then the
        Gaussian will be elongated (fwhm in x, fwhm in y). For the Moffat, it is
        related to the gamma and alpha parameters. For the Airy disk, it is
        related to the radius (of the first zero) parameter. If ``msdi`` is True
        then ``fwhm`` must be a list of 1d np.ndarray (for example for
        SPHERE/IFS this sounds like a reasonable FWHM: np.linspace(4.5,6.7,39)).
    theta : float, optional
        Rotation angle in degrees of the Gaussian.
    gamma : float or None, optional
        Gamma parameter of core width of the Moffat model. If None, then it is
        calculated to correspond to the given ``fwhm``.
    alpha : float, optional
        Power index of the Moffat model.
    radius : float or None, optional
        The radius of the Airy disk (radius of the first zero). If None, then it
        is calculated to correspond to the given ``fwhm``.
    msdi : bool, optional
        Creates a 3d PSF, for emulating an IFS PSF.

    Returns
    -------
    im : array_like
        2d array with given ``shape`` and containing the synthetic PSF.

    Notes
    -----
    http://docs.astropy.org/en/stable/api/astropy.modeling.functional_models.Gaussian2D.html
    http://docs.astropy.org/en/stable/api/astropy.modeling.functional_models.Moffat2D.html
    http://docs.astropy.org/en/stable/api/astropy.modeling.functional_models.AiryDisk2D.html

    https://www.gnu.org/software/gnuastro/manual/html_node/PSF.html
    web.ipac.caltech.edu/staff/fmasci/home/astro_refs/PSFtheory.pdf
    web.ipac.caltech.edu/staff/fmasci/home/astro_refs/PSFsAndSampling.pdf
    """
    # 2d case
    if not msdi:
        sizex, sizey = shape
        if x_mean is None or y_mean is None:
            y_mean, x_mean = frame_center(np.zeros((sizey, sizex)))
        x = np.arange(sizex)
        y = np.arange(sizey)
        x, y = np.meshgrid(x, y)

        if model == 'gauss':
            if isinstance(fwhm, (tuple, list)):
                fwhm_x, fwhm_y = fwhm
            else:
                fwhm_y = fwhm
                fwhm_x = fwhm
            gauss = models.Gaussian2D(amplitude=amplitude, x_mean=x_mean,
                                      y_mean=y_mean,
                                      x_stddev=fwhm_x * gaussian_fwhm_to_sigma,
                                      y_stddev=fwhm_y * gaussian_fwhm_to_sigma,
                                      theta=np.deg2rad(theta))
            im = gauss(x, y)
        elif model == 'moff':
            if gamma is None and fwhm is not None:
                gamma = fwhm / (2. * np.sqrt(2 ** (1 / alpha) - 1))
            moffat = models.Moffat2D(amplitude=amplitude, x_0=x_mean,
                                     y_0=y_mean, gamma=gamma, alpha=alpha)
            im = moffat(x, y)
        elif model == 'airy':
            if radius is None and fwhm is not None:
                diam_1st_zero = (fwhm * 2.44) / 1.028
                radius = diam_1st_zero / 2.
            airy = models.AiryDisk2D(amplitude=amplitude, x_0=x_mean,
                                     y_0=y_mean, radius=radius)
            im = airy(x, y)
        return im
    # 3d case
    else:
        if not isinstance(fwhm, (list, np.ndarray)):
            raise ValueError('`Fwhm` must be a 1d vector')

        cube = []
        for fwhm_i in fwhm:
            cube.append(create_synth_psf(model, shape, amplitude, x_mean,
                                         y_mean, fwhm_i, theta, gamma, alpha,
                                         radius))
        cube = np.array(cube)
        return cube


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
        
    If ``full_output`` is True it returns a Pandas dataframe containing the
    following columns:
    'amplitude' : Float value. Amplitude of the Gaussian.
    'centroid_x' : Float value. X coordinate of the centroid.
    'centroid_y' : Float value. Y coordinate of the centroid.
    'fwhm_x' : Float value. FHWM in X [px].
    'fwhm_y' : Float value. FHWM in Y [px].
    'theta' : Float value. Rotation angle.
    
    """
    if array.ndim != 2:
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
    fitter = fitting.LevMarLSQFitter()
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
    theta = np.rad2deg(fit.theta.value)
    
    if debug:
        if threshold:
            msg = ['Subimage thresholded', 'Model', 'Residuals']
        else:
            msg = ['Subimage', 'Model', 'Residuals']
        pp_subplots(psf_subimage, fit(x, y), psf_subimage-fit(x, y),
                    grid=True, gridspacing=1, label=msg)
        print('FWHM_y =', fwhm_y)
        print('FWHM_x =', fwhm_x, '\n')
        print('centroid y =', mean_y)
        print('centroid x =', mean_x)
        print('centroid y subim =', fit.y_mean.value)
        print('centroid x subim =', fit.x_mean.value, '\n')
        print('amplitude =', amplitude)
        print('theta =', theta)
    
    if full_output:
        return pd.DataFrame({'centroid_y': mean_y, 'centroid_x': mean_x,
                             'fwhm_y': fwhm_y, 'fwhm_x': fwhm_x,
                             'amplitude': amplitude, 'theta': theta}, index=[0])
    else:
        return mean_y, mean_x


def fit_2dmoffat(array, crop=False, cent=None, cropsize=15, fwhm=4,
                 threshold=False, sigfactor=6, full_output=False, debug=False):
    """ Fitting a 2D Moffat to the 2D distribution of the data.

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
    fwhm : float, optional
        Initial values for the FWHM of the fitted 2d Moffat, in px.
    threshold : bool, optional
        If True the background pixels (estimated using sigma clipped statistics)
        will be replaced by small random Gaussian noise.
    sigfactor : int, optional
        The background pixels will be thresholded before fitting a 2d Moffat
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

    If ``full_output`` is True it returns a Pandas dataframe containing the
    following columns:
    'alpha': Float value. Alpha parameter.
    'amplitude' : Float value. Moffat Amplitude.
    'centroid_x' : Float value. X coordinate of the centroid.
    'centroid_y' : Float value. Y coordinate of the centroid.
    'fwhm' : Float value. FHWM [px].
    'gamma' : Float value. Gamma parameter.

    """
    if array.ndim != 2:
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

    # Creating the 2D Moffat model
    init_amplitude = np.ptp(psf_subimage)
    xcom, ycom = photutils.centroid_com(psf_subimage)
    moffat = models.Moffat2D(amplitude=init_amplitude, x_0=xcom, y_0=ycom,
                             gamma=fwhm / 2., alpha=1)
    # Levenberg-Marquardt algorithm
    fitter = fitting.LevMarLSQFitter()
    y, x = np.indices(psf_subimage.shape)
    fit = fitter(moffat, x, y, psf_subimage)

    if crop:
        mean_y = fit.y_0.value + suby
        mean_x = fit.x_0.value + subx
    else:
        mean_y = fit.y_0.value
        mean_x = fit.x_0.value

    fwhm = fit.fwhm
    amplitude = fit.amplitude.value
    alpha = fit.alpha.value
    gamma = fit.gamma.value

    if debug:
        if threshold:
            msg = ['Subimage thresholded', 'Model', 'Residuals']
        else:
            msg = ['Subimage', 'Model', 'Residuals']
        pp_subplots(psf_subimage, fit(x, y), psf_subimage - fit(x, y),
                    grid=True, gridspacing=1, label=msg)
        print('FWHM =', fwhm)
        print('centroid y =', mean_y)
        print('centroid x =', mean_x)
        print('centroid y subim =', fit.y_0.value)
        print('centroid x subim =', fit.x_0.value, '\n')
        print('amplitude =', amplitude)
        print('alpha =', alpha)
        print('gamma =', gamma)

    if full_output:
        return pd.DataFrame({'centroid_y': mean_y, 'centroid_x': mean_x,
                             'fwhm': fwhm, 'alpha': alpha, 'gamma': gamma,
                             'amplitude': amplitude}, index=[0])
    else:
        return mean_y, mean_x


def fit_2dairydisk(array, crop=False, cent=None, cropsize=15, fwhm=4,
                 threshold=False, sigfactor=6, full_output=False, debug=False):
    """ Fitting a 2D Moffat to the 2D distribution of the data.

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
    fwhm : float, optional
        Initial values for the FWHM of the fitted 2d Moffat, in px.
    threshold : bool, optional
        If True the background pixels (estimated using sigma clipped statistics)
        will be replaced by small random Gaussian noise.
    sigfactor : int, optional
        The background pixels will be thresholded before fitting a 2d Moffat
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

    If ``full_output`` is True it returns a Pandas dataframe containing the
    following columns:
    'alpha': Float value. Alpha parameter.
    'amplitude' : Float value. Moffat Amplitude.
    'centroid_x' : Float value. X coordinate of the centroid.
    'centroid_y' : Float value. Y coordinate of the centroid.
    'fwhm' : Float value. FHWM [px].
    'gamma' : Float value. Gamma parameter.

    """
    if array.ndim != 2:
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

    # Creating the 2d Airy disk model
    init_amplitude = np.ptp(psf_subimage)
    xcom, ycom = photutils.centroid_com(psf_subimage)
    diam_1st_zero = (fwhm * 2.44) / 1.028
    airy = models.AiryDisk2D(amplitude=init_amplitude, x_0=xcom, y_0=ycom,
                             radius=diam_1st_zero/2.)
    # Levenberg-Marquardt algorithm
    fitter = fitting.LevMarLSQFitter()
    y, x = np.indices(psf_subimage.shape)
    fit = fitter(airy, x, y, psf_subimage)

    if crop:
        mean_y = fit.y_0.value + suby
        mean_x = fit.x_0.value + subx
    else:
        mean_y = fit.y_0.value
        mean_x = fit.x_0.value

    amplitude = fit.amplitude.value
    radius = fit.radius.value
    fwhm = ((radius * 1.028) / 2.44) * 2

    if debug:
        if threshold:
            msg = ['Subimage thresholded', 'Model', 'Residuals']
        else:
            msg = ['Subimage', 'Model', 'Residuals']
        pp_subplots(psf_subimage, fit(x, y), psf_subimage - fit(x, y),
                    grid=True, gridspacing=1, label=msg)
        print('FWHM =', fwhm)
        print('centroid y =', mean_y)
        print('centroid x =', mean_x)
        print('centroid y subim =', fit.y_0.value)
        print('centroid x subim =', fit.x_0.value, '\n')
        print('amplitude =', amplitude)
        print('radius =', radius)

    if full_output:
        return pd.DataFrame({'centroid_y': mean_y, 'centroid_x': mean_x,
                             'fwhm': fwhm, 'radius': radius,
                             'amplitude': amplitude}, index=[0])
    else:
        return mean_y, mean_x