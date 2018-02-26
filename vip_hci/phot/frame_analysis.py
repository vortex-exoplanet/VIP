#! /usr/bin/env python

"""

"""

from __future__ import division
from __future__ import print_function

__author__ = 'Carlos Alberto Gomez Gonzalez'
__all__ = ['frame_quick_report']

import numpy as np
import photutils
from skimage import draw
from .snr import snr_ss
from ..conf.utils_conf import sep


def frame_quick_report(array, fwhm, source_xy=None, verbose=True):
    """ Gets information from given frame: Integrated flux in aperture, SNR of
    central pixel (max or given coordinates), mean SNR in aperture.
    
    Parameters
    ----------
    array : array_like
        2d array or input frame.
    fwhm : float
        Size of the FWHM in pixels.
    source_xy : tuple of floats or list (of tuples of floats)
        X and Y coordinates of the center(s) of the source(s).
    verbose : {True, False}, bool optional
        If True prints to stdout the frame info.
        
    Returns
    -------
    source_xy : tuple of floats or list (of tuples of floats)
        X and Y coordinates of the center(s) of the source(s).
    obj_flux : list of floats
        Integrated flux in aperture.
    snr_centpx : list of floats
        S/N of the ``source_xy`` pixels.
    meansnr_pixels : list of floats
        Mean S/N of pixels in 1xFWHM apertures centered on ``source_xy``.
    """
    if not array.ndim == 2:
        raise TypeError('Array is not 2d.')

    if not isinstance(source_xy, list):
        source_xy = [source_xy]

    obj_flux = []
    meansnr_pixels = []
    snr_centpx = []
    for xy in source_xy:
        if xy is not None:
            x, y = xy
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
        obj_flux_i = photutils.aperture_photometry(array, aper, method='exact')
        obj_flux_i = obj_flux_i['aperture_sum'][0]

        # we get the mean and stddev of SNRs on aperture
        yy, xx = draw.circle(y, x, fwhm/2.)
        snr_pixels_i = [snr_ss(array, (x_, y_), fwhm, plot=False, verbose=False)\
                        for y_, x_ in zip(yy, xx)]
        meansnr_i = np.mean(snr_pixels_i)
        stdsnr_i = np.std(snr_pixels_i)
        pxsnr_i = snr_ss(array, (x, y), fwhm, plot=False, verbose=False)

        obj_flux.append(obj_flux_i)
        meansnr_pixels.append(meansnr_i)
        snr_centpx.append(pxsnr_i)

        if verbose:
            msg0 = 'Flux in a centered 1xFWHM circular aperture = {:.3f}'
            print(msg0.format(obj_flux_i))
            print('Central pixel S/N = {:.3f}'.format(pxsnr_i))
            print(sep)
            print('Inside a centered 1xFWHM circular aperture:')
            msg1 = 'Mean S/N (shifting the aperture center) = {:.3f}'
            print(msg1.format(meansnr_i))
            msg2 = 'Max S/N (shifting the aperture center) = {:.3f}'
            print(msg2.format(np.max(snr_pixels_i)))
            msg3 = 'stddev S/N (shifting the aperture center) = {:.3f}'
            print(msg3.format(stdsnr_i))
            print(sep)
    
    return source_xy, obj_flux, snr_centpx, meansnr_pixels


