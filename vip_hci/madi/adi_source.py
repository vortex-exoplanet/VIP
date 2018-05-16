#! /usr/bin/env python

"""
Implementation of a median subtraction algorithm for model PSF subtraction in
high-contrast imaging sequences. In the case of ADI, the algorithm is based on
[MAR06]_. The ADI+IFS method, is an extension of this basic idea to
multi-spectral cubes.

.. [MAR06]
   | Marois et al. 2006
   | **Angular Differential Imaging: A Powerful High-Contrast Imaging
     Technique**
   | *The Astrophysical Journal, Volume 641, Issue 1, pp. 556-564*
   | `https://arxiv.org/abs/astro-ph/0512335
     <https://arxiv.org/abs/astro-ph/0512335>`_
"""

from __future__ import division, print_function

__author__ = 'Carlos Alberto Gomez Gonzalez'
__all__ = ['adi']

import numpy as np
import itertools as itt
from multiprocessing import Pool, cpu_count
from ..conf import time_ini, timing
from ..var import get_annulus, mask_circle
from ..preproc import (cube_derotate, cube_collapse, check_pa_vector,
                       check_scal_vector)
from ..preproc import cube_rescaling_wavelengths as scwave
from ..conf import Progressbar
from ..conf.utils_conf import eval_func_tuple as EFT
from .adi_utils import _find_indices, _define_annuli


array = None


def adi(cube, angle_list, scale_list=None, fwhm=4, radius_int=0, asize=2,
        delta_rot=1, mode='fullfr', nframes=4, imlib='opencv',
        interpolation='lanczos4', collapse='median', nproc=1, full_output=False,
        verbose=True):
    """ Implementation of a median subtraction algorithm for model PSF
     subtraction in high-contrast imaging sequences. In the case of ADI, the
    algorithm is based on [MAR06]_. The ADI+IFS method, is an extension of this
    basic idea to multi-spectral cubes.
    
    Parameters
    ----------
    cube : array_like, 3d
        Input cube.
    angle_list : array_like, 1d
        Corresponding parallactic angle for each frame.
    scale_list :
        Scaling factors in case of IFS data (ADI+mSDI cube). Usually, the
        scaling factors are the central channel wavelength divided by the
        shortest wavelength in the cube (more thorough approaches can be used
        to get the scaling factors). This scaling factors are used to re-scale
        the spectral channels and align the speckles.
    fwhm : float
        Known size of the FHWM in pixels to be used. Default is 4.
    radius_int : int, optional
        The radius of the innermost annulus. By default is 0, if >0 then the 
        central circular area is discarded.
    asize : int, optional
        The size of the annuli, in FWHM. Default is 2.
    delta_rot : int, optional
        Factor for increasing the parallactic angle threshold, expressed in
        FWHM. Default is 1 (excludes 1 FHWM on each side of the considered
        frame).
    mode : {"fullfr","annular"}, str optional
        In "simple" mode only the median frame is subtracted, in "annular" mode
        also the 4 closest frames given a PA threshold (annulus-wise) are 
        subtracted.
    nframes : even int optional
        Number of frames to be used for building the optimized reference PSF 
        when working in annular mode.
    imlib : str, optional
        See the documentation of the ``vip_hci.preproc.frame_rotate`` function.
    interpolation : str, optional
        See the documentation of the ``vip_hci.preproc.frame_rotate`` function.
    collapse : {'median', 'mean', 'sum', 'trimmean'}, str optional
        Sets the way of collapsing the frames for producing a final image.
    nproc : None or int, optional
        Number of processes for parallel computing. If None the number of
        processes will be set to cpu_count()/2. By default the algorithm works
        in single-process mode.
    full_output: bool, optional
        Whether to return the final median combined image only or with other 
        intermediate arrays. 
    verbose : bool, optional
        If True prints to stdout intermediate info.
        
    Returns
    -------
    frame : array_like, 2d
        Median combination of the de-rotated cube.
    If full_output is True:  
    cube_out : array_like, 3d
        The cube of residuals.
    cube_der : array_like, 3d
        The derotated cube of residuals.

    References
    ----------
    .. [MAR06]
       | Marois et al. 2006
       | **Angular Differential Imaging: A Powerful High-Contrast Imaging
         Technique**
       | *The Astrophysical Journal, Volume 641, Issue 1, pp. 556-564*
       | `https://arxiv.org/abs/astro-ph/0512335
         <https://arxiv.org/abs/astro-ph/0512335>`_

    """
    global array
    array = cube
    
    if not (array.ndim == 3 or array.ndim == 4):
        raise TypeError('Input array is not a 3d or 4d array')

    if verbose:
        start_time = time_ini()

    if nproc is None:
        nproc = cpu_count() // 2        # Hyper-threading doubles the # of cores

    angle_list = check_pa_vector(angle_list)

    if array.ndim == 3:
        n, y, _ = array.shape

        if array.shape[0] != angle_list.shape[0]:
            msg = 'Input vector or parallactic angles has wrong length'
            raise TypeError(msg)

        # The median frame is first subtracted from each frame
        model_psf = np.median(array, axis=0)
        array = array - model_psf

        # Depending on the ``mode``
        if mode == 'fullfr':
            if radius_int > 0:
                cube_out = mask_circle(array, radius_int)
            else:
                cube_out = array
            if verbose:
                print('Median psf reference subtracted')

        elif mode == 'annular':
            if nframes % 2 != 0:
                raise TypeError('nframes argument must be even value')

            annulus_width = int(asize * fwhm)  # equal size for all annuli
            n_annuli = int((y / 2 - radius_int) / annulus_width)
            if verbose:
                print('N annuli = {}, FWHM = {}'.format(n_annuli, fwhm))

            cube_out = np.zeros_like(array)

            if nproc == 1:
                for ann in range(n_annuli):
                    mres, yy, xx = _median_subt_ann(ann, angle_list, n_annuli,
                                                    fwhm, radius_int,
                                                    annulus_width, delta_rot,
                                                    nframes, verbose)
                    cube_out[:, yy, xx] = mres
            elif nproc > 1:
                pool = Pool(processes=nproc)
                res = pool.map(EFT, zip(itt.repeat(_median_subt_ann),
                                        range(n_annuli), itt.repeat(angle_list),
                                        itt.repeat(n_annuli), itt.repeat(fwhm),
                                        itt.repeat(radius_int),
                                        itt.repeat(annulus_width),
                                        itt.repeat(delta_rot),
                                        itt.repeat(nframes),
                                        itt.repeat(verbose)))
                res = np.array(res)
                pool.close()
                mres = res[:, 0]
                yy = res[:, 1]
                xx = res[:, 2]
                for ann in range(n_annuli):
                    cube_out[:, yy[ann], xx[ann]] = mres[ann]

            if verbose:
                print('Optimized median psf reference subtracted')

        else:
            raise RuntimeError('Mode not recognized')

        cube_der = cube_derotate(cube_out, angle_list, imlib=imlib,
                                 interpolation=interpolation)
        frame = cube_collapse(cube_der, mode=collapse)

    elif array.ndim == 4:
        z, n, y_in, x_in = array.shape

        if scale_list is None:
            raise ValueError('Scaling factors vector must be provided')
        else:
            if np.array(scale_list).ndim > 1:
                raise ValueError('Scaling factors vector is not 1d')
            if not scale_list.shape[0] == z:
                raise ValueError('Scaling factors vector has wrong length')

        scale_list = check_scal_vector(scale_list)

        if verbose:
            print('{} spectral channels in IFS cube'.format(z))
        residuals_cube_channels = np.zeros((n, y_in, x_in))

        if mode == 'fullfr':
            if verbose:
                print('First median subtraction exploiting spectral '
                      'variability')
            for i in Progressbar(range(n), verbose=verbose):
                cube_resc, _, _, _, _, _ = scwave(array[:, i, :, :], scale_list)
                median_frame = np.median(cube_resc, axis=0)
                residuals_cube = cube_resc - median_frame
                frame_i = scwave(residuals_cube, scale_list,
                                 full_output=full_output, inverse=True,
                                 y_in=y_in, x_in=x_in)
                residuals_cube_channels[i] = frame_i

            if verbose:
                timing(start_time)
                print('{} ADI frames'.format(n))
                print('Median subtraction in the ADI fashion')
            median_frame = np.median(residuals_cube_channels, axis=0)
            residuals_final = residuals_cube_channels - median_frame
            residuals_final_der = cube_derotate(residuals_final, angle_list,
                                                imlib=imlib,
                                                interpolation=interpolation)
            frame = cube_collapse(residuals_final_der, mode=collapse)
            if verbose:
                timing(start_time)

        elif mode == 'annular':
            # TODO : implement the radial movement exclusion
            if verbose:
                print('First median subtraction exploiting spectral '
                      'variability')
            for i in Progressbar(range(n), verbose=verbose):
                cube_resc, _, _, _, _, _ = scwave(array[:, i, :, :], scale_list)
                median_frame = np.median(cube_resc, axis=0)
                residuals_cube = cube_resc - median_frame
                frame_i = scwave(residuals_cube, scale_list,
                                 full_output=full_output, inverse=True,
                                 y_in=y_in, x_in=x_in)
                residuals_cube_channels[i] = frame_i

            if nframes % 2 != 0:
                raise TypeError('nframes argument must be even value')

            fwhm = int(np.round(np.mean(fwhm)))
            annulus_width = int(asize * fwhm)  # equal size for all annuli
            n_annuli = int((y_in / 2 - radius_int) / annulus_width)
            if verbose:
                timing(start_time)
                print('{} ADI frames'.format(n))
                print('Median subtraction in the ADI fashion')
                print('N annuli = {}, FWHM = {}'.format(n_annuli, fwhm))
                print('PA thresholds: (not in order when `nrpoc`>1)')
            cube_out = np.zeros_like(array[0])
            array = residuals_cube_channels

            if nproc == 1:
                for ann in range(n_annuli):
                    mres, yy, xx = _median_subt_ann(ann, angle_list, n_annuli,
                                                    fwhm, radius_int,
                                                    annulus_width, delta_rot,
                                                    nframes, verbose)
                    cube_out[:, yy, xx] = mres

            elif nproc > 1:
                pool = Pool(processes=nproc)
                res = pool.map(EFT, zip(itt.repeat(_median_subt_ann),
                                        range(n_annuli), itt.repeat(angle_list),
                                        itt.repeat(n_annuli), itt.repeat(fwhm),
                                        itt.repeat(radius_int),
                                        itt.repeat(annulus_width),
                                        itt.repeat(delta_rot),
                                        itt.repeat(nframes),
                                        itt.repeat(verbose)))
                res = np.array(res)
                pool.close()
                mres = res[:, 0]
                yy = res[:, 1]
                xx = res[:, 2]
                for ann in range(n_annuli):
                    cube_out[:, yy[ann], xx[ann]] = mres[ann]

            cube_der = cube_derotate(cube_out, angle_list, imlib=imlib,
                                     interpolation=interpolation)
            frame = cube_collapse(cube_der, mode=collapse)

    if verbose:
        print('\nDone derotating and combining')
        timing(start_time)
    if full_output:
        return cube_out, cube_der, frame 
    else:
        return frame 


def _median_subt_ann(ann, angle_list, n_annuli, fwhm, radius_int, annulus_width,
                     delta_rot, nframes, verbose):
    """ Optimized median subtraction for a given annulus.
    """
    if array.ndim == 3:
        n = array.shape[0]
    elif array.ndim == 4:
        n = array.shape[1]

    # The annulus is built, and the corresponding PA thresholds for frame
    # rejection are calculated. The PA rejection is calculated at center of
    # the annulus
    pa_thr, inner_radius, _ = _define_annuli(angle_list, ann, n_annuli, fwhm,
                                             radius_int, annulus_width,
                                             delta_rot, 1, False)
    if verbose:
        print('{:.2f}'.format(pa_thr), end=' ')

    if array.ndim == 3:
        indices = get_annulus(array[0], inner_radius, annulus_width,
                              output_indices=True)
    elif array.ndim == 4:
        indices = get_annulus(array[0, 0], inner_radius, annulus_width,
                              output_indices=True)
    yy = indices[0]
    xx = indices[1]
    matrix = array[:, yy, xx]  # shape [n x npx_annulus]
    matrix_res = np.zeros_like(matrix)

    # A second optimized psf reference is subtracted from each frame.
    # For each frame we find ``nframes``, depending on the PA threshold,
    # to construct this optimized psf reference
    for frame in range(n):
        if pa_thr != 0:
            indices_left = _find_indices(angle_list, frame, pa_thr, nframes)
            matrix_disc = matrix[indices_left]
        else:
            matrix_disc = matrix

        ref_psf_opt = np.median(matrix_disc, axis=0)
        curr_frame = matrix[frame]
        subtracted = curr_frame - ref_psf_opt
        matrix_res[frame] = subtracted

    return matrix_res, yy, xx


