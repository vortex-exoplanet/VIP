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
__all__ = ['median_sub']

import numpy as np
from multiprocessing import cpu_count
from ..conf import time_ini, timing
from ..var import get_annulus, mask_circle
from ..preproc import (cube_derotate, cube_collapse, check_pa_vector,
                       check_scal_vector)
from ..preproc import cube_rescaling_wavelengths as scwave
from ..conf.utils_conf import pool_map, fixed, print_precision
from ..preproc.derotation import _find_indices_adi, _define_annuli
from ..preproc.rescaling import _find_indices_sdi


def median_sub(cube, angle_list, scale_list=None, fwhm=4, radius_int=0, asize=4,
               delta_rot=1, delta_sep=(0.1, 1), mode='fullfr', nframes=4,
               imlib='opencv', interpolation='lanczos4', collapse='median',
               nproc=1, full_output=False, verbose=True):
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
        The size of the annuli, in pixels.
    delta_rot : int, optional
        Factor for increasing the parallactic angle threshold, expressed in
        FWHM. Default is 1 (excludes 1 FHWM on each side of the considered
        frame).
    delta_sep : float or tuple of floats, optional
        The threshold separation in terms of the mean FWHM (for ADI+mSDI data).
        If a tuple of two values is provided, they are used as the lower and
        upper intervals for the threshold (grows as a function of the
        separation).
    mode : {'fullfr', 'annular'}, str optional
        In "simple" mode only the median frame is subtracted, in "annular" mode
        also the 4 closest frames given a PA threshold (annulus-wise) are 
        subtracted.
    nframes : int or None, optional
        Number of frames (even value) to be used for building the optimized
        reference PSF when working in annular mode. None by default, which means
        that all frames, excluding the thresholded ones, are used.
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
    global ARRAY
    ARRAY = cube
    
    if not (ARRAY.ndim == 3 or ARRAY.ndim == 4):
        raise TypeError('Input array is not a 3d or 4d array')

    if verbose:
        start_time = time_ini()

    if nproc is None:
        nproc = cpu_count() // 2        # Hyper-threading doubles the # of cores

    angle_list = check_pa_vector(angle_list)

    if ARRAY.ndim == 3:
        n, y, _ = ARRAY.shape

        if ARRAY.shape[0] != angle_list.shape[0]:
            msg = 'Input vector or parallactic angles has wrong length'
            raise TypeError(msg)

        # The median frame is first subtracted from each frame
        model_psf = np.median(ARRAY, axis=0)
        ARRAY -= model_psf

        # Depending on the ``mode``
        if mode == 'fullfr':
            if radius_int > 0:
                cube_out = mask_circle(ARRAY, radius_int)
            else:
                cube_out = ARRAY
            if verbose:
                print('Median psf reference subtracted')

        elif mode == 'annular':
            if nframes is not None:
                if nframes % 2 != 0:
                    raise TypeError('`nframes` argument must be even value')

            n_annuli = int((y / 2 - radius_int) / asize)
            if verbose:
                print('N annuli = {}, FWHM = {}'.format(n_annuli, fwhm))

            res = pool_map(nproc, _median_subt_ann_adi, fixed(range(n_annuli)),
                           angle_list, n_annuli, fwhm, radius_int, asize,
                           delta_rot, nframes, verbose=verbose,
                           msg='Processing annuli:', progressbar_single=True)

            res = np.array(res)
            mres = res[:, 0]
            yy = res[:, 1]
            xx = res[:, 2]
            cube_out = np.zeros_like(ARRAY)
            for ann in range(n_annuli):
                cube_out[:, yy[ann], xx[ann]] = mres[ann]

            if verbose:
                print('Optimized median psf reference subtracted')

        else:
            raise RuntimeError('Mode not recognized')

        cube_der = cube_derotate(cube_out, angle_list, imlib=imlib,
                                 interpolation=interpolation)
        frame = cube_collapse(cube_der, mode=collapse)

    elif ARRAY.ndim == 4:
        z, n, y_in, x_in = ARRAY.shape

        if scale_list is None:
            raise ValueError('Scaling factors vector must be provided')
        else:
            if np.array(scale_list).ndim > 1:
                raise ValueError('Scaling factors vector is not 1d')
            if not scale_list.shape[0] == z:
                raise ValueError('Scaling factors vector has wrong length')

        # Exploiting spectral variability (radial movement)
        fwhm = int(np.round(np.mean(fwhm)))
        n_annuli = int((y_in / 2 - radius_int) / asize)

        if nframes is not None:
            if nframes % 2 != 0:
                raise TypeError('`nframes` argument must be even value')

        if verbose:
            print('{} spectral channels per IFS frame'.format(z))
            print('First median subtraction exploiting spectral variability')
            if mode == 'annular':
                print('N annuli = {}, mean FWHM = {:.3f}'.format(n_annuli,
                                                                 fwhm))

        res = pool_map(nproc, _median_subt_fr_sdi, fixed(range(n)),
                       scale_list, n_annuli, fwhm, radius_int, asize,
                       delta_sep, nframes, imlib, interpolation, collapse,
                       mode)
        residuals_cube_channels = np.array(res)

        if verbose:
            timing(start_time)
            print('{} ADI frames'.format(n))
            print('Median subtraction in the ADI fashion')

        if mode == 'fullfr':
            median_frame = np.median(residuals_cube_channels, axis=0)
            residuals_final = residuals_cube_channels - median_frame
            residuals_final_der = cube_derotate(residuals_final, angle_list,
                                                imlib=imlib,
                                                interpolation=interpolation)
            frame = cube_collapse(residuals_final_der, mode=collapse)
            if verbose:
                timing(start_time)

        elif mode == 'annular':
            if verbose:
                print('N annuli = {}, mean FWHM = {:.3f}'.format(n_annuli,
                                                                 fwhm))
            ARRAY = residuals_cube_channels

            res = pool_map(nproc, _median_subt_ann_adi, fixed(range(n_annuli)),
                           angle_list, n_annuli, fwhm, radius_int, asize,
                           delta_rot, nframes)

            res = np.array(res)
            mres = res[:, 0]
            yy = res[:, 1]
            xx = res[:, 2]
            pa_thrs = np.array(res[:, 3])
            if verbose:
                print('PA thresholds: ')
                print_precision(pa_thrs)

            cube_out = np.zeros_like(ARRAY)
            for ann in range(n_annuli):
                cube_out[:, yy[ann], xx[ann]] = mres[ann]

            cube_der = cube_derotate(cube_out, angle_list, imlib=imlib,
                                     interpolation=interpolation)
            frame = cube_collapse(cube_der, mode=collapse)

        else:
            raise RuntimeError('Mode not recognized')

    if verbose:
        print('Done derotating and combining')
        timing(start_time)
    if full_output:
        return cube_out, cube_der, frame 
    else:
        return frame 


def _median_subt_fr_sdi(fr, wl, n_annuli, fwhm, radius_int, annulus_width,
                        delta_sep, nframes, imlib, interpolation, collapse,
                        mode):
    """ Optimized median subtraction on a multi-spectral frame (IFS data).
    """
    z, n, y_in, x_in = ARRAY.shape
    scale_list = check_scal_vector(wl)
    multispec_fr = scwave(ARRAY[:, fr, :, :], scale_list, imlib=imlib,
                          interpolation=interpolation)[0]    # rescaled cube

    if mode == 'annular':
        cube_res = np.zeros_like(multispec_fr)  # shape (z, resc_y, resc_x)

        if isinstance(delta_sep, tuple):
            delta_sep_vec = np.linspace(delta_sep[0], delta_sep[1], n_annuli)
        else:
            delta_sep_vec = [delta_sep] * n_annuli

        for ann in range(n_annuli):
            if ann == n_annuli - 1:
                inner_radius = radius_int + (ann * annulus_width - 1)
            else:
                inner_radius = radius_int + ann * annulus_width
            ann_center = inner_radius + (annulus_width / 2)

            indices = get_annulus(multispec_fr[0], inner_radius, annulus_width,
                                  output_indices=True)
            yy = indices[0]
            xx = indices[1]
            matrix = multispec_fr[:, yy, xx]  # shape (z, npx_annulus)

            for j in range(z):
                indices_left = _find_indices_sdi(wl, ann_center, j, fwhm,
                                                 delta_sep_vec[ann], nframes)
                matrix_masked = matrix[indices_left]
                ref_psf_opt = np.median(matrix_masked, axis=0)
                curr_wv = matrix[j]
                subtracted = curr_wv - ref_psf_opt
                cube_res[j, yy, xx] = subtracted

    elif mode == 'fullfr':
        median_frame = np.median(multispec_fr, axis=0)
        cube_res = multispec_fr - median_frame

    frame_desc = scwave(cube_res, scale_list, full_output=False, inverse=True,
                        y_in=y_in, x_in=x_in, imlib=imlib,
                        interpolation=interpolation, collapse=collapse)
    return frame_desc


def _median_subt_ann_adi(ann, angle_list, n_annuli, fwhm, radius_int,
                         annulus_width, delta_rot, nframes):
    """ Optimized median subtraction for a given annulus.
    """
    if ARRAY.ndim == 3:
        n = ARRAY.shape[0]
    elif ARRAY.ndim == 4:
        n = ARRAY.shape[1]

    # The annulus is built, and the corresponding PA thresholds for frame
    # rejection are calculated. The PA rejection is calculated at center of
    # the annulus
    pa_thr, inner_radius, _ = _define_annuli(angle_list, ann, n_annuli, fwhm,
                                             radius_int, annulus_width,
                                             delta_rot, 1, False)
    if ARRAY.ndim == 3:
        indices = get_annulus(ARRAY[0], inner_radius, annulus_width,
                              output_indices=True)
    elif ARRAY.ndim == 4:
        indices = get_annulus(ARRAY[0, 0], inner_radius, annulus_width,
                              output_indices=True)
    yy = indices[0]
    xx = indices[1]
    matrix = ARRAY[:, yy, xx]  # shape [n x npx_annulus]
    matrix_res = np.zeros_like(matrix)

    # A second optimized psf reference is subtracted from each frame.
    # For each frame we find ``nframes``, depending on the PA threshold,
    # to construct this optimized psf reference
    for frame in range(n):
        if pa_thr != 0:
            indices_left = _find_indices_adi(angle_list, frame, pa_thr, nframes)
            matrix_disc = matrix[indices_left]
        else:
            matrix_disc = matrix

        ref_psf_opt = np.median(matrix_disc, axis=0)
        curr_frame = matrix[frame]
        subtracted = curr_frame - ref_psf_opt
        matrix_res[frame] = subtracted

    return matrix_res, yy, xx, pa_thr


