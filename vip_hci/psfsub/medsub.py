#! /usr/bin/env python
"""
Implementation of a median subtraction algorithm for model PSF subtraction in
high-contrast imaging sequences. Median-ADI was originally proposed in [MAR06]_,
while median-SDI (also referred to as spectral deconvolution) was proposed in
[SPA02]_ and further developed in [THA07]_.

.. [MAR06]
   | Marois et al. 2006
   | **Angular Differential Imaging: A Powerful High-Contrast Imaging
     Technique**
   | *The Astrophysical Journal, Volume 641, Issue 1, pp. 556-564*
   | `https://arxiv.org/abs/astro-ph/0512335
     <https://arxiv.org/abs/astro-ph/0512335>`_

.. [SPA02]
   | Sparks & Ford 2002
   | **Imaging Spectroscopy for Extrasolar Planet Detection**
   | *The Astrophysical Journal, Volume 578, Issue 1, pp. 543-564*
   | `https://arxiv.org/abs/astro-ph/0209078
     <https://arxiv.org/abs/astro-ph/0209078>`_

.. [THA07]
   | Thatte et al. 2007
   | **Very high contrast integral field spectroscopy of AB Doradus C: 9-mag
     contrast at 0.2arcsec without a coronagraph using spectral deconvolution**
   | *MNRAS, Volume 378, Issue 4, pp. 1229-1236*
   | `https://arxiv.org/abs/astro-ph/0703565
     <https://arxiv.org/abs/astro-ph/0703565>`_


"""

__author__ = "C. A. Gomez Gonzalez, T. Bédrine, V. Christiaens"
__all__ = ["median_sub", "MEDIAN_SUB_Params"]

import numpy as np
from multiprocessing import cpu_count
from dataclasses import dataclass
from enum import Enum
from typing import Tuple, Union, List
from ..config import time_ini, timing
from ..config.paramenum import Imlib, Interpolation, Collapse, ALGO_KEY
from ..config.utils_conf import pool_map, iterable, print_precision
from ..config.utils_param import setup_parameters, separate_kwargs_dict
from ..preproc import (cube_derotate, cube_collapse, check_pa_vector,
                       check_scal_vector)
from ..preproc import cube_rescaling_wavelengths as scwave
from ..preproc.derotation import _find_indices_adi, _define_annuli
from ..preproc.rescaling import _find_indices_sdi
from ..var import get_annulus_segments, mask_circle


@dataclass
class MEDIAN_SUB_Params:
    """
    Set of parameters for the median subtraction module.

    See function `median_sub` for documentation.
    """

    cube: np.ndarray = None
    angle_list: np.ndarray = None
    scale_list: np.ndarray = None
    flux_sc_list: np.ndarray = None
    fwhm: float = 4
    radius_int: int = 0
    asize: int = 4
    delta_rot: int = 1
    delta_sep: Union[float, Tuple[float]] = (0.1, 1)
    mode: str = "fullfr"
    nframes: int = 4
    sdi_only: bool = False
    imlib: Enum = Imlib.VIPFFT
    interpolation: Enum = Interpolation.LANCZOS4
    collapse: Enum = Collapse.MEDIAN
    cube_ref: np.ndarray = None
    collapse_ref: str = 'median'
    nproc: int = 1
    full_output: bool = False
    verbose: bool = True


def median_sub(*all_args: List, **all_kwargs: dict):
    """Perform (smart) median-ADI, median-SDI or median-RDI.

    In the case of angular differential imaging (ADI), the algorithm is based on
    [MAR06]_. The ADI+IFS method is an extension of this basic idea to
    multi-spectral cubes, combining ADI with spectral deconvolution (also called
    spectral differential imaging or SDI).

    References: [MAR06]_ for median-ADI; [SPA02]_ and [THA07]_ for SDI.

    Parameters
    ----------
    all_args: list, optional
        Positionnal arguments for the median_sub algorithm. Full list of
        parameters is provided below.
    all_kwargs: dictionary, optional
        Mix of keyword arguments that can initialize a MEDIAN_SUB_Params and the
        optional ``rot_options`` dictionary (with keywords ``border_mode``,
        ``mask_val``, ``edge_blend``, ``interp_zeros``, ``ker``; see docstrings
        of ``vip_hci.preproc.frame_rotate``). Can also contain a
        MEDIAN_SUB_Params object/dictionary named ``algo_params``.

    Parameters
    ----------
    cube : numpy ndarray, 3d or 4d
        Input cube.
    angle_list : 1d numpy ndarray
        Vector of derotation angles to align North up in your cube images.
    scale_list : numpy ndarray, 1d, optional
        If provided, triggers mSDI reduction. These should be the scaling
        factors used to re-scale the spectral channels and align the speckles
        in case of IFS data (ADI+mSDI cube). Usually, these can be approximated
        by the last channel wavelength divided by the other wavelengths in the
        cube (more thorough approaches can be used to get the scaling factors,
        e.g. with ``vip_hci.preproc.find_scal_vector``).
    flux_sc_list : numpy ndarray, 1d
        In the case of IFS data (ADI+SDI), this is the list of flux scaling
        factors applied to each spectral frame after geometrical rescaling.
        These should be set to either the ratio of stellar fluxes between the
        last spectral channel and the other channels, or to the second output
        of `preproc.find_scal_vector` (when using 2 free parameters). If not
        provided, the algorithm will still work, but with a lower efficiency
        at subtracting the stellar halo.
    fwhm : float or 1d numpy array
        Known size of the FWHM in pixels to be used. Default is 4.
    radius_int : int, optional
        The radius of the innermost annulus. By default is 0, if >0 then the
        central circular area is discarded.
    asize : int, optional
        The size of the annuli, in pixels.
    delta_rot : int, optional
        Factor for increasing the parallactic angle threshold, expressed in
        FWHM. Default is 1 (excludes 1 FWHM on each side of the considered
        frame).
    delta_sep : float or tuple of floats, optional
        The threshold separation in terms of the mean FWHM (for ADI+mSDI data).
        If a tuple of two values is provided, they are used as the lower and
        upper intervals for the threshold (grows as a function of the
        separation).
    mode : {'fullfr', 'annular'}, str optional
        Whether to run the algorithm in full-frame or concentric annuli.

        * median-ADI case (cube_ref = None): in ``fullfr`` mode only the median
        frame is subtracted, in ``annular`` mode the 4 closest frames meeting a
        PA threshold condition (annulus-wise) are subtracted.

        * median RDI case (cube_ref != None): in ``fullfr`` mode only a single
        (scaled) reference frame is subtracted, while in ``annular`` mode the
        reference frame is subtracted in after scaling to each annulus.
        PA threshold condition (annulus-wise) are subtracted.
    nframes : int or None, optional
        Number of frames (even value) to be used for building the optimized
        reference PSF when working in ``annular`` mode. None by default, which
        means that all frames, excluding the thresholded ones, are used.
    sdi_only: bool, optional
        In the case of IFS data (ADI+SDI), whether to perform median-SDI, or
        median-ASDI (default).
    imlib : Enum, see `vip_hci.config.paramenum.Imlib`
        See the documentation of ``vip_hci.preproc.frame_rotate``.
    interpolation : Enum, see `vip_hci.config.paramenum.Interpolation`
        See the documentation of the ``vip_hci.preproc.frame_rotate`` function.
    collapse : Enum, see `vip_hci.config.paramenum.Collapse`
        Sets how temporal residual frames should be combined to produce an
        ADI image.
    cube_ref : 3d numpy ndarray, optional
        Reference library cube for Reference Star Differential Imaging. If not
        None will automatically trigger median-RDI.
    collapse_ref : {'median', 'mean', 'sc_median', 'sc_mean'}, str optional
        Method to consider for collapsing the reference cube to create a
        reference PSF image to be subtracted to the science images. Choice
        between median, mean, scaled median ('sc_median') or scaled mean
        ('sc_mean'). In the latter 2 cases the reference image is scaled in flux
        based on the ratio of integrated flux in the reference image and each
        science image, before subtraction. By default, the whole images are
        considered to calculate the flux ratio, but if a pair of integers
        separated by '-' are provided, these will be considered as inner and
        outer radius to be used for flux scaling. For example 'sc_median0-30'
        means a median reference image will be used, scaled based on the ratio
        of integrated flux measured within a radius of 30px. If scaling of the
        reference is requested ('sc' in collapse_ref) while ``mode`` is set to
        'annular', the scaling will be done annulus per annulus.
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
    cube_out : numpy ndarray, 3d
        [full_output=True] The cube of residuals.
    cube_der : numpy ndarray, 3d
        [full_output=True] The derotated cube of residuals.
    frame : numpy ndarray, 2d
        Median combination of the de-rotated cube.

    """
    # Separating the parameters of the ParamsObject from optional rot_options
    class_params, rot_options = separate_kwargs_dict(
        initial_kwargs=all_kwargs, parent_class=MEDIAN_SUB_Params
    )

    # Extracting the object of parameters (if any)
    algo_params = None
    if ALGO_KEY in rot_options.keys():
        algo_params = rot_options[ALGO_KEY]
        del rot_options[ALGO_KEY]

    if algo_params is None:
        algo_params = MEDIAN_SUB_Params(*all_args, **class_params)

    # by default, interpolate masked area before derotation if a mask is used
    if algo_params.radius_int and len(rot_options) == 0:
        rot_options['mask_val'] = 0
        rot_options['ker'] = 1
        rot_options['interp_zeros'] = True

    global ARRAY
    ARRAY = algo_params.cube.copy()

    if not (ARRAY.ndim == 3 or ARRAY.ndim == 4):
        raise TypeError("Input array is not a 3d or 4d array")

    if algo_params.verbose:
        start_time = time_ini()

    if algo_params.nproc is None:
        algo_params.nproc = cpu_count() // 2

    algo_params.angle_list = check_pa_vector(algo_params.angle_list)

    if ARRAY.ndim == 3:
        n, y, x = ARRAY.shape
        if algo_params.cube_ref is not None:
            condx = algo_params.cube_ref.shape[-1] != x
            condy = algo_params.cube_ref.shape[-2] != y
            if condx or condy:
                msg = "Reference cube shape should have same xy dimensions as "
                msg += "science cube"
                raise TypeError(msg)
            if 'median' in algo_params.collapse_ref:
                ref_frame = np.median(algo_params.cube_ref, axis=0)
            elif 'mean' in algo_params.collapse_ref:
                ref_frame = np.mean(algo_params.cube_ref, axis=0)
            else:
                idx0 = 0
                if "sc_" in algo_params.collapse_ref:
                    idx0 = 3
                if '-' in algo_params.collapse_ref:
                    idxN = algo_params.collapse_ref.index('-') - 1  # or -2
                try:
                    collapse_mode = algo_params.collapse_ref[idx0:idxN]
                    ref_frame = cube_collapse(algo_params.cube_ref,
                                              mode=collapse_mode)
                except TypeError:
                    collapse_mode = algo_params.collapse_ref[idx0:idxN-1]
                    ref_frame = cube_collapse(algo_params.cube_ref,
                                              mode=collapse_mode)
        if ARRAY.shape[0] != algo_params.angle_list.shape[0]:
            msg = "Input vector or parallactic angles has wrong length"
            raise TypeError(msg)

        # The median frame is first subtracted from each frame (if no RDI)
        if algo_params.cube_ref is None:
            model_psf = np.median(ARRAY, axis=0)
            ARRAY -= model_psf

        # Depending on the ``mode``
        if algo_params.mode == "fullfr":
            # MASK AFTER DEROTATION TO AVOID ARTEFACTS
            # if radius_int > 0:
            #     cube_out = mask_circle(ARRAY, radius_int, fillwith=np.nan)
            # else:
            cube_out = ARRAY
            if algo_params.cube_ref is not None:
                if 'sc' in algo_params.collapse_ref:
                    if len(algo_params.collapse_ref) > 9:  # ie radii given?
                        idx_rin = algo_params.collapse_ref.index('n')+1
                        idx_rout = algo_params.collapse_ref.index('-')
                        rin = algo_params.collapse_ref[idx_rin:idx_rout]
                        rin = int(rin)
                        rout = algo_params.collapse_ref[idx_rout+1:]
                        rout = int(rout)
                    else:
                        rin = 0
                        rout = y//2 - 1
                    mask_ref = mask_circle(ref_frame, rin, fillwith=np.nan)
                    mask_ref = mask_circle(mask_ref, rout, fillwith=np.nan,
                                           mode='out')
                    for i in range(n):
                        mask_sci = mask_circle(ARRAY[i], rin, fillwith=np.nan)
                        mask_sci = mask_circle(mask_sci, rout, fillwith=np.nan,
                                               mode='out')
                        scal_fac = np.nansum(mask_sci)/np.nansum(mask_ref)
                        ARRAY[i] -= scal_fac*ref_frame
                else:
                    ARRAY -= ref_frame
            if algo_params.verbose:
                print("Median psf reference subtracted")

        elif algo_params.mode == "annular":
            cube_out = np.zeros_like(ARRAY)
            n_annuli = int((y / 2 - algo_params.radius_int) / algo_params.asize)
            if algo_params.verbose:
                print("N annuli = {}, FWHM = {}".format(
                    n_annuli, algo_params.fwhm))

            add_params = {
                "ann": iterable(range(n_annuli)),
                "n_annuli": n_annuli,
                "annulus_width": algo_params.asize,
            }

            if algo_params.cube_ref is not None:
                add_params["frame_ref"] = ref_frame
                func_params = setup_parameters(
                    params_obj=algo_params,
                    fkt=_median_subt_ann_rdi,
                    as_list=True,
                    **add_params,
                )
                res = pool_map(
                    algo_params.nproc,
                    _median_subt_ann_rdi,
                    msg="Processing annuli:",
                    progressbar_single=True,
                    *func_params,
                )
            else:
                if algo_params.nframes is not None:
                    if algo_params.nframes % 2 != 0:
                        raise TypeError("`nframes` argument must be even value")
                func_params = setup_parameters(
                    params_obj=algo_params,
                    fkt=_median_subt_ann_adi,
                    as_list=True,
                    **add_params,
                )
                res = pool_map(
                    algo_params.nproc,
                    _median_subt_ann_adi,
                    msg="Processing annuli:",
                    progressbar_single=True,
                    *func_params,
                )

            res = np.array(res, dtype=object)
            mres = res[:, 0]
            yy = res[:, 1]
            xx = res[:, 2]
            # cube_out = np.zeros_like(ARRAY)
            # cube_out[:] = np.nan
            for ann in range(n_annuli):
                cube_out[:, yy[ann], xx[ann]] = mres[ann]

            if algo_params.verbose:
                print("Optimized median psf reference subtracted")

        else:
            raise RuntimeError("Mode not recognized")

        cube_der = cube_derotate(
            cube_out,
            algo_params.angle_list,
            nproc=algo_params.nproc,
            imlib=algo_params.imlib,
            interpolation=algo_params.interpolation,
            **rot_options,
        )
        if algo_params.radius_int:
            cube_out = mask_circle(cube_out, algo_params.radius_int)
            cube_der = mask_circle(cube_der, algo_params.radius_int)
        frame = cube_collapse(cube_der, mode=algo_params.collapse)

    elif ARRAY.ndim == 4:
        z, n, y_in, x_in = ARRAY.shape

        if algo_params.scale_list is None:
            raise ValueError("Scaling factors vector must be provided")
        else:
            if np.array(algo_params.scale_list).ndim > 1:
                raise ValueError("Scaling factors vector is not 1d")
            if not algo_params.scale_list.shape[0] == z:
                raise ValueError("Scaling factors vector has wrong length")

        if algo_params.flux_sc_list is not None:
            if np.array(algo_params.flux_sc_list).ndim > 1:
                raise ValueError("Scaling factors vector is not 1d")
            if not algo_params.flux_sc_list.shape[0] == z:
                raise ValueError("Scaling factors vector has wrong length")

        # Exploiting spectral variability (radial movement)
        algo_params.fwhm = int(np.round(np.mean(algo_params.fwhm)))
        n_annuli = int((y_in / 2 - algo_params.radius_int) / algo_params.asize)

        if algo_params.nframes is not None:
            if algo_params.nframes % 2 != 0:
                raise TypeError("`nframes` argument must be even value")

        if algo_params.verbose:
            print("{} spectral channels per IFS frame".format(z))
            print("First median subtraction exploiting spectral variability")
            if algo_params.mode == "annular":
                print(
                    "N annuli = {}, mean FWHM = {:.3f}".format(
                        n_annuli, algo_params.fwhm
                    )
                )

        add_params = {
            "fr": iterable(range(n)),
            "scal": algo_params.scale_list,
            "flux_scal": algo_params.flux_sc_list,
            "n_annuli": n_annuli,
            "annulus_width": algo_params.asize,
        }

        func_params = setup_parameters(params_obj=algo_params,
                                       fkt=_median_subt_fr_sdi, as_list=True,
                                       **add_params)
        res = pool_map(
            algo_params.nproc,
            _median_subt_fr_sdi,
            *func_params,
        )
        residuals_cube_channels = np.array(res)

        if algo_params.verbose:
            timing(start_time)
            print("{} ADI frames".format(n))
            print("Median subtraction in the ADI fashion")

        if algo_params.sdi_only:
            cube_out = residuals_cube_channels
        else:
            if algo_params.mode == "fullfr":
                median_frame = np.nanmedian(residuals_cube_channels, axis=0)
                cube_out = residuals_cube_channels - median_frame

            elif algo_params.mode == "annular":
                if algo_params.verbose:
                    print(
                        "N annuli = {}, mean FWHM = {:.3f}".format(
                            n_annuli, algo_params.fwhm
                        )
                    )
                ARRAY = residuals_cube_channels

                add_params = {
                    "ann": iterable(range(n_annuli)),
                    "n_annuli": n_annuli,
                    "annulus_width": algo_params.asize,
                }

                func_params = setup_parameters(
                    params_obj=algo_params,
                    fkt=_median_subt_ann_adi,
                    as_list=True,
                    **add_params,
                )

                res = pool_map(
                    algo_params.nproc,
                    _median_subt_ann_adi,
                    msg="Processing annuli:",
                    progressbar_single=True,
                    *func_params,
                )

                res = np.array(res, dtype=object)
                mres = res[:, 0]
                yy = res[:, 1]
                xx = res[:, 2]
                pa_thrs = np.array(res[:, 3])
                if algo_params.verbose:
                    print("PA thresholds: ")
                    print_precision(pa_thrs)

                cube_out = np.zeros_like(ARRAY)
                cube_out[:] = np.nan
                for ann in range(n_annuli):
                    cube_out[:, yy[ann], xx[ann]] = mres[ann]

            else:
                raise RuntimeError("Mode not recognized")

        cube_der = cube_derotate(
            cube_out,
            algo_params.angle_list,
            imlib=algo_params.imlib,
            interpolation=algo_params.interpolation,
            nproc=algo_params.nproc,
            **rot_options,
        )
        if algo_params.radius_int:
            cube_der = mask_circle(cube_der, algo_params.radius_int)
        frame = cube_collapse(cube_der, mode=algo_params.collapse)

    if algo_params.verbose:
        print("Done derotating and combining")
        timing(start_time)
    if algo_params.full_output:
        return cube_out, cube_der, frame
    else:
        return frame


def _median_subt_fr_sdi(
    fr,
    scal,
    flux_scal,
    n_annuli,
    fwhm,
    radius_int,
    annulus_width,
    delta_sep,
    nframes,
    imlib,
    interpolation,
    collapse,
    mode,
):
    """Optimized median subtraction on a multi-spectral frame (IFS data)."""
    z, n, y_in, x_in = ARRAY.shape
    scale_list = check_scal_vector(scal)
    multispec_fr = scwave(
        ARRAY[:, fr, :, :], scale_list, imlib=imlib, interpolation=interpolation
    )[
        0
    ]  # rescaled cube
    if flux_scal is not None:
        for i in range(z):
            multispec_fr[i] *= flux_scal[i]

    if mode == "annular":
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

            indices = get_annulus_segments(
                multispec_fr[0], inner_radius, annulus_width
            )[0]
            yy = indices[0]
            xx = indices[1]
            matrix = multispec_fr[:, yy, xx]  # shape (z, npx_annulus)

            for j in range(z):
                indices_left = _find_indices_sdi(
                    scal, ann_center, j, fwhm, delta_sep_vec[ann], nframes
                )
                matrix_masked = matrix[indices_left]
                ref_psf_opt = np.nanmedian(matrix_masked, axis=0)
                curr_wv = matrix[j]
                subtracted = curr_wv - ref_psf_opt
                cube_res[j, yy, xx] = subtracted

    elif mode == "fullfr":
        median_frame = np.nanmedian(multispec_fr, axis=0)
        cube_res = multispec_fr - median_frame

    if flux_scal is not None:
        for i in range(z):
            cube_res[i] /= flux_scal[i]

    frame_desc = scwave(
        cube_res,
        scale_list,
        full_output=False,
        inverse=True,
        y_in=y_in,
        x_in=x_in,
        imlib=imlib,
        interpolation=interpolation,
        collapse=collapse,
    )
    return frame_desc


def _median_subt_ann_adi(ann, angle_list, n_annuli, fwhm, radius_int,
                         annulus_width, delta_rot, nframes):
    """Optimized median subtraction for a given annulus."""
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
        indices = get_annulus_segments(ARRAY[0], inner_radius, annulus_width)[0]
    elif ARRAY.ndim == 4:
        indices = get_annulus_segments(
            ARRAY[0, 0], inner_radius, annulus_width)[0]
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

        ref_psf_opt = np.nanmedian(matrix_disc, axis=0)
        curr_frame = matrix[frame]
        subtracted = curr_frame - ref_psf_opt
        matrix_res[frame] = subtracted

    return matrix_res, yy, xx, pa_thr


def _median_subt_ann_rdi(frame_ref, collapse_ref, ann, n_annuli,
                         radius_int, annulus_width):
    """RDI median subtraction for a given annulus."""
    if ARRAY.ndim == 3:
        n = ARRAY.shape[0]

    inner_radius = radius_int + ann * annulus_width

    if ARRAY.ndim == 3:
        indices = get_annulus_segments(ARRAY[0], inner_radius, annulus_width)[0]
    elif ARRAY.ndim == 4:
        indices = get_annulus_segments(
            ARRAY[0, 0], inner_radius, annulus_width)[0]
    yy = indices[0]
    xx = indices[1]

    matrix_ref = frame_ref[yy, xx]
    matrix = ARRAY[:, yy, xx]  # shape [n x npx_annulus]
    matrix_res = np.zeros_like(matrix)

    # A second optimized psf reference is subtracted from each frame.
    # For each frame we find ``nframes``, depending on the PA threshold,
    # to construct this optimized psf reference
    for frame in range(n):
        curr_frame = matrix[frame]
        if 'sc' in collapse_ref:
            scal_fac = np.nansum(curr_frame)/np.nansum(matrix_ref)
            ref_psf_opt = scal_fac*matrix_ref
        else:
            ref_psf_opt = matrix_ref.copy()
        subtracted = curr_frame - ref_psf_opt
        matrix_res[frame] = subtracted

    return matrix_res, yy, xx
