#! /usr/bin/env python

"""
Module with a frame differencing algorithm for ADI and ADI+mSDI post-processing.
"""

__author__ = 'Carlos Alberto Gomez Gonzalez'
__all__ = ['xloci']

import numpy as np
import scipy as sp
import pandas as pn
from multiprocessing import cpu_count
from sklearn.metrics import pairwise_distances
from ..var import get_annulus_segments
from ..preproc import (cube_derotate, cube_collapse, check_pa_vector,
                       check_scal_vector)
from ..preproc.rescaling import _find_indices_sdi
from ..config import time_ini, timing
from ..preproc import cube_rescaling_wavelengths as scwave
from ..preproc.derotation import _find_indices_adi, _define_annuli
from ..config.utils_conf import pool_map, iterable, Progressbar


def xloci(cube, angle_list, scale_list=None, fwhm=4, metric='manhattan',
          dist_threshold=100, delta_rot=(0.1, 1), delta_sep=(0.1, 1),
          radius_int=0, asize=4, n_segments=4, nproc=1, solver='lstsq',
          tol=1e-2, optim_scale_fact=2, adimsdi='skipadi', imlib='vip-fft',
          interpolation='lanczos4', collapse='median', verbose=True,
          full_output=False, **rot_options):
    """ LOCI style algorithm that models a PSF (for ADI and ADI+mSDI) with a
    least-square combination of neighbouring frames (solving the equation
    a x = b by computing a vector x of coefficients that minimizes the
    Euclidean 2-norm || b - a x ||^2).

    Parameters
    ----------
    cube : numpy ndarray, 3d or 4d
        Input cube.
    angle_list : numpy ndarray, 1d
        Corresponding parallactic angle for each frame.
    fwhm : float, optional
        Size of the FHWM in pixels. Default is 4.
    metric : str, optional
        Distance metric to be used ('cityblock', 'cosine', 'euclidean', 'l1',
        'l2', 'manhattan', 'correlation', etc). It uses the scikit-learn
        function ``sklearn.metrics.pairwise.pairwise_distances`` (check its
        documentation).
    dist_threshold : int, optional
        Indices with a distance larger than ``dist_threshold`` percentile will
        initially discarded. 100 by default.
    delta_rot : float or tuple of floats, optional
        Factor for adjusting the parallactic angle threshold, expressed in
        FWHM. Default is 1 (excludes 1 FHWM on each side of the considered
        frame). If a tuple of two floats is provided, they are used as the lower
        and upper intervals for the threshold (grows linearly as a function of
        the separation).
    delta_sep : float or tuple of floats, optional
        The threshold separation in terms of the mean FWHM (for ADI+mSDI data).
        If a tuple of two values is provided, they are used as the lower and
        upper intervals for the threshold (grows as a function of the
        separation).
    radius_int : int, optional
        The radius of the innermost annulus. By default is 0, if >0 then the
        central circular region is discarded.
    asize : int, optional
        The size of the annuli, in pixels.
    n_segments : int or list of int or 'auto', optional
        The number of segments for each annulus. When a single integer is given
        it is used for all annuli. When set to 'auto', the number of segments is
        automatically determined for every annulus, based on the annulus width.
    nproc : None or int, optional
        Number of processes for parallel computing. If None the number of
        processes will be set to cpu_count()/2. By default the algorithm works
        in single-process mode.
    solver : {'lstsq', 'nnls'}, str optional
        Choosing the solver of the least squares problem. ``lstsq`` uses the
        standard scipy least squares solver. ``nnls`` uses the scipy
        non-negative least-squares solver.
    tol : float, optional
        Valid when ``solver`` is set to lstsq. Sets the cutoff for 'small'
        singular values; used to determine effective rank of a. Singular values
        smaller than ``tol * largest_singular_value`` are considered zero.
        Smaller values of ``tol`` lead to smaller residuals (more aggressive
        subtraction).
    optim_scale_fact : float, optional
        If >1, the least-squares optimization is performed on a larger segment,
        similar to LOCI. The optimization segments share the same inner radius,
        mean angular position and angular width as their corresponding
        subtraction segments.
    adimsdi : {'skipadi', 'double'}, str optional
        Changes the way the 4d cubes (ADI+mSDI) are processed.

        ``skipadi``: the multi-spectral frames are rescaled wrt the largest
        wavelength to align the speckles and the least-squares model is
        subtracted on each spectral cube separately.

        ``double``: a first subtraction is done on the rescaled spectral frames
        (as in the ``skipadi`` case). Then the residuals are processed again in
        an ADI fashion.

    imlib : str, optional
        See the documentation of the ``vip_hci.preproc.frame_rotate`` function.
    interpolation : str, optional
        See the documentation of the ``vip_hci.preproc.frame_rotate`` function.
    collapse : {'median', 'mean', 'sum', 'trimmean'}, str optional
        Sets the way of collapsing the frames for producing a final image.
    verbose: bool, optional
        If True prints info to stdout.
    full_output: bool, optional
        Whether to return the final median combined image only or with other
        intermediate arrays.
    rot_options: dictionary, optional
        Dictionary with optional keyword values for "border_mode", "mask_val",  
        "edge_blend", "interp_zeros", "ker" (see documentation of 
        ``vip_hci.preproc.frame_rotate``)

    Returns
    -------
    frame_der_median : numpy ndarray, 2d
        Median combination of the de-rotated cube of residuals.

    If ``full_output`` is True, the following intermediate arrays are returned:
    cube_res, cube_der, frame_der_median

    """
    global ARRAY
    ARRAY = cube

    if verbose:
        start_time = time_ini()

    # ADI datacube
    if cube.ndim == 3:
        res = _leastsq_adi(cube, angle_list, fwhm, metric, dist_threshold,
                           delta_rot, radius_int, asize, n_segments, nproc,
                           solver, tol, optim_scale_fact, imlib,
                           interpolation, collapse, verbose, full_output)

        if verbose:
            timing(start_time)

        if full_output:
            cube_res, cube_der, frame = res
            return cube_res, cube_der, frame
        else:
            frame = res
            return frame

    # ADI+mSDI (IFS) datacubes
    elif cube.ndim == 4:
        z, n, y_in, x_in = cube.shape
        fwhm = int(np.round(np.mean(fwhm)))
        n_annuli = int((y_in / 2 - radius_int) / asize)

        # Processing separately each wavelength in ADI fashion
        if adimsdi == 'skipsdi':
            if verbose:
                print('ADI lst-sq modeling for each wavelength individually')
                print('{} frames per wavelength'.format(n))

            cube_res = np.zeros((z, y_in, x_in))
            for z in Progressbar(range(z)):
                ARRAY = cube[z]
                res = _leastsq_adi(cube[z], angle_list, fwhm, metric,
                                   dist_threshold, delta_rot, radius_int, asize,
                                   n_segments, nproc, solver, tol,
                                   optim_scale_fact, imlib, interpolation,
                                   collapse, verbose=False, full_output=False)
                cube_res[z] = res

            frame = cube_collapse(cube_res, collapse)
            if verbose:
                print('Done combining the residuals')
                timing(start_time)

            if full_output:
                return cube_res, frame
            else:
                return frame

        else:
            if scale_list is None:
                raise ValueError('Scaling factors vector must be provided')
            else:
                if np.array(scale_list).ndim > 1:
                    raise ValueError('Scaling factors vector is not 1d')
                if not scale_list.shape[0] == z:
                    raise ValueError('Scaling factors vector has wrong length')

            if verbose:
                print('SDI lst-sq modeling exploiting the spectral variability')
                print('{} spectral channels per IFS frame'.format(z))
                print('N annuli = {}, mean FWHM = '
                      '{:.3f}'.format(n_annuli, fwhm))
            res = pool_map(nproc, _leastsq_sdi_fr, iterable(range(n)),
                           scale_list, radius_int, fwhm, asize, n_segments,
                           delta_sep, tol, optim_scale_fact, metric,
                           dist_threshold, solver, imlib, interpolation,
                           collapse)
            cube_out = np.array(res)

            # Choosing not to exploit the rotational variability
            if adimsdi == 'skipadi':
                if verbose:
                    print('Skipping the ADI least-squares subtraction')
                    print('{} ADI frames'.format(n))
                    timing(start_time)

                cube_der = cube_derotate(cube_out, angle_list, imlib=imlib, 
                                         interpolation=interpolation, 
                                         nproc=nproc, **rot_options)
                frame = cube_collapse(cube_der, mode=collapse)

            # Exploiting rotational variability
            elif adimsdi == 'double':
                if verbose:
                    print('ADI lst-sq modeling exploiting the angular '
                          'variability')
                    print('{} ADI frames'.format(n))
                    timing(start_time)

                ARRAY = cube_out
                res = _leastsq_adi(cube_out, angle_list, fwhm, metric,
                                   dist_threshold, delta_rot, radius_int, asize,
                                   n_segments, nproc, solver, tol,
                                   optim_scale_fact, imlib, interpolation,
                                   collapse, verbose, full_output, 
                                   **rot_options)
                if full_output:
                    cube_out, cube_der, frame = res
                else:
                    frame = res

            if verbose:
                timing(start_time)

            if full_output:
                return cube_out, cube_der, frame
            else:
                return frame


def _leastsq_adi(cube, angle_list, fwhm=4, metric='manhattan',
                 dist_threshold=50, delta_rot=0.5, radius_int=0, asize=4,
                 n_segments=4, nproc=1, solver='lstsq', tol=1e-2,
                 optim_scale_fact=1, imlib='vip-fft', interpolation='lanczos4',
                 collapse='median', verbose=True, full_output=False,
                 **rot_options):
    """ Least-squares model PSF subtraction for ADI.
    """
    y = cube.shape[1]
    if not asize < y // 2:
        raise ValueError("asize is too large")

    angle_list = check_pa_vector(angle_list)
    n_annuli = int((y / 2 - radius_int) / asize)
    if verbose:
        print("Building {} annuli:".format(n_annuli))

    if isinstance(delta_rot, tuple):
        delta_rot = np.linspace(delta_rot[0], delta_rot[1], num=n_annuli)
    elif isinstance(delta_rot, (int, float)):
        delta_rot = [delta_rot] * n_annuli

    if nproc is None:
        nproc = cpu_count() // 2        # Hyper-threading doubles the # of cores

    annulus_width = asize
    if isinstance(n_segments, int):
        n_segments = [n_segments]*n_annuli
    elif n_segments == 'auto':
        n_segments = list()
        n_segments.append(2)    # for first annulus
        n_segments.append(3)    # for second annulus
        ld = 2 * np.tan(360/4/2) * annulus_width
        for i in range(2, n_annuli):    # rest of annuli
            radius = i * annulus_width
            ang = np.rad2deg(2 * np.arctan(ld / (2 * radius)))
            n_segments.append(int(np.ceil(360/ang)))

    # annulus-wise least-squares combination and subtraction
    cube_res = np.zeros_like(cube)

    ayxyx = []  # contains per-segment data
    pa_thresholds = []

    for ann in range(n_annuli):
        n_segments_ann = n_segments[ann]
        inner_radius_ann = radius_int + ann*annulus_width

        # angles
        pa_threshold = _define_annuli(angle_list, ann, n_annuli, fwhm,
                                      radius_int, asize, delta_rot[ann],
                                      n_segments_ann, verbose)[0]

        # indices
        indices = get_annulus_segments(cube[0], inner_radius=inner_radius_ann,
                                       width=asize, nsegm=n_segments_ann)
        ind_opt = get_annulus_segments(cube[0], inner_radius=inner_radius_ann,
                                       width=asize, nsegm=n_segments_ann,
                                       optim_scale_fact=optim_scale_fact)

        # store segment data for multiprocessing
        ayxyx += [(ann, indices[nseg][0], indices[nseg][1],
                   ind_opt[nseg][0], ind_opt[nseg][1]) for nseg in
                  range(n_segments_ann)]

        pa_thresholds.append(pa_threshold)

    msg = 'Patch-wise least-square combination and subtraction:'
    # reverse order of processing, as outer segments take longer
    res_patch = pool_map(nproc, _leastsq_patch, iterable(ayxyx[::-1]),
                         pa_thresholds, angle_list, metric, dist_threshold,
                         solver, tol, verbose=verbose, msg=msg,
                         progressbar_single=True)

    for patch in res_patch:
        matrix_res, yy, xx = patch
        cube_res[:, yy, xx] = matrix_res

    cube_der = cube_derotate(cube_res, angle_list, imlib, interpolation,
                             nproc=nproc, **rot_options)
    frame_der_median = cube_collapse(cube_der, collapse)

    if verbose:
        print('Done processing annuli')

    if full_output:
        return cube_res, cube_der, frame_der_median
    else:
        return frame_der_median


def _leastsq_patch(ayxyx,  pa_thresholds, angles, metric, dist_threshold,
                   solver, tol):
    """ Helper function for _leastsq_ann.

    Parameters
    ----------
    axyxy : tuple
        This tuple contains all per-segment data.
    pa_thresholds : list of list
        This is a per-annulus list of thresholds. 
    angles, metric, dist_threshold, solver, tol
        These parameters are the same for each annulus or segment.
    """
    iann, yy, xx, yy_opt, xx_opt = ayxyx
    pa_threshold = pa_thresholds[iann]
    
    values = ARRAY[:, yy, xx]  # n_frames x n_pxs_segment
   
    values_opt = ARRAY[:, yy_opt, xx_opt]

    n_frames = ARRAY.shape[0]

    if dist_threshold < 100:
        mat_dists_ann_full = pairwise_distances(values, metric=metric)
    else:
        mat_dists_ann_full = np.ones((values.shape[0], values.shape[0]))

    if pa_threshold > 0:
        mat_dists_ann = np.zeros_like(mat_dists_ann_full)
        for i in range(n_frames):
            ind_fr_i = _find_indices_adi(angles, i, pa_threshold, None, False)
            mat_dists_ann[i][ind_fr_i] = mat_dists_ann_full[i][ind_fr_i]
    else:
        mat_dists_ann = mat_dists_ann_full
        
    threshold = np.percentile(mat_dists_ann[mat_dists_ann != 0], dist_threshold)
    mat_dists_ann[mat_dists_ann > threshold] = np.nan
    mat_dists_ann[mat_dists_ann == 0] = np.nan

    matrix_res = np.zeros((values.shape[0], yy.shape[0]))
    for i in range(n_frames):
        vector = pn.DataFrame(mat_dists_ann[i])
        if vector.sum().values > 0:
            ind_ref = np.where(~np.isnan(vector))[0]
            A = values_opt[ind_ref]
            b = values_opt[i]
            if solver == 'lstsq':
                coef = sp.linalg.lstsq(A.T, b, cond=tol)[0]     # SVD method
            elif solver == 'nnls':
                coef = sp.optimize.nnls(A.T, b)[0]
            elif solver == 'lsq':   # TODO
                coef = sp.optimize.lsq_linear(A.T, b, bounds=(0, 1),
                                              method='trf',
                                              lsq_solver='lsmr')['x']
            else:
                raise ValueError("`solver` not recognized")
        else:
            msg = "No frames left in the reference set. Try increasing "
            msg += "`dist_threshold` or decreasing `delta_rot`."
            raise RuntimeError(msg)

        recon = np.dot(coef, values[ind_ref])
        matrix_res[i] = values[i] - recon

    return matrix_res, yy, xx


def _leastsq_sdi_fr(fr, wl, radius_int, fwhm, asize, n_segments, delta_sep,
                    tol, optim_scale_fact, metric, dist_threshold, solver,
                    imlib, interpolation, collapse):
    """ Optimized least-squares based subtraction on a multi-spectral frame
    (IFS data).
    """
    z, n, y_in, x_in = ARRAY.shape

    scale_list = check_scal_vector(wl)
    # rescaled cube, aligning speckles
    global MULTISPEC_FR
    MULTISPEC_FR = scwave(ARRAY[:, fr, :, :], scale_list, imlib=imlib,
                          interpolation=interpolation)[0]

    # Exploiting spectral variability (radial movement)
    fwhm = int(np.round(np.mean(fwhm)))
    annulus_width = int(np.ceil(asize))  # equal size for all annuli
    n_annuli = int(np.floor((y_in / 2 - radius_int) / annulus_width))

    if isinstance(n_segments, int):
        n_segments = [n_segments for _ in range(n_annuli)]
    elif n_segments == 'auto':
        n_segments = list()
        n_segments.append(2)  # for first annulus
        n_segments.append(3)  # for second annulus
        ld = 2 * np.tan(360 / 4 / 2) * annulus_width
        for i in range(2, n_annuli):  # rest of annuli
            radius = i * annulus_width
            ang = np.rad2deg(2 * np.arctan(ld / (2 * radius)))
            n_segments.append(int(np.ceil(360 / ang)))

    cube_res = np.zeros_like(MULTISPEC_FR)    # shape (z, resc_y, resc_x)

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

        indices = get_annulus_segments(MULTISPEC_FR[0], inner_radius,
                                       annulus_width, n_segments[ann])

        ind_opt = get_annulus_segments(MULTISPEC_FR[0], inner_radius,
                                       annulus_width, n_segments[ann],
                                       optim_scale_fact=optim_scale_fact)

        for seg in range(n_segments[ann]):
            yy = indices[seg][0]
            xx = indices[seg][1]
            segm_res = _leastsq_patch_ifs(seg, indices, ind_opt, wl, ann_center,
                                          fwhm, delta_sep_vec[ann], metric,
                                          dist_threshold, solver, tol)
            cube_res[:, yy, xx] = segm_res

    frame_desc = scwave(cube_res, scale_list, full_output=False, inverse=True,
                        y_in=y_in, x_in=x_in, imlib=imlib,
                        interpolation=interpolation, collapse=collapse)
    return frame_desc


def _leastsq_patch_ifs(nseg, indices, indices_opt, wl, ann_center, fwhm,
                       delta_sep, metric, dist_threshold, solver, tol):
    """ Helper function.
    """
    yy = indices[nseg][0]
    xx = indices[nseg][1]
    values = MULTISPEC_FR[:, yy, xx]

    yy_opt = indices_opt[nseg][0]
    xx_opt = indices_opt[nseg][0]
    values_opt = MULTISPEC_FR[:,  yy_opt, xx_opt]

    n_wls = ARRAY.shape[0]

    if dist_threshold < 100:
        mat_dists_ann_full = pairwise_distances(values, metric=metric)
    else:
        mat_dists_ann_full = np.ones((values.shape[0], values.shape[0]))

    if delta_sep > 0:
        mat_dists_ann = np.zeros_like(mat_dists_ann_full)
        for z in range(n_wls):
            ind_fr_i = _find_indices_sdi(wl, ann_center, z, fwhm, delta_sep)
            mat_dists_ann[z][ind_fr_i] = mat_dists_ann_full[z][ind_fr_i]
    else:
        mat_dists_ann = mat_dists_ann_full

    threshold = np.percentile(mat_dists_ann[mat_dists_ann != 0],
                              dist_threshold)
    mat_dists_ann[mat_dists_ann > threshold] = np.nan
    mat_dists_ann[mat_dists_ann == 0] = np.nan

    matrix_res = np.zeros((values.shape[0], yy.shape[0]))
    for z in range(n_wls):
        vector = pn.DataFrame(mat_dists_ann[z])
        if vector.sum().values != 0:
            ind_ref = np.where(~np.isnan(vector))[0]
            A = values_opt[ind_ref]
            b = values_opt[z]
            if solver == 'lstsq':
                coef = sp.linalg.lstsq(A.T, b, cond=tol)[0]     # SVD method
            elif solver == 'nnls':
                coef = sp.optimize.nnls(A.T, b)[0]
            elif solver == 'lsq':   # TODO
                coef = sp.optimize.lsq_linear(A.T, b, bounds=(0, 1),
                                              method='trf',
                                              lsq_solver='lsmr')['x']
            else:
                raise ValueError("solver not recognized")

        else:
            msg = "No frames left in the reference set. Try increasing "
            msg += "`dist_threshold` or decreasing `delta_sep`."
            raise RuntimeError(msg)

        recon = np.dot(coef, values[ind_ref])
        matrix_res[z] = values[z] - recon

    return matrix_res

