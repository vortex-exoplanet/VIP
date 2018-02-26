#! /usr/bin/env python

"""
Module with a frame differencing algorithm for ADI post-processing.
"""

from __future__ import division
from __future__ import print_function

__author__ = 'Carlos Alberto Gomez Gonzalez'
__all__ = ['xloci']

import numpy as np
import scipy as sp
import pandas as pn
import itertools as itt
from multiprocessing import Pool, cpu_count
from sklearn.metrics import pairwise_distances
from ..var import get_annulus_segments
from ..preproc import cube_derotate, cube_collapse, check_pa_vector
from ..conf import time_ini, timing
from ..madi.adi_utils import _find_indices, _define_annuli
from ..conf.utils_conf import eval_func_tuple as EFT


array = None


def xloci(cube, angle_list, fwhm=4, metric='manhattan', dist_threshold=50,
          delta_rot=0.5, radius_int=0, asize=4, n_segments=4, nproc=1,
          solver='lstsq', tol=1e-3, verbose=True, full_output=False):
    """ LOCI style algorithm that models a PSF (for ADI data) with a
    least-square combination of neighbouring frames (solving the equation
    a x = b by computing a vector x of coefficients that minimizes the
    Euclidean 2-norm || b - a x ||^2).
    
    Parameters
    ----------
    cube : array_like, 3d
        Input cube.
    angle_list : array_like, 1d
        Corresponding parallactic angle for each frame.
    fwhm : float, optional
        Known size of the FHWM in pixels to be used. Default is 4.
    metric : str, optional
        Distance metric to be used ('cityblock', 'cosine', 'euclidean', 'l1',
        'l2', 'manhattan', 'correlation', etc). It uses the scikit-learn
        function ``sklearn.metrics.pairwise.pairwise_distances`` (check its
        documentation).
    dist_threshold : int
        Indices with a distance larger thatn ``dist_threshold`` percentile will
        initially discarded.
    delta_rot : int
        Minimum parallactic angle distance between the pairs.
    radius_int : int, optional
        The radius of the innermost annulus. By default is 0, if >0 then the
        central circular area is discarded.
    asize : int, optional
        The size of the annuli, in pixels.
    n_segments : int or list of ints, optional
        The number of segments for each annulus. When a single integer is given
        it is used for all annuli.
    nproc : None or int, optional
        Number of processes for parallel computing. If None the number of
        processes will be set to (cpu_count()/2). By default the algorithm works
        in single-process mode.
    solver : {'lstsq', 'nnls'}, str optional
        Choosing the solver of the least squares problem. ``lstsq`` uses the
        standard scipy least squares solver. ``nnls`` uses the scipy
        non-negative least squares solver.
    tol : float, optional
        Valid when ``solver`` is set to lstsq. Sets the cutoff for 'small'
        singular values; used to determine effective rank of a. Singular values
        smaller than ``tol * largest_singular_value`` are considered zero.
        Smaller values of ``tol`` lead to smaller residuals (more aggressive
        subtraction).
    verbose: bool, optional
        If True prints info to stdout.
    full_output: bool, optional
        Whether to return the final median combined image only or with other
        intermediate arrays.
        
    Returns
    -------
    frame_der_median : array_like, 2d
        Median combination of the de-rotated cube of residuals.

    If ``full_output`` is True, the following intermediate arrays are returned:
    cube_res, cube_der, frame_der_median

    """
    global array
    array = cube

    if verbose:
        start_time = time_ini()

    y = array.shape[1]
    if not asize < np.floor((y / 2)):
        raise ValueError("asize is too large")

    angle_list = check_pa_vector(angle_list)
    n_annuli = int(np.floor((y / 2 - radius_int) / asize))
    if verbose:
        msg = "{:} annuli. Performing least-square combination and "
        msg += "subtraction:\n"
        print(msg.format(n_annuli))

    if nproc is None:   # Hyper-threading "duplicates" the cores -> cpu_count/2
        nproc = (cpu_count()/2)

    annulus_width = asize
    if isinstance(n_segments, int):
        n_segments = [n_segments for _ in range(n_annuli)]
    elif n_segments == 'auto':
        n_segments = []
        n_segments.append(2)    # for first annulus
        n_segments.append(3)    # for second annulus
        ld = 2 * np.tan(360/4/2) * annulus_width
        for i in range(2, n_annuli):    # rest of annuli
            radius = i * annulus_width
            ang = np.rad2deg(2 * np.arctan(ld / (2 * radius)))
            n_segments.append(int(np.ceil(360/ang)))

    # annulus-wise least-squares combination and subtraction
    cube_res = np.zeros((array.shape[0], array.shape[1], array.shape[2]))
    for ann in range(n_annuli):
        n_segments_ann = n_segments[ann]
        inner_radius_ann = radius_int + ann*annulus_width

        indices = get_annulus_segments(array[0], inner_radius=inner_radius_ann,
                                       width=asize, nsegm=n_segments_ann)

        res_ann = _leastsq_ann(indices, ann, n_annuli, fwhm, angle_list,
                               delta_rot, metric, dist_threshold, radius_int,
                               asize, n_segments_ann, nproc, solver, tol,
                               verbose)

        for j in range(n_segments_ann):
            yy = indices[j][0]
            xx = indices[j][1]
            cube_res[:, yy, xx] = res_ann[j]

    cube_der = cube_derotate(cube_res, angle_list)
    frame_der_median = cube_collapse(cube_der, 'median')

    if verbose:
        print('Done processing annuli')
        timing(start_time)

    if full_output:
        return cube_res, cube_der, frame_der_median
    else:
        return frame_der_median


def _leastsq_ann(indices, ann, n_annuli, fwhm, angles, delta_rot, metric,
                 dist_threshold, radius_int, asize, n_segments_ann, nproc,
                 solver, tol, verbose):
    """ Helper function for xloci. Least-squares combination and subtraction
    for each segment in an annulus, applying a rotation threshold.
    """
    start_time = time_ini(False)

    pa_threshold, _, _ = _define_annuli(angles, ann, n_annuli, fwhm, radius_int,
                                        asize, delta_rot, n_segments_ann,
                                        verbose)
    res = []
    if nproc == 1:
        for j in range(n_segments_ann):
            segm_res = _leastsq_patch(j, indices, angles, pa_threshold, metric,
                                      dist_threshold, solver, tol)
            res.append(segm_res)

    elif nproc > 1:
        pool = Pool(processes=int(nproc))
        res = pool.map(EFT, zip(itt.repeat(_leastsq_patch),
                                range(n_segments_ann), itt.repeat(indices),
                                itt.repeat(angles), itt.repeat(pa_threshold),
                                itt.repeat(metric), itt.repeat(dist_threshold),
                                itt.repeat(solver), itt.repeat(tol)))
        pool.close()

    else:
        raise ValueError("nproc must be a positive integer")

    if verbose and nproc == 1:
        timing(start_time)

    return res


def _leastsq_patch(nseg, indices, angles, pa_threshold, metric, dist_threshold,
                   solver, tol):
    """ Helper function for _leastsq_ann.
    """
    yy = indices[nseg][0]
    xx = indices[nseg][1]
    values = array[:, yy, xx]  # n_frames x n_pxs_segment
    n_frames = array.shape[0]

    mat_dists_ann_full = pairwise_distances(values, metric=metric)

    if pa_threshold > 0:
        mat_dists_ann = np.zeros_like(mat_dists_ann_full)
        for i in range(n_frames):
            ind_fr_i = _find_indices(angles, i, pa_threshold, None, False)
            mat_dists_ann[i][ind_fr_i] = mat_dists_ann_full[i][ind_fr_i]
    else:
        mat_dists_ann = mat_dists_ann_full

    threshold = np.percentile(mat_dists_ann[mat_dists_ann != 0.0],
                              dist_threshold)
    mat_dists_ann[mat_dists_ann > threshold] = np.nan
    mat_dists_ann[mat_dists_ann == 0.0] = np.nan

    matrix_res = np.zeros((values.shape[0], yy.shape[0]))
    for i in range(n_frames):
        vector = pn.DataFrame(mat_dists_ann[i])
        vector.columns = ['i']
        if vector.sum().values != 0:
            ind_ref = np.where(~np.isnan(vector))[0]
            A = values[ind_ref]
            b = values[i]
            if solver == 'lstsq':
                coef = sp.linalg.lstsq(A.T, b, cond=tol)[0]     # SVD method
            elif solver == 'nnls':
                coef = sp.optimize.nnls(A.T, b)[0]
            elif solver == 'lsq':
                coef = sp.optimize.lsq_linear(A.T, b, bounds=(0, 1),
                                              method='trf',
                                              lsq_solver='lsmr')['x']
            else:
                raise ValueError("solver not recognized")

        recon = np.dot(coef, values[ind_ref])
        matrix_res[i] = values[i] - recon

    return matrix_res

