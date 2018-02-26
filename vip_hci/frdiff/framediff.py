#! /usr/bin/env python

"""
Module with a frame differencing algorithm for ADI post-processing.
"""

from __future__ import division
from __future__ import print_function

__author__ = 'Carlos Alberto Gomez Gonzalez'
__all__ = ['frame_diff']

import numpy as np
import pandas as pn
import itertools as itt
from multiprocessing import Pool, cpu_count
from sklearn.metrics import pairwise_distances
from ..var import get_annulus_segments, pp_subplots
from ..preproc import cube_derotate, cube_collapse, check_pa_vector
from ..conf import time_ini, timing
from ..pca.utils_pca import pca_annulus
from ..madi.adi_utils import _compute_pa_thresh, _find_indices, _define_annuli
from ..conf.utils_conf import eval_func_tuple as EFT


array = None


def frame_diff(cube, angle_list, fwhm=4, metric='manhattan', dist_threshold=50,
               n_similar=None, delta_rot=0.5, radius_int=2, asize=4, ncomp=None,
               nproc=1, verbose=True, debug=False):
    """ Frame differencing algorithm. It uses vector distance (depending on
    ``metric``), using separately the pixels from different annuli of ``asize``
    width, to create pairs of most similar images. Then it performs pair-wise
    subtraction and combines the residuals.
    
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
    n_similar : None or int, optional
        If a postive integer value is given, then a median combination of
        ``n_similar`` frames will be used instead of the most similar one.
    delta_rot : int
        Minimum parallactic angle distance between the pairs.
    radius_int : int, optional
        The radius of the innermost annulus. By default is 0, if >0 then the
        central circular area is discarded.
    asize : int, optional
        The size of the annuli, in pixels.
    ncomp : None or int, optional
        If a positive integer value is given, then the annulus-wise PCA low-rank
        approximation with ``ncomp`` principal components will be subtracted.
        The pairwise subtraction will be performed on these residuals.
    nproc : None or int, optional
        Number of processes for parallel computing. If None the number of
        processes will be set to (cpu_count()/2). By default the algorithm works
        in single-process mode.
    verbose: bool, optional
        If True prints info to stdout.
    debug : bool, optional
        If True the distance matrices will be plotted and additional information
        will be given.
        
    Returns
    -------
    final_frame : array_like, 2d
        Median combination of the de-rotated cube.
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
        if ncomp is not None:
            msg = "{:} annuli. Performing annular PCA subtraction with {:} PCs "
            msg += "and pair-wise subtraction:\n"
            print(msg.format(n_annuli, ncomp))
        else:
            msg = "{:} annuli. Performing pair-wise subtraction:\n"
            print(msg.format(n_annuli))

    if nproc is None:   # Hyper-threading "duplicates" the cores -> cpu_count/2
        nproc = (cpu_count()/2)

    # annulus-wise pair-wise subtraction
    final_frame = []
    if nproc == 1:
        for ann in range(n_annuli):
            res_ann = _pairwise_ann(ann, n_annuli, fwhm, angle_list, delta_rot,
                                    metric, dist_threshold, n_similar,
                                    radius_int, asize, ncomp, verbose, debug)
            final_frame.append(res_ann)
    elif nproc > 1:
        pool = Pool(processes=int(nproc))
        res = pool.map(EFT, zip(itt.repeat(_pairwise_ann), range(n_annuli),
                                itt.repeat(n_annuli), itt.repeat(fwhm),
                                itt.repeat(angle_list), itt.repeat(delta_rot),
                                itt.repeat(metric), itt.repeat(dist_threshold),
                                itt.repeat(n_similar), itt.repeat(radius_int),
                                itt.repeat(asize), itt.repeat(ncomp),
                                itt.repeat(verbose), itt.repeat(debug),
                                itt.repeat(nproc)))
        final_frame = np.array(res)
        pool.close()
    final_frame = np.sum(final_frame, axis=0)

    if verbose:
        print('Done processing annuli')
        timing(start_time)

    return final_frame


def _pairwise_ann(ann, n_annuli, fwhm, angles, delta_rot, metric,
                  dist_threshold, n_similar, radius_int, asize, ncomp, verbose,
                  debug=False, nproc=1):
    """
    Helper functions for pair-wise subtraction for a single annulus.
    """
    start_time = time_ini(False)

    n_frames = array.shape[0]

    pa_threshold, in_rad, ann_center = _define_annuli(angles, ann, n_annuli,
                                                      fwhm, radius_int, asize,
                                                      delta_rot, 1, verbose)
    if ncomp is not None:
        arrayin = pca_annulus(array, None, ncomp, asize, ann_center,
                              svd_mode='lapack', scaling=None, collapse=None)
    else:
        arrayin = array

    yy, xx = get_annulus_segments(array[0], inner_radius=in_rad, width=asize,
                                  nsegm=1)[0]
    values = arrayin[:, yy, xx]  # n_frames x n_pxs_annulus

    if debug:
        print('Done taking pixel intensities from annulus.')
        timing(start_time)

    mat_dists_ann_full = pairwise_distances(values, metric=metric)

    if pa_threshold > 0:
        mat_dists_ann = np.zeros_like(mat_dists_ann_full)
        for i in range(n_frames):
            ind_fr_i = _find_indices(angles, i, pa_threshold, None, False)
            mat_dists_ann[i][ind_fr_i] = mat_dists_ann_full[i][ind_fr_i]
    else:
        mat_dists_ann = mat_dists_ann_full

    if debug:
        #pp_subplots(mat_dists_ann)
        msg = 'Done calculating the {:} distance for annulus {:}'
        print(msg.format(metric, ann+1))
        timing(start_time)

    threshold = np.percentile(mat_dists_ann[mat_dists_ann != 0.0],
                              dist_threshold)
    mat_dists_ann[mat_dists_ann > threshold] = np.nan
    mat_dists_ann[mat_dists_ann == 0.0] = np.nan
    if not mat_dists_ann[~np.isnan(mat_dists_ann)].size > 0:
        raise RuntimeError('No pairs left. Decrease thresholds')

    if debug:
        pp_subplots(mat_dists_ann)
        print('Done thresholding/checking distances.')
        timing(start_time)

    # median of n ``n_similar`` most similar patches
    cube_res = []
    if n_similar is not None:
        if not n_similar >= 3:
            raise ValueError("n_similar must be >= 3 or None")
        for i in range(n_frames):
            vector = pn.DataFrame(mat_dists_ann[i])
            vector.columns = ['i']
            if vector.sum().values == 0:
                continue
            else:
                vector_sorted = vector.i.sort_values()[:n_similar]
                ind_n_similar = vector_sorted.index.values
                # median subtraction
                res = values[i] - np.median((values[ind_n_similar]), axis=0)
                cube_res.append(res)

        cube_res = np.array(cube_res)
        angles_list = angles

    # taking just the most similar frame
    else:
        ind = []
        for i in range(n_frames):
            vector = pn.DataFrame(mat_dists_ann[i])
            vector.columns = ['i']
            if vector.sum().values == 0:
                continue
            else:
                ind.append((i, vector.idxmin().tolist()[0]))
                ind.append((vector.idxmin().tolist()[0], i))

        if debug:
            print('Done finding pairs. Total found: ', len(ind)/2)
            timing(start_time)

        df = pn.DataFrame(ind)  # sorting using pandas dataframe
        df.columns = ['i', 'j']
        df = df.sort_values('i')

        indices = df.values
        indices = indices.astype(int)  # back to a ndarray int type

        size = indices.shape[0]
        angles_list = np.zeros((size))
        for i in range(size):
            angles_list[i] = angles[indices[i][0]]  # filter of the angles vector

        cube_res = np.zeros((size, yy.shape[0]))
        # pair-wise subtraction
        for i in range(size):
            res = values[indices[i][0]] - values[indices[i][1]]
            cube_res[i] = res

    cube_out = np.zeros((cube_res.shape[0], array.shape[1], array.shape[2]))
    cube_out[:, yy, xx] = cube_res
    cube_der = cube_derotate(cube_out, angles_list)
    frame_der_median = cube_collapse(cube_der, 'median')

    if verbose and nproc == 1:
        timing(start_time)

    return frame_der_median


def _pw_rot_res(cube, angle_list, fwhm=4, delta_rot=0.5, inner_radius=2,
                asize=4, verbose=True, debug=False):
    """

    Parameters
    ----------
    cube : array_like, 3d
        Input cube.
    angle_list : array_like, 1d
        Corresponding parallactic angle for each frame.
    fwhm : float, optional
        Known size of the FHWM in pixels to be used. Default is 4.
    delta_rot : int
        Minimum parallactic angle distance between the pairs.
    inner_radius : int, optional
        The radius of the innermost annulus. By default is 0, if >0 then the
        central circular area is discarded.
    asize : int, optional
        The size of the annuli, in pixels.
    verbose: bool, optional
        If True prints info to stdout.
    debug : bool, optional
        If True the distance matrices will be plotted and additional information
        will be given.

    Returns
    -------
    final_frame : array_like, 2d

    """
    array = cube

    if verbose:
        start_time = time_ini()

    n_frames = array.shape[0]
    y = array.shape[1]
    if not asize < np.floor((y / 2)):
        raise ValueError("asize is too large")

    angle_list = check_pa_vector(angle_list)

    ann_center = (inner_radius + (asize / 2.0))
    pa_threshold = _compute_pa_thresh(ann_center, fwhm, delta_rot)
    if debug:
        print(pa_threshold)

    # annulus-wise pair-wise subtraction
    res = []
    for i in range(n_frames):
        indp, indn = _find_indices(angle_list, i, pa_threshold,
                                   out_closest=True)
        if debug:
            print(indp, indn)

        res.append(array[i] - array[indn])
        if indn == n_frames-1: break

    if verbose:
        print('Done processing annulus')
        timing(start_time)

    return np.array(res)
