#! /usr/bin/env python
"""Module with a frame differencing algorithm for ADI post-processing."""

__author__ = "Carlos Alberto Gomez Gonzalez, Thomas Bédrine"
__all__ = ["frame_diff", "FRAME_DIFF_Params"]

import numpy as np
import pandas as pn
from hciplot import plot_frames
from multiprocessing import cpu_count
from dataclasses import dataclass
from typing import List
from enum import Enum
from sklearn.metrics import pairwise_distances
from .utils_pca import pca_annulus
from ..var import get_annulus_segments
from ..config.utils_param import setup_parameters, separate_kwargs_dict
from ..config.paramenum import Metric, Imlib, Interpolation, Collapse, ALGO_KEY
from ..config import time_ini, timing
from ..config.utils_conf import pool_map, iterable
from ..preproc import cube_derotate, cube_collapse, check_pa_vector
from ..preproc.derotation import _find_indices_adi, _define_annuli


@dataclass
class FRAME_DIFF_Params:
    """
    Set of parameters for the frame differencing module.

    See the function `frame_diff` below for the documentation.
    """

    cube: np.ndarray = None
    angle_list: np.ndarray = None
    fwhm: float = 4
    metric: Enum = Metric.MANHATTAN
    dist_threshold: int = 50
    n_similar: int = None
    delta_rot: int = 0.5
    radius_int: int = 2
    asize: int = 4
    ncomp: int = None
    imlib: Enum = Imlib.VIPFFT
    interpolation: Enum = Interpolation.LANCZOS4
    collapse: Enum = Collapse.MEDIAN
    nproc: int = 1
    verbose: bool = True
    debug: bool = False
    full_output: bool = False


def frame_diff(*all_args: List, **all_kwargs: dict):
    """Run the frame differencing algorithm.

    It uses vector distance (depending on
    ``metric``), using separately the pixels from different annuli of ``asize``
    width, to create pairs of most similar images. Then it performs pair-wise
    subtraction and combines the residuals.

    Parameters
    ----------
    all_args: list, optional
        Positionnal arguments for the frame diff algorithm. Full list of parameters
        below.
    all_kwargs: dictionary, optional
        Mix of keyword arguments that can initialize a FrameDiffParams and the optional
        'rot_options' dictionnary, with keyword values for "border_mode", "mask_val",
        "edge_blend", "interp_zeros", "ker" (see documentation of
        ``vip_hci.preproc.frame_rotate``). Can also contain a FrameDiffParams named as
        `algo_params`.

    Frame differencing parameters
    ----------
    cube : numpy ndarray, 3d
        Input cube.
    angle_list : 1d numpy ndarray
        Vector of derotation angles to align North up in your cube images.
    fwhm : float, optional
        Known size of the FWHM in pixels to be used. Default is 4.
    metric : str, optional
        Distance metric to be used ('cityblock', 'cosine', 'euclidean', 'l1',
        'l2', 'manhattan', 'correlation', etc). It uses the scikit-learn
        function ``sklearn.metrics.pairwise.pairwise_distances`` (check its
        documentation).
    dist_threshold : int
        Indices with a distance larger than ``dist_threshold`` percentile will
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
        processes will be set to cpu_count()/2. By default the algorithm works
        in single-process mode.
    imlib : str, opt
        See description in vip.preproc.frame_rotate()
    interpolation : str, opt
        See description in vip.preproc.frame_rotate()
    collapse: str, opt
        What to do with derotated residual cube? See options of
        vip.preproc.cube_collapse()
    verbose: bool, optional
        If True prints info to stdout.
    debug : bool, optional
        If True the distance matrices will be plotted and additional information
        will be given.

    Returns
    -------
    final_frame : numpy ndarray, 2d
        Median combination of the de-rotated cube.
    """

    # Separating the parameters of the ParamsObject from the optionnal rot_options
    class_params, rot_options = separate_kwargs_dict(
        initial_kwargs=all_kwargs, parent_class=FRAME_DIFF_Params
    )

    # Extracting the object of parameters (if any)
    algo_params = None
    if ALGO_KEY in rot_options.keys():
        algo_params = rot_options[ALGO_KEY]
        del rot_options[ALGO_KEY]

    if algo_params is None:
        algo_params = FRAME_DIFF_Params(*all_args, **class_params)

    global array
    array = algo_params.cube

    if algo_params.verbose:
        start_time = time_ini()

    y = array.shape[1]
    if not algo_params.asize < y // 2:
        raise ValueError("asize is too large")

    algo_params.angle_list = check_pa_vector(algo_params.angle_list)
    n_annuli = int((y / 2 - algo_params.radius_int) / algo_params.asize)
    if algo_params.verbose:
        if algo_params.ncomp is not None:
            msg = "{} annuli. Performing annular PCA subtraction with {} PCs "
            msg += "and pair-wise subtraction:"
            print(msg.format(n_annuli, algo_params.ncomp))
        else:
            msg = "{} annuli. Performing pair-wise subtraction:"
            print(msg.format(n_annuli))

    if algo_params.nproc is None:
        algo_params.nproc = cpu_count() // 2  # Hyper-threading doubles the # of cores

    # rotation options
    # border_mode = rot_options.get('border_mode','constant')
    # edge_blend = rot_options.get('edge_blend',None)
    # interp_zeros = rot_options.get('interp_zeros',False)
    # ker = rot_options.get('ker',1)

    add_params = {
        "ann": iterable(range(n_annuli)),
        "n_annuli": n_annuli,
        "angles": algo_params.angle_list,
        "debug": False,
    }

    func_params = setup_parameters(
        params_obj=algo_params, fkt=_pairwise_ann, as_list=True, **add_params
    )

    res = pool_map(
        algo_params.nproc,
        _pairwise_ann,
        *func_params,
        **rot_options,
    )  # border_mode, edge_blend,
    # interp_zeros, ker)

    final_frame = np.sum(res, axis=0)

    if algo_params.verbose:
        print("Done processing annuli")
        timing(start_time)

    return final_frame


def _pairwise_ann(
    ann,
    n_annuli,
    fwhm,
    angles,
    delta_rot,
    metric,
    dist_threshold,
    n_similar,
    radius_int,
    asize,
    ncomp,
    imlib,
    interpolation,
    collapse,
    verbose,
    debug=False,
    **rot_options,
):
    """
    Helper functions for pair-wise subtraction for a single annulus.
    """
    start_time = time_ini(False)

    n_frames = array.shape[0]

    pa_threshold, in_rad, ann_center = _define_annuli(
        angles, ann, n_annuli, fwhm, radius_int, asize, delta_rot, 1, verbose
    )
    if ncomp is not None:
        arrayin = pca_annulus(
            array,
            None,
            ncomp,
            asize,
            ann_center,
            svd_mode="lapack",
            scaling=None,
            collapse=None,
        )
    else:
        arrayin = array

    yy, xx = get_annulus_segments(array[0], inner_radius=in_rad, width=asize, nsegm=1)[
        0
    ]
    values = arrayin[:, yy, xx]  # n_frames x n_pxs_annulus

    if debug:
        print("Done taking pixel intensities from annulus.")
        timing(start_time)

    mat_dists_ann_full = pairwise_distances(values, metric=metric)

    if pa_threshold > 0:
        mat_dists_ann = np.zeros_like(mat_dists_ann_full)
        for i in range(n_frames):
            ind_fr_i = _find_indices_adi(angles, i, pa_threshold, None, False)
            mat_dists_ann[i][ind_fr_i] = mat_dists_ann_full[i][ind_fr_i]
    else:
        mat_dists_ann = mat_dists_ann_full

    if debug:
        msg = "Done calculating the {} distance for annulus {}"
        print(msg.format(metric, ann + 1))
        timing(start_time)

    threshold = np.percentile(mat_dists_ann[mat_dists_ann != 0], dist_threshold)
    mat_dists_ann[mat_dists_ann > threshold] = np.nan
    mat_dists_ann[mat_dists_ann == 0] = np.nan
    if not mat_dists_ann[~np.isnan(mat_dists_ann)].size > 0:
        raise RuntimeError("No pairs left. Decrease thresholds")

    if debug:
        plot_frames(mat_dists_ann)
        print("Done thresholding/checking distances.")
        timing(start_time)

    # median of n ``n_similar`` most similar patches
    cube_res = []
    if n_similar is not None:
        angles_list = []
        if n_similar < 3:
            raise ValueError("n_similar must be >= 3 or None")
        for i in range(n_frames):
            vector = pn.DataFrame(mat_dists_ann[i])
            if vector.sum().values == 0:
                continue
            else:
                vector_sorted = vector[:][0].sort_values()[:n_similar]
                ind_n_similar = vector_sorted.index.values
                # median subtraction
                res = values[i] - np.median((values[ind_n_similar]), axis=0)
                cube_res.append(res)
                angles_list.append(angles[i])
        angles_list = np.array(angles_list)
        cube_res = np.array(cube_res)

    # taking just the most similar frame
    else:
        ind = []
        for i in range(n_frames):
            vector = pn.DataFrame(mat_dists_ann[i])
            if vector.sum().values == 0:
                continue
            else:
                ind.append((i, vector.idxmin().tolist()[0]))
                ind.append((vector.idxmin().tolist()[0], i))

        if debug:
            print("Done finding pairs. Total found: ", len(ind) / 2)
            timing(start_time)

        df = pn.DataFrame(ind)  # sorting using pandas dataframe
        df.columns = ["i", "j"]
        df = df.sort_values("i")

        indices = df.values
        indices = indices.astype(int)  # back to a ndarray int type

        size = indices.shape[0]
        angles_list = np.zeros((size))
        for i in range(size):
            # filter of the angles vector
            angles_list[i] = angles[indices[i][0]]

        cube_res = np.zeros((size, yy.shape[0]))
        # pair-wise subtraction
        for i in range(size):
            res = values[indices[i][0]] - values[indices[i][1]]
            cube_res[i] = res

    cube_out = np.zeros((cube_res.shape[0], array.shape[1], array.shape[2]))
    for i in range(cube_res.shape[0]):
        cube_out[i, yy, xx] = cube_res[i]

    cube_der = cube_derotate(
        cube_out,
        angles_list,
        imlib=imlib,
        interpolation=interpolation,
        mask_val=0,
        **rot_options,
    )
    frame_collapse = cube_collapse(cube_der, collapse)

    return frame_collapse
