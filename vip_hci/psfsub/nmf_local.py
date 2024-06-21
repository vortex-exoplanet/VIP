#! /usr/bin/env python
"""
Module with NMF algorithm in concentric annuli for ADI/RDI.
"""

__author__ = "Valentin Christiaens, Thomas Bédrine"
__all__ = ["nmf_annular", "NMF_ANNULAR_Params"]

import numpy as np
from multiprocessing import cpu_count
from sklearn.decomposition import NMF
from dataclasses import dataclass, field
from typing import Tuple, List, Union
from enum import Enum
from ..preproc import cube_derotate, cube_collapse, check_pa_vector
from ..preproc.derotation import _find_indices_adi, _define_annuli
from ..var import get_annulus_segments, matrix_scaling
from ..config.utils_param import setup_parameters, separate_kwargs_dict
from ..config.paramenum import Initsvd, Imlib, Interpolation, HandleNeg, Collapse, ALGO_KEY
from ..config import timing, time_ini
from ..config.utils_conf import pool_map, iterable


@dataclass
class NMF_ANNULAR_Params:
    """
    Set of parameters for the NMF annular algorithm.

    See function `nmf_annular` below for the documentation.
    """

    cube: np.ndarray = None
    angle_list: np.ndarray = None
    cube_ref: np.ndarray = None
    radius_int: int = 0
    fwhm: float = 4
    asize: int = 4
    n_segments: int = 1
    delta_rot: Union[float, Tuple[float]] = (0.1, 1)
    ncomp: int = 1
    init_svd: Enum = Initsvd.NNDSVD
    nproc: int = 1
    min_frames_lib: int = 2
    max_frames_lib: int = 200
    scaling: Enum = None
    imlib: Enum = Imlib.VIPFFT
    interpolation: Enum = Interpolation.LANCZOS4
    collapse: Enum = Collapse.MEDIAN
    full_output: bool = False
    verbose: bool = True
    theta_init: float = 0
    weights: List = None
    cube_sig: np.ndarray = None
    handle_neg: Enum = HandleNeg.MASK
    max_iter: int = 1000
    random_state: int = None
    nmf_args: dict = field(default_factory=lambda: {})


# TODO: update the doc of the params (some are missing)
def nmf_annular(*all_args: List, **all_kwargs: dict):
    """Non Negative Matrix Factorization in concentric annuli, for ADI/RDI
    sequences. Alternative to the annular ADI-PCA processing that does not rely
    on SVD or ED for obtaining a low-rank approximation of the datacube.
    This function embeds the scikit-learn NMF algorithm solved through either
    the coordinate descent or the multiplicative update method.

    Parameters
    ----------
    all_args: list, optional
        Positionnal arguments for the NMF annular algorithm. Full list of parameters
        below.
    all_kwargs: dictionary, optional
        Mix of keyword arguments that can initialize a NMFAnnParams and the optional
        'rot_options' dictionnary, with keyword values for "border_mode", "mask_val",
        "edge_blend", "interp_zeros", "ker" (see documentation of
        ``vip_hci.preproc.frame_rotate``). Can also contain a NMFAnnParams named as
        `algo_params`..

    NMF annular parameters
    ----------
    cube : numpy ndarray, 3d
        Input cube.
    angle_list : 1d numpy ndarray
        Vector of derotation angles to align North up in your cube images.
    cube_ref : numpy ndarray, 3d, optional
        Reference library cube. For Reference Star Differential Imaging.
    radius_int : int, optional
        The radius of the innermost annulus. By default is 0, if >0 then the
        central circular region is discarded.
    fwhm : float, optional
        Size of the FWHM in pixels. Default is 4.
    asize : float, optional
        The size of the annuli, in pixels.
    n_segments : int or list of ints or 'auto', optional
        The number of segments for each annulus. When a single integer is given
        it is used for all annuli. When set to 'auto', the number of segments is
        automatically determined for every annulus, based on the annulus width.
    delta_rot : float or tuple of floats, optional
        Factor for adjusting the parallactic angle threshold, expressed in
        FWHM. Default is 1 (excludes 1 FWHM on each side of the considered
        frame). If a tuple of two floats is provided, they are used as the lower
        and upper intervals for the threshold (grows linearly as a function of
        the separation). !!! Important: this is used even if a reference cube
        is provided for RDI. This is to allow ARDI (PCA library built from both
        science and reference cubes). If you want to do pure RDI, set delta_rot
        to an arbitrarily high value such that the condition is never fulfilled
        for science frames to make it in the PCA library.
    ncomp : int, optional
        How many components are used as for low-rank approximation of the
        datacube.
    scaling : {None, "temp-mean", spat-mean", "temp-standard",
        "spat-standard"}, None or str optional
        Pixel-wise scaling mode using ``sklearn.preprocessing.scale``
        function. If set to None, the input matrix is left untouched. Otherwise:

        * ``temp-mean``: temporal px-wise mean is subtracted.

        * ``spat-mean``: spatial mean is subtracted.

        * ``temp-standard``: temporal mean centering plus scaling pixel values
          to unit variance (temporally).

        * ``spat-standard``: spatial mean centering plus scaling pixel values
          to unit variance (spatially).

        DISCLAIMER: Using ``temp-mean`` or ``temp-standard`` scaling can improve
        the speckle subtraction for ASDI or (A)RDI reductions. Nonetheless, this
        involves a sort of c-ADI preprocessing, which (i) can be dangerous for
        datasets with low amount of rotation (strong self-subtraction), and (ii)
        should probably be referred to as ARDI (i.e. not RDI stricto sensu).
    max_iter : int optional
        The number of iterations for the coordinate descent solver.
    random_state : int or None, optional
        Controls the seed for the Pseudo Random Number generator.
    mask_center_px : None or int
        If None, no masking is done. If an integer > 1 then this value is the
        radius of the circular mask.
    source_xy : tuple of int, optional
        For ADI-PCA, this triggers a frame rejection in the PCA library, with
        ``source_xy`` as the coordinates X,Y of the center of the annulus where
        the PA criterion is estimated. When ``ncomp`` is a tuple, a PCA grid is
        computed and the S/Ns (mean value in a 1xFWHM circular aperture) of the
        given (X,Y) coordinates are computed.
    delta_rot : int, optional
        Factor for tunning the parallactic angle threshold, expressed in FWHM.
        Default is 1 (excludes 1xFWHM on each side of the considered frame).
    init_svd: str, optional {'nnsvd','nnsvda','random'}
        Method used to initialize the iterative procedure to find H and W.
        'nndsvd': non-negative double SVD recommended for sparseness
        'nndsvda': NNDSVD where zeros are filled with the average of cube;
        recommended when sparsity is not desired
        'random': random initial non-negative matrix
    collapse : {'median', 'mean', 'sum', 'trimmean'}, str optional
        Sets the way of collapsing the frames for producing a final image.
    full_output: boolean, optional
        Whether to return the final median combined image only or with other
        intermediate arrays.
    verbose : {True, False}, bool optional
        If True prints intermediate info and timing.
    nmf_args: dictionary, optional
        Additional arguments for scikit-learn NMF algorithm. See:
        https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html

    Returns
    -------
    If full_output is False the final frame is returned. If True the algorithm
    returns the reshaped NMF components, the reconstructed cube, the residuals,
    the residuals derotated and the final frame.

    """
    # Separating the parameters of the ParamsObject from the optionnal rot_options
    class_params, rot_options = separate_kwargs_dict(initial_kwargs=all_kwargs,
                                                     parent_class=NMF_ANNULAR_Params
                                                     )

    # Extracting the object of parameters (if any)
    algo_params = None
    if ALGO_KEY in rot_options.keys():
        algo_params = rot_options[ALGO_KEY]
        del rot_options[ALGO_KEY]

    if algo_params is None:
        algo_params = NMF_ANNULAR_Params(*all_args, **class_params)

    if algo_params.verbose:
        global start_time
        start_time = time_ini()

    array = algo_params.cube
    if array.ndim != 3:
        raise TypeError("Input array is not a cube or 3d array")
    if array.shape[0] != algo_params.angle_list.shape[0]:
        raise TypeError("Input vector or parallactic angles has wrong length")

    n, y, _ = array.shape

    algo_params.angle_list = check_pa_vector(algo_params.angle_list)
    n_annuli = int((y / 2 - algo_params.radius_int) / algo_params.asize)

    if isinstance(algo_params.delta_rot, tuple):
        algo_params.delta_rot = np.linspace(
            algo_params.delta_rot[0], algo_params.delta_rot[1], num=n_annuli
        )
    elif isinstance(algo_params.delta_rot, (int, float)):
        algo_params.delta_rot = [algo_params.delta_rot] * n_annuli

    if isinstance(algo_params.n_segments, int):
        algo_params.n_segments = [
            algo_params.n_segments for _ in range(n_annuli)]
    elif algo_params.n_segments == "auto":
        algo_params.n_segments = list()
        algo_params.n_segments.append(2)  # for first annulus
        algo_params.n_segments.append(3)  # for second annulus
        ld = 2 * np.tan(360 / 4 / 2) * algo_params.asize
        for i in range(2, n_annuli):  # rest of annuli
            radius = i * algo_params.asize
            ang = np.rad2deg(2 * np.arctan(ld / (2 * radius)))
            algo_params.n_segments.append(int(np.ceil(360 / ang)))

    if algo_params.verbose:
        msg = "N annuli = {}, FWHM = {:.3f}"
        print(msg.format(n_annuli, algo_params.fwhm))
        print("NMF per annulus (or annular sectors):")

    if (
        algo_params.nproc is None
    ):  # Hyper-threading "duplicates" the cores -> cpu_count/2
        algo_params.nproc = cpu_count() // 2

    # how to handle negative values
    if algo_params.handle_neg == HandleNeg.NULL:
        array[np.where(array < 0)] = 0
    elif algo_params.handle_neg == HandleNeg.SUBTR_MIN:
        array -= np.amin(array)
    elif not algo_params.handle_neg == HandleNeg.MASK:
        raise ValueError("Mode to handle neg. pixels not recognized")

    # The annuli are built, and the corresponding PA thresholds for frame
    # rejection are calculated (at the center of the annulus)
    cube_out = np.zeros_like(array)
    cube_recon = np.zeros_like(array)
    H_comps = np.zeros([algo_params.ncomp, array.shape[1], array.shape[2]])
    if algo_params.cube_ref is None:
        strict = False
    else:
        strict = True
    for ann in range(n_annuli):
        if isinstance(algo_params.ncomp, tuple) or isinstance(
            algo_params.ncomp, np.ndarray
        ):
            if len(algo_params.ncomp) == n_annuli:
                ncompann = algo_params.ncomp[ann]
            else:
                raise TypeError(
                    "If `ncomp` is a tuple, it must match the " "number of annuli"
                )
        else:
            ncompann = algo_params.ncomp

        n_segments_ann = algo_params.n_segments[ann]

        add_params = {
            "ann": ann,
            "n_annuli": n_annuli,
            "annulus_width": algo_params.asize,
            "n_segments": n_segments_ann,
            "delta_rot": algo_params.delta_rot[ann],
            "strict": strict,
        }

        func_params = setup_parameters(
            params_obj=algo_params, fkt=_define_annuli, show_params=False, **add_params
        )

        res_ann_par = _define_annuli(**func_params)
        pa_thr, inner_radius, ann_center = res_ann_par
        indices = get_annulus_segments(
            array[0],
            inner_radius,
            algo_params.asize,
            n_segments_ann,
            algo_params.theta_init,
        )
        # Library matrix is created for each segment and scaled if needed
        for j in range(n_segments_ann):
            yy = indices[j][0]
            xx = indices[j][1]
            if algo_params.handle_neg == "mask":
                npts = range(len(yy))
                if algo_params.cube_sig is not None:
                    yp = [
                        yy[i]
                        for i in npts
                        if np.amin(
                            array[:, yy[i], xx[i]]
                            - np.abs(algo_params.cube_sig[:, yy[i], xx[i]])
                        )
                        > 0
                    ]
                    xp = [
                        xx[i]
                        for i in npts
                        if np.amin(
                            array[:, yy[i], xx[i]]
                            - np.abs(algo_params.cube_sig[:, yy[i], xx[i]])
                        )
                        > 0
                    ]
                else:
                    yp = [yy[i]
                          for i in npts if np.amin(array[:, yy[i], xx[i]]) > 0]
                    xp = [xx[i]
                          for i in npts if np.amin(array[:, yy[i], xx[i]]) > 0]
                yy = tuple(yp)
                xx = tuple(xp)
            matrix_segm = array[:, yy, xx]  # shape [nframes x npx_segment]
            matrix_segm = matrix_scaling(matrix_segm, algo_params.scaling)
            if algo_params.cube_ref is not None:
                matrix_segm_ref = algo_params.cube_ref[:, yy, xx]
                matrix_segm_ref = matrix_scaling(
                    matrix_segm_ref, algo_params.scaling)
            else:
                matrix_segm_ref = None
            if algo_params.cube_sig is not None:
                matrix_sig_segm = algo_params.cube_sig[:, yy, xx]
            else:
                matrix_sig_segm = None

            add_params = {
                "matrix": matrix_segm,
                "frame": iterable(range(n)),
                "pa_threshold": pa_thr,
                "ann_center": ann_center,
                "ncomp": ncompann,
                "matrix_ref": matrix_segm_ref,
                "matrix_sig_segm": matrix_sig_segm,
            }
            func_params = setup_parameters(
                params_obj=algo_params,
                fkt=do_nmf_patch,
                as_list=True,
                show_params=False,
                **add_params,
            )
            res = pool_map(
                algo_params.nproc,
                do_nmf_patch,
                *func_params,
            )

            res = np.array(res, dtype=object)
            residuals = np.array(res[:, 0])
            # ncomps = res[:, 1]
            # nfrslib = res[:, 2]
            recon = np.array(res[:, 1])
            H = np.array(res[:, 2])
            for fr in range(n):
                cube_out[fr][yy, xx] = residuals[fr]
                cube_recon[fr][yy, xx] = recon[fr]
            for pp in range(algo_params.ncomp):
                H_comps[pp][yy, xx] = H[0][pp]  # just save H inferred for fr=0

        if algo_params.verbose:
            timing(start_time)

    # Cube is derotated according to the parallactic angle and collapsed
    cube_der = cube_derotate(
        cube_out, algo_params.angle_list, nproc=algo_params.nproc, **rot_options
    )
    frame = cube_collapse(
        cube_der, mode=algo_params.collapse, w=algo_params.weights)
    if algo_params.verbose:
        print("Done derotating and combining.")
        timing(start_time)
    if algo_params.full_output:
        return cube_out, cube_der, cube_recon, H_comps, frame
    else:
        return frame


def do_nmf_patch(
    matrix,
    frame,
    angle_list,
    fwhm,
    pa_threshold,
    ann_center,
    ncomp,
    max_iter,
    random_state,
    init_svd,
    min_frames_lib,
    max_frames_lib,
    matrix_ref,
    matrix_sig_segm,
    handle_neg,
    **kwargs,
):
    """Solves the NMF for each frame patch (small matrix). For each frame we
    find the frames to be rejected depending on the amount of rotation. The
    library is also truncated on the other end (frames too far or which have
    rotated more) which are more decorrelated to keep the computational cost
    lower. This truncation is done on the annuli after 10*FWHM and the goal is
    to keep min(num_frames/2, 200) in the library.
    """

    # Note: blocks below allow the possibility of ARDI
    if pa_threshold != 0:
        indices_left = _find_indices_adi(
            angle_list, frame, pa_threshold, truncate=True, max_frames=max_frames_lib
        )
        msg = "Too few frames left in the PCA library. "
        msg += "Accepted indices length ({:.0f}) less than {:.0f}. "
        msg += "Try decreasing either delta_rot or min_frames_lib."
        try:
            if matrix_sig_segm is not None:
                data_ref = matrix[indices_left] - matrix_sig_segm[indices_left]
            else:
                data_ref = matrix[indices_left]
        except IndexError:
            if matrix_ref is None:
                raise RuntimeError(msg.format(0, min_frames_lib))
            data_ref = None

        if data_ref.shape[0] < min_frames_lib and matrix_ref is None:
            raise RuntimeError(msg.format(data_ref.shape[0], min_frames_lib))
    elif pa_threshold == 0:
        if matrix_sig_segm is not None:
            data_ref = matrix - matrix_sig_segm
        else:
            data_ref = matrix
    if matrix_ref is not None:
        if data_ref is not None:
            data_ref = np.vstack((matrix_ref, data_ref))
        else:
            data_ref = matrix_ref

    # to avoid bug, just consider positive values
    if np.median(data_ref) < 0:
        raise ValueError("Mostly negative values in the cube")
    else:
        # how to handle negative values
        if handle_neg == "null":
            data_ref[np.where(data_ref < 0)] = 0
        elif handle_neg == "subtr_min":
            data_ref -= np.amin(data_ref)
        else:  # 'mask'
            zp = np.nonzero(np.amin(data_ref, axis=0) > 0)

    solver = "mu"
    # if init_svd == 'nndsvda':
    #     solver = 'mu'
    # else:
    #     solver = 'cd'
    mod = NMF(
        n_components=ncomp,
        solver=solver,
        init=init_svd,
        max_iter=max_iter,
        random_state=random_state,
        **kwargs,
    )

    curr_frame = matrix[frame]  # current frame
    if matrix_sig_segm is not None:
        curr_frame_emp = matrix[frame] - matrix_sig_segm[frame]
    else:
        curr_frame_emp = curr_frame.copy()
    # how to handle negative values
    if handle_neg == "null":
        curr_frame_emp[np.where(curr_frame_emp < 0)] = 0
    elif handle_neg == "subtr_min":
        curr_frame_emp -= np.amin(curr_frame_emp)
    else:  # 'mask'
        zzp = np.nonzero(curr_frame_emp > 0)
        pos_p = np.intersect1d(zp[0], zzp[0])
        curr_frame_emp = curr_frame_emp[pos_p]
        data_ref = data_ref[:, pos_p]

    H = mod.fit(data_ref).components_
    W = mod.transform(curr_frame_emp[np.newaxis, ...])
    reconstructed = np.dot(W, H)
    # if masked neg values, reshape
    if handle_neg == "mask":  # 'mask'
        recon = np.zeros(matrix.shape[1])
        recon[pos_p] = reconstructed
        reconstructed = recon.copy()
        H_tmp = np.zeros([ncomp, matrix.shape[1]])
        for pp in range(ncomp):
            H_tmp[pp, pos_p] = H[pp]
        H = H_tmp.copy()
    residuals = curr_frame - reconstructed
    return residuals, reconstructed, H
    # return residuals, V.shape[0], data_ref.shape[0]
