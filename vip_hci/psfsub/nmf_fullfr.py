#! /usr/bin/env python
"""
Module with PSF reference approximation using Non-negative matrix factorization
for ADI and RDI data, in full frames.

.. [LEE99]
   | Lee & Seung 1999
   | **Learning the parts of objects by non-negative matrix factorization**
   | *Nature, Volume 401, Issue 6755, pp. 788-791*
   | `https://ui.adsabs.harvard.edu/abs/1999Natur.401..788L
     <https://ui.adsabs.harvard.edu/abs/1999Natur.401..788L>`_

"""

__author__ = "Thomas Bédrine, Carlos Alberto Gomez Gonzalez, Valentin Christiaens"
__all__ = ["nmf", "NMF_Params"]

import numpy as np
from sklearn.decomposition import NMF
from dataclasses import dataclass, field
from enum import Enum
from typing import Tuple, List
from ..config.utils_param import setup_parameters, separate_kwargs_dict
from ..config.paramenum import Collapse, HandleNeg, Initsvd, ALGO_KEY
from ..config import timing, time_ini
from ..preproc import cube_derotate, cube_collapse
from ..preproc.derotation import _compute_pa_thresh, _find_indices_adi
from ..var import (prepare_matrix, reshape_matrix, frame_center, dist,
                   matrix_scaling, mask_circle)


@dataclass
class NMF_Params:
    """
    Set of parameters for the NMF full-frame algorithm.

    See function `nmf` below for the documentation.
    """

    cube: np.ndarray = None
    angle_list: np.ndarray = None
    cube_ref: np.ndarray = None
    ncomp: int = 1
    scaling: Enum = None
    max_iter: int = 10000
    random_state: int = None
    mask_center_px: int = None
    source_xy: Tuple[int] = None
    delta_rot: float = 1
    fwhm: float = 4
    init_svd: Enum = Initsvd.NNDSVD
    collapse: Enum = Collapse.MEDIAN
    full_output: bool = False
    verbose: bool = True
    cube_sig: np.ndarray = None
    handle_neg: Enum = HandleNeg.MASK
    nmf_args: dict = field(default_factory=lambda: {})


def nmf(*all_args: List, **all_kwargs: dict):
    """Non Negative Matrix Factorization [LEE99]_ for ADI sequences [GOM17]_.
    Alternative to the full-frame ADI-PCA processing that does not rely on SVD
    or ED for obtaining a low-rank approximation of the datacube. This function
    embeds the scikit-learn NMF algorithm solved through either the coordinate
    descent or the multiplicative update method.

    Parameters
    ----------
    all_args: list, optional
        Positionnal arguments for the NMF algorithm. Full list of parameters
        below.
    all_kwargs: dictionary, optional
        Mix of keyword arguments that can initialize a NMFParams and the optional
        'rot_options' dictionnary, with keyword values for "border_mode", "mask_val",
        "edge_blend", "interp_zeros", "ker" (see documentation of
        ``vip_hci.preproc.frame_rotate``). Can also contain a NMFParams named as
        `algo_params`.

    NMF parameters
    ----------
    cube : numpy ndarray, 3d
        Input cube.
    angle_list : 1d numpy ndarray
        Vector of derotation angles to align North up in your cube images.
    cube_ref : numpy ndarray, 3d, optional
        Reference library cube. For Reference Star Differential Imaging.
    ncomp : int, optional
        How many components are used as for low-rank approximation of the
        datacube.
    scaling : Enum, see `vip_hci.config.paramenum.Scaling`
        With None, no scaling is performed on the input data before SVD. With
        "temp-mean" then temporal px-wise mean subtraction is done, with
        "spat-mean" then the spatial mean is subtracted, with "temp-standard"
        temporal mean centering plus scaling to unit variance is done and with
        "spat-standard" spatial mean centering plus scaling to unit variance is
        performed.
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
    delta_rot : float, optional
        Factor for tunning the parallactic angle threshold, expressed in FWHM.
        Default is 1 (excludes 1xFWHM on each side of the considered frame).
    fwhm : float, optional
        Known size of the FWHM in pixels to be used. Default value is 4.
    init_svd: Enum, see `vip_hci.config.paramenum.Initsvd`
        Method used to initialize the iterative procedure to find H and W.
        'nndsvd': non-negative double SVD recommended for sparseness
        'nndsvda': NNDSVD where zeros are filled with the average of cube;
        recommended when sparsity is not desired
        'random': random initial non-negative matrix
    collapse : Enum, see `vip_hci.config.paramenum.Collapse`
        Sets the way of collapsing the frames for producing a final image.
    full_output: boolean, optional
        Whether to return the final median combined image only or with other
        intermediate arrays.
    verbose : {True, False}, bool optional
        If True prints intermediate info and timing.
    handle_neg: Enum, see `vip_hci.config.paramenum.HandleNeg`
        Determines how to handle negative values: mask them, set them to zero,
        or subtract the minimum value in the arrays. Note: 'mask' or 'null'
        may leave significant artefacts after derotation of residual cube
        => those options should be used carefully (e.g. with proper treatment
        of masked values in non-derotated cube of residuals).
    nmf_args : dictionary, optional
        Additional arguments for scikit-learn NMF algorithm. See:
        https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html

    Returns
    -------
    If full_output is False the final frame is returned. If True the algorithm
    returns the reshaped NMF components, the reconstructed cube, the residuals,
    the residuals derotated and the final frame.

    """
    # Separating the parameters of the ParamsObject from the optionnal rot_options
    class_params, rot_options = separate_kwargs_dict(
        initial_kwargs=all_kwargs, parent_class=NMF_Params
    )

    # Extracting the object of parameters (if any)
    algo_params = None
    if ALGO_KEY in rot_options.keys():
        algo_params = rot_options[ALGO_KEY]
        del rot_options[ALGO_KEY]

    if algo_params is None:
        algo_params = NMF_Params(*all_args, **class_params)

    array = algo_params.cube.copy()
    if algo_params.verbose:
        start_time = time_ini()
    n, y, x = array.shape

    matrix_ref = None
    matrix_sig = None

    # how to handle negative values
    if algo_params.handle_neg == HandleNeg.MASK:
        if algo_params.mask_center_px:
            array = mask_circle(array, algo_params.mask_center_px)
        if algo_params.cube_sig is not None:
            yy, xx = np.where(
                np.amin(array - np.abs(algo_params.cube_sig), axis=0) > 0)
        else:
            yy, xx = np.where(np.amin(array, axis=0) > 0)
        H_tmp = np.zeros([algo_params.ncomp, y, x])
        if len(yy) > 0:
            matrix = array[:, yy, xx]
            matrix = matrix_scaling(matrix, algo_params.scaling)
            if algo_params.cube_ref is not None:
                matrix_ref = algo_params.cube_ref[:, yy, xx]
                matrix_ref = matrix_scaling(matrix_ref, algo_params.scaling)
            if algo_params.cube_sig is not None:
                matrix_sig = algo_params.cube_sig[:, yy, xx]
        else:
            raise ValueError("Remove frame(s) with negative values")
    else:
        if algo_params.handle_neg == HandleNeg.NULL:
            if algo_params.cube_sig is not None:
                array[np.where(array - algo_params.cube_sig < 0)] = 0
                algo_params.cube_sig[np.where(
                    array - algo_params.cube_sig < 0)] = 0
            else:
                array[np.where(array < 0)] = 0

        elif algo_params.handle_neg == HandleNeg.SUBTR_MIN:
            if algo_params.cube_sig is not None:
                array -= np.amin(array - algo_params.cube_sig)
            else:
                array -= np.amin(array)
        else:
            raise ValueError("Mode to handle neg. pixels not recognized")

        matrix = prepare_matrix(
            array,
            algo_params.scaling,
            algo_params.mask_center_px,
            mode="fullfr",
            verbose=algo_params.verbose,
        )
        if algo_params.cube_ref is not None:
            matrix_ref = prepare_matrix(
                algo_params.cube_ref,
                algo_params.scaling,
                algo_params.mask_center_px,
                mode="fullfr",
                verbose=algo_params.verbose,
            )
        if algo_params.cube_sig is not None:
            matrix_sig = prepare_matrix(
                algo_params.cube_sig,
                algo_params.scaling,
                algo_params.mask_center_px,
                mode="fullfr",
                verbose=algo_params.verbose,
            )

    if algo_params.cube_sig is not None:
        # derotate
        residuals_cube = algo_params.cube_sig.copy()
    else:
        residuals_cube = np.zeros_like(array)

    if algo_params.source_xy is None:
        add_params = {
            "matrix": matrix,
            "matrix_ref": matrix_ref,
            "matrix_sig": matrix_sig,
        }
        func_params = setup_parameters(
            params_obj=algo_params, fkt=_project_subtract, **add_params
        )
        residuals = _project_subtract(**func_params, **algo_params.nmf_args)
        if algo_params.verbose:
            timing(start_time)
        if algo_params.full_output:
            # reconstructed = residuals[1]
            H = residuals[2]
            reconstructed = residuals[1]
            residuals = residuals[0]
        recon_cube = residuals_cube.copy()
        if algo_params.handle_neg == "mask":
            for fr in range(n):
                residuals_cube[fr][yy, xx] = residuals[fr]
            if algo_params.full_output:
                for fr in range(n):
                    recon_cube[fr][yy, xx] = reconstructed[fr]
                for pp in range(algo_params.ncomp):
                    H_tmp[pp][yy, xx] = H[pp]
                H = H_tmp
        else:
            for fr in range(n):
                residuals_cube[fr] = residuals[fr].reshape((y, x))
            if algo_params.full_output:
                recon_cube = reshape_matrix(reconstructed, y, x)
                H = H.reshape(algo_params.ncomp, y, x)
    else:
        if algo_params.delta_rot is None or algo_params.fwhm is None:
            msg = "Delta_rot or fwhm parameters missing. Needed for the"
            msg += "PA-based rejection of frames from the library"
            raise TypeError(msg)
        recon_cube = np.zeros_like(algo_params.cube)
        yc, xc = frame_center(algo_params.cube[0], False)
        x1, y1 = algo_params.source_xy
        ann_center = dist(yc, xc, y1, x1)
        pa_thr = _compute_pa_thresh(
            ann_center, algo_params.fwhm, algo_params.delta_rot)
        mid_range = (
            np.abs(np.amax(algo_params.angle_list) -
                   np.amin(algo_params.angle_list))
            / 2
        )
        if pa_thr >= mid_range - mid_range * 0.1:
            new_pa_th = float(mid_range - mid_range * 0.1)
            if algo_params.verbose:
                msg = "PA threshold {:.2f} is too big, will be set to "
                msg += "{:.2f}"
                print(msg.format(pa_thr, new_pa_th))
            pa_thr = new_pa_th

        for fr in range(n):
            ind = _find_indices_adi(algo_params.angle_list, fr, pa_thr)
            add_params = {
                "matrix": matrix,
                "matrix_ref": matrix_ref,
                "matrix_sig": matrix_sig,
                "indices": ind,
                "frame": fr,
            }
            func_params = setup_parameters(
                params_obj=algo_params, fkt=_project_subtract, **add_params
            )
            res_result = _project_subtract(
                **func_params, **algo_params.nmf_args)
            # ! Instead of reshaping, fill frame using get_annulus?
            if algo_params.full_output:
                residuals = res_result[0]
                recon_frame = res_result[1]
                H = res_result[2]
                if algo_params.handle_neg == "mask":
                    recon_cube[fr][yy, xx] = recon_frame
                else:
                    recon_cube[fr] = recon_frame.reshape((y, x))
            else:
                residuals = res_result
            if algo_params.handle_neg == "mask":
                residuals_cube[fr][yy, xx] = residuals
                if fr == n - 1 and algo_params.full_output:
                    for pp in range(algo_params.ncomp):
                        H_tmp[pp][yy, xx] = H[pp]
                    H = H_tmp
            else:
                residuals_cube[fr] = residuals.reshape((y, x))
                if fr == n - 1 and algo_params.full_output:
                    H = H.reshape(algo_params.ncomp, y, x)

    if algo_params.verbose:
        print("Done NMF with sklearn.NMF.")
        timing(start_time)

    residuals_cube_ = cube_derotate(
        residuals_cube, algo_params.angle_list, **rot_options
    )
    frame = cube_collapse(residuals_cube_, mode=algo_params.collapse)

    if algo_params.verbose:
        print("Done derotating and combining.")
        timing(start_time)
    if algo_params.full_output:
        return (H, recon_cube, residuals_cube, residuals_cube_, frame)
    else:
        return frame


def _project_subtract(
    matrix,
    matrix_ref,
    ncomp,
    scaling,
    mask_center_px,
    verbose,
    full_output,
    indices=None,
    frame=None,
    matrix_sig=None,
    max_iter=100,
    random_state=None,
    init_svd="nndsvda",
    **kwargs
):
    """
    PCA projection and model PSF subtraction. Used as a helping function by
    each of the PCA modes (ADI, ADI+RDI, ADI+mSDI).

    Parameters
    ----------
    cube : numpy ndarray
        Input cube.
    cube_ref : numpy ndarray
        Reference cube.
    ncomp : int
        Number of principal components.
    scaling : str
        Scaling of pixel values. See ``pca`` docstrings.
    mask_center_px : int
        Masking out a centered circular aperture.
    verbose : bool
        Verbosity.
    full_output : bool
        Whether to return intermediate arrays or not.
    indices : list
        Indices to be used to discard frames (a rotation threshold is used).
    frame : int
        Index of the current frame (when indices is a list and a rotation
        threshold was applied).
    cube_sig: numpy ndarray, opt
        Cube with estimate of significant authentic signals. If provided, this
        will be subtracted from both the cube and the PCA library, before
        projecting the cube onto the principal components.

    Returns
    -------
    ref_lib_shape : int
        [indices is not None, frame is not None] Number of
        rows in the reference library for the given frame.
    residuals: numpy ndarray
        Residuals, returned in every case.
    reconstructed : numpy ndarray
        [full_output=True] The reconstructed array.
    V : numpy ndarray
        [full_output=True, indices is None, frame is None] The right singular
        vectors of the input matrix, as returned by ``svd/svd_wrapper()``
    """
    if matrix_sig is None:
        matrix_emp = matrix.copy()
    else:
        matrix_emp = matrix - matrix_sig

    if matrix_ref is not None:
        ref_lib = matrix_ref
    elif indices is not None and frame is not None:
        ref_lib = matrix_emp[indices].copy()
    else:
        ref_lib = matrix_emp.copy()

    # to avoid bug, just consider positive values
    if np.median(ref_lib) < 0:
        raise ValueError("Mostly negative values in the cube")
    else:
        ref_lib[np.where(ref_lib < 0)] = 0

    solver = "mu"
    # if init_svd != 'nndsvd':
    #     solver = 'mu'
    # else:
    #     solver = 'cd'
    mod = NMF(
        n_components=ncomp,
        solver=solver,
        init=init_svd,
        max_iter=max_iter,
        random_state=random_state,
        **kwargs
    )
    # a rotation threshold is used (frames are processed one by one)
    if indices is not None and frame is not None:
        if ref_lib.shape[0] <= 10:
            raise RuntimeError(
                "Less than 10 frames left in the PCA library"
                ", Try decreasing the parameter delta_rot"
            )
        curr_frame = matrix[frame]  # current frame
        curr_frame_emp = matrix_emp[frame]
        # H [ncomp, n_pixels]
        H = mod.fit(ref_lib).components_
        # W: coefficients [1, ncomp]
        W = mod.transform(curr_frame_emp[np.newaxis, ...])
        # V = svd_wrapper(ref_lib, svd_mode, ncomp, False)
        # transformed = np.dot(curr_frame_emp, V.T)
        # reconstructed = np.dot(transformed.T, V)
        reconstructed = np.dot(W, H)
        residuals = curr_frame - reconstructed
        if full_output:
            return residuals, reconstructed, H
        else:
            return residuals

    # the whole matrix is processed at once
    else:
        # H [ncomp, n_pixels]: Non-negative components of the data
        # if cube_ref is not None:
        H = mod.fit(ref_lib).components_
        # else:
        #    H = mod.fit(matrix).components_

        # W: coefficients [n_frames, ncomp]
        W = mod.transform(matrix_emp)

        reconstructed = np.dot(W, H)
        residuals = matrix - reconstructed

        # V = svd_wrapper(ref_lib, svd_mode, ncomp, verbose)
        # transformed = np.dot(V, matrix_emp.T)
        # reconstructed = np.dot(transformed.T, V)
        # residuals = matrix - reconstructed
        # residuals_res = reshape_matrix(residuals, y, x)
        if full_output:
            return residuals, reconstructed, H
        else:
            return residuals
