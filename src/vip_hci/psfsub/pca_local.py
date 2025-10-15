#! /usr/bin/env python
"""
Module with local/smart PCA (annulus or patch-wise in a multi-processing
fashion) model PSF subtraction for ADI, ADI+SDI (IFS) and ADI+RDI datasets.

.. [ABS13]
   | Absil et al. 2013
   | **Searching for companions down to 2 AU from beta Pictoris using the
     L'-band AGPM coronagraph on VLT/NACO**
   | *Astronomy & Astrophysics, Volume 559, Issue 1, p. 12*
   | `https://arxiv.org/abs/1311.4298
     <https://arxiv.org/abs/1311.4298>`_

"""

__author__ = "C. A. Gomez Gonzalez, V. Christiaens, T. Bédrine"
__all__ = ["pca_annular", "PCA_ANNULAR_Params"]

import numpy as np
from multiprocessing import cpu_count
from typing import Tuple, List, Union
from enum import Enum
from dataclasses import dataclass
from .svd import get_eigenvectors
from ..preproc import (cube_derotate, cube_collapse, check_pa_vector,
                       check_scal_vector)
from ..preproc import cube_rescaling_wavelengths as scwave
from ..preproc.derotation import _find_indices_adi, _define_annuli
from ..preproc.rescaling import _find_indices_sdi
from ..config import time_ini, timing
from ..config.paramenum import SvdMode, Imlib, Interpolation, Collapse, ALGO_KEY
from ..config.utils_conf import pool_map, iterable
from ..config.utils_param import setup_parameters, separate_kwargs_dict
from ..stats import descriptive_stats
from ..var import get_annulus_segments, matrix_scaling
AUTO = "auto"


@dataclass
class PCA_ANNULAR_Params:
    """Set of parameters for the annular PCA module."""

    cube: np.ndarray = None
    angle_list: np.ndarray = None
    cube_ref: np.ndarray = None
    scale_list: np.ndarray = None
    radius_int: int = 0
    fwhm: float = 4
    asize: float = 4
    n_segments: Union[int, List[int], AUTO] = 1
    delta_rot: Union[float, Tuple[float], List[float]] = (0.1, 1)
    delta_sep: Union[float, Tuple[float], List[float]] = (0.1, 1)
    ncomp: Union[int, Tuple, np.ndarray, AUTO] = 1
    svd_mode: Enum = SvdMode.LAPACK
    nproc: int = 1
    min_frames_lib: int = 2
    max_frames_lib: int = 200
    tol: float = 1e-1
    scaling: Enum = None
    imlib: Enum = Imlib.VIPFFT
    interpolation: Enum = Interpolation.LANCZOS4
    collapse: Enum = Collapse.MEDIAN
    collapse_ifs: Enum = Collapse.MEAN
    ifs_collapse_range: Union["all", Tuple[int]] = "all"
    theta_init: int = 0
    weights: np.ndarray = None
    cube_sig: np.ndarray = None
    full_output: bool = False
    verbose: bool = True
    left_eigv: bool = False


def pca_annular(*all_args: List, **all_kwargs: dict):
    """PCA model PSF subtraction for ADI, ADI+RDI or ADI+mSDI (IFS) data.

    The PCA model is computed locally in each annulus (or annular sectors
    according to ``n_segments``). For each sector we discard reference frames
    taking into account a parallactic angle threshold (``delta_rot``) and
    optionally a radial movement threshold (``delta_sep``) for 4d cubes.

    For ADI+RDI data, it computes the principal components from the reference
    library/cube, forcing pixel-wise temporal standardization. The number of
    principal components can be automatically adjusted by the algorithm by
    minimizing the residuals inside each patch/region.

    References: [AMA12]_ for PCA-ADI; [ABS13]_ for PCA-ADI in concentric annuli
    considering a parallactic angle threshold; [CHR19]_ for PCA-ASDI and
    PCA-SADI in one or two steps.

    Parameters
    ----------
    all_args: list, optional
        Positionnal arguments for the PCA annular algorithm. Full list of
        parameters below.
    all_kwargs: dictionary, optional
        Mix of keyword arguments that can initialize a PCA_ANNULAR_Params and
        the optional 'rot_options' dictionary, with keyword values for
        "border_mode", "mask_val", "edge_blend", "interp_zeros", "ker" (see
        documentation of ``vip_hci.preproc.frame_rotate``). Can also contain a
        PCA_ANNULAR_Params named as `algo_params`.

    PCA annular parameters
    ----------
    cube : numpy ndarray, 3d or 4d
        Input cube.
    angle_list : 1d numpy ndarray
        Vector of derotation angles to align North up in your cube images.
    cube_ref : numpy ndarray, 3d, optional
        Reference library cube. For Reference Star Differential Imaging.
    scale_list : numpy ndarray, 1d, optional
        If provided, triggers mSDI reduction. These should be the scaling
        factors used to re-scale the spectral channels and align the speckles
        in case of IFS data (ADI+mSDI cube). Usually, these can be approximated
        by the last channel wavelength divided by the other wavelengths in the
        cube (more thorough approaches can be used to get the scaling factors,
        e.g. with ``vip_hci.preproc.find_scal_vector``).
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
    delta_rot : float, tuple of floats or list of floats, optional
        Parallactic angle threshold, expressed in FWHM, used to build the PCA
        library. If a tuple of 2 floats is provided, they are used as the lower
        and upper bounds of a linearly increasing threshold as a function of
        separation. If a list is provided, it will correspond to the threshold
        to be adopted for each annulus (length should match number of annuli).
        Default is (0.1, 1), which excludes 0.1 FWHM for the innermost annulus
        up to 1 FWHM for the outermost annulus.
    delta_sep : float, tuple of floats or list of floats, optional
        The radial threshold in terms of the mean FWHM, used to build the PCA
        library (for ADI+mSDI data). If a tuple of 2 floats is provided, they
        are used as the lower and upper bounds of a linearly increasing
        threshold as a function of separation. If a list is provided, it will
        correspond to the threshold to be adopted for each annulus (length
        should match number of annuli). Default is (0.1, 1), which excludes 0.1
        FWHM for the innermost annulus up to 1 FWHM for the outermost annulus.
    ncomp : 'auto', int, tuple/1d numpy array of int, list, tuple of lists, opt
        How many PCs are used as a lower-dimensional subspace to project the
        target (sectors of) frames. Depends on the dimensionality of `cube`.

        * ADI and ADI+RDI (``cube`` is a 3d array): if a single integer is
        provided, then the same number of PCs will be subtracted at each
        separation (annulus). If a tuple is provided, then a different number
        of PCs will be used for each annulus (starting with the innermost
        one). If ``ncomp`` is set to ``auto`` then the number of PCs are
        calculated for each region/patch automatically. If a list of int is
        provided, several npc will be tried at once, but the same value of npc
        will be used for all annuli. If a tuple of lists of int is provided,
        the length of tuple should match the number of annuli and different sets
        of npc will be calculated simultaneously for each annulus, with the
        exact values of npc provided in the respective lists.

        * ADI or ADI+RDI (``cube`` is a 4d array): same input format allowed as
        above, but with a slightly different behaviour if ncomp is a list: if it
        has the same length as the number of channels, each element of the list
        will be used as ``ncomp`` value (whether int, float or tuple) for each
        spectral channel. Otherwise the same behaviour as above is assumed.

        * ADI+mSDI case: ``ncomp`` must be a tuple of two integers or a list of
        tuples of two integers, with the number of PCs obtained from each
        multi-spectral frame (for each sector) and the number of PCs used in the
        second PCA stage (ADI fashion, using the residuals of the first stage).
        If None then the second PCA stage is skipped and the residuals are
        de-rotated and combined.

    svd_mode : Enum, see `vip_hci.config.paramenum.SvdMode`
        Switch for the SVD method/library to be used.
    nproc : None or int, optional
        Number of processes for parallel computing. If None the number of
        processes will be set to (cpu_count()/2).
    min_frames_lib : int, optional
        Minimum number of frames in the PCA reference library.
    max_frames_lib : int, optional
        Maximum number of frames in the PCA reference library. The more
        distant/decorrelated frames are removed from the library.
    tol : float, optional
        Stopping criterion for choosing the number of PCs when ``ncomp``
        is None. Lower values will lead to smaller residuals and more PCs.
    scaling : Enum, see `vip_hci.config.paramenum.Scaling`
        Pixel-wise scaling mode using ``sklearn.preprocessing.scale``
        function. If set to None, the input matrix is left untouched.
    imlib : Enum, see `vip_hci.config.paramenum.Imlib`
        See the documentation of the ``vip_hci.preproc.frame_rotate`` function.
    interpolation :  Enum, see `vip_hci.config.paramenum.Interpolation`
        See the documentation of the ``vip_hci.preproc.frame_rotate`` function.
    collapse : Enum, see `vip_hci.config.paramenum.Collapse`
        Sets the way of collapsing the frames for producing a final image.
    collapse_ifs : Enum, see `vip_hci.config.paramenum.Collapse`
        Sets how spectral residual frames should be combined to produce an
        mSDI image.
    ifs_collapse_range: str 'all' or tuple of 2 int
        If a tuple, it should contain the first and last channels where the mSDI
        residual channels will be collapsed (by default collapses all channels).
    full_output: boolean, optional
        Whether to return the final median combined image only or with other
        intermediate arrays.
    verbose : bool, optional
        If True prints to stdout intermediate info.
    theta_init : int
        Initial azimuth [degrees] of the first segment, counting from the
        positive x-axis counterclockwise (irrelevant if n_segments=1).
    weights: 1d numpy array or list, optional
        Weights to be applied for a weighted mean. Need to be provided if
        collapse mode is 'wmean'.
    cube_sig: numpy ndarray, opt
        Cube with estimate of significant authentic signals. If provided, this
        will be subtracted before projecting cube onto reference cube.

    Returns
    -------
    frame : numpy ndarray, 2d
        [full_output=False] Median combination of the de-rotated cube.
    array_out : numpy ndarray, 3d or 4d
        [full_output=True] Cube of residuals.
    array_der : numpy ndarray, 3d or 4d
        [full_output=True] Cube residuals after de-rotation.
    frame : numpy ndarray, 2d
        [full_output=True] Median combination of the de-rotated cube.
    """
    # Separate parameters of the ParamsObject from the optionnal rot_options
    class_params, rot_options = separate_kwargs_dict(all_kwargs,
                                                     PCA_ANNULAR_Params
                                                     )

    # Extracting the object of parameters (if any)
    algo_params = None
    if ALGO_KEY in rot_options.keys():
        algo_params = rot_options[ALGO_KEY]
        del rot_options[ALGO_KEY]

    if algo_params is None:
        algo_params = PCA_ANNULAR_Params(*all_args, **class_params)

    # by default, interpolate masked area before derotation if a mask is used
    if algo_params.radius_int and len(rot_options) == 0:
        rot_options['mask_val'] = 0
        rot_options['ker'] = 1
        rot_options['interp_zeros'] = True

    global start_time
    if algo_params.verbose:
        start_time = time_ini()

    if algo_params.left_eigv:
        if (
            (algo_params.cube_ref is not None)
            or (algo_params.cube_sig is not None)
            or (algo_params.ncomp == "auto")
        ):
            raise NotImplementedError(
                "left_eigv is not compatible"
                "with 'cube_ref', 'cube_sig', ncomp='auto'"
            )

    # ADI or ADI+RDI data
    if algo_params.cube.ndim == 3:
        if algo_params.verbose:
            add_params = {"start_time": start_time, "full_output": True}
        else:
            add_params = {"full_output": True}

        func_params = setup_parameters(
            params_obj=algo_params, fkt=_pca_adi_rdi, **add_params
        )
        res = _pca_adi_rdi(**func_params, **rot_options)

        cube_out, cube_der, frame = res
        if algo_params.full_output:
            return cube_out, cube_der, frame
        else:
            return frame

    # 4D cube, but no mSDI desired
    elif algo_params.cube.ndim == 4 and algo_params.scale_list is None:
        nch, nz, ny, nx = algo_params.cube.shape
        ifs_adi_frames = np.zeros([nch, ny, nx])
        if not isinstance(algo_params.ncomp, list):
            algo_params.ncomp = [algo_params.ncomp] * nch
        elif len(algo_params.ncomp) != nch:
            algo_params.ncomp = [algo_params.ncomp] * nch
        if np.isscalar(algo_params.fwhm):
            algo_params.fwhm = [algo_params.fwhm] * nch

        cube_out = []
        cube_der = []
        # ADI or RDI in each channel
        for ch in range(nch):
            if algo_params.cube_ref is not None:
                if algo_params.cube_ref[ch].ndim != 3:
                    msg = "Ref cube has wrong format for 4d input cube"
                    raise TypeError(msg)
                cube_ref_tmp = algo_params.cube_ref[ch]
            else:
                cube_ref_tmp = algo_params.cube_ref

            add_params = {
                "cube": algo_params.cube[ch],
                "fwhm": algo_params.fwhm[ch],
                "ncomp": algo_params.ncomp[ch],
                "full_output": True,
                "cube_ref": cube_ref_tmp,
            }

            func_params = setup_parameters(
                params_obj=algo_params, fkt=_pca_adi_rdi, **add_params
            )
            res_pca = _pca_adi_rdi(**func_params, **rot_options)
            cube_out.append(res_pca[0])
            cube_der.append(res_pca[1])
            ifs_adi_frames[ch] = res_pca[-1]

        if algo_params.collapse_ifs is not None:
            frame = cube_collapse(ifs_adi_frames, mode=algo_params.collapse_ifs)
        else:
            frame = ifs_adi_frames

        # convert to numpy arrays
        cube_out = np.array(cube_out)
        cube_der = np.array(cube_der)
        if algo_params.full_output:
            return cube_out, cube_der, frame
        else:
            return frame

    # ADI+mSDI (IFS) datacubes
    elif algo_params.cube.ndim == 4:
        global ARRAY
        ARRAY = algo_params.cube

        if algo_params.cube_ref is not None:
            global ARRAY_REF
            ARRAY_REF = algo_params.cube_ref

        z, n, y_in, x_in = algo_params.cube.shape
        algo_params.fwhm = int(np.round(np.mean(algo_params.fwhm)))
        n_annuli = int((y_in / 2 - algo_params.radius_int) / algo_params.asize)

        if np.array(algo_params.scale_list).ndim > 1:
            raise ValueError("Scaling factors vector is not 1d")
        if not algo_params.scale_list.shape[0] == z:
            raise ValueError("Scaling factors vector has wrong length")

        if not isinstance(algo_params.ncomp, tuple):
            msg = "`ncomp` must be a tuple of two integers when "
            msg += "`cube` is a 4d array"
            raise TypeError(msg)
        else:
            ncomp2 = algo_params.ncomp[1]
            algo_params.ncomp = algo_params.ncomp[0]

        if algo_params.verbose:
            print("First PCA subtraction exploiting the spectral variability")
            print("{} spectral channels per IFS frame".format(z))
            print(
                "N annuli = {}, mean FWHM = {:.3f}".format(
                    n_annuli, algo_params.fwhm)
            )

        add_params = {
            "fr": iterable(range(n)),
            "scal": algo_params.scale_list,
            "collapse": algo_params.collapse_ifs,
        }

        func_params = setup_parameters(
            params_obj=algo_params, fkt=_pca_sdi_fr, as_list=True, **add_params
        )
        res = pool_map(
            algo_params.nproc,
            _pca_sdi_fr,
            verbose=algo_params.verbose,
            *func_params,
        )
        residuals_cube_channels = np.array(res)

        # Exploiting rotational variability
        if algo_params.verbose:
            timing(start_time)
            print("{} ADI frames".format(n))

        if ncomp2 is None:
            if algo_params.verbose:
                print("Skipping the second PCA subtraction")

            cube_out = residuals_cube_channels
            cube_der = cube_derotate(
                cube_out,
                algo_params.angle_list,
                nproc=algo_params.nproc,
                imlib=algo_params.imlib,
                interpolation=algo_params.interpolation,
                **rot_options,
            )
            frame = cube_collapse(
                cube_der, mode=algo_params.collapse, w=algo_params.weights
            )

        else:
            if algo_params.cube_ref is not None:
                if algo_params.verbose:
                    print("First PCA subtraction (spectral) on REF cube")
                    print("{} spectral channels per IFS frame".format(z))
                    print(
                        "N annuli = {}, mean FWHM = {:.3f}".format(
                            n_annuli, algo_params.fwhm)
                    )
                # apply the same first pass to cube ref
                nr = algo_params.cube_ref.shape[0]
                add_params['do_ref'] = True
                add_params["fr"] = iterable(range(nr))
                func_params_ref = setup_parameters(
                    params_obj=algo_params, fkt=_pca_sdi_fr, as_list=True,
                    **add_params
                )
                res = pool_map(
                    algo_params.nproc,
                    _pca_sdi_fr,
                    verbose=algo_params.verbose,
                    *func_params_ref,
                )
                residuals_cube_channels_ref = np.array(res)

                # Exploiting rotational variability
                if algo_params.verbose:
                    timing(start_time)
                    print("{} REF frames".format(n))
            else:
                residuals_cube_channels_ref = None

            if algo_params.verbose:
                print("Second PCA subtraction exploiting angular variability")

            add_params = {
                "cube": residuals_cube_channels,
                "ncomp": ncomp2,
                "cube_ref": residuals_cube_channels_ref,
            }

            func_params = setup_parameters(
                params_obj=algo_params, fkt=_pca_adi_rdi, **add_params
            )
            res = _pca_adi_rdi(**func_params, **rot_options)

            if algo_params.full_output:
                cube_out, cube_der, frame = res
            else:
                frame = res

        if algo_params.full_output:
            return cube_out, cube_der, frame
        else:
            return frame

    else:
        raise TypeError("Input array is not a 4d or 3d array")


################################################################################
# Functions encapsulating portions of the main algorithm
################################################################################


def _pca_sdi_fr(
    fr,
    scal,
    radius_int,
    fwhm,
    asize,
    n_segments,
    delta_sep,
    ncomp,
    svd_mode,
    tol,
    scaling,
    imlib,
    interpolation,
    collapse,
    ifs_collapse_range,
    theta_init,
    do_ref=False
):
    """Optimized PCA subtraction on a multi-spectral frame (IFS data)."""
    scale_list = check_scal_vector(scal)

    if do_ref:  # do it on REF cube
        z, n, y_in, x_in = ARRAY_REF.shape
        # rescaled cube, aligning speckles
        multispec_fr = scwave(
            ARRAY_REF[:, fr, :, :], scale_list, imlib=imlib,
            interpolation=interpolation
        )[0]
    else:  # do it on SCI cube
        z, n, y_in, x_in = ARRAY.shape

        # rescaled cube, aligning speckles
        multispec_fr = scwave(
            ARRAY[:, fr, :, :], scale_list, imlib=imlib,
            interpolation=interpolation
        )[0]

    # Exploiting spectral variability (radial movement)
    fwhm = int(np.round(np.mean(fwhm)))
    n_annuli = int((y_in / 2 - radius_int) / asize)

    if isinstance(n_segments, int):
        n_segments = [n_segments for _ in range(n_annuli)]
    elif n_segments == "auto":
        n_segments = list()
        n_segments.append(2)  # for first annulus
        n_segments.append(3)  # for second annulus
        ld = 2 * np.tan(360 / 4 / 2) * asize
        for i in range(2, n_annuli):  # rest of annuli
            radius = i * asize
            ang = np.rad2deg(2 * np.arctan(ld / (2 * radius)))
            n_segments.append(int(np.ceil(360 / ang)))

    cube_res = np.zeros_like(multispec_fr)  # shape (z, resc_y, resc_x)

    if isinstance(delta_sep, (tuple, list)):
        delta_sep_vec = np.linspace(delta_sep[0], delta_sep[1], n_annuli)
    elif np.isscalar(delta_sep):
        delta_sep_vec = [delta_sep] * n_annuli
    else:
        if len(delta_sep) != n_annuli:
            msg = "If delta_sep is a list it should have n_annuli elements."
            raise TypeError(msg)
        delta_sep_vec = delta_sep

    for ann in range(n_annuli):
        if ann == n_annuli - 1:
            inner_radius = radius_int + (ann * asize - 1)
        else:
            inner_radius = radius_int + ann * asize
        ann_center = inner_radius + (asize / 2)

        indices = get_annulus_segments(
            multispec_fr[0], inner_radius, asize, n_segments[ann], theta_init
        )
        # Library matrix is created for each segment and scaled if needed
        for seg in range(n_segments[ann]):
            yy = indices[seg][0]
            xx = indices[seg][1]
            matrix = multispec_fr[:, yy, xx]  # shape (z, npx_annsegm)
            matrix = matrix_scaling(matrix, scaling)

            for j in range(z):
                indices_left = _find_indices_sdi(
                    scal, ann_center, j, fwhm, delta_sep_vec[ann]
                )
                matrix_ref = matrix[indices_left]
                curr_frame = matrix[j]  # current frame
                V = get_eigenvectors(
                    ncomp,
                    matrix_ref,
                    svd_mode,
                    noise_error=tol,
                    debug=False,
                    scaling=scaling,
                )
                transformed = np.dot(curr_frame, V.T)
                reconstructed = np.dot(transformed.T, V)
                residuals = curr_frame - reconstructed
                # return residuals, V.shape[0], matrix_ref.shape[0]
                cube_res[j, yy, xx] = residuals

    if ifs_collapse_range == "all":
        idx_ini = 0
        idx_fin = z
    else:
        idx_ini = ifs_collapse_range[0]
        idx_fin = ifs_collapse_range[1]

    frame_desc = scwave(
        cube_res[idx_ini:idx_fin],
        scale_list[idx_ini:idx_fin],
        full_output=False,
        inverse=True,
        y_in=y_in,
        x_in=x_in,
        imlib=imlib,
        interpolation=interpolation,
        collapse=collapse,
    )
    return frame_desc


def _pca_adi_rdi(
    cube,
    angle_list,
    radius_int=0,
    fwhm=4,
    asize=2,
    n_segments=1,
    delta_rot=1,
    ncomp=1,
    svd_mode="lapack",
    nproc=None,
    min_frames_lib=2,
    max_frames_lib=200,
    tol=1e-1,
    scaling=None,
    imlib="vip-fft",
    interpolation="lanczos4",
    collapse="median",
    full_output=False,
    verbose=1,
    cube_ref=None,
    theta_init=0,
    weights=None,
    cube_sig=None,
    left_eigv=False,
    **rot_options,
):
    """PCA exploiting angular variability (ADI fashion)."""
    array = cube
    if array.ndim != 3:
        raise TypeError("Input array is not a cube or 3d array")
    if array.shape[0] != angle_list.shape[0]:
        raise TypeError("Input vector or parallactic angles has wrong length")

    n, y, x = array.shape

    angle_list = check_pa_vector(angle_list)
    n_annuli = int((y / 2 - radius_int) / asize)

    if isinstance(delta_rot, tuple):
        delta_rot = np.linspace(delta_rot[0], delta_rot[1], num=n_annuli)
    elif np.isscalar(delta_rot):
        delta_rot = [delta_rot] * n_annuli
    else:
        if len(delta_rot) != n_annuli:
            msg = "If delta_rot is a list it should have n_annuli elements."
            raise TypeError(msg)

    if isinstance(n_segments, int):
        n_segments = [n_segments for _ in range(n_annuli)]
    elif n_segments == "auto":
        n_segments = list()
        n_segments.append(2)  # for first annulus
        n_segments.append(3)  # for second annulus
        ld = 2 * np.tan(360 / 4 / 2) * asize
        for i in range(2, n_annuli):  # rest of annuli
            radius = i * asize
            ang = np.rad2deg(2 * np.arctan(ld / (2 * radius)))
            n_segments.append(int(np.ceil(360 / ang)))

    if verbose:
        msg = "N annuli = {}, FWHM = {:.3f}"
        print(msg.format(n_annuli, fwhm))
        print("PCA per annulus (or annular sectors):")

    if nproc is None:  # Hyper-threading "duplicates" the cores -> cpu_count/2
        nproc = cpu_count() // 2

    # The annuli are built, and the corresponding PA thresholds for frame
    # rejection are calculated (at the center of the annulus)
    cube_out = np.zeros_like(array)
    if isinstance(ncomp, list):
        nncomp = len(ncomp)
        cube_out = np.zeros([nncomp, array.shape[0], array.shape[1],
                             array.shape[2]])
    if verbose:
        #  verbosity set to 2 only for ADI
        verbose_ann = int(verbose) + int(cube_ref is None)
    else:
        verbose_ann = verbose

    for ann in range(n_annuli):
        if isinstance(ncomp, tuple) or isinstance(ncomp, np.ndarray):
            if len(ncomp) == n_annuli:
                ncompann = ncomp[ann]
            else:
                msg = "If `ncomp` is a tuple, its length must match the number "
                msg += "of annuli"
                raise TypeError(msg)
        else:
            ncompann = ncomp

        n_segments_ann = n_segments[ann]
        res_ann_par = _define_annuli(
            angle_list,
            ann,
            n_annuli,
            fwhm,
            radius_int,
            asize,
            delta_rot[ann],
            n_segments_ann,
            verbose_ann,
            True,
        )
        pa_thr, inner_radius, ann_center = res_ann_par
        indices = get_annulus_segments(
            array[0], inner_radius, asize, n_segments_ann, theta_init
        )

        if left_eigv:
            indices_out = get_annulus_segments(array[0], inner_radius, asize,
                                               n_segments_ann, theta_init,
                                               out=True)

        # Library matrix is created for each segment and scaled if needed
        for j in range(n_segments_ann):
            yy = indices[j][0]
            xx = indices[j][1]
            matrix_segm = array[:, yy, xx]  # shape [nframes x npx_segment]
            matrix_segm = matrix_scaling(matrix_segm, scaling)
            if cube_ref is not None:
                matrix_segm_ref = cube_ref[:, yy, xx]
                matrix_segm_ref = matrix_scaling(matrix_segm_ref, scaling)
            else:
                matrix_segm_ref = None
            if cube_sig is not None:
                matrix_sig_segm = cube_sig[:, yy, xx]
            else:
                matrix_sig_segm = None

            if not left_eigv:
                res = pool_map(
                    nproc,
                    do_pca_patch,
                    matrix_segm,
                    iterable(range(n)),
                    angle_list,
                    fwhm,
                    pa_thr,
                    ann_center,
                    svd_mode,
                    ncompann,
                    min_frames_lib,
                    max_frames_lib,
                    tol,
                    matrix_segm_ref,
                    matrix_sig_segm,
                )

                if isinstance(ncomp, list):
                    nncomp = len(ncomp)
                    residuals = []
                    for nn in range(nncomp):
                        tmp = np.array([res[i][0][nn] for i in range(n)])
                        residuals.append(tmp)
                else:
                    res = np.array(res, dtype=object)
                    residuals = np.array(res[:, 0])
                    ncomps = res[:, 1]
                    nfrslib = res[:, 2]
            else:
                yy_out = indices_out[j][0]
                xx_out = indices_out[j][1]
                matrix_out_segm = array[
                    :, yy_out, xx_out
                ]  # shape [nframes x npx_out_segment]
                matrix_out_segm = matrix_scaling(matrix_out_segm, scaling)
                if isinstance(ncomp, list):
                    npc = max(ncomp)
                else:
                    npc = ncomp
                V = get_eigenvectors(npc, matrix_out_segm, svd_mode,
                                     noise_error=tol, left_eigv=True)

                if isinstance(ncomp, list):
                    residuals = []
                    for nn, npc_tmp in enumerate(ncomp):
                        transformed = np.dot(V[:npc_tmp], matrix_segm)
                        reconstructed = np.dot(transformed.T, V[:npc_tmp])
                        residuals.append(matrix_segm - reconstructed.T)
                else:
                    transformed = np.dot(V, matrix_segm)
                    reconstructed = np.dot(transformed.T, V)
                    residuals = matrix_segm - reconstructed.T
                    nfrslib = matrix_out_segm.shape[0]

            if isinstance(ncomp, list):
                for nn, npc in enumerate(ncomp):
                    for fr in range(n):
                        cube_out[nn, fr][yy, xx] = residuals[nn][fr]
            else:
                for fr in range(n):
                    cube_out[fr][yy, xx] = residuals[fr]

            # number of frames in library printed for each annular quadrant
            # number of PCs printed for each annular quadrant
            if verbose == 2 and not isinstance(ncomp, list):
                descriptive_stats(nfrslib, verbose=verbose, label="\tLIBsize: ")
                descriptive_stats(ncomps, verbose=verbose, label="\tNum PCs: ")

        if verbose == 1:
            print("Done PCA with {} for current annulus".format(svd_mode))
            timing(start_time)

    if isinstance(ncomp, list):
        cube_der = np.zeros_like(cube_out)
        frame = []
        for nn, npc in enumerate(ncomp):
            cube_der[nn] = cube_derotate(cube_out[nn], angle_list, nproc=nproc,
                                         imlib=imlib,
                                         interpolation=interpolation,
                                         **rot_options)
            frame.append(cube_collapse(cube_der[nn], mode=collapse, w=weights))
    else:
        # Cube is derotated according to the parallactic angle and collapsed
        cube_der = cube_derotate(
            cube_out,
            angle_list,
            nproc=nproc,
            imlib=imlib,
            interpolation=interpolation,
            **rot_options,
        )
        frame = cube_collapse(cube_der, mode=collapse, w=weights)

    if verbose:
        print("Done derotating and combining.")
        timing(start_time)

    if full_output:
        return cube_out, cube_der, frame
    else:
        return frame


def do_pca_patch(
    matrix,
    frame,
    angle_list,
    fwhm,
    pa_threshold,
    ann_center,
    svd_mode,
    ncomp,
    min_frames_lib,
    max_frames_lib,
    tol,
    matrix_ref,
    matrix_sig_segm,
):
    """Do the SVD/PCA for each frame patch (small matrix).

    For each frame, frames to be rejected from the PCA library are found
    depending on the criterion in field rotation. The library is also truncated
    on the other end (frames too far in time, which have rotated more) which are
    more decorrelated, to keep the computational cost lower. This truncation is
    done on the annuli beyong 10*FWHM radius and the goal is to keep
    min(num_frames/2, 200) in the library.

    """
    if pa_threshold != 0:
        indices_left = _find_indices_adi(angle_list, frame, pa_threshold,
                                         truncate=True,
                                         max_frames=max_frames_lib)
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
            raise RuntimeError(msg.format(len(indices_left), min_frames_lib))
    else:
        if matrix_sig_segm is not None:
            data_ref = matrix - matrix_sig_segm
        else:
            data_ref = matrix

    if matrix_ref is not None:
        # Stacking the ref and the target ref (pa thresh) libraries
        if data_ref is not None:
            data_ref = np.vstack((matrix_ref, data_ref))
        else:
            data_ref = matrix_ref

    curr_frame = matrix[frame]  # current frame
    if matrix_sig_segm is not None:
        curr_frame_emp = matrix[frame] - matrix_sig_segm[frame]
    else:
        curr_frame_emp = curr_frame
    if isinstance(ncomp, list):
        npc = max(ncomp)
    else:
        npc = ncomp
    V = get_eigenvectors(npc, data_ref, svd_mode, noise_error=tol)

    if isinstance(ncomp, list):
        residuals = []
        for nn, npc_tmp in enumerate(ncomp):
            transformed = np.dot(curr_frame_emp, V[:npc_tmp].T)
            reconstructed = np.dot(transformed.T, V[:npc_tmp])
            residuals.append(curr_frame - reconstructed)
    else:
        transformed = np.dot(curr_frame_emp, V.T)
        reconstructed = np.dot(transformed.T, V)
        residuals = curr_frame - reconstructed

    return residuals, V.shape[0], data_ref.shape[0]
