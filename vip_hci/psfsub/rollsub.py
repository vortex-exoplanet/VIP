#! /usr/bin/env python
"""
Implementation of a roll subtraction algorithm for PSF subtraction in imaging
sequences obtained with space-based instruments (e.g. JWST or HST) with
different roll angles. The concept was proposed in [SCH03]_ for application to
HST/NICMOS observations.

.. [SCH03]
   | Schneider & Silverstone 2003
   | **NICMOS Coronagraphic Observations of the GM Aurigae Circumstellar Disk**
   | *The Astronomical Journal, Volume 6125, Issue 3, pp. 1467-1479*
   | `https://arxiv.org/abs/astro-ph/0512335
     <https://arxiv.org/abs/astro-ph/0512335>`_

"""

__author__ = "Valentin Christiaens"
__all__ = ["roll_sub", "ROLL_SUB_Params"]

import numpy as np
from multiprocessing import cpu_count
from dataclasses import dataclass
from enum import Enum
from typing import List
from ..config import time_ini, timing
from ..config.paramenum import Imlib, Interpolation, Collapse, ALGO_KEY
from ..config.utils_param import separate_kwargs_dict
from ..preproc import cube_derotate, cube_collapse, frame_rotate
from ..var import mask_circle, frame_filter_lowpass, cube_filter_lowpass


@dataclass
class ROLL_SUB_Params:
    """
    Set of parameters for the roll subtraction module.

    See function `roll_sub` for documentation.
    """

    cube: np.ndarray = None
    angle_list: np.ndarray = None
    mode: str = "mean"
    imlib: Enum = Imlib.VIPFFT
    interpolation: Enum = Interpolation.LANCZOS4
    collapse: Enum = Collapse.MEAN
    fwhm_lp_bef: float = 0.
    fwhm_lp_aft: float = 0.
    mask_rad: float = 0.
    nproc: int = 1
    full_output: bool = False
    verbose: bool = True


def roll_sub(*all_args: List, **all_kwargs: dict):
    """Perform roll-subtraction, followed by derotation and stacking of\
    residual images.

    Reference: [SCH03]_.

    Parameters
    ----------
    all_args: list, optional
        Positionnal arguments for the roll_sub algorithm. Full list of
        parameters below.
    all_kwargs: dictionary, optional
        Mix of keyword arguments that can initialize a MEDIAN_SUB_Params and the
        optional ``rot_options`` dictionary (with keywords ``border_mode``,
        ``mask_val``, ``edge_blend``, ``interp_zeros``, ``ker``; see docstrings
        of ``vip_hci.preproc.frame_rotate``). Can also contain a
        MEDIAN_SUB_Params object/dictionary named ``algo_params``.

    Parameters
    ----------
    cube : 3d numpy ndarray, or tuple of two 3d numpy ndarray
        Input cube. Can also be a 4d array, with images obtained with the 1st
        and 2nd roll angle values are provided in the 1st and 2nd dimension of
        the 4d cube, respectively.
    angle_list : 1d numpy ndarray, or list/tuple of 2 elements
        Roll angles associated to each frame. Can also be a list/tuple of 2
        elements. In the latter case, if input cube is 3D, it will assume that
        the first half of the frames are associated to the first roll angle,
        and the second half to the second roll angle.
    mode : {'mean', 'median', 'individual'}, str optional
        If ``mode`` is set to 'mean' or 'median', only the mean/median frame of
        the image sequence obtained at the first roll angle is subtracted to the
        mean/media image from the sequence obtained with the second roll angle,
        and vice-versa. If ``mode`` is set to 'individual' a pair-wise
        subtraction of individual images obtained at each roll angle is
        performed, following the same order as in the input cube. To work, this
        mode requires the same number of images obtained with each roll angle.
    imlib : Enum, see `vip_hci.config.paramenum.Imlib`
        See the documentation of ``vip_hci.preproc.frame_rotate``.
    interpolation : Enum, see `vip_hci.config.paramenum.Interpolation`
        See the documentation of the ``vip_hci.preproc.frame_rotate`` function.
    collapse : Enum, see `vip_hci.config.paramenum.Collapse`
        Sets how temporal residual frames should be combined to produce an
        ADI image.
    fwhm_lp_bef : float, optional
        FWHM of the Gaussian kernel used for low-pass filtering of the input
        images. Can be useful for mode='individual' and spatially subsampled
        input images. If set to 0 (default), no low-pass filtering is performed.
    fwhm_lp_aft : float, optional
        FWHM of the Gaussian kernel used for low-pass filtering of the final
        image. Can be useful for spatially subsampled input images. If set to 0
        (default), no low-pass filtering is performed.
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
    cube_res : numpy ndarray, 3d
        [full_output=True] The cube of residuals.
    cube_der : numpy ndarray, 3d
        [full_output=True] The derotated cube of residuals.
    frame : numpy ndarray, 2d
        Median combination of the de-rotated cube.

    """
    # Separating the parameters of the ParamsObject from optional rot_options
    class_params, rot_options = separate_kwargs_dict(
        initial_kwargs=all_kwargs, parent_class=ROLL_SUB_Params
    )

    # Extracting the object of parameters (if any)
    algo_params = None
    if ALGO_KEY in rot_options.keys():
        algo_params = rot_options[ALGO_KEY]
        del rot_options[ALGO_KEY]

    if algo_params is None:
        algo_params = ROLL_SUB_Params(*all_args, **class_params)

    global ARRAY

    mang = np.mean(algo_params.angle_list)
    if len(algo_params.angle_list) == 2:
        ang1, ang2 = algo_params.angle_list
    else:
        rang1 = algo_params.angle_list[np.where(algo_params.angle_list <= mang)]
        ang1 = np.mean(rang1)
        rang2 = algo_params.angle_list[np.where(algo_params.angle_list > mang)]
        ang2 = np.mean(rang2)

    if isinstance(algo_params.cube, tuple):
        nh1 = len(algo_params.cube[0])
        nh2 = len(algo_params.cube[1])
        ARRAY = np.concatenate((ARRAY[0], ARRAY[1]), axis=0)
        algo_params.angle_list = [ang1]*nh1
        algo_params.angle_list.extend([ang2]*nh2)
        algo_params.angle_list = np.array(algo_params.angle_list)
    elif algo_params.cube.ndim == 3:
        ARRAY = algo_params.cube.copy()
        nfr = ARRAY.shape[0]
        nh1 = nfr//2
        nh2 = nfr-nfr//2
        if len(algo_params.angle_list) != nfr:
            if len(algo_params.angle_list) == 2:
                algo_params.angle_list = [ang1]*nh1
                algo_params.angle_list.extend([ang2]*nh2)
                algo_params.angle_list = np.array(algo_params.angle_list)
            else:
                msg = "Input angle_list has wrong length (should be 2 or {}"
                raise ValueError(msg.format(nfr))
    else:
        raise TypeError("Input array is not a 3d array or tuple of 2 3d arrays")

    if algo_params.verbose:
        start_time = time_ini()

    if algo_params.nproc is None:
        algo_params.nproc = cpu_count() // 2

    if algo_params.fwhm_lp_bef > 0:
        cube = cube_filter_lowpass(ARRAY, fwhm_size=algo_params.fwhm_lp_bef)
    else:
        cube = ARRAY.copy()

    idx1 = np.where(algo_params.angle_list <= mang)
    idx2 = np.where(algo_params.angle_list > mang)

    if algo_params.mode == 'individual':
        if nh1 != nh2:
            msg = "In 'individual' mode, the same number of images is required "
            msg += "for both roll angles."
            raise ValueError(msg)
        cube1 = cube[idx1]
        cube2 = cube[idx2]
        cube_res1 = np.array([cube1[i]-cube2[i] for i in range(nh1)])
        cube_res2 = np.array([cube2[i]-cube1[i] for i in range(nh2)])
        cube_res = np.concantenate((cube_res1, cube_res2), axis=0)
        cube_der = cube_derotate(cube_res, algo_params.angle_list,
                                 imlib=algo_params.imlib,
                                 interpolation=algo_params.interpolation,
                                 nproc=algo_params.nproc,
                                 **rot_options,)
        fin_roll = cube_collapse(cube_der, mode=algo_params.collapse)

    else:
        mr1 = np.mean(cube[idx1], axis=0)
        mr2 = np.mean(cube[idx2], axis=0)

        dr12 = mr1-mr2
        dr12_drot = frame_rotate(dr12, -algo_params.angle_list[idx1],
                                 imlib=algo_params.imlib,
                                 interpolation=algo_params.interpolation,
                                 **rot_options)

        dr21 = mr2-mr1
        dr21_drot = frame_rotate(dr21, -algo_params.angle_list[idx2],
                                 imlib=algo_params.imlib,
                                 interpolation=algo_params.interpolation,
                                 **rot_options)

        cube_res = np.array([dr12, dr21])
        cube_der = np.array([dr12_drot, dr21_drot])
        fin_roll = cube_collapse(cube_der, mode=algo_params.collapse)

    if algo_params.fwhm_lp_aft > 0:
        fin_roll = frame_filter_lowpass(fin_roll,
                                        fwhm_size=algo_params.fwhm_lp_aft)

    if algo_params.mask_rad > 0:
        fin_roll = mask_circle(fin_roll, algo_params.mask_rad)

    if algo_params.verbose:
        print("Done derotating and combining")
        timing(start_time)

    if algo_params.full_output:
        return cube_res, cube_der, fin_roll
    else:
        return fin_roll
