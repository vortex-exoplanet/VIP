#! /usr/bin/env python
"""
Implementation of an iterative roll subtraction algorithm for PSF subtraction\
in imaging sequences obtained with space-based instruments (e.g. JWST or HST)\
with different roll angles.

The concept was proposed in [HEA00]_ for application to HST/NICMOS observations.

.. [HEA00]
   | Heap et al. 2000
   | **Space Telescope Imaging Spectrograph Coronagraphic Observations of β
   Pictoris**
   | *The Astrophysical Journal, Volume 539, Issue 1, pp. 435-444*
   | `https://arxiv.org/abs/astro-ph/9911363
     <https://arxiv.org/abs/astro-ph/9911363>`_

"""

__author__ = "Valentin Christiaens"
__all__ = ["iroll"]

import numpy as np
from dataclasses import dataclass
from typing import Union, List
from ..config.paramenum import ALGO_KEY
from ..config.utils_param import separate_kwargs_dict
from ..metrics import stim_map, inverse_stim_map
from ..preproc import cube_derotate
from ..psfsub import ROLL_SUB_Params, roll_sub
from ..var import mask_circle


@dataclass
class IROLL_SUB_Params(ROLL_SUB_Params):
    """
    Set of parameters for the roll subtraction routine.

    The class inherits from ROLL_SUB_Params.

    See function `iroll_sub` for documentation.
    """

    nit: int = 1
    thr: Union[float, str] = 0.
    thr_mode: str = 'STIM'
    r_out: float = None
    r_max: float = None


def iroll(*all_args: List, **all_kwargs: dict):
    """
    Perform iterative roll-subtraction, followed by derotation and stacking of\
    residual images.

    Reference: [HEA00]_.

    Parameters
    ----------
    all_args: list, optional
        Positionnal arguments for the iroll_sub algorithm. Full list of
        parameters combines those of ``vip_hci.psfsub.roll_sub`` and those
        below.
    all_kwargs: dictionary, optional
        Mix of keyword arguments that can initialize a IROLL_SUB_Params and the
        optional ``rot_options`` dictionary (with keywords ``border_mode``,
        ``mask_val``, ``edge_blend``, ``interp_zeros``, ``ker``; see docstrings
        of ``vip_hci.preproc.frame_rotate``). Can also contain a
        IROLL_SUB_Params object/dictionary named ``algo_params``.

    Parameters
    ----------
    nit : int, optional
        Number of iterations for the algorithm.
    thr : float, optional
        Threshold for estimation of significant signals at each iteration. The
        threshold is expressed in terms of normalized STIM map value. A value of
        0 will propagate all positive signals at the subsequent iteration.
    r_out: float or None, opt
        Outermost radius in pixels of circumstellar signals (estimated). This
        will be used if thr is set to 'auto'. The max STIM value beyond that
        radius will be used as minimum threshold. If r_out is set to None, the
        quarter-width of the frames will be used.
    r_max: float or None, opt
        Max radius in pixels where the STIM map will be considered. The max STIM
        value beyond r_out but within r_max will be used as minimum threshold.
        If r_max is set to None, the half-width of the frames will be used.

    Returns
    -------
    cube_resi : numpy ndarray, 3d
        [full_output=True] The cube of residuals after nit iterations.
    cube_deri : numpy ndarray, 3d
        [full_output=True] The derotated cube of residuals after nit iterations.
    rolli : numpy ndarray, 2d
        Median combination of the de-rotated cube after nit iterations.
    all_rolli : numpy ndarray, 3d
        Cube of nit images containing the post-processed image at each iteration
        up to nit iterations.
    """

    def _find_significant_signals(residuals_cube, residuals_cube_, angle_list,
                                  thr, mask=0, r_out=None, r_max=None):
        # Identifies significant signals with STIM map (outside mask)
        stim = stim_map(residuals_cube_)
        good_mask = np.zeros_like(stim)
        if thr == 0:
            good_mask[np.where(stim > thr)] = 1
            return good_mask, stim
        inv_stim = inverse_stim_map(residuals_cube, angle_list)
        if mask:
            inv_stim = mask_circle(inv_stim, mask)
        max_inv = np.amax(inv_stim)
        if max_inv == 0:
            max_inv = 1  # np.amin(stim[np.where(stim>0)])
        if thr == 'auto':
            if r_out is None:
                r_out = residuals_cube.shape[-1]//4
            if r_max is None:
                r_max = residuals_cube.shape[-1]//2
            inv_stim_rout = mask_circle(inv_stim, r_out)
            inv_stim_rmax = mask_circle(inv_stim_rout, r_max, mode='out')
            thr = np.amax(inv_stim_rmax)/max_inv
        norm_stim = stim/max_inv
        good_mask[np.where(norm_stim > thr)] = 1

        return good_mask, norm_stim

    # Separating the parameters of the ParamsObject from optional rot_options
    class_params, rot_options = separate_kwargs_dict(
        initial_kwargs=all_kwargs, parent_class=IROLL_SUB_Params
    )
    # Do the same to separate IROLL and ROLL params
    roll_params, iroll_params = separate_kwargs_dict(
        initial_kwargs=class_params, parent_class=ROLL_SUB_Params
    )

    # Extracting the object of parameters (if any)
    algo_params = None
    if ALGO_KEY in rot_options.keys():
        algo_params = rot_options[ALGO_KEY]
        del rot_options[ALGO_KEY]

    if algo_params is None:
        algo_params = IROLL_SUB_Params(*all_args, **class_params)

    # force full_output
    roll_params['full_output'] = True

    global ARRAY
    ARRAY = algo_params.cube.copy()
    nframes = ARRAY.shape[0]

    # first run
    cube_res0, cube_der0, roll0 = roll_sub(**roll_params, **rot_options)

    # estimate significant signal
    if algo_params.thr_mode == 'STIM':
        sig_mask, _ = _find_significant_signals(cube_res0, cube_der0,
                                                algo_params.angle_list,
                                                algo_params.thr,
                                                mask=algo_params.mask_rad,
                                                r_out=algo_params.r_out)
    else:
        sig_mask = np.ones_like(roll0)
        sig_mask[np.where(roll0 < algo_params.thr)] = 0
    # subtract significant signals to original cube
    roll0_cube = np.repeat(roll0[np.newaxis, :, :], nframes, axis=0)
    mask_cube = np.repeat(sig_mask[np.newaxis, :, :], nframes, axis=0)
    img_cube = cube_derotate(roll0_cube, -algo_params.angle_list, **rot_options)
    sig_cube = cube_derotate(mask_cube, -algo_params.angle_list,
                             imlib='skimage', interpolation='bilinear')
    sig_cube[np.where(sig_cube < 0.5)] = 0
    sig_cube[np.where(sig_cube >= 0.5)] = 1
    img_cube *= sig_cube

    # run for number of requested iterations on adapted cube
    all_rolli = [roll0]
    for i in range(algo_params.nit):
        roll_params['cube'] = ARRAY
        roll_params['cube_sig'] = img_cube
        cube_resi, cube_deri, rolli = roll_sub(**roll_params, **rot_options)

        # estimate significant signal
        if algo_params.thr_mode == 'STIM':
            sig_mask, _ = _find_significant_signals(cube_resi, cube_deri,
                                                    algo_params.angle_list,
                                                    algo_params.thr,
                                                    mask=algo_params.mask_rad,
                                                    r_out=algo_params.r_out)
        else:
            sig_mask = np.ones_like(rolli)
            sig_mask[np.where(rolli < algo_params.thr)] = 0
        # subtract significant signals to original cube
        rolli_cube = np.repeat(rolli[np.newaxis, :, :], nframes, axis=0)
        mask_cube = np.repeat(sig_mask[np.newaxis, :, :], nframes, axis=0)
        img_cube = cube_derotate(rolli_cube, -algo_params.angle_list,
                                 **rot_options)
        sig_cube = cube_derotate(mask_cube, -algo_params.angle_list,
                                 imlib='skimage', interpolation='bilinear')
        sig_cube[np.where(sig_cube < 0.5)] = 0
        sig_cube[np.where(sig_cube >= 0.5)] = 1
        img_cube *= sig_cube
        all_rolli.append(rolli)

    all_rolli = np.array(all_rolli)

    if algo_params.full_output:
        return cube_resi, cube_deri, rolli, all_rolli
    else:
        return rolli
