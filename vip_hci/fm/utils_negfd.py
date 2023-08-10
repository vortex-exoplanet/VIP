#! /usr/bin/env python
"""
Module with post-processing related functions called from within the NEGFD
algorithm.
"""

__author__ = "Valentin Christiaens, Carlos Alberto Gomez Gonzalez"
__all__ = ["cube_disk_free"]

import numpy as np
from ..preproc import (
    cube_crop_frames,
    frame_pad,
    cube_shift,
    frame_shift,
    cube_derotate,
    frame_rotate,
    cube_rescaling,
    frame_rescaling,
)
from .fakedisk import cube_inject_fakedisk


def cube_disk_free(
    disk_parameter,
    cube,
    derot_angs,
    disk_img,
    psfn=None,
    imlib="vip-fft",
    interpolation="lanczos4",
    imlib_sh="vip-fft",
    interpolation_sh="lanczos4",
    imlib_sc="vip-fft",
    interpolation_sc="lanczos4",
    transmission=None,
    weights=None,
    **rot_options
):
    """

    Return a cube in which we have injected a negative fake disk model image \
    with xy shifts, spatial scaling, rotation and flux scaling given by \
    disk_parameter.

    Parameters
    ----------
    disk_parameter: numpy.array or list or tuple
        The (delta_x, delta_y, theta, scal, flux) for the model disk image. For
        a 4d cube, disk_parameter should be either (i) a numpy array with shape
        (5,n_ch), (ii) a 5-element list/tuple where delta_x, delta_y, theta and
        scal can be floats or 1d-arrays, but flux mandatorily as a 1d array with
        length equal to cube.shape[0].
    cube: numpy.array
        The cube of fits images expressed as a numpy.array.
    angs: numpy.array
        The parallactic angle fits image expressed as a numpy.array.
    disk_img: 2d or 3d numpy ndarray
        The disk image to be injected, as a 2d ndarray (for either 3D or 4D
        input cubes) or a 3d numpy array (for a 4D spectral+ADI input cube). In
        the latter case, the images should correspond to different wavelengths,
        and the zeroth shape of disk_model and cube should match.
    psfn: 2d or 3d numpy ndarray
        The normalized psf expressed as a numpy ndarray. Can be 3d for a 4d
        (spectral+ADI) input cube. This would only be used to convolve disk_img.
        Leave to None if the disk_img is already convolved.
    imlib : str, optional
        See the documentation of the ``vip_hci.preproc.frame_rotate`` function.
    interpolation : str, optional
        See the documentation of the ``vip_hci.preproc.frame_rotate`` function.
    imlib_sh : str, optional
        See the documentation of the ``vip_hci.preproc.frame_shift`` function.
    interpolation_sh : str, optional
        See the documentation of the ``vip_hci.preproc.frame_shift`` function.
    imlib_sc : str, optional
        See the documentation of the ``vip_hci.preproc.frame_rescaling``
        function.
    interpolation_sc : str, optional
        See the documentation of the ``vip_hci.preproc.frame_rescaling``
        function.
    transmission: numpy array, optional
        Radial transmission of the coronagraph, if any. Array with either
        2 x n_rad, 1+n_ch x n_rad columns. The first column should contain the
        radial separation in pixels, while the other column(s) are the
        corresponding off-axis transmission (between 0 and 1), for either all,
        or each spectral channel (only relevant for a 4D input cube).
    rot_options: dictionary, optional
        Dictionary with optional keyword values for "nproc", "cxy", "imlib",
        "interpolation, "border_mode", "mask_val",  "edge_blend",
        "interp_zeros", "ker" (see documentation of
        ``vip_hci.preproc.frame_rotate``)

    Returns
    -------
    cdf : numpy.array
        The cube with negative disk model injected at the position given in
        disk_parameter.

    """
    cdf = np.zeros_like(cube)
    disk_model_tmp = disk_img.copy()

    # unify planet_parameter format
    if isinstance(disk_parameter, np.ndarray):
        cond1 = cube.ndim == 3 and disk_parameter.ndim < 2
        cond2 = cube.ndim == 4 and disk_parameter.ndim < 3
        if not cond1 or not cond2:
            raise TypeError("Wrong dimensions for disk_parameter")
    elif len(disk_parameter) != 5:
        raise TypeError("Wrong length for disk_parameter")

    if cube.ndim == 4:
        if len(disk_parameter[-1]) != cube.shape[0]:
            msg = "Length of flux scaling parameter should match cube axis 0"
            raise ValueError(msg)
        if not isinstance(disk_parameter, np.ndarray):
            cond1 = np.isscalar(disk_parameter[0])
            cond2 = np.isscalar(disk_parameter[1])
            cond3 = np.isscalar(disk_parameter[2])
            if cond1 or cond2 or cond3:
                ndisk_parameter = np.zeros([5, cube.shape[0]])
                for j in range(5):
                    ndisk_parameter[j, :] = disk_parameter[j]
                disk_parameter = ndisk_parameter.copy()
            else:
                disk_parameter = np.array(disk_parameter)
                if disk_parameter.shape[-1] != cube.shape[0]:
                    raise TypeError("Input disk parameter has wrong dimensions")

        if disk_model_tmp.ndim == 2:
            disk_model_tmp = np.array([disk_model_tmp] * cube.shape[0])
        if psfn is None:
            psfn = [None] * cube.shape[0]

        delta_x = disk_parameter[0, :]
        delta_y = disk_parameter[1, :]
        theta = disk_parameter[2, :]
        disk_model_tmp = cube_shift(
            disk_model_tmp,
            delta_y,
            delta_x,
            imlib=imlib_sh,
            interpolation=interpolation_sh,
        )
        angs = np.array([theta] * disk_model_tmp.shape[0])
        disk_model_tmp = cube_derotate(
            disk_model_tmp,
            -angs,
            imlib=imlib,
            interpolation=interpolation,
            **rot_options
        )
        for j in range(cube.shape[0]):
            sc = disk_parameter[3, j]
            disk_model_tmp[j] = frame_rescaling(
                disk_model_tmp[j],
                scale=sc,
                imlib=imlib_sc,
                interpolation=interpolation_sc,
            )
            flevel = disk_parameter[4, j]
            disk_rot = cube_inject_fakedisk(
                flevel * disk_model_tmp[j],
                derot_angs,
                psf=psfn[j],
                transmission=transmission,
                **rot_options
            )
            if weights is not None:
                if len(weights) != cube.shape[1]:
                    raise TypeError("weights length should match cube axis 1")
                else:
                    disk_rot *= weights
            if disk_rot.shape[-1] < cube.shape[-1]:
                disk_rot_pad = np.zeros_like(cube[j])
                pad_fac = cube.shape[-1] / disk_rot.shape[-1]
                for i in range(cube.shape[1]):
                    disk_rot_pad[i] = frame_pad(
                        disk_rot[i],
                        pad_fac,
                        fillwith=0,
                        keep_parity=False,
                        full_output=False,
                    )
                disk_rot = disk_rot_pad
            elif disk_rot.shape[-1] > cube.shape[-1]:
                disk_rot = cube_crop_frames(disk_rot, cube.shape[-1], force=True)
            cdf[j] = cube[j] - disk_rot

    else:
        delta_x = disk_parameter[0]
        delta_y = disk_parameter[1]
        theta = disk_parameter[2]
        sc = disk_parameter[3]
        flevel = disk_parameter[4]
        disk_model_tmp = frame_shift(
            disk_model_tmp,
            delta_y,
            delta_x,
            imlib=imlib_sh,
            interpolation=interpolation_sh,
        )
        disk_model_tmp = frame_rotate(
            disk_model_tmp,
            theta,
            imlib=imlib,
            interpolation=interpolation,
            **rot_options
        )

        disk_model_tmp = frame_rescaling(
            disk_model_tmp, scale=sc, imlib=imlib_sc, interpolation=interpolation_sc
        )
        disk_rot = cube_inject_fakedisk(
            flevel * disk_model_tmp,
            derot_angs,
            psf=psfn,
            transmission=transmission,
            **rot_options
        )
        if weights is not None:
            if len(weights) != cube.shape[0]:
                raise TypeError("weights length should match cube axis 0")
            else:
                disk_rot *= weights

        if disk_rot.shape[-1] < cube.shape[-1]:
            disk_rot_pad = np.zeros_like(cube)
            pad_fac = cube.shape[-1] / disk_rot.shape[-1]
            for i in range(cube.shape[0]):
                disk_rot_pad[i] = frame_pad(
                    disk_rot[i],
                    pad_fac,
                    fillwith=0,
                    keep_parity=False,
                    full_output=False,
                )
            disk_rot = disk_rot_pad
        elif disk_rot.shape[-1] > cube.shape[-1]:
            disk_rot = cube_crop_frames(disk_rot, cube.shape[-1], force=True)
        cdf = cube - disk_rot

    return cdf
