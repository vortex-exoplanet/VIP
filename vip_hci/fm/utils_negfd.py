#! /usr/bin/env python

"""
Module with post-processing related functions called from within the NEGFD
algorithm.
"""

__author__ = 'Valentin Christiaens, Carlos Alberto Gomez Gonzalez'
__all__ = ['cube_disk_free']

import numpy as np
from ..preproc import cube_crop_frames, frame_pad, cube_shift, frame_shift
from .fakedisk import cube_inject_fakedisk


def cube_disk_free(disk_parameter, cube, angs, disk_model, psfn, imlib='vip-fft',
                   interpolation='lanczos4', imlib_sh='vip-fft',
                   interpolation_sh='lanczos4', transmission=None):
    """
    Return a cube in which we have injected a negative fake disk model image at 
    the position and flux scaling given by disk_parameter.

    Parameters
    ----------
    disk_parameter: numpy.array or list or tuple
        The (delta_x, delta_y, flux) for the model disk image. For a 4d cube,
        delta_x, delta_y and flux must all be 1d arrays with length equal to 
        cube.shape[0]; i.e. disk_parameter should have shape: (3,n_ch).
    cube: numpy.array
        The cube of fits images expressed as a numpy.array.
    angs: numpy.array
        The parallactic angle fits image expressed as a numpy.array.
    disk_model: 2d or 3d numpy ndarray
        The disk image to be injected, as a 2d ndarray (for either 3D or 4D 
        input cubes) or a 3d numpy array (for a 4D spectral+ADI input cube). In 
        the latter case, the images should correspond to different wavelengths, 
        and the zeroth shape of disk_model and cube should match.
    psfn: 2d or 3d numpy ndarray
        The normalized psf expressed as a numpy ndarray. Can be 3d for a 4d 
        (spectral+ADI) input cube.
    imlib : str, optional
        See the documentation of the ``vip_hci.preproc.frame_rotate`` function.
    interpolation : str, optional
        See the documentation of the ``vip_hci.preproc.frame_rotate`` function.
    imlib_sh : str, optional
        See the documentation of the ``vip_hci.preproc.frame_shift`` function.
    interpolation_sh : str, optional
        See the documentation of the ``vip_hci.preproc.frame_shift`` function.
    transmission: numpy array, optional
        Radial transmission of the coronagraph, if any. Array with either
        2 x n_rad, 1+n_ch x n_rad columns. The first column should contain the
        radial separation in pixels, while the other column(s) are the
        corresponding off-axis transmission (between 0 and 1), for either all,
        or each spectral channel (only relevant for a 4D input cube).

    Returns
    -------
    cdf : numpy.array
        The cube with negative disk model injected at the position given in
        disk_parameter.

    """
    cdf = np.zeros_like(cube)
    disk_model_tmp = disk_model.copy()

    # unify planet_parameter format
    disk_parameter = np.array(disk_parameter)
    cond1 = cube.ndim == 3 and disk_parameter.ndim < 2
    cond2 = cube.ndim == 4 and disk_parameter.ndim < 3
    if cond1 or cond2:
        disk_parameter = disk_parameter[np.newaxis, :]

    if cube.ndim == 4:
        if disk_parameter.shape[2] != cube.shape[0]:
            raise TypeError("Input planet parameter with wrong dimensions.")
        if disk_model_tmp.ndim == 2:
            disk_model_tmp = np.array([disk_model_tmp]*cube.shape[0])
        if psfn is None:
            psfn = [None]*cube.shape[0]

    if cube.ndim == 4:
        delta_x = disk_parameter[0, :]
        delta_y = disk_parameter[1, :]
        disk_model_tmp = cube_shift(disk_model_tmp, delta_y, delta_x,
                                    imlib=imlib_sh,
                                    interpolation=interpolation_sh)
        for j in range(cube.shape[0]):
            flevel = -disk_parameter[2, j]
            disk_rot = cube_inject_fakedisk(flevel*disk_model_tmp[j], angs,
                                            psf=psfn[j],
                                            transmission=transmission,
                                            imlib=imlib,
                                            interpolation=interpolation)
            if disk_rot.shape[-1] < cube.shape[-1]:
                disk_rot_pad = np.zeros_like(cube[j])
                pad_fac = cube.shape[-1]/disk_rot.shape[-1]
                for i in range(cube.shape[1]):
                    disk_rot_pad[i] = frame_pad(disk_rot[i], pad_fac,
                                                fillwith=0,
                                                keep_parity=False,
                                                full_output=False)
                disk_rot = disk_rot_pad
            elif disk_rot.shape[-1] > cube.shape[-1]:
                disk_rot = cube_crop_frames(disk_rot, cube.shape[-1],
                                            force=True)
            cdf[j] = cube[j]+disk_rot

    else:
        flevel = -disk_parameter[2]
        delta_x = disk_parameter[0]
        delta_y = disk_parameter[1]
        disk_model_tmp = frame_shift(disk_model_tmp, delta_y, delta_x,
                                     imlib=imlib_sh,
                                     interpolation=interpolation_sh)
        disk_rot = cube_inject_fakedisk(flevel*disk_model_tmp, angs, psf=psfn,
                                        transmission=transmission, imlib=imlib,
                                        interpolation=interpolation)
        # TO DO
        if disk_rot.shape[-1] < cube.shape[-1]:
            disk_rot_pad = np.zeros_like(cube)
            pad_fac = cube.shape[-1]/disk_rot.shape[-1]
            for i in range(cube.shape[0]):
                disk_rot_pad[i] = frame_pad(disk_rot[i], pad_fac, fillwith=0,
                                            keep_parity=False,
                                            full_output=False)
            disk_rot = disk_rot_pad
        elif disk_rot.shape[-1] > cube.shape[-1]:
            disk_rot = cube_crop_frames(disk_rot, cube.shape[-1], force=True)
        cdf = cube+disk_rot

    return cdf
