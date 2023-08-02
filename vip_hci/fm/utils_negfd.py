#! /usr/bin/env python

"""
Module with post-processing related functions called from within the NEGFD
algorithm.
"""

__author__ = 'Valentin Christiaens, Carlos Alberto Gomez Gonzalez'
__all__ = ['cube_disk_free']

import numpy as np
from .fakecomp import cube_inject_companions


def cube_disk_free(disk_parameter, cube, angs, psfn, imlib='vip-fft',
                   interpolation='lanczos4', transmission=None):
    """
    Return a cube in which we have injected negative fake companion at the
    position/flux given by planet_parameter.

    Parameters
    ----------
    disk_parameter: numpy.array or list or tuple
        The (delta_x, delta_y, flux) for the model disk image. For a 4d cube r,
        theta and flux must all be 1d arrays with length equal to cube.shape[0];
        i.e. planet_parameter should have shape: (n_pl,3,n_ch).
    cube: numpy.array
        The cube of fits images expressed as a numpy.array.
    angs: numpy.array
        The parallactic angle fits image expressed as a numpy.array.
    psfn: numpy.array
        The scaled psf expressed as a numpy.array.
    imlib : str, optional
        See the documentation of the ``vip_hci.preproc.frame_rotate`` function.
    interpolation : str, optional
        See the documentation of the ``vip_hci.preproc.frame_rotate`` function.

    Returns
    -------
    cpf : numpy.array
        The cube with negative companions injected at the position given in
        planet_parameter.

    """
    cpf = np.zeros_like(cube)

    # unify planet_parameter format
    disk_parameter = np.array(disk_parameter)
    cond1 = cube.ndim == 3 and disk_parameter.ndim < 2
    cond2 = cube.ndim == 4 and disk_parameter.ndim < 3
    if cond1 or cond2:
        disk_parameter = disk_parameter[np.newaxis, :]

    if cube.ndim == 4:
        if disk_parameter.shape[2] != cube.shape[0]:
            raise TypeError("Input planet parameter with wrong dimensions.")

    for i in range(disk_parameter.shape[0]):
        if i == 0:
            cube_temp = cube
        else:
            cube_temp = cpf

        if cube.ndim == 4:
            for j in range(cube.shape[0]):
                flevel = -disk_parameter[i, 2, j]
                r = disk_parameter[i, 0, j]
                theta = disk_parameter[i, 1, j]
                cpf[j] = cube_inject_companions(cube_temp[j], psfn[j], angs,
                                                flevel=flevel,
                                                rad_dists=[r],
                                                n_branches=1,
                                                theta=theta,
                                                imlib=imlib,
                                                interpolation=interpolation,
                                                verbose=False,
                                                transmission=transmission)
        else:
            cpf = cube_inject_companions(cube_temp, psfn, angs, n_branches=1,
                                         flevel=-disk_parameter[i, 2],
                                         rad_dists=[disk_parameter[i, 0]],
                                         theta=disk_parameter[i, 1],
                                         imlib=imlib, verbose=False,
                                         interpolation=interpolation,
                                         transmission=transmission)
    return cpf
