#! /usr/bin/env python

"""
Module with post-processing related functions called from within the NFC
algorithm.
"""

__author__ = 'Carlos Alberto Gomez Gonzalez'
__all__ = ['cube_planet_free']

import numpy as np
from ..fm import cube_inject_companions


def cube_planet_free(planet_parameter, cube, angs, psfn, plsc, imlib='vip-fft',
                     interpolation='lanczos4',transmission=None):
    """
    Return a cube in which we have injected negative fake companion at the
    position/flux given by planet_parameter.

    Parameters
    ----------
    planet_parameter: numpy.array or list
        The (r, theta, flux) for all known companions. For a 4d cube r, 
        theta and flux must all be 1d arrays with length equal to cube.shape[0];
        i.e. planet_parameter should have shape: (n_pl,3,n_ch).
    cube: numpy.array
        The cube of fits images expressed as a numpy.array.
    angs: numpy.array
        The parallactic angle fits image expressed as a numpy.array.
    psfn: numpy.array
        The scaled psf expressed as a numpy.array.
    plsc: float
        The platescale, in arcsec per pixel.
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

    planet_parameter = np.array(planet_parameter)

    if cube.ndim == 4:
        if planet_parameter.shape[3] != cube.shape[0]:
            raise TypeError("Input planet parameter with wrong dimensions.")
        
    for i in range(planet_parameter.shape[0]):
        if i == 0:
            cube_temp = cube
        else:
            cube_temp = cpf

        if cube.ndim == 4:
            for j in cube.shape[0]:
                cpf[j] = cube_inject_companions(cube_temp[j], psfn[j], angs,
                                                flevel=-planet_parameter[i, 2, j], 
                                                plsc=plsc,
                                                rad_dists=[planet_parameter[i, 0, j]],
                                                n_branches=1, 
                                                theta=planet_parameter[i, 1, j],
                                                imlib=imlib, 
                                                interpolation=interpolation,
                                                verbose=False,
                                                transmission=transmission)
        else:
            cpf = cube_inject_companions(cube_temp, psfn, angs,
                                         flevel=-planet_parameter[i, 2], plsc=plsc,
                                         rad_dists=[planet_parameter[i, 0]],
                                         n_branches=1, theta=planet_parameter[i, 1],
                                         imlib=imlib, interpolation=interpolation,
                                         verbose=False, transmission=transmission)
    return cpf