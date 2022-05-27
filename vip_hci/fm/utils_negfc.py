#! /usr/bin/env python

"""
Module with post-processing related functions called from within the NFC
algorithm.
"""

__author__ = 'Carlos Alberto Gomez Gonzalez'
__all__ = ['cube_planet_free',
           'find_nearest']

import numpy as np
from ..fm import cube_inject_companions


def cube_planet_free(planet_parameter, cube, angs, psfn, imlib='vip-fft',
                     interpolation='lanczos4', transmission=None):
    """
    Return a cube in which we have injected negative fake companion at the
    position/flux given by planet_parameter.

    Parameters
    ----------
    planet_parameter: numpy.array or list or tuple
        The (r, theta, flux) for all known companions. For a 4d cube r,
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
    planet_parameter = np.array(planet_parameter)
    cond1 = cube.ndim == 3 and planet_parameter.ndim < 2
    cond2 = cube.ndim == 4 and planet_parameter.ndim < 3
    if cond1 or cond2:
        planet_parameter = planet_parameter[np.newaxis, :]

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
                flevel = -planet_parameter[i, 2, j]
                r = planet_parameter[i, 0, j]
                theta = planet_parameter[i, 1, j]
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
                                         flevel=-planet_parameter[i, 2],
                                         rad_dists=[planet_parameter[i, 0]],
                                         theta=planet_parameter[i, 1],
                                         imlib=imlib, verbose=False,
                                         interpolation=interpolation,
                                         transmission=transmission)
    return cpf


def find_nearest(array, value, output='index', constraint=None, n=1):
    """
    Function to find the indices, and optionally the values, of an array's n
    closest elements to a certain value.

    Parameters
    ----------
    array: 1d numpy array or list
        Array in which to check the closest element to value.
    value: float
        Value for which the algorithm searches for the n closest elements in
        the array.
    output: str, opt {'index','value','both' }
        Set what is returned
    constraint: str, opt {None, 'ceil', 'floor'}
        If not None, will check for the closest element larger than value (ceil)
        or closest element smaller than value (floor).
    n: int, opt
        Number of elements to be returned, sorted by proximity to the values.
        Default: only the closest value is returned.

    Returns
    -------
    Either:
        (output='index'): index/indices of the closest n value(s) in the array;
        (output='value'): the closest n value(s) in the array,
        (output='both'): closest value(s) and index/-ices, respectively.
    By default, only returns the index/indices.

    Possible constraints: 'ceil', 'floor', None ("ceil" will return the closest
    element with a value greater than 'value', "floor" the opposite)
    """
    if isinstance(array, np.ndarray):
        pass
    elif isinstance(array, list):
        array = np.array(array)
    else:
        raise ValueError("Input type for array should be np.ndarray or list.")

    if constraint is None:
        fm = np.absolute(array-value)
        idx = fm.argsort()[:n]
    elif constraint == 'floor' or constraint == 'ceil':
        indices = np.arange(len(array), dtype=np.int32)
        if constraint == 'floor':
            fm = -(array-value)
        else:
            fm = array-value
        crop_indices = indices[np.where(fm > 0)]
        fm = fm[np.where(fm > 0)]
        idx = fm.argsort()[:n]
        idx = crop_indices[idx]
        if len(idx) == 0:
            msg = "No indices match the constraint ({} w.r.t {:.2f})"
            print(msg.format(constraint, value))
            raise ValueError("No indices match the constraint")
    else:
        raise ValueError("Constraint not recognised")

    if n == 1:
        idx = idx[0]

    if output == 'index':
        return idx
    elif output == 'value':
        return array[idx]
    else:
        return array[idx], idx
