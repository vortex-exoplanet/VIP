#! /usr/bin/env python

"""
Module with pixel and frame subsampling functions.
"""

from __future__ import division
from __future__ import print_function

__author__ = 'C. Gomez @ ULg'
__all__ = ['cube_collapse',
           'cube_subsample',
           'cube_subsample_trimmean']

import numpy as np


def cube_collapse(cube, mode='median', n=50, wl_cube=False):
    """ Collapses a cube into a frame (4d or 3d array to 2d array) depending on the
    parameter *mode*. It's possible to perform a trimmed mean combination of the
    frames based on description in Brandt+ 2012.

    Parameters
    ----------
    cube : array_like
        Cube.
    mode : {'median', 'mean', 'sum', 'trimmean'}, str optional
        Sets the way of collapsing the images in the cube.
    n : int, optional
        Sets the discarded values at high and low ends. When n = N is the same
        as taking the mean, when n = 1 is like taking the median.
    wl_cube : True or False
        Enable if you only want to collapse the rotations. Set to False to collapse wavelengths

    Returns
    -------
    frame : array_like
        Output array, cube combined.
    """
    arr = cube
    if not (arr.ndim == 3 or arr.ndim==4):
        raise TypeError('The input array is not a cube or 3d or 4d array.')

    if arr.ndim==3:
        if mode=='mean':
            frame = np.mean(arr, axis=0)
        elif mode=='median':
            frame = np.median(arr, axis=0)
        elif mode=='sum':
            frame = np.sum(arr, axis=0)
        elif mode=='trimmean':
            N = arr.shape[0]
            if N % 2 == 0:
                k = (N - n)//2
            else:
                k = (N - n)/2

            frame = np.empty_like(arr[0])
            for index, _ in np.ndenumerate(arr[0]):
                sort = np.sort(arr[:,index[0],index[1]])
                frame[index] = np.mean(sort[k:N-k])

    elif arr.ndim==4:
        if mode=='mean':
            cube_wl = np.mean(arr, axis=1)
        elif mode=='median':
            cube_wl = np.median(arr, axis=1)
        elif mode=='sum':
            cube_wl = np.sum(arr, axis=1)
        elif mode=='trimmean':
            print 'Sorry: no trimmed mean for 4d cubes'
            return 0
        if wl_cube:
            return cube_wl
        else:
            if mode=='mean':
                frame = np.mean(cube_wl, axis=0)
            elif mode=='median':
                frame = np.median(cube_wl, axis=0)
            elif mode=='sum':
                frame = np.sum(cube_wl, axis=0)
    return frame


def cube_subsample(array, n, mode="mean", parallactic=None, verbose=True):
    """Mean/Median combines frames in cube with window n.

    Parameters
    ----------
    n : int
        Window for mean/median.
    array : array_like
        Input 3d array, cube.
    mode : {'mean','median'}
        Switch for choosing mean or median.
    parallactic: array_like
        List of corresponding parallactic angles.

    Returns
    -------
    arr_view : array_like
        Resulting array.
    angles : array_like
        Parallactic angles.
    """
    if not array.ndim == 3:
        raise TypeError('The input array is not a cube or 3d array.')
    m = int(array.shape[0]/n)
    resid = array.shape[0]%n
    y = array.shape[1]
    x = array.shape[2]
    arr = np.empty([m, y, x])
    if parallactic is not None:
        angles = np.zeros(m)

    if mode == 'median':  func = np.median
    elif mode=='mean':  func = np.mean
    else:
        raise ValueError('Mode should be either Mean or Median.')

    for i in range(m):
        arr[i, :, :] = func(array[:n, :, :], axis=0)
        if parallactic is not None:  angles[i] = func(parallactic[:n])
        if i >= 1:
            arr[i, :, :] = func(array[n*i:n*i+n, :, :], axis=0)
            if parallactic is not None:
                angles[i] = func(parallactic[n*i:n*i+n])

    if verbose:
        print("Datacube subsampled by taking the {:} of {:} frames".format(mode ,n))
        if resid > 0:
            msg = "Initial # of frames and window are not multiples ({:} frames were dropped)"
<<<<<<< HEAD:vip/preproc/subsampling.py
            print msg.format(resid)
        print "New cube contains {:} frames".format(m)

=======
            print(msg.format(resid))     
        print("New cube contains {:} frames".format(m))
                                   
>>>>>>> 3bf9edb7f4129af6591c67341fd3048fce28df4a:vip_hci/preproc/subsampling.py
    if parallactic is not None:
        return arr, angles
    else:
        return arr


def cube_subsample_trimmean(arr, n, m):
    """Performs a trimmed mean combination every m frames in a cube. Based on
    description in Brandt+ 2012.

    Parameters
    ----------
    arr : array_like
        Cube.
    n : int
        Sets the discarded values at high and low ends. When n = N is the same
        as taking the mean, when n = 1 is like taking the median.
    m : int
        Window from the trimmed mean.

    Returns
    -------
    arr_view : array_like
        Output array, cube combined.
    """
    if not arr.ndim == 3:
        raise TypeError('The input array is not a cube or 3d array.')
    num = int(arr.shape[0]/m)
    res = int(arr.shape[0]%m)
    y = arr.shape[1]
    x = arr.shape[2]
    arr2 = np.empty([num+2, y, x])
    for i in range(num):
        arr2[0] = cube_collapse(arr[:m, :, :], 'trimmean', n)
        if i > 0:
            arr2[i] = cube_collapse(arr[m*i:m*i+m, :, :], 'trimmean', n)
    arr2[num] = cube_collapse(arr[-res:, :, :], 'trimmean', n)
    arr_view = arr2[:num+1]                                                      # slicing until m+1 - last index not included
    print("\nDone trimmed mean over FITS-Cube with window m=" + str(m))
    return arr_view
