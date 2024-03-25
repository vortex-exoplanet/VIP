#! /usr/bin/env python
"""
Module with pixel and frame subsampling functions.

.. [BRA13]
   | Brandt et al. 2013
   | **New Techniques for High-contrast Imaging with ADI: The ACORNS-ADI SEEDS
     Data Reduction Pipeline**
   | *The Astrophysical Journal, Volume 764, Issue 2, p. 183*
   | `https://arxiv.org/abs/1209.3014
     <https://arxiv.org/abs/1209.3014>`_

"""

__author__ = 'Carlos Alberto Gomez Gonzalez, Valentin Christiaens'
__all__ = ['cube_collapse',
           'cube_subsample',
           'cube_subsample_trimmean']

import numpy as np


def cube_collapse(cube, mode='median', n=50, w=None):
    """Collapse a 3D or 4D cube into a 2D frame or 3D cube, respectively.

    The  ``mode`` parameter determines how the collapse should be done. It is
    possible to perform a trimmed mean combination of the frames, as in
    [BRA13]_. In case of a 4D input cube, it is assumed to be an IFS dataset
    with the zero-th axis being the spectral dimension, and the first axis the
    temporal dimension.


    Parameters
    ----------
    cube : numpy ndarray
        Cube.
    mode : {'median', 'mean', 'sum', 'max', 'trimmean', 'absmean', 'wmean'}
        Sets the way of collapsing the images in the cube.
        'wmean' stands for weighted mean and requires weights w to be provided.
        'absmean' stands for the mean of absolute values (potentially useful
        for negfc).
    n : int, optional
        Sets the discarded values at high and low ends. When n = N is the same
        as taking the mean, when n = 1 is like taking the median.
    w: 1d numpy array or list, optional
        Weights to be applied for a weighted mean. Need to be provided if
        collapse mode is 'wmean'.

    Returns
    -------
    frame : numpy ndarray
        Output array, cube combined.
    """
    arr = cube
    if arr.ndim == 3:
        ax = 0
    elif arr.ndim == 4:
        nch = arr.shape[0]
        ax = 1
    else:
        raise TypeError('The input array is not a cube or 3d array.')

    if mode == 'wmean':
        if w is None:
            raise ValueError(
                "Weights have to be provided for weighted mean mode")
        if len(w) != cube.shape[0]:
            raise TypeError("Weights need same length as cube")
        if isinstance(w, list):
            w = np.array(w)

    if mode == 'mean':
        frame = np.nanmean(arr, axis=ax)
    elif mode == 'median':
        frame = np.nanmedian(arr, axis=ax)
    elif mode == 'sum':
        frame = np.nansum(arr, axis=ax)
    elif mode == 'max':
        frame = np.nanmax(arr, axis=ax)
    elif mode == 'trimmean':
        N = arr.shape[ax]
        k = (N - n)//2
        if N % 2 != n % 2:
            n += 1
        if ax == 0:
            frame = np.empty_like(arr[0])
            for index, _ in np.ndenumerate(arr[0]):
                sort = np.sort(arr[:, index[0], index[1]])
                frame[index] = np.nanmean(sort[k:k+n])
        else:
            frame = np.empty_like(arr[:, 0])
            for j in range(nch):
                for index, _ in np.ndenumerate(arr[:, 0]):
                    sort = np.sort(arr[j, :, index[0], index[1]])
                    frame[j][index] = np.nanmean(sort[k:k+n])
    elif mode == 'wmean':
        arr[np.where(np.isnan(arr))] = 0  # to avoid product with nan
        if ax == 0:
            frame = np.inner(w, np.moveaxis(arr, 0, -1))
        else:
            frame = np.empty_like(arr[:, 0])
            for j in range(nch):
                frame[j] = np.inner(w, np.moveaxis(arr[j], 0, -1))
    elif mode == 'absmean':
        frame = np.nanmean(np.abs(arr), axis=ax)
    else:
        raise TypeError("mode not recognized")

    return frame


def cube_subsample(array, n, mode="mean", w=None, parallactic=None,
                   verbose=True):
    """Mean/Median combines frames in 3d or 4d cube with window ``n``.

    Parameters
    ----------
    array : numpy ndarray
        Input 3d array, cube.
    n : int
        Window for mean/median.
    mode : {'mean','median', 'wmean'}, optional
        Switch for choosing mean, median or weighted average.
    parallactic : numpy ndarray, optional
        List of corresponding parallactic angles.
    verbose : bool optional

    Returns
    -------
    arr_view : numpy ndarray
        Resulting array.
    If ``parallactic`` is provided the the new cube and angles are returned.
    """
    if array.ndim not in [3, 4]:
        raise TypeError('The input array is not a cube or 3d or 4d array')

    kwargs = {}
    if mode == 'median':
        func = np.median
    elif mode == 'mean':
        func = np.mean
    elif mode == 'wmean':
        func = np.average
        w = w/np.sum(w)
        kwargs = {'weights': w}
    else:
        raise ValueError('`Mode` should be either Mean or Median')

    if array.ndim == 3:
        m = int(array.shape[0] / n)
        resid = array.shape[0] % n
        y = array.shape[1]
        x = array.shape[2]
        arr = np.empty([m, y, x])
        if parallactic is not None:
            angles = np.zeros(m)

        for i in range(m):
            arr[i, :, :] = func(array[n * i:n * i + n, :, :], axis=0, **kwargs)
            if parallactic is not None:
                angles[i] = func(parallactic[n * i:n * i + n], **kwargs)

    elif array.ndim == 4:
        m = int(array.shape[1] / n)
        resid = array.shape[1] % n
        w = array.shape[0]
        y = array.shape[2]
        x = array.shape[3]
        arr = np.empty([w, m, y, x])
        if parallactic is not None:
            angles = np.zeros(m)

        for j in range(w):
            for i in range(m):
                arr[j, i, :, :] = func(array[j, n * i:n * i + n, :, :], axis=0,
                                       **kwargs)
                if parallactic is not None:
                    angles[i] = func(parallactic[n * i:n * i + n], **kwargs)

    if verbose:
        msg = "Cube temporally subsampled by taking the {} of every {} frames"
        print(msg.format(mode, n))
        if resid > 0:
            print("Initial # of frames and window are not multiples ({} "
                  "frames were dropped)".format(resid))
        print("New shape: {}".format(arr.shape))

    if parallactic is not None:
        return arr, angles
    else:
        return arr


def cube_subsample_trimmean(arr, n, m):
    """Perform a trimmed mean combination every m frames in a cube.

    Details in [BRA13]_.

    Parameters
    ----------
    arr : numpy ndarray
        Cube.
    n : int
        Sets the discarded values at high and low ends. When n = N is the same
        as taking the mean, when n = 1 is like taking the median.
    m : int
        Window from the trimmed mean.

    Returns
    -------
    arr_view : numpy ndarray
        Output array, cube combined.
    """
    if arr.ndim != 3:
        raise TypeError('The input array is not a cube or 3d array')
    num = int(arr.shape[0]/m)
    res = int(arr.shape[0] % m)
    y = arr.shape[1]
    x = arr.shape[2]
    arr2 = np.empty([num+2, y, x])
    for i in range(num):
        arr2[0] = cube_collapse(arr[:m, :, :], 'trimmean', n)
        if i > 0:
            arr2[i] = cube_collapse(arr[m*i:m*i+m, :, :], 'trimmean', n)
    arr2[num] = cube_collapse(arr[-res:, :, :], 'trimmean', n)
    arr_view = arr2[:num+1]        # slicing until m+1 - last index not included
    msg = "Cube temporally subsampled by taking the trimmed mean of every {} "
    msg += "frames"
    print(msg.format(m))
    return arr_view
