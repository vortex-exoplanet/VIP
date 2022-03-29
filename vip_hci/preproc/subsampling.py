#! /usr/bin/env python

"""
Module with pixel and frame subsampling functions.
"""

__author__ = 'Carlos Alberto Gomez Gonzalez, Valentin Christiaens'
__all__ = ['cube_collapse',
           'cube_subsample',
           'cube_subsample_trimmean']

import numpy as np


def cube_collapse(cube, mode='median', n=50, w=None):
    """ Collapses a cube into a frame (3D array -> 2D array) depending on the
    parameter ``mode``. It's possible to perform a trimmed mean combination of
    the frames based on description in Brandt+ 2012.
    
    Parameters
    ----------
    cube : numpy ndarray
        Cube.
    mode : {'median', 'mean', 'sum', 'trimmean', 'max', 'wmean'}, str optional
        Sets the way of collapsing the images in the cube.
        'wmean' stands for weighted mean and requires weights w to be provided.
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
    if arr.ndim != 3:
        raise TypeError('The input array is not a cube or 3d array.')
    
    if mode == 'wmean':
        if w is None:
            raise ValueError("Weights have to be provided for weighted mean mode")
        if len(w) != cube.shape[0]:
            raise TypeError("Weights need same length as cube")
        if isinstance(w,list):
            w = np.array(w)
    
    if mode == 'mean':
        frame = np.nanmean(arr, axis=0)
    elif mode == 'median':
        frame = np.nanmedian(arr, axis=0)
    elif mode == 'sum':
        frame = np.nansum(arr, axis=0)
    elif mode == 'max':
        frame = np.nanmax(arr, axis=0)
    elif mode == 'trimmean':
        N = arr.shape[0]
        k = (N - n)//2                                                          
        if N%2 != n%2:
            n+=1
        frame = np.empty_like(arr[0])                                    
        for index, _ in np.ndenumerate(arr[0]):
            sort = np.sort(arr[:, index[0], index[1]])
            frame[index] = np.nanmean(sort[k:k+n])
    elif mode == 'wmean':
        arr[np.where(np.isnan(arr))]=0 # to avoid product with nan
        frame = np.inner(w, np.moveaxis(arr,0,-1))
        
    return frame


def cube_subsample(array, n, mode="mean", parallactic=None, verbose=True):
    """Mean/Median combines frames in 3d or 4d cube with window ``n``.
    
    Parameters
    ----------
    array : numpy ndarray
        Input 3d array, cube.
    n : int
        Window for mean/median.
    mode : {'mean','median'}, optional
        Switch for choosing mean or median.
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
        
    if mode == 'median':
        func = np.median
    elif mode == 'mean':
        func = np.mean
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
            arr[i, :, :] = func(array[n * i:n * i + n, :, :], axis=0)
            if parallactic is not None:
                angles[i] = func(parallactic[n * i:n * i + n])

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
                arr[j, i, :, :] = func(array[j, n * i:n * i + n, :, :], axis=0)
                if parallactic is not None:
                    angles[i] = func(parallactic[n * i:n * i + n])

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
    """Performs a trimmed mean combination every m frames in a cube. Based on 
    description in Brandt+ 2012.
    
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
    res = int(arr.shape[0]%m)                                                       
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