#! /usr/bin/env python

"""
Module with pixel and frame subsampling functions.
"""

from __future__ import division

__author__ = 'C. Gomez @ ULg'
__all__ = ['cube_subsample',
           'cube_collapse_trimmean',
           'cube_subsample_trimmean']

import numpy as np
import pandas as pn
    

def cube_subsample(array, n, mode="mean", parallactic=None, verbose=False):
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
        PA.
    """
    if not array.ndim == 3:
        raise TypeError('The input array is not a cube or 3d array.')
    m = int(array.shape[0]/n)                                                        
    y = array.shape[1]
    x = array.shape[2]
    arr = np.empty([m, y, x]) 
    if mode == 'median':
        for i in xrange(m):                                                  
            arr[i, :, :] = np.median(array[:n, :, :], axis=0)                   # first new frame,  mean of first n frames
            if i >= 1:
                arr[i, :, :] = np.median(array[n*i:n*i+n, :, :], axis=0) 
    if mode=='mean':
        for i in xrange(m):                                                  
            arr[i, :, :] = np.mean(array[:n, :, :], axis=0)                         
            if i >= 1:
                arr[i, :, :] = np.mean(array[n*i:n*i+n, :, :], axis=0)
    else:  
        raise ValueError('Mode should be either Mean or Median.')
    if parallactic is not None:
        angles = parallactic.byteswap().newbyteorder()
        angles = pn.rolling_mean(angles, n, center=True)
        angles = pn.DataFrame(angles)
        angles = pn.DataFrame.fillna(angles, method='pad')
        angles = pn.DataFrame.fillna(angles, method='bfill')
        angles = np.array(angles)
        angles = angles[0:m*n:n]
    if array.shape[0]/n % 1 != 0:
        print '\nInitial # of frames and window are not multiples. A few frames were dropped.'   
    if verbose:
        print "Done {:} over FITS-Cube with window n = {:}".format(mode ,n)                            
    if parallactic is not None:
        return arr, angles
    else:
        return arr


def cube_collapse_trimmean(arr, n):
    """Performs a trimmed mean combination of the frames in a cube. Based on 
    description in Brandt+ 2012.
    
    Parameters
    ----------
    arr : array_like
        Cube.
    n : int
        Sets the discarded values at high and low ends. When n = N is the same
        as taking the mean, when n = 1 is like taking the median.
        
    Returns
    -------
    arr2 : array_like
        Output array, cube combined. 
    """
    if not arr.ndim == 3:
        raise TypeError('The input array is not a cube or 3d array.')
    N = arr.shape[0]
    if N % 2 == 0:
        k = (N - n)//2
    else:
        k = (N - n)/2                                                               
    arr2 = np.empty_like(arr[0])                                    
    for index, _ in np.ndenumerate(arr[0]):
        sort = np.sort(arr[:,index[0],index[1]])
        arr2[index] = np.mean(sort[k:N-k])
    return arr2


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
    for i in xrange(num):                                                  
        arr2[0] = cube_collapse_trimmean(arr[:m, :, :], n)                         
        if i > 0:
            arr2[i] = cube_collapse_trimmean(arr[m*i:m*i+m, :, :], n)
    arr2[num] = cube_collapse_trimmean(arr[-res:, :, :], n)     
    arr_view = arr2[:num+1]                                                      # slicing until m+1 - last index not included
    print "\nDone trimmed mean over FITS-Cube with window m=" + str(m)
    return arr_view


