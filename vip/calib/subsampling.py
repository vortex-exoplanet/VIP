#! /usr/bin/env python

"""
Module with pixel and frame subsampling functions.
"""

from __future__ import division

__author__ = 'C. Gomez @ ULg'
__all__ = ['dataset_stacking',
           'frame_resize',
           'cube_pixel_resize',
           'cube_subsample',
           'cube_collapse_trimmean',
           'cube_subsample_trimmean']

import numpy as np
import pandas as pn
import cv2
from datetime import datetime
from ..fits import open_fits

### TODO: check available memory
def dataset_stacking(fi_list, mode, window):
    """ Creates a cube a list of FITS files and stacks them with a given window.
    This function might be heavy on memory usage, depending on the number of
    FITS files to handle, must be used carefully.
    
    Parameters
    ----------
    fi_list : list of str
        FITS filenames list.
    mode : {"mean", "median"}, str optional
        Stacking (temporal subsampling) mode.
    window : Int optional
        Window for the frames subsampling.
        
    Returns
    _______
    cube_out : array_like
        Cube with the stacked frames.    
        
    """
    nfits = len(fi_list)
    fiframe = open_fits(fi_list[0], verbose=False)
    y, x = fiframe.shape
    cube = np.empty((nfits, y, x), dtype=np.float64)
    
    for i, filename in enumerate(fi_list):
        frame = open_fits(filename, verbose=False)
        cube[i] = frame

    cube_out = cube_subsample(cube, window, mode, None, False)
    print "Finished {:} stacking with window n = {:}".format(mode, window) 
    return cube_out


def frame_resize(array, scale=0.5, interpolation='bicubic'):
    """ Resizes the input frame with a given scale using OpenCV libraries. 
    Depending on the scale the operation can lead to pixel binning or pixel
    upsampling.
    
    Parameters
    ----------
    array : array_like
        Input 2d array.
    scale : float
        Scale factor along both X and Y axes.
    interpolation : {'bicubic', 'bilinear', 'nearneig'}, optional
        'nneighbor' stands for nearest-neighbor interpolation,
        'bilinear' stands for bilinear interpolation,
        'bicubic' for interpolation over 4x4 pixel neighborhood.
        The 'bicubic' is the default. The 'nearneig' is the fastest method and
        the 'bicubic' the slowest of the three. The 'nearneig' is the poorer
        option for interpolation of noisy astronomical images.
        
    Returns
    -------
    resized : array_like
        Resized 2d array.
        
    """
    if not array.ndim == 2:
        raise TypeError('The input array is not a frame or 2d array.')
    
    if interpolation == 'bilinear':
        intp = cv2.INTER_LINEAR
    elif interpolation == 'bicubic':
        intp= cv2.INTER_CUBIC
    elif interpolation == 'nneighbor':
        intp = cv2.INTER_NEAREST
    else:
        raise TypeError('Interpolation method not recognized.')

    resized = cv2.resize(array, (0,0), fx=scale, fy=scale, interpolation=intp)
    return resized


def cube_pixel_resize(array, scale, interpolation):
    """ Calls frame_resize for resizing the frames of a cube.
    
    Parameters
    ----------
    array : array_like
        Input cube.
    scale : float
        Scale factor along both X and Y axes.
    interpolation : {'bicubic', 'bilinear', 'nearneig'}, optional
        'nneighbor' stands for nearest-neighbor interpolation,
        'bilinear' stands for bilinear interpolation,
        'bicubic' for interpolation over 4x4 pixel neighborhood.
        The 'bicubic' is the default. The 'nearneig' is the fastest method and
        the 'bicubic' the slowest of the three. The 'nearneig' is the poorer
        option for interpolation of noisy astronomical images.
    
    Returns
    -------
    cube : array_like
        Cube with resized frames.
    """
    if not array.ndim == 3:
        raise TypeError('The input array is not a cube or 3d array.')
    
    cube = array.copy()
    for frame in cube:                                                       
        frame = frame_resize(frame, scale=scale, interpolation=interpolation)
    
    print "Done resizing the frames in the cube"
    return cube
    

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
            arr[i, :, :] = np.median(array[:n, :, :], axis=0)
            # first new frame,  mean of first n frames
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


