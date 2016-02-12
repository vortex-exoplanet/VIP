#! /usr/bin/env python

"""
Module with cosmetics procedures. Contains the function for bad pixel fixing. 
Also functions for cropping cubes. 
"""

from __future__ import division

__author__ = 'C. Gomez @ ULg'
__all__ = ['cube_crop_frames',
           'cube_drop_frames',
           'frame_crop']


import numpy as np
from ..var import frame_center, get_square


def cube_crop_frames(array, size, xy=None, verbose=True):                          
    """Crops frames in a cube (3d array). If size is an even value it'll be 
    increased by one to make it odd.
    
    Parameters
    ----------
    array : array_like 
        Input 3d array, cube.
    size : int
        Size of the desired central sub-array in each frame.
    xy : tuple of ints
        X, Y coordinates of new frame center. If you are getting the 
        coordinates from ds9 subtract 1, python has 0-based indexing.
    verbose : {True, False}, bool optional
        If True message of completion is showed.
    
    Returns
    -------
    array_view : array_like
        Cube with cropped frames.
        
    """
    if not array.ndim == 3:
        raise TypeError('\nArray is not 3d, not a cube.')
    
    if size%2!=0:  size -= 1
    if size >= array.shape[1]:
        raise ValueError('The crop size is equal or bigger than the actual size.')
    
    # wing is added to the sides of the subframe center.
    wing = size/2           
    
    if xy is not None:
        cenx, ceny = xy 
        # Note the +1 when closing the interval (python doesn't include the 
        # endpoint)
        array_view = array[:,ceny-wing:ceny+wing+1,cenx-wing:cenx+wing+1].copy()
        if verbose:
            print 
            msg = "Cube cropped; new size [{:},{:},{:}] centered at ({:},{:})."
            print msg.format(array.shape[0], size+1, size+1, ceny, cenx)
    else:  
        cy, cx = frame_center(array[0], verbose=False)
        array_view = array[:, cy-wing:cy+wing+1, cx-wing:cx+wing+1].copy()
        if verbose:
            print
            msg = "Cube cropped with new size [{:},{:},{:}]."
            print msg.format(array.shape[0], size+1, size+1)

    return array_view


def frame_crop(array, size, ceny=None, cenx=None, verbose=True):                         
    """Crops frame (2d array). Wrapper of function vortex.var.shapes.get_square.
    If size is an even value it'll be increased by one to make it odd.
    
    Parameters
    ----------
    array : array_like
        Input frame.
    size : int, odd
        Size of the subframe.
    ceny, cenx : int
        Coordinates of the center of the subframe.
    verbose : {True, False}, bool optional
        If True message of completion is showed.
        
    Returns
    -------
    array_view : array_like
        Sub array.
        
    """
    if not array.ndim == 2:
        raise TypeError('Array is not a frame or 2d array')
    
    if not ceny and not cenx:
        ceny, cenx = frame_center(array, verbose=False)
    array_view = get_square(array, size, ceny, cenx)    
    
    if verbose:
        print
        print "Done frame cropping"
    return array_view


def cube_drop_frames(array, n, m):
    """Discards frames at the beginning or the end of a cube (axis 0).
    
    Parameters
    ----------
    array : array_like 
        Input 3d array, cube.
    n : int
        Index of the first frame to be kept. Frames before this one are dropped.
    m : int
        Index of the last frame to be kept. Frames after this one are dropped.
    
    Returns
    -------
    array_view : array_like
        Cube with new size (axis 0).
        
    """
    if m>array.shape[0]:
        raise TypeError('End index must be smaller than the # of frames')
    
    array_view = array[n:m+1, :, :].copy()
    print
    print "Done discarding frames from FITS-Cube"
    return array_view  


def frame_remove_stripes(array):
    """ Removes unwanted stripe artifact in frames with non-perfect bias or sky
    subtraction. Encountered this case on an LBT data cube.
    """
    lines = array[:50]
    lines = np.vstack((lines, array[-50:]))
    mean = lines.mean(axis=0)
    for i in range(array.shape[1]):
        array[:,i] = array[:,i] - mean[i]
    return array

