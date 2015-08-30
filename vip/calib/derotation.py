#! /usr/bin/env python

"""
Module with frame de-rotation routine for ADI.
"""
__author__ = 'C. Gomez @ ULg'
__all__ = ['cube_derotate',
           'frame_rotate']

import numpy as np
import cv2
from ..var import frame_center


def frame_rotate(array, angle, interpolation='bicubic', cy=None, cx=None):
    """ Rotates a frame.
    
    Parameters
    ----------
    array : array_like 
        Input frame, 2d array.
    angle : float
        Rotation angle.
    interpolation : {'bicubic', 'bilinear', 'nearneig'}, optional
        'nneighbor' stands for nearest-neighbor interpolation,
        'bilinear' stands for bilinear interpolation,
        'bicubic' for interpolation over 4x4 pixel neighborhood.
        The 'bicubic' is the default. The 'nearneig' is the fastest method and
        the 'bicubic' the slowest of the three. The 'nearneig' is the poorer
        option for interpolation of noisy astronomical images.
    cy, cx : float, optional
        Coordinates X,Y  of the point with respect to which the rotation will be
        performed. By default the rotation is done with respect to the center 
        of the frame; central pixel if frame has odd size.
        
    Returns
    -------
    array_out : array_like
        Resulting frame.
        
    """
    if not array.ndim == 2:
        raise TypeError('Input array is not a frame or 2d array.')
    array = np.float32(array)
    y, x = array.shape
    
    if not cy and not cx:  cy, cx = frame_center(array)
    
    if interpolation == 'bilinear':
        intp = cv2.INTER_LINEAR
    elif interpolation == 'bicubic':
        intp= cv2.INTER_CUBIC
    elif interpolation == 'nearneig':
        intp = cv2.INTER_NEAREST
    else:
        raise TypeError('Interpolation method not recognized.')
    
    M = cv2.getRotationMatrix2D((cx,cy), angle, 1)
    array_out = cv2.warpAffine(array.astype(np.float32), M, (x, y), flags=intp)
             
    return array_out
    
    
def cube_derotate(array, angle_list, cy=None, cx=None, collapse='median'):
    """ Rotates an ADI cube to a common north given a vector with the 
    corresponding parallactic angles for each frame of the sequence. By default
    bicubic interpolation is used (opencv). 
    
    Parameters
    ----------
    array : array_like 
        Input 3d array, cube.
    angle_list : list or 1D-array
        Vector containing the parallactic angles.
    cy, cx : float, optional
        Coordinates X,Y  of the point with respect to which the rotation will be
        performed. By default the rotation is done with respect to the center 
        of the frames; central pixel if the frames have odd size.
    collapse : string, {'median','mean'}, optional
        Way of collapsing the derotated cube.       
 
    Returns
    -------
    array_der : array_like
        Resulting cube with de-rotated frames.
    array_out : array_like
        Collapsed image of the de-rotated cube.
        
    """
    if not array.ndim == 3:
        raise TypeError('Input array is not a cube or 3d array.')
    array_der = np.zeros_like(array) 
    y, x = array[0].shape
    
    if not cy and not cx:  cy, cx = frame_center(array[0])
    
    for i in xrange(array.shape[0]): 
        M = cv2.getRotationMatrix2D((cx,cy), -angle_list[i], 1)
        array_der[i] = cv2.warpAffine(array[i].astype(np.float32), M, (x, y))
            
    if collapse=='median':        
        array_out = np.median(array_der, axis=0)
    elif collapse=='mean':
        array_out = np.mean(array_der, axis=0)
             
    return array_der, array_out
