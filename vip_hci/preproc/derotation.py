#! /usr/bin/env python

"""
Module with frame de-rotation routine for ADI.
"""
__author__ = 'C. Gomez @ ULg'
__all__ = ['cube_derotate',
           'frame_rotate']

import numpy as np
import warnings
try:
    import cv2
    no_opencv = False
except ImportError:
    msg = "Opencv python binding are missing (consult VIP documentation for "
    msg += "Opencv installation instructions). Scikit-image will be used instead."
    warnings.warn(msg, ImportWarning)
    no_opencv = True

from skimage.transform import rotate
from multiprocessing import Pool, cpu_count
import itertools as itt
from ..conf import eval_func_tuple as futup
from ..var import frame_center

data_array = None # holds the (implicitly mem-shared) data array


def frame_rotate(array, angle, imlib='opencv', interpolation='bicubic', cxy=None):
    """ Rotates a frame.
    
    Parameters
    ----------
    array : array_like 
        Input frame, 2d array.
    angle : float
        Rotation angle.
    imlib : {'opencv', 'skimage'}, str optional
        Library used for image transformations. Opencv is faster than ndimage or
        skimage.
    interpolation : {'bicubic', 'bilinear', 'nearneig'}, optional
        'nneighbor' stands for nearest-neighbor interpolation,
        'bilinear' stands for bilinear interpolation,
        'bicubic' for interpolation over 4x4 pixel neighborhood.
        The 'bicubic' is the default. The 'nearneig' is the fastest method and
        the 'bicubic' the slowest of the three. The 'nearneig' is the poorer
        option for interpolation of noisy astronomical images.
    cxy : float, optional
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

    y, x = array.shape
    
    if not cxy:  
        cy, cx = frame_center(array)
    else:
        cx, cy = cxy

    if imlib not in ['skimage', 'opencv']:
        raise ValueError('Imlib not recognized, try opencv or ndimage')

    if imlib=='skimage' or no_opencv:
        if interpolation == 'bilinear':
            order = 1
        elif interpolation == 'bicubic':
            order = 3
        elif interpolation == 'nearneig':
            order = 0
        else:
            raise TypeError('Interpolation method not recognized.')

        min_val = np.min(array)
        im_temp = array - min_val
        max_val = np.max(im_temp)
        im_temp /= max_val

        array_out = rotate(im_temp, angle, order=order, center=cxy, cval=np.nan)

        array_out *= max_val
        array_out += min_val
        array_out = np.nan_to_num(array_out)

    else:
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
    
    
def cube_derotate(array, angle_list, imlib='opencv', interpolation='bicubic',
                  cxy=None, nproc=1):
    """ Rotates an cube (3d array or image sequence) providing a vector or
    corrsponding angles. Serves for rotating an ADI sequence to a common north
    given a vector with the corresponding parallactic angles for each frame. By
    default bicubic interpolation is used (opencv).
    
    Parameters
    ----------
    array : array_like 
        Input 3d array, cube.
    angle_list : list
        Vector containing the parallactic angles.
    imlib : {'opencv', 'skimage'}, str optional
        Library used for image transformations. Opencv is usually faster than
        ndimage or skimage.
    interpolation : {'bicubic', 'bilinear', 'nearneig'}, optional
        'nneighbor' stands for nearest-neighbor interpolation,
        'bilinear' stands for bilinear interpolation,
        'bicubic' for interpolation over 4x4 pixel neighborhood.
        The 'bicubic' is the default. The 'nearneig' is the fastest method and
        the 'bicubic' the slowest of the three. The 'nearneig' is the poorer
        option for interpolation of noisy astronomical images.
    cxy : tuple of int, optional
        Coordinates X,Y  of the point with respect to which the rotation will be 
        performed. By default the rotation is done with respect to the center 
        of the frames, as it is returned by the function vip_hci.var.frame_center. 
    collapse : {'median','mean'}
        Way of collapsing the derotated cube.
        
    Returns
    -------
    array_der : array_like
        Resulting cube with de-rotated frames.
        
    """
    if not array.ndim == 3:
        raise TypeError('Input array is not a cube or 3d array.')
    array_der = np.empty_like(array)
    n_frames = array.shape[0]
    
    if not cxy:
        cy, cx = frame_center(array[0])
        cxy = (cx, cy)

    if not nproc: nproc = int((cpu_count() / 2))

    if nproc==1:
        for i in range(n_frames):
            array_der[i] = frame_rotate(array[i], -angle_list[i], imlib=imlib,
                                        interpolation=interpolation, cxy=cxy)
    elif nproc>1:
        global data_array
        data_array = array

        pool = Pool(processes=int(nproc))
        res = pool.map(futup, itt.izip(itt.repeat(_cube_rotate_mp),
                                       range(n_frames), itt.repeat(angle_list),
                                       itt.repeat(imlib),
                                       itt.repeat(interpolation),
                                       itt.repeat(cxy)))
        pool.close()
        array_der = np.array(res)

    return array_der


def _cube_rotate_mp(num_fr, angle_list, imlib, interpolation, cxy):
    framerot = frame_rotate(data_array[num_fr], -angle_list[num_fr],
                            imlib, interpolation, cxy)
    return framerot
