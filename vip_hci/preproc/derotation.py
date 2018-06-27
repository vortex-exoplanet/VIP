#! /usr/bin/env python

"""
Module with frame de-rotation routine for ADI.
"""
from __future__ import division, print_function

__author__ = 'Carlos Alberto Gomez Gonzalez'
__all__ = ['cube_derotate',
           'frame_rotate']

import numpy as np
import warnings
try:
    import cv2
    no_opencv = False
except ImportError:
    msg = "Opencv python bindings are missing."
    warnings.warn(msg, ImportWarning)
    no_opencv = True

from skimage.transform import rotate
from multiprocessing import Pool, cpu_count
import itertools as itt
from ..conf.utils_conf import eval_func_tuple as EFT
from ..var import frame_center

data_array = None # holds the (implicitly mem-shared) data array


def frame_rotate(array, angle, imlib='opencv', interpolation='lanczos4',
                 cxy=None, border_mode='constant'):
    """ Rotates a frame or 2D array.
    
    Parameters
    ----------
    array : array_like 
        Input image, 2d array.
    angle : float
        Rotation angle.
    imlib : {'opencv', 'skimage'}, str optional
        Library used for image transformations. Opencv is faster than
        Skimage or scipy.ndimage.
    interpolation : str, optional
        For Skimage the options are: 'nearneig', bilinear', 'bicuadratic',
        'bicubic', 'biquartic' or 'biquintic'. The 'nearneig' interpolation is
        the fastest and the 'biquintic' the slowest. The 'nearneig' is the
        poorer option for interpolation of noisy astronomical images.
        For Opencv the options are: 'nearneig', 'bilinear', 'bicubic' or
        'lanczos4'. The 'nearneig' interpolation is the fastest and the
        'lanczos4' the slowest and more accurate. 'lanczos4' is the default for
        Opencv and 'biquartic' for Skimage.
    cxy : float, optional
        Coordinates X,Y  of the point with respect to which the rotation will be 
        performed. By default the rotation is done with respect to the center 
        of the frame; central pixel if frame has odd size.
    border_mode : {'constant', 'edge', 'symmetric', 'reflect', 'wrap'}, str optional
        Pixel extrapolation method for handling the borders. 'constant' for
        padding with zeros. 'edge' for padding with the edge values of the
        image. 'symmetric' for padding with the reflection of the vector
        mirrored along the edge of the array. 'reflect' for padding with the
        reflection of the vector mirrored on the first and last values of the
        vector along each axis. 'wrap' for padding with the wrap of the vector
        along the axis (the first values are used to pad the end and the end
        values are used to pad the beginning). Default is 'constant'.
        
    Returns
    -------
    array_out : array_like
        Resulting frame.
        
    """
    if array.ndim != 2:
        raise TypeError('Input array is not a frame or 2d array')

    y, x = array.shape
    
    if cxy is None:
        cy, cx = frame_center(array)
    else:
        cx, cy = cxy

    if imlib == 'skimage':
        if interpolation == 'nearneig':
            order = 0
        elif interpolation == 'bilinear':
            order = 1
        elif interpolation == 'bicuadratic':
            order = 2
        elif interpolation == 'bicubic':
            order = 3
        elif interpolation == 'biquartic' or interpolation == 'lanczos4':
            order = 4
        elif interpolation == 'biquintic':
            order = 5
        else:
            raise ValueError('Skimage interpolation method not recognized')

        if border_mode not in ['constant', 'edge', 'symmetric', 'reflect',
                               'wrap']:
            raise ValueError('Skimage `border_mode` not recognized.')

        min_val = np.min(array)
        im_temp = array - min_val
        max_val = np.max(im_temp)
        im_temp /= max_val

        array_out = rotate(im_temp, angle, order=order, center=cxy, cval=np.nan,
                           mode=border_mode)

        array_out *= max_val
        array_out += min_val
        array_out = np.nan_to_num(array_out)

    elif imlib == 'opencv':
        if no_opencv:
            msg = 'Opencv python bindings cannot be imported. Install opencv or'
            msg += ' set imlib to skimage'
            raise RuntimeError(msg)

        if interpolation == 'bilinear':
            intp = cv2.INTER_LINEAR
        elif interpolation == 'bicubic':
            intp= cv2.INTER_CUBIC
        elif interpolation == 'nearneig':
            intp = cv2.INTER_NEAREST
        elif interpolation == 'lanczos4':
            intp = cv2.INTER_LANCZOS4
        else:
            raise ValueError('Opencv interpolation method not recognized')

        if border_mode == 'constant':
            bormo = cv2.BORDER_CONSTANT  # iiiiii|abcdefgh|iiiiiii
        elif border_mode == 'edge':
            bormo = cv2.BORDER_REPLICATE  # aaaaaa|abcdefgh|hhhhhhh
        elif border_mode == 'symmetric':
            bormo = cv2.BORDER_REFLECT  # fedcba|abcdefgh|hgfedcb
        elif border_mode == 'reflect':
            bormo = cv2.BORDER_REFLECT_101  # gfedcb|abcdefgh|gfedcba
        elif border_mode == 'wrap':
            bormo = cv2.BORDER_WRAP  # cdefgh|abcdefgh|abcdefg
        else:
            raise ValueError('Opencv `border_mode` not recognized.')

        M = cv2.getRotationMatrix2D((cx,cy), angle, 1)
        array_out = cv2.warpAffine(array.astype(np.float32), M, (x, y),
                                   flags=intp, borderMode=bormo)

    else:
        raise ValueError('Image transformation library not recognized')
             
    return array_out
    
    
def cube_derotate(array, angle_list, imlib='opencv', interpolation='lanczos4',
                  cxy=None, nproc=1, border_mode='constant'):
    """ Rotates an cube (3d array or image sequence) providing a vector or
    corresponding angles. Serves for rotating an ADI sequence to a common north
    given a vector with the corresponding parallactic angles for each frame. By
    default bicubic interpolation is used (opencv).
    
    Parameters
    ----------
    array : array_like 
        Input 3d array, cube.
    angle_list : list
        Vector containing the parallactic angles.
    imlib : str, optional
        See the documentation of the ``vip_hci.preproc.frame_rotate`` function.
    interpolation : str, optional
        See the documentation of the ``vip_hci.preproc.frame_rotate`` function.
    cxy : tuple of int, optional
        Coordinates X,Y  of the point with respect to which the rotation will be 
        performed. By default the rotation is done with respect to the center 
        of the frames, as it is returned by the function
        vip_hci.var.frame_center.
    nproc : int, optional
        Whether to rotate the frames in the sequence in a multi-processing
        fashion. Only useful if the cube is significantly large (frame size and
        number of frames).
    border_mode : str, optional
        See the documentation of the ``vip_hci.preproc.frame_rotate`` function.
        
    Returns
    -------
    array_der : array_like
        Resulting cube with de-rotated frames.
        
    """
    if array.ndim != 3:
        raise TypeError('Input array is not a cube or 3d array.')
    n_frames = array.shape[0]

    if nproc is None:
        nproc = cpu_count() // 2        # Hyper-threading doubles the # of cores

    if nproc == 1:
        array_der = np.zeros_like(array)
        for i in range(n_frames):
            array_der[i] = frame_rotate(array[i], -angle_list[i], imlib=imlib,
                                        interpolation=interpolation, cxy=cxy,
                                        border_mode=border_mode)
    elif nproc > 1:
        global data_array
        data_array = array

        pool = Pool(processes=nproc)
        res = pool.map(EFT, zip(itt.repeat(_cube_rotate_mp), range(n_frames),
                                itt.repeat(angle_list), itt.repeat(imlib),
                                itt.repeat(interpolation), itt.repeat(cxy),
                                itt.repeat(border_mode)))
        pool.close()
        array_der = np.array(res)

    return array_der


def _cube_rotate_mp(num_fr, angle_list, imlib, interpolation, cxy, border_mode):
    framerot = frame_rotate(data_array[num_fr], -angle_list[num_fr],
                            imlib, interpolation, cxy, border_mode)
    return framerot


def _find_indices_adi(angle_list, frame, thr, nframes=None, out_closest=False,
                      truncate=False, max_frames=200):
    """ Returns the indices to be left in frames library for annular ADI median
    subtraction, LOCI or annular PCA.

    # TODO: find a more pythonic way to to this!

    Parameters
    ----------
    angle_list : array_like, 1d
        Vector of parallactic angle (PA) for each frame.
    frame : int
        Index of the current frame for which we are applying the PA threshold.
    thr : float
        PA threshold.
    nframes : int or None, optional
        Number of indices to be left. For annular ADI median subtraction,
        where we keep the closest frames (after the PA threshold). If None then
        all the indices are returned (after the PA threshold).
    out_closest : bool, optional
        If True then the function returns the indices of the 2 closest frames.
    truncate : bool, optional
        Useful for annular PCA, when we want to discard too far away frames and
        avoid increasing the computational cost.
    max_frames : int, optional
        Max frames to leave if ``truncate`` is True.

    Returns
    -------
    indices : array_like, 1d
        Vector with the indices left.

    If ``out_closest`` is True then the function returns instead:
    index_prev, index_foll
    """
    n = angle_list.shape[0]
    index_prev = 0
    index_foll = frame
    for i in range(0, frame):
        if np.abs(angle_list[frame] - angle_list[i]) < thr:
            index_prev = i
            break
        else:
            index_prev += 1
    for k in range(frame, n):
        if np.abs(angle_list[k] - angle_list[frame]) > thr:
            index_foll = k
            break
        else:
            index_foll += 1

    if out_closest:
        return index_prev, index_foll - 1
    else:
        if nframes is not None:
            # For annular ADI median subtraction, returning n_frames closest
            # indices (after PA thresholding)
            window = nframes // 2
            ind1 = index_prev - window
            ind1 = max(ind1, 0)
            ind2 = index_prev
            ind3 = index_foll
            ind4 = index_foll + window
            ind4 = min(ind4, n)
            indices = np.array(list(range(ind1, ind2)) +
                               list(range(ind3, ind4)))
        else:
            # For annular PCA, returning all indices (after PA thresholding)
            half1 = range(0, index_prev)
            half2 = range(index_foll, n)

            # This truncation is done on the annuli after 10*FWHM and the goal
            # is to keep min(num_frames/2, 200) in the library after discarding
            # those based on the PA threshold
            if truncate:
                thr = min(n//2, max_frames)
                if frame < thr:
                    half1 = range(max(0, index_prev - thr // 2), index_prev)
                    half2 = range(index_foll,
                                  min(index_foll + thr - len(half1), n))
                else:
                    half2 = range(index_foll, min(n, thr // 2 + index_foll))
                    half1 = range(max(0, index_prev - thr + len(half2)),
                                  index_prev)
            indices = np.array(list(half1) + list(half2))

        return indices


def _compute_pa_thresh(ann_center, fwhm, delta_rot=1):
    """ Computes the parallactic angle theshold[degrees]
    Replacing approximation: delta_rot * (fwhm/ann_center) / np.pi * 180
    """
    return np.rad2deg(2 * np.arctan(delta_rot * fwhm / (2 * ann_center)))


def _define_annuli(angle_list, ann, n_annuli, fwhm, radius_int, annulus_width,
                   delta_rot, n_segments, verbose):
    """ Function that defines the annuli geometry using the input parameters.
    Returns the parallactic angle threshold, the inner radius and the annulus
    center for each annulus.
    """
    if ann == n_annuli - 1:
        inner_radius = radius_int + (ann * annulus_width - 1)
    else:
        inner_radius = radius_int + ann * annulus_width
    ann_center = inner_radius + (annulus_width / 2)
    pa_threshold = _compute_pa_thresh(ann_center, fwhm, delta_rot)

    mid_range = np.abs(np.amax(angle_list) - np.amin(angle_list)) / 2
    if pa_threshold >= mid_range - mid_range * 0.1:
        new_pa_th = float(mid_range - mid_range * 0.1)
        if verbose:
            msg = '\tPA threshold {:.2f} is too big, will be set to {:.2f}'
            print(msg.format(pa_threshold, new_pa_th))
        pa_threshold = new_pa_th

    if verbose:
        msg2 = '\tAnnulus {}, PA thresh = {:.2f}, Inn radius = {:.2f}, '
        msg2 += 'Ann center = {:.2f}, N segments = {} '
        print(msg2.format(ann+1, pa_threshold, inner_radius, ann_center,
                          n_segments))
    return pa_threshold, inner_radius, ann_center