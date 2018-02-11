#! /usr/bin/env python

"""
Module with frame resampling/rescaling functions.
"""
__author__ = 'Carlos Alberto Gomez Gonzalez, V. Christiaens @ ULg'
__all__ = ['frame_px_resampling',
           'cube_px_resampling',
           'frame_rescaling',
           'cube_rescaling',
           'check_scal_vector']

import numpy as np
import warnings
try:
    import cv2
    no_opencv = False
except ImportError:
    msg = "Opencv python bindings are missing."
    warnings.warn(msg, ImportWarning)
    no_opencv = True

from skimage.transform import rescale
from scipy.ndimage.interpolation import geometric_transform
from ..var import frame_center


def frame_px_resampling(array, scale, imlib='opencv', interpolation='lanczos4',
                        scale_y=None, scale_x=None, keep_odd=True,
                        full_output=False):
    """ Resamples the pixels of a frame wrt to the center, changing the size
    of the frame. If ``scale`` < 1 then the frame is downsampled and if
    ``scale`` > 1 then its pixels are upsampled.

    Parameters
    ----------
    array : array_like 
        Input frame, 2d array.
    scale : int or float
        Scale factor for upsampling or downsampling the frame.
    imlib : {'opencv', 'skimage'}, str optional
        Library used for image transformations. Opencv is faster than ndimage or
        skimage.
    interpolation : str, optional
        For 'skimage' library: 'nearneig', bilinear', 'bicuadratic', 'bicubic',
        'biquartic', 'biquintic'. The 'nearneig' interpolation is the fastest
        and the 'biquintic' the slowest. The 'nearneig' is the poorer
        option for interpolation of noisy astronomical images.
        For 'opencv' library: 'nearneig', 'bilinear', 'bicubic', 'lanczos4'.
        The 'nearneig' interpolation is the fastest and the 'lanczos4' the
        slowest and accurate. 'lanczos4' is the default.
    scale_y : int or float, opt
        Scale factor for upsampling or downsampling the frame along y only. If 
        provided, it takes priority on scale parameter.
    scale_x : int or float, opt
        Scale factor for upsampling or downsampling the frame along x only. If 
        provided, it takes priority on scale parameter.
    keep_odd: bool, opt
        Will slightly modify the scale factor in order for the final frame size
        to keep y and x sizes odd. This keyword does nothing if the input array
        has even dimensions.
    full_output: bool, opt
        If True, it will also return the scale factor (slightly modified if
        ``keep_odd`` was set to True).

    Returns
    -------
    array_resc : array_like 
        Output resampled frame.
    """
    if not array.ndim == 2:
        raise TypeError('Input array is not a frame or 2d array.')

    if scale_y is None: scale_y = scale
    if scale_x is None: scale_x = scale

    ny = array.shape[0]
    nx = array.shape[1]

    if keep_odd:
        if ny % 2 != 0:  # original y size is odd?
            new_ny = ny * scale_y  # check final size
            if not new_ny > 0.5:
                raise ValueError(
                    "scale_y is too small; resulting array would be 0 size.")
            # final size is decimal? => make it integer
            if new_ny % 1 > 0.5:
                scale_y = (new_ny + 1 - (new_ny % 1)) / ny
            elif new_ny % 1 > 0:
                scale_y = (new_ny - (new_ny % 1)) / ny
            new_ny = ny * scale_y
            # final size is even?
            if new_ny % 2 == 0:
                if scale_y > 1:  # if upscaling => go to closest odd with even-1
                    scale_y = float(new_ny - 1) / ny
                else:
                    scale_y = float(new_ny + 1) / ny
                    # if downscaling => go to closest odd with even+1 (reversible)

        if nx % 2 != 0:  # original x size is odd?
            new_nx = nx * scale_x  # check final size
            if not new_nx > 0.5:
                raise ValueError(
                    "scale_x is too small; resulting array would be 0 size.")
            # final size is decimal? => make it integer
            if new_nx % 1 > 0.5:
                scale_x = (new_nx + 1 - (new_nx % 1)) / nx
            elif new_nx % 1 > 0:
                scale_x = (new_nx - (new_nx % 1)) / nx
            new_nx = nx * scale_x
            # final size is even?
            if new_nx % 2 == 0:
                if scale_x > 1:  # if upscaling => go to closest odd with even-1
                    scale_x = float(new_nx - 1) / nx
                else:  # if downscaling => go to closest odd with even+1 (reversible)
                    scale_x = float(new_nx + 1) / nx

    if imlib == 'skimage':
        if interpolation == 'nearneig':
            order = 0
        elif interpolation == 'bilinear':
            order = 1
        elif interpolation == 'bicuadratic':
            order = 2
        elif interpolation == 'bicubic':
            order = 3
        elif interpolation == 'biquartic':
            order = 4
        elif interpolation == 'biquintic':
            order = 5
        else:
            raise TypeError('Skimage interpolation method not recognized.')

        min_val = np.min(array)
        im_temp = array - min_val
        max_val = np.max(im_temp)
        im_temp /= max_val

        array_resc = rescale(im_temp, scale=(scale_y, scale_x), order=order)
        array_resc *= max_val
        array_resc += min_val

    elif imlib == 'opencv':
        if no_opencv:
            msg = 'Opencv python bindings cannot be imported. Install opencv or'
            msg += ' set imlib to skimage.'
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
            raise TypeError('Opencv interpolation method not recognized.')

        array_resc = cv2.resize(array.astype(np.float32), (0,0), fx=scale_x,
                                fy=scale_y, interpolation=intp)

    else:
        raise ValueError('Image transformation library not recognized.')

    array_resc /= (scale_y * scale_x)

    if full_output:
        if scale_y == scale_x:
            scale = scale_y
        else:
            scale = (scale_y, scale_x)
        return array_resc, scale
    else:
        return array_resc



def cube_px_resampling(array, scale, imlib='opencv', interpolation='lanczos4',
                       scale_y=None, scale_x=None):
    """ Wrapper of ``frame_px_resampling`` for resampling the frames of a cube
    with a single scale factor. Useful when we need to upsample (upscaling) or
    downsample (pixel binning) a set of frames, e.g. an ADI cube.
    
    Parameters
    ----------
    array : array_like 
        Input frame, 2d array.
    scale : float
        Scale factor for upsampling or downsampling the frames in the cube.
    imlib : str optional
        See the documentation of the ``vip_hci.preproc.frame_px_resampling``
        function.
    interpolation : str, optional
        See the documentation of the ``vip_hci.preproc.frame_px_resampling``
        function.
    scale_y : float
        Scale factor for upsampling or downsampling the frame along y. If 
        provided, it takes priority on scale parameter.
    scale_x : float
        Scale factor for upsampling or downsampling the frame along x. If 
        provided, it takes priority on scale parameter.
        
    Returns
    -------
    array_resc : array_like 
        Output cube with resampled frames.

    Notes
    -----
    Be aware that the interpolation used for the pixel rescaling can create
    negative values at the transition from a background composed mostly of zeros
    to a positive valued patch (e.g. an injected PSF template).

    """
    if not array.ndim == 3:
        raise TypeError('Input array is not a cube or 3d array.')
    
    if scale_y is None: scale_y = scale
    if scale_x is None: scale_x = scale

    array_resc = []
    for i in range(array.shape[0]):
        array_resc.append(frame_px_resampling(array[i], scale=scale, imlib=imlib,
                                              interpolation=interpolation,
                                              scale_y=scale_y, scale_x=scale_x))

    return np.array(array_resc)


### TODO: Merge with ``frame_px_resampling``?
def frame_rescaling(array, ref_y=None, ref_x=None, scale=1.0, imlib='opencv',
                    interpolation='lanczos4', scale_y=None, scale_x=None):
    """ Function to rescale a frame by a factor ``scale``, wrt a reference point
    which by default is the center of the frame (typically the exact location
    of the star). However, it keeps the same dimensions.
    
    Parameters
    ----------
    array : array_like 
        Input frame, 2d array.
    ref_y, ref_x : float, optional
        Coordinates X,Y  of the point with respect to which the rotation will be
        performed. By default the rescaling is done with respect to the center
        of the frame; central pixel if frame has odd size.
    scale : float
        Scaling factor. If > 1, it will upsample the input array equally along y
        and x by this factor.      
    imlib : {'opencv', 'ndimage'}, str optional
        Library used for image transformations. Opencv is faster than ndimage or
        skimage.
    interpolation : str, optional
        For 'ndimage' library: 'nearneig', bilinear', 'bicuadratic', 'bicubic',
        'biquartic', 'biquintic'. The 'nearneig' interpolation is the fastest
        and the 'biquintic' the slowest. The 'nearneig' is the poorer
        option for interpolation of noisy astronomical images.
        For 'opencv' library: 'nearneig', 'bilinear', 'bicubic', 'lanczos4'.
        The 'nearneig' interpolation is the fastest and the 'lanczos4' the
        slowest and accurate. 'lanczos4' is the default.
    scale_y : float
        Scaling factor only for y axis. If provided, it takes priority on scale
        parameter.
    scale_x : float
        Scaling factor only for x axis. If provided, it takes priority on scale
        parameter.

    Returns
    -------
    array_out : array_like
        Resulting frame.
    """
    def _scale_func(output_coords,ref_y=0,ref_x=0, scaling=1.0, scaling_y=None,
                    scaling_x=None):    
        """
        For each coordinate point in a new scaled image (output_coords), 
        coordinates in the image before the scaling are returned. This scaling
        function is used within geometric_transform which, for each point in the
        output image, will compute the (spline) interpolated value at the
        corresponding frame coordinates before the scaling.
        """
        if scaling_y is None:
            scaling_y = scaling
        if scaling_x is None:
            scaling_x = scaling
        return (ref_y+((output_coords[0]-ref_y)/scaling_y), 
                ref_x+((output_coords[1]-ref_x)/scaling_x))
    #---------------------------------------------------------------------------
    if not array.ndim == 2:
        raise TypeError('Input array is not a frame or 2d array.')

    if not ref_y and not ref_x:
        ref_y, ref_x = frame_center(array)

    if imlib == 'ndimage':
        outshap = array.shape

        if interpolation == 'nearneig':
            order = 0
        elif interpolation == 'bilinear':
            order = 1
        elif interpolation == 'bicuadratic':
            order = 2
        elif interpolation == 'bicubic':
            order = 3
        elif interpolation == 'biquartic':
            order = 4
        elif interpolation == 'biquintic':
            order = 5
        else:
            raise TypeError('Scipy.ndimage interpolation method not recognized.')

        array_out = geometric_transform(array, _scale_func, order=order,
                                        output_shape=outshap,
                                        extra_keywords={'ref_y':ref_y,
                                                        'ref_x':ref_x,
                                                        'scaling':scale,
                                                        'scaling_y':scale_y,
                                                        'scaling_x':scale_x})

    elif imlib == 'opencv':
        if no_opencv:
            msg = 'Opencv python bindings cannot be imported. Install opencv or'
            msg += ' set imlib to skimage.'
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
            raise TypeError('Opencv interpolation method not recognized.')

        if scale_y is None: scale_y = scale
        if scale_x is None: scale_x = scale
        M = np.array([[scale_x,0,(1.-scale_x)*ref_x],
                      [0,scale_y,(1.-scale_y)*ref_y]])
        array_out = cv2.warpAffine(array.astype(np.float32), M, 
                                   (array.shape[1], array.shape[0]), flags=intp)

    else:
        raise ValueError('Image transformation library not recognized.')

    if scale_y==scale_x:
        array_out /= scale ** 2

    return array_out

    
    
def cube_rescaling(array, scaling_list, ref_y=None, ref_x=None, imlib='opencv',
                   interpolation='lanczos4', scaling_y=None, scaling_x=None):
    """ Function to rescale a cube, frame by frame, by a factor ``scale``, with
    respect to position (``ref_y``, ``ref_x``). It calls ``frame_rescaling``
    function.
    
    Parameters
    ----------
    array : array_like 
        Input 3d array, cube.
    scaling_list : 1D-array
        Scale corresponding to each frame in the cube.
    ref_y, ref_x : float, optional
        Coordinates X,Y  of the point with respect to which the rescaling will be
        performed. By default the rescaling is done with respect to the center 
        of the frames; central pixel if the frames have odd size.
    imlib : str optional
        See the documentation of the ``vip_hci.preproc.frame_rescaling``
        function.
    interpolation : str, optional
        See the documentation of the ``vip_hci.preproc.frame_rescaling``
        function.
    scaling_y : 1D-array or list
        Scaling factor only for y axis. If provided, it takes priority on 
        scaling_list.
    scaling_x : 1D-array or list
        Scaling factor only for x axis. If provided, it takes priority on 
        scaling_list.
        
    Returns
    -------
    array_sc : array_like
        Resulting cube with rescaled frames.
    array_out : array_like
        Median combined image of the rescaled cube.
    """
    if not array.ndim == 3:
        raise TypeError('Input array is not a cube or 3d array')

    array_sc = np.zeros_like(array)
    if scaling_y is None: scaling_y = [None]*array.shape[0]
    if scaling_x is None: scaling_x = [None]*array.shape[0]
    
    if not ref_y and not ref_x:
        ref_y, ref_x = frame_center(array[0])
    
    for i in range(array.shape[0]):
        array_sc[i] = frame_rescaling(array[i], ref_y=ref_y, ref_x=ref_x, 
                                      scale=scaling_list[i], imlib=imlib,
                                      interpolation=interpolation,
                                      scale_y=scaling_y[i],
                                      scale_x=scaling_x[i])
            
    array_out = np.median(array_sc, axis=0)              
    return array_sc, array_out


def check_scal_vector(scal_vec):
    """
    Function to check if the scaling list has the right format to avoid any bug
    in the pca algorithm, in the case of ifs data.
    Indeed, all scaling factors should be >= 1 (i.e. the scaling should be done
    to match the longest wavelength of the cube)

    Parameter:
    ----------
    scal_vec: array_like, 1d OR list

    Returns:
    --------
    scal_vec: array_like, 1d 
        Vector containing the scaling factors (after correction to comply with
        the condition >= 1)

    """
    correct = False

    if isinstance(scal_vec, list):
        scal_list = scal_vec[:]
        nz = len(scal_list)
        scal_vec = np.zeros(nz)
        for ii in range(nz):
            scal_vec[ii] = scal_list[ii]
        correct = True
    elif isinstance(scal_vec,np.ndarray):
        nz = scal_vec.shape[0]
    else:
        raise TypeError('scal_vec is neither a list or an np.ndarray')

    scal_min = np.amin(scal_vec)

    if scal_min < 1:
        correct = True

    if correct:
        for ii in range(nz):
            scal_vec[ii] = scal_vec[ii]/scal_min

    return scal_vec

