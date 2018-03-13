#! /usr/bin/env python

"""
Module with frame px resampling/rescaling functions.
"""
__author__ = 'Carlos Alberto Gomez Gonzalez, V. Christiaens @ ULg'
__all__ = ['frame_px_resampling',
           'cube_px_resampling',
           'cube_rescaling_wavelengths',
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

from scipy.ndimage.interpolation import geometric_transform, zoom
from ..var import frame_center


def cube_px_resampling(array, scale, imlib='ndimage', interpolation='bicubic'):
    """ Wrapper of ``frame_px_resampling`` for resampling the frames of a cube
    with a single scale factor. Useful when we need to upsample (upscaling) or
    downsample (pixel binning) a set of frames, e.g. an ADI cube.

    Parameters
    ----------
    array : array_like
        Input frame, 2d array.
    scale : int, float or tuple
        Scale factor for upsampling or downsampling the frames in the cube. If
        a tuple it corresponds to the scale along x and y.
    imlib : str optional
        See the documentation of the ``vip_hci.preproc.frame_px_resampling``
        function.
    interpolation : str, optional
        See the documentation of the ``vip_hci.preproc.frame_px_resampling``
        function.

    Returns
    -------
    array_resc : array_like
        Output cube with resampled frames.

    """
    if not array.ndim == 3:
        raise TypeError('Input array is not a cube or 3d array.')

    array_resc = []
    for i in range(array.shape[0]):
        imresc = frame_px_resampling(array[i], scale=scale, imlib=imlib,
                                     interpolation=interpolation)
        array_resc.append(imresc)

    return np.array(array_resc)


def frame_px_resampling(array, scale, imlib='ndimage', interpolation='bicubic'):
    """ Resamples the pixels of a frame wrt to the center, changing the size
    of the frame. If ``scale`` < 1 then the frame is downsampled and if
    ``scale`` > 1 then its pixels are upsampled.

    Parameters
    ----------
    array : array_like 
        Input frame, 2d array.
    scale : int, float or tuple
        Scale factor for upsampling or downsampling the frame. If a tuple it
        corresponds to the scale along x and y.
    imlib : {'ndimage', 'opencv'}, str optional
        Library used for image transformations. ndimage is the default.
    interpolation : str, optional
        For 'ndimage' library: 'nearneig', bilinear', 'bicuadratic', 'bicubic',
        'biquartic', 'biquintic'. The 'nearneig' interpolation is the fastest
        and the 'biquintic' the slowest. The 'nearneig' is the poorer
        option for interpolation of noisy astronomical images.
        For 'opencv' library: 'nearneig', 'bilinear', 'bicubic', 'lanczos4'.
        The 'nearneig' interpolation is the fastest and the 'lanczos4' the
        slowest and accurate.

    Returns
    -------
    array_resc : array_like 
        Output resampled frame.
    """
    if not array.ndim == 2:
        raise TypeError('Input array is not a frame or 2d array')

    if isinstance(scale, tuple):
        scale_x, scale_y = scale
    elif isinstance(scale, (float, int)):
        scale_x = scale
        scale_y = scale
    else:
        raise TypeError('`scale` must be float, int or tuple')

    if imlib == 'ndimage':
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
            raise TypeError('Scipy.ndimage interpolation method not recognized')

        array_resc = zoom(array, zoom=(scale_y, scale_x), order=order)

    elif imlib == 'opencv':
        if no_opencv:
            msg = 'Opencv python bindings cannot be imported. Install opencv or'
            msg += ' set imlib to ndimage'
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
            raise TypeError('Opencv interpolation method not recognized')

        array_resc = cv2.resize(array.astype(np.float32), (0, 0), fx=scale_x,
                                fy=scale_y, interpolation=intp)

    else:
        raise ValueError('Image transformation library not recognized')

    array_resc /= (scale_y * scale_x)
    return array_resc


def cube_rescaling_wavelengths(array, scaling_list, ref_xy=None, imlib='opencv',
                               interpolation='lanczos4', scaling_y=None,
                               scaling_x=None):
    """ Function to rescale a cube, frame by frame by a factor stored in
    ``scaling_list``, with respect to position (``ref_xy`` which by default
    is the center of the frames).
    
    Parameters
    ----------
    array : array_like 
        Input 3d array, cube.
    scaling_list : 1D-array
        Scale corresponding to each frame in the cube.
    ref_xy : float, optional
        Coordinates X,Y  of the point with respect to which the rescaling will be
        performed. By default the rescaling is done with respect to the center 
        of the frames; central pixel if the frames have odd size.
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

    """
    def _scale_func(output_coords, ref_xy=0, scaling=1.0,
                    scale_y=None, scale_x=None):
        """
        For each coordinate point in a new scaled image (output_coords),
        coordinates in the image before the scaling are returned. This scaling
        function is used within geometric_transform which, for each point in the
        output image, will compute the (spline) interpolated value at the
        corresponding frame coordinates before the scaling.
        """
        ref_x, ref_y = ref_xy
        if scale_y is None:
            scale_y = scaling
        if scale_x is None:
            scale_x = scaling
        return (ref_y + ((output_coords[0] - ref_y) / scale_y),
                ref_x + ((output_coords[1] - ref_x) / scale_x))

    def _frame_rescaling(array, ref_xy=None, scale=1.0, imlib='opencv',
                         interpolation='lanczos4', scale_y=None, scale_x=None):
        """ Function to rescale a frame by a factor ``scale``, wrt a reference point
        which by default is the center of the frame (typically the exact location
        of the star). However, it keeps the same dimensions.

        Parameters
        ----------
        array : array_like
            Input frame, 2d array.
        ref_xy : float, optional
            Coordinates X,Y  of the point wrt which the rescaling will be
            applied. By default the rescaling is done with respect to the center
            of the frame.
        scale : float
            Scaling factor. If > 1, it will upsample the input array equally
            along y and x by this factor.
        scale_y : float
            Scaling factor only for y axis. If provided, it takes priority on
            scale parameter.
        scale_x : float
            Scaling factor only for x axis. If provided, it takes priority on
            scale parameter.

        Returns
        -------
        array_out : array_like
            Resulting frame.
        """
        if not array.ndim == 2:
            raise TypeError('Input array is not a frame or 2d array.')

        if scale_y is None:
            scale_y = scale
        if scale_x is None:
            scale_x = scale

        outshape = array.shape
        if ref_xy is None:
            ref_xy = frame_center(array)

        if imlib == 'ndimage':
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
                raise TypeError(
                    'Scipy.ndimage interpolation method not recognized')

            array_out = geometric_transform(array, _scale_func, order=order,
                                            output_shape=outshape,
                                            extra_keywords={'ref_xy': ref_xy,
                                                            'scaling': scale,
                                                            'scale_y': scale_y,
                                                            'scale_x': scale_x})

        elif imlib == 'opencv':
            if no_opencv:
                msg = 'Opencv python bindings cannot be imported. Install '
                msg += ' opencv or set imlib to skimage'
                raise RuntimeError(msg)

            if interpolation == 'bilinear':
                intp = cv2.INTER_LINEAR
            elif interpolation == 'bicubic':
                intp = cv2.INTER_CUBIC
            elif interpolation == 'nearneig':
                intp = cv2.INTER_NEAREST
            elif interpolation == 'lanczos4':
                intp = cv2.INTER_LANCZOS4
            else:
                raise TypeError('Opencv interpolation method not recognized')

            M = np.array([[scale_x, 0, (1. - scale_x) * ref_xy[0]],
                          [0, scale_y, (1. - scale_y) * ref_xy[1]]])
            array_out = cv2.warpAffine(array.astype(np.float32), M, outshape,
                                       flags=intp)

        else:
            raise ValueError('Image transformation library not recognized')

        array_out /= (scale_y * scale_x)
        return array_out

    ############################################################################
    if not array.ndim == 3:
        raise TypeError('Input array is not a cube or 3d array')

    array_sc = []
    for i in range(array.shape[0]):
        array_sc.append(_frame_rescaling(array[i], ref_xy=ref_xy,
                                         scale=scaling_list[i], imlib=imlib,
                                         interpolation=interpolation,
                                         scale_y=scaling_y, scale_x=scaling_x))
    return np.array(array_sc)


def check_scal_vector(scal_vec):
    """
    Function to check if the scaling list has the right format to avoid any bug
    in the pca algorithm, in the case of ifs data.
    Indeed, all scaling factors should be >= 1 (i.e. the scaling should be done
    to match the longest wavelength of the cube)

    Parameter:
    ----------
    scal_vec: 1d array or list
        Vector with the scaling factors.

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
    elif isinstance(scal_vec, np.ndarray):
        nz = scal_vec.shape[0]
    else:
        raise TypeError('scal_vec is neither a list or an np.ndarray')

    scal_min = np.amin(scal_vec)

    if scal_min < 1:
        correct = True

    if correct:
        for ii in range(nz):
            scal_vec[ii] = scal_vec[ii] / scal_min

    return scal_vec

