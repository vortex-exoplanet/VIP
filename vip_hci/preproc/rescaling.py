#! /usr/bin/env python

"""
Module with frame px resampling/rescaling functions.
"""
from __future__ import division, print_function

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
    warnings.warn("Opencv python bindings are missing.", ImportWarning)
    no_opencv = True

from scipy.ndimage.interpolation import geometric_transform, zoom
from ..var import frame_center, get_square
from .subsampling import cube_collapse


def cube_px_resampling(array, scale, imlib='ndimage', interpolation='bicubic',
                       verbose=True):
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
    verbose : bool, optional
        Whether to print out additional info such as the new cube shape.

    Returns
    -------
    array_resc : array_like
        Output cube with resampled frames.

    """
    if array.ndim != 3:
        raise TypeError('Input array is not a cube or 3d array.')

    array_resc = []
    for i in range(array.shape[0]):
        imresc = frame_px_resampling(array[i], scale=scale, imlib=imlib,
                                     interpolation=interpolation)
        array_resc.append(imresc)

    array_resc = np.array(array_resc)

    if verbose:
        print("Cube successfully rescaled")
        print("New shape: {}".format(array_resc.shape))
    return array_resc


def frame_px_resampling(array, scale, imlib='ndimage', interpolation='bicubic',
                        verbose=False):
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
        and the 'biquintic' the slowest. The 'nearneig' is the worst
        option for interpolation of noisy astronomical images.
        For 'opencv' library: 'nearneig', 'bilinear', 'bicubic', 'lanczos4'.
        The 'nearneig' interpolation is the fastest and the 'lanczos4' the
        slowest and accurate.
    verbose : bool, optional
        Whether to print out additional info such as the new image shape.

    Returns
    -------
    array_resc : array_like 
        Output resampled frame.
    """
    if array.ndim != 2:
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
            intp = cv2.INTER_CUBIC
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

    array_resc /= scale_y * scale_x

    if verbose:
        print("Image successfully rescaled")
        print("New shape: {}".format(array_resc.shape))

    return array_resc


def cube_rescaling_wavelengths(cube, scal_list, full_output=True, inverse=False,
                               y_in=1, x_in=1, imlib='opencv',
                               interpolation='lanczos4', collapse='median'):
    """ Wrapper to scale or descale a cube by factors given in scal_list,
    without any loss of information (zero-padding if scaling > 1).
    Important: in case of ifs data, the scaling factors in var_list should be
    >= 1 (ie. provide the scaling factors as for scaling to the longest
    wavelength channel).

    Parameters:
    -----------
    cube: 3D-array
       Datacube that whose frames have to be rescaled.
    scal_list: 1D-array
       Vector of same dimension as the first dimension of datacube, containing
       the scaling factor for each frame.
    full_output: bool, optional
       Whether to output just the rescaled cube (False) or also its median,
       the new y and x shapes of the cube, and the new centers cy and cx of the
       frames (True).
    inverse: bool, optional
       Whether to inverse the scaling factors in scal_list before applying them
       or not; i.e. True is to descale the cube (typically after a first scaling
       has already been done)
    y_in, x-in: int, optional
       Initial y and x sizes. In case the cube is descaled, these values will
       be used to crop back the cubes/frames to their original size.
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
    collapse : {'median', 'mean', 'sum', 'trimmean'}, str optional
        Sets the way of collapsing the frames for producing a final image.

    Returns:
    --------
    frame: 2D-array
        The median of the rescaled cube.
    If full_output is set to True, the function returns:
    cube,frame,y,x,cy,cx: 3D-array,2D-array,int,int,int,int
        The rescaled cube, its median, the new y and x shapes of the cube, and
        the new centers cy and cx of the frames
    """
    n, y, x = cube.shape

    max_sc = np.amax(scal_list)

    if not inverse and max_sc > 1:
        new_y = int(np.ceil(max_sc * y))
        new_x = int(np.ceil(max_sc * x))
        if (new_y - y) % 2 != 0:
            new_y = new_y+1
        if (new_x - x) % 2 != 0:
            new_x = new_x + 1
        pad_len_y = (new_y - y) // 2
        pad_len_x = (new_x - x) // 2
        pad_width = ((0, 0), (pad_len_y, pad_len_y), (pad_len_x, pad_len_x))
        big_cube = np.pad(cube, pad_width, 'reflect', reflect_type='even')
    else:
        big_cube = cube.copy()

    n, y, x = big_cube.shape
    cy, cx = frame_center(big_cube[0])
    var_list = scal_list

    if inverse:
        var_list = 1. / scal_list
        cy, cx = frame_center(cube[0])

    # (de)scale the cube, so that a planet would now move radially
    cube = _cube_resc_wave(big_cube, var_list, ref_xy=(cx, cy),
                           imlib=imlib, interpolation=interpolation)
    frame = cube_collapse(cube, collapse)

    if inverse and max_sc > 1:
        siz = max(y_in, x_in)
        frame = get_square(frame, siz, cy, cx)
        if full_output:
            n_z = cube.shape[0]
            array_old = cube.copy()
            cube = np.zeros([n_z, siz, siz])
            for zz in range(n_z):
                cube[zz] = get_square(array_old[zz], siz, cy, cx)

    if full_output:
        return cube, frame, y, x, cy, cx
    else:
        return frame


def _cube_resc_wave(array, scaling_list, ref_xy=None, imlib='opencv',
                    interpolation='lanczos4', scaling_y=None, scaling_x=None):
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
    imlib : str optional
        See the documentation of ``vip_hci.preproc.cube_rescaling_wavelengths``.
    interpolation : str, optional
        See the documentation of ``vip_hci.preproc.cube_rescaling_wavelengths``.
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
        return (ref_y + (output_coords[0] - ref_y) / scale_y,
                ref_x + (output_coords[1] - ref_x) / scale_x)

    def _frame_rescaling(array, ref_xy=None, scale=1.0, imlib='opencv',
                         interpolation='lanczos4', scale_y=None, scale_x=None):
        """ Function to rescale a frame by a factor ``scale``, wrt a reference
        point which by default is the center of the frame (typically the exact
        location of the star). However, it keeps the same dimensions.

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
        if array.ndim != 2:
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

        array_out /= scale_y * scale_x
        return array_out

    ############################################################################
    if array.ndim != 3:
        raise TypeError('Input array is not a cube or 3d array')

    array_sc = []
    for i in range(array.shape[0]):
        array_sc.append(_frame_rescaling(array[i], ref_xy=ref_xy,
                                         scale=scaling_list[i], imlib=imlib,
                                         interpolation=interpolation,
                                         scale_y=scaling_y, scale_x=scaling_x))
    return np.array(array_sc)


def check_scal_vector(scal_vec):
    """ Function to turn the wavelengths (in the case of IFS data) into a
    scaling factor list. It checksthat it has the right format: all scaling
    factors should be >= 1 (i.e. the scaling should be done wrt the longest
    wavelength of the cube).

    Parameter:
    ----------
    scal_vec: 1d array or list
        Vector with the wavelengths.

    Returns:
    --------
    scal_vec: array_like, 1d 
        Vector containing the scaling factors (after correction to comply with
        the condition >= 1).
    """
    if not isinstance(scal_vec, (list, np.ndarray)):
        raise TypeError('`Scal_vec` is neither a list or an np.ndarray')

    scal_vec = np.array(scal_vec)

    # checking if min factor is 1
    if scal_vec.min() != 1:
        scal_vec = 1/scal_vec
        scal_vec /= scal_vec.min()

    return scal_vec


def _find_indices_sdi(wl, dist, index_ref, fwhm, delta_sep=1, nframes=None,
                      debug=False):
    """ Finds a radial threshold for avoiding as much as possible self-
    subtraction while doing model PSF subtraction.

    # TODO: check this output. Also, find a more pythonic way to to this!

    Parameters
    ----------
    wl : 1d array or list
        Vector with the scaling factors.
    dist : int
        Separation or distance from the center of the array (star).
    index_ref : int
        The `wl` index for which we are finding the pairs.
    fwhm : float
        Mean FWHM of all the wavelengths.
    delta_sep : float, optional
        The threshold separation in terms of the mean FWHM.
    debug : bool, optional
        It True it prints out debug information.

    Returns
    -------
    index_w2 : int
        Index of the frame located (outwards) at a `delta_sep` separation from
        `index_ref`.
    index_w3 :
        Index of the frame located (inwards) at a `delta_sep` separation from
        `index_ref`.
    """
    nwvs = wl.shape[0]
    index_w2 = 0
    index_ref = int(index_ref)
    for i in range(0, index_ref):
        index_w2 = i
        sep = ((wl[index_ref] - wl[index_w2]) / wl[index_ref]) * (
                (dist + fwhm * delta_sep) / fwhm)
        if debug:
            sep_pxs = ((wl[index_ref] - wl[index_w2]) / wl[index_ref]) * (
                        dist + fwhm * delta_sep)
            print(sep, sep_pxs)
        if sep <= delta_sep:
            if index_w2 == 0:
                index_w2 += 1
            break
    if debug:
        print('Index 1 = ', index_w2)

    index_w3 = nwvs
    for i in range(index_ref, nwvs)[::-1]:
        index_w3 = i
        sep = ((wl[index_w3] - wl[index_ref]) / wl[index_ref]) * (
                (dist - fwhm * delta_sep) / fwhm)
        if debug:
            sep_pxs = ((wl[index_w3] - wl[index_ref]) / wl[index_ref]) * (
                        dist - fwhm * delta_sep)
            print(sep, sep_pxs)
        if sep <= delta_sep:
            if index_w3 == nwvs - 1:
                index_w3 += 1
            break

    if debug:
        print('Index 2 = ', index_w3)

    if nframes is not None:
        window = nframes // 2
        ind1 = max(index_w2 - window, 0)
        ind2 = index_w2
        ind3 = index_w3
        ind4 = min(index_w3 + window, nwvs)
        indices = np.array(list(range(ind1, ind2)) + list(range(ind3, ind4)))
    else:
        half1 = range(0, index_w2)
        half2 = range(index_w3, nwvs)
        indices = np.array(list(half1) + list(half2))

    indices = indices.astype(int)
    if indices.shape[0] == 1:
        msg = 'No frames left after the radial motion threshold. Try decreasing'
        msg += ' the value of `delta_sep` (for dist: {} pxs)'
        raise RuntimeError(msg.format(dist))
    return indices

