#! /usr/bin/env python

"""
Module with frame px resampling/rescaling functions.
"""
__author__ = 'Carlos Alberto Gomez Gonzalez, V. Christiaens, R. Farkas'
__all__ = ['frame_px_resampling',
           'cube_px_resampling',
           'cube_rescaling_wavelengths',
           'frame_rescaling',
           'check_scal_vector',
           'find_scal_vector',
           'scale_fft']

import numpy as np
import warnings
try:
    import cv2
    no_opencv = False
except ImportError:
    warnings.warn("Opencv python bindings are missing.", ImportWarning)
    no_opencv = True

from scipy.ndimage.interpolation import geometric_transform, zoom
from scipy.optimize import minimize
from ..var import frame_center, get_square
from .subsampling import cube_collapse
from .recentering import frame_shift
from .cosmetics import frame_crop


def cube_px_resampling(array, scale, imlib='vip-fft', interpolation='lanczos4',
                       keep_center=False, verbose=True):
    """
    Resample the frames of a cube with a single scale factor. Can deal with NaN 
    values.

    Wrapper of ``frame_px_resampling``. Useful when we need to upsample
    (upscaling) or downsample (pixel binning) a set of frames, e.g. an ADI cube.

    Parameters
    ----------
    array : 3d numpy ndarray
        Input cube, 3d array.
    scale : int, float or tuple
        Scale factor for upsampling or downsampling the frames in the cube. If
        a tuple it corresponds to the scale along x and y.
    imlib : str, optional
        See the documentation of the ``vip_hci.preproc.frame_px_resampling``
        function.
    interpolation : str, optional
        See the documentation of the ``vip_hci.preproc.frame_px_resampling``
        function.
    keep_center: bool, opt
        If input dimensions are even and the star centered (i.e. on 
        dim//2, dim//2), whether to keep the star centered after scaling, i.e.
        on (new_dim//2, new_dim//2). For a non-centered input cube, better to
        leave it to False.
    verbose : bool, optional
        Whether to print out additional info such as the new cube shape.

    Returns
    -------
    array_resc : numpy ndarray
        Output cube with resampled frames.

    """
    if array.ndim != 3:
        raise TypeError('Input array is not a cube or 3d array.')

    array_resc = []
    for i in range(array.shape[0]):
        imresc = frame_px_resampling(array[i], scale=scale, imlib=imlib,
                                     interpolation=interpolation, 
                                     keep_center=keep_center)
        array_resc.append(imresc)

    array_resc = np.array(array_resc)

    if verbose:
        print("Cube successfully rescaled")
        print("New shape: {}".format(array_resc.shape))
    return array_resc


def frame_px_resampling(array, scale, imlib='vip-fft', interpolation='lanczos4',
                        keep_center=False, verbose=False):
    """
    Resample the pixels of a frame changing the frame size.
    Can deal with NaN values.
    
    If ``scale`` < 1 then the frame is downsampled and if ``scale`` > 1 then its
    pixels are upsampled.
    
    Warning: if imlib is not 'vip-fft', the input size is even and keep_center
    set to True, an additional interpolation (shifting by (0.5,0.5)px) may 
    occur after rescaling, to ensure center location stays on (dim//2,dim//2).

    Parameters
    ----------
    array : numpy ndarray
        Input frame, 2d array.
    scale : int, float or tuple
        Scale factor for upsampling or downsampling the frame. If a tuple it
        corresponds to the scale along x and y.
    imlib : {'ndimage', 'opencv', 'vip-fft'}, optional
        Library used for image transformations. 'vip-fft' corresponds to a 
        FFT-based rescaling algorithm implemented in VIP 
        (``vip_hci.preproc.scale_fft``).
    interpolation : str, optional
        For 'ndimage' library: 'nearneig', bilinear', 'biquadratic', 'bicubic',
        'biquartic', 'biquintic'. The 'nearneig' interpolation is the fastest
        and the 'biquintic' the slowest. The 'nearneig' is the worst
        option for interpolation of noisy astronomical images.
        For 'opencv' library: 'nearneig', 'bilinear', 'bicubic', 'lanczos4'.
        The 'nearneig' interpolation is the fastest and the 'lanczos4' the
        slowest and accurate.
    keep_center: bool, opt
        If input dimensions are even and the star centered (i.e. on 
        dim//2, dim//2), whether to keep the star centered after scaling, i.e.
        on (new_dim//2, new_dim//2). For a non-centered input frame, better to
        leave it to False.
    verbose : bool, optional
        Whether to print out additional info such as the new image shape.

    Returns
    -------
    array_resc : numpy ndarray
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

    # Replace any NaN with real values before scaling
    mask = None
    nan_mask = np.isnan(array)
    if np.any(nan_mask):
        medval = np.nanmedian(array)
        array[nan_mask] = medval

        mask = np.zeros_like(array)
        mask[nan_mask] = 1
    
    if array.shape[0]%2:
        odd=True
    else:
        odd=False
            
    # expected output size
    out_sz = int(round(array.shape[0]*scale_y)), int(round(array.shape[1]*scale_x))
    
    if not odd and keep_center and imlib != 'vip-fft':
        def _make_odd(img):
            img_odd = np.zeros([img.shape[0]+1,img.shape[1]+1])
            img_odd[:-1,:-1] = img.copy()
            img_odd[-1,:-1] = img[-1].copy()
            img_odd[:-1,-1] = img[:,-1].copy()
            img_odd[-1,-1] = np.mean([img[-1,-2],img[-2,-1],img[-2,-2]])
            return img_odd
        array = _make_odd(array)
        if mask is not None:
            mask = _make_odd(mask)
            

    if imlib == 'ndimage':
        if interpolation == 'nearneig':
            order = 0
        elif interpolation == 'bilinear':
            order = 1
        elif interpolation == 'biquadratic':
            order = 2
        elif interpolation == 'bicubic':
            order = 3
        elif interpolation == 'biquartic' or interpolation == 'lanczos4':
            order = 4
        elif interpolation == 'biquintic':
            order = 5
        else:
            raise TypeError('Scipy.ndimage interpolation method not recognized')

        if mask is not None:
            mask = zoom(mask, zoom=(scale_y, scale_x), order=order)
        array_resc = zoom(array, zoom=(scale_y, scale_x), order=order)
        # For flux conservation:
        array_resc /= scale_y * scale_x
        
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
            
        if mask is not None:
            mask = cv2.resize(mask.astype(np.float32), (0, 0), fx=scale_x,
                                fy=scale_y, interpolation=intp)
            
        array_resc = cv2.resize(array.astype(np.float32), (0, 0), fx=scale_x,
                                fy=scale_y, interpolation=intp)
        
        # For flux conservation:
        array_resc /= scale_y * scale_x

    elif imlib == 'vip-fft':
        if scale_x != scale_y:
            msg='FFT scaling only supports identical factors along x and y'
            raise ValueError(msg)
        if array.shape[0] != array.shape[1]:
            msg='FFT scaling only supports square input arrays'
            raise ValueError(msg)   
            
        # make array with even dimensions before FFT-scaling
        if odd:
            array_even = np.zeros([array.shape[0]+1,array.shape[1]+1])
            array_even[1:,1:] = array
            array = array_even

        if mask is not None:
            if odd:
                mask_even = np.zeros([mask.shape[0]+1,mask.shape[1]+1])
                mask_even[1:,1:] = mask
                mask = mask_even             
            mask = scale_fft(mask, scale_x)
            if odd:
                mask_odd = np.zeros([mask.shape[0]-1,mask.shape[1]-1])
                mask_odd = mask[1:,1:]
                mask = mask_odd
            
        array_resc = scale_fft(array, scale_x)
        if odd:
            array = np.zeros([array_resc.shape[0]-1,array_resc.shape[1]-1])
            array = array_resc[1:,1:]
            array_resc = array
        
        #Note: FFT preserves flux - no need to scale flux separately
        
    else:
        raise ValueError('Image transformation library not recognized')

    # Place back NaN values in scaled array
    if mask is not None:
        array_resc[mask >= 0.5] = np.nan
 
    if keep_center and not array_resc.shape[0]%2 and imlib != 'vip-fft':
        if imlib == 'ndimage':
            imlib_s = 'ndimage-interp'
        else:
            imlib_s = imlib
        array_resc = frame_shift(array_resc, 0.5, 0.5, imlib_s, interpolation)
        
    if array_resc.shape != out_sz and imlib != 'vip-fft':
        if out_sz[0] == out_sz[1]:
            if out_sz[0]<array_resc.shape[0]:
                array_resc = frame_crop(array_resc, out_sz[0], force=True,
                                        verbose=False)
        else:
            # crop manually along each axis
            cy, cx = frame_center(array_resc)
            wing_y = (out_sz[0]-1)/2
            y0 = int(cy-wing_y)
            yN = int(cy+wing_y+1)
            wing_x = (out_sz[1]-1)/2
            x0 = int(cx-wing_x)
            xN = int(cx+wing_x+1)
            array_resc = array_resc[y0:yN,x0:xN]
            
    if verbose:
        print("Image successfully rescaled")
        print("New shape: {}".format(array_resc.shape))

    return array_resc


def cube_rescaling_wavelengths(cube, scal_list, full_output=True, inverse=False,
                               y_in=None, x_in=None, imlib='vip-fft',
                               interpolation='lanczos4', collapse='median',
                               pad_mode='reflect'):
    """
    Scale/Descale a cube by scal_list, with padding. Can deal with NaN values.

    Wrapper to scale or descale a cube by factors given in scal_list,
    without any loss of information (zero-padding if scaling > 1).
    Important: in case of IFS data, the scaling factors in scal_list should be
    >= 1 (ie. provide the scaling factors as for scaling to the longest
    wavelength channel).

    Parameters
    ----------
    cube: 3D-array
       Data cube with frames to be rescaled.
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
    y_in, x_in: int
       Initial y and x sizes, required for ``inverse=True``. In case the cube is
       descaled, these values will be used to crop back the cubes/frames to
       their original size.
    imlib : {'opencv', 'ndimage', 'vip-fft'}, str optional
        Library used for image transformations. Opencv is faster than ndimage or
        skimage. 'vip-fft' corresponds to a FFT-based rescaling algorithm 
        implemented in VIP (``vip_hci.preproc.scale_fft``).
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
    pad_mode : str, optional
        One of the following string values:

            ``'constant'``
                pads with a constant value
            ``'edge'``
                pads with the edge values of array
            ``'linear_ramp'``
                pads with the linear ramp between end_value and the array edge
                value.
            ``'maximum'``
                pads with the maximum value of all or part of the vector along
                each axis
            ``'mean'``
                pads with the mean value of all or part of the vector along each
                axis
            ``'median'``
                pads with the median value of all or part of the vector along
                each axis
            ``'minimum'``
                pads with the minimum value of all or part of the vector along
                each axis
            ``'reflect'``
                pads with the reflection of the vector mirrored on the first and
                last values of the vector along each axis
            ``'symmetric'``
                pads with the reflection of the vector mirrored along the edge
                of the array
            ``'wrap'``
                pads with the wrap of the vector along the axis. The first
                values are used to pad the end and the end values are used to
                pad the beginning

    Returns
    -------
    frame: 2d array
        The median of the rescaled cube.
    cube : 3d array
        [full_output] rescaled cube
    frame : 2d array
        [full_output] median of the rescaled cube
    y,x,cy,cx : float
        [full_output] New y and x shapes of the cube, and the new centers cy and
        cx of the frames

    """
    n, y, x = cube.shape

    max_sc = np.amax(scal_list)

    if not inverse and max_sc > 1:
        new_y = int(np.ceil(max_sc * y))
        new_x = int(np.ceil(max_sc * x))
        if (new_y - y) % 2 != 0:
            new_y += 1
        if (new_x - x) % 2 != 0:
            new_x += 1
        pad_len_y = (new_y - y) // 2
        pad_len_x = (new_x - x) // 2
        pad_width = ((0, 0), (pad_len_y, pad_len_y), (pad_len_x, pad_len_x))
        big_cube = np.pad(cube, pad_width, pad_mode)
    else:
        big_cube = cube.copy()

    n, y, x = big_cube.shape
    cy, cx = frame_center(big_cube[0])

    if inverse:
        scal_list = 1. / scal_list
        cy, cx = frame_center(cube[0])

    # (de)scale the cube, so that a planet would now move radially
    cube = _cube_resc_wave(big_cube, scal_list, ref_xy=(cx, cy),
                           imlib=imlib, interpolation=interpolation)
    frame = cube_collapse(cube, collapse)

    if inverse and max_sc > 1:
        if y_in is None or x_in is None:
            raise ValueError("You need to provide y_in and x_in when "
                             "inverse=True!")
        siz = max(y_in, x_in)
        if frame.shape[0] > siz:
            frame = get_square(frame, siz, cy, cx)
        if full_output and cube.shape[-1]>siz:
            n_z = cube.shape[0]
            array_old = cube.copy()
            cube = np.zeros([n_z, siz, siz])
            for zz in range(n_z):
                cube[zz] = get_square(array_old[zz], siz, cy, cx)

    if full_output:
        return cube, frame, y, x, cy, cx
    else:
        return frame


def _scale_func(output_coords, ref_xy=0, scaling=1.0, scale_y=None,
                scale_x=None):
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


def frame_rescaling(array, ref_xy=None, scale=1.0, imlib='vip-fft',
                    interpolation='lanczos4', scale_y=None, scale_x=None):
    """
    Rescale a frame by a factor wrt a reference point.

    The reference point is by default the center of the frame (typically the
    exact location of the star). However, it keeps the same dimensions.

    Parameters
    ----------
    array : numpy ndarray
        Input frame, 2d array.
    ref_xy : float, optional
        Coordinates X,Y  of the point wrt which the rescaling will be
        applied. By default the rescaling is done with respect to the center
        of the frame.
    scale : float
        Scaling factor. If > 1, it will upsample the input array equally
        along y and x by this factor.
    imlib : {'ndimage', 'opencv', 'vip-fft'}, optional
        Library used for image transformations. 'vip-fft' corresponds to a 
        FFT-based rescaling algorithm implemented in VIP 
        (``vip_hci.preproc.scale_fft``).
    interpolation : str, optional
        For 'ndimage' library: 'nearneig', bilinear', 'biquadratic', 'bicubic',
        'biquartic', 'biquintic'. The 'nearneig' interpolation is the fastest
        and the 'biquintic' the slowest. The 'nearneig' is the worst
        option for interpolation of noisy astronomical images.
        For 'opencv' library: 'nearneig', 'bilinear', 'bicubic', 'lanczos4'.
        The 'nearneig' interpolation is the fastest and the 'lanczos4' the
        slowest and accurate.
    scale_y : float
        Scaling factor only for y axis. If provided, it takes priority on
        scale parameter.
    scale_x : float
        Scaling factor only for x axis. If provided, it takes priority on
        scale parameter.

    Returns
    -------
    array_out : numpy ndarray
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
    else:
        if imlib == 'vip-fft' and ref_xy != frame_center(array):
            msg = "'vip-fft'imlib does not yet allow for custom center to be "
            msg+= " provided "
            raise ValueError(msg)
        
    # Replace any NaN with real values before scaling
    mask = None
    nan_mask = np.isnan(array)
    if np.any(nan_mask):
        medval = np.nanmedian(array)
        array[nan_mask] = medval

        mask = np.zeros_like(array)
        mask[nan_mask] = 1
        
    if imlib == 'ndimage':
        if interpolation == 'nearneig':
            order = 0
        elif interpolation == 'bilinear':
            order = 1
        elif interpolation == 'biquadratic':
            order = 2
        elif interpolation == 'bicubic':
            order = 3
        elif interpolation == 'biquartic' or interpolation == 'lanczos4':
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
        array_out /= scale_y * scale_x

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
        array_out /= scale_y * scale_x

    elif imlib == 'vip-fft':
        if scale_x != scale_y:
            msg='FFT scaling only supports identical factors along x and y'
            raise ValueError(msg)
        if array.shape[0] != array.shape[1]:
            msg='FFT scaling only supports square input arrays'
            raise ValueError(msg)   
            
        # make array with even dimensions before FFT-scaling
        if array.shape[0]%2:
            odd=True
            array_even = np.zeros([array.shape[0]+1,array.shape[1]+1])
            array_even[1:,1:] = array
            array = array_even
        else:
            odd = False

        if mask is not None:
            if odd:
                mask_even = np.zeros([mask.shape[0]+1,mask.shape[1]+1])
                mask_even[1:,1:] = mask
                mask = mask_even             
            mask = scale_fft(mask, scale_x, ori_dim=True)
            if odd:
                mask_odd = np.zeros([mask.shape[0]-1,mask.shape[1]-1])
                mask_odd = mask[1:,1:]
                mask = mask_odd
            
        array_out = scale_fft(array, scale_x, ori_dim=True)
        if odd:
            array = np.zeros([array_out.shape[0]-1,array_out.shape[1]-1])
            array = array_out[1:,1:]
            array_out = array

    else:
        raise ValueError('Image transformation library not recognized')

    # Place back NaN values in scaled array
    if mask is not None:
        array_out[mask >= 0.5] = np.nan

    
    return array_out


def _cube_resc_wave(array, scaling_list, ref_xy=None, imlib='vip-fft',
                    interpolation='lanczos4', scaling_y=None, scaling_x=None):
    """
    Rescale a cube by factors from ``scaling_list`` wrt a position.

    Parameters
    ----------
    array : numpy ndarray
        Input 3d array, cube.
    scaling_list : 1D-array
        Scale corresponding to each frame in the cube.
    ref_xy : float, optional
        Coordinates X,Y of the point with respect to which the rescaling will be
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
    array_sc : numpy ndarray
        Resulting cube with rescaled frames.

    """

    if array.ndim != 3:
        raise TypeError('Input array is not a cube or 3d array')

    array_sc = []
    if scaling_list is None:
        scaling_list = [None]*array.shape[0]
    for i in range(array.shape[0]):
        array_sc.append(frame_rescaling(array[i], ref_xy=ref_xy,
                                        scale=scaling_list[i], imlib=imlib,
                                        interpolation=interpolation,
                                        scale_y=scaling_y, scale_x=scaling_x))
    return np.array(array_sc)


def check_scal_vector(scal_vec):
    """
    Turn wavelengths (IFS data) into a scaling factor list.

    It checks that it has the right format: all scaling factors should be >= 1
    (i.e. the scaling should be done wrt the longest wavelength of the cube).

    Parameters
    ----------
    scal_vec: 1d array or list
        Vector with the wavelengths.

    Returns
    -------
    scal_vec: numpy ndarray, 1d
        Vector containing the scaling factors (after correction to comply with
        the condition >= 1).

    """
    if not isinstance(scal_vec, (list, np.ndarray)):
        raise TypeError('`Scal_vec` is neither a list or an np.ndarray')

    scal_vec = np.array(scal_vec)

    # checking if min factor is 1:
    if scal_vec.min() != 1:
        scal_vec = 1 / scal_vec
        scal_vec /= scal_vec.min()

    return scal_vec


def find_scal_vector(cube, lbdas, fluxes, mask=None, nfp=2, fm="stddev", 
                     simplex_options=None, debug=False, **kwargs):
    """
    Find the optimal scaling factor for the channels of an IFS cube (or of 
    dual-band pairs of images).

    The algorithm finds the optimal scaling factor that minimizes residuals in
    the rescaled frames. It takes the inverse of the wavelength vector as a 
    first guess, and uses a similar method as the negative fake companion 
    technique, but minimizing residuals in either a mask or the whole field.

    Parameters
    ----------
    cube: 3D-array
       Data cube with frames to be rescaled.
    lbdas: 1d array or list
        Vector with the wavelengths, used for first guess on scaling factor.
    fluxes: 1d array or list
        Vector with the (unsaturated) fluxes at the different wavelengths, 
        used for first guess on flux factor.
    mask: 2D-array, opt
        Binary mask, with ones where the residual intensities should be 
        evaluated. If None is provided, the whole field is used.
    nfp: int, opt, {1,2}
        Number of free parameters: spatial scaling alone or spatial scaling + 
        flux scaling.
    fm: str, opt, {"sum","stddev"}
        Figure of merit to use: sum of squared residuals or stddev of residual 
        pixels.
    options: dict, optional
        The scipy.optimize.minimize options.
    **kwargs: optional
        Optional arguments to the scipy.optimize.minimize function
        
    Returns
    -------
    scal_vec: numpy ndarray, 1d
        Vector containing the scaling factors (after correction to comply with
        the condition >= 1).
    if nfp==2, also returns:
    flux_vec: numpy ndarray, 1d
        Vector containing the associated flux factors.
    """

    scal_vec_ini = lbdas[-1]/lbdas
    n_z = len(lbdas)
    if n_z != len(fluxes) or n_z != cube.shape[0]:
        msg = "first axis of cube, fluxes and lbda must have same length"
        raise TypeError(msg)

    if simplex_options is None:
        simplex_options = {'xatol': 1e-6, 'fatol': 1e-6, 'maxiter': 800,
                           'maxfev': 2000}
    scal_vec = np.ones(n_z)
    flux_vec = np.ones(n_z)
    for z in range(n_z-1):
        flux_scal = fluxes[-1]/fluxes[z]
        cube_tmp = np.array([cube[z],cube[-1]])
        if nfp==1:
            p_ini = (scal_vec_ini[z],)
            solu = minimize(_chisquare_scal, p_ini, args=(cube_tmp, flux_scal, 
                                                          mask, fm),
                            method='Nelder-Mead', options=simplex_options, 
                            **kwargs)
            scal_fac, =  solu.x
            flux_fac = flux_scal
        else:
            p_ini = (scal_vec_ini[z],flux_scal)
            solu = minimize(_chisquare_scal_2fp, p_ini, args=(cube_tmp,mask,fm),
                            method='Nelder-Mead', options=simplex_options, 
                            **kwargs)      
            scal_fac, flux_fac =  solu.x
        if debug:
            print("channel {:.0f}:".format(z), solu.x)
        scal_vec[z] = scal_fac
        flux_vec[z] = flux_fac

    scal_vec = check_scal_vector(scal_vec)
    
    return scal_vec, flux_vec


def _find_indices_sdi(wl, dist, index_ref, fwhm, delta_sep=1, nframes=None,
                      debug=False):
    """
    Find optimal wavelengths which minimize self-subtraction in model PSF
    subtraction.

    Parameters
    ----------
    wl : numpy ndarray or list
        Vector with the scaling factors.
    dist : float
        Separation or distance (in pixels) from the center of the array.
    index_ref : int
        The `wl` index for which we are finding the pairs.
    fwhm : float
        Mean FWHM of all the wavelengths (in pixels).
    delta_sep : float, optional
        The threshold separation in terms of the mean FWHM.
    nframes : None or int, optional
        Must be an even value. In not None, then between 2 and adjacent
        ``nframes`` are kept.
    debug : bool, optional
        It True it prints out debug information.

    Returns
    -------
    indices : numpy ndarray
        List of good indices.

    """
    wl = np.asarray(wl)
    wl_ref = wl[index_ref]
    sep_lft = (wl_ref - wl) / wl_ref * ((dist + fwhm * delta_sep) / fwhm)
    sep_rgt = (wl - wl_ref) / wl_ref * ((dist - fwhm * delta_sep) / fwhm)
    map_lft = sep_lft >= delta_sep
    map_rgt = sep_rgt >= delta_sep
    indices = np.nonzero(map_lft | map_rgt)[0]

    if debug:
        print("dist: {}, index_ref: {}".format(dist, index_ref))
        print("sep_lft:", "  ".join(["{:+.2f}".format(x) for x in sep_lft]))
        print("sep_rgt:", "  ".join(["{:+.2f}".format(x) for x in sep_rgt]))
        print("indices:", indices)
        print("indices size: {}".format(indices.size))

    if indices.size == 0:
        raise RuntimeError("No frames left after radial motion threshold. Try "
                           "decreasing the value of `delta_sep`")

    if nframes is not None:
        i1 = map_lft.sum()
        window = nframes // 2
        if i1 - window < 0 or i1 + window > indices[-1]:
            window = nframes
        ind1 = max(0, i1 - window)
        ind2 = min(wl.size, i1 + window)
        indices = indices[ind1: ind2]

        if indices.size < 2:
            raise RuntimeError("No frames left after radial motion threshold. "
                               "Try decreasing the value of `delta_sep` or "
                               "`nframes`")

    if debug:
        print("indices (nframes):", indices)

    return indices


def _chisquare_scal(modelParameters, cube, flux_fac=1, mask=None, fm='sum'):
    r"""
    Calculate the reduced math:`\chi^2`:
    .. math:: \chi^2_r = \frac{1}{N-3}\sum_{j=1}^{N} |I_j|,
    where N is the number of pixels in the image (or mask if provided), and 
    :math:`I_j` the j-th pixel intensity, considering one free parameter: the 
    physical scaling factor between images of the cube, for a given
    input flux scaling factor.
    
    Parameters
    ----------    
    modelParameters: tuple
        The model parameters, typically (scal_fac, flux_fac).
    cube: numpy.array
        The cube of fits images expressed as a numpy.array.
    flux_fac:     
        
    mask: 2D-array, opt
        Binary mask, with ones where the residual intensities should be 
        evaluated. If None is provided, the whole field is used.
    fm: str, opt, {"sum","stddev"}
        Figure of merit to use: sum of squared residuals or stddev of residual 
        pixels.
        
    Returns
    -------
    chi: float
        The reduced chi squared.
        
    """
    # rescale in flux and spatially
    array = cube.copy()
    #scale_fac, flux_fac = modelParameters
    scale_fac, = modelParameters
    array[0]*=flux_fac
    scaling_list = np.array([scale_fac,1])
    array = _cube_resc_wave(array, scaling_list)

    frame = array[1]-array[0]
    if mask is None:
        mask = np.ones_like(frame)
    
    
    if fm == 'sum':
        chi = np.sum(np.power(frame[np.where(mask)],2))
    elif fm == 'stddev':
        values = frame[np.where(mask)]
        values = values[values != 0]
        chi = np.std(values)
    else:
        raise RuntimeError('fm choice not recognized.')
        
    return chi

def _chisquare_scal_2fp(modelParameters, cube, mask=None, fm='sum'):
    r"""
    Calculate the reduced :math:`\chi^2`:
    .. math:: \chi^2_r = \frac{1}{N-3}\sum_{j=1}^{N} |I_j|,
    where N is the number of pixels within a circular aperture centered on the 
    first estimate of the planet position, and :math:`I_j` the j-th pixel 
    intensity. Two free parameters: physical and flux scaling factors.
    
    Parameters
    ----------    
    modelParameters: tuple
        The model parameters, typically (scal_fac, flux_fac).
    cube: numpy.array
        The cube of fits images expressed as a numpy.array.
    mask: 2D-array, opt
        Binary mask, with ones where the residual intensities should be 
        evaluated. If None is provided, the whole field is used.
    fm: str, opt, {"sum","stddev"}
        Figure of merit to use: sum of squared residuals or stddev of residual 
        pixels.
        
    Returns
    -------
    chi: float
        The reduced chi squared.
        
    """
    # rescale in flux and spatially
    array = cube.copy()
    scale_fac, flux_fac = modelParameters
    array[0]*=flux_fac
    scaling_list = np.array([scale_fac,1])
    array = _cube_resc_wave(array, scaling_list)

    frame = array[1]-array[0]
    if mask is None:
        mask = np.ones_like(frame)
    
    
    if fm == 'sum':
        chi = np.sum(np.power(frame[np.where(mask)],2))
    elif fm == 'stddev':
        values = frame[np.where(mask)]
        values = values[values != 0]
        chi = np.std(values)
    else:
        raise RuntimeError('fm choice not recognized.')
        
    return chi


    
def scale_fft(array, scale, ori_dim=False):
    """
    Resample the frames of a cube with a single scale factor using a FFT-based
    method.

    Parameters
    ----------
    array : 3d numpy ndarray
        Input cube, 3d array.
    scale : int or float
        Scale factor for upsampling or downsampling the frames in the cube. If
        a tuple it corresponds to the scale along x and y.
    ori_dim: bool, opt
        Whether to crop/pad scaled array in order to have the output with the
        same dimensions as the input array. By default, the x,y dimensions of 
        the output are the closest integer to scale*dim_input, with the same 
        parity as the input.
    
    Returns
    -------
    array_resc : numpy ndarray
        Output cube with resampled frames.
    
    """
    if scale == 1:
        return array
    dim = array.shape[0] # even square
    dtype = array.dtype.kind

    kd_array = np.arange(dim/2 + 1, dtype=int)
    
    # scaling factor chosen as *close* as possible to N''/N', where: 
    #   N' = N + 2*KD (N': dim after FT)
    #   N" = N + 2*KF (N'': dim after FT-1 of FT image), 
    #   => N" = 2*round(N'*sc/2)
    #   => KF = (N"-N)/2 = round(N'*sc/2 - N/2) 
    #         = round(N/2*(sc-1) + KD*sc)
    # We call yy=N/2*(sc-1) +KD*sc   
    yy = dim/2 * (scale - 1) + kd_array.astype(float)*scale
    
    # We minimize the difference between the `ideal' N" and its closest 
    # integer value by minimizing |yy-int(yy)|.
    kf_array = np.round(yy).astype(int)
    tmp = np.abs(yy-kf_array)
    imin = np.nanargmin(tmp)

    kd_io = kd_array[imin]
    kf_io = kf_array[imin]
    
    # Extract a part of array and place into dim_p array
    dim_p = int(dim + 2*kd_io)
    tmp = np.zeros((dim_p, dim_p), dtype=dtype)
    tmp[kd_io:kd_io+dim, kd_io:kd_io+dim] = array

    # Fourier-transform the larger array
    array_f = np.fft.fftshift(np.fft.fft2(tmp))
    
    # Extract a part of, or expand, the FT to dim_pp pixels
    dim_pp = int(dim + 2*kf_io)
    
    if dim_pp > dim_p:
        tmp = np.zeros((dim_pp, dim_pp), dtype=np.complex)
        tmp[(dim_pp-dim_p)//2:(dim_pp+dim_p)//2, 
            (dim_pp-dim_p)//2:(dim_pp+dim_p)//2] = array_f
    else:
        tmp = array_f[kd_io-kf_io:kd_io-kf_io+dim_pp, 
                      kd_io-kf_io:kd_io-kf_io+dim_pp]

    # inverse Fourier-transform the FT
    tmp = np.fft.ifft2(np.fft.fftshift(tmp))
    array_resc = tmp.real
    del tmp

    # Extract a part of or expand the scaled image to desired number of pixels
    dim_resc = int(round(scale*dim))
    if dim_resc>dim and dim_resc%2 != dim%2:
         dim_resc+=1
    elif dim_resc<dim and dim_resc%2 != dim%2:
         dim_resc-=1 # for reversibility
    
    if not ori_dim and dim_pp > dim_resc:
        array_resc = array_resc[(dim_pp-dim_resc)//2:(dim_pp+dim_resc)//2,
                                (dim_pp-dim_resc)//2:(dim_pp+dim_resc)//2]
    elif not ori_dim and dim_pp <= dim_resc:
        array = np.zeros((dim_resc,dim_resc))
        array[(dim_resc-dim_pp)//2:(dim_resc+dim_pp)//2,
              (dim_resc-dim_pp)//2:(dim_resc+dim_pp)//2] = array_resc
        array_resc = array
    elif dim_pp > dim:
        array_resc = array_resc[kf_io:kf_io+dim, kf_io:kf_io+dim]
    elif dim_pp  <= dim:
        scaled = array*0
        scaled[-kf_io:-kf_io+dim_pp, -kf_io:-kf_io+dim_pp] = array_resc
        array_resc = scaled

    return array_resc
