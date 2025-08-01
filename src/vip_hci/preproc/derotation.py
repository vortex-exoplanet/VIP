#! /usr/bin/env python
"""
Module with frame de-rotation routine for ADI.

.. [LAR97]
   | Larkin et al. 1997
   | **Fast Fourier method for the accurate rotation of sampled images**
   | *Optics Communications, Volume 154, Issue 1-3, pp. 99-106*
   | `https://ui.adsabs.harvard.edu/abs/1997OptCo.139...99L
     <https://ui.adsabs.harvard.edu/abs/1997OptCo.139...99L>`_

"""
from multiprocessing import cpu_count

from skimage.transform import rotate

from ..config.utils_conf import iterable
from ..config.utils_conf import pool_map
from ..var import frame_center
from ..var import frame_filter_lowpass
from .cosmetics import frame_pad
__author__ = 'C. A. Gomez Gonzalez, V. Christiaens, S. Juillard'
__all__ = ['cube_derotate',
           'frame_rotate',
           'rotate_fft']

from astropy.stats import sigma_clipped_stats
import numpy as np
from numpy.fft import fft, ifft, fftshift, fftfreq
import warnings
from astropy.utils.exceptions import AstropyWarning
# intentionally ignore NaN warnings from astropy - won't ignore other warnings
warnings.simplefilter('ignore', category=AstropyWarning)
try:
    import cv2
    no_opencv = False
except ImportError:
    msg = "Opencv python bindings are missing."
    warnings.warn(msg, ImportWarning)
    no_opencv = True

try:
    import torch as torch
    no_torch = False
except ImportError:
    msg = "Pytorch python bindings are missing"
    warnings.warn(msg, ImportWarning)
    no_torch = True


def frame_rotate(array, angle, imlib='vip-fft', interpolation='lanczos4',
                 cxy=None, border_mode='constant', mask_val=np.nan,
                 edge_blend=None, interp_zeros=False, ker=1):
    """Rotate a frame or 2D array.

    Parameters
    ----------
    array : numpy ndarray
        Input image, 2d array.
    angle : float
        Rotation angle.
    imlib : {'opencv', 'skimage', 'vip-fft', 'torch-fft'}, str optional
        Library used for image transformations. Opencv is faster than skimage or
        'vip-fft', but vip-fft slightly better preserves the flux in the image
        (followed by skimage with a biquintic interpolation). 'vip-fft'
        corresponds to the FFT-based rotation method described in [LAR97]_, and
        implemented in this module. Best results are obtained with images
        without any sharp intensity change (i.e. no numerical mask).
        Edge-blending and/or zero-interpolation may help if sharp transitions
        are unavoidable.
    interpolation : str, optional
        [Only used for imlib='opencv' or imlib='skimage']
        For Skimage the options are: 'nearneig', bilinear', 'biquadratic',
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
        of the frame.
    border_mode : {'constant', 'edge', 'symmetric', 'reflect', 'wrap'}, str opt
        Pixel extrapolation method for handling the borders. 'constant' for
        padding with zeros. 'edge' for padding with the edge values of the
        image. 'symmetric' for padding with the reflection of the vector
        mirrored along the edge of the array. 'reflect' for padding with the
        reflection of the vector mirrored on the first and last values of the
        vector along each axis. 'wrap' for padding with the wrap of the vector
        along the axis (the first values are used to pad the end and the end
        values are used to pad the beginning). Default is 'constant'.
    mask_val: flt, opt
        If any numerical mask in the image to be rotated, what are its values?
        Will only be used if a strategy to mitigate Gibbs effects is adopted -
        see below.
    edge_blend: str, opt {None,'noise','interp','noise+interp'}
        Whether to blend the edges, by padding nans then inter/extrapolate them
        with a gaussian filter. Slower but can significantly reduce ringing
        artefacts from Gibbs phenomenon, in particular if several consecutive
        rotations are involved in your image processing.

        - 'noise': pad with small amplitude noise inferred from neighbours
        - 'interp': interpolated from neighbouring pixels using Gaussian kernel.
        - 'noise+interp': sum both components above at masked locations.

        Original mask will be placed back after rotation.
    interp_zeros: bool, opt
        [only used if edge_blend is not None]
        Whether to interpolate zeros in the frame before (de)rotation. Not
        dealing with them can induce a Gibbs phenomenon near their location.
        However, this flag should be false if rotating a binary mask.
    ker: float, opt
        Size of the Gaussian kernel used for interpolation.

    Returns
    -------
    array_out : numpy ndarray
        Resulting frame.

    """
    if array.ndim != 2:
        raise TypeError('Input array is not a frame or 2d array')

    if edge_blend is None:
        edge_blend = ''

    if edge_blend != '' or imlib in ['vip-fft', 'torch-fft']:
        # fill with nans
        cy_ori, cx_ori = frame_center(array)
        y_ori, x_ori = array.shape
        if np.isnan(mask_val):
            mask_ori = np.where(np.isnan(array))
        else:
            mask_ori = np.where(array == mask_val)
        array_nan = array.copy()
        array_zeros = array.copy()
        if interp_zeros == 1 or mask_val != 0:  # set to nans for interpolation
            array_nan[np.where(array == mask_val)] = np.nan
        else:
            array_zeros[np.where(np.isnan(array))] = 0
        if 'noise' in edge_blend:
            # evaluate std and med far from the star, avoiding nans
            _, med, stddev = sigma_clipped_stats(array_nan, sigma=1.5,
                                                 cenfunc=np.nanmedian,
                                                 stdfunc=np.nanstd)

        # pad and interpolate, about 1.2x original size
        if imlib in ['vip-fft', 'torch-fft']:
            fac = 1.5
        else:
            fac = 1.1
        new_y = int(y_ori*fac)
        new_x = int(x_ori*fac)
        if y_ori % 2 != new_y % 2:
            new_y += 1
        if x_ori % 2 != new_x % 2:
            new_x += 1
        array_prep = np.empty([new_y, new_x])
        array_prep1 = np.zeros([new_y, new_x])
        array_prep[:] = np.nan
        if 'interp' in edge_blend:
            array_prep2 = array_prep.copy()
            med = 0  # local level will be added with Gaussian kernel
        if 'noise' in edge_blend:
            array_prep = np.random.normal(loc=med, scale=stddev,
                                          size=(new_y, new_x))
        cy, cx = frame_center(array_prep)
        y0_p = int(cy-cy_ori)
        y1_p = int(cy+cy_ori)
        if new_y % 2:
            y1_p += 1
        x0_p = int(cx-cx_ori)
        x1_p = int(cx+cx_ori)
        if new_x % 2:
            x1_p += 1
        if interp_zeros:
            array_prep[y0_p:y1_p, x0_p:x1_p] = array_nan.copy()
            array_prep1[y0_p:y1_p, x0_p:x1_p] = array_nan.copy()
        else:
            array_prep[y0_p:y1_p, x0_p:x1_p] = array_zeros.copy()
        # interpolate nans with a Gaussian filter
        if 'interp' in edge_blend:
            array_prep2[y0_p:y1_p, x0_p:x1_p] = array_nan.copy()
            cond1 = array_prep1 == 0
            cond2 = np.isnan(array_prep2)
            new_nan = np.where(cond1 & cond2)
            mask_nan = np.where(np.isnan(array_prep2))
            if not ker:
                ker = array_nan.shape[0]/5
            ker2 = 1
            array_prep_corr1 = frame_filter_lowpass(array_prep2, mode='gauss',
                                                    fwhm_size=ker)
            if 'noise' in edge_blend:
                array_prep_corr2 = frame_filter_lowpass(array_prep2,
                                                        mode='gauss',
                                                        fwhm_size=ker2)
                ori_nan = np.where(np.isnan(array_prep1))
                array_prep[ori_nan] = array_prep_corr2[ori_nan]
                array_prep[new_nan] += array_prep_corr1[new_nan]
            else:
                array_prep[mask_nan] = array_prep_corr1[mask_nan]

        # finally pad zeros for 4x larger images before FFT
        if imlib == 'vip-fft':
            array_prep, new_idx = frame_pad(array_prep, fac=4/fac, fillwith=0,
                                            full_output=True)
            y0 = new_idx[0]+y0_p
            y1 = new_idx[0]+y1_p
            x0 = new_idx[2]+x0_p
            x1 = new_idx[2]+x1_p
        else:
            y0 = y0_p
            y1 = y1_p
            x0 = x0_p
            x1 = x1_p
    else:
        array_prep = array.copy()

    # residual (non-interp) nans should be set to 0 to avoid bug in rotation
    array_prep[np.where(np.isnan(array_prep))] = 0

    y, x = array_prep.shape

    if cxy is None:
        cy, cx = frame_center(array_prep)
    else:
        cx, cy = cxy
        cond_imlib = imlib in ['vip-fft', 'torch-fft']
        if cond_imlib and (cy, cx) != frame_center(array_prep):
            msg = "'vip-fft' imlib does not yet allow for custom center to be "
            msg += " provided "
            raise ValueError(msg)

    if imlib == 'vip-fft':
        array_out = rotate_fft(array_prep, angle)

    elif imlib == 'skimage':
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
            raise ValueError('Skimage interpolation method not recognized')

        if border_mode not in ['constant', 'edge', 'symmetric', 'reflect',
                               'wrap']:
            raise ValueError('Skimage `border_mode` not recognized.')

        # for a non-constant image, normalize manually
        min_val = np.nanmin(array_prep)
        max_val = np.nanmax(array_prep)
        if min_val != max_val:
            norm = True
            im_temp = array_prep - min_val
            max_val = np.nanmax(im_temp)
            im_temp /= max_val
        else:
            norm = False
            im_temp = array_prep.copy()

        array_out = rotate(im_temp, angle, order=order, center=(cx, cy),
                           cval=0, mode=border_mode)

        if norm:
            array_out *= max_val
            array_out += min_val
        array_out = np.nan_to_num(array_out, copy=False)

    elif imlib == 'opencv':
        if no_opencv:
            msg = 'Opencv python bindings cannot be imported. Install opencv or'
            msg += ' set imlib to skimage'
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
            raise ValueError(f'Opencv interpolation method `{interpolation}` is not recognized')

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

        M = cv2.getRotationMatrix2D((cx, cy), angle, 1)
        array_out = cv2.warpAffine(array_prep.astype(np.float32), M, (x, y),
                                   flags=intp, borderMode=bormo)
    elif imlib == 'torch-fft':
        if no_torch:
            msg = 'Pytorch bindings cannot be imported. Install torch or'
            msg += ' set imlib to skimage'
            raise RuntimeError(msg)

        array_out = (tensor_rotate_fft(torch.unsqueeze(
            torch.from_numpy(array_prep), 0), angle)[0]).numpy()

    else:
        raise ValueError('Image transformation library not recognized')

    if edge_blend != '' or imlib in ['vip-fft', 'torch-fft']:
        array_out = array_out[y0:y1, x0:x1]  # remove padding
        array_out[mask_ori] = mask_val      # mask again original masked values

    return array_out


def cube_derotate(array, angle_list, imlib='vip-fft', interpolation='lanczos4',
                  cxy=None, nproc=1, border_mode='constant', mask_val=np.nan,
                  edge_blend=None, interp_zeros=False, ker=1):
    """Rotate a cube (3d array or image sequence) providing a vector or\
    corresponding angles.

    Serves for rotating an ADI sequence to a common north given a vector with
    the corresponding parallactic angles for each frame.

    Parameters
    ----------
    array : numpy.ndarray
        Input 3d array, cube.
    angle_list : list or 1D numpy.ndarray
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
    mask_val: flt, opt
        See the documentation of the ``vip_hci.preproc.frame_rotate`` function.
    edge_blend : str, optional
        See the documentation of the ``vip_hci.preproc.frame_rotate`` function.
    interp_zeros : str, optional
        See the documentation of the ``vip_hci.preproc.frame_rotate`` function.
    ker: int, optional
        See the documentation of the ``vip_hci.preproc.frame_rotate`` function.

    Returns
    -------
    array_der : numpy ndarray
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
                                        border_mode=border_mode,
                                        mask_val=mask_val,
                                        edge_blend=edge_blend,
                                        interp_zeros=interp_zeros, ker=ker)
    elif nproc > 1:

        res = pool_map(nproc, _frame_rotate_mp, iterable(array),
                       iterable(-angle_list), imlib, interpolation, cxy,
                       border_mode, mask_val, edge_blend, interp_zeros, ker)
        array_der = np.array(res)

    return array_der


def _frame_rotate_mp(frame, angle, imlib, interpolation, cxy, border_mode,
                     mask_val, edge_blend, interp_zeros, ker):
    framerot = frame_rotate(frame, angle, imlib, interpolation, cxy,
                            border_mode, mask_val, edge_blend, interp_zeros,
                            ker)
    return framerot


def _find_indices_adi(angle_list, frame, thr, nframes=None, out_closest=False,
                      truncate=False, max_frames=200):
    """Return the indices to be left in frames library for annular ADI median\
    subtraction, LOCI or annular PCA.

    Parameters
    ----------
    angle_list : numpy ndarray, 1d
        Vector of parallactic angle (PA) for each frame.
    frame : int
        Index of the current frame for which we are applying the PA threshold.
    thr : float
        PA threshold.
    nframes : int or None, optional
        Exact number of indices to be left. For annular median-ADI subtraction,
        where we keep the closest frames (after the PA threshold). If None then
        all the indices are returned (after the PA threshold).
    out_closest : bool, optional
        If True then the function returns the indices of the 2 closest frames.
    truncate : bool, optional
        Useful for annular PCA, when we want to discard too far away frames and
        avoid increasing the computational cost.
    max_frames : int, optional
        Max number of indices to be left. To be provided if ``truncate`` is
        True (used e.g. in pca_annular).

    Returns
    -------
    indices : numpy ndarray, 1d
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
                               list(range(ind3, ind4)), dtype='int32')
        else:
            # For annular PCA, returning all indices (after PA thresholding)
            half1 = range(0, index_prev)
            half2 = range(index_foll, n)
            indices = np.array(list(half1) + list(half2), dtype='int32')

            # The goal is to keep min(num_frames/2, ntrunc) in the library after
            # discarding those based on the PA threshold
            if truncate:
                thr = min(n-1, max_frames)
                all_indices = np.array(list(half1)+list(half2))
                if len(all_indices) > thr:
                    # then truncate and update indices
                    # first sort by dPA
                    dPA = np.abs(angle_list[all_indices]-angle_list[frame])
                    sort_indices = all_indices[np.argsort(dPA)]
                    # keep the ntrunc first ones
                    good_indices = sort_indices[:thr]
                    # sort again, this time by increasing indices
                    indices = np.sort(good_indices)

        return indices


def _compute_pa_thresh(ann_center, fwhm, delta_rot=1):
    """Compute the parallactic angle threshold [degrees].

    Replacing approximation: delta_rot * (fwhm/ann_center) / np.pi * 180
    """
    return np.rad2deg(2 * np.arctan(delta_rot * fwhm / (2 * ann_center)))


def _define_annuli(angle_list, ann, n_annuli, fwhm, radius_int, annulus_width,
                   delta_rot, n_segments, verbose, strict=False):
    """Define and return the requested annuli geometry: parallactic angle\
    threshold, inner radius and annulus center for each annulus."""
    verbosity = int(verbose)
    if ann == n_annuli - 1:
        inner_radius = radius_int + (ann * annulus_width - 1)
    else:
        inner_radius = radius_int + ann * annulus_width
    ann_center = inner_radius + (annulus_width / 2)
    pa_threshold = _compute_pa_thresh(ann_center, fwhm, delta_rot)
    mid_range = np.abs(np.amax(angle_list) - np.amin(angle_list)) / 2
    if pa_threshold >= mid_range - mid_range * 0.1:
        new_pa_th = float(mid_range - mid_range * 0.1)
        msg = 'WARNING: PA threshold {:.2f} is too big, recommended '
        msg += ' value for annulus {:.0f}: {:.2f}'
        if strict:
            if verbosity > 1:
                print(msg.format(pa_threshold, ann, new_pa_th))
        else:
            print('PA threshold {:.2f} is likely too big, will be set to '
                  '{:.2f}'.format(pa_threshold, new_pa_th))
            pa_threshold = new_pa_th

    if verbosity:
        if pa_threshold > 0:
            print('Ann {}    PA thresh: {:5.2f}    Ann center: '
                  '{:3.0f}    N segments: {} '.format(ann + 1, pa_threshold,
                                                      ann_center, n_segments))
        else:
            print('Ann {}    Ann center: {:3.0f}    N segments: '
                  '{} '.format(ann + 1, ann_center, n_segments))
    return pa_threshold, inner_radius, ann_center


def rotate_fft(array, angle):
    """Rotate a frame or 2D array using Fourier transforms.

    Rotation is equivalent to 3 consecutive linear shears, or 3 consecutive 1D
    FFT phase shifts. See details in [LAR97]_.

    Parameters
    ----------
    array : numpy ndarray
        Input image, 2d array.
    angle : float
        Rotation angle.

    Returns
    -------
    array_out : numpy ndarray
        Resulting frame.

    Note
    ----
    This method is slower than interpolation methods (e.g. opencv/lanczos4 or
    ndimage), but preserves the flux better (by construction it preserves the
    total power). It is more prone to large-scale Gibbs artefacts, so make sure
    no sharp edge nor bad pixels are present in the image to be rotated.

    Note
    ----
    Warning: if input frame has even dimensions, the center of rotation
    will NOT be between the 4 central pixels, instead it will be on the top
    right of those 4 pixels. Make sure your images are centered with
    respect to that pixel before rotation.

    """
    y_ori, x_ori = array.shape

    while angle < 0:
        angle += 360
    while angle > 360:
        angle -= 360

    # first convert to odd size before multiple 90deg rotations
    if not y_ori % 2 or not x_ori % 2:
        array_in = np.zeros([array.shape[0]+1, array.shape[1]+1])
        array_in[:-1, :-1] = array.copy()
    else:
        array_in = array.copy()

    if angle > 45:
        dangle = angle % 90
        if dangle > 45:
            dangle = -(90-dangle)
        nangle = np.rint(angle/90)
        array_in = np.rot90(array_in, nangle)
    else:
        dangle = angle

    # remove last row and column to make it even size before FFT
    array_in = array_in[:-1, :-1]

    a = np.tan(np.deg2rad(dangle)/2)
    b = -np.sin(np.deg2rad(dangle))

    ori_y, ori_x = array_in.shape

    cy, cx = frame_center(array)
    arr_xy = np.mgrid[0:ori_y, 0:ori_x]
    arr_y = arr_xy[0]-cy
    arr_x = arr_xy[1]-cx

    # TODO: make FFT padding work for other option than '0'.
    s_x = _fft_shear(array_in, arr_x, a, ax=1, pad=0)
    s_xy = _fft_shear(s_x, arr_y, b, ax=0, pad=0)
    s_xyx = _fft_shear(s_xy, arr_x, a, ax=1, pad=0)

    if y_ori % 2 or x_ori % 2:
        # set it back to original dimensions
        array_out = np.zeros([s_xyx.shape[0]+1, s_xyx.shape[1]+1])
        array_out[:-1, :-1] = np.real(s_xyx)
    else:
        array_out = np.real(s_xyx)

    return array_out


def _fft_shear(arr, arr_ori, c, ax, pad=0, shift_ini=True):
    ax2 = 1-ax % 2
    freqs = fftfreq(arr_ori.shape[ax2])
    sh_freqs = fftshift(freqs)
    arr_u = np.tile(sh_freqs, (arr_ori.shape[ax], 1))
    if ax == 1:
        arr_u = arr_u.T
    s_x = fftshift(arr)
    s_x = fft(s_x, axis=ax)
    s_x = fftshift(s_x)
    s_x = np.exp(-2j*np.pi*c*arr_u*arr_ori)*s_x
    s_x = fftshift(s_x)
    s_x = ifft(s_x, axis=ax)
    s_x = fftshift(s_x)

    return s_x


def tensor_rotate_fft(tensor, angle):
    """Rotate Tensor using Fourier transform phases.

    This is a Tensor adaptation of the vip-fft rotation function, and supposes
    that the frame(s) are perfectly centred. If input frame(s) has even
    dimensions, the center of rotation will NOT be between the 4 central pixels,
    instead it will be on the top right of those 4 pixels. Make sure your images
    are centered with respect to that pixel before rotation.

    Parameters
    ----------
    tensor : torch.Tensor
        Input image, 2d array.
    angle : float
        Rotation angle.

    Returns
    -------
    array_out : torch.Tensor
        Resulting frame.

    """
    y_ori, x_ori = tensor.shape[1:]

    while angle < 0:
        angle += 360
    while angle > 360:
        angle -= 360

    if angle > 45:
        dangle = angle % 90
        if dangle > 45:
            dangle = -(90 - dangle)
        nangle = int(np.rint(angle / 90))
        tensor_in = torch.rot90(tensor, nangle, [1, 2])
    else:
        dangle = angle
        tensor_in = tensor.clone()

    if y_ori % 2 or x_ori % 2:
        # NO NEED TO SHIFT BY 0.5px: FFT assumes rot. center on cx+0.5, cy+0.5!
        tensor_in = tensor_in[:, :-1, :-1]

    a = np.tan(np.deg2rad(dangle) / 2).item()
    b = -np.sin(np.deg2rad(dangle)).item()

    y_new, x_new = tensor_in.shape[1:]
    arr_xy = torch.from_numpy(np.mgrid[0:y_new, 0:x_new])
    cy, cx = frame_center(tensor[0])
    arr_y = arr_xy[0] - cy
    arr_x = arr_xy[1] - cx

    s_x = tensor_fft_shear(tensor_in, arr_x, a, ax=2)
    s_xy = tensor_fft_shear(s_x, arr_y, b, ax=1)
    s_xyx = tensor_fft_shear(s_xy, arr_x, a, ax=2)

    if y_ori % 2 or x_ori % 2:
        # set it back to original dimensions
        array_out = torch.zeros([1, s_xyx.shape[1]+1, s_xyx.shape[2]+1])
        array_out[0, :-1, :-1] = torch.real(s_xyx)
    else:
        array_out = torch.real(s_xyx)

    return array_out


def tensor_fft_shear(arr, arr_ori, c, ax):
    ax2 = 1 - (ax-1) % 2
    freqs = torch.fft.fftfreq(arr_ori.shape[ax2], dtype=torch.float64)
    sh_freqs = torch.fft.fftshift(freqs)
    arr_u = torch.tile(sh_freqs, (arr_ori.shape[ax-1], 1))
    if ax == 2:
        arr_u = torch.transpose(arr_u, 0, 1)
    s_x = torch.fft.fftshift(arr)
    s_x = torch.fft.fft(s_x, dim=ax)
    s_x = torch.fft.fftshift(s_x)
    s_x = torch.exp(-2j * torch.pi * c * arr_u * arr_ori) * s_x
    s_x = torch.fft.fftshift(s_x)
    s_x = torch.fft.ifft(s_x, dim=ax)
    s_x = torch.fft.fftshift(s_x)

    return s_x
