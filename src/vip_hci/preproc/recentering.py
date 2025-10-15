#! /usr/bin/env python
"""
Module containing functions for cubes frame registration.

.. [GUI08]
   | Guizar-Sicairos et al. 2008
   | **Efficient subpixel image registration algorithms**
   | *Optics Letters, Volume 33, Issue 2, p. 156*
   | `https://ui.adsabs.harvard.edu/abs/2008OptL...33..156G
     <https://ui.adsabs.harvard.edu/abs/2008OptL...33..156G>`_

.. [PUE15]
   | Pueyo et al. 2015
   | **Reconnaissance of the HR 8799 Exosolar System. II. Astrometry and Orbital
     Motion**
   | *The Astrophysical Journal, Volume 803, Issue 1, p. 31*
   | `https://arxiv.org/abs/1409.6388
     <https://arxiv.org/abs/1409.6388>`_

"""

__author__ = 'C. A. Gomez Gonzalez, V. Christiaens, G. Ruane, R. Farkas'
__all__ = ['frame_shift',
           'cube_shift',
           'frame_center_radon',
           'frame_center_satspots',
           'cube_recenter_satspots',
           'cube_recenter_radon',
           'cube_recenter_dft_upsampling',
           'cube_recenter_2dfit',
           'cube_recenter_via_speckles']

import warnings
from importlib.metadata import version

import numpy as np

try:
    import cv2
    no_opencv = False
except ImportError:
    msg = "Opencv python bindings are missing."
    warnings.warn(msg, ImportWarning)
    no_opencv = True

from hciplot import plot_frames
from scipy.ndimage import fourier_shift
from scipy.ndimage import shift
from skimage.transform import radon
from skimage.registration import phase_cross_correlation
from multiprocessing import cpu_count
from matplotlib import pyplot as plt

from ..config import time_ini, timing, Progressbar
from ..config.utils_conf import vip_figsize, check_array
from ..config.utils_conf import pool_map, iterable
from ..stats import frame_basic_stats
from ..var import (get_square, frame_center, get_annulus_segments,
                   fit_2dmoffat, fit_2dgaussian, fit_2dairydisk,
                   fit_2d2gaussian, cube_filter_lowpass, cube_filter_highpass,
                   frame_filter_highpass, frame_filter_lowpass)
from .cosmetics import cube_crop_frames, frame_crop
from .subsampling import cube_collapse


def frame_shift(array, shift_y, shift_x, imlib='vip-fft',
                interpolation='lanczos4', border_mode='reflect'):
    """Shift a 2D array by shift_y, shift_x.

    Parameters
    ----------
    array : numpy ndarray
        Input 2d array.
    shift_y, shift_x: float
        Shifts in y and x directions.
    imlib : {'opencv', 'ndimage-fourier', 'ndimage-interp', 'vip-fft'}, str opt
        Library or method used for performing the image shift.
        'ndimage-fourier' or 'vip-fft': does a fourier shift operation and
        preserves better the pixel values - therefore the flux and photometry
        (wrapper of scipy.ndimage.fourier_shift). Interpolation-based shift
        ('opencv' and 'ndimage-interp') is faster but less accurate than the
        fourier shift. 'opencv' is recommended when speed is critical.
    interpolation : str, optional
        Only used in case of imlib is set to 'opencv' or 'ndimage-interp'
        (Scipy.ndimage), where the images are shifted via interpolation.
        For Scipy.ndimage the options are: 'nearneig', bilinear', 'biquadratic',
        'bicubic', 'biquartic' or 'biquintic'. The 'nearneig' interpolation is
        the fastest and the 'biquintic' the slowest. The 'nearneig' is the
        poorer option for interpolation of noisy astronomical images.
        For Opencv the options are: 'nearneig', 'bilinear', 'bicubic' or
        'lanczos4'. The 'nearneig' interpolation is the fastest and the
        'lanczos4' the slowest and accurate. 'lanczos4' is the default for
        Opencv and 'biquartic' for Scipy.ndimage.
    border_mode : {'reflect', 'nearest', 'constant', 'mirror', 'wrap'}
        For 'opencv' and 'ndimage-interp', points outside the boundaries of the
        input are filled according to the value of this parameter.
        With 'reflect', the input is extended by reflecting about the edge of
        the last pixel. With 'nearest', the input is extended by replicating the
        last pixel. With 'constant', the input is extended by filling all values
        beyond the edge with zeros. With 'mirror', the input is extended by
        reflecting about the center of the last pixel. With 'wrap', the input is
        extended by wrapping around to the opposite edge. Default is 'reflect'.
        Note: for 'ndimage-fourier' default is 'wrap' (impossible to change),
        while border_mode is 'constant' (zeros) for 'vip-fft'.

    Returns
    -------
    array_shifted : numpy ndarray
        Shifted 2d array.

    """
    check_array(array, dim=2)
    image = array.copy()

    if imlib == 'ndimage-fourier':
        # Warning: default border mode is 'wrap' (cannot be changed)
        shift_val = (shift_y, shift_x)
        array_shifted = fourier_shift(np.fft.fftn(image), shift_val)
        array_shifted = np.fft.ifftn(array_shifted)
        array_shifted = array_shifted.real

    elif imlib == 'vip-fft':
        ny_ori, nx_ori = image.shape

        # First pad to avoid 'wrapping' values at the edges
        npad = int(np.ceil(np.amax(np.abs([shift_y, shift_x]))))
        cy_ori, cx_ori = frame_center(array)
        new_y = int(ny_ori+2*npad)
        new_x = int(nx_ori+2*npad)
        new_image = np.zeros([new_y, new_x], dtype=array.dtype)
        cy, cx = frame_center(new_image)
        y0 = int(cy-cy_ori)
        y1 = int(cy+cy_ori)
        if new_y % 2:
            y1 += 1
        x0 = int(cx-cx_ori)
        x1 = int(cx+cx_ori)
        if new_x % 2:
            x1 += 1
        new_image[y0:y1, x0:x1] = array.copy()
        p_y0 = npad
        p_x0 = npad
        npix = new_y

        # If non-square, add extra pad to make it square
        if new_y != new_x:
            if new_y > new_x:
                npix = new_y
                image = np.zeros([npix, npix])
                x0 = int(cy-cx)
                x1 = x0+new_x
                image[:, x0:x1] = new_image.copy()
                p_x0 += x0
            else:
                npix = new_x
                image = np.zeros([npix, npix])
                y0 = int(cx-cy)
                y1 = y0+new_y
                image[y0:y1] = new_image.copy()
                p_y0 += y0
            new_image = image.copy()

        # If odd, add an extra pad layer to make it even
        if npix % 2:
            npix += 1
            image = np.zeros([npix, npix])
            if shift_x > 0:
                x0 = 0
            else:
                x0 = 1
                p_x0 += 1
            if shift_y > 0:
                y0 = 0
            else:
                y0 = 1
                p_y0 += 1
            image[y0:y0+npix-1, x0:x0+npix-1] = new_image.copy()
            new_image = image.copy()

        # actual FT-based shift
        ramp = np.outer(np.ones(npix), np.arange(npix) - npix/2)
        tilt = (-2*np.pi / npix) * (shift_x*ramp + shift_y*ramp.T)
        fact = np.fft.fftshift(np.cos(tilt) + 1j*np.sin(tilt))

        image_ft = np.fft.fft2(new_image)  # no np.fft.fftshift applied!
        array_shifted = np.fft.ifft2(image_ft * fact).real

        # final crop to compensate padding
        array_shifted = array_shifted[p_y0:p_y0+ny_ori, p_x0:p_x0+nx_ori]

    elif imlib == 'ndimage-interp':
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
            raise ValueError('Scipy.ndimage interpolation method not '
                             'recognized')

        if border_mode not in ['reflect', 'nearest', 'constant', 'mirror',
                               'wrap']:
            raise ValueError('`border_mode` not recognized')

        array_shifted = shift(image, (shift_y, shift_x), order=order,
                              mode=border_mode)

    elif imlib == 'opencv':
        if no_opencv:
            msg = 'Opencv python bindings cannot be imported. Install opencv or'
            msg += ' set imlib to ndimage-fourier or ndimage-interp'
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
            raise ValueError('Opencv interpolation method not recognized')

        if border_mode == 'mirror':
            bormo = cv2.BORDER_REFLECT_101  # gfedcb|abcdefgh|gfedcba
        elif border_mode == 'reflect':
            bormo = cv2.BORDER_REFLECT  # fedcba|abcdefgh|hgfedcb
        elif border_mode == 'wrap':
            bormo = cv2.BORDER_WRAP  # cdefgh|abcdefgh|abcdefg
        elif border_mode == 'constant':
            bormo = cv2.BORDER_CONSTANT  # iiiiii|abcdefgh|iiiiiii
        elif border_mode == 'nearest':
            bormo = cv2.BORDER_REPLICATE  # aaaaaa|abcdefgh|hhhhhhh
        else:
            raise ValueError('`border_mode` not recognized')

        image = np.float32(image)
        y, x = image.shape
        M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        array_shifted = cv2.warpAffine(image, M, (x, y), flags=intp,
                                       borderMode=bormo)

    else:
        raise ValueError('Image transformation library not recognized')

    return array_shifted


def cube_shift(cube, shift_y, shift_x, imlib='vip-fft',
               interpolation='lanczos4', border_mode='reflect', nproc=None):
    """Shift the X-Y coordinates of a cube or 3D array by x and y values.

    Parameters
    ----------
    cube : numpy ndarray, 3d
        Input cube.
    shift_y, shift_x: float, list of floats or np.ndarray of floats
        Shifts in y and x directions for each frame. If the a single value is
        given then all the frames will be shifted by the same amount.
    imlib : str, optional
        See the documentation of the ``vip_hci.preproc.frame_shift`` function.
    interpolation : str, optional
        See the documentation of the ``vip_hci.preproc.frame_shift`` function.
    border_mode : str, optional
        See the documentation of the ``vip_hci.preproc.frame_shift`` function.
    nproc: int or None, optional
        Number of CPUs to use for multiprocessing. If None, will be
        automatically set to half the number of available CPUs.

    Returns
    -------
    cube_out : numpy ndarray, 3d
         Cube with shifted frames.

    """
    check_array(cube, dim=3)

    nfr = cube.shape[0]
    if np.isscalar(shift_x):
        shift_x = np.ones([nfr]) * shift_x
    if np.isscalar(shift_y):
        shift_y = np.ones([nfr]) * shift_y

    if nproc is None:
        nproc = cpu_count()//2

    if nproc == 1:
        cube_out = np.zeros_like(cube)
        for i in range(cube.shape[0]):
            cube_out[i] = frame_shift(cube[i], shift_y[i], shift_x[i], imlib,
                                      interpolation, border_mode)
    elif nproc > 1:
        res = pool_map(nproc, frame_shift, iterable(cube), iterable(shift_y),
                       iterable(shift_x), imlib, interpolation, border_mode)
        cube_out = np.array(res)

    return cube_out


def frame_center_satspots(array, xy, subi_size=19, sigfactor=6, shift=False,
                          imlib='vip-fft', interpolation='lanczos4',
                          fit_type='moff', filter_freq=(0, 0),
                          border_mode='reflect', debug=False, verbose=True):
    """Find the center of a frame with satellite spots (relevant e.g. for\
    VLT/SPHERE data).

    The method used to determine the center is by centroiding the 4 spots via a
    2D Gaussian fit and finding the intersection of the lines they create (see
    Notes). This method is very sensitive to the SNR of the satellite spots,
    therefore thresholding of the background pixels is performed. If the results
    are too extreme, the debug parameter will allow to see in depth what is
    going on with the fit (you may have to adjust the `sigfactor` for the
    background pixels thresholding).

    Parameters
    ----------
    array : numpy ndarray, 2d
        Image or frame.
    xy : tuple of 4 tuples of 2 elements
        Tuple with coordinates X,Y of the 4 satellite spots. When the spots are
        in an X configuration, the order is the following: top-left, top-right,
        bottom-left and bottom-right. When the spots are in an + (cross-like)
        configuration, the order is the following: top, right, left, bottom.
    subi_size : int, optional
        Size of subimage where the fitting is done.
    sigfactor : int, optional
        The background pixels will be thresholded before fitting a 2d Gaussian
        to the data using sigma clipped statistics. All values smaller than
        (MEDIAN + sigfactor*STDDEV) will be replaced by small random Gaussian
        noise.
    shift : bool, optional
        If True the image is shifted.
    imlib : str, optional
        See the documentation of the ``vip_hci.preproc.frame_shift`` function.
    interpolation : str, optional
        See the documentation of the ``vip_hci.preproc.frame_shift`` function.
    fit_type: str, optional {'gaus','moff'}
        Type of 2d fit to infer the centroid of the satellite spots.
    filter_freq: tuple of 2 floats, optional
        If the first (resp. second) element of the tuple is larger than 0,
        a high-pass (resp. low-pass) filter is applied to the image,
        before fitting the satellite spots. The elements should correspond to
        the fwhm_size of the frame_filter_highpass and frame_filter_lowpass
        functions, respectively. If both elements are non-zero, both high-pass
        and low-pass filter of the image are applied, in that order. This can be
        useful to better isolate the signal from the satellite spots.
    border_mode : {'reflect', 'nearest', 'constant', 'mirror', 'wrap'}
        Points outside the boundaries of the input are filled accordingly.
        With 'reflect', the input is extended by reflecting about the edge of
        the last pixel. With 'nearest', the input is extended by replicating the
        last pixel. With 'constant', the input is extended by filling all values
        beyond the edge with zeros. With 'mirror', the input is extended by
        reflecting about the center of the last pixel. With 'wrap', the input is
        extended by wrapping around to the opposite edge. Default is 'reflect'.
    imlib : str, optional
        See the documentation of the ``vip_hci.preproc.frame_shift`` function.
    interpolation : str, optional
        See the documentation of the ``vip_hci.preproc.frame_shift`` function.
    debug : bool, optional
        If True debug information is printed and plotted.
    verbose : bool, optional
        If True the intersection and shifts information is printed out.

    Returns
    -------
    array_rec : 2d numpy array
        Shifted images. *Only returned if ``shift=True``.*
    shifty, shiftx : floats
        Shift Y,X to get to the true center.
    ceny, cenx : floats
        Center Y,X coordinates of the true center. *Only returned if
        ``shift=True``.*

    Note
    ----
    We are solving a linear system:

    .. code-block:: python

        A1 * x + B1 * y = C1
        A2 * x + B2 * y = C2

    Cramer's rule - solution can be found in determinants:

    .. code-block:: python

        x = Dx/D
        y = Dy/D

    where D is main determinant of the system:

    .. code-block:: python

        A1 B1
        A2 B2

    and Dx and Dy can be found from matrices:

    .. code-block:: python

        C1 B1
        C2 B2

    and

    .. code-block:: python

        A1 C1
        A2 C2

    C column consequently substitutes the coef. columns of x and y

    L stores our coefs A, B, C of the line equations.

    .. code-block:: python

        For D: L1[0] L1[1]   for Dx: L1[2] L1[1]   for Dy: L1[0] L1[2]
               L2[0] L2[1]           L2[2] L2[1]           L2[0] L2[2]

    """
    def line(p1, p2):
        """Calculate coefs A, B, C of line equation by 2 points."""
        A = (p1[1] - p2[1])
        B = (p2[0] - p1[0])
        C = (p1[0] * p2[1] - p2[0] * p1[1])
        return A, B, -C

    def intersection(L1, L2):
        """Find intersection point (if any) of 2 lines provided by coefs."""
        D = L1[0] * L2[1] - L1[1] * L2[0]
        Dx = L1[2] * L2[1] - L1[1] * L2[2]
        Dy = L1[0] * L2[2] - L1[2] * L2[0]
        if D != 0:
            x = Dx / D
            y = Dy / D
            return x, y
        else:
            return None
    # --------------------------------------------------------------------------
    check_array(array, dim=2)
    if fit_type not in ['gaus', 'moff']:
        raise TypeError('fit_type is not recognized')
    if not isinstance(xy, (tuple, list)) or len(xy) != 4:
        raise TypeError('Input waffle spot coordinates in wrong format (must '
                        'be a tuple of 4 tuples')

    cy, cx = frame_center(array)
    centx = []
    centy = []
    subims = []

    if filter_freq[0] > 0:
        array = frame_filter_highpass(array, mode='gauss-subt',
                                      fwhm_size=filter_freq[0])
    if filter_freq[1] > 0:
        array = frame_filter_lowpass(array, fwhm_size=filter_freq[1])

    for i in range(len(xy)):
        sim, y, x = get_square(array, subi_size, xy[i][1], xy[i][0],
                               position=True, verbose=False)
        if fit_type == 'gaus':
            cent2dgy, cent2dgx = fit_2dgaussian(sim, crop=False, threshold=True,
                                                sigfactor=sigfactor,
                                                debug=debug, full_output=False)
        else:
            cent2dgy, cent2dgx = fit_2dmoffat(sim, crop=False, threshold=True,
                                              sigfactor=sigfactor, debug=debug,
                                              full_output=False)
        centx.append(cent2dgx + x)
        centy.append(cent2dgy + y)
        subims.append(sim)

    cent2dgx_1, cent2dgx_2, cent2dgx_3, cent2dgx_4 = centx
    cent2dgy_1, cent2dgy_2, cent2dgy_3, cent2dgy_4 = centy
    si1, si2, si3, si4 = subims

    if debug:
        plot_frames((si1, si2, si3, si4), colorbar=True)
        print('Centroids X,Y:')
        print(cent2dgx_1, cent2dgy_1)
        print(cent2dgx_2, cent2dgy_2)
        print(cent2dgx_3, cent2dgy_3)
        print(cent2dgx_4, cent2dgy_4)

    L1 = line([cent2dgx_1, cent2dgy_1], [cent2dgx_4, cent2dgy_4])
    L2 = line([cent2dgx_2, cent2dgy_2], [cent2dgx_3, cent2dgy_3])
    R = intersection(L1, L2)

    msgerr = "Check that the order of the tuples in `xy` is correct and"
    msgerr += " the satellite spots have good S/N"
    if R is not None:
        shiftx = cx - R[0]
        shifty = cy - R[1]

        if np.abs(shiftx) < cx * 2 and np.abs(shifty) < cy * 2:
            if debug or verbose:
                print('Intersection coordinates (X,Y):', R[0], R[1], '\n')
                print('Shifts (X,Y): {:.3f}, {:.3f}'.format(shiftx, shifty))

            if shift:
                array_rec = frame_shift(array, shifty, shiftx, imlib=imlib,
                                        interpolation=interpolation,
                                        border_mode=border_mode)
                return array_rec, shifty, shiftx, centy, centx
            else:
                return shifty, shiftx
        else:
            raise RuntimeError("Too large shifts. " + msgerr)
    else:
        raise RuntimeError("Something went wrong, no intersection found. " +
                           msgerr)


def cube_recenter_satspots(array, xy, subi_size=19, sigfactor=6, plot=True,
                           fit_type='moff', lbda=None, filter_freq=(0, 0),
                           border_mode='constant', imlib='vip-fft',
                           interpolation='lanczos4', debug=False, verbose=True,
                           full_output=False):
    """Recenter an image cube based on satellite spots.

    The function relies on ``frame_center_satspots`` to align each image of the
    cube individually (details in ``vip_hci.preproc.frame_center_satspots``).
    The function returns the recentered image cube, abd can also plot the
    histogram of the shifts, and calculate its statistics. The latter can help
    to assess the dispersion of the star center by using waffle/satellite spots
    (like those in VLT/SPHERE images) and evaluate the uncertainty of the
    position of the center.

    Parameters
    ----------
    array : numpy ndarray, 3d
        Input cube.
    xy : tuple of 4 tuples of 2 elements
        Tuple with coordinates X,Y of the 4 satellite spots. When the spots are
        in an X configuration, the order is the following: top-left, top-right,
        bottom-left and bottom-right. When the spots are in an + (plus-like)
        configuration, the order is the following: top, right, left, bottom.
        If wavelength vector is not provided, assumes all sat spots of the cube
        are at a similar location. If wavelength is provided, only coordinates
        of the sat spots in the first channel should be provided. The boxes
        location in other channels will be scaled accordingly.
    subi_size : int, optional
        Size of subimage where the fitting is done.
    sigfactor : int, optional
        The background pixels will be thresholded before fitting a 2d Gaussian
        to the data using sigma clipped statistics. All values smaller than
        (MEDIAN + sigfactor*STDDEV) will be replaced by small random Gaussian
        noise.
    plot : bool, optional
        Whether to plot the shifts.
    fit_type: str, optional {'gaus','moff'}
        Type of 2d fit to infer the centroid of the satellite spots.
    lbda: 1d array or list, opt
        Wavelength vector. If provided, the subimages will be scaled accordingly
        to follow the motion of the satellite spots.
    filter_freq: tuple of 2 floats, optional
        If the first (resp. second) element of the tuple is larger than 0,
        a high-pass (resp. low-pass) filter is applied to the image,
        before fitting the satellite spots. The elements should correspond to
        the fwhm_size of the frame_filter_highpass and frame_filter_lowpass
        functions, respectively. If both elements are non-zero, both high-pass
        and low-pass filter of the image are applied, in that order.
        This can be useful to isolate the signal from the satellite spots.
    border_mode : {'reflect', 'nearest', 'constant', 'mirror', 'wrap'}
        Points outside the boundaries of the input are filled accordingly.
        With 'reflect', the input is extended by reflecting about the edge of
        the last pixel. With 'nearest', the input is extended by replicating the
        last pixel. With 'constant', the input is extended by filling all values
        beyond the edge with zeros. With 'mirror', the input is extended by
        reflecting about the center of the last pixel. With 'wrap', the input is
        extended by wrapping around to the opposite edge. Default is 'reflect'.
    imlib : str, optional
        See the documentation of the ``vip_hci.preproc.frame_shift`` function.
    interpolation : str, optional
        See the documentation of the ``vip_hci.preproc.frame_shift`` function.
    debug : bool, optional
        If True debug information is printed and plotted (fit and residuals,
        intersections and shifts). This has to be used carefully as it can
        produce too much output and plots.
    verbose : bool, optional
        Whether to print to stdout the timing and additional info.
    full_output : bool, optional
        Whether to return 2 1d arrays of shifts along with the recentered cube
        or not.

    Returns
    -------
    array_rec
        The shifted cube.
    shift_y, shift_x
        [full_output==True] Shifts Y,X to get to the true center for each image.
    sat_y, sat_x
        [full_output==True] Y,X positions of the satellite spots in each image.
        Order: top-left, top-right, bottom-left and bottom-right.
    """
    check_array(array, dim=3)

    if verbose:
        start_time = time_ini()

    n_frames = array.shape[0]
    shift_x = np.zeros((n_frames))
    shift_y = np.zeros((n_frames))
    sat_y = np.zeros([n_frames, 4])
    sat_x = np.zeros([n_frames, 4])
    array_rec = []

    if lbda is not None:
        cy, cx = frame_center(array[0])
        final_xy = []
        rescal = lbda/lbda[0]
        for i in range(n_frames):
            xy_new = []
            for s in range(4):
                xy_new.append(
                    (cx+rescal[i]*(xy[s][0]-cx), cy+rescal[i]*(xy[s][1]-cy)))
            xy_new = tuple(xy_new)
            final_xy.append(xy_new)
    else:
        final_xy = [xy for i in range(n_frames)]

    if verbose:
        print("Final xy positions for sat spots:", final_xy)
        print('Looping through the frames, fitting the intersections:')
    for i in Progressbar(range(n_frames), verbose=verbose):
        res = frame_center_satspots(array[i], final_xy[i], debug=debug,
                                    shift=True, subi_size=subi_size,
                                    sigfactor=sigfactor, fit_type=fit_type,
                                    filter_freq=filter_freq, imlib=imlib,
                                    interpolation=interpolation, verbose=False,
                                    border_mode=border_mode)
        array_rec.append(res[0])
        shift_y[i] = res[1]
        shift_x[i] = res[2]
        sat_y[i] = res[3]
        sat_x[i] = res[4]

    if verbose:
        timing(start_time)

    if plot:
        plt.figure(figsize=vip_figsize)
        plt.plot(shift_x, 'o-', label='Shifts in x', alpha=0.5)
        plt.plot(shift_y, 'o-', label='Shifts in y', alpha=0.5)
        plt.legend(loc='best')
        plt.grid('on', alpha=0.2)
        plt.ylabel('Pixels')
        plt.xlabel('Frame number')

        plt.figure(figsize=vip_figsize)
        b = int(np.sqrt(n_frames))
        la = 'Histogram'
        _ = plt.hist(shift_x, bins=b, alpha=0.5, label=la + ' shifts X')
        _ = plt.hist(shift_y, bins=b, alpha=0.5, label=la + ' shifts Y')
        plt.legend(loc='best')
        plt.ylabel('Bin counts')
        plt.xlabel('Pixels')

    if verbose:
        msg1 = 'MEAN X,Y: {:.3f}, {:.3f}'
        print(msg1.format(np.mean(shift_x), np.mean(shift_y)))
        msg2 = 'MEDIAN X,Y: {:.3f}, {:.3f}'
        print(msg2.format(np.median(shift_x), np.median(shift_y)))
        msg3 = 'STDDEV X,Y: {:.3f}, {:.3f}'
        print(msg3.format(np.std(shift_x), np.std(shift_y)))

    array_rec = np.array(array_rec)

    if full_output:
        return array_rec, shift_y, shift_x, sat_y, sat_x
    else:
        return array_rec


def frame_center_radon(array, cropsize=None, hsize_ini=1., step_ini=0.1,
                       n_iter=5, tol=0.1, mask_center=None, nproc=None,
                       satspots_cfg=None, theta_0=0, delta_theta=5,
                       gauss_fit=True, hpf=True, filter_fwhm=8, imlib='vip-fft',
                       interpolation='lanczos4', full_output=False,
                       verbose=True, plot=True, debug=False):
    """Find the center of a broadband (co-added) frame with speckles and\
    satellite spots elongated towards the star (center).

    The function uses the Radon transform implementation from scikit-image, and
    follow the algorithm presented in [PUE15]_.

    Parameters
    ----------
    array : numpy ndarray
        Input 2d array or image.
    cropsize : None or odd int, optional
        Size in pixels of the cropped central area of the input array that will
        be used. It should be large enough to contain the bright elongated
        speckle or satellite spots.
    hsize_ini : float, optional
        Size of the box for the grid search for first centering iteration. The
        frame is shifted to each direction from the center in a hsize length
        with a given step.
    step_ini : float, optional
        The step of the coordinates change in the first step. Note: should not
        be too fine for efficiency as it is automatically refined at each step.
    n_iter : int, optional
        Number of iterations for finer recentering. At each step, a finer
        step is considered based on the amplitude of the shifts found in the
        previous step. Iterations are particularly relevant when mask_center is
        not None, as the masked area will change from one iteration to the next.
    tol : float, optional
        Absolute tolerance on relative shift from one iteration to the next to
        consider convergence. If the absolute value of the shift is found to be
        less than tol, the iterative algorithm is stopped.
    mask_center : None or int, optional
        If None the central area of the frame is kept. If int a centered zero
        mask will be applied to the frame. By default the center isn't masked.
    nproc : int, optional
        Number of processes for parallel computing. If None the number of
        processes will be set to cpu_count()/2.
    satspots_cfg: None or str ('x', '+' or 'custom'), opt
        If satellite spots are present, provide a string corresponding to the
        configuration of the satellite spots: as a cross ('x'), as a
        plus sign ('+') or 'custom' (provide theta_0). Leave to None if no
        satellite spots present. Note: setting satspots_cfg to non-None value
        leads to varying performance depending on dataset.
    theta_0: float between [0,90[, optional
        Azimuth of the first satellite spot. Only considered if satspots_cfg is
        set to 'custom'.
    delta_theta: float, optional
        Azimuthal half-width in degrees of the slices considered along a '+' or
        'x' pattern to calculate the Radon transform. E.g. if set to 5 for 'x'
        configuration, it will consider slices from 40 to 50 deg in each
        quadrant.
    hpf: bool, optional
        Whether to high-pass filter the images
    filter_fwhm: float, optional
        In case of high-pass filtering, this is the FWHM of the low-pass filter
        used for subtraction to the original image to get the high-pass
        filtered image (i.e. should be >~ 2 x FWHM).
    imlib : str, optional
        See the documentation of the ``vip_hci.preproc.frame_shift`` function.
    interpolation : str, optional
        See the documentation of the ``vip_hci.preproc.frame_shift`` function.
    full_output: bool, optional
        Whether to also return the cost map, and uncertainty on centering.
    verbose : bool optional
        Whether to print to stdout some messages and info.
    plot : bool, optional
        Whether to plot the radon cost function.
    debug : bool, optional
        Whether to print and plot intermediate info.

    Returns
    -------
    optimy, optimx : floats
        Values of the Y, X coordinates of the center of the frame based on the
        radon optimization. (always returned)
    dxy : float
        [full_output=True] Uncertainty on center in pixels.
    cost_bound : 2d numpy array
        [full_output=True] Radon cost function surface.

    """
    if array.ndim != 2:
        raise TypeError('Input array is not a frame or 2d array')

    if verbose:
        start_time = time_ini()

    def _center_radon(array, cropsize=None, hsize=1., step=0.1,
                      mask_center=None, nproc=None, satspots_cfg=None,
                      theta_0=0, d_theta=5, gauss_fit=False,
                      imlib='vip-fft', interpolation='lanczos4',
                      verbose=True, plot=True, debug=False):

        frame = array.copy()
        ori_cent_y, ori_cent_x = frame_center(frame)
        if cropsize is not None:
            if not cropsize % 2:
                raise TypeError("If not None, cropsize should be odd integer")
            frame = frame_crop(frame, cropsize, verbose=False)
        listyx = np.linspace(start=-hsize, stop=hsize, num=int(2*hsize/step)+1,
                             endpoint=True)
        if not mask_center:
            radint = 0
        else:
            if not isinstance(mask_center, int):
                raise TypeError
            radint = mask_center

        coords = [(y, x) for y in listyx for x in listyx]
        #  coords = [(x, y) for y in listyx for x in listyx]
        cent, _ = frame_center(frame)

        frame = get_annulus_segments(frame, radint, cent-radint, mode="mask")[0]

        if debug:
            if satspots_cfg is not None:
                samples = 10
                if satspots_cfg == 'x':
                    theta = np.hstack((np.linspace(start=45-d_theta,
                                                   stop=45+d_theta,
                                                   num=samples,
                                                   endpoint=False),
                                       np.linspace(start=135-d_theta,
                                                   stop=135+d_theta,
                                                   num=samples,
                                                   endpoint=False),
                                       np.linspace(start=225-d_theta,
                                                   stop=225+d_theta,
                                                   num=samples,
                                                   endpoint=False),
                                       np.linspace(start=315-d_theta,
                                                   stop=315+d_theta,
                                                   num=samples,
                                                   endpoint=False)))
                elif satspots_cfg == '+':
                    theta = np.hstack((np.linspace(start=-d_theta,
                                                   stop=d_theta,
                                                   num=samples,
                                                   endpoint=False),
                                       np.linspace(start=90-d_theta,
                                                   stop=90+d_theta,
                                                   num=samples,
                                                   endpoint=False),
                                       np.linspace(start=180-d_theta,
                                                   stop=180+d_theta,
                                                   num=samples,
                                                   endpoint=False),
                                       np.linspace(start=270-d_theta,
                                                   stop=270+d_theta,
                                                   num=samples,
                                                   endpoint=False)))
                elif satspots_cfg == 'custom':
                    theta = np.hstack((np.linspace(start=90-theta_0-d_theta,
                                                   stop=90-theta_0+d_theta,
                                                   num=samples, endpoint=False),
                                       np.linspace(start=180-theta_0-d_theta,
                                                   stop=180-theta_0+d_theta,
                                                   num=samples, endpoint=False),
                                       np.linspace(start=270-theta_0-d_theta,
                                                   stop=270-theta_0+d_theta,
                                                   num=samples, endpoint=False),
                                       np.linspace(start=360-theta_0-d_theta,
                                                   stop=360-theta_0+d_theta,
                                                   num=samples,
                                                   endpoint=False)))
                else:
                    msg = "If not None, satspots_cfg can only be 'x' or '+'."
                    raise ValueError(msg)
                sinogram = radon(frame, theta=theta, circle=True)
                plot_frames((frame, sinogram))
                # print(np.sum(np.abs(sinogram[int(cent), :])))
            else:
                theta = np.linspace(start=0, stop=360, num=int(cent*2),
                                    endpoint=False)
                sinogram = radon(frame, theta=theta, circle=True)
                plot_frames((frame, sinogram))
                # print(np.sum(np.abs(sinogram[int(cent), :])))

        if nproc is None:
            nproc = cpu_count() // 2    # Hyper-threading doubles the # of cores

        if nproc == 1:
            costf = []
            for coord in coords:
                res = _radon_costf(frame, cent, radint, coord, satspots_cfg,
                                   theta_0, d_theta, imlib, interpolation)
                costf.append(res)
            costf = np.array(costf)
        elif nproc > 1:
            res = pool_map(nproc, _radon_costf, frame, cent, radint,
                           iterable(coords), satspots_cfg, theta_0, d_theta,
                           imlib, interpolation)
            costf = np.array(res)

        if verbose:
            msg = 'Done {} radon transform calls distributed in {} processes'
            print(msg.format(len(coords), nproc))

        cost_bound = costf.reshape(listyx.shape[0], listyx.shape[0])
        if plot:
            plt.contour(cost_bound, cmap='CMRmap', origin='lower')
            plt.imshow(cost_bound, cmap='CMRmap', origin='lower',
                       interpolation='nearest')
            plt.colorbar()
            plt.grid('off')
            plt.show()

        if gauss_fit:  # or full_output:
            # fit a 2d gaussian to the surface
            fit_res = fit_2dgaussian(cost_bound-np.amin(cost_bound), crop=False,
                                     threshold=False, sigfactor=3, debug=debug,
                                     full_output=True)
            # optimal shift -> optimal position
            opt_yind = float(fit_res['centroid_y'].iloc[0])
            opt_xind = float(fit_res['centroid_x'].iloc[0])
            opt_yshift = -hsize + opt_yind*step
            opt_xshift = -hsize + opt_xind*step
            optimy = ori_cent_y - opt_yshift
            optimx = ori_cent_x - opt_xshift

            # find uncertainty on centering
            unc_y = float(fit_res['fwhm_y'].iloc[0])*step
            unc_x = float(fit_res['fwhm_x'].iloc[0])*step
            dyx = (unc_y, unc_x)  # np.sqrt(unc_y**2 + unc_x**2)

        # Replace the position found by Gaussian fit
        if not gauss_fit:
            # OLD CODE:
            argm = np.argmax(costf)  # index of 1st max in 1d cost function
            opt_yshift, opt_xshift = coords[argm]

            # maxima in the 2d cost function surface
            # num_max = np.where(cost_bound == cost_bound.max())[0].shape[0]
            # ind_maxy, ind_maxx = np.where(cost_bound == cost_bound.max())
            # argmy = ind_maxy[int(np.ceil(num_max/2)) - 1]
            # argmx = ind_maxx[int(np.ceil(num_max/2)) - 1]
            # y_grid = np.array(coords)[:, 0].reshape(listyx.shape[0],
            #                                         listyx.shape[0])
            # x_grid = np.array(coords)[:, 1].reshape(listyx.shape[0],
            #                                         listyx.shape[0])
            # optimy = ori_cent_y-y_grid[argmy, 0]  # subtract optimal shift
            # optimx = ori_cent_x-x_grid[0, argmx]  # subtract optimal shift
            optimy = ori_cent_y - opt_yshift
            optimx = ori_cent_x - opt_xshift
            dyx = (step, step)

        if verbose:
            print('Cost function max: {}'.format(costf.max()))
            # print('Cost function # maxima: {}'.format(num_max))
            m = 'Finished grid search radon optimization: dy={:.3f}, dx={:.3f}'
            print(m.format(opt_yshift, opt_xshift))
            timing(start_time)

        return optimy, optimx, opt_yshift, opt_xshift, dyx, cost_bound

    # high-pass filtering if requested
    if hpf:
        array = frame_filter_highpass(array, mode='gauss-subt',
                                      fwhm_size=filter_fwhm)

    ori_cent_y, ori_cent_x = frame_center(array)
    hsize = hsize_ini
    step = step_ini
    opt_yshift = 0
    opt_xshift = 0
    for i in range(n_iter):
        if verbose:
            print("*** Iteration {}/{} ***".format(i+1, n_iter))
        res = _center_radon(array, cropsize=cropsize, hsize=hsize, step=step,
                            mask_center=mask_center, nproc=nproc,
                            satspots_cfg=satspots_cfg, theta_0=theta_0,
                            d_theta=delta_theta, gauss_fit=gauss_fit,
                            imlib=imlib, interpolation=interpolation,
                            verbose=verbose, plot=plot, debug=debug)
        _, _, y_shift, x_shift, dyx, cost_bound = res
        array = frame_shift(array, y_shift, x_shift, imlib=imlib,
                            interpolation=interpolation)
        opt_yshift += y_shift
        opt_xshift += x_shift

        abs_shift = np.sqrt(y_shift**2 + x_shift**2)
        if abs_shift < tol:
            if i == 0:
                msg = "Null shifts found at first iteration for step = {}. Try"
                msg += " with a finer step."
                raise ValueError(msg.format(step))
            else:
                msg = "Convergence found after {} iterations (final step = {})."
            print(msg.format(i+1, step))
            break
        # refine box - OR NOT?!
        # max_sh = np.amax(np.abs(np.array([y_shift, x_shift])))
        hsize *= 0.75  # *max_sh
        step *= 0.75

    optimy = ori_cent_y+opt_yshift  # ORI: -
    optimx = ori_cent_x+opt_xshift  # ORI: -

    if verbose:
        print("Star (x,y) location: {:.2f}, {:.2f}".format(optimx, optimy))
        print("Final (x,y) shifts: {:.2f}, {:.2f}".format(opt_xshift,
                                                          opt_yshift))

    if full_output:
        return optimy, optimx, dyx, cost_bound
    else:
        return optimy, optimx


def _radon_costf(frame, cent, radint, coords, satspots_cfg=None, theta_0=0,
                 delta_theta=5, imlib='vip-fft', interpolation='lanczos4'):
    """Calculate Radon cost function used in frame_center_radon()."""
    frame_shifted = frame_shift(frame, coords[0], coords[1], imlib=imlib,
                                interpolation=interpolation)
    frame_shifted_ann = get_annulus_segments(frame_shifted, radint,
                                             cent-radint, mode="mask")[0]

    if satspots_cfg is None:
        theta = np.linspace(start=0, stop=360, num=frame_shifted_ann.shape[0],
                            endpoint=False)
    elif satspots_cfg == '+':
        samples = 10
        theta = np.hstack((np.linspace(start=-delta_theta, stop=delta_theta,
                                       num=samples, endpoint=False),
                           np.linspace(start=90-delta_theta,
                                       stop=90+delta_theta, num=samples,
                                       endpoint=False),
                           np.linspace(start=180-delta_theta,
                                       stop=180+delta_theta, num=samples,
                                       endpoint=False),
                           np.linspace(start=270-delta_theta,
                                       stop=270+delta_theta, num=samples,
                                       endpoint=False)))
    elif satspots_cfg == 'x':
        samples = 10
        theta = np.hstack((np.linspace(start=45-delta_theta,
                                       stop=45+delta_theta, num=samples,
                                       endpoint=False),
                           np.linspace(start=135-delta_theta,
                                       stop=135+delta_theta, num=samples,
                                       endpoint=False),
                           np.linspace(start=225-delta_theta,
                                       stop=225+delta_theta, num=samples,
                                       endpoint=False),
                           np.linspace(start=315-delta_theta,
                                       stop=315+delta_theta, num=samples,
                                       endpoint=False)))
    elif satspots_cfg == 'custom':
        samples = 10
        theta = np.hstack((np.linspace(start=theta_0-delta_theta,
                                       stop=theta_0+delta_theta,
                                       num=samples, endpoint=False),
                           np.linspace(start=theta_0+90-delta_theta,
                                       stop=theta_0+90+delta_theta,
                                       num=samples, endpoint=False),
                           np.linspace(start=theta_0+180-delta_theta,
                                       stop=theta_0+180+delta_theta,
                                       num=samples, endpoint=False),
                           np.linspace(start=theta_0+270-delta_theta,
                                       stop=theta_0+270+delta_theta,
                                       num=samples, endpoint=False)))
    sinogram = radon(frame_shifted_ann, theta=theta, circle=True)
    # costf = np.sum(np.abs(sinogram[int(cent), :])) # ORI DEF
    # rather consider the sum of the 4 top values?
    qstep = len(theta)//4
    sort_sin = []
    for i in range(4):
        sort_sin.append(np.nanmax(sinogram[int(cent), i*qstep:(i+1)*qstep]))
    costf = np.nansum(sort_sin)
    return costf


def cube_recenter_radon(array, full_output=False, verbose=True, imlib='vip-fft',
                        interpolation='lanczos4', border_mode='reflect',
                        nproc=None, **kwargs):
    """Recenter a cube using the Radon transform, as in [PUE15]_.

    The function loops through its frames, relying on the ``frame_center_radon``
    function for the recentering.

    Parameters
    ----------
    array : numpy ndarray
        Input 3d array or cube.
    full_output : {False, True}, bool optional
        If True the recentered cube is returned along with the y and x shifts.
    verbose : {True, False}, bool optional
        Whether to print timing and intermediate information to stdout.
    imlib : str, optional
        See the documentation of the ``vip_hci.preproc.frame_shift`` function.
    interpolation : str, optional
        See the documentation of the ``vip_hci.preproc.frame_shift`` function.
    border_mode : {'reflect', 'nearest', 'constant', 'mirror', 'wrap'}
        Points outside the boundaries of the input are filled accordingly.
        With 'reflect', the input is extended by reflecting about the edge of
        the last pixel. With 'nearest', the input is extended by replicating the
        last pixel. With 'constant', the input is extended by filling all values
        beyond the edge with zeros. With 'mirror', the input is extended by
        reflecting about the center of the last pixel. With 'wrap', the input is
        extended by wrapping around to the opposite edge. Default is 'reflect'.
    nproc : int, optional
        Number of processes for parallel computing. If None the number of
        processes will be set to cpu_count()/2.
    kwargs:
        Other optional parameters for ``vip_hci.preproc.frame_center_radon``
        function, such as cropsize, hsize, step, satspots_cfg, mask_center,
        hpf, filter_fwhm, nproc or debug.


    Returns
    -------
    array_rec : 3d ndarray
        Recentered cube.
    y, x : 1d arrays of floats
        [full_output] Shifts in y and x.
    dyx: 1d array of floats
        [full_output] Array of uncertainty on center in pixels.

    """
    check_array(array, dim=3)
    if nproc is None:
        nproc = int(cpu_count() / 2)

    if verbose:
        start_time = time_ini()

    n_frames = array.shape[0]
    x = np.zeros((n_frames))
    y = np.zeros((n_frames))
    dyx = np.zeros((n_frames, 2))
    cy, cx = frame_center(array[0])
    array_rec = array.copy()

    for i in Progressbar(range(n_frames), desc="Recentering frames...",
                         verbose=verbose):
        res = frame_center_radon(array[i], verbose=False, plot=False,
                                 imlib=imlib, interpolation=interpolation,
                                 full_output=True, nproc=nproc, **kwargs)
        y[i] = res[0]
        x[i] = res[1]
        dyx[i] = res[2]
        array_rec[i] = frame_shift(array[i], cy-y[i], cx-x[i], imlib=imlib,
                                   interpolation=interpolation,
                                   border_mode=border_mode)

    if verbose:
        timing(start_time)

    if full_output:
        return array_rec, y-cy, x-cx, dyx
    else:
        return array_rec


def cube_recenter_dft_upsampling(array, upsample_factor=100, subi_size=None,
                                 center_fr1=None, negative=False, fwhm=4,
                                 imlib='vip-fft', interpolation='lanczos4',
                                 mask=None, border_mode='reflect', log=False,
                                 collapse='median', full_output=False,
                                 verbose=True, nproc=None, save_shifts=False,
                                 debug=False, plot=True, **collapse_args):
    """Recenter a cube of frames using the DFT upsampling method as proposed\
    in [GUI08]_ and implemented in the ``phase_cross_correlation`` function\
    from scikit-image.

    The algorithm (DFT upsampling) obtains an initial estimate of the
    cross-correlation peak by an FFT and then refines the shift estimation by
    upsampling the DFT only in a small neighborhood of that estimate by means
    of a matrix-multiply DFT.

    Optionally, after alignment of all images to the first one, a 2D Gaussian
    fit can be made to the mean image to recenter them based on the location of
    the Gaussian centroid. This second stage is performed if subi_size is not
    None.

    Parameters
    ----------
    array : numpy ndarray
        Input cube.
    upsample_factor : int, optional
        Upsampling factor (default 100). Images will be registered to within
        1/upsample_factor of a pixel. The larger the slower the algorithm.
    subi_size : int or None, optional
        Size of the square subimage sides in pixels, used to find the centroid
        of the mean aligned cube image (i.e. after DFT-based registration). If
        subi_size is None then the first frame is assumed to be
        centered already.
    center_fr1 = (cy_1, cx_1) : Tuple, optional
        [subi_size != None] Coordinates of the center of the subimage for
        fitting a 2d Gaussian. Since the first part of the function of the
        algorithm aligns all subsequent frames to the first one. This tuple
        should be the rough coordinates of the centroid in the first frame. If
        not provided, the function considers the center of the images.
    negative : bool, optional
        [subi_size != None] If True the final centroiding is done with a
        negative 2D Gaussian fit, instead of a positive one.
    fwhm : float, optional
        [subi_size != None] First guess of the FWHM in pixels for the Gaussian
        fit.
    nproc : int or None, optional
        Number of processes (>1) for parallel computing. If 1 then it runs in
        serial. If None the number of processes will be set to (cpu_count()/2).
    imlib : str, optional
        See the documentation of the ``vip_hci.preproc.frame_shift`` function.
    interpolation : str, optional
        See the documentation of the ``vip_hci.preproc.frame_shift`` function.
    mask: 2D np.ndarray, optional
        Binary mask indicating where the cross-correlation should be calculated
        in the images. If provided, should be the same size as array frames.
        [Note: requires skimage >= 0.18.0]
    border_mode : {'reflect', 'nearest', 'constant', 'mirror', 'wrap'}
        Points outside the boundaries of the input are filled accordingly.
        With 'reflect', the input is extended by reflecting about the edge of
        the last pixel. With 'nearest', the input is extended by replicating the
        last pixel. With 'constant', the input is extended by filling all values
        beyond the edge with zeros. With 'mirror', the input is extended by
        reflecting about the center of the last pixel. With 'wrap', the input is
        extended by wrapping around to the opposite edge. Default is 'reflect'.
    log : bool, optional
        Whether to run the cross-correlation algorithm on images converted in
        log scale. This can be useful to leverage the whole extent of the PSF
        and be less dominated by the brightest central pixels.
    collapse : {'median', 'mean', 'sum', 'max', 'trimmean', 'absmean', 'wmean'}
        Method used to collapse the aligned cube before 2D Gaussian fit. Should
        be an argument accepted by the ``vip_hci.preproc.cube_collapse``
        function.
    full_output : bool, optional
        Whether to return 2 1d arrays of shifts along with the recentered cube
        or not.
    verbose : bool, optional
        Whether to print to stdout the timing or not.
    save_shifts : bool, optional
        Whether to save the shifts to a file in disk.
    debug : bool, optional
        Whether to print to stdout the shifts or not.
    plot : bool, optional
        If True, the shifts are plotted.
    collapse_args: opt
        Additional options passed to the ``vip_hci.preproc.cube_collapse``
        function.

    Returns
    -------
    array_recentered : numpy ndarray
        The recentered cube.
    y : numpy ndarray
        [full_output=True] 1d array with the shifts in y.
    x : numpy ndarray
        [full_output=True] 1d array with the shifts in x.

    Note
    ----
    This function uses the implementation from scikit-image of the algorithm
    described in [GUI08]_. This algorithm registers two images (2-D rigid
    translation) within a fraction of a pixel specified by the user. Instead of
    computing a zero-padded FFT (fast Fourier transform), this code
    uses selective upsampling by a matrix-multiply DFT (discrete FT) to
    dramatically reduce computation time and memory without sacrificing
    accuracy. With this procedure all the image points are used to compute the
    upsampled cross-correlation in a very small neighborhood around its peak.

    """
    if verbose:
        start_time = time_ini()

    check_array(array, dim=3)
    if mask is not None:
        if mask.shape != array.shape[-2:]:
            msg = "If provided, mask should have same shape as frames"
            raise TypeError(msg)

    n_frames, sizey, sizex = array.shape
    if subi_size is not None:
        if center_fr1 is None:
            print("`center_fr1` not provided")
            print("Using the coordinates of the 1st frame center for "
                  "the Gaussian 2d fit")
            cy_1, cx_1 = frame_center(array[0])
        else:
            cy_1, cx_1 = center_fr1
        if not isinstance(subi_size, int):
            raise ValueError('subi_size must be an integer or None')
        if subi_size < fwhm:
            raise ValueError('`subi_size` (value in pixels) is too small')
        if sizey % 2 == 0:
            if subi_size % 2 != 0:
                subi_size += 1
                print('`subi_size` is odd (while frame size is even)')
                print('Setting `subi_size` to {} pixels'.format(subi_size))
        else:
            if subi_size % 2 == 0:
                subi_size += 1
                print('`subi_size` is even (while frame size is odd)')
                print('Setting `subi_size` to {} pixels'.format(subi_size))

    n_frames = array.shape[0]
    x = np.zeros((n_frames))
    y = np.zeros((n_frames))
    array_rec = array.copy()

    cy, cx = frame_center(array[0])

    # convert to log scale
    if log:
        array_rec -= (np.nanmin(array_rec)-1)
        array_rec = np.log(array_rec)

    # Finding the shifts with DFT upsampling of each frame wrt the first
    if nproc is None:
        nproc = cpu_count() // 2  # Hyper-threading doubles the # of cores

    if nproc == 1:
        for i in Progressbar(range(1, n_frames),
                             desc="frames", verbose=verbose):
            y[i], x[i], array_rec[i] = _shift_dft(array_rec, array_rec, i,
                                                  upsample_factor, mask,
                                                  interpolation, imlib,
                                                  border_mode)
    elif nproc > 1:
        res = pool_map(nproc, _shift_dft, array_rec, array_rec,
                       iterable(range(1, n_frames)), upsample_factor, mask,
                       interpolation, imlib, border_mode)
        res = np.array(res, dtype=object)

        y[1:] = res[:, 0]
        x[1:] = res[:, 1]
        array_rec[1:] = [frames for frames in res[:, 2]]
    else:
        raise ValueError("nproc should be an int > 0.")

    if debug:
        print("\nShifts in X and Y")
        for i in range(n_frames):
            print(x[i], y[i])

    # Centroiding mean frame with 2d gaussian and shifting (only necessary if
    # first frame was not well-centered)
    msg0 = "The rest of the frames will be shifted by cross-correlation wrt "
    msg0 += "the 1st"
    if subi_size is not None:
        # before 2D gaussian fit, take non-log images
        if log:
            array_rec = cube_shift(array, shift_y=y, shift_x=x, imlib=imlib,
                                   interpolation=interpolation, nproc=nproc)
        marray_al = cube_collapse(array_rec, mode=collapse, **collapse_args)
        y1, x1 = _centroid_2dg_frame([marray_al], 0, subi_size,
                                     cy_1, cx_1, negative, debug, fwhm)
        x[:] += cx - x1
        y[:] += cy - y1

        if verbose:
            msg = "Shift for first frame X,Y=({:.3f}, {:.3f})"
            print(msg.format(x[0], y[0]))
            print(msg0)
        if debug:
            titd = "original / shifted 1st frame subimage"
            plot_frames((frame_crop(array[0], subi_size, verbose=False),
                        frame_crop(array_rec[0], subi_size, verbose=False)),
                        grid=True, title=titd)
    else:
        if verbose:
            print("The first frame is assumed to be well centered wrt the"
                  "center of the array")
            print(msg0)

    array_rec = cube_shift(array, shift_y=y, shift_x=x, imlib=imlib,
                           interpolation=interpolation, nproc=nproc)

    if verbose:
        timing(start_time)

    if plot:
        plt.figure(figsize=vip_figsize)
        plt.plot(x, 'o-', label='shifts in x', alpha=0.5)
        plt.plot(y, 'o-', label='shifts in y', alpha=0.5)
        plt.legend(loc='best')
        plt.grid('on', alpha=0.2)
        plt.ylabel('Pixels')
        plt.xlabel('Frame number')

        plt.figure(figsize=vip_figsize)
        b = int(np.sqrt(n_frames))
        la = 'Histogram'
        _ = plt.hist(x, bins=b, alpha=0.5, label=la + ' shifts X')
        _ = plt.hist(y, bins=b, alpha=0.5, label=la + ' shifts Y')
        plt.legend(loc='best')
        plt.ylabel('Bin counts')
        plt.xlabel('Pixels')

    if save_shifts:
        np.savetxt('recent_dft_shifts.txt', np.transpose([y, x]), fmt='%f')
    if full_output:
        return array_rec, y, x
    else:
        return array_rec


def _shift_dft(array_rec, array, frnum, upsample_factor, mask, interpolation,
               imlib, border_mode):
    """Align images using a DFT-based cross-correlation algorithm, used in\
    cube_recenter_dft_upsampling.

    See the docstring of skimage.register.phase_cross_correlation for a
    description of the ``normalization`` parameter which was added in
    scikit-image 0.19. This should be set to None to maintain the original
    behaviour of _shift_dft.
    """
    shifts = phase_cross_correlation(array_rec[0], array[frnum],
                                     upsample_factor=upsample_factor,
                                     reference_mask=mask,
                                     normalization=None)
    # from skimage 0.22, phase_cross_correlation returns two more variables
    # in addition to the array of shifts
    if len(shifts) == 3:
        shifts = shifts[0]

    y_i, x_i = shifts
    array_rec_i = frame_shift(array[frnum], shift_y=y_i, shift_x=x_i,
                              imlib=imlib, interpolation=interpolation,
                              border_mode=border_mode)
    return y_i, x_i, array_rec_i


def cube_recenter_2dfit(array, xy=None, fwhm=4, subi_size=5, model='gauss',
                        nproc=1, imlib='vip-fft', interpolation='lanczos4',
                        offset=None, negative=False, threshold=False,
                        sigfactor=2, fix_neg=False, params_2g=None,
                        border_mode='reflect', save_shifts=False,
                        full_output=False, verbose=True, debug=False,
                        plot=True):
    """Recenter the frames of a cube such that the centroid of a fitted 2D\
    model is placed at the center of the frames.

    The shifts are found by fitting a 2D Gaussian, Moffat, Airy or double
    Gaussian (positive+negative), as set by ``model``, to a subimage centered at
    ``xy``. This assumes the frames don't have too large shifts (<5px). The
    frames are then shifted using the function frame_shift().

    Parameters
    ----------
    array : numpy ndarray
        Input cube.
    xy : tuple of integers or floats
        Integer coordinates of the center of the subimage. For the double
        Gaussian fit with fixed negative gaussian (``fix_neg=True``), this
        should correspond to the exact location of the center of the negative
        Gaussian (e.g. the center of the coronagraph mask). In that case a tuple
        of floats is accepted.
    fwhm : float or numpy ndarray
        FWHM size in pixels, either one value (float) that will be the same for
        the whole cube, or an array of floats with the same dimension as the
        0th dim of array, containing the fwhm for each channel (e.g. in the case
        of an IFS cube, where the FWHM varies with wavelength).
    subi_size : int, optional
        Size of the square subimage sides in pixels.
    model : str, optional
        Sets the type of fit to be used. 'gauss' for a 2d Gaussian fit,
        'moff' for a 2d Moffat fit, 'airy' for a 2d Airy disk fit, and
        '2gauss' for a 2d double Gaussian (positive+negative) fit.
    nproc : int or None, optional
        Number of processes (>1) for parallel computing. If 1 then it runs in
        serial. If None the number of processes will be set to (cpu_count()/2).
    imlib : str, optional
        See the documentation of the ``vip_hci.preproc.frame_shift`` function.
    interpolation : str, optional
        See the documentation of the ``vip_hci.preproc.frame_shift`` function.
    offset : tuple of floats, optional
        If None the region of the frames used for the 2d Gaussian/Moffat fit is
        shifted to the center of the images (2d arrays). If a tuple is given it
        serves as the offset of the fitted area wrt the center of the 2d arrays.
    negative : bool, optional
        If True a negative 2d Gaussian/Moffat fit is performed.
    fix_neg: bool, optional
        In case of a double gaussian fit, whether to fix the parameters of the
        megative gaussian. If True, they should be provided in params_2g.
    params_2g: None or dictionary, optional
        In case of a double gaussian fit, dictionary with either fixed or first
        guess parameters for the double gaussian. E.g.:
        params_2g = {'fwhm_neg': 3.5, 'fwhm_pos': (3.5,4.2), 'theta_neg': 48.,
        'theta_pos':145., 'neg_amp': 0.5}

        - fwhm_neg: float or tuple with fwhm of neg gaussian
        - fwhm_pos: can be a tuple for x and y axes of pos gaussian (replaces
          fwhm)
        - theta_neg: trigonometric angle of the x axis of the neg gaussian (deg)
        - theta_pos: trigonometric angle of the x axis of the pos gaussian (deg)
        - neg_amp: amplitude of the neg gaussian wrt the amp of the positive one

        Note: it is always recommended to provide theta_pos and theta_neg for a
        better fit.
    threshold : bool, optional
        If True the background pixels (estimated using sigma clipped statistics)
        will be replaced by small random Gaussian noise (recommended for 2g).
    sigfactor: float, optional
        If thresholding is performed, set the the threshold in terms of
        gaussian sigma in the subimage (will depend on your cropping size).
    border_mode : {'reflect', 'nearest', 'constant', 'mirror', 'wrap'}
        Points outside the boundaries of the input are filled accordingly.
        With 'reflect', the input is extended by reflecting about the edge of
        the last pixel. With 'nearest', the input is extended by replicating the
        last pixel. With 'constant', the input is extended by filling all values
        beyond the edge with zeros. With 'mirror', the input is extended by
        reflecting about the center of the last pixel. With 'wrap', the input is
        extended by wrapping around to the opposite edge. Default is 'reflect'.
    save_shifts : bool, optional
        Whether to save the shifts to a file in disk.
    full_output : bool, optional
        Whether to return 2 1d arrays of shifts along with the recentered cube
        or not.
    verbose : bool, optional
        Whether to print to stdout the timing or not.
    debug : bool, optional
        If True the details of the fitting are shown. Won't work when the cube
        contains >20 frames (as it might produce an extremely long output).
    plot : bool, optional
        If True, the shifts are plotted.

    Returns
    -------
    array_rec: numpy ndarray
        The recentered cube.
    y : numpy ndarray
        [full_output=True] 1d array with the shifts in y.
    x : numpy ndarray
        [full_output=True] 1d array with the shifts in x.

    """
    if verbose:
        start_time = time_ini()

    check_array(array, dim=3)

    n_frames, sizey, sizex = array.shape

    if not isinstance(subi_size, int):
        raise ValueError('`subi_size` must be an integer')

    if sizey % 2 == 0:
        if subi_size % 2 != 0:
            subi_size += 1
            print('`subi_size` is odd (while frame size is even)')
            print('Setting `subi_size` to {} pixels'.format(subi_size))
    else:
        if subi_size % 2 == 0:
            subi_size += 1
            print('`subi_size` is even (while frame size is odd)')
            print('Setting `subi_size` to {} pixels'.format(subi_size))

    if isinstance(fwhm, (float, int, np.float32, np.float64)):
        fwhm = np.ones(n_frames) * fwhm

    if debug and array.shape[0] > 20:
        msg = 'Debug with a big array will produce a very long output. '
        msg += 'Try with less than 20 frames in debug mode'
        raise RuntimeWarning(msg)

    if xy is not None:
        pos_x, pos_y = xy
        cond = model != '2gauss'
        if (not isinstance(pos_x, int) or not isinstance(pos_y, int)) and cond:
            raise TypeError('`xy` must be a tuple of integers')
    else:
        pos_y, pos_x = frame_center(array[0])

    cy, cx = frame_center(array[0])
    array_rec = np.empty_like(array)

    if model == 'gauss':
        func = _centroid_2dg_frame
    elif model == 'moff':
        func = _centroid_2dm_frame
    elif model == 'airy':
        func = _centroid_2da_frame
    elif model == '2gauss':
        func = _centroid_2d2g_frame
    else:
        raise ValueError('model not recognized')

    if nproc is None:
        nproc = cpu_count() // 2        # Hyper-threading doubles the # of cores

    if nproc == 1:
        res = []
        if verbose:
            print('2d {}-fitting'.format(model))
        for i in Progressbar(range(n_frames), desc="frames", verbose=verbose):
            if model == "2gauss":
                args = [array, i, subi_size, pos_y, pos_x, debug, fwhm[i],
                        fix_neg, params_2g, threshold, sigfactor]
            else:
                args = [array, i, subi_size, pos_y, pos_x, negative, debug,
                        fwhm[i], threshold, sigfactor]

            res.append(func(*args))
        #res = np.array(res, dtype=object)
    elif nproc > 1:
        if model == "2gauss":
            args = [array, iterable(range(n_frames)), subi_size, pos_y, pos_x,
                    debug, iterable(fwhm), fix_neg, params_2g, threshold,
                    sigfactor]
        else:
            args = [array, iterable(range(n_frames)), subi_size, pos_y, pos_x,
                    negative, debug, iterable(fwhm), threshold, sigfactor]
        res = pool_map(nproc, func, *args)
        #res = np.array(res, dtype=object)
    y = cy - np.array([res[i][0] for i in range(len(res))])
    x = cx - np.array([res[i][1] for i in range(len(res))])

    if model == "2gauss" and not fix_neg:
        y_neg = np.array([res[i][2] for i in range(len(res))])
        x_neg = np.array([res[i][3] for i in range(len(res))])
        fwhm_x = np.array([res[i][4] for i in range(len(res))])
        fwhm_y = np.array([res[i][5] for i in range(len(res))])
        fwhm_neg_x = np.array([res[i][6] for i in range(len(res))])
        fwhm_neg_y = np.array([res[i][7] for i in range(len(res))])
        theta = np.array([res[i][8] for i in range(len(res))])
        theta_neg = np.array([res[i][9] for i in range(len(res))])
        amp_pos = np.array([res[i][10] for i in range(len(res))])
        amp_neg = np.array([res[i][11] for i in range(len(res))])

    if offset is not None:
        offx, offy = offset
        y -= offy
        x -= offx

    for i in Progressbar(range(n_frames), desc="Shifting", verbose=verbose):
        if debug:
            print("\nShifts in X and Y")
            print(x[i], y[i])
        array_rec[i] = frame_shift(array[i], y[i], x[i], imlib=imlib,
                                   interpolation=interpolation,
                                   border_mode=border_mode)

    if verbose:
        timing(start_time)

    if plot:
        if nproc > 1:
            print("Warning: plots may not show well if nproc is set to a value")
            print(" different than 1.")
        plt.figure(figsize=vip_figsize)
        b = int(np.sqrt(n_frames))
        la = 'Histogram'
        _ = plt.hist(x, bins=b, alpha=0.5, label=la + ' shifts X')
        _ = plt.hist(y, bins=b, alpha=0.5, label=la + ' shifts Y')
        if model == "2gauss" and not fix_neg:
            _ = plt.hist(cx-x_neg, bins=b, alpha=0.5,
                         label=la + ' shifts X (neg gaussian)')
            _ = plt.hist(cy-y_neg, bins=b, alpha=0.5,
                         label=la + ' shifts Y (neg gaussian)')
        plt.legend(loc='best')
        plt.ylabel('Bin counts')
        plt.xlabel('Pixels')

        plt.figure(figsize=vip_figsize)
        plt.plot(x, 'o-', label='shifts in x', alpha=0.5)
        plt.plot(y, 'o-', label='shifts in y', alpha=0.5)
        plt.legend(loc='best')
        plt.grid('on', alpha=0.2)
        plt.ylabel('Pixels')
        plt.xlabel('Frame number')

    if save_shifts:
        np.savetxt('recent_gauss_shifts.txt', np.transpose([y, x]), fmt='%f')
    if full_output:
        if model == "2gauss" and not fix_neg:
            return (array_rec, y, x, y_neg, x_neg, fwhm_x, fwhm_y, fwhm_neg_x,
                    fwhm_neg_y, theta, theta_neg, amp_pos, amp_neg)

        return array_rec, y, x
    else:
        return array_rec


def _centroid_2dg_frame(cube, frnum, size, pos_y, pos_x, negative, debug,
                        fwhm, threshold=False, sigfactor=1):
    """Find the centroid with a 2d Gaussian fit in a frame of the cube."""
    sub_image, y1, x1 = get_square(cube[frnum], size=size, y=pos_y, x=pos_x,
                                   position=True)
    # negative gaussian fit
    if negative:
        sub_image = -sub_image + np.abs(np.min(-sub_image))

    y_i, x_i = fit_2dgaussian(sub_image, crop=False, fwhmx=fwhm, fwhmy=fwhm,
                              threshold=threshold, sigfactor=sigfactor,
                              debug=debug,
                              full_output=False)
    y_i = y1 + y_i
    x_i = x1 + x_i
    return y_i, x_i


def _centroid_2dm_frame(cube, frnum, size, pos_y, pos_x, negative, debug,
                        fwhm, threshold=False, sigfactor=1):
    """Find the centroid with a 2d Moffat fit in a frame of the cube."""
    sub_image, y1, x1 = get_square(cube[frnum], size=size, y=pos_y, x=pos_x,
                                   position=True)
    # negative fit
    if negative:
        sub_image = -sub_image + np.abs(np.min(-sub_image))

    y_i, x_i = fit_2dmoffat(sub_image, crop=False, fwhm=fwhm, debug=debug,
                            threshold=threshold, sigfactor=sigfactor,
                            full_output=False)
    y_i = y1 + y_i
    x_i = x1 + x_i
    return y_i, x_i


def _centroid_2da_frame(cube, frnum, size, pos_y, pos_x, negative, debug,
                        fwhm, threshold=False, sigfactor=1):
    """Find the centroid with a 2d Airy fit in a frame of the cube."""
    sub_image, y1, x1 = get_square(cube[frnum], size=size, y=pos_y, x=pos_x,
                                   position=True)
    # negative fit
    if negative:
        sub_image = -sub_image + np.abs(np.min(-sub_image))

    y_i, x_i = fit_2dairydisk(sub_image, crop=False, fwhm=fwhm,
                              threshold=threshold, sigfactor=sigfactor,
                              full_output=False, debug=debug)
    y_i = y1 + y_i
    x_i = x1 + x_i
    return y_i, x_i


def _centroid_2d2g_frame(cube, frnum, size, pos_y, pos_x, debug=False, fwhm=4,
                         fix_neg=True, params_2g=None, threshold=False,
                         sigfactor=1):
    """Find the centroid with a 2d double Gaussian fit in a cube frame."""
    size = min(cube[frnum].shape[0], cube[frnum].shape[1], size)
    if isinstance(params_2g, dict):
        fwhm_neg = params_2g.get('fwhm_neg', 0.8*fwhm)
        fwhm_pos = params_2g.get('fwhm_pos', 2*fwhm)
        theta_neg = params_2g.get('theta_neg', 0.)
        theta_pos = params_2g.get('theta_pos', 0.)
        neg_amp = params_2g.get('neg_amp', 1)

    res_DF = fit_2d2gaussian(cube[frnum], crop=True, cent=(pos_x, pos_y),
                             cropsize=size, fwhm_neg=fwhm_neg,
                             fwhm_pos=fwhm_pos, neg_amp=neg_amp,
                             fix_neg=fix_neg, theta_neg=theta_neg,
                             theta_pos=theta_pos, threshold=threshold,
                             sigfactor=sigfactor, full_output=True, debug=debug)
    y_i = res_DF['centroid_y']
    x_i = res_DF['centroid_x']

    if not fix_neg:
        y_neg = res_DF['centroid_y_neg']
        x_neg = res_DF['centroid_x_neg']
        fwhm_x = res_DF['fwhm_x']
        fwhm_y = res_DF['fwhm_y']
        fwhm_neg_x = res_DF['fwhm_x_neg']
        fwhm_neg_y = res_DF['fwhm_y_neg']
        theta = res_DF['theta']
        theta_neg = res_DF['theta_neg']
        amp_pos = res_DF['amplitude']
        amp_neg = res_DF['amplitude_neg']
        return (y_i, x_i, y_neg, x_neg, fwhm_x, fwhm_y, fwhm_neg_x, fwhm_neg_y,
                theta, theta_neg, amp_pos, amp_neg)
    return y_i, x_i


def cube_recenter_via_speckles(cube_sci, cube_ref=None, alignment_iter=5,
                               gammaval=1, min_spat_freq=0.5, max_spat_freq=3,
                               fwhm=4, upsample_factor=100, debug=False,
                               recenter_median=False, fit_type='gaus',
                               negative=True, crop=True, subframesize=25,
                               mask=None, ann_rad=0.5, ann_rad_search=False,
                               ann_width=0.5, collapse='median',
                               imlib='vip-fft', interpolation='lanczos4',
                               border_mode='reflect', log=True, plot=True,
                               full_output=False, nproc=1, **collapse_args):
    """Register frames based on the median speckle pattern.

    The function also optionally centers images based on the position of the
    vortex null in the median frame (through a negative Gaussian fit or an
    annulus fit). By default, images are filtered to isolate speckle spatial
    frequencies, and converted to log-scale before the cross-correlation based
    alignment. The image cube should already be centered within ~2px accuracy
    before being passed to this function (e.g. through an eyeball crop using
    ``vip_hci.preproc.cube_crop_frames``).

    Parameters
    ----------
    cube_sci : numpy ndarray
        Science cube.
    cube_ref : numpy ndarray
        Reference cube (e.g. for NIRC2 data in RDI mode).
    alignment_iter : int, optional
        Number of alignment iterations (recomputes median after each iteration).
    gammaval : int, optional
        Applies a gamma correction to emphasize speckles (useful for faint
        stars).
    min_spat_freq : float, optional
        Spatial frequency for low pass filter.
    max_spat_freq : float, optional
        Spatial frequency for high pass filter.
    fwhm : float, optional
        Full width at half maximum.
    upsample_factor: int, optional
        Upsampling factor (default 100). Images will be registered to within
        1/upsample_factor of a pixel. The larger the slower the algorithm.
    debug : bool, optional
        Outputs extra info.
    recenter_median : bool, optional
        Recenter the frames at each iteration based on a 2d fit.
    fit_type : str, optional
        If recenter_median is True, this is the model to which the image is
        fitted to for recentering. 'gaus' works well for NIRC2_AGPM data.
        'ann' works better for NACO+AGPM data.
    negative : bool, optional
        If True, uses a negative gaussian fit to determine the center of the
        median frame.
    crop: bool, optional
        Whether to calculate the recentering on a cropped version of the cube
        that is speckle-dominated (recommended).
    subframesize : int, optional
        Sub-frame window size used. Should cover the region where speckles are
        the dominant noise source.
    mask: 2D np.ndarray, optional
        Binary mask indicating where the cross-correlation should be calculated
        in the images. If provided, should be the same size as array frames.
    ann_rad: float, optional
        [if fit_type='ann'] The expected inner radius of the annulus in FWHM.
    ann_rad_search: bool
        [if fit_type='ann'] Whether to also search for optimal radius.
    ann_width: float, optional
        [if fit_type='ann'] The expected radial width of the annulus in FWHM.
    collapse : {'median', 'mean', 'sum', 'max', 'trimmean', 'absmean', 'wmean'}
        Method used to collapse the aligned cube before 2D Gaussian fit. Should
        be an argument accepted by the ``vip_hci.preproc.cube_collapse``
        function.
    imlib : str, optional
        Image processing library to use.
    interpolation : str, optional
        Interpolation method to use.
    border_mode : {'reflect', 'nearest', 'constant', 'mirror', 'wrap'}
        Points outside the boundaries of the input are filled accordingly.
        With 'reflect', the input is extended by reflecting about the edge of
        the last pixel. With 'nearest', the input is extended by replicating the
        last pixel. With 'constant', the input is extended by filling all values
        beyond the edge with zeros. With 'mirror', the input is extended by
        reflecting about the center of the last pixel. With 'wrap', the input is
        extended by wrapping around to the opposite edge. Default is 'reflect'.
    log : bool
        Whether to run the cross-correlation algorithm on images converted in
        log scale. This can be useful to leverage the whole extent of the PSF
        and be less dominated by the brightest central pixels.
    plot : bool, optional
        If True, the shifts are plotted.
    full_output: bool, optional
        Whether to return more variables, useful for debugging.
    nproc: int or None, optional
        Number of CPUs to use if frame shifts are to be done in multiprocessing.
        None assigns half of available CPUs.
    **collapse_args:
        Additional arguments passed to the ``vip_hci.preproc.cube_collapse``
        function.

    Returns
    -------
    cube_reg_sci : numpy 3d ndarray
        Registered science cube
    cube_reg_ref : numpy 3d ndarray
        [cube_ref!=None] Cube registered to science frames
    cube_sci_lpf : numpy 3d ndarray
        [full_output=True] Low+high-pass filtered science cube
    cube_stret : numpy 3d ndarray
        [full_output=True] cube_stret with stretched values used for cross-corr
    cum_x_shifts_sci: numpy 1d array
        [full_output=True] Vector of x shifts for science frames
    cum_y_shifts_sci: numpy 1d array
        [full_output=True] Vector of y shifts for science frames
    cum_x_shifts_ref: numpy 1d array
        [full_output=True & cube_ref!=None] Vector of x shifts for ref frames
    cum_y_shifts_ref: numpy 1d array
        [full_output=True & cube_ref!=None] Vector of y shifts for ref frames
    """
    n, y, x = cube_sci.shape
    check_array(cube_sci, dim=3)
    gam = gammaval

    if nproc is None:
        nproc = cpu_count()//2

    if recenter_median and fit_type not in {'gaus', 'ann'}:
        raise TypeError("fit type not recognized. Should be 'ann' or 'gaus'")

    if crop and not subframesize < y:
        raise ValueError('`Subframesize` is too large')

    if cube_ref is not None:
        ref_star = True
        nref = cube_ref.shape[0]
    else:
        ref_star = False

    if crop:
        cube_sci_subframe = cube_crop_frames(cube_sci, subframesize, force=True,
                                             verbose=False)
        if ref_star:
            cube_ref_subframe = cube_crop_frames(cube_ref, subframesize,
                                                 force=True, verbose=False)
    else:
        subframesize = cube_sci.shape[-1]
        cube_sci_subframe = np.copy(cube_sci)
        if ref_star:
            cube_ref_subframe = np.copy(cube_ref)

    ceny, cenx = frame_center(cube_sci_subframe[0])
    print('Sub frame shape: {}'.format(cube_sci_subframe.shape))
    print('Center pixel: ({}, {})'.format(ceny, cenx))

    # Filtering cubes. Will be used for alignment purposes
    cube_sci_lpf = np.copy(cube_sci_subframe)
    if ref_star:
        cube_ref_lpf = np.copy(cube_ref_subframe)

    cube_sci_lpf = cube_sci_lpf - np.min(cube_sci_lpf)
    if ref_star:
        cube_ref_lpf = cube_ref_lpf - np.min(cube_ref_lpf)

    if max_spat_freq > 0:
        median_size = int(fwhm * max_spat_freq)
        # Remove spatial frequencies <0.5 lam/D and >3lam/D to isolate speckles
        cube_sci_hpf = cube_filter_highpass(cube_sci_lpf, 'median-subt',
                                            median_size=median_size,
                                            verbose=False)
    else:
        cube_sci_hpf = cube_sci_lpf
    if min_spat_freq > 0:
        cube_sci_lpf = cube_filter_lowpass(cube_sci_hpf, 'gauss',
                                           fwhm_size=min_spat_freq * fwhm,
                                           verbose=False)
    else:
        cube_sci_lpf = np.copy(cube_sci_hpf)

    if ref_star:
        if max_spat_freq > 0:
            cube_ref_hpf = cube_filter_highpass(cube_ref_lpf, 'median-subt',
                                                median_size=median_size,
                                                verbose=False)
        else:
            cube_ref_hpf = cube_ref_lpf
        if min_spat_freq > 0:
            cube_ref_lpf = cube_filter_lowpass(cube_ref_hpf, 'gauss',
                                               fwhm_size=min_spat_freq * fwhm,
                                               verbose=False)
        else:
            cube_ref_lpf = np.copy(cube_ref_hpf)

    if ref_star:
        align_cube = np.zeros((1 + n + nref, subframesize, subframesize))
        align_cube[1:(n + 1), :, :] = cube_sci_lpf
        align_cube[(n + 1):(n + 2 + nref), :, :] = cube_ref_lpf
    else:
        align_cube = np.zeros((1 + n, subframesize, subframesize))
        align_cube[1:(n + 1), :, :] = cube_sci_lpf

    n_frames = align_cube.shape[0]  # 1+n or 1+n+nref
    cum_y_shifts = 0
    cum_x_shifts = 0

    # no iteration => just align with respect to first frame => converges
    if alignment_iter == 1:
        align_cube[0] = cube_sci_lpf[0]
        # center the cube with stretched values
        if log:
            cube_stret = np.log10((align_cube-np.min(align_cube)+1)**gam)
        else:
            cube_stret = align_cube.copy()
        if mask is not None and crop:
            mask_tmp = frame_crop(mask, subframesize)
        else:
            mask_tmp = mask
        res = cube_recenter_dft_upsampling(cube_stret, center_fr1=(ceny, cenx),
                                           upsample_factor=upsample_factor,
                                           fwhm=fwhm, subi_size=None,
                                           full_output=True,
                                           verbose=debug, plot=plot,
                                           mask=mask_tmp, imlib=imlib,
                                           interpolation=interpolation,
                                           nproc=nproc)
        cube_stret, y_shift, x_shift = res
        sqsum_shifts = np.sum(np.sqrt(y_shift ** 2 + x_shift ** 2))
        print('Square sum of shift vecs: ' + str(sqsum_shifts))

        for j in range(1, n_frames):
            align_cube[j] = frame_shift(align_cube[j], y_shift[j], x_shift[j],
                                        imlib=imlib,
                                        interpolation=interpolation,
                                        border_mode=border_mode)
        cum_y_shifts += y_shift
        cum_x_shifts += x_shift

        if recenter_median:
            align_cube[0] = cube_collapse(align_cube[1:(n + 1)], mode=collapse,
                                          **collapse_args)
            # Recenter the median frame using a 2d fit
            if fit_type == 'gaus' and negative:
                crop_sz = int(fwhm)
            elif fit_type == 'gaus':
                crop_sz = int(3*fwhm)
            else:
                crop_sz = int(6*fwhm)
            if not crop_sz % 2:
                # size should be odd and small, between 5 and 7
                if crop_sz > 7:
                    crop_sz -= 1
                else:
                    crop_sz += 1
            sub_image, y1, x1 = get_square(align_cube[0], size=crop_sz,
                                           y=ceny, x=cenx, position=True)

            if fit_type == 'gaus':
                if negative:
                    sub_image = -sub_image + np.abs(np.min(-sub_image))
                y_i, x_i = fit_2dgaussian(sub_image, crop=False,
                                          threshold=False, sigfactor=1,
                                          debug=debug, full_output=False)
            elif fit_type == 'ann':
                if upsample_factor > 20:
                    print("WARNING: the annulus centering may be slow for ")
                    print("upsample_factor larger than 20")
                sampl_cen = 1./upsample_factor
                if ann_rad_search:
                    sampl_rad = fwhm*ann_rad/10  # 1/10 of estimated radius
                else:
                    sampl_rad = None
                y_i, x_i, rad = _fit_2dannulus(sub_image, fwhm=fwhm, crop=False,
                                               ann_rad=ann_rad,
                                               sampl_cen=sampl_cen,
                                               sampl_rad=sampl_rad,
                                               ann_width=ann_width,
                                               unc_in=2.)
            yshift = ceny - (y1 + y_i)
            xshift = cenx - (x1 + x_i)

            cum_y_shifts += yshift
            cum_x_shifts += xshift
    else:
        for i in range(alignment_iter):
            align_cube[0] = cube_collapse(align_cube[1:(n + 1)], mode=collapse,
                                          **collapse_args)
            if recenter_median:
                # Recenter the median frame using a 2d fit
                if fit_type == 'gaus' and negative:
                    crop_sz = int(fwhm)
                elif fit_type == 'gaus':
                    crop_sz = int(3*fwhm)
                else:
                    crop_sz = int(6*fwhm)
                if not crop_sz % 2:
                    # size should be odd and small
                    if crop_sz > 7:
                        crop_sz -= 1
                    else:
                        crop_sz += 1
                sub_image, y1, x1 = get_square(align_cube[0], size=crop_sz,
                                               y=ceny, x=cenx, position=True)

                if fit_type == 'gaus':
                    if negative:
                        sub_image = -sub_image + np.abs(np.min(-sub_image))
                    y_i, x_i = fit_2dgaussian(sub_image, crop=False,
                                              threshold=False, sigfactor=1,
                                              debug=debug, full_output=False)
                elif fit_type == 'ann':
                    sampl_cen = 1./upsample_factor
                    if ann_rad_search:
                        sampl_rad = fwhm*ann_rad/10  # 1/10 of estimated radius
                    else:
                        sampl_rad = None
                    y_i, x_i, rad = _fit_2dannulus(sub_image, fwhm=fwhm,
                                                   crop=False, ann_rad=ann_rad,
                                                   sampl_cen=sampl_cen,
                                                   sampl_rad=sampl_rad,
                                                   ann_width=ann_width,
                                                   unc_in=2.)
                yshift = ceny - (y1 + y_i)
                xshift = cenx - (x1 + x_i)

                align_cube[0] = frame_shift(align_cube[0, :, :], yshift, xshift,
                                            imlib=imlib,
                                            interpolation=interpolation,
                                            border_mode=border_mode)

            # center the cube with stretched values
            if log:
                cube_stret = np.log10((align_cube-np.min(align_cube)+1)**gam)
            else:
                cube_stret = align_cube.copy()
            if mask is not None and crop:
                mask_tmp = frame_crop(mask, subframesize)
            else:
                mask_tmp = mask
            res = cube_recenter_dft_upsampling(cube_stret, subi_size=None,
                                               upsample_factor=upsample_factor,
                                               center_fr1=(ceny, cenx),
                                               fwhm=fwhm, full_output=True,
                                               verbose=False, plot=False,
                                               mask=mask_tmp, imlib=imlib,
                                               interpolation=interpolation,
                                               nproc=nproc)
            _, y_shift, x_shift = res
            sqsum_shifts = np.sum(np.sqrt(y_shift ** 2 + x_shift ** 2))
            print('Square sum of shift vecs: ' + str(sqsum_shifts))

            for j in range(1, n_frames):
                align_cube[j] = frame_shift(align_cube[j], y_shift[j],
                                            x_shift[j], imlib=imlib,
                                            interpolation=interpolation,
                                            border_mode=border_mode)

            cum_y_shifts += y_shift
            cum_x_shifts += x_shift

    cube_reg_sci = np.copy(cube_sci)
    cum_y_shifts_sci = cum_y_shifts[1:(n + 1)]
    cum_x_shifts_sci = cum_x_shifts[1:(n + 1)]
    cube_reg_sci = cube_shift(cube_sci, cum_y_shifts_sci, cum_x_shifts_sci,
                              imlib=imlib, interpolation=interpolation,
                              border_mode=border_mode, nproc=nproc)

    if plot:
        plt.figure(figsize=vip_figsize)
        plt.plot(cum_x_shifts_sci, 'o-', label='Shifts in x', alpha=0.5)
        plt.plot(cum_y_shifts_sci, 'o-', label='Shifts in y', alpha=0.5)
        plt.legend(loc='best')
        plt.grid('on', alpha=0.2)
        plt.ylabel('Pixels')
        plt.xlabel('Frame number')

        plt.figure(figsize=vip_figsize)
        b = int(np.sqrt(n))
        la = 'Histogram'
        _ = plt.hist(cum_x_shifts_sci, bins=b, alpha=0.5, label=la+' shifts X')
        _ = plt.hist(cum_y_shifts_sci, bins=b, alpha=0.5, label=la+' shifts Y')
        plt.legend(loc='best')
        plt.ylabel('Bin counts')
        plt.xlabel('Pixels')

    if ref_star:
        cube_reg_ref = np.copy(cube_ref)
        cum_y_shifts_ref = cum_y_shifts[(n + 1):]
        cum_x_shifts_ref = cum_x_shifts[(n + 1):]
        cube_reg_ref = cube_shift(cube_ref, cum_y_shifts_ref, cum_x_shifts_ref,
                                  imlib=imlib, interpolation=interpolation,
                                  border_mode=border_mode, nproc=nproc)

    if ref_star:
        if full_output:
            return (cube_reg_sci, cube_reg_ref, cube_sci_lpf, cube_stret,
                    cum_x_shifts_sci, cum_y_shifts_sci, cum_x_shifts_ref,
                    cum_y_shifts_ref)
        else:
            return (cube_reg_sci, cube_reg_ref)
    else:
        if full_output:
            return (cube_reg_sci, cube_sci_lpf, cube_stret,
                    cum_x_shifts_sci, cum_y_shifts_sci)
        else:
            return cube_reg_sci


def _fit_2dannulus(array, fwhm=4, crop=False, cent=None, cropsize=15,
                   ann_rad=0.5, ann_width=0.5, sampl_cen=0.1, sampl_rad=None,
                   unc_in=2.):
    """Find the center of a donut-shape signal (e.g. a coronagraphic PSF) with\
    an annulus fit.

    The function uses a grid of positions for the center and radius of the
    annulus. The best fit is found by maximizing the mean flux measured in the
    annular mask. This requires the image to be already roughly centered
    (by an uncertainty provided by unc_in).

    Parameters
    ----------
    array : array_like
        Image with a single donut-like source, already approximately at the
        center of the frame.
    fwhm : float
        Gaussian PSF full width half maximum from fitting (in pixels).
    ann_rad: float, opt
        First estimate of the hole radius (in terms of fwhm). The grid search
        on the radius of the optimal annulus goes from 0.5 to 2 times hole_rad.
        Note: for the AGPM PSF of VLT/NACO, the optimal hole_rad ~ 0.5FWHM.
    ann_width: float, opt
        Width of the annulus in FWHM; default is 0.5 FWHM.
    sampl_cen: float, opt
        Precision of the grid sampling to find the center of the annulus (in
        pixels)
    sampl_rad: float, opt or None.
        Precision of the grid sampling to find the optimal radius of the
        annulus (in pixels). If set to None, there is no grid search for the
        optimal radius of the annulus, the value given by hole_rad is used.
    unc_in: float, opt
        Initial uncertainty on the center location (with respect to center of
        input subframe) in pixels; this will set the grid width.

    Returns
    -------
    mean_y : float
        Source centroid y position on the full image from fitting.
    mean_x : float
        Source centroid x position on the full image from fitting.
    final_hole_rad : float
        [if sampl_rad != None] Best fit radius of the annulus, in pixels.
        [if sampl_rad = None] Input radius of the annulus, in pixels.

    """
    if cent is None:
        ceny, cenx = frame_center(array)
    else:
        cenx, ceny = cent

    if crop:
        x_sub_px = cenx % 1
        y_sub_px = ceny % 1

        imside = array.shape[0]
        psf_subimage, suby, subx = get_square(array, min(cropsize, imside),
                                              int(ceny), int(cenx),
                                              position=True)
        ceny, cenx = frame_center(psf_subimage)
        ceny += y_sub_px
        cenx += x_sub_px

    ann_sz = ann_width*fwhm

    grid_sh_x = np.arange(-unc_in, unc_in, sampl_cen)
    grid_sh_y = np.arange(-unc_in, unc_in, sampl_cen)
    if sampl_rad is None:
        rads = [ann_rad*fwhm]
    else:
        rads = np.arange(0.5*ann_rad*fwhm, 2*ann_rad*fwhm, sampl_rad)
    flux_ann = np.zeros([grid_sh_x.shape[0], grid_sh_y.shape[0]])
    best_rad = np.zeros([grid_sh_x.shape[0], grid_sh_y.shape[0]])

    for ii, xx in enumerate(grid_sh_x):
        for jj, yy in enumerate(grid_sh_y):
            tmp_tmp = frame_shift(array, yy, xx)
            for rr, rad in enumerate(rads):
                # mean flux in the annulus
                tmp = frame_basic_stats(tmp_tmp, 'annulus', inner_radius=rad,
                                        size=ann_sz, plot=False)

                if tmp > flux_ann[ii, jj]:
                    flux_ann[ii, jj] = tmp
                    best_rad[ii, jj] = rad
    i_max, j_max = np.unravel_index(np.argmax(flux_ann), flux_ann.shape)
    mean_x = cenx - grid_sh_x[i_max]
    mean_y = ceny - grid_sh_y[j_max]

    if sampl_rad is None:
        return mean_y, mean_x, ann_rad*fwhm
    else:
        final_hole_rad = best_rad[i_max, j_max]/fwhm
        return mean_y, mean_x, final_hole_rad
