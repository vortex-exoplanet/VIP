#! /usr/bin/env python

"""
Module with detection algorithms.
"""

__author__ = 'Carlos Alberto Gomez Gonzalez, Valentin Christiaens'
__all__ = ['detection',
           'mask_source_centers',
           'mask_sources',
           'peak_coordinates']

import numpy as np
import pandas as pn
from hciplot import plot_frames
from scipy.ndimage.filters import correlate
from skimage import feature
from astropy.stats import sigma_clipped_stats
from astropy.stats import gaussian_fwhm_to_sigma, gaussian_sigma_to_fwhm
from astropy.modeling import models, fitting
from skimage.feature import peak_local_max
from ..var import (mask_circle, get_square, frame_center, fit_2dgaussian,
                   frame_filter_lowpass, dist_matrix)
from ..config.utils_conf import sep
from .snr_source import snr, snrmap, frame_report


def detection(array, fwhm=4, psf=None, mode='lpeaks', bkg_sigma=5,
              matched_filter=False, mask=True, snr_thresh=5, nproc=1, plot=True,
              debug=False, full_output=False, verbose=True, **kwargs):
    """ Finds blobs in a 2d array. The algorithm is designed for automatically
    finding planets in post-processed high contrast final frames. Blob can be
    defined as a region of an image in which some properties are constant or
    vary within a prescribed range of values. See ``Notes`` below to read about
    the algorithm details.

    Parameters
    ----------
    array : numpy ndarray, 2d
        Input frame.
    fwhm : None or int, optional
        Size of the FWHM in pixels. If None and a ``psf`` is provided, then the
        FWHM is measured on the PSF image.
    psf : numpy ndarray
        Input PSF template. It must be normalized with the
        ``vip_hci.metrics.normalize_psf`` function.
    mode : {'lpeaks', 'log', 'dog', 'snrmap', 'snrmapf'}, optional
        Sets with algorithm to use. Each algorithm yields different results. See
        notes for the details of each method.
    bkg_sigma : int or float, optional
        The number standard deviations above the clipped median for setting the
        background level. Used when ``mode`` is either 'lpeaks', 'dog' or 'log'.
    matched_filter : bool, optional
        Whether to correlate with the psf of not. Used when ``mode`` is either
        'lpeaks', 'dog' or 'log'.
    mask : bool, optional
        If True the central region (circular aperture of 2*FWHM radius) of the
        image will be masked out.
    snr_thresh : float, optional
        S/N threshold for deciding whether the blob is a detection or not. Used
        to threshold the S/N map when ``mode`` is set to 'snrmap' or 'snrmapf'.
    nproc : None or int, optional
        The number of processes for running the ``snrmap`` function.
    plot : bool, optional
        If True plots the frame showing the detected blobs on top.
    debug : bool, optional
        Whether to print and plot additional/intermediate results.
    full_output : bool, optional
        Whether to output just the coordinates of blobs that fulfill the SNR
        constraint or a table with all the blobs and the peak pixels and SNR.
    verbose : bool, optional
        Whether to print to stdout information about found blobs.
    **kwargs : dictionary, optional
        Arguments to be passed to ``plot_frames`` to customize the plot (and to
        save it to disk).

    Returns
    -------
    yy, xx : numpy ndarray
        Two vectors with the y and x coordinates of the centers of the sources
        (potential planets).
    If full_output is True then a table with all the candidates that passed the
    2d Gaussian fit constrains and their S/N is returned.

    Notes
    -----
    When ``mode`` is either 'lpeaks', 'dog' or 'log', the detection might happen
    in the input frame or in a match-filtered version of it (by setting
    ``matched_filter`` to True and providing a PSF template, to run a
    correlation filter). Filtering the image will smooth the noise and maximize
    detectability of objects with a shape similar to the kernel. When ``mode``
    is either 'snrmap' or 'snrmapf', the detection is done on an S/N map
    directly.

    When ``mode`` is set to:
        'lpeaks' (Local maxima): The local peaks above the background (computed
        using sigma clipped statistics) on the (correlated) frame are detected.
        A maximum filter is used for finding local maxima. This operation
        dilates the original image and merges neighboring local maxima closer
        than the size of the dilation. Locations where the original image is
        equal to the dilated image are returned as local maxima. The minimum
        separation between the peaks is 1*FWHM.

        'log' (Laplacian of Gaussian): It computes the Laplacian of Gaussian
        images with successively increasing standard deviation and stacks them
        up in a cube. Blobs are local maximas in this cube. LOG assumes that the
        blobs are again assumed to be bright on dark.

        'dog' (Difference of Gaussians): This is a faster approximation of the
        Laplacian of Gaussian approach. In this case the image is blurred with
        increasing standard deviations and the difference between two
        successively blurred images are stacked up in a cube. DOG assumes that
        the blobs are again assumed to be bright on dark.

        'snrmap' or 'snrmapf': A threshold is applied to the S/N map, computed
        with the ``snrmap`` function (``snrmapf`` calls ``snrmap`` with
        ``approximated`` set to True). The threshold is given by ``snr_thresh``
        and local maxima are found as in the case of 'lpeaks'.

    Finally, a 2d Gaussian fit is done on each of the potential blobs
    constraining the position on a cropped sub-image and the sigma of the fit
    (to match the input FWHM). Finally the blobs are filtered based on its S/N
    value, according to ``snr_thresh``.

    """
    def check_blobs(array, coords_temp, fwhm, debug):
        y_temp = coords_temp[:, 0]
        x_temp = coords_temp[:, 1]
        coords = []
        # Fitting a 2d gaussian to each local maxima position
        for y, x in zip(y_temp, x_temp):
            subsi = 3 * int(np.ceil(fwhm))
            if subsi % 2 == 0:
                subsi += 1

            if mode in ('lpeaks', 'log', 'dog'):
                scy = y + pad
                scx = x + pad
            elif mode in ('snrmap', 'snrmapf'):
                scy = y
                scx = x
            subim, suby, subx = get_square(array, subsi, scy, scx,
                                           position=True, force=True,
                                           verbose=False)
            cy, cx = frame_center(subim)

            gauss = models.Gaussian2D(amplitude=subim.max(), x_mean=cx,
                                      y_mean=cy, theta=0,
                                      x_stddev=fwhm*gaussian_fwhm_to_sigma,
                                      y_stddev=fwhm*gaussian_fwhm_to_sigma)

            sy, sx = np.indices(subim.shape)
            fitter = fitting.LevMarLSQFitter()
            fit = fitter(gauss, sx, sy, subim)

            # checking that the amplitude is positive > 0
            # checking whether the x and y centroids of the 2d gaussian fit
            # coincide with the center of the subimage (within 2px error)
            # checking whether the mean of the fwhm in y and x of the fit
            # are close to the FWHM_PSF with a margin of 3px
            fwhm_y = fit.y_stddev.value * gaussian_sigma_to_fwhm
            fwhm_x = fit.x_stddev.value * gaussian_sigma_to_fwhm
            mean_fwhm_fit = np.mean([np.abs(fwhm_x), np.abs(fwhm_y)])
            condyf = np.allclose(fit.y_mean.value, cy, atol=2)
            condxf = np.allclose(fit.x_mean.value, cx, atol=2)
            condmf = np.allclose(mean_fwhm_fit, fwhm, atol=3)
            if fit.amplitude.value > 0 and condxf and condyf and condmf:
                coords.append((suby + fit.y_mean.value,
                               subx + fit.x_mean.value))

                if debug:
                    print('Coordinates (Y,X): {:.3f},{:.3f}'.format(y, x))
                    print('fit peak = {:.3f}'.format(fit.amplitude.value))
                    msg = 'fwhm_y in px = {:.3f}, fwhm_x in px = {:.3f}'
                    print(msg.format(fwhm_y, fwhm_x))
                    print('mean fit fwhm = {:.3f}'.format(mean_fwhm_fit))
                    if plot:
                        plot_frames(subim, colorbar=True, axis=False, dpi=60)
        return coords

    def print_coords(coords):
        print('Blobs found:', len(coords))
        print(' ycen   xcen')
        print('------ ------')
        for j in range(len(coords[:, 0])):
            print('{:.3f} \t {:.3f}'.format(coords[j, 0], coords[j, 1]))

    def print_abort():
        if verbose:
            print(sep)
            print('No potential sources found')
            print(sep)

    # --------------------------------------------------------------------------
    if array.ndim != 2:
        raise TypeError('Input array is not a frame or 2d array')
    if psf is not None:
        if psf.ndim != 2 and psf.shape[0] < array.shape[0]:
            raise TypeError('Input psf is not a 2d array or has wrong size')
    else:
        if matched_filter:
            raise ValueError('`psf` must be provided when `matched_filter` is '
                             'True')

    if fwhm is None:
        if psf is not None:
            # Getting the FWHM from the PSF array
            cenpsf = frame_center(psf)
            outdf = fit_2dgaussian(psf, cent=(cenpsf), debug=debug,
                                   full_output=True)
            fwhm_x, fwhm_y = outdf['fwhm_x'], outdf['fwhm_y']
            fwhm = np.mean([fwhm_x, fwhm_y])
            if verbose:
                print('FWHM = {:.2f} pxs\n'.format(fwhm))
            if debug:
                print('FWHM_y', fwhm_y)
                print('FWHM_x', fwhm_x)
        else:
            raise ValueError('`fwhm` or `psf` must be provided')

    # Masking the center, 2*lambda/D is the expected IWA
    if mask:
        array = mask_circle(array, radius=fwhm)

    # Generating a detection map: Match-filtered frame or SNRmap
    # For 'lpeaks', 'dog', 'log' it is possible to skip this step
    if mode in ('lpeaks', 'log', 'dog'):
        if matched_filter:
            frame_det = correlate(array, psf)
        else:
            frame_det = array

        if debug and plot and matched_filter:
            print('Match-filtered frame:')
            plot_frames(frame_det, colorbar=True)

        # Estimation of background level
        _, median, stddev = sigma_clipped_stats(frame_det, sigma=5,
                                                maxiters=None)
        bkg_level = median + (stddev * bkg_sigma)
        if debug:
            print('Sigma clipped median = {:.3f}'.format(median))
            print('Sigma clipped stddev = {:.3f}'.format(stddev))
            print('Background threshold = {:.3f}'.format(bkg_level), '\n')

    elif mode in ('snrmap', 'snrmapf'):
        if mode == 'snrmap':
            approx = False
        elif mode == 'snrmapf':
            approx = True
        frame_det = snrmap(array, fwhm=fwhm, approximated=approx, plot=False,
                           nproc=nproc, verbose=verbose)

        if debug and plot:
            print('Signal-to-noise ratio map:')
            plot_frames(frame_det, colorbar=True)

    if mode in ('lpeaks', 'log', 'dog'):
        # Padding the image with zeros to avoid errors at the edges
        pad = 10
        array_padded = np.lib.pad(array, pad, 'constant', constant_values=0)
    else:
        pad=0

    if mode in ('lpeaks', 'snrmap', 'snrmapf'):
        if mode == 'lpeaks':
            threshold = bkg_level
        else:
            threshold = snr_thresh

        coords_temp = peak_local_max(frame_det, threshold_abs=threshold,
                                     min_distance=int(np.ceil(fwhm)),
                                     num_peaks=20)

        if mode == 'lpeaks':
            coords = check_blobs(array_padded, coords_temp, fwhm, debug)
        else:
            coords = check_blobs(array, coords_temp, fwhm, debug)
        coords = np.array(coords)
        if verbose and coords.shape[0] > 0:
            print_coords(coords-pad)

    elif mode == 'log':
        sigma = fwhm * gaussian_fwhm_to_sigma
        coords = feature.blob_log(frame_det.astype('float'),
                                  threshold=bkg_level, min_sigma=sigma-.5,
                                  max_sigma=sigma+.5)
        if len(coords) == 0:
            print_abort()
            return 0, 0
        coords = coords[:, :2]
        coords = check_blobs(array_padded, coords, fwhm, debug)
        coords = np.array(coords)
        if coords.shape[0] > 0 and verbose:
            print_coords(coords-pad)

    elif mode == 'dog':
        sigma = fwhm * gaussian_fwhm_to_sigma
        coords = feature.blob_dog(frame_det.astype('float'),
                                  threshold=bkg_level, min_sigma=sigma-.5,
                                  max_sigma=sigma+.5)
        if len(coords) == 0:
            print_abort()
            return 0, 0
        coords = coords[:, :2]
        coords = check_blobs(array_padded, coords, fwhm, debug)
        coords = np.array(coords)
        if coords.shape[0] > 0 and verbose:
            print_coords(coords-pad)

    else:
        raise ValueError('`mode` not recognized')

    if coords.shape[0] == 0:
        print_abort()
        return 0, 0

    yy = coords[:, 0]
    xx = coords[:, 1]
    yy_final = []
    xx_final = []
    yy_out = []
    xx_out = []
    snr_list = []

    if mode in ('lpeaks', 'log', 'dog'):
        xx -= pad
        yy -= pad

    # Checking S/N for potential sources
    for i in range(yy.shape[0]):
        y = yy[i]
        x = xx[i]
        if verbose:
            print('')
            print(sep)
            print('X,Y = ({:.1f},{:.1f})'.format(x, y))
        snr_value = snr(array, (x, y), fwhm, False, verbose=False)
        snr_list.append(snr_value)
        if snr_value >= snr_thresh:
            if verbose:
                _ = frame_report(array, fwhm, (x, y), verbose=verbose)
            yy_final.append(y)
            xx_final.append(x)
        else:
            yy_out.append(y)
            xx_out.append(x)
            if verbose:
                msg = 'S/N constraint NOT fulfilled (S/N = {:.3f})'
                print(msg.format(snr_value))
            if debug:
                _ = frame_report(array, fwhm, (x, y), verbose=verbose)
    if verbose:
        print(sep)

    if debug or full_output:
        table_full = pn.DataFrame({'y': yy.tolist(),
                                   'x': xx.tolist(),
                                   'px_snr': snr_list})
        table_full.sort_values('px_snr')

    yy_final = np.array(yy_final)
    xx_final = np.array(xx_final)
    yy_out = np.array(yy_out)
    xx_out = np.array(xx_out)
    table = pn.DataFrame({'y': yy_final.tolist(), 'x': xx_final.tolist()})

    if plot:
        coords = tuple(zip(xx_out.tolist() + xx_final.tolist(),
                           yy_out.tolist() + yy_final.tolist()))
        circlealpha = [0.3] * len(xx_out)
        circlealpha += [1] * len(xx_final)
        plot_frames(array, dpi=120, circle=coords, circle_alpha=circlealpha,
                    circle_label=True, circle_radius=fwhm, **kwargs)

    if debug:
        print(table_full)

    if full_output:
        return table_full
    else:
        return table


def peak_coordinates(obj_tmp, fwhm, approx_peak=None, search_box=None,
                     channels_peak=False):
    """Find the pixel coordinates of maximum in either a frame or a cube,
    after convolution with gaussian. It first applies a gaussian filter, to
    lower the probability of returning a hot pixel (although it may still
    happen with clumps of hot pixels, hence parameter "approx_peak").

    Parameters
    ----------
    obj_tmp : cube_like or frame_like
        Input 3d cube or image.
    fwhm : float_like or 1d array
        Input full width half maximum value of the PSF in pixels. This will be
        used as the standard deviation for Gaussian kernel of the Gaussian
        filtering. Can be a 1d array if obj_tmp is a 3D cube.
    approx_peak: 2 components list or array, opt
        Gives the approximate yx coordinates of the peak.
    search_box: float or 2 components list or array, opt
        Gives the half-size in pixels of a box in which the peak is searched,
        around approx_peak. If float, it is assumed the same box size is wanted
        in both y and x. Note that this parameter should be provided if
        approx_peak is provided.
    channels_peak: bool, {False, True}, opt
        Whether returns the indices of the peak in each channel in addition to
        the global indices of the peak in the cube. If True, it would hence
        return two 1d-arrays. (note: only available if the input is a 3d cube)

    Returns
    -------
    (zz_max,) yy_max, xx_max : integers
        Indices of peak in either 3D or 2D array
    ind_ch_max: 2d array
        Coordinates of the peak in each channel

    """

    ndims = len(obj_tmp.shape)
    assert ndims == 2 or ndims == 3, "Array is not two or three dimensional"

    if approx_peak is not None:
        assert len(approx_peak) == 2, "Approx peak is not two dimensional"
        if isinstance(search_box,float) or isinstance(search_box,int):
            sbox_y = search_box
            sbox_x = search_box
        elif len(search_box) == 2:
            sbox_y = search_box[0]
            sbox_x = search_box[1]
        else:
            msg = "The search box does not have the right number of elements"
            raise ValueError(msg)
        if ndims == 3:
            n_z = obj_tmp.shape[0]
            sbox = np.zeros([n_z,2*sbox_y+1,2*sbox_x+1])

    if ndims == 2:
        med_filt_tmp = frame_filter_lowpass(obj_tmp, 'median', 
                                            median_size=int(fwhm))
        if approx_peak is None:
            ind_max = np.unravel_index(med_filt_tmp.argmax(),
                                       med_filt_tmp.shape)
        else:
            sbox = med_filt_tmp[approx_peak[0]-sbox_y:approx_peak[0]+sbox_y+1,
                                  approx_peak[1]-sbox_x:approx_peak[1]+sbox_x+1]
            ind_max_sbox = np.unravel_index(sbox.argmax(), sbox.shape)
            ind_max = (approx_peak[0]-sbox_y+ind_max_sbox[0],
                       approx_peak[1]-sbox_x+ind_max_sbox[1])

        return ind_max

    if ndims == 3:
        n_z = obj_tmp.shape[0]
        med_filt_tmp = np.zeros_like(obj_tmp)
        ind_ch_max = np.zeros([n_z,2])
        if isinstance(fwhm, float) or isinstance(fwhm, int):
            fwhm = [fwhm]*n_z

        for zz in range(n_z):
            med_filt_tmp[zz] = frame_filter_lowpass(obj_tmp[zz], 'median', 
                                                    median_size=int(fwhm[zz]))
            if approx_peak is None:
                ind_ch_max[zz] = np.unravel_index(med_filt_tmp[zz].argmax(),
                                                  med_filt_tmp[zz].shape)
            else:
                sbox[zz] = med_filt_tmp[zz, approx_peak[0]-sbox_y:\
                                          approx_peak[0]+sbox_y+1,
                                          approx_peak[1]-sbox_x:\
                                          approx_peak[1]+sbox_x+1]
                ind_max_sbox = np.unravel_index(sbox[zz].argmax(),
                                                sbox[zz].shape)
                ind_ch_max[zz] = (approx_peak[0]-sbox_y+ind_max_sbox[0],
                                  approx_peak[1]-sbox_x+ind_max_sbox[1])

        if approx_peak is None:
            ind_max = np.unravel_index(med_filt_tmp.argmax(),
                                       med_filt_tmp.shape)
        else:
            ind_max_tmp = np.unravel_index(sbox.argmax(),
                                           sbox.shape)
            ind_max = (ind_max_tmp[0]+approx_peak[0]-sbox_y,
                       ind_max_tmp[1]+approx_peak[1]-sbox_x)

        if channels_peak:
            return ind_max, ind_ch_max
        else:
            return ind_max


def mask_source_centers(array, fwhm, y=None, x=None):
    """ Creates a mask of ones with the size of the input frame and zeros at
    the center of the sources (planets) with coordinates x, y.
    If y and x are not provided, the sources will be found automatically using
    'detection()' ('log' mode).

    Parameters
    ----------
    array : numpy ndarray
        Input frame.
    fwhm : float
        Size in pixels of the FWHM.
    y, x : tuples of int (optional)
        Coordinates of the center of the sources.

    Returns
    -------
    mask : numpy ndarray
        Mask frame.

    """
    if array.ndim != 2:
        raise TypeError('Wrong input array shape.')

    frame = array.copy()
    if y is None or x is None:
        frame = mask_circle(frame, radius=2*fwhm)
        yy, xx = detection(frame, fwhm, plot=False, mode='log')
    else:
        yy = np.array(y)
        xx = np.array(x)
    mask = np.ones_like(array)
    # center sources become zeros
    mask[yy.astype('int'), xx.astype('int')] = 0
    return mask



def mask_sources(mask, ap_rad):
    """ Given an input mask with zeros only at the *center* of source locations
    (ones elsewhere), returns a mask with zeros within a radius ap_rad of all 
    sources.

    Parameters
    ----------
    mask : numpy ndarray
        Input mask with zeros at sources center. Mask has to be square.
    ap_rad : float
        Size in pixels of the apertures that should be filled with zeros 
        around each source in the mask.

    Returns
    -------
    mask_out : numpy ndarray
        Output mask frame.

    """
    if mask.ndim != 2:
        raise TypeError('Wrong input array shape.')
    mask[np.where(mask>1)] = 1

    ny, nx = mask.shape
    n_s = int(ny*nx - np.sum(mask))
    mask_out = np.ones([ny,nx])
    
    if n_s == 0:
        return mask_out
    else:
        s_coords = np.where(mask==0)
        for s in range(n_s):
            rad_arr = dist_matrix(ny, cx=s_coords[1][s], cy=s_coords[0][s]) 
            mask_out[np.where(rad_arr<ap_rad)] = 0

        return mask_out