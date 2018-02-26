#! /usr/bin/env python

"""
Module with detection algorithms.
"""

from __future__ import division
from __future__ import print_function

__author__ = 'Carlos Alberto Gomez Gonzalez'
__all__ = ['detection',
           'mask_source_centers',
           'peak_coordinates']

import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage.filters import correlate
from skimage import feature
from astropy.stats import sigma_clipped_stats
from astropy.stats import gaussian_fwhm_to_sigma, gaussian_sigma_to_fwhm
from astropy.table import Table
from astropy.modeling.models import Gaussian2D
from astropy.modeling.fitting import LevMarLSQFitter
from skimage.feature import peak_local_max
from ..var import (mask_circle, pp_subplots, get_square, frame_center,
                   frame_filter_gaussian2d, fit_2dgaussian)
from ..conf.utils_conf import sep
from .snr import snr_ss
from .frame_analysis import frame_quick_report


# TODO: Add the option of computing and thresholding an S/N map

def detection(array, psf, bkg_sigma=1, mode='lpeaks', matched_filter=False,
              mask=True, snr_thresh=5, plot=True, debug=False,
              full_output=False, verbose=True, save_plot=None,
              object_name = None, frame_size=None, inner_rad=None,
              pca_type = None, ncomp = None, NIRC2angscale = False):
    """ Finds blobs in a 2d array. The algorithm is designed for automatically
    finding planets in post-processed high contrast final frames. Blob can be
    defined as a region of an image in which some properties are constant or
    vary within a prescribed range of values. See <Notes> below to read about
    the algorithm details.

    Parameters
    ----------
    array : array_like, 2d
        Input frame.
    psf : array_like
        Input psf.
    bkg_sigma : float, optional
        The number standard deviations above the clipped median for setting the
        background level.
    mode : {'lpeaks','log','dog'}, optional
        Sets with algorithm to use. Each algorithm yields different results.
    matched_filter : {True, False}, bool optional
        Whether to correlate with the psf of not.
    mask : {True, False}, optional
        Whether to mask the central region (circular aperture of 2*fwhm radius).
    snr_thresh : float, optional
        SNR threshold for deciding whether the blob is a detection or not.
    plot {True, False}, bool optional
        If True plots the frame showing the detected blobs on top.
    debug : {False, True}, bool optional
        Whether to print and plot additional/intermediate results.
    full_output : {False, True}, bool optional
        Whether to output just the coordinates of blobs that fulfill the SNR
        constraint or a table with all the blobs and the peak pixels and SNR.
    verbose : {True,False}, bool optional
        Whether to print to stdout information about found blobs.
    save_plot: string
        If provided, the frames processed by blob detection are saved to that path.
    object_name: string
        Target name, used in the plot title
    frame_size: int
        Frame size of the pca, used in the plot title
    inner_rad: int
        Size of the mask in pca, as a unit of the FWHM, used in the plot title
    pca_type: string
        adi or rdi, used in the title
    ncomp: int
        Number of principal components used to compute the reduced frame, used in the title
    NIRC2angscale: {False, True}
        If True the plot axes are converted to angular scale (arcseconds,
        assuming NIRC2's ~ 0.01 pixel scale)

    Returns
    -------
    yy, xx : array_like
        Two vectors with the y and x coordinates of the centers of the sources
        (potential planets).
    If full_output is True then a table with all the candidates that passed the
    2d Gaussian fit constrains and their SNR is returned. Also the count of
    companions with SNR>5 (those with highest probability of being true
    detections).

    Notes
    -----
    The FWHM of the PSF is measured directly on the provided array. If the
    parameter matched_filter==True then the PSF is used to run a matched filter
    (correlation) which is equivalent to a convolution filter. Filtering the
    image will smooth the noise and maximize detectability of objects with a
    shape similar to the kernel.
    The background level or threshold is found with sigma clipped statistics
    (5 sigma over the median) on the image/correlated image. Then 5 different
    strategies can be used to detect the blobs (potential planets):

    Local maxima + 2d Gaussian fit. The local peaks above the background on the
    (correlated) frame are detected. A maximum filter is used for finding local
    maxima. This operation dilates the original image and merges neighboring
    local maxima closer than the size of the dilation. Locations where the
    original image is equal to the dilated image are returned as local maxima.
    The minimum separation between the peaks is 1*FWHM. A 2d Gaussian fit is
    done on each of the maxima constraining the position on the subimage and the
    sigma of the fit. Finally the blobs are filtered based on its SNR.

    Laplacian of Gaussian + 2d Gaussian fit. It computes the Laplacian of
    Gaussian images with successively increasing standard deviation and stacks
    them up in a cube. Blobs are local maximas in this cube. LOG assumes that
    the blobs are again assumed to be bright on dark. A 2d Gaussian fit is done
    on each of the candidates constraining the position on the subimage and the
    sigma of the fit. Finally the blobs are filtered based on its SNR.

    Difference of Gaussians. This is a faster approximation of LoG approach. In
    this case the image is blurred with increasing standard deviations and the
    difference between two successively blurred images are stacked up in a cube.
    DOG assumes that the blobs are again assumed to be bright on dark. A 2d
    Gaussian fit is done on each of the candidates constraining the position on
    the subimage and the sigma of the fit. Finally the blobs are filtered based
    on its SNR.

    """
    def check_blobs(array_padded, coords_temp, fwhm, debug):
        y_temp = coords_temp[:,0]
        x_temp = coords_temp[:,1]
        coords = []
        # Fitting a 2d gaussian to each local maxima position
        for y,x in zip(y_temp,x_temp):
            subim, suby, subx = get_square(array_padded,
                                           2*int(np.ceil(fwhm)),
                                           y+pad, x+pad, position=True)
            cy, cx = frame_center(subim)

            gauss = Gaussian2D(amplitude=subim.max(),
                               x_mean=cx, y_mean=cy,
                               x_stddev=fwhm*gaussian_fwhm_to_sigma,
                               y_stddev=fwhm*gaussian_fwhm_to_sigma, theta=0)

            sy, sx = np.indices(subim.shape)
            fitter = LevMarLSQFitter()
            fit = fitter(gauss, sx, sy, subim)

            # checking that the amplitude is positive > 0
            # checking whether the x and y centroids of the 2d gaussian fit
            # coincide with the center of the subimage (within 2px error)
            # checking whether the mean of the fwhm in y and x of the fit
            # are close to the FWHM_PSF with a margin of 3px
            fwhm_y = fit.y_stddev.value*gaussian_sigma_to_fwhm
            fwhm_x = fit.x_stddev.value*gaussian_sigma_to_fwhm
            mean_fwhm_fit = np.mean([np.abs(fwhm_x), np.abs(fwhm_y)])
            if fit.amplitude.value>0 \
            and np.allclose(fit.y_mean.value, cy, atol=2) \
            and np.allclose(fit.x_mean.value, cx, atol=2) \
            and np.allclose(mean_fwhm_fit, fwhm, atol=3):
                coords.append((suby+fit.y_mean.value,subx+fit.x_mean.value))

            if debug:
                print('Coordinates (Y,X): {:.3f},{:.3f}'.format(y, x))
                print('fit peak = {:.3f}'.format(fit.amplitude.value))
                #print fit
                msg = 'fwhm_y in px = {:.3f}, fwhm_x in px = {:.3f}'
                print(msg.format(fwhm_y, fwhm_x))
                print('mean fit fwhm = {:.3f}'.format(mean_fwhm_fit))
                pp_subplots(subim, colorb=True)
        return coords

    def print_coords(coords):
        print('Blobs found:', len(coords))
        print(' ycen   xcen')
        print('------ ------')
        for i in range(len(coords[:,0])):
            print('{:.3f} \t {:.3f}'.format(coords[i,0], coords[i,1]))

    def print_abort():
        if verbose:
            print(sep)
            print('No potential sources found')
            print(sep)

    # --------------------------------------------------------------------------

    if not array.ndim == 2:
        raise TypeError('Input array is not a frame or 2d array')
    if not psf.ndim == 2 and psf.shape[0] < array.shape[0]:
        raise TypeError('Input psf is not a 2d array or has wrong size')
        
    # Getting the FWHM from the PSF array
    cenpsf = frame_center(psf)
    outdf = fit_2dgaussian(psf, cent=(cenpsf), debug=debug, full_output=True)
    fwhm_x, fwhm_y = outdf.at[0,'fwhm_x'],outdf.at[0,'fwhm_y']
    fwhm = np.mean([fwhm_x, fwhm_y])
    if verbose:
        print('FWHM =', fwhm)
        print()
    if debug:
        print('FWHM_y', fwhm_y)
        print('FWHM_x', fwhm_x)

    # Masking the center, 2*lambda/D is the expected IWA
    if mask: array = mask_circle(array, radius=fwhm)

    # Matched filter
    if matched_filter:
        frame_det = correlate(array, psf)
    else:
        frame_det = array

    # Estimation of background level
    _, median, stddev = sigma_clipped_stats(frame_det, sigma=5, iters=None)
    bkg_level = median + (stddev * bkg_sigma)
    if debug:
        print('Sigma clipped median = {:.3f}'.format(median))
        print('Sigma clipped stddev = {:.3f}'.format(stddev))
        print('Background threshold = {:.3f}'.format(bkg_level))
        print()

    if mode == 'lpeaks' or mode == 'log' or mode == 'dog':
        # Padding the image with zeros to avoid errors at the edges
        pad = 10
        array_padded = np.lib.pad(array, pad, 'constant', constant_values=0)

    if debug and plot and matched_filter:
        print('Input frame after matched filtering:')
        pp_subplots(frame_det, rows=2, colorb=True)

    if mode == 'lpeaks':
        # Finding local peaks (can be done in the correlated frame)
        coords_temp = peak_local_max(frame_det, threshold_abs=bkg_level,
                                     min_distance=int(np.ceil(fwhm)),
                                     num_peaks=20)
        coords = check_blobs(array_padded, coords_temp, fwhm, debug)
        coords = np.array(coords)
        if verbose and coords.shape[0]>0:  print_coords(coords)

    elif mode == 'log':
        sigma = fwhm*gaussian_fwhm_to_sigma
        coords = feature.blob_log(frame_det.astype('float'),
                                  threshold=bkg_level,
                                  min_sigma=sigma-.5, max_sigma=sigma+.5)
        if len(coords)==0:
            print_abort()
            return 0, 0
        coords = coords[:,:2]
        coords = check_blobs(array_padded, coords, fwhm, debug)
        coords = np.array(coords)
        if coords.shape[0]>0 and verbose:  print_coords(coords)

    elif mode == 'dog':
        sigma = fwhm*gaussian_fwhm_to_sigma
        coords = feature.blob_dog(frame_det.astype('float'),
                                  threshold=bkg_level,
                                  min_sigma=sigma-.5, max_sigma=sigma+.5)
        if len(coords)==0:
            print_abort()
            return 0, 0
        coords = coords[:,:2]
        coords = check_blobs(array_padded, coords, fwhm, debug)
        coords = np.array(coords)
        if coords.shape[0]>0 and verbose:  print_coords(coords)

    else:
        msg = 'Wrong mode. Available modes: lpeaks, log, dog.'
        raise TypeError(msg)

    if coords.shape[0]==0:
        print_abort()
        return 0, 0

    yy = coords[:,0]
    xx = coords[:,1]
    yy_final = []
    xx_final = []
    yy_out = []
    xx_out = []
    snr_list = []
    xx -= pad
    yy -= pad

    # Checking SNR for potential sources
    for i in range(yy.shape[0]):
        y = yy[i]
        x = xx[i]
        if verbose:
            print(sep)
            print('X,Y = ({:.1f},{:.1f})'.format(x,y))
        subim = get_square(array, size=15, y=y, x=x)
        snr = snr_ss(array, (x,y), fwhm, False, verbose=False)
        snr_list.append(snr)
        if snr >= snr_thresh:
            if verbose:
                _ = frame_quick_report(array, fwhm, (x,y), verbose=verbose)
            yy_final.append(y)
            xx_final.append(x)
        else:
            yy_out.append(y)
            xx_out.append(x)
            if verbose:  print('S/N constraint NOT fulfilled (S/N = {:.3f})'.format(snr))
            if debug:
                _ = frame_quick_report(array, fwhm, (x,y), verbose=verbose)

    if debug or full_output:
        table = Table([yy.tolist(), xx.tolist(), snr_list],
                      names=('y','x','px_snr'))
        table.sort('px_snr')
    yy_final = np.array(yy_final)
    xx_final = np.array(xx_final)
    yy_out = np.array(yy_out)
    xx_out = np.array(xx_out)

    if plot:
        #print
        #print sep
        #print 'Input frame showing all the detected blobs / potential sources:'
        #print 'In RED circles those that did not pass the SNR and 2dGaussian '
        #print 'fit constraints while in CYAN circles those that passed them.'
        fig, ax = plt.subplots(figsize=(6,6))
        im = ax.imshow(array, origin='lower', interpolation='nearest',
                       cmap='gray', alpha=0.8)

        # Option to plot axes in angular scale
        if NIRC2angscale and frame_size is not None:
            from scipy.ndimage import gaussian_filter
            # Converting axes from pixels to arcseconds
            # Find the middle value in the odd frame sizes
            center_val = int((frame_size / 2.0) + 0.5)
            # Place a tick every 0.5 arcseconds
            half_num_ticks = center_val // 50

            # Calculate the pixel locations at which to put ticks
            ticks = []
            for i in range(half_num_ticks, -half_num_ticks-1, -1):
                # Avoid ticks not showing on the last pixel
                if not center_val - (i) * 50 == frame_size:
                    ticks.append(center_val - (i) * 50)
                else:
                    ticks.append((center_val - (i) * 50) - 1)
                #print xticks
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)

            # Calculate the corresponding distance in arcseconds, measured from the center
            labels = []
            for i in range(half_num_ticks, -half_num_ticks-1, -1):
                labels.append(0.0 - (i) * 0.5)
                #print xlabels
            ax.set_xticklabels(labels)
            ax.set_yticklabels(labels)

            ax.set_xlabel("arcseconds", fontsize=12)
            ax.set_ylabel("arcseconds", fontsize=12)
            plt.tick_params(axis='both', which='major', labelsize=10)

            # Set the title of the plot
            if object_name is not None and inner_rad is not None:
                ax.set_title(pca_type + ' ' + object_name+' '+ str(ncomp) + 'pc ' + str(frame_size)+'+'+str(inner_rad),
                             fontsize=14)
            array_smoothed = gaussian_filter(array, sigma=(2.3, 2.3), order=0)
            plt.imshow(array_smoothed, origin='lower')

        else:
            colorbar_ax = fig.add_axes([0.92, 0.12, 0.03, 0.78])
            fig.colorbar(im, cax=colorbar_ax)
            ax.grid('off')

        for i in range(yy_out.shape[0]):
            y = yy_out[i]
            x = xx_out[i]
            circ = plt.Circle((x, y), radius=fwhm, color='red', fill=False,
                              linewidth=1.5, alpha=0.6)
            ax.text(x, y+1.5*fwhm, (int(x), int(y)), fontsize=10, color='red',
                    family='monospace', ha='center', va='top', weight='bold',
                    alpha=0.6)
            ax.add_patch(circ)
        for i in range(yy_final.shape[0]):
            y = yy_final[i]
            x = xx_final[i]
            circ = plt.Circle((x, y), radius=fwhm, color='cyan', fill=False,
                              linewidth=2)
            ax.text(x, y+1.5*fwhm, (int(x), int(y)), fontsize=10, color='cyan',
                    weight='heavy', family='monospace', ha='center', va='top')
            ax.add_patch(circ)
        # Save the plot if output path is provided
        # Don't show the plot when running pipeline (i.e. when saving figures)
        if save_plot is not None:
            plt.savefig(save_plot, dpi= 100, bbox_inches='tight')
        else:
            plt.show()

    if debug:  print(table)

    if full_output:
        return table, yy_final.shape[0]
    else:
        return yy_final, xx_final


def peak_coordinates(obj_tmp, fwhm, approx_peak=None, search_box=None,
                     channels_peak=False):
    """Find the pixel coordinates of maximum in either a frame or a cube,
    after convolution with gaussian. It first applies a gaussian filter, to
    lower the probability of returning a hot pixel (although it may still
    happen with clumps of hot pixels, hence the need for function
    "approx_stellar_position").

    Parameters
    ----------
    obj_tmp : cube_like or frame_like
        Input 3d cube or image.
    fwhm : float_like
        Input full width half maximum value of the PSF in pixels. This will be
        used as the standard deviation for Gaussian kernel of the Gaussian
        filtering.
    approx_peak: 2 components list or array, opt
        Gives the approximate coordinates of the peak.
    search_box: float or 2 components list or array, opt
        Gives the half-size in pixels of a box in which the peak is searched,
        around approx_peak. If float, it is assumed the same box size is wanted
        in both y and x. Note that this parameter should be provided if
        approx_peak is provided.
    channels_peak: bool, {False, True}, opt
        Whether returns the indices of the peak in each channel in addition to
        the global indices of the peak in the cube. If True, it would hence also
        return two 1d-arrays. (note: only available if the input is a 3d cube)

    Returns
    -------
    zz_max, yy_max, xx_max : integers
        Indices of highest throughput channel

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
        gauss_filt_tmp = frame_filter_gaussian2d(obj_tmp,
                                            fwhm/gaussian_sigma_to_fwhm)
        if approx_peak is None:
            ind_max = np.unravel_index(gauss_filt_tmp.argmax(),
                                       gauss_filt_tmp.shape)
        else:
            sbox = gauss_filt_tmp[approx_peak[0]-sbox_y:approx_peak[0]+sbox_y+1,
                                  approx_peak[1]-sbox_x:approx_peak[1]+sbox_x+1]
            ind_max_sbox = np.unravel_index(sbox.argmax(), sbox.shape)
            ind_max = (approx_peak[0]-sbox_y+ind_max_sbox[0],
                       approx_peak[1]-sbox_x+ind_max_sbox[1])

        return ind_max

    if ndims == 3:
        n_z = obj_tmp.shape[0]
        gauss_filt_tmp = np.zeros_like(obj_tmp)
        ind_ch_max = np.zeros([n_z,2])

        for zz in range(n_z):
            gauss_filt_tmp[zz] = frame_filter_gaussian2d(obj_tmp[zz],
                                                    fwhm[zz]/gaussian_sigma_to_fwhm)
            if approx_peak is None:
                ind_ch_max[zz] = np.unravel_index(gauss_filt_tmp[zz].argmax(),
                                                  gauss_filt_tmp[zz].shape)
            else:
                sbox[zz] = gauss_filt_tmp[zz, approx_peak[0]-sbox_y:\
                                          approx_peak[0]+sbox_y+1,
                                          approx_peak[1]-sbox_x:\
                                          approx_peak[1]+sbox_x+1]
                ind_max_sbox = np.unravel_index(sbox[zz].argmax(),
                                                sbox[zz].shape)
                ind_ch_max[zz] = (approx_peak[0]-sbox_y+ind_max_sbox[0],
                                  approx_peak[1]-sbox_x+ind_max_sbox[1])

        if approx_peak is None:
            ind_max = np.unravel_index(gauss_filt_tmp.argmax(),
                                       gauss_filt_tmp.shape)
        else:
            ind_max_tmp = np.unravel_index(sbox.argmax(),
                                           sbox.shape)
            ind_max = (ind_max_tmp[0]+approx_peak[0]-sbox_y,
                       ind_max_tmp[1]+approx_peak[1]-sbox_x)

        if channels_peak:
            return ind_max, ind_ch_max
        else:
            return ind_max


def mask_source_centers(array, fwhm, y, x):
    """ Creates a mask of ones with the size of the input frame and zeros at
    the center of the sources (planets) with coordinates x, y.

    Parameters
    ----------
    array : array_like
        Input frame.
    fwhm : float
        Size in pixels of the FWHM.
    y, x : tuples of int
        Coordinates of the center of the sources.

    Returns
    -------
    mask : array_like
        Mask frame.

    """
    if not array.ndim==2:
        raise TypeError('Wrong input array shape.')

    frame = array.copy()
    if not y and x:
        frame = mask_circle(frame, radius=2*fwhm)
        yy, xx = detection(frame, fwhm, plot=False, mode='log')
    else:
        yy = np.array(y); xx = np.array(x)
    mask = np.ones_like(array)
    # center sources become zeros
    mask[yy.astype('int'), xx.astype('int')] = 0
    return mask


