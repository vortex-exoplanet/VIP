#! /usr/bin/env python

"""
Module with detection algorithms.
"""

from __future__ import division

__author__ = 'C. Gomez @ ULg'
__all__ = ['detection', 
           'mask_source_centers',
           'peak_coordinates']

import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage.filters import correlate, gaussian_filter
from skimage import feature
from astropy.stats import sigma_clipped_stats
from astropy.table import Table
from astropy.modeling.models import Gaussian2D
from astropy.modeling.fitting import LevMarLSQFitter
from photutils import CircularAperture, aperture_photometry
from photutils.detection import findstars
from skimage.feature import peak_local_max
from ..var import mask_circle, pp_subplots, get_square_robust, fit_2dgaussian, frame_center
from ..var.filters import SIGMA2FWHM, gaussian_kernel
from .snr import snr_student
from .frame_analysis import frame_quick_report


def detection(array, psf, bkg_sigma=3, mode='lpeaks', matched_filter=True, 
              mask=True, snr_thresh=5, plot=True, debug=False, 
              full_output=False, verbose=True):                 
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
    mode : {'lpeaks','irafsf','daofind','log','dog'}, optional
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
    
    Returns
    -------
    yy, xx : array_like
        Two vectors with the y and x coordinates of the centers of the sources 
        (putative planets). 
    If full_output is True then a table with all the candidates that passed the
    2d Gaussian fit constrains and their SNR is returned. Also the count of 
    companions with SNR>5 (those with highest probability of being true 
    detections).
                
    Notes
    -----
    The PSF is used to run a matched filter (correlation) which is equivalent 
    to a convolution filter. Filtering the image will smooth the noise and
    maximize detectability of objects with a shape similar to the kernel. 
    The background level or threshold is found with sigma clipped statistics 
    (5 sigma over the median) on the image. Then 5 different strategies can be 
    used to detect the blobs (planets):
    
    Local maxima + 2d Gaussian fit. The local peaks above the background on the 
    (correlated) frame are detected. A maximum filter is used for finding local 
    maxima. This operation dilates the original image and merges neighboring 
    local maxima closer than the size of the dilation. Locations where the 
    original image is equal to the dilated image are returned as local maxima.
    The minimum separation between the peaks is 1*FWHM. A 2d Gaussian fit is 
    done on each of the maxima constraining the position on the subimage and the
    sigma of the fit. Finally an SNR criterion can be applied. 
    
    Laplacian of Gaussian. It computes the Laplacian of Gaussian images with 
    successively increasing standard deviation and stacks them up in a cube. 
    Blobs are local maximas in this cube. Detecting larger blobs is especially 
    slower because of larger kernel sizes during convolution. Only bright blobs 
    on dark backgrounds are detected. This is the most accurate and slowest 
    approach.
    
    Difference of Gaussians. This is a faster approximation of LoG approach. In 
    this case the image is blurred with increasing standard deviations and the 
    difference between two successively blurred images are stacked up in a cube. 
    This method suffers from the same disadvantage as LoG approach for detecting 
    larger blobs. Blobs are again assumed to be bright on dark.
    
    Irafsf. starfind algorithm (IRAF software) searches images for local density 
    maxima that have a peak amplitude greater than threshold above the local 
    background and have a PSF full-width half-maximum similar to the input fwhm. 
    The objects' centroid, roundness (ellipticity), and sharpness are calculated 
    using image moments.
    
    Daofind. Searches images for local density maxima that have a peak amplitude 
    greater than threshold (approximately; threshold is applied to a convolved 
    image) and have a size and shape similar to the defined 2D Gaussian kernel. 
    The Gaussian kernel is defined by the fwhm, ratio, theta, and sigma_radius 
    input parameters. Daofind finds the object centroid by fitting the the 
    marginal x and y 1D distributions of the Gaussian kernel to the marginal x 
    and y distributions of the input (unconvolved) data image.
                
    """
    def print_coords(coords):
        print 'Blobs found:', len(coords)
        print ' ycen   xcen'
        print '------ ------'
        for i in range(len(coords[:,0])):
            print ' ', coords[i,0], '\t', coords[i,1]
    
    if not array.ndim == 2:
        raise TypeError('Input array is not a frame or 2d array')
    if not psf.ndim == 2 and psf.shape[0] < array.shape[0]:
        raise TypeError('Input psf is not a 2d array or has wrong size')
    
    # Getting the FWHM with a 2d gaussian fit on the PSF
    gauss = Gaussian2D(amplitude=1, x_mean=5, y_mean=5, x_stddev=3.5, 
                       y_stddev=3.5, theta=0)
    fitter = LevMarLSQFitter()                  # Levenberg-Marquardt algorithm
    psf_subimage = get_square_robust(psf, 9, frame_center(psf)[0],frame_center(psf)[1])
    y, x = np.indices(psf_subimage.shape)
    fit = fitter(gauss, x, y, psf_subimage)
    fwhm = np.mean([fit.y_stddev.value*SIGMA2FWHM, 
                    fit.x_stddev.value*SIGMA2FWHM])
    if verbose:  
        print 'FWHM =', fwhm
        print
    if debug:  
        print 'FWHM_y ', fit.y_stddev.value*SIGMA2FWHM
        print 'FWHM_x ', fit.x_stddev.value*SIGMA2FWHM  
        print
    
    # Masking the center, 2*lambda/D is the expected IWA
    if mask:  array = mask_circle(array, radius=2*fwhm)
    
    # Matched filter
    if matched_filter:  
        frame_det = correlate(array, psf)
    else:
        frame_det = array
    
    # Estimation of background level
    _, median, stddev = sigma_clipped_stats(frame_det, sigma=5, iters=None)
    bkg_level = median + (stddev * bkg_sigma)
    if debug:  
        print 'Sigma clipped median = {:.3f}'.format(median)
        print 'Sigma clipped stddev = {:.3f}'.format(stddev)
        print 'Background threshold = {:.3f}'.format(bkg_level)
        print
    
    round = 0.3   # roundness constraint
    
    # Padding the image with zeros to avoid errors at the edges
    pad = 10
    array_padded = np.lib.pad(array, pad, 'constant', constant_values=0)
        
    if debug and plot and matched_filter:  
        print 'Input frame after matched filtering'
        pp_subplots(frame_det, size=6, rows=2, colorb=True)
        
    if mode=='lpeaks':
        # Finding local peaks (can be done in the correlated frame)                                           
        coords_temp = peak_local_max(frame_det, threshold_abs=bkg_level, 
                                     min_distance=fwhm, num_peaks=20)
        y_temp = coords_temp[:,0]
        x_temp = coords_temp[:,1]
        coords = []
        # Fitting a 2d gaussian to each local maxima position
        for y,x in zip(y_temp,x_temp):
            subim, suby, subx = get_square_robust(array_padded, 2*int(np.ceil(fwhm)), 
                                           y+pad, x+pad, position=True) 
            cy, cx = frame_center(subim)
            
            gauss = Gaussian2D(amplitude=subim.max(), 
                               x_mean=cx, y_mean=cy, 
                               x_stddev=fwhm/SIGMA2FWHM, 
                               y_stddev=fwhm/SIGMA2FWHM, theta=0)
            
            sy, sx = np.indices(subim.shape)
            fit = fitter(gauss, sx, sy, subim)
            
            # checking that the amplitude is positive > 0
            # checking whether the x and y centroids of the 2d gaussian fit 
            # coincide with the center of the subimage (within 2px error)
            # checking whether the mean of the fwhm in y and x of the fit are
            # close to the FWHM_PSF with a margin of 3px
            fwhm_y = fit.y_stddev.value*SIGMA2FWHM
            fwhm_x = fit.x_stddev.value*SIGMA2FWHM
            mean_fwhm_fit = np.mean([np.abs(fwhm_x), np.abs(fwhm_y)]) 
            if fit.amplitude.value>0 \
            and np.allclose(fit.y_mean.value, cy, atol=2) \
            and np.allclose(fit.x_mean.value, cx, atol=2) \
            and np.allclose(mean_fwhm_fit, fwhm, atol=3):                     
                coords.append((suby+fit.y_mean.value, subx+fit.x_mean.value))
        
            if debug:  
                print 'Coordinates (Y,X): {:.3f},{:.3f}'.format(y, x)
                print 'fit peak = {:.3f}'.format(fit.amplitude.value)
                #print fit
                msg = 'fwhm_y in px = {:.3f}, fwhm_x in px = {:.3f}'
                print msg.format(fwhm_y, fwhm_x) 
                print 'mean fit fwhm = {:.3f}'.format(mean_fwhm_fit)
                pp_subplots(subim, colorb=True)
        
        coords = np.array(coords)
        if verbose and coords.shape[0]>0:  print_coords(coords)
    
    elif mode=='daofind':                 
        tab = findstars.daofind(frame_det, fwhm=fwhm, threshold=bkg_level,
                                roundlo=-round,roundhi=round)
        coords = np.transpose((np.array(tab['ycentroid']), 
                               np.array(tab['xcentroid'])))
        if verbose:
            print 'Blobs found:', len(coords)
            print tab['ycentroid','xcentroid','roundness1','roundness2','flux']
                  
    elif mode=='irafsf':                
        tab = findstars.irafstarfind(frame_det, fwhm=fwhm, 
                                     threshold=bkg_level,
                                     roundlo=0, roundhi=round)
        coords = np.transpose((np.array(tab['ycentroid']), 
                               np.array(tab['xcentroid'])))
        if verbose:
            print 'Blobs found:', len(coords)
            print tab['ycentroid','xcentroid','fwhm','flux','roundness']
        
    elif mode=='log':
        sigma = fwhm/SIGMA2FWHM
        coords = feature.blob_log(frame_det.astype('float'), 
                                  threshold=bkg_level, 
                                  min_sigma=sigma-.5, max_sigma=sigma+.5)
        coords = coords[:,:2]
        if coords.shape[0]>0 and verbose:  print_coords(coords)
     
    elif mode=='dog':
        sigma = fwhm/SIGMA2FWHM
        coords = feature.blob_dog(frame_det.astype('float'), 
                                  threshold=bkg_level, 
                                  min_sigma=sigma-.5, max_sigma=sigma+.5)
        coords = coords[:,:2]
        if coords.shape[0]>0 and verbose:  print_coords(coords)
        
    else:
        msg = 'Wrong mode. Available modes: lpeaks, daofind, irafsf, log, dog.'
        raise TypeError(msg)

    if coords.shape[0]==0:
        if verbose:  
            print '_________________________________________'
            print 'No potential sources found'
            print '_________________________________________'
        return 0, 0
    
    yy = coords[:,0]
    xx = coords[:,1]
    yy_final = [] 
    xx_final = []
    yy_out = []
    xx_out = []
    snr_list = []
    px_list = []
    if mode=='lpeaks':
        xx -= pad
        yy -= pad
    
    # Checking SNR for potential sources
    for i in xrange(yy.shape[0]):
        y = yy[i] 
        x = xx[i] 
        if verbose: 
            print '_________________________________________'
            print 'Y,X = ({:.1f},{:.1f}) -----------------------'.format(y, x)
        subim = get_square_robust(array, size=15, y=y, x=x)
        snr = snr_student(array, y, x, fwhm, False, verbose=False)
        snr_list.append(snr)
        px_list.append(array[y,x])
        if snr >= snr_thresh and array[y,x]>0:
            if plot:
                pp_subplots(subim, size=2)
            if verbose:  
                _ = frame_quick_report(array, fwhm, y=y, x=x , verbose=verbose)
            yy_final.append(y)
            xx_final.append(x)
        else:
            yy_out.append(y)
            xx_out.append(x)
            if verbose:  print 'SNR constraint NOT fulfilled'
            if debug:
                if plot:
                    pp_subplots(subim, size=2)
                _ = frame_quick_report(array, fwhm, y=y, x=x , verbose=verbose)
            else:
                if verbose:  print 'SNR = {:.3f}'.format(snr)

    if debug or full_output:
        table = Table([yy.tolist(), xx.tolist(), px_list, snr_list], 
                      names=('y','x','px_val','px_snr'))
        table.sort('px_snr')
    yy_final = np.array(yy_final) 
    xx_final = np.array(xx_final) 
    yy_out = np.array(yy_out) 
    xx_out = np.array(xx_out) 
    
    if plot: 
        print
        print '_________________________________________'           
        print'Input frame showing all the detected blobs / potential sources'
        print 'In red circles those that did not pass the SNR and 2dGauss fit constraints'
        print 'In cyan circles those that passed the constraints'
        fig, ax = plt.subplots(figsize=(8,8))
        im = ax.imshow(array, origin='lower', interpolation='nearest', 
                       cmap='gray')
        colorbar_ax = fig.add_axes([0.92, 0.12, 0.03, 0.78])
        fig.colorbar(im, cax=colorbar_ax)
        ax.grid('off')
        
        for i in xrange(yy_out.shape[0]):
            y = yy_out[i]
            x = xx_out[i]
            circ = plt.Circle((x, y), radius=2*fwhm, color='red', fill=False,
                              linewidth=2)
            ax.text(x, y+5*fwhm, (int(y),int(x)), fontsize=10, color='red', 
                    family='monospace', ha='center', va='top', weight='bold')
            ax.add_patch(circ)
        for i in xrange(yy_final.shape[0]):
            y = yy_final[i]
            x = xx_final[i]
            circ = plt.Circle((x, y), radius=2*fwhm, color='cyan', fill=False, 
                              linewidth=2)
            ax.text(x, y+5*fwhm, (int(y),int(x)), fontsize=10, color='cyan', 
                    weight='heavy', family='monospace', ha='center', va='top')
            ax.add_patch(circ)
        plt.show()
    
    if debug:  print table
    
    if full_output:
        return table, yy_final.shape[0]
    else:
        return yy_final, xx_final

    
def peak_coordinates(obj_tmp, fwhm, approx_peak=None, search_box=None):
    """Find the pixel coordinates of maximum in either a frame or a cube, 
    after convolution with gaussian. It first applies a gaussian filter, to 
    lower the probability of returning a hot pixel (although it may still 
    happen with clumps of hot pixels, hence the need for function 
    "approx_stellar_position".
    
    Parameters
    ----------
    obj_tmp : cube_like or frame_like
        Input 3d cube or image.
    fwhm     : float_like
        Input full width half maximum value of the PSF. This will be used as 
        the standard deviation for Gaussian kernel of the Gaussian filtering.
    approx_peak: 
        List_like, vector of 2 components giving the approximate coordinates 
        of the peak.
    search_box: 
        Scalar or list_like (of 2 components) giving the half-size in pixels 
        of a box in which the peak is searched, aroung approx_peak.

    Returns
    -------
    zz_max, yy_max, xx_max : integers
        Indices of highest throughput channel
    """

    ndims = len(obj_tmp.shape)
    assert ndims == 2 or ndims == 3, "Array is not two or three dimensional"

    if ndims == 2:
        gauss_filt_tmp = gaussian_filter(obj_tmp, fwhm)
        if approx_peak == None:
            ind_max = np.unravel_index(gauss_filt_tmp.argmax(), 
                                       gauss_filt_tmp.shape)
        else:
            assert len(approx_peak) == 2, "Approx peak is not two dimensional"
            nel_sbox = len(search_box)
            msg = "The search box does not have the right number of elements"
            assert nel_sbox == 1 or nel_sbox == 2, msg
            if nel_sbox == 1:
                search_box_y = search_box
                search_box_x = search_box
            else:
                search_box_y = search_box[0]
                search_box_x = search_box[1]
            sbox = gauss_filt_tmp[approx_peak[0]-search_box_y:approx_peak[0]\
                                  +search_box_y+1,approx_peak[1]-search_box_x\
                                  :approx_peak[1]+search_box_x+1]
            ind_max_sbox = np.unravel_index(sbox.argmax(), sbox.shape)
            ind_max = (approx_peak[0]-search_box_y+ind_max_sbox[0],
                       approx_peak[1]-search_box_x+ind_max_sbox[1])

    if ndims == 3:
        n_z = obj_tmp.shape[0]
        gauss_filt_tmp = np.zeros_like(obj_tmp)

        msg2 = "The search for the peak in a 3D cube, with the approx peak"
        msg2 +=  "option is not implemented.\n"
        assert approx_peak == None, msg2
        for zz in range(n_z):
            gauss_filt_tmp[zz] = gaussian_filter(obj_tmp[zz], fwhm[zz])

        ind_max = np.unravel_index(gauss_filt_tmp.argmax(), 
                                   gauss_filt_tmp.shape)
        
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
