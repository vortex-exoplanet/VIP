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
from photutils.detection import findstars
from skimage.feature import peak_local_max
from ..var import mask_circle, pp_subplots, get_square
from ..var.filters import SIGMA2FWHM
from .snr import snr_student
from .frame_analysis import frame_quick_report


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


def detection(array, fwhm, psf, mode='irafsf', mask=True, snr_thresh=5,
              plot=True, debug=False, full_output=False, verbose=True):                 
    """ Finds blobs in a 2d array. The algorithm is designed for automatically 
    finding planets in post-processed high contrast final frames. Blob can be 
    defined as a region of an image in which some properties are constant or 
    vary within a prescribed range of values.
     
    The PSF is used to run a matched filter (correlation) which is equivalent 
    to a convolution filter. Filtering the image will smooth the noise and
    maximize detectability of objects with a shape similar to the kernel. 
    The background level or threshold is found with sigma clipped statistics 
    (5 sigma over the median) on the original image. Then 5 different strategies 
    can be used to detect the blobs (planets):
    
    Local maxima. The local peaks above the background threshold on the 
    correlated frame are detected. The minimum separation between the peaks is 
    2*FWHM. 
    
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
    
    Parameters
    ----------
    array : array_like, 2d
        Input frame.
    fwhm : float
        Size of the fwhm in pixels.
    psf : array_like
        Input psf.
    mode : {'irafsf','daofind','log','dog','matched'}, optional
        Sets with algorithm to use. Each algorithm yields different results.
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
        They will be Int for modes LOG or DOG and Float for modes daofind and
        irafsf.
                
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
        
    # Masking the center, 2*lambda/D is the expected IWA
    if mask:  array = mask_circle(array, radius=2*fwhm)
    
    # Estimation of background level
    _, median, stddev = sigma_clipped_stats(array, sigma=5)
    bkg_level = median + (stddev * 5)
    if debug:  
        print 'Sigma clipped median = {:.3f}'.format(median)
        print 'Sigma clipped stddev = {:.3f}'.format(stddev)
        print 'Background threshold = {:.3f}'.format(bkg_level)
        print
    
    round = 0.3   # roundness constraint
    
    # Matched filter
    array_correl = correlate(array, psf)
        
    if debug and plot:  
        print 'Input frame after matched filtering'
        pp_subplots(array_correl, size=6, rows=2)
        
    if mode=='lpeaks':
        # Finding local peaks                                            
        coords = peak_local_max(array_correl, threshold_abs=bkg_level, 
                                min_distance=fwhm*2, num_peaks=20)
        if verbose:  print_coords(coords)
    
    elif mode=='daofind':                 
        tab = findstars.daofind(array_correl, fwhm=fwhm, threshold=bkg_level,
                                roundlo=-round,roundhi=round)
        coords = np.transpose((np.array(tab['ycentroid']), 
                               np.array(tab['xcentroid'])))
        if verbose:
            print 'Blobs found:', len(coords)
            print tab['ycentroid','xcentroid','roundness1','roundness2','flux']
                  
    elif mode=='irafsf':                
        tab = findstars.irafstarfind(array_correl, fwhm=fwhm, 
                                     threshold=bkg_level,
                                     roundlo=0, roundhi=round)
        coords = np.transpose((np.array(tab['ycentroid']), 
                               np.array(tab['xcentroid'])))
        if verbose:
            print 'Blobs found:', len(coords)
            print tab['ycentroid','xcentroid','fwhm','flux','roundness']
        
    elif mode=='log':
        sigma = fwhm/SIGMA2FWHM
        coords = feature.blob_log(array_correl.astype('float'), 
                                  threshold=bkg_level, 
                                  min_sigma=sigma-.5, max_sigma=sigma+.5)
        coords = coords[:,:2]
        if verbose:  print_coords(coords)
     
    elif mode=='dog':
        sigma = fwhm/SIGMA2FWHM
        coords = feature.blob_dog(array_correl.astype('float'), 
                                  threshold=bkg_level, 
                                  min_sigma=sigma-.5, max_sigma=sigma+.5)
        coords = coords[:,:2]
        if verbose:  print_coords(coords)
        
    else:
        msg = 'Wrong mode. Available modes: lpeaks, daofind, irafsf, log, dog.'
        raise TypeError(msg)

    yy = coords[:,0]
    xx = coords[:,1]
    yy_final = [] 
    xx_final = []
    yy_out = []
    xx_out = []
    snr_list = []
    px_list = []
    
    for i in xrange(yy.shape[0]):
        y = yy[i]
        x = xx[i]
        if verbose: 
            print
            print '_________________________________________'
            print 'Y,X = ({:.0f},{:.0f}) -------------------------'.format(y, x)
        subim = get_square(array, size=15, y=y, x=x)
        snr = snr_student(array, y, x, fwhm, False, verbose=False)
        snr_list.append(snr)
        px_list.append(array[y,x])
        if snr >= snr_thresh:
            if plot:
                pp_subplots(subim, size=2)
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
        table.sort('px_val')
    yy_final = np.array(yy_final)
    xx_final = np.array(xx_final)
    yy_out = np.array(yy_out)
    xx_out = np.array(xx_out)
    
    if plot: 
        print
        print '_________________________________________'           
        print'Input frame showing all the detected blobs'
        print 'In red circles those that did not pass the SNR constraint'
        print 'In cyan circles those that have and SNR >= 5'
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
        return table 
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
