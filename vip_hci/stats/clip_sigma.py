#! /usr/bin/env python

"""
Module with sigma clipping functions.
"""

from __future__ import division, print_function

__author__ = 'Carlos Alberto Gomez Gonzalez', 'V. Christiaens'
__all__ = ['clip_array',
           'sigma_filter']

import numpy as np
from scipy.ndimage.filters import generic_filter
from astropy.stats import median_absolute_deviation


# TODO: If possible, replace this function using
# scipy.ndimage.filters.generic_filter and astropy.stats.sigma_clip
def sigma_filter(frame_tmp, bpix_map, neighbor_box=3, min_neighbors=3,
                 verbose=False):
    """Sigma filtering of pixels in a 2d array.
    
    Parameters
    ----------
    frame_tmp : array_like 
        Input 2d array, image.
    bpix_map: array_like
        Input array of the same size as frame_tmp, indicating the locations of 
        bad/nan pixels by 1 (the rest of the array is set to 0)
    neighbor_box : int, optional
        The side of the square window around each pixel where the sigma and 
        median are calculated.
    min_neighbors : int, optional
        Minimum number of good neighboring pixels to be able to correct the 
        bad/nan pixels
    verbose : bool, optional
        Prints out the number of iterations.
        
    Returns
    -------
    frame_corr : array_like
        Output array with corrected bad/nan pixels
    
    """
    if frame_tmp.ndim != 2:
        raise TypeError('Input array is not a frame or 2d array')

    sz_y = frame_tmp.shape[0]  # get image y-dim
    sz_x = frame_tmp.shape[1]  # get image x-dim
    bp = bpix_map.copy()       # temporary bpix map; important to make a copy!
    im = frame_tmp             # corrected image
    nb = int(np.sum(bpix_map)) # number of bad pixels remaining
    #In each iteration, correct only the bpix with sufficient good 'neighbors'
    nit = 0                                 # number of iterations
    while nb > 0:
        nit += 1
        wb = np.where(bp)                   # find bad pixels
        gp = 1 - bp                         # temporary good pixel map
        for n in range(nb):
            #0/ Determine the box around each pixel
            half_box = np.floor(neighbor_box/2)
            hbox_b = min(half_box, wb[0][n])       # half size of the box at the
                                                   # bottom of the pixel
            hbox_t = min(half_box, sz_y-1-wb[0][n])# half size of the box at the
                                                   # top of the pixel
            hbox_l = min(half_box, wb[1][n])       # half size of the box to the
                                                   # left of the pixel
            hbox_r = min(half_box, sz_x-1-wb[1][n])# half size of the box to the
                                                   # right of the pixel
            # but in case we are at an edge, we want to extend the box by one 
            # row/column of pixels in the direction opposite to the edge to 
            # have 9 px instead of 6: 
            if half_box == 1:
                if wb[0][n] == sz_y-1:
                    hbox_b = hbox_b+1
                elif wb[0][n] == 0:
                    hbox_t = hbox_t+1
                if wb[1][n] == sz_x-1:
                    hbox_l = hbox_l+1
                elif wb[1][n] == 0:
                    hbox_r = hbox_r+1

            sgp = gp[int(wb[0][n]-hbox_b): int(wb[0][n]+hbox_t+1),
                     int(wb[1][n]-hbox_l): int(wb[1][n]+hbox_r+1)]
            if int(np.sum(sgp)) >= min_neighbors:
                sim = im[int(wb[0][n]-hbox_b): int(wb[0][n]+hbox_t+1),
                         int(wb[1][n]-hbox_l): int(wb[1][n]+hbox_r+1)]
                im[wb[0][n],wb[1][n]] = np.median(sim[np.where(sgp)])
                bp[wb[0][n],wb[1][n]] = 0
        nb = int(np.sum(bp))
    if verbose:
        print('Required number of iterations in the sigma filter: ', nit)
    return im


# TODO: If possible, replace this function using astropy.stats.sigma_clip
def clip_array(array, lower_sigma, upper_sigma, out_good=False, neighbor=False,
               num_neighbor=None, mad=False):
    """Sigma clipping for detecting outlying values in 2d array. If the
    parameter 'neighbor' is True the clipping can be performed in a local patch
    around each pixel, whose size depends on 'neighbor' parameter.
    
    Parameters
    ----------
    array : array_like 
        Input 2d array, image.
    lower_sigma : float 
        Value for sigma, lower boundary.
    upper_sigma : float 
        Value for sigma, upper boundary.
    out_good : bool, optional
        For choosing different outputs.
    neighbor : bool optional
        For clipping over the median of the contiguous pixels.
    num_neighbor : int, optional
        The side of the square window around each pixel where the sigma and 
        median are calculated. 
    mad : bool, optional
        If True, the median absolute deviation will be used instead of the 
        standard deviation.
        
    Returns
    -------
    good : array_like
        If out_good argument is true, returns the indices of not-outlying px.
    bad : array_like 
        If out_good argument is false, returns a vector with the outlier px.
    
    """
    if array.ndim != 2:
        raise TypeError("Input array is not two dimensional (frame)\n")

    values = array.copy()
    if neighbor and num_neighbor:
        median = generic_filter(array, function=np.median, 
                                size=(num_neighbor,num_neighbor), mode="mirror")
        if mad:
            sigma = generic_filter(array, function=median_absolute_deviation,                                 
                                   size=(num_neighbor,num_neighbor), 
                                   mode="mirror")
        else:
            sigma = generic_filter(array, function=np.std,                                 
                                   size=(num_neighbor,num_neighbor), 
                                   mode="mirror")
    else:
        median = np.median(values)
        sigma = values.std()
        
    good1 = values > (median - lower_sigma * sigma) 
    good2 = values < (median + upper_sigma * sigma)
    bad1 = values < (median - lower_sigma * sigma)
    bad2 = values > (median + upper_sigma * sigma)
    
    if out_good:
        # normal px indices in both good1 and good2
        good = np.where(good1 & good2)
        return good
    else:
        # deviating px indices in either bad1 or bad2
        bad = np.where(bad1 | bad2)
        return bad
    
  
