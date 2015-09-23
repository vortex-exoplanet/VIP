#! /usr/bin/env python

"""
Module with cube cosmetic functions for SDI datasets.
"""

from __future__ import division

__author__ = 'V. Christiaens'
__all__ = ['cube_correct_nan',
           'approx_stellar_position']

import copy
import numpy as np
from skimage.draw import circle
from astropy.stats import sigma_clipped_stats
from ..stats import sigma_filter


def cube_correct_nan(cube, neighbor_box=3, min_neighbors=3, verbose=False):
    """Sigma filtering of nan pixels in a whole frame or cube (originally 
    intended for SINFONI data).
    
    Parameters
    ----------
    cube : cube_like 
        Input 3d or 2d array.
    neighbor_box_corr : int, optional
        The side of the square window around each pixel where the sigma and 
        median are calculated for the nan pixel correction.
    min_neighbors : int, optional
        Minimum number of good neighboring pixels to be able to correct the 
        bad/nan pixels.
        
    Returns
    -------
    obj_tmp : array_like
        Output cube with corrected nan pixels in each frame
    """

    obj_tmp = cube.copy()
    if obj_tmp.ndim==2:
        nan_indices = np.where(np.isnan(obj_tmp))
        nan_map = np.zeros_like(obj_tmp)
        nan_map[nan_indices] = 1
        nnanpix = int(np.sum(nan_map))
        if verbose == True:
            msg = 'In frame there are {} nan pixels to be corrected.'
            print msg.format(nnanpix)
        #Correct nan with iterative sigma filter
        obj_tmp = sigma_filter(obj_tmp, nan_map, 
                                   neighbor_box=neighbor_box, 
                                   min_neighbors=min_neighbors, 
                                   verbose=verbose) 
        if verbose == True:
            print 'All nan pixels are corrected.'
            
    elif obj_tmp.ndim==3:
        n_z = obj_tmp.shape[0]
        for zz in range(n_z):
            nan_indices = np.where(np.isnan(obj_tmp[zz]))
            nan_map = np.zeros_like(obj_tmp[zz])
            nan_map[nan_indices] = 1
            nnanpix = int(np.sum(nan_map))
            if verbose == True:
                msg = 'In channel {} there are {} nan pixels to be corrected.'
                print msg.format(zz, nnanpix)
            #Correct nan with iterative sigma filter
            obj_tmp[zz] = sigma_filter(obj_tmp[zz], nan_map, 
                                       neighbor_box=neighbor_box, 
                                       min_neighbors=min_neighbors, 
                                       verbose=verbose) 
            if verbose == True:
                print 'All nan pixels are corrected.'
                
    return obj_tmp
    

def approx_stellar_position(cube, fwhm, return_test=False):
    """FIND THE APPROX COORDS OF THE STAR IN EACH CHANNEL (even the ones 
    dominated by noise)
    
    Parameters
    ----------
    obj_tmp : array_like
        Input 3d cube
    fwhm : float or array 1D
        Input full width half maximum value of the PSF for each channel. 
        This will be used as the standard deviation for Gaussian kernel 
        of the Gaussian filtering.
        If float, it is assumed the same for all channels.
    return_test: bool, {False,True}, optional
        Whether the test result vector (a bool vector with whether the star 
        centroid could be find in the corresponding channel) should be returned
        as well, along with the approx stellar coordinates.

    Returns:
    --------
    Array of x and y approx coordinates of the star in each channel of the cube
    if return_test: it also returns the test result vector
    """
    from ..phot import peak_coordinates

    obj_tmp = cube.copy()
    n_z = obj_tmp.shape[0]

    if isinstance(fwhm,float) or isinstance(fwhm,int):
        fwhm_scal = fwhm
        fwhm = np.zeros((n_z))
        fwhm[:] = fwhm_scal

    #1/ Write a 2-columns array with indices of all max pixel values in the cube
    star_tmp_idx = np.zeros([n_z,2])
    star_approx_idx = np.zeros([n_z,2])
    test_result = np.zeros([n_z,2])
    for zz in range(n_z):
        star_tmp_idx[zz] = peak_coordinates(obj_tmp[zz], fwhm[zz])
        
    #2/ Detect the outliers in each column
    _, med_y, stddev_y = sigma_clipped_stats(star_tmp_idx[:,0],sigma=2.5)
    _, med_x, stddev_x = sigma_clipped_stats(star_tmp_idx[:,1],sigma=2.5)
    lim_inf_y = med_y-3*stddev_y
    lim_sup_y = med_y+3*stddev_y
    lim_inf_x = med_x-3*stddev_x
    lim_sup_x = med_x+3*stddev_x

    print "median y of star - 3sigma = ", lim_inf_y
    print "median y of star + 3sigma = ", lim_sup_y
    print "median x of star - 3sigma = ", lim_inf_x
    print "median x of star + 3sigma = ", lim_sup_x

     for zz in range(n_z):
        if ((star_tmp_idx[zz,0]<lim_inf_y) or (star_tmp_idx[zz,0]>lim_sup_y) or
            (star_tmp_idx[zz,1]<lim_inf_x) or (star_tmp_idx[zz,1]>lim_sup_x)):
            test_result[zz] = 1

    #3/ Replace by the median of neighbouring good coordinates if need be
    for zz in range(n_z):             
        if test_result[zz] == 1:
            ii= 1
            inf_neigh = max(0,zz-ii)
            sup_neigh = min(n_z-1,zz+ii)
            while test_result[inf_neigh] == 1 and test_result[sup_neigh] == 1:
                ii=ii+1
                inf_neigh = max(0,zz-ii)
                sup_neigh = min(n_z-1,zz+ii)
            if test_result[inf_neigh] == 0 and test_result[sup_neigh] == 0:
                star_approx_idx[zz] = np.floor((star_tmp_idx[sup_neigh]+ \
                                                star_tmp_idx[inf_neigh])/2.)
            elif test_result[inf_neigh] == 0: 
                star_approx_idx[zz] = star_tmp_idx[inf_neigh]
            else: star_approx_idx[zz] = star_tmp_idx[sup_neigh]
        else: star_approx_idx[zz] = star_tmp_idx[zz]

    if return_test:
        return star_approx_idx, test_result.astype(bool)
    else:
        return star_approx_idx
