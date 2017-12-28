#! /usr/bin/env python

"""
Module with cube cosmetic functions for SDI datasets.
"""

from __future__ import division
from __future__ import print_function

__author__ = 'V. Christiaens @ UChile/ULg, Carlos Alberto Gomez Gonzalez'
__all__ = ['cube_correct_nan',
           'approx_stellar_position']

import copy
import numpy as np
from skimage.draw import circle
from astropy.stats import sigma_clipped_stats
from ..stats import sigma_filter


def cube_correct_nan(cube, neighbor_box=3, min_neighbors=3, verbose=False,
                     half_res_y=False):
    """Sigma filtering of nan pixels in a whole frame or cube. Intended for 
    SINFONI data.
    
    Parameters
    ----------
    cube : cube_like 
        Input 3d or 2d array.
    neighbor_box : int, optional
        The side of the square window around each pixel where the sigma and 
        median are calculated for the nan pixel correction.
    min_neighbors : int, optional
        Minimum number of good neighboring pixels to be able to correct the 
        bad/nan pixels.
    verbose: {False,True} bool, optional
        Whether to print more information or not during processing
    half_res_y: bool, {True,False}, optional
        Whether the input data have every couple of 2 rows identical, i.e. there
        is twice less angular resolution vertically than horizontally (e.g. 
        SINFONI data). The algorithm goes twice faster if this option is 
        rightfully set to True.
        
    Returns
    -------
    obj_tmp : array_like
        Output cube with corrected nan pixels in each frame
    """

    obj_tmp= cube.copy()

    ndims = obj_tmp.ndim
    if ndims != 2 and ndims != 3:
        raise TypeError("Input object is not two or three dimensional")

    if neighbor_box < 3 or neighbor_box%2 == 0:
        raise ValueError('neighbor_box should be an odd value greater than 3')
    max_neigh = sum(range(3,neighbor_box+2,2))
    if min_neighbors > max_neigh:
        min_neighbors = max_neigh
        msg = "Warning! min_neighbors was reduced to "+str(max_neigh) +\
              " to avoid bugs. \n"
        print(msg)

    def nan_corr_2d(obj_tmp):
        n_x = obj_tmp.shape[1]
        n_y = obj_tmp.shape[0]

        if half_res_y:
            if n_y%2 != 0: 
                msg = 'The input frames do not have an even number of rows. '
                msg2 = 'Hence, you should probably not be using the option '
                msg3 = 'half_res_y = True.'
                raise ValueError(msg+msg2+msg3)
            n_y = int(n_y/2)
            frame = obj_tmp
            obj_tmp = np.zeros([n_y,n_x])
            for yy in range(n_y):
                obj_tmp[yy] = frame[2*yy]

        # tuple with the 2D indices of each nan value of the frame
        nan_indices = np.where(np.isnan(obj_tmp))  
        nan_map = np.zeros_like(obj_tmp)
        nan_map[nan_indices] = 1
        nnanpix = int(np.sum(nan_map))
        # Correct nan with iterative sigma filter
        obj_tmp = sigma_filter(obj_tmp, nan_map, neighbor_box=neighbor_box, 
                               min_neighbors=min_neighbors, verbose=verbose) 
        if half_res_y:
            frame = obj_tmp
            n_y = 2*n_y
            obj_tmp = np.zeros([n_y,n_x])
            for yy in range(n_y):
                obj_tmp[yy] = frame[int(yy/2)]

        return obj_tmp, nnanpix


    if ndims == 2:
        obj_tmp, nnanpix = nan_corr_2d(obj_tmp)
        if verbose:
            print('\n There were ', nnanpix, ' nan pixels corrected.')

    elif ndims == 3:
        n_z = obj_tmp.shape[0]
        for zz in range(n_z):
            obj_tmp[zz], nnanpix = nan_corr_2d(obj_tmp[zz])
            if verbose:
                msg = 'In channel '+str(zz)+', there were '+str(nnanpix)
                msg2 = ' nan pixels corrected.'
                print(msg+msg2)

    if verbose:
        print('All nan pixels are corrected.')

    return obj_tmp
    

def approx_stellar_position(cube, fwhm, return_test=False, verbose=False):
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
    verbose: {True, False}, bool optional
        Chooses whether to print some additional information. 

    Returns:
    --------
    Array of y and x approx coordinates of the star in each channel of the cube
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
    test_result = np.ones(n_z)
    for zz in range(n_z):
        star_tmp_idx[zz] = peak_coordinates(obj_tmp[zz], fwhm[zz])
        
    #2/ Detect the outliers in each column
    _, med_y, stddev_y = sigma_clipped_stats(star_tmp_idx[:,0],sigma=2.5)
    _, med_x, stddev_x = sigma_clipped_stats(star_tmp_idx[:,1],sigma=2.5)
    lim_inf_y = med_y-3*stddev_y
    lim_sup_y = med_y+3*stddev_y
    lim_inf_x = med_x-3*stddev_x
    lim_sup_x = med_x+3*stddev_x
    
    if verbose:
        print("median y of star - 3sigma = ", lim_inf_y)
        print("median y of star + 3sigma = ", lim_sup_y)
        print("median x of star - 3sigma = ", lim_inf_x)
        print("median x of star + 3sigma = ", lim_sup_x)

    for zz in range(n_z):
        if ((star_tmp_idx[zz,0]<lim_inf_y) or (star_tmp_idx[zz,0]>lim_sup_y) or
            (star_tmp_idx[zz,1]<lim_inf_x) or (star_tmp_idx[zz,1]>lim_sup_x)):
            test_result[zz] = 0

    #3/ Replace by the median of neighbouring good coordinates if need be
    for zz in range(n_z):             
        if test_result[zz] == 0:
            ii= 1
            inf_neigh = max(0,zz-ii)
            sup_neigh = min(n_z-1,zz+ii)
            while test_result[inf_neigh] == 0 and test_result[sup_neigh] == 0:
                ii=ii+1
                inf_neigh = max(0,zz-ii)
                sup_neigh = min(n_z-1,zz+ii)
            if test_result[inf_neigh] == 1 and test_result[sup_neigh] == 1:
                star_approx_idx[zz] = np.floor((star_tmp_idx[sup_neigh]+ \
                                                star_tmp_idx[inf_neigh])/2.)
            elif test_result[inf_neigh] == 1: 
                star_approx_idx[zz] = star_tmp_idx[inf_neigh]
            else: star_approx_idx[zz] = star_tmp_idx[sup_neigh]
        else: star_approx_idx[zz] = star_tmp_idx[zz]

    if return_test:
        return star_approx_idx, test_result.astype(bool)
    else:
        return star_approx_idx
