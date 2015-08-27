#! /usr/bin/env python

"""
Module with cube cosmetic functions for SDI datasets.
"""

from __future__ import division

__author__ = 'V. Christiaens', 'C. Gomez @ ULg'
__all__ = ['cube_correct_nan',
           'bad_pixel_removal',
           'approx_stellar_position',
           'find_outliers',
           'reject_outliers']

import numpy as np
from skimage.draw import circle
from astropy.stats import sigma_clipped_stats
from ..stats import sigma_filter


def cube_correct_nan(cube, neighbor_box=3, min_neighbors=3, verbose=False):
    """Sigma filtering of nan pixels in a whole frame or cube. Intended for 
    SINFONI data.
    
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
        nan_indices = np.where(np.isnan(obj_tmp)) # tuple with the 2D indices of each nan value
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
            nan_indices = np.where(np.isnan(obj_tmp[zz])) # tuple with the 2D indices of each nan value
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
    

def approx_stellar_position(cube, fwhm):
    """FIND THE APPROX COORDS OF THE STAR IN EACH CHANNEL (even the ones 
    dominated by noise)
    
    Parameters
    ----------
    obj_tmp : array_like
        Input 3d cube
    fwhm : float
        Input full width half maximum value of the PSF. This will be used as 
        the standard deviation for Gaussian kernel of the Gaussian filtering.
    """
    from ..phot import peak_coordinates

    obj_tmp = cube
    n_z = obj_tmp.shape[0]

    #1/ Write a 2-columns matrix with the coords of all the max pixel values in the cube
    star_tentative_coords = np.zeros([n_z,2])
    star_approx_coords = np.zeros([n_z,2])
    test_result = np.zeros([n_z,2])
    for zz in range(n_z):
        star_tentative_coords[zz] = peak_coordinates(obj_tmp[zz], fwhm[zz])
        
    #2/ Detect the outliers in each column
    lim_inf_y = np.median(star_tentative_coords[:,0])-(3*np.std(star_tentative_coords[:,0]))
    print "median y of star - 3sigma = ", lim_inf_y
    lim_sup_y = np.median(star_tentative_coords[:,0])+(3*np.std(star_tentative_coords[:,0]))
    print "median y of star + 3sigma = ", lim_sup_y
    lim_inf_x = np.median(star_tentative_coords[:,1])-(3*np.std(star_tentative_coords[:,1]))
    print "median x of star - 3sigma = ", lim_inf_x
    lim_sup_x = np.median(star_tentative_coords[:,1])+(3*np.std(star_tentative_coords[:,1]))
    print "median x of star + 3sigma = ", lim_sup_x

    for zz in range(n_z):
        if star_tentative_coords[zz,0] < lim_inf_y or star_tentative_coords[zz,0] > lim_sup_y: test_result[zz,0] = 1
        if star_tentative_coords[zz,1] < lim_inf_x or star_tentative_coords[zz,1] > lim_sup_x: test_result[zz,1] = 1

    #3/ Replace by the median of neighbouring good coordinates if need be
    for zz in range(n_z):             
        if test_result[zz,0] == 1 or test_result[zz,1] == 1:
            ii= 1
            inf_neigh = max(0,zz-ii)
            sup_neigh = min(n_z-1,zz+ii)
            while (test_result[inf_neigh,0] == 1 or test_result[inf_neigh,1] == 1) and (test_result[sup_neigh,0] == 1 or test_result[sup_neigh,1] == 1):
                ii=ii+1
                inf_neigh = max(0,zz-ii)
                sup_neigh = min(n_z-1,zz+ii)
            if (test_result[inf_neigh,0] == 0 and test_result[inf_neigh,1] == 0) and (test_result[sup_neigh,0] == 0 and test_result[sup_neigh,1] == 0):
                star_approx_coords[zz] = np.floor((star_tentative_coords[sup_neigh]+star_tentative_coords[inf_neigh])/2.)
            elif (test_result[inf_neigh,0] == 0 and test_result[inf_neigh,1] == 0): 
                star_approx_coords[zz] = star_tentative_coords[inf_neigh]
            else: star_approx_coords[zz] = star_tentative_coords[sup_neigh]
        else: star_approx_coords[zz] = star_tentative_coords[zz]

    return star_approx_coords


def bad_pixel_removal(obj_tmp, center, fwhm, sig=5., protect_psf=True, verbose=False, hcube_ii='',DEBUG=False):
    """
    Function to correct the bad pixels in obj_tmp; either a frame or a wavelength cube.
    No bad pixel is corrected in a circle of 2 fwhm radius around the approximate location of the star.
    Parameters
    ----------
    obj_tmp : cube_like or frame_like 
        Input 3d cube or 2d image.
    center : float scalar or vector
        Vector with approximate y and x coordinates of the star for each channel (cube_like), or single 2-elements vector (frame_like)
    fwhm: float scalar or vector
        Vector containing the full width half maximum of the PSF in pixels, for each channel (cube_like); or single value (frame_like)
    sig=5.: Float scalar
        Value representing the number of "sigmas" above or below the "median" of the neighbouring pixel, to consider a pixel as bad. See details on parameter "m" fo function reject_outlier
    protect_psf: True if you want to protect a circular region centered on the star (2*fwhm radius) from any bpix corr (if False, maybe risk to modify a peak value; but if True real bpix within the core are not corrected)
    verbose: Boolean
        If true, it will print the number of bad pixels and number of iterations required in each frame 
    hcube_ii:
        Optional, the header of the cube. It is required in case DEBUG is True, to write some intermediate fits files.
    DEBUG: Boolean
        If true, it will create fits files containing the bad pixel map and corrected frame at each iteration.
    """
    ndims = obj_tmp.ndim
    assert ndims == 2 or ndims == 3, "Object is not two or three dimensional.\n"


    def bp_removal_2d(obj_tmp, center, fwhm, sig, protect_psf, verbose):        
        n_x = obj_tmp.shape[1]
        n_y = obj_tmp.shape[0]
        fwhm_round = int(round(fwhm))
        if fwhm_round % 2 == 0:
            neighbor_box = max(3,fwhm_round+1)     # This should reduce the chance to accidently correct a bright planet
        else:
            neighbor_box = max(3,fwhm_round)       # This should reduce the chance to accidently correct a bright planet
        nneig = sum(np.arange(3,neighbor_box+2,2)) # This way, the sigma clipping will never get stuck due to a nan/bpix in a corner of the frame 
        
        #1/ Create a tuple-array with coordinates of a circle of radius 2*fwhm_round centered on the approximate coordinates of the star
        if protect_psf: circl_new = circle(cy=center[0], cx=center[1], radius=2.0*fwhm_round, shape=(n_y, n_x))
        else: circl_new = []
        
        #2/ Compute stddev of background
        # tested that 2.5 is the best to represent typical variation of background noise
        _, _, stddev = sigma_clipped_stats(obj_tmp, sigma=2.5) 

        #3/ Create a bad pixel map, by detecting them with find_outliers
        bpix_map = find_outliers(obj_tmp, sig_dist=sig, stddev = stddev, 
                                 neighbor_box=neighbor_box)
        bpix_map_cumul = np.zeros_like(bpix_map)
        bpix_map_cumul[:] = bpix_map[:] 
        nbpix_tot = np.sum(bpix_map)
        nbpix_tbc = nbpix_tot - np.sum(bpix_map[circl_new])
        bpix_map[circl_new] = 0
        nit = 0

        #4/ Loop over the bpix correction with sigma_filter, until there is 0 bpix left
        while nbpix_tbc > 0:
            nit = nit+1
            if verbose:
                print 'Iteration ',nit, ': ', nbpix_tot, ' bpixels in total, and ', nbpix_tbc, ' to be corrected.'
            obj_tmp = sigma_filter(obj_tmp, bpix_map, neighbor_box=neighbor_box, 
                                   min_neighbors=nneig, verbose=verbose)
            bpix_map = find_outliers(obj_tmp, sig_dist=sig, stddev = stddev, 
                                     neighbor_box=neighbor_box,DEBUG=True)
            bpix_map_cumul = bpix_map_cumul+bpix_map
            nbpix_tot = np.sum(bpix_map)
            nbpix_tbc = nbpix_tot - np.sum(bpix_map[circl_new])
            bpix_map[circl_new] = 0

        if verbose:  print 'All bad pixels are corrected.'
            
        return obj_tmp, bpix_map_cumul

    if ndims == 2:
        obj_tmp, bpix_map_cumul = bp_removal_2d(obj_tmp, center, fwhm, sig, protect_psf, verbose)

    if ndims == 3:
        n_z = obj_tmp.shape[0]
        bpix_map_cumul = np.zeros_like(obj_tmp)
        for i in range(n_z):
            if verbose: print '************Frame # ', i,' *************'
            obj_tmp[i], bpix_map_cumul[i] = bp_removal_2d(obj_tmp[i], center[i], fwhm[i], sig, protect_psf, verbose)
 
    return obj_tmp, bpix_map_cumul
    
    
def find_outliers(frame, sig_dist, stddev=None, neighbor_box=3, DEBUG=False):
    """ Provides a bad pixel (or outlier) map for a given frame.

    Parameters
    ----------
    frame : image_like 
        Input 2d image.
    sig_dist: float
        Threshold used to discriminate good from bad neighbours, in terms of 
        normalized distance to the median value of the set (see reject_outliers).
    neighbor_box : int, optional
        The side of the square window around each pixel where the sigma and 
        median are calculated for the bad pixel DETECTION and CORRECTION.
        Can only be 3 or 5.
        
    Returns
    -------
    obj_tmp : array_like
        Output cube with corrected bad pixels in each frame"""

    ndims = len(frame.shape)
    assert ndims == 2, "Object is not two dimensional.\n"

    nx = frame.shape[1]
    ny = frame.shape[0]
    bpix_map = np.zeros_like(frame)
    if stddev == None: stddev = np.std(frame)

    for xx in range(nx):
        for yy in range(ny):

            #0/ Determine the box of neighbouring pixels
            half_box = np.floor(neighbor_box/2.)
            hbox_b = min(half_box,yy)         # half size of the box at the bottom of the pixel
            hbox_t = min(half_box,ny-1-yy)    # half size of the box at the top of the pixel
            hbox_l = min(half_box,xx)         # half size of the box to the left of the pixel
            hbox_r = min(half_box,nx-1-xx)    # half size of the box to the right of the pixel
            # but in case we are at an edge, we want to extend the box by one row/column of pixels in the direction opposite to the edge:
            if half_box == 1:
                if yy == ny-1: hbox_b = hbox_b+1 
                elif yy == 0: hbox_t = hbox_t+1
                if xx == nx-1: hbox_l = hbox_l+1 
                elif xx == 0: hbox_r = hbox_r+1

            #1/ list neighbouring pixels, at least 9 pixels (including the pixel itself)
            neighbours = frame[yy-hbox_b:yy+hbox_t+1,xx-hbox_l:xx+hbox_r+1] 

            #2/ Reject the outliers from that list
            _,test_result = reject_outliers(neighbours, m=sig_dist,
                                            test_value=frame[yy,xx],
                                            stddev=stddev)

            #3/ Assign the value of the test to bpix_map
            bpix_map[yy,xx] = test_result

    return bpix_map


def reject_outliers(data, m = 5., test_value=0,stddev=None, DEBUG = False):
    """ FUNCTION TO REJECT OUTLIERS FROM A SET
    Instead of the classic standard deviation criterion (e.g. 5-sigma), the discriminant is determined as follow:
    - for each value in data, an absolute distance to the median of data is computed and put in a new array "d" (of same size as data)
    - the absolute distances "s" are then scaled by the median value of "d"
    - each "s" is then compared to "m" (parameter): if s < m, we have a good neighbour.

A specific value can also be tested as outlier, it would follow the same process as described above.


    Parameters:
    ----------------

data         : Input array with respect to which we determine if a test_value consists in an outlier.
m             : Criterion used to test if test_value or a neighbour pixel is an outlier (similar to the number of "sigma" in std_dev statistics)
test_value: Value to be tested as an outlier in the context of the input array (data)
stddev      : Global std dev of the non-PSF part of the considered frame. It is needed to avoid detecting outliers out of very close pixel values
    (i.e. if the 9 pixels happen to be very uniform in values at some location). If not provided, a stddev of the sample is used (not recommended).
DEBUG    : Bool. If True, the different variables involved will be printed out.

    Returns:
    -----------
good_neighbours, test_result

good_neighbours: sub-array of data containing the good neighbours, i.e. the pixels that are not outliers.
test_result           : 0 if test_value is not an outlier. 1 otherwise. 
    """

    if stddev == None: stddev = np.std(data)

    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    if DEBUG:
        print "data = ", data
        print "median(data)= ", np.median(data)
        print "d = ", d
        print "mdev = ", mdev
        print "stddev(box) = ", np.std(data)
        print "stddev(frame) = ", stddev
        print "max(d) = ", np.max(d)

    if np.max(d) > stddev:
        mdev = mdev if mdev>stddev else stddev
        s = d/mdev
        good_neighbours = data[s<m]
        if DEBUG: print "s =", s

        test = np.abs((test_value-np.median(data))/mdev)
        if DEBUG: print "test =", test
        if test < m:
            test_result = 0
        else:
            test_result = 1
    else:
        good_neighbours = data
        test_result = 0

    return good_neighbours, test_result
