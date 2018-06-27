#! /usr/bin/env python

"""
Module with functions for correcting bad pixels in cubes.
"""

from __future__ import division, print_function

__author__ = 'Carlos Alberto Gomez Gonzalez, V. Christiaens'
__all__ = ['frame_fix_badpix_isolated',
           'cube_fix_badpix_isolated',
           'cube_fix_badpix_annuli',
           'cube_fix_badpix_clump']

import numpy as np
from skimage.draw import circle, ellipse
from scipy.ndimage import median_filter
from astropy.stats import sigma_clipped_stats
from ..stats import sigma_filter
from ..var import dist, frame_center, pp_subplots
from ..stats import clip_array
from ..conf import timing, time_ini, Progressbar


def frame_fix_badpix_isolated(array, bpm_mask=None, sigma_clip=3, num_neig=5,
                              size=5, protect_mask=False, radius=30,
                              verbose=True, debug=False):
    """ Corrects the bad pixels, marked in the bad pixel mask. The bad pixel is
     replaced by the median of the adjacent pixels. This function is very fast
     but works only with isolated (sparse) pixels.

     Parameters
     ----------
     array : array_like
         Input 2d array.
     bpm_mask : array_like, optional
         Input bad pixel map. Zeros frame where the bad pixels have a value of
         1.
         If None is provided a bad pixel map will be created per frame using
         sigma clip statistics. In the case of a cube the bad pixels will be
         computed on the mean frame of the stack.
     sigma_clip : int, optional
         In case no bad pixel mask is provided all the pixels above and below
         sigma_clip*STDDEV will be marked as bad.
     num_neig : int, optional
         The side of the square window around each pixel where the sigma clipped
         statistics are calculated (STDDEV and MEDIAN). If the value is equal to
         0 then the statistics are computed in the whole frame.
     size : odd int, optional
         The size the box (size x size) of adjacent pixels for the median
         filter.
     protect_mask : bool, optional
         If True a circular aperture at the center of the frames will be
         protected from any operation. With this we protect the star and
         vicinity.
     radius : int, optional
         Radius of the circular aperture (at the center of the frames) for the
         protection mask.
     verbose : bool, optional
         If True additional information will be printed.
     debug : bool, optional
         If debug is True, the bpm_mask and the input array are plotted. If the
         input array is a cube, a long output is to be expected. Better check
         the results with single images.

     Return
     ------
     frame : array_like
         Frame with bad pixels corrected.
     """
    if array.ndim != 2:
        raise TypeError('Array is not a 2d array or single frame')
    if size % 2 == 0:
        raise TypeError('Size of the median blur kernel must be an odd integer')

    if bpm_mask is not None:
        bpm_mask = bpm_mask.astype('bool')

    if verbose:  start = time_ini()

    if num_neig > 0:
        neigh = True
    else:
        neigh = False

    frame = array.copy()
    cy, cx = frame_center(frame)
    if bpm_mask is None:
        ind = clip_array(frame, sigma_clip, sigma_clip, neighbor=neigh,
                         num_neighbor=num_neig, mad=True)
        bpm_mask = np.zeros_like(frame)
        bpm_mask[ind] = 1
        if protect_mask:
            cir = circle(cy, cx, radius)
            bpm_mask[cir] = 0
        bpm_mask = bpm_mask.astype('bool')
        if debug:
            pp_subplots(frame, bpm_mask, title='Frame / Bad pixel mask')

    smoothed = median_filter(frame, size, mode='mirror')
    frame[np.where(bpm_mask)] = smoothed[np.where(bpm_mask)]
    array_out = frame

    if verbose:
        print("\nDone replacing bad pixels using the median of the neighbors")
        timing(start)
    return array_out


def cube_fix_badpix_isolated(array, bpm_mask=None, sigma_clip=3, num_neig=5, 
                             size=5, protect_mask=False, radius=30,
                             verbose=True, debug=False):
    """ Corrects the bad pixels, marked in the bad pixel mask. The bad pixel is 
    replaced by the median of the adjacent pixels. This function is very fast
    but works only with isolated (sparse) pixels. 
     
    Parameters
    ----------
    array : array_like
        Input 3d array.
    bpm_mask : array_like, optional
        Input bad pixel map. Zeros frame where the bad pixels have a value of 1.
        If None is provided a bad pixel map will be created per frame using 
        sigma clip statistics. In the case of a cube the bad pixels will be 
        computed on the mean frame of the stack.
    sigma_clip : int, optional
        In case no bad pixel mask is provided all the pixels above and below
        sigma_clip*STDDEV will be marked as bad. 
    num_neig : int, optional
        The side of the square window around each pixel where the sigma clipped
        statistics are calculated (STDDEV and MEDIAN). If the value is equal to
        0 then the statistics are computed in the whole frame.
    size : odd int, optional
        The size the box (size x size) of adjacent pixels for the median filter. 
    protect_mask : bool, optional
        If True a circular aperture at the center of the frames will be 
        protected from any operation. With this we protect the star and its
        vicinity.
    radius : int, optional 
        Radius of the circular aperture (at the center of the frames) for the 
        protection mask.
    verbose : bool, optional
        If True additional information will be printed. 
    debug : bool, optional
        If debug is True, the bpm_mask and the input array are plotted. If the
        input array is a cube, a long output is to be expected. Better check the
        results with single images.
    
    Return
    ------
    array_out : array_like
        Cube with bad pixels corrected.
    """
    if array.ndim != 3:
        raise TypeError('Array is not a 3d array or cube')
    if size % 2 == 0:
        raise TypeError('Size of the median blur kernel must be an odd integer')
    
    if bpm_mask is not None:
        bpm_mask = bpm_mask.astype('bool')
    
    if verbose:  start = time_ini()
    
    if num_neig > 0:
        neigh = True
    else:
        neigh = False
    
    cy, cx = frame_center(array[0])
    cube_out = array.copy()
    n_frames = array.shape[0]

    if bpm_mask is None:
        ind = clip_array(np.mean(array, axis=0), sigma_clip, sigma_clip,
                         neighbor=neigh, num_neighbor=num_neig, mad=True)
        bpm_mask = np.zeros_like(array[0])
        bpm_mask[ind] = 1
        if protect_mask:
            cir = circle(cy, cx, radius)
            bpm_mask[cir] = 0
        bpm_mask = bpm_mask.astype('bool')

    if debug:
        pp_subplots(bpm_mask, title='Bad pixel mask')

    for i in Progressbar(range(n_frames), desc="frames"):
        frame = cube_out[i]
        smoothed = median_filter(frame, size, mode='mirror')
        frame[np.where(bpm_mask)] = smoothed[np.where(bpm_mask)]
    array_out = cube_out
    
    if verbose: 
        print("/nDone replacing bad pixels using the median of the neighbors")
        timing(start)
    return array_out


def cube_fix_badpix_annuli(array, cy, cx, fwhm, sig=5., protect_psf=True, 
                           verbose=True, half_res_y=False, min_thr=None, 
                           mid_thr=None, full_output=False):
    """
    Function to correct the bad pixels annulus per annulus (centered on the 
    provided location of the star), in an input frame or cube.
    This function is MUCH FASTER than bp_clump_removal (about 20 times faster);
    hence to be prefered in all cases where there is only one bright source.
    The bad pixel values are replaced by: ann_median + ann_stddev*random_gauss;
    where ann_median is the median of the annulus, ann_stddev is the standard 
    deviation in the annulus, and random_gauss is a random factor picked from a 
    gaussian distribution centered on 0 and with variance 1.

    Parameters
    ----------
    array : 3D or 2D array 
        Input 3d cube or 2d image.
    cy, cx : float or 1D array
        Vector with approximate y and x coordinates of the star for each channel
        (cube_like), or single 2-elements vector (frame_like)
    fwhm: float or 1D array
        Vector containing the full width half maximum of the PSF in pixels, for 
        each channel (cube_like); or single value (frame_like)
    sig: Float scalar, optional
        Number of stddev above or below the median of the pixels in the same 
        annulus, to consider a pixel as bad.
    protect_psf: bool, {True, False}, optional
        Whether to protect a circular region centered on the star (1.8*fwhm 
        radius) from any bpix corr. If False, there is a risk of modifying a 
        centroid peak value if it is too "peaky"; but if True real bad pixels 
        within the core are not corrected.
    verbose: bool, {False, True}, optional
        Whether to print out the number of bad pixels in each frame. 
    half_res_y: bool, {True,False}, optional
        Whether the input data have only half the angular resolution vertically 
        compared to horizontally (e.g. SINFONI data).
        The algorithm will then correct the bad pixels every other row.
    min_thr: {None,float}, optional
        Any pixel whose value is lower than this threshold (expressed in stddev)
        will be automatically considered bad and hence sigma_filtered. If None, 
        it is not used.
    mid_thr: {None, float}, optional
        Pixels whose value is lower than this threshold (expressed in stddev) 
        will have its neighbours checked; if there is at max. 1 neighbour pixel
        whose value is lower than (5+mid_thr)*stddev, then the pixel is 
        considered bad (as it means it is a cold pixel in the middle of 
        significant signal). If None, it is not used.
    full_output: bool, {False,True}, optional
        Whether to return as well the cube of bad pixel maps and the cube of 
        defined annuli.

    Returns:
    --------
    obj_tmp: 2d or 3d array; the bad pixel corrected frame/cube.
    If full_output is set to True, it returns as well:
    bpix_map: 2d or 3d array; the bad pixel map or the cube of bpix maps
    ann_frame_cumul: 2 or 3d array; the cube of defined annuli
    """

    obj_tmp = array.copy()
    ndims = obj_tmp.ndim
    assert ndims == 2 or ndims == 3, "Object is not two or three dimensional.\n"

    #thresholds
    if min_thr is None:
        min_thr = np.amin(obj_tmp)-1
    if mid_thr is None:
        mid_thr = np.amin(obj_tmp)-1

    def bp_removal_2d(obj_tmp, cy, cx, fwhm, sig, protect_psf, verbose):

        n_x = obj_tmp.shape[1]
        n_y = obj_tmp.shape[0]

        # Squash the frame if twice less resolved vertically than horizontally
        if half_res_y:
            if n_y % 2 != 0:
                msg = 'The input frames do not have of an even number of rows. '
                msg2 = 'Hence, you should not use option half_res_y = True'
                raise ValueError(msg+msg2)
            n_y = int(n_y/2)
            cy = int(cy/2)
            frame = obj_tmp.copy()
            obj_tmp = np.zeros([n_y,n_x])
            for yy in range(n_y):
                obj_tmp[yy] = frame[2*yy]

        #1/ Stddev of background
        _, _, stddev = sigma_clipped_stats(obj_tmp, sigma=2.5)

        #2/ Define each annulus, its median and stddev
        
        ymax = max(cy, n_y-cy)
        xmax = max(cx, n_x-cx)
        if half_res_y:
            ymax *= 2
        rmax = np.sqrt(ymax**2+xmax**2)
        # the annuli definition is optimized for Airy rings
        ann_width = max(1.5, 0.61*fwhm) 
        nrad = int(rmax/ann_width)+1
        d_bord_max = max(n_y-cy, cy, n_x-cx, cx)
        if half_res_y:
            d_bord_max = max(2*(n_y-cy), 2*cy, n_x-cx, cx)

        big_ell_frame = np.zeros_like(obj_tmp)
        sma_ell_frame = np.zeros_like(obj_tmp)
        ann_frame_cumul = np.zeros_like(obj_tmp)
        n_neig = np.zeros(nrad)
        med_neig = np.zeros(nrad)
        std_neig = np.zeros(nrad)
        neighbours = np.zeros([nrad,n_y*n_x])

        for rr in range(nrad):
            if rr > int(d_bord_max/ann_width):
                # just to merge farthest annuli with very few elements 
                rr_big = nrad  
                rr_sma = int(d_bord_max/ann_width)
            else: 
                rr_big = rr
                rr_sma= rr
            if half_res_y:
                big_ell_idx = ellipse(cy=cy, cx=cx, 
                                      yradius=((rr_big+1)*ann_width)/2, 
                                      xradius=(rr_big+1)*ann_width, 
                                      shape=(n_y,n_x))
                if rr != 0:
                    small_ell_idx = ellipse(cy=cy, cx=cx, 
                                            yradius=(rr_sma*ann_width)/2, 
                                            xradius=rr_sma*ann_width, 
                                            shape=(n_y,n_x))
            else:
                big_ell_idx = circle(cy, cx, radius=(rr_big+1)*ann_width,
                                     shape=(n_y,n_x))
                if rr != 0:
                    small_ell_idx = circle(cy, cx, radius=rr_sma*ann_width, 
                                           shape=(n_y,n_x))
            big_ell_frame[big_ell_idx] = 1
            if rr!=0: sma_ell_frame[small_ell_idx] = 1
            ann_frame = big_ell_frame - sma_ell_frame
            n_neig[rr] = ann_frame[np.where(ann_frame)].shape[0]
            neighbours[rr,:n_neig[rr]] = obj_tmp[np.where(ann_frame)]
            ann_frame_cumul[np.where(ann_frame)] = rr

            # We delete iteratively max and min outliers in each annulus, 
            # so that the annuli median and stddev are not corrupted by bpixs
            neigh = neighbours[rr,:n_neig[rr]]
            n_removal = 0
            n_pix_init = neigh.shape[0]
            while neigh.shape[0] > 10 and n_removal < n_pix_init/10:
                min_neigh = np.amin(neigh)
                if reject_outliers(neigh, min_neigh, m=5, stddev=stddev,
                                   min_thr=min_thr*stddev,
                                   mid_thr=mid_thr*stddev):
                    min_idx = np.argmin(neigh)
                    neigh = np.delete(neigh,min_idx)
                    n_removal += 1
                else:
                    max_neigh = np.amax(neigh)
                    if reject_outliers(neigh, max_neigh, m=5, stddev=stddev,
                                       min_thr=min_thr*stddev, 
                                       mid_thr=mid_thr*stddev):
                        max_idx = np.argmax(neigh)
                        neigh = np.delete(neigh,max_idx)
                        n_removal += 1
                    else: break
            n_neig[rr] = neigh.shape[0]
            neighbours[rr,:n_neig[rr]] = neigh
            neighbours[rr,n_neig[rr]:] = 0
            med_neig[rr] = np.median(neigh)
            std_neig[rr] = np.std(neigh)
        
        #3/ Create a tuple-array with coordinates of a circle of radius 1.8*fwhm
        # centered on the provided coordinates of the star
        if protect_psf:
            if half_res_y: 
                circl_new = ellipse(cy, cx, yradius=0.9*fwhm, 
                                    xradius=1.8*fwhm, shape=(n_y,n_x))
            else: 
                circl_new = circle(cy, cx, radius=1.8*fwhm, 
                                   shape=(n_y, n_x))
        else: circl_new = []

        #4/ Loop on all pixels to check bpix
        bpix_map = np.zeros_like(obj_tmp)
        obj_tmp_corr = obj_tmp.copy()
        
        for yy in range(n_y):
            for xx in range(n_x):
                if half_res_y:
                    rad = np.sqrt((2*(cy-yy))**2+(cx-xx)**2)
                else:
                    rad = dist(cy, cx, yy, xx)
                rr = int(rad/ann_width)
                neigh = neighbours[rr,:n_neig[rr]]
                dev = max(stddev,min(std_neig[rr],med_neig[rr]))

                # check min_thr
                if obj_tmp[yy,xx] < min_thr*stddev:
                    bpix_map[yy,xx] = 1
                    # Gaussian noise
                    #obj_tmp_corr[yy,xx] = med_neig[rr] +dev*np.random.randn()
                    # Poisson noise
                    rand_fac= 2*(np.random.rand()-0.5)
                    obj_tmp_corr[yy,xx] = med_neig[rr] + \
                                          np.sqrt(np.abs(med_neig[rr]))*rand_fac

                # check median +- sig*stddev
                elif (obj_tmp[yy,xx] < med_neig[rr]-sig*dev or 
                      obj_tmp[yy,xx] > med_neig[rr]+sig*dev):
                    bpix_map[yy,xx] = 1  
                    # Gaussian noise
                    #obj_tmp_corr[yy,xx] = med_neig[rr] +dev*np.random.randn()
                    # Poisson noise
                    rand_fac= 2*(np.random.rand()-0.5)
                    obj_tmp_corr[yy,xx] = med_neig[rr] + \
                                          np.sqrt(np.abs(med_neig[rr]))*rand_fac

                # check mid_thr and neighbours
                else:
                    min_el = max(2, 0.05*n_neig[rr])
                    if (obj_tmp[yy,xx] < mid_thr*stddev and 
                        neigh[neigh<(mid_thr+5)*stddev].shape[0] < min_el):
                            bpix_map[yy,xx] = 1
                            # Gaussian noise
                            #obj_tmp_corr[yy,xx] = med_neig[rr] + \
                            #                      dev*np.random.randn()
                            # Poisson noise
                            rand_fac = 2*(np.random.rand()-0.5)
                            obj_tmp_corr[yy,xx] = med_neig[rr] + \
                                          np.sqrt(np.abs(med_neig[rr]))*rand_fac

        #5/ Count bpix and uncorrect if within the circle
        nbpix_tot = np.sum(bpix_map)
        nbpix_tbc = nbpix_tot - np.sum(bpix_map[circl_new])
        bpix_map[circl_new] = 0
        obj_tmp_corr[circl_new] = obj_tmp[circl_new]
        if verbose:
            print(nbpix_tot, ' bpix in total, and ', nbpix_tbc, ' corrected.')

        # Unsquash all the frames
        if half_res_y:
            frame = obj_tmp_corr.copy()
            frame_bpix = bpix_map.copy()
            n_y = 2*n_y
            obj_tmp_corr = np.zeros([n_y,n_x])
            bpix_map = np.zeros([n_y,n_x])
            ann_frame = ann_frame_cumul.copy()
            ann_frame_cumul = np.zeros([n_y,n_x])
            for yy in range(n_y):
                obj_tmp_corr[yy] = frame[int(yy/2)]
                bpix_map[yy] = frame_bpix[int(yy/2)]
                ann_frame_cumul[yy] = ann_frame[int(yy/2)]

        return obj_tmp_corr, bpix_map, ann_frame_cumul

    if ndims == 2:
        obj_tmp, bpix_map, ann_frame_cumul = bp_removal_2d(obj_tmp, cy, cx, 
                                                           fwhm, sig, 
                                                           protect_psf, verbose)
    if ndims == 3:
        n_z = obj_tmp.shape[0]
        bpix_map = np.zeros_like(obj_tmp)
        ann_frame_cumul = np.zeros_like(obj_tmp)
        if cy.shape[0] != n_z or cx.shape[0] != n_z: 
            raise ValueError('Please provide cy and cx as 1d-arr of size n_z')
        for i in range(n_z):
            if verbose:
                print('************Frame # ', i,' *************')
            res_i = bp_removal_2d(obj_tmp[i], cy[i], cx[i], fwhm[i], sig,
                                  protect_psf, verbose)
            obj_tmp[i], bpix_map[i], ann_frame_cumul[i] = res_i
 
    if full_output:
        return obj_tmp, bpix_map, ann_frame_cumul
    else:
        return obj_tmp


def cube_fix_badpix_clump(array, cy, cx, fwhm, sig=4., protect_psf=True, 
                          verbose=True, half_res_y=False, min_thr=None, 
                          mid_thr=None, max_nit=15, full_output=False):
    """
    Function to correct the bad pixels in obj_tmp. Slow alternative to 
    bp_annuli_removal to correct for bad pixels in either a frame or a spectral
    cube. It should be used instead of bp_annuli_removal only if the observed 
    field is not composed of a single bright object (producing a circularly 
    symmetric PSF). As it is based on an iterative process, it is able to 
    correct clumps of bad pixels. The bad pixel values are sigma filtered 
    (replaced by the median of neighbouring pixel values).


    Parameters
    ----------
    array : 3D or 2D array 
        Input 3d cube or 2d image.
    cy,cx : float or 1D array
        Vector with approximate y and x coordinates of the star for each channel
        (cube_like), or single 2-elements vector (frame_like).
    fwhm: float or 1D array
        Vector containing the full width half maximum of the PSF in pixels, for
        each channel (cube_like); or single value (frame_like)
    sig: float, optional
        Value representing the number of "sigmas" above or below the "median" of
        the neighbouring pixel, to consider a pixel as bad. See details on 
        parameter "m" of function reject_outlier.
    protect_psf: bool, {True, False}, optional
        True if you want to protect a circular region centered on the star 
        (1.8*fwhm radius) from any bpix corr. If False, there is a risk to 
        modify a psf peak value; but if True, real bpix within the core are 
        not corrected.
    verbose: bool, {False,True}, optional
        Whether to print the number of bad pixels and number of iterations 
        required for each frame.
    half_res_y: bool, {True,False}, optional
        Whether the input data has only half the angular resolution vertically 
        compared to horizontally (e.g. the case of SINFONI data); in other words
        there are always 2 rows of pixels with exactly the same values.
        The algorithm will just consider every other row (hence making it
        twice faster), then apply the bad pixel correction on all rows.
    min_thr: {None,float}, optional
        Any pixel whose value is lower than this threshold (expressed in adu)
        will be automatically considered bad and hence sigma_filtered. If None, 
        it is not used (not recommended).
    mid_thr: {None, float}, optional
        Pixels whose value is lower than this threshold (expressed in adu) will
        have its neighbours checked; if there is at max 1 neighbour pixel whose
        value is lower than mid_thr+(5*stddev), then the pixel is considered bad
        (because it means it is a cold pixel in the middle of significant 
        signal). If None, it is not used (not recommended).
    max_nit: float, optional
        Maximum number of iterations on a frame to correct bpix. Typically, it 
        should be set to less than ny/2 or nx/2. This is a mean of precaution in
        case the algorithm gets stuck with 2 neighbouring pixels considered bpix
        alternately on two consecutively iterations hence leading to an infinite
        loop (very very rare case).
    full_output: bool, {False,True}, optional
        Whether to return as well the cube of bad pixel maps and the cube of 
        defined annuli.

    Returns:
    --------
    obj_tmp: 2d or 3d array; the bad pixel corrected frame/cube.
    If full_output is set to True, it returns as well:
    bpix_map: 2d or 3d array; the bad pixel map or the cube of bpix maps
    """

    obj_tmp = array.copy()
    ndims = obj_tmp.ndim
    assert ndims == 2 or ndims == 3, "Object is not two or three dimensional.\n"

    def bp_removal_2d(obj_tmp, cy, cx, fwhm, sig, protect_psf, verbose):    
        n_x = obj_tmp.shape[1]
        n_y = obj_tmp.shape[0]

        if half_res_y:
            if n_y%2 != 0: 
                msg = 'The input frames do not have of an even number of rows. '
                msg2 = 'Hence, you should not use option half_res_y = True'
                raise ValueError(msg+msg2)
            n_y = int(n_y/2)
            frame = obj_tmp.copy()
            obj_tmp = np.zeros([n_y,n_x])
            for yy in range(n_y):
                obj_tmp[yy] = frame[2*yy]
            
        fwhm_round = int(round(fwhm))
        # This should reduce the chance to accidently correct a bright planet:
        if fwhm_round % 2 == 0:
            neighbor_box = max(3, fwhm_round+1) 
        else:
            neighbor_box = max(3, fwhm_round)
        nneig = sum(np.arange(3, neighbor_box+2, 2))

        
        #1/ Create a tuple-array with coordinates of a circle of radius 1.8*fwhm
        # centered on the approximate coordinates of the star
        if protect_psf:
            if half_res_y: 
                circl_new = ellipse(int(cy/2), cx, yradius=0.9*fwhm, 
                                    xradius=1.8*fwhm, shape=(n_y, n_x))
            else: circl_new = circle(cy, cx, radius=1.8*fwhm, 
                                     shape=(n_y, n_x))
        else: circl_new = []
        
        #2/ Compute stddev of background
        # empirically 2.5 is best to find stddev of background noise
        _, _, stddev = sigma_clipped_stats(obj_tmp, sigma=2.5) 

        #3/ Create a bad pixel map, by detecting them with find_outliers
        bpix_map = find_outliers(obj_tmp, sig_dist=sig, stddev=stddev, 
                                 neighbor_box=neighbor_box,min_thr=min_thr,
                                 mid_thr=mid_thr)
        nbpix_tot = np.sum(bpix_map)
        nbpix_tbc = nbpix_tot - np.sum(bpix_map[circl_new])
        bpix_map[circl_new] = 0
        bpix_map_cumul = np.zeros_like(bpix_map)
        bpix_map_cumul[:] = bpix_map[:]
        nit = 0

        #4/ Loop over the bpix correction with sigma_filter, until 0 bpix left
        while nbpix_tbc > 0 and nit < max_nit:
            nit = nit+1
            if verbose:
                print("Iteration {}: {} bpix in total, {} to be "
                      "corrected".format(nit, nbpix_tot, nbpix_tbc))
            obj_tmp = sigma_filter(obj_tmp, bpix_map, neighbor_box=neighbor_box,
                                   min_neighbors=nneig, verbose=verbose)
            bpix_map = find_outliers(obj_tmp, sig_dist=sig, in_bpix=bpix_map,
                                     stddev=stddev, neighbor_box=neighbor_box,
                                     min_thr=min_thr,mid_thr=mid_thr)
            nbpix_tot = np.sum(bpix_map)
            nbpix_tbc = nbpix_tot - np.sum(bpix_map[circl_new])
            bpix_map[circl_new] = 0
            bpix_map_cumul = bpix_map_cumul+bpix_map

        if verbose:
            print('All bad pixels are corrected.')
            
        if half_res_y:
            frame = obj_tmp.copy()
            frame_bpix = bpix_map_cumul.copy()
            n_y = 2*n_y
            obj_tmp = np.zeros([n_y,n_x])
            bpix_map_cumul = np.zeros([n_y,n_x])
            for yy in range(n_y):
                obj_tmp[yy] = frame[int(yy/2)]
                bpix_map_cumul[yy] = frame_bpix[int(yy/2)]

        return obj_tmp, bpix_map_cumul

    if ndims == 2:
        obj_tmp, bpix_map_cumul = bp_removal_2d(obj_tmp, cy, cx, fwhm, sig, 
                                                protect_psf, verbose)

    if ndims == 3:
        n_z = obj_tmp.shape[0]
        bpix_map_cumul = np.zeros_like(obj_tmp)
        for i in range(n_z):
            if verbose: print('************Frame # ', i,' *************')
            obj_tmp[i], bpix_map_cumul[i] = bp_removal_2d(obj_tmp[i], cy[i], 
                                                          cx[i], fwhm[i], sig, 
                                                          protect_psf, verbose)
    if full_output:
        return obj_tmp, bpix_map_cumul
    else:
        return obj_tmp
    
    
def find_outliers(frame, sig_dist, in_bpix=None, stddev=None, neighbor_box=3,
                  min_thr=None, mid_thr=None):
    """ Provides a bad pixel (or outlier) map for a given frame.

    Parameters
    ----------
    frame: 2d array 
        Input 2d image.
    sig_dist: float
        Threshold used to discriminate good from bad neighbours, in terms of 
        normalized distance to the median value of the set (see reject_outliers)
    in_bpix: 2d array, optional
        Input bpix map (typically known from the previous iteration), to only 
        look for bpix around those locations.
    neighbor_box: int, optional
        The side of the square window around each pixel where the sigma and 
        median are calculated for the bad pixel DETECTION and CORRECTION.
    min_thr: {None,float}, optional
        Any pixel whose value is lower than this threshold (expressed in adu)
        will be automatically considered bad and hence sigma_filtered. If None,
        it is not used.
    mid_thr: {None, float}, optional
        Pixels whose value is lower than this threshold (expressed in adu) will
        have its neighbours checked; if there is at max. 1 neighbour pixel whose
        value is lower than mid_thr+(5*stddev), then the pixel is considered bad
        (because it means it is a cold pixel in the middle of significant 
        signal). If None, it is not used.

    Returns
    -------
    bpix_map : array_like
        Output cube with frames indicating the location of bad pixels"""

    ndims = len(frame.shape)
    assert ndims == 2, "Object is not two dimensional.\n"

    nx = frame.shape[1]
    ny = frame.shape[0]
    bpix_map = np.zeros_like(frame)
    if stddev is None: stddev = np.std(frame)
    half_box = int(neighbor_box/2)
    
    if in_bpix is None:
        for xx in range(nx):
            for yy in range(ny):
                #0/ Determine the box of neighbouring pixels
                # half size of the box at the bottom of the pixel
                hbox_b = min(half_box, yy)        
                # half size of the box at the top of the pixel
                hbox_t = min(half_box, ny-1-yy)   
                # half size of the box to the left of the pixel
                hbox_l = min(half_box, xx)
                # half size of the box to the right of the pixel  
                hbox_r = min(half_box, nx-1-xx)    
                # but in case we are at an edge, we want to extend the box by 
                # one row/column of px in the direction opposite to the edge:
                if yy > ny-1-half_box:
                    hbox_b = hbox_b + (yy-(ny-1-half_box))
                elif yy < half_box:
                    hbox_t = hbox_t+(half_box-yy)
                if xx > nx-1-half_box:
                    hbox_l = hbox_l + (xx-(nx-1-half_box))
                elif xx < half_box:
                    hbox_r = hbox_r+(half_box-xx)

                #1/ list neighbouring pixels, >8 (NOT including pixel itself)
                neighbours = frame[yy-hbox_b:yy+hbox_t+1,
                                   xx-hbox_l:xx+hbox_r+1]
                idx_px = ([[hbox_b],[hbox_l]])
                flat_idx = np.ravel_multi_index(idx_px,(hbox_t+hbox_b+1,
                                                        hbox_r+hbox_l+1))
                neighbours = np.delete(neighbours,flat_idx)

                #2/ Det if central pixel is outlier
                test_result = reject_outliers(neighbours, frame[yy,xx], 
                                              m=sig_dist, stddev=stddev, 
                                              min_thr=min_thr, mid_thr=mid_thr)

                #3/ Assign the value of the test to bpix_map
                bpix_map[yy,xx] = test_result

    else:
        nb = int(np.sum(in_bpix))  # number of bad pixels at previous iteration
        wb = np.where(in_bpix)     # pixels to check
        bool_bpix= np.zeros_like(in_bpix)
        for n in range(nb):
            for yy in [max(0,wb[0][n]-1),wb[0][n],min(ny-1,wb[0][n]+1)]:
                for xx in [max(0,wb[1][n]-1),wb[1][n],min(ny-1,wb[1][n]+1)]:
                    bool_bpix[yy,xx] = 1
        nb = int(np.sum(bool_bpix))# true number of px to check  (including 
                                   # neighbours of bpix from previous iteration)
        wb = np.where(bool_bpix)   # true px to check
        for n in range(nb):
            #0/ Determine the box of neighbouring pixels
            # half size of the box at the bottom of the pixel
            hbox_b = min(half_box,wb[0][n])
            # half size of the box at the top of the pixel     
            hbox_t = min(half_box,ny-1-wb[0][n])
            # half size of the box to the left of the pixel
            hbox_l = min(half_box,wb[1][n])
            # half size of the box to the right of the pixel
            hbox_r = min(half_box,nx-1-wb[1][n])
            # but in case we are at an edge, we want to extend the box by one 
            # row/column of pixels in the direction opposite to the edge:
            if wb[0][n] > ny-1-half_box:
                hbox_b = hbox_b + (wb[0][n]-(ny-1-half_box))
            elif wb[0][n] < half_box:
                hbox_t = hbox_t+(half_box-wb[0][n])
            if wb[1][n] > nx-1-half_box:
                hbox_l = hbox_l + (wb[1][n]-(nx-1-half_box))
            elif wb[1][n] < half_box:
                hbox_r = hbox_r+(half_box-wb[1][n])

            #1/ list neighbouring pixels, > 8, not including the pixel itself
            neighbours = frame[wb[0][n]-hbox_b:wb[0][n]+hbox_t+1,
                               wb[1][n]-hbox_l:wb[1][n]+hbox_r+1]
            c_idx_px = ([[hbox_b],[hbox_l]])
            flat_c_idx = np.ravel_multi_index(c_idx_px,(hbox_t+hbox_b+1,
                                                        hbox_r+hbox_l+1))
            neighbours = np.delete(neighbours,flat_c_idx)

            #2/ test if bpix
            test_result = reject_outliers(neighbours, frame[wb[0][n],wb[1][n]],
                                          m=sig_dist, stddev=stddev, 
                                          min_thr=min_thr, mid_thr=mid_thr)

            #3/ Assign the value of the test to bpix_map
            bpix_map[wb[0][n],wb[1][n]] = test_result


    return bpix_map


def reject_outliers(data, test_value, m=5., stddev=None, debug=False,
                    min_thr=None, mid_thr=None):
    """ FUNCTION TO REJECT OUTLIERS FROM A SET
    Instead of the classic standard deviation criterion (e.g. 5-sigma), the 
    discriminant is determined as follow:
    - for each value in data, an absolute distance to the median of data is
    computed and put in a new array "d" (of same size as data)
    - scaling each element of "d" by the median value of "d" gives the absolute
    distances "s" of each element
    - each "s" is then compared to "m" (parameter): if s < m, we have a good 
    neighbour, otherwise we have an outlier. A specific value test_value is 
    tested as outlier.

    Parameters:
    -----------
    data: array_like
        Input array with respect to which either a test_value or the central a 
        value of data is determined to be an outlier or not
    test_value: float
        Value to be tested as an outlier in the context of the input array data
    m: float, optional
        Criterion used to test if test_value is or pixels of data are outlier(s)
        (similar to the number of "sigma" in std_dev statistics)
    stddev: float, optional (but strongly recommended)
        Global std dev of the non-PSF part of the considered frame. It is needed
        as a reference to know the typical variation of the noise, and hence 
        avoid detecting outliers out of very close pixel values. If the 9 pixels
        of data happen to be very uniform in values at some location, the 
        departure in value of only one pixel could make it appear as a bad 
        pixel. If stddev is not provided, the stddev of data is used (not 
        recommended).
    debug: Bool, {False,True}.
        If True, the different variables involved will be printed out.
    min_thr: {None,float}, optional
        Any pixel whose value is lower than this threshold (expressed in adu)
        will be automatically considered bad and hence sigma_filtered. If None,
        it is not used.
    mid_thr: {None, float}, optional
        Pixels whose value is lower than this threshold (expressed in adu) will
        have its neighbours checked; if there is at max. 1 neighbour pixel whose
        value is lower than mid_thr+(5*stddev), then the pixel is considered bad
        (because it means it is a cold pixel in the middle of significant 
        signal). If None, it is not used.

    Returns:
    --------
    test_result: 0 or 1
        0 if test_value is not an outlier. 1 otherwise. 
    """

    if stddev is None:
        stddev = np.std(data)
    if min_thr is None:
        min_thr = min(np.amin(data), test_value)-1
    if mid_thr is None:
        mid_thr = min(np.amin(data), test_value)-1

    med = np.median(data)
    d = np.abs(data - med)
    mdev = np.median(d)
    if debug:
        print("data = ", data)
        print("median(data)= ", np.median(data))
        print("d = ", d)
        print("mdev = ", mdev)
        print("stddev(box) = ", np.std(data))
        print("stddev(frame) = ", stddev)
        print("max(d) = ", np.max(d))

    n_el = max(2, 0.05*data.shape[0])
    if test_value < min_thr or (test_value < mid_thr and 
                                data[data<(mid_thr+5*stddev)].shape[0] < n_el):
        test_result = 1

    elif max(np.max(d),np.abs(test_value-med)) > stddev:
        mdev = mdev if mdev>stddev else stddev
        s = d/mdev
        if debug:
            print("s =", s)
        test = np.abs((test_value-np.median(data))/mdev)
        if debug:
            print("test =", test)
        else:
            if test < m:
                test_result = 0
            else:
                test_result = 1
    else:
        test_result = 0

    return test_result
