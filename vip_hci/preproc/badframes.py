#! /usr/bin/env python

"""
Module with functions for outlier frame detection.
"""

__author__ = 'Carlos Alberto Gomez Gonzalez'
__all__ = ['cube_detect_badfr_pxstats',
           'cube_detect_badfr_ellipticity',
           'cube_detect_badfr_correlation']

import numpy as np
import pandas as pn
from matplotlib import pyplot as plt
from photutils import DAOStarFinder
from astropy.stats import sigma_clip
from ..var import get_annulus_segments
from ..config import time_ini, timing, check_array
from ..config.utils_conf import vip_figsize
from ..stats import cube_basic_stats, cube_distance


def cube_detect_badfr_pxstats(array, mode='annulus', in_radius=10, width=10, 
                              top_sigma=1.0, low_sigma=1.0, window=None, 
                              plot=True, verbose=True):
    """ Returns the list of bad frames from a cube using the px statistics in 
    a centered annulus or circular aperture. Frames that are more than a few 
    standard deviations discrepant are rejected. Should be applied on a 
    recentered cube.
    
    Parameters
    ----------
    array : numpy ndarray 
        Input 3d array, cube.
    mode : {'annulus', 'circle'}, string optional
        Whether to take the statistics from a circle or an annulus.
    in_radius : int optional
        If mode is 'annulus' then 'in_radius' is the inner radius of the annular 
        region. If mode is 'circle' then 'in_radius' is the radius of the 
        aperture.
    width : int optional
        Size of the annulus. Ignored if mode is 'circle'.
    top_sigma : int, optional
        Top boundary for rejection.
    low_sigma : int, optional
        Lower boundary for rejection.
    window : int, optional
        Window for smoothing the mean and getting the rejection statistic. If
        None, it is defined as ``n_frames//3``.
    plot : bool, optional
        If true it plots the mean fluctuation as a function of the frames and 
        the boundaries.
    verbose : bool, optional
        Whether to print to stdout or not.
    
    Returns
    -------
    good_index_list : numpy ndarray
        1d array of good indices.
    bad_index_list : numpy ndarray
        1d array of bad frames indices.
    
    """
    check_array(array, 3, msg='array')

    if in_radius+width > array[0].shape[0]/2:
        msgve = 'Inner radius and annulus size are too big (out of boundaries)'
        raise ValueError(msgve)
    
    if verbose:
        start_time = time_ini()
    
    n = array.shape[0]
    
    mean_values = cube_basic_stats(array, mode, radius=in_radius,
                                   inner_radius=in_radius, size=width)

    if window is None:
        window = n//3
    mean_smooth = pn.Series(mean_values).rolling(window, center=True).mean()
    mean_smooth = mean_smooth.fillna(method='backfill')
    mean_smooth = mean_smooth.fillna(method='ffill')
    sigma = np.std(mean_values)
    bad_index_list = []
    good_index_list = []
    top_boundary = np.empty([n])
    bot_boundary = np.empty([n])
    for i in range(n):
        if mode == 'annulus':
            i_mean_value = get_annulus_segments(array[i], width=width,
                                                inner_radius=in_radius,
                                                mode="val")[0].mean()
        elif mode == 'circle':
            i_mean_value = mean_values[i]
        top_boundary[i] = mean_smooth[i] + top_sigma*sigma
        bot_boundary[i] = mean_smooth[i] - low_sigma*sigma
        if i_mean_value > top_boundary[i] or i_mean_value < bot_boundary[i]:
            bad_index_list.append(i)
        else:
            good_index_list.append(i)                       

    if verbose:
        bad = len(bad_index_list)
        percent_bad_frames = (bad * 100) / n
        msg1 = "Done detecting bad frames from cube: {} out of {} ({:.3}%)"
        print(msg1.format(bad, n, percent_bad_frames)) 

    if plot:
        plt.figure(figsize=vip_figsize)
        plt.plot(mean_values, 'o', alpha=0.6)
        plt.plot(mean_smooth, label='smoothed mean fluctuation', lw=2, ls='-',
                 alpha=0.5)
        plt.plot(top_boundary, label='upper threshold', lw=1.4, ls='-',
                 color='#9467bd', alpha=0.8)
        plt.plot(bot_boundary, label='lower threshold', lw=1.4, ls='-',
                 color='#9467bd', alpha=0.8)
        plt.legend(fancybox=True, framealpha=0.5, loc='best')
        plt.grid('on', alpha=0.2)
        plt.ylabel('Mean value in '+mode)
        plt.xlabel('Frame number')

    if verbose:
        timing(start_time)

    good_index_list = np.array(good_index_list)
    bad_index_list = np.array(bad_index_list)
    
    return good_index_list, bad_index_list


def cube_detect_badfr_ellipticity(array, fwhm, crop_size=30, roundlo=-0.2,
                                  roundhi=0.2, plot=True, verbose=True):
    """ Returns the list of bad frames  from a cube by measuring the PSF 
    ellipticity of the central source. Should be applied on a recentered cube.
    
    Parameters
    ----------
    array : numpy ndarray 
        Input 3d array, cube.
    fwhm : float
        FWHM size in pixels.
    crop_size : int, optional
        Size in pixels of the square subframe to be analyzed.
    roundlo, roundhi : float, optional
        Lower and higher bounds for the ellipticity. See ``Notes`` below for
        details.
    plot : bool, optional
        If true it plots the central PSF roundness for each frame.
    verbose : bool, optional
        Whether to print to stdout or not.
        
    Returns
    -------
    good_index_list : numpy ndarray
        1d array of good indices.
    bad_index_list : numpy ndarray
        1d array of bad frames indices.
    
    Notes
    -----
    From photutils.DAOStarFinder documentation:
    DAOFIND calculates the object roundness using two methods. The 'roundlo'
    and 'roundhi' bounds are applied to both measures of roundness. The first
    method ('roundness1'; called 'SROUND' in DAOFIND) is based on the source 
    symmetry and is the ratio of a measure of the object's bilateral (2-fold) 
    to four-fold symmetry. The second roundness statistic ('roundness2'; called 
    'GROUND' in DAOFIND) measures the ratio of the difference in the height of
    the best fitting Gaussian function in x minus the best fitting Gaussian 
    function in y, divided by the average of the best fitting Gaussian 
    functions in x and y. A circular source will have a zero roundness. A source
    extended in x or y will have a negative or positive roundness, respectively.
    
    """
    from .cosmetics import cube_crop_frames

    check_array(array, 3, msg='array')
    
    if verbose:
        start_time = time_ini()

    array = cube_crop_frames(array, crop_size, verbose=False)
    n = array.shape[0]
    goodfr = []
    badfr = []
    roundness1 = []
    roundness2 = []
    for i in range(n):
        ff_clipped = sigma_clip(array[i], sigma=3, maxiters=None)
        thr = ff_clipped.max()
        DAOFIND = DAOStarFinder(threshold=thr, fwhm=fwhm)
        tbl = DAOFIND.find_stars(array[i])
        table_mask = (tbl['peak'] == tbl['peak'].max())
        tbl = tbl[table_mask]
        roun1 = tbl['roundness1'][0]
        roun2 = tbl['roundness2'][0]
        roundness1.append(roun1)
        roundness2.append(roun2)
        # we check the roundness
        if roundhi > roun1 > roundlo and roundhi > roun2 > roundlo:
            goodfr.append(i)
        else:
            badfr.append(i)
    
    bad_index_list = np.array(badfr)
    good_index_list = np.array(goodfr)

    if plot:
        _, ax = plt.subplots(figsize=vip_figsize)
        x = range(len(roundness1))
        if n > 5000:
            marker = ','
        else:
            marker = 'o'
        ax.plot(x, roundness1, '-', alpha=0.6, color='#1f77b4',
                label='roundness1')
        ax.plot(x, roundness1, marker=marker, alpha=0.4, color='#1f77b4')
        ax.plot(x, roundness2, '-', alpha=0.6, color='#9467bd',
                label='roundness2')
        ax.plot(x, roundness2, marker=marker, alpha=0.4, color='#9467bd')
        ax.hlines(roundlo, xmin=-1, xmax=n + 1, lw=2, colors='#ff7f0e',
                  linestyles='dashed', label='roundlo', alpha=0.6)
        ax.hlines(roundhi, xmin=-1, xmax=n + 1, lw=2, colors='#ff7f0e',
                  linestyles='dashdot', label='roundhi', alpha=0.6)
        plt.xlabel('Frame number')
        plt.ylabel('Roundness')
        plt.xlim(xmin=-1, xmax=n + 1)
        plt.legend(fancybox=True, framealpha=0.5, loc='best')
        plt.grid('on', alpha=0.2)

    if verbose:
        bad = len(bad_index_list)
        percent_bad_frames = (bad*100)/n
        msg1 = "Done detecting bad frames from cube: {} out of {} ({:.3}%)"
        print(msg1.format(bad, n, percent_bad_frames))
        timing(start_time)
    
    return good_index_list, bad_index_list


def cube_detect_badfr_correlation(array, frame_ref, crop_size=30,
                                  dist='pearson', percentile=20, threshold=None, 
                                  mode='full', inradius=None, width=None, 
                                  plot=True, verbose=True, full_output=False):
    """ Returns the list of bad frames from a cube by measuring the distance 
    (similarity) or correlation of the frames (cropped to a 30x30 subframe) 
    wrt a reference frame from the same cube. Then the distance/correlation 
    level is thresholded (percentile parameter) to find the outliers. Should be 
    applied on a recentered cube.
    
    Parameters
    ----------
    array : numpy ndarray 
        Input 3d array, cube.
    frame_ref : int or 2d array
        Index of the frame that will be used as a reference or 2d reference
        array.
    crop_size : int, optional
        Size in pixels of the square subframe to be analyzed.
    dist : {'sad','euclidean','mse','pearson','spearman','ssim'}, str optional
        One of the similarity or dissimilarity measures from function
        vip_hci.stats.distances.cube_distance(). 
    percentile : int, optional
        The percentage of frames that will be discarded, if threshold is not 
        provided.
    threshold: None or float, optional
        If provided, corresponds to the threshold 'distance' value above/below 
        which (depending on index of similarity/dissimilarity resp.) will be 
        discarded. If not None, supersedes 'percentile'.
    mode : {'full','annulus'}, string optional
        Whether to use the full frames or a centered annulus.
    inradius : None or int, optional
        The inner radius when mode is 'annulus'.
    width : None or int, optional
        The width when mode is 'annulus'.
    plot : bool, optional
        If true it plots the mean fluctuation as a function of the frames and 
        the boundaries.
    verbose : bool, optional
        Whether to print to stdout or not.
    full_output: bool, optional
        Whether to also return the array of distances.
        
    Returns
    -------
    good_index_list : numpy ndarray
        1d array of good indices.
    bad_index_list : numpy ndarray
        1d array of bad frames indices.
        
    """
    from .cosmetics import cube_crop_frames, frame_crop
    
    check_array(array, 3, msg='array')
    
    if verbose:
        start_time = time_ini()
    
    n = array.shape[0]
    # the cube is cropped to the central area
    subarray = cube_crop_frames(array, crop_size, verbose=False)
    if isinstance(frame_ref, np.ndarray):
        frame_ref = frame_crop(frame_ref, crop_size, verbose=False)
    distances = cube_distance(subarray, frame_ref, mode, dist, 
                              inradius=inradius, width=width, plot=False)
    
    if dist == 'pearson' or dist == 'spearman' or dist == 'ssim':
        # measures of correlation or similarity
        minval = np.min(distances[~np.isnan(distances)])
        distances = np.nan_to_num(distances)
        distances[np.where(distances == 0)] = minval
        if threshold is None:
            threshold = np.percentile(distances, percentile)
        indbad = np.where(distances <= threshold)
        indgood = np.where(distances > threshold)
    else:
        # measures of dissimilarity
        if threshold is None:
            threshold = np.percentile(distances, 100-percentile)
        indbad = np.where(distances >= threshold)
        indgood = np.where(distances < threshold)
        
    bad_index_list = indbad[0]
    good_index_list = indgood[0]
    if verbose:
        bad = len(bad_index_list)
        percent_bad_frames = (bad*100)/n
        msg1 = "Done detecting bad frames from cube: {} out of {} ({:.3}%)"
        print(msg1.format(bad, n, percent_bad_frames))
    
    if plot:
        lista = distances
        _, ax = plt.subplots(figsize=vip_figsize)
        x = range(len(lista))
        ax.plot(x, lista, '-', alpha=0.6, color='#1f77b4')
        if n > 5000:
            marker = ','
        else:
            marker = 'o'
        ax.plot(x, lista, marker=marker, alpha=0.4, color='#1f77b4')
        if isinstance(frame_ref, int):
            ax.vlines(frame_ref, ymin=np.nanmin(lista), ymax=np.nanmax(lista), 
                      colors='green', linestyles='dashed', lw=2, alpha=0.6,
                      label='Reference frame '+str(frame_ref))
        ax.hlines(threshold, xmin=-1, xmax=n+1, lw=2, colors='#ff7f0e',
                  linestyles='dashed', label='Threshold', alpha=0.6)
        plt.xlabel('Frame number')
        if dist == 'sad':
            plt.ylabel('SAD - Manhattan distance')
        elif dist == 'euclidean':
            plt.ylabel('Euclidean distance')
        elif dist == 'pearson':
            plt.ylabel('Pearson correlation coefficient')
        elif dist == 'spearman':
            plt.ylabel('Spearman correlation coefficient')
        elif dist == 'mse':
            plt.ylabel('Mean squared error')
        elif dist == 'ssim':
            plt.ylabel('Structural Similarity Index')
        
        plt.xlim(xmin=-1, xmax=n+1)
        plt.legend(fancybox=True, framealpha=0.5, loc='best')
        plt.grid('on', alpha=0.2)
    
    if verbose:
        timing(start_time)
    
    if full_output:
        return good_index_list, bad_index_list, distances
    else:   
        return good_index_list, bad_index_list