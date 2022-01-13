#! /usr/bin/env python

"""
Module for image statistics.
"""

__author__ = 'Carlos Alberto Gomez Gonzalez'
__all__ = ['frame_histo_stats',
           'frame_average_radprofile']

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from ..var import frame_center
from ..config.utils_conf import check_array, vip_figsize


def frame_average_radprofile(frame, sep=1, init_rad=None, plot=True):
    """ Calculates the average radial profile of an image.

    Parameters
    ----------
    frame : numpy ndarray
        Input image or 2d array.
    sep : int, optional
        The average radial profile is recorded every ``sep`` pixels.
    plot : bool, optional
        If True the profile is plotted.

    Returns
    -------
    df : dataframe
        Pandas dataframe with the radial profile and distances.

    Notes
    -----
    https://stackoverflow.com/questions/21242011/most-efficient-way-to-calculate-radial-profile
    https://stackoverflow.com/questions/48842320/what-is-the-best-way-to-calculate-radial-average-of-the-image-with-python
    https://github.com/keflavich/image_tools/blob/master/image_tools/radialprofile.py

    """
    check_array(frame, dim=2)
    cy, cx = frame_center(frame)

    if init_rad is None:
        init_rad = 1
    x, y = np.indices((frame.shape))
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    r = r.astype(int)
    tbin = np.bincount(r.ravel(), frame.ravel())
    nr = np.bincount(r.ravel())
    radprofile = tbin / nr

    radists = np.arange(init_rad + 1, int(cy), sep) - 1
    radprofile_radists = radprofile[radists]
    nr_radists = nr[radists]
    df = pd.DataFrame({'rad': radists, 'radprof': radprofile_radists,
                       'npx': nr_radists})

    if plot:
        plt.figure(figsize=vip_figsize)
        plt.plot(radists, radprofile_radists, '.-', alpha=0.6)
        plt.grid(which='both', alpha=0.4)
        plt.xlabel('Pixels')
        plt.ylabel('Counts')
        plt.minorticks_on()
        plt.xlim(0)

    return df


def frame_histo_stats(image_array, plot=True):
    """Plots a frame with a colorbar, its histogram and some statistics: mean,
    median, maximum, minimum and standard deviation values.  
    
    Parameters
    ----------
    image_array : numpy ndarray
        The input frame.  
    plot : bool, optional
        If True plots the frame and the histogram with the values.
        
    Return
    ------
    mean : float
        Mean value of array.
    median : float
        Median value of array.
    std : float
        Standard deviation of array.
    maxim : int or float
        Maximum value.
    minim : int or float
        Minimum value.
        
    """
    vector = image_array.flatten()
    mean = vector.mean()
    median = np.median(vector)
    maxim = vector.max()
    minim = vector.min()
    std = vector.std()
   
    if plot:
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
        ax0, ax1 = axes.flat
        bins = int(np.sqrt(vector.shape[0]))
        txt = 'Mean = {:.3f}\n'.format(mean) + \
              'Median = {:.3f}\n'.format(median) +\
              'Stddev = {:.3f}\n'.format(std) +\
              'Max = {:.3f}\n'.format(maxim) +\
              'Min = {:.3f}\n\n'.format(minim)

        ax0.imshow(image_array, interpolation="nearest", origin ="lower",
                   cmap='viridis')
        ax0.set_title('Frame')
        ax0.grid('off')

        ax1.hist(vector, bins=bins, label=txt, alpha=0.5, histtype='stepfilled')
        ax1.set_yscale('log')
        ax1.set_title('Histogram')
        ax1.text(0.98, 0.98, txt, transform=ax1.transAxes, fontsize=12,
                 verticalalignment='top', horizontalalignment='right')
        plt.show()
        
    return mean, median, std, maxim, minim

