#! /usr/bin/env python

"""
Module for image statistics.
"""

from __future__ import division

__author__ = 'Carlos Alberto Gomez Gonzalez'
__all__ = ['frame_histo_stats']

import numpy as np
from matplotlib import pyplot as plt


def frame_histo_stats(image_array, plot=True):
    """Plots a frame with a colorbar, its histogram and some statistics: mean,
    median, maximum, minimum and standard deviation values.  
    
    Parameters
    ----------
    image_array : array_like
        The input frame.  
    plot : {True, False}, bool optional
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
    minim " int or float
        Minimum value.
        
    """
    vector = image_array.flatten()
    mean = vector.mean()
    median = np.median(vector)
    maxim = vector.max()
    minim = vector.min()
    std = vector.std()
    
    if plot is True:
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14,5))
        ax0, ax1 = axes.flat
        bins = np.sqrt(vector.shape[0])
        txt = 'Mean = {:.3f}\n'.format(mean) + \
              'Median = {:.3f}\n'.format(median) +\
              'Stddev = {:.3f}\n'.format(std) +\
              'Max = {:.3f}\n'.format(maxim) +\
              'Min = {:.3f}\n\n'.format(minim)
        ax0.hist(vector, bins=bins, label=txt, alpha=0.5, histtype='stepfilled')
        ax0.set_yscale('log')
        ax0.set_title('Histogram')
        ax0.text(0.98, 0.98, txt, transform=ax0.transAxes, fontsize=12,
                 verticalalignment='top', horizontalalignment='right')

        ima = ax1.imshow(image_array, interpolation="nearest", origin ="lower",
                         cmap='CMRmap')
        ax1.set_title('Frame')
        ax1.grid('off')
        fig.colorbar(ima)
        plt.jet()
        plt.show()
        
    return mean, median, std, maxim, minim




    
