#! /usr/bin/env python

"""
Module for stats of a fits-cube.
"""

from __future__ import division 

__author__ = 'Carlos Alberto Gomez Gonzalez'
__all__ = ['cube_stats_aperture',
           'cube_stats_annulus']

import numpy as np
import scipy.stats
from matplotlib import pyplot as plt
from ..var import get_annulus, get_circle, get_annulus_cube


def cube_stats_aperture(arr, radius, xy=None, plot=False, full_output=False):                                                 
    """Calculates statistics in an aperture on a 2D or 3D array and plots the 
    variation of the mean, median and standard deviation as a functions of time.
    
    Parameters
    ----------
    arr : array_like
        Input array.
    radius : int
        Radius.
    xy : tuple of ints
        Corrdinated of the center of the aperture
    plot : None,1,2, optional
        If 1 or True it plots the mean, std_dev and max. Also the histogram. 
        If 2 it also plots the linear correlation between the median and the
        std_dev. 
    full_output : {False,True}, optional
        If true it returns mean, std_dev, median, if false just the mean.
        
    Returns
    -------
    If full_out is true it returns the sum, mean, std_dev, median. If false 
    only the mean.
    """
    if arr.ndim == 2:        
        if xy is not None:
            x, y = xy
            circle = get_circle(arr, radius, output_values=True, cy=y, cx=x)
        else:
            circle = get_circle(arr, radius, output_values=True)
        maxi = circle.max()
        mean = circle.mean()
        std_dev = circle.std()
        median = np.median(circle)
            
    if arr.ndim == 3:
        n = arr.shape[0]
        mean = np.empty(n)
        std_dev = np.empty(n)
        median = np.empty(n)
        maxi = np.empty(n)
        
        values_circle = []
        for i in range(n):
            if xy is not None:
                x, y = xy
                circle = get_circle(arr[i], radius, output_values=True, cy=y, cx=x)
            else:
                circle = get_circle(arr[i], radius, output_values=True)
            values_circle.append(circle)
            maxi[i] = circle.max()
            mean[i] = circle.mean()
            std_dev[i] = circle.std()
            median[i] = np.median(circle)
        values_circle = np.array(values_circle).flatten()
        
        if plot==1 or plot==2:
            plt.figure('Image crop (first slice)', figsize=(12,3))
            if xy is not None:
                x, y = xy
                temp = get_circle(arr[0], radius, cy=y, cx=x)
            else:
                temp = get_circle(arr[0], radius)
            ax1 = plt.subplot(1, 2, 1)
            ax1.imshow(arr[0], origin = 'lower', interpolation="nearest", 
                       cmap = plt.get_cmap('gray'), alpha=0.8)
            ax1.imshow(temp, origin = 'lower', interpolation="nearest", 
                       cmap = plt.get_cmap('CMRmap'), alpha=0.6)                     
            plt.axis('on')
            ax2 = plt.subplot(1, 2, 2)
            ax2.hist(values_circle, bins=int(np.sqrt(values_circle.shape[0])),
                     alpha=0.5, histtype='stepfilled', label='Histogram')
            ax2.legend()
            ax2.tick_params(axis='x', labelsize=8)

            fig = plt.figure('Stats in annulus', figsize=(12, 6))
            fig.subplots_adjust(hspace=0.15)
            ax1 = plt.subplot(3, 1, 1)
            std_of_means = np.std(mean)
            median_of_means = np.median(mean)
            lab = 'mean (median={:.1f}, stddev={:.1f})'.format(median_of_means,
                                                               std_of_means)
            ax1.axhline(median_of_means, alpha=0.5, color='gray', lw=2, ls='--')
            ax1.plot(mean, '.-', label=lab, lw = 0.8, alpha=0.6, marker='o', 
                     color='b')
            ax1.legend(loc='best', fancybox=True).get_frame().set_alpha(0.5)
            ax1.grid(True)
            plt.setp(ax1.get_xticklabels(), visible=False)
            
            ax2 = plt.subplot(3, 1, 2, sharex=ax1)
            ax2.plot(std_dev, '.-', label='std_dev', lw = 0.8, alpha=0.6, 
                     marker='o', color='r')
            ax2.legend(loc='best', fancybox=True).get_frame().set_alpha(0.5)
            ax2.grid(True)
            plt.setp(ax2.get_xticklabels(), visible=False)

            ax3 = plt.subplot(3, 1, 3, sharex=ax1)
            ax3.plot(maxi, '.-', label='max', lw=0.8, alpha=0.6, marker='o',
                     color='g')
            ax3.legend(loc='best', fancybox=True).get_frame().set_alpha(0.5)  
            ax3.grid(True)  

            if plot==2:
                plt.figure('Std_dev - mean in annulus', figsize=(4, 4))                
                plt.scatter(std_dev, mean, alpha=0.6)
                m, b = np.polyfit(std_dev, mean, 1)
                corr, _ = scipy.stats.pearsonr(mean, std_dev)
                plt.plot(std_dev, m*std_dev + b, '-', label=corr, alpha=0.6)
                plt.xlabel('Mean')
                plt.ylabel('Standard deviation')
                plt.legend()
    
    if full_output:
        return mean, std_dev, median, maxi
    else:
        return mean


def cube_stats_annulus(array, inner_radius, size, plot=None, full_out=False):
    """Calculates statistics in a centered annulus on a 2D or 3D array and 
    plots the variation of the mean, median and standard deviation as a 
    functions of time.
    
    Parameters
    ----------
    array : array_like
        Input array.
    inner_radius : int
        Annulus inner radius.
    size : int
        How many pixels in radial direction contains the annulus.
    plot : None,1,2, optional
        If 1 or True it plots the mean, std_dev and max. Also the histogram. 
        If 2 it also plots the linear correlation between the median and the
        std_dev. 
    full_out : {False,True}, optional
        If true it returns mean, std_dev, median, if false just the mean.
        
    Returns
    -------
    If full_out is true it returns mean, std_dev, median, if false 
    only the mean.
    """
    if array.ndim==2:
        arr = array.copy()    
        
        annulus = get_annulus(arr, inner_radius, size, output_values=True)
        mean = annulus.mean()
        std_dev = annulus.std()
        median = np.median(annulus)
        maxi = annulus.max()
            
    if array.ndim==3:
        n = array.shape[0]
        mean = np.empty(n)
        std_dev = np.empty(n)
        median = np.empty(n)
        maxi = np.empty(n)
        
        for i in range(n):
            arr = array[i].copy() 
            annulus = get_annulus(arr, inner_radius, size, output_values=True)
            mean[i] = annulus.mean()
            std_dev[i] = annulus.std()
            median[i] = np.median(annulus)
            maxi[i] = annulus.max()
            
        if plot==1 or plot==2:
            plt.figure('Image crop (first slice)', figsize=(12,3))
            temp = get_annulus_cube(array, inner_radius, size)
            ax1 = plt.subplot(1, 2, 1)
            ax1.imshow(array[0], origin = 'lower', interpolation="nearest", 
                       cmap = plt.get_cmap('gray'), alpha=0.8) 
            ax1.imshow(temp[0], origin = 'lower', interpolation="nearest", 
                       cmap = plt.get_cmap('CMRmap'), alpha=0.6)                           
            plt.axis('on')
            ax2 = plt.subplot(1, 2, 2)
            values = temp[np.where(temp>0)]
            ax2.hist(values.ravel(), bins=int(np.sqrt(values.shape[0])),
                     alpha=0.5, histtype='stepfilled', label='Histogram')
            ax2.legend()
            ax2.tick_params(axis='x', labelsize=8)
            
            fig = plt.figure('Stats in annulus', figsize=(12, 6))
            fig.subplots_adjust(hspace=0.15)
            ax1 = plt.subplot(3, 1, 1)
            std_of_means = np.std(mean)
            median_of_means = np.median(mean)
            lab = 'mean (median={:.1f}, stddev={:.1f})'.format(median_of_means,
                                                               std_of_means)
            ax1.axhline(median_of_means, alpha=0.5, color='gray', lw=2, ls='--')
            ax1.plot(mean, '.-', label=lab, lw = 0.8, alpha=0.6, marker='o', 
                     color='b')
            ax1.legend(loc='best', fancybox=True).get_frame().set_alpha(0.5)
            ax1.grid(True)
            plt.setp(ax1.get_xticklabels(), visible=False)
            
            ax2 = plt.subplot(3, 1, 2, sharex=ax1)
            ax2.plot(std_dev, '.-', label='std_dev', lw = 0.8, alpha=0.6, 
                     marker='o', color='r')
            ax2.legend(loc='best', fancybox=True).get_frame().set_alpha(0.5)
            ax2.grid(True)
            plt.setp(ax2.get_xticklabels(), visible=False)

            ax3 = plt.subplot(3, 1, 3, sharex=ax1)
            ax3.plot(maxi, '.-', label='max', lw=0.8, alpha=0.6, marker='o',
                     color='g')
            ax3.legend(loc='best', fancybox=True).get_frame().set_alpha(0.5)  
            ax3.grid(True)          
            
            if plot==2:
                plt.figure('Std_dev - mean in annulus', figsize=(4, 4))                
                plt.scatter(std_dev, mean, alpha=0.6)
                m, b = np.polyfit(std_dev, mean, 1)
                corr, _ = scipy.stats.pearsonr(mean, std_dev)
                plt.plot(std_dev, m*std_dev + b, '-', label=corr, alpha=0.6)
                plt.xlabel('Mean')
                plt.ylabel('Standard deviation')
                plt.legend()

    if full_out:
        return mean, std_dev, median, maxi
    else:
        return mean






