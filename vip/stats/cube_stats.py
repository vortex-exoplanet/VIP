#! /usr/bin/env python

"""
Module for stats of a fits-cube.
"""

from __future__ import division 

__author__ = 'C. Gomez @ ULg'
__all__ = ['cube_stats_aperture',
           'cube_stats_annulus']

import numpy as np
import scipy.stats
from matplotlib import pyplot as plt
from ..var import get_annulus, get_circle


def cube_stats_aperture(arr, radius, y=None, x=None, plot=False, 
                        full_output=False, verbose=True):                                                 
    """Calculates statistics in an aperture on a 2D or 3D array and plots the 
    variation of the mean, median and standard deviation as a functions of time.
    
    Parameters
    ----------
    arr : array_like
        Input array.
    radius : int
        Radius.
    plot : {False,True}, optional
        If true it plots the mean, median and std dev.
    full_output : {False,True}, optional
        If true it returns mean, std_dev, median, if false just the 
        mean.
    verbose : {True,False}, bool optional
        If false it turns off the print message.
        
    Returns
    -------
    If full_out is true it returns the sum, mean, std_dev, median. If false 
    only the mean.
    """
    if arr.ndim == 2:
        height, widht = arr.shape
        
        if x and y:
            circle = get_circle(arr, radius, output_values=True, cy=y, cx=x)
        else:
            circle = get_circle(arr, radius, output_values=True)
        suma = circle.sum()
        mean = circle.mean()
        std_dev = circle.std()
        median = np.median(circle)
            
    if arr.ndim == 3:
        n, height, widht = arr.shape
        mean = np.empty(n)
        std_dev = np.empty(n)
        median = np.empty(n)
        suma = np.empty(n)
        
        values_circle = []
        for i in xrange(n):
            if x and y:
                circle = get_circle(arr[i], radius, output_values=True, cy=y, cx=x)
            else:
                circle = get_circle(arr[i], radius, output_values=True)
            values_circle.append(circle)
            suma[i] = circle.sum()
            mean[i] = circle.mean()
            std_dev[i] = circle.std()
            median[i] = np.median(circle)
        values_circle = np.array(values_circle).flatten()
        
        if plot:
            title1 = 'Stats in aperture'
            title2 = 'Std_dev - mean in aperture'
            title3 = 'Histogram'
            plt.close(title1) 
            plt.close(title2)
            plt.close(title3)
            
            plt.figure(title1, figsize=(12, 8))    
            plt.subplot(211)
            plt.grid(which='major')
            #plt.yscale('log')
            if n < 400:
                plt.minorticks_on()
                plt.xticks(range(0, n, 10))
            plt.plot(mean, label='mean', lw = 0.8, ls='-', alpha=0.6)
            #plt.plot(suma, label='sum', lw = 0.8, alpha=0.6)
            #plt.plot(mean/std_dev, label='mean/std_dev', lw = 0.8)
            plt.legend(fancybox=True).get_frame().set_alpha(0.5)
            plt.subplot(212)
            plt.grid()
            plt.minorticks_on()
            #plt.yscale('log')
            plt.plot(std_dev, label='std dev', lw = 0.5, ls='-', color='red', 
                     alpha=0.6)
            plt.plot(median, label='median', lw = 0.8, ls='-', alpha=0.6)
            plt.legend(fancybox=True).get_frame().set_alpha(0.5) 
            
            plt.axes([0.80,0.55,0.1,0.1])
            plt.axis('off')
            if x and y:
                circle_im = get_circle(arr.mean(axis=0), radius=radius, cy=y, cx=x)           
                plt.imshow(circle_im, origin='lower', interpolation="nearest", 
                           cmap=plt.get_cmap('gray'), alpha=0.5)                     
                plt.xlim(x-radius, x+radius)
                plt.ylim(y-radius, y+radius)
            else:
                circle_im = get_circle(arr.mean(axis=0), radius)
                plt.imshow(circle_im, origin='lower', interpolation="nearest", 
                           cmap=plt.get_cmap('gray'), alpha=0.5) 
                plt.xlim((widht/2)-(radius), (widht/2)+(radius))
                plt.ylim((height/2)-(radius), (height/2)+(radius))
            
            plt.figure(title2, figsize=(12, 8))
            plt.scatter(std_dev, mean)
            m, b = np.polyfit(std_dev, mean, 1)
            corr, _ = scipy.stats.pearsonr(mean, std_dev)
            plt.plot(std_dev, m*std_dev + b, '-', label=corr)
            plt.xlabel('Mean')
            plt.ylabel('Standard deviation')
            plt.legend()
            #print 'm = ', m, '  angle = ', np.rad2deg(np.arctan(m))
            
            plt.figure(title3)
            plt.hist(values_circle, bins=np.sqrt(values_circle.shape[0]),
                     alpha=0.5, histtype='stepfilled', label='Px values')
            plt.legend()
        
            #plt.show(block=False)
    
    if verbose:        
        print "Done calculating stats in circular aperture"
    
    if full_output:
        return suma, mean, std_dev, median
    else:
        return mean


def cube_stats_annulus(array, inner_radius, size, plot=None, full_out=False, 
                       verbose=True):
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
        If 1 it plots the mean, std_dev and max.
        If 2 it also plots the median, mean/std_dev and the mean and std_dev 
        correlation. 
    full_out : {False,True}, optional
        If true it returns mean, std_dev, median, if false just the 
        mean.
    verbose : {True,False}, bool optional
        If false it turns off the print message.
        
    Returns
    -------
    If full_out is true it returns mean, std_dev, median, if false 
    only the mean.
    """
    if len(array.shape) == 2:
        height, widht = array.shape
        arr = array.copy()    
        
        annulus = get_annulus(arr, inner_radius, size, output_values=True)
        mean = annulus.mean()
        std_dev = annulus.std()
        median = np.median(annulus)
        maxi = annulus.max()
            
    if len(array.shape) == 3:
        n, height, widht = array.shape
        mean = np.empty(n)
        std_dev = np.empty(n)
        median = np.empty(n)
        maxi = np.empty(n)
        
        for i in xrange(n):
            arr = array[i].copy() 
            
            annulus = get_annulus(arr, inner_radius, size, output_values=True)
            mean[i] = annulus.mean()
            std_dev[i] = annulus.std()
            median[i] = np.median(annulus)
            maxi[i] = annulus.max()
            
        if plot==1 or plot==2:
            plt.close('Stats in annulus 1') 
            plt.close('Std_dev - mean in annulus')
            
            plt.figure('Stats in annulus 1', figsize=(12, 4))    
            plt.grid('on')
            if n < 400:
                plt.minorticks_on()
                plt.xticks(range(0, n, 10))
            plt.plot(mean, 'o-', label='mean', lw = 0.8, alpha=0.6)
            plt.plot(std_dev, 'o-', label='std_dev', lw = 0.8, alpha=0.6)
            plt.plot(maxi, 'o-', label='max', lw=0.8, alpha=0.6)
            plt.legend(loc='best', fancybox=True).get_frame().set_alpha(0.5)
            
            plt.axes([0.095,0.79,0.1,0.1])
            plt.axis('off')
            if len(arr.shape) == 3:
                annulus = get_annulus(arr[0], inner_radius, size)
            else:
                annulus = get_annulus(arr, inner_radius, size)
            plt.imshow(annulus, origin = 'lower', interpolation="nearest", 
                       cmap = plt.get_cmap('CMRmap'))                           # shows annulus from 1st fr
            plt.xlim((widht/2)-(inner_radius+size), 
                     (widht/2)+(inner_radius+size))
            plt.ylim((height/2)-(inner_radius+size), 
                     (height/2)+(inner_radius+size))
            
            if plot==2:
                plt.figure('Stats in annulus 2', figsize=(12, 8))
                plt.grid('on')
                plt.minorticks_on()
                plt.plot(mean/std_dev, label='mean/std dev', lw = 0.5, ls='-')
                plt.plot(median, label='median', lw = 0.8, ls='-')
                plt.legend(fancybox=True).get_frame().set_alpha(0.5) 
                
                plt.figure('Std_dev - mean in annulus', figsize=(12, 8))
                plt.scatter(std_dev, mean)
                m, b = np.polyfit(std_dev, mean, 1)
                corr, _ = scipy.stats.pearsonr(mean, std_dev)
                plt.plot(std_dev, m*std_dev + b, '-', label=corr)
                plt.xlabel('Mean')
                plt.ylabel('Standard deviation')
                plt.legend()
                #print 'm = ', m, '  angle = ', np.rad2deg(np.arctan(m))
    
    if verbose:        
        print "Done calculating stats in annulus"
    
    if full_out:
        return mean, std_dev, median, maxi
    else:
        return mean






