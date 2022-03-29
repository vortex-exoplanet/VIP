#! /usr/bin/env python

"""
Various stat functions.
"""



__author__ = 'Carlos Alberto Gomez Gonzalez'
__all__ = ['descriptive_stats',
           'frame_basic_stats',
           'cube_basic_stats']

import numpy as np
from matplotlib.pyplot import boxplot
from matplotlib import pyplot as plt
from ..var import get_annulus_segments, get_circle
from ..config.utils_conf import vip_figsize


def descriptive_stats(array, verbose=True, label='', mean=False, plot=False):
    """ Simple statistics from vector.
    """
    if mean:
        mean_ = np.mean(array)
    median = np.median(array)
    mini = np.min(array)
    maxi = np.max(array)
    first_qu = np.percentile(array, 25)
    third_qu = np.percentile(array, 75)
    
    if verbose:
        if mean:
            label += 'min={:.1f} / 1st QU={:.1f} / ave={:.1f} / med={:.1f} / '
            label += '3rd QU={:.1f} / max={:.1f}'
            print(label.format(mini, first_qu, mean_, median, third_qu, maxi))
        else:
            label += 'min={:.1f} / 1st QU={:.1f} / med={:.1f} / 3rd QU={:.1f} '
            label += '/ max={:.1f}'
            print(label.format(mini, first_qu, median, third_qu, maxi))
    
    if plot:
        boxplot(array, vert=False, meanline=mean, showfliers=True, sym='.')
    
    if mean:
        return mini, first_qu, mean_, median, third_qu, maxi
    else:
        return mini, first_qu, median, third_qu, maxi


def frame_basic_stats(arr, region='circle', radius=5, xy=None, inner_radius=0,
                      size=5, plot=True, full_output=False):
    """ Calculates statistics in a ``region`` of a 2D array.

    Parameters
    ----------
    arr : numpy ndarray
        Input array.
    region : {'circle', 'annulus'}, str optional
        Pixels are extracted either from a centered annulus or a circular
        aperture centered on ``xy``.
    radius : int
        Radius of the circular aperture.
    xy : tuple of ints
        Coordinates of the center of the circular aperture.
    inner_radius : int
        Annulus inner radius of the annulus.
    size : int
        Width of the annulus.
    plot : bool, optional
        If True it plots the histogram and the region.
    full_output : bool, optional
        If true it returns mean, std_dev, median, if false just the mean.

    Returns
    -------
    If full_out is true it returns the sum, mean, std_dev, median. If false
    only the mean.
    """
    if region == 'circle':
        if xy is not None:
            x, y = xy
        else:
            x, y = None, None
        region_pxs = get_circle(arr, radius, cy=y, cx=x, mode="val")
    elif region == 'annulus':
        region_pxs = get_annulus_segments(arr, inner_radius, size,
                                          mode="val")[0]
    else:
        raise ValueError('Region not recognized')

    maxi = region_pxs.max()
    mean = region_pxs.mean()
    std_dev = region_pxs.std()
    median = np.median(region_pxs)

    if plot:
        plt.figure('Image crop (first slice)', figsize=(10, 4))
        if region == 'circle':
            temp = get_circle(arr, radius, cy=y, cx=x)
        elif region == 'annulus':
            temp = get_annulus_segments(arr, inner_radius, size, mode="mask")[0]
        else:
            raise ValueError('Region not recognized')
        temp[temp == 0] = np.nan
        ax1 = plt.subplot(1, 2, 1)
        ax1.imshow(arr, origin='lower', interpolation="nearest", cmap='gray')
        ax1.imshow(temp, origin='lower', interpolation="nearest",
                   cmap='viridis')
        ax1.set_title('Frame region')
        plt.axis('on')
        ax2 = plt.subplot(1, 2, 2)
        ax2.hist(region_pxs, bins=int(np.sqrt(region_pxs.shape[0])),
                 alpha=0.5, histtype='stepfilled')
        ax2.set_title('Histogram')
        ax2.tick_params(axis='x', labelsize=8)
        plt.show()

    if full_output:
        return mean, std_dev, median, maxi
    else:
        return mean


def cube_basic_stats(arr, region='circle', radius=5, xy=None, inner_radius=0,
                     size=5, plot=False, full_output=False):
    """ Calculates statistics in a region on a 3D array and plots the variation
    of the mean, median and standard deviation as a functions of time.

    Parameters
    ----------
    arr : numpy ndarray
        Input array.
    region : {'circle', 'annulus'}, str optional
        Pixels are extracted either from a centered annulus or a circular
        aperture centered on ``xy``.
    radius : int
        Radius of the circular aperture.
    xy : tuple of ints
        Coordinates of the center of the circular aperture.
    inner_radius : int
        Annulus inner radius of the annulus.
    size : int
        Width of the annulus.
    plot : bool, optional
        If True it plots the mean, std_dev and max. Also the histogram.
    full_output : bool, optional
        If true it returns mean, std_dev, median, if false just the mean.

    Returns
    -------
    If full_out is true it returns the sum, mean, std_dev, median. If false
    only the mean.
    """
    n = arr.shape[0]
    mean = np.empty(n)
    std_dev = np.empty(n)
    median = np.empty(n)
    maxi = np.empty(n)

    values_region = []
    for i in range(n):
        if region == 'circle':
            if xy is not None:
                x, y = xy
            else:
                x, y = None, None
            region_pxs = get_circle(arr[i], radius, cy=y, cx=x, mode="val")
        elif region == 'annulus':
            region_pxs = get_annulus_segments(arr[i], inner_radius, size,
                                              mode="val")[0]
        else:
            raise ValueError('Region not recognized')

        values_region.append(region_pxs)
        maxi[i] = region_pxs.max()
        mean[i] = region_pxs.mean()
        std_dev[i] = region_pxs.std()
        median[i] = np.median(region_pxs)

    values_region = np.array(values_region).flatten()

    if plot:
        plt.figure('Image crop (first slice)', figsize=vip_figsize)
        if region == 'circle':
            temp = get_circle(arr[0], radius, cy=y, cx=x)
        elif region == 'annulus':
            temp = get_annulus_segments(arr[0], inner_radius, size, mode="mask")[0]
        else:
            raise ValueError('Region not recognized')
        temp[temp == 0] = np.nan
        ax1 = plt.subplot(1, 2, 1)
        ax1.imshow(arr[0], origin='lower', interpolation="nearest", cmap='gray')
        ax1.imshow(temp, origin='lower', interpolation="nearest",
                   cmap='viridis')
        ax1.set_title('Frame region')
        plt.axis('on')
        ax2 = plt.subplot(1, 2, 2)
        ax2.hist(values_region, bins=int(np.sqrt(values_region.shape[0])),
                 alpha=0.5, histtype='stepfilled')
        ax2.set_title('Histogram')
        ax2.tick_params(axis='x', labelsize=8)

        fig = plt.figure('Stats in annulus', figsize=vip_figsize)
        fig.subplots_adjust(hspace=0.15)
        ax1 = plt.subplot(3, 1, 1)
        lab1 = 'Mean value in {}'.format(region)
        ax1.plot(mean, '.-', label=lab1, lw=0.8, alpha=0.6)
        ax1.legend(loc=1, fancybox=True).get_frame().set_alpha(0.5)
        ax1.grid(True, alpha=0.2)
        plt.setp(ax1.get_xticklabels(), visible=False)

        ax2 = plt.subplot(3, 1, 2, sharex=ax1)
        lab2 = 'Px std dev in {}'.format(region)
        ax2.plot(std_dev, '.-', label=lab2, lw=0.8, alpha=0.6)
        ax2.legend(loc=1, fancybox=True).get_frame().set_alpha(0.5)
        ax2.grid(True, alpha=0.2)
        plt.setp(ax2.get_xticklabels(), visible=False)

        ax3 = plt.subplot(3, 1, 3, sharex=ax1)
        lab3 = 'Max value in {}'.format(region)
        ax3.plot(maxi, '.-', label=lab3, lw=0.8, alpha=0.6)
        ax3.legend(loc=1, fancybox=True).get_frame().set_alpha(0.5)
        ax3.grid(True, alpha=0.2)
        ax3.set_xlabel('Frame number')
        plt.show()

    if full_output:
        return mean, std_dev, median, maxi
    else:
        return mean