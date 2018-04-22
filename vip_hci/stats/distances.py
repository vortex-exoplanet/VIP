#! /usr/bin/env python

"""
Distance between images.
"""

from __future__ import division, print_function

__author__ = 'Carlos Alberto Gomez Gonzalez'
__all__ = ['cube_distance']

import numpy as np
import scipy.stats
from matplotlib import pyplot as plt
from ..var import get_annulus
from skimage.measure import compare_ssim as ssim


def cube_distance(array, frame, mode='full', dist='sad', inradius=None,
                  width=None, plot=True):
    """ Computes the distance (or similarity) between frames in a cube, using
    one as the reference (it can be either a frame from the same cube or a
    separate 2d array). Depending on the mode, the whole image can be used,
    or just the pixels in a given annulus. The criteria used are:
    - the Manhattan distance (SAD or sum of absolute differences),
    - the Euclidean distance (square root of the sum of the squared differences),
    - the Mean Squared Error,
    - the Spearman correlation coefficient,
    - the Pearson correlation coefficient,
    - the Structural Similarity Index (SSIM).

    The SAD, MSE and Ecuclidean criteria are dissimilarity criteria, which
    means that 0 is perfect similarity.
    The Spearman and Pearson correlation coefficients, vary between -1 and +1
    with 0 implying no correlation. Correlations of -1 or +1 imply an exact
    linear relationship.
    The Structural Similarity Index was proposed by Wang et al. 2004.
    (http://www.cns.nyu.edu/pub/eero/wang03-reprint.pdf)
    SSIM varies between -1 and 1, where 1 means perfect similarity. SSIM
    attempts to model the perceived change in the structural information of the
    image. The mean SSIM is reported.

    Parameters
    ----------
    array : array_like
        Input cube or 3d array.
    frame : int or 2d array
        Reference frame in the cube or 2d array.
    mode : {'full','annulus'}, string optional
        Whether to use the full frames or a centered annulus.
    dist : {'sad','euclidean','mse','pearson','spearman', 'ssim'}, str optional
        Which criterion to use.
    inradius : None or int, optional
        The inner radius when mode is 'annulus'.
    width : None or int, optional
        The width when mode is 'annulus'.
    plot : bool, optional
        Whether to plot the distances or not.

    Returns
    -------
    lista : array_like
        1d array of distances for each frame wrt the reference one.

    """
    if array.ndim != 3:
        raise TypeError('The input array is not a cube or 3d array')
    lista = []
    n = array.shape[0]
    if mode == 'full':
        if isinstance(frame, int):
            frame_ref = array[frame]
        elif isinstance(frame, np.ndarray):
            frame_ref = frame
    elif mode == 'annulus':
        if inradius is None:
            raise ValueError('`Inradius` has not been set')
        if width is None:
            raise ValueError('`Width` has not been set')
        if isinstance(frame, int):
            frame_ref = array[frame]
        elif isinstance(frame, np.ndarray):
            frame_ref = frame
        frame_ref = get_annulus(frame_ref, inradius, width, output_values=True)
    else:
        raise TypeError('Mode not recognized or missing parameters')

    for i in range(n):
        if mode == 'full':
            framei = array[i]
        elif mode == 'annulus':
            framei = get_annulus(array[i], inradius, width, True)

        if dist == 'sad':
            lista.append(np.sum(abs(frame_ref - framei)))
        elif dist == 'euclidean':
            lista.append(np.sqrt(np.sum((frame_ref - framei)**2)))
        elif dist == 'mse':
            lista.append((np.sum((frame_ref - framei)**2))/len(frame_ref))
        elif dist == 'pearson':
            pears, _ = scipy.stats.pearsonr(frame_ref.ravel(), framei.ravel())
            lista.append(pears)
        elif dist == 'spearman':
            spear, _ = scipy.stats.spearmanr(frame_ref.ravel(), framei.ravel())
            lista.append(spear)
        elif dist == 'ssim':
            mean_ssim = ssim(frame_ref, framei, win_size=7,
                             dynamic_range=frame_ref.max() - frame_ref.min(),
                             gaussian_weights=True, sigma=1.5,
                             use_sample_covariance=True)
            lista.append(mean_ssim)
        else:
            raise ValueError('Distance not recognized')
    lista = np.array(lista)

    median_cor = np.median(lista)
    mean_cor = np.mean(lista)
    if plot:
        _, ax = plt.subplots(figsize=vip_figsize)

        if isinstance(frame, int):
            ax.vlines(frame, ymin=np.nanmin(lista), ymax=np.nanmax(lista),
                      colors='green', linestyles='dashed', lw=2, alpha=0.8,
                      label='Frame '+str(frame))
        ax.hlines(median_cor, xmin=-1, xmax=n+1, colors='purple', alpha=0.3,
                  linestyles='dashed', label='Median value : '+str(median_cor))
        ax.hlines(mean_cor, xmin=-1, xmax=n+1, colors='green', alpha=0.3,
                  linestyles='dashed', label='Mean value : '+str(mean_cor))

        x = range(len(lista))
        ax.plot(x, lista, '-', alpha=0.6, color='#1f77b4')
        ax.plot(x, lista, 'o', alpha=0.4, color='#1f77b4')
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
        plt.minorticks_on()
        plt.legend(fancybox=True, framealpha=0.5, fontsize=12, loc='best')
        plt.grid(which='major', alpha=0.2)

    return lista

