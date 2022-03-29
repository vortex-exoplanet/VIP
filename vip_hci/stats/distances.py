#! /usr/bin/env python

"""
Distance and correlation between images.
"""

__author__ = 'Carlos Alberto Gomez Gonzalez; V. Christiaens'
__all__ = ['cube_distance',
           'spectral_correlation']

from astropy.stats import gaussian_fwhm_to_sigma
import numpy as np
import scipy.stats
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt
try:
    # for skimage version >= '0.16' use skimage.metrics.structural_similarity
    from skimage.metrics import structural_similarity as ssim
except:
    # before skimage version '0.16' use skimage.measure.compare_ssim
    from skimage.measure import compare_ssim as ssim
from ..var import get_annulus_segments, get_circle
from ..config import vip_figsize


def cube_distance(array, frame, mode='full', dist='sad', inradius=None,
                  width=None, mask=None, plot=True):
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
    array : numpy ndarray
        Input cube or 3d array.
    frame : int, 2d array or None
        Reference frame in the cube or 2d array. If None, will take the median
        frame of the 3d array.
    mode : {'full','annulus', 'mask'}, string optional
        Whether to use the full frames, a centered annulus or a provided mask.
    dist : {'sad','euclidean','mse','pearson','spearman', 'ssim'}, str optional
        Which criterion to use.
    inradius : None or int, optional
        The inner radius when mode is 'annulus'.
    width : None or int, optional
        The width when mode is 'annulus'.
    mask: 2d array, optional
        If mode is 'mask', this is the mask within which the metrics is
        calculated in the images.
    plot : bool, optional
        Whether to plot the distances or not.

    Returns
    -------
    lista : numpy ndarray
        1d array of distances for each frame wrt the reference one.

    """
    if array.ndim != 3:
        raise TypeError('The input array is not a cube or 3d array')
    lista = []
    n = array.shape[0]
    if isinstance(frame, int):
        frame_ref = array[frame]
    elif isinstance(frame, np.ndarray):
        frame_ref = frame
    elif frame is None:
        frame_ref = np.median(array, axis=0)
    else:
        raise TypeError('Input ref frame format not recognized')
    if mode == 'full':
        pass
    elif mode == 'annulus':
        if inradius is None:
            raise ValueError('`Inradius` has not been set')
        if width is None:
            raise ValueError('`Width` has not been set')
        frame_ref = get_annulus_segments(frame_ref, inradius, width,
                                         mode="val")[0]
    elif mode == 'mask':
        if mask is None:
            raise ValueError('mask has not been set')
        frame_ref = frame_ref[np.where(mask)]
    else:
        raise TypeError('Mode not recognized or missing parameters')

    for i in range(n):
        if mode == 'full':
            framei = array[i]
        elif mode == 'annulus':
            framei = get_annulus_segments(array[i], inradius, width,
                                          mode="val")[0]
        elif mode == 'mask':
            framei = array[i][np.where(mask)]
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
                             data_range=frame_ref.max() - frame_ref.min(),
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



def spectral_correlation(array, ann_width=2, r_in=1, r_out=None, pl_xy=None,
                         mask_r=4, fwhm=4, sp_fwhm_guess=3,full_output=False):
    """ Computes the spectral correlation between (post-processed) IFS frames, 
    as a function of radius, implemented as Eq. 7 of Greco & Brandt 2017. This 
    is a crucial step for an unbias fit of a measured IFS spectrum to either 
    synthetic or template spectra.
    
    Parameters
    ----------
    array : numpy ndarray
        Input cube or 3d array, of dimensions n_ch x n_y x n_x; where n_y and 
        n_x should be odd values (star should be centered on central pixel).
    ann_width : int, optional
        Width in pixels of the concentric annuli used to compute the spectral 
        correlation as a function of radial separation. Greco & Brandt 2017 
        noted no significant differences for annuli between 1 and 3 pixels 
        width on GPI data.
    r_in: int, optional
        Innermost radius where the spectral correlation starts to be computed.
    r_out: int, optional
        Outermost radius where the spectral correlation is computed.If left as 
        None, it will automatically be computed up to the edge of the frame. 
    pl_xy: tuple of tuples of 2 floats, optional
        x,y coordiantes of all companions present in the images.
        If provided, a circle centered on the location of each
        companion will be masked out for the spectral correlation computation.
    mask_r: float, optional
        if pl_xy is provided, this should also be provided. Size of the 
        aperture around each companion (in terms of fwhm) that is discarded to 
        not bias the spectral correlation computation.
    fwhm: float, optional
        if pl_xy is provided, this should also be provided. By default we  
        consider a 2FWHM aperture mask around each companion to not bias the 
        spectral correlation computation.
    sp_fwhm_guess: float, optional
        Initial guess on the spectral FWHM of all channels.
    full_output: bool, opt
        Whether to also output the fitted spectral FWHM for each channel.
    Note: radii that are skipped will be filled with zeros in the output cube.

    Returns
    -------
    sp_corr : numpy ndarray
        3d array of spectral correlation, as a function of radius with 
        dimensions: n_r x n_ch x n_ch, where n_r = min((n_y-1)/2,(n_x-1)/2)
        Starts at r = 1 (not r=0) px.
    sp_fwhm: numpy ndarray
        (if full_output is True) 2d array containing the spectral fwhm at each 
        radius, for each spectral channel. Dims: n_r x n_ch
        
    """

    if not isinstance(ann_width,int) or not isinstance(r_in,int):
        raise TypeError("Inputs should be integers")

    if array.ndim != 3:
        raise TypeError("Input array should be 3D.")
        
    n_ch, n_y, n_x = array.shape
    n_r = min((n_y-1)/2.,(n_x-1)/2.)
    if n_r%1:
        raise TypeError("Input array y and x dimensions should be odd")
    
    if r_out is None:
        r_out = n_r

    test_rads = np.arange(r_in-1,r_out-1)
    n_rad = int(np.floor(test_rads.shape[0]/ann_width))
    
    #n_rad = int(np.ceil(n_r/ann_width)) # effective number of annuli probed
    
    sp_corr = np.zeros([int(n_r),n_ch,n_ch])
    if full_output:
        sp_fwhm = np.zeros([int(n_r),n_ch])
        def gauss_1fp(x, *p):
            sigma = p[0]*gaussian_fwhm_to_sigma
            return np.exp(-x**2/(2.*sigma**2))
    mask_final = np.zeros_like(array[0])

    if pl_xy is not None:
        mask = np.ones_like(array[0])
        for i in range(len(pl_xy)):
            if not isinstance(pl_xy[i], tuple):
                raise TypeError("Format of companions coordinates incorrect")
            mask_i = get_circle(mask, radius=mask_r*fwhm, cy=pl_xy[i][1], cx=pl_xy[i][0], mode="mask")
            mask_final[np.where(mask_i)] = 1

    for ann in range(n_rad):
        inner_radius = r_in+ (ann * ann_width)
        indices = get_annulus_segments(array[0], inner_radius, ann_width)
        yy = indices[0][0]
        xx = indices[0][1]
        yy_final = [yy[i] for i in range(len(indices[0][0])) if not mask_final[yy[i],xx[i]]]
        xx_final = [xx[i] for i in range(len(indices[0][0])) if not mask_final[yy[i],xx[i]]]
        matrix = array[:, yy_final, xx_final]  # shape (z, npx_annsegm)
        for zi in range(n_ch):
            for zj in range(n_ch):
                num = np.nanmean(matrix[zi]*matrix[zj])
                denom = np.sqrt(np.nanmean(matrix[zi]*matrix[zi])* \
                                np.nanmean(matrix[zj]*matrix[zj]))
                sp_corr[r_in+ann*ann_width:r_in+(ann+1)*ann_width,zi,zj] = num/denom
            if full_output:
                p0 = (sp_fwhm_guess,)
                x = np.arange(n_ch)-zi
                y = sp_corr[r_in+ann*ann_width,zi]# norm y
                y = y-np.amin(y)
                y = y/np.amax(y)
                coeff, var_matrix = curve_fit(gauss_1fp, x, y, p0=p0)
                sp_fwhm[r_in+ann*ann_width:r_in+(ann+1)*ann_width,zi] = coeff[0]

                
    if full_output:
        return sp_corr, sp_fwhm
    else:
        return sp_corr
    