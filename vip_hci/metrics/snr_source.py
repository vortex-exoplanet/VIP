#! /usr/bin/env python

"""
Module with S/N calculation functions.
We strongly recommend users to read Mawet et al. (2014) before using routines 
of this module: https://ui.adsabs.harvard.edu/abs/2014ApJ...792...97M/abstract
"""

__author__ = 'Carlos Alberto Gomez Gonzalez, O. Absil @ ULg, V. Christiaens'
__all__ = ['snr',
           'snrmap',
           'significance',
           'frame_report']

import numpy as np
import photutils
from scipy.stats import norm, t
from hciplot import plot_frames
from skimage.draw import disk, circle_perimeter
from matplotlib import pyplot as plt
from astropy.convolution import convolve, Tophat2DKernel
from astropy.stats import median_absolute_deviation as mad
from multiprocessing import cpu_count
from ..config.utils_conf import pool_map, iterable, sep
from ..config import time_ini, timing, check_array
from ..var import get_annulus_segments, frame_center, dist


def snrmap(array, fwhm, approximated=False, plot=False, known_sources=None,
           nproc=None, array2=None, use2alone=False, 
           exclude_negative_lobes=False, verbose=True, **kwargs):
    """Parallel implementation of the S/N map generation function. Applies the
    S/N function (small samples penalty) at each pixel.
    
    The S/N is computed as in Mawet et al. (2014) for each radial separation.    
    https://ui.adsabs.harvard.edu/abs/2014ApJ...792...97M/abstract
    
    *** DISCLAIMER ***
    Signal-to-noise ratio is not significance! For a conversion from snr to 
    n-sigma (i.e. the equivalent confidence level of a Gaussian n-sigma), use 
    the significance() function.    
    
    
    Parameters
    ----------
    array : numpy ndarray
        Input frame (2d array).
    fwhm : float
        Size in pixels of the FWHM.
    approximated : bool, optional
        If True, an approximated S/N map is generated.
    plot : bool, optional
        If True plots the S/N map. False by default.
    known_sources : None, tuple or tuple of tuples, optional
        To take into account existing sources. It should be a tuple of float/int
        or a tuple of tuples (of float/int) with the coordinate(s) of the known
        sources.
    nproc : int or None
        Number of processes for parallel computing.
    array2 : numpy ndarray, optional
        Additional image (e.g. processed image with negative derotation angles) 
        enabling to have more noise samples. Should have the 
        same dimensions as array.
    use2alone: bool, optional
        Whether to use array2 alone to estimate the noise (might be useful to 
        estimate the snr of extended disk features).
    verbose: bool, optional
        Whether to print timing or not.
    **kwargs : dictionary, optional
        Arguments to be passed to ``plot_frames`` to customize the plot (and to
        save it to disk).

    Returns
    -------
    snrmap : 2d numpy ndarray
        Frame with the same size as the input frame with each pixel.
    """
    if verbose:
        start_time = time_ini()

    check_array(array, dim=2, msg='array')
    sizey, sizex = array.shape
    snrmap_array = np.zeros_like(array)
    width = min(sizey, sizex) / 2 - 1.5 * fwhm
    mask = get_annulus_segments(array, (fwhm / 2) + 2, width, mode="mask")[0]
    mask = np.ma.make_mask(mask)
    # by making a bool mask *after* applying the mask to the array, we also mask
    # out zero values from the array. This logic cannot be simplified by using
    # mode="ind"!
    yy, xx = np.where(mask)
    coords = zip(xx, yy)

    if nproc is None:
        nproc = cpu_count() // 2        # Hyper-threading doubles the # of cores

    if known_sources is None:

        # proxy to S/N calculation
        if approximated:
            cy, cx = frame_center(array)
            tophat_kernel = Tophat2DKernel(fwhm / 2)
            array = convolve(array, tophat_kernel)
            width = min(sizey, sizex) / 2 - 1.5 * fwhm
            mask = get_annulus_segments(array, (fwhm / 2) + 1, width - 1,
                                        mode="mask")[0]
            mask = np.ma.make_mask(mask)
            yy, xx = np.where(mask)
            coords = [(int(x), int(y)) for (x, y) in zip(xx, yy)]
            res = pool_map(nproc, _snr_approx, array, iterable(coords), fwhm,
                           cy, cx)
            res = np.array(res, dtype=object)
            yy = res[:, 0]
            xx = res[:, 1]
            snr_value = res[:, 2]
            snrmap_array[yy.astype(int), xx.astype(int)] = snr_value

        # computing s/n map with Mawet+14 definition
        else:
            res = pool_map(nproc, snr, array, iterable(coords), fwhm, True,
                           array2, use2alone, exclude_negative_lobes)
            res = np.array(res, dtype=object)
            yy = res[:, 0]
            xx = res[:, 1]
            snr_value = res[:, -1]
            snrmap_array[yy.astype('int'), xx.astype('int')] = snr_value

    # masking known sources
    else:
        if not isinstance(known_sources, tuple):
            raise TypeError("`known_sources` must be a tuple or tuple of "
                            "tuples")
        else:
            source_mask = np.zeros_like(array)
            if isinstance(known_sources[0], tuple):
                for coor in known_sources:
                    source_mask[coor[::-1]] = 1
            elif isinstance(known_sources[0], int):
                source_mask[known_sources[1], known_sources[0]] = 1
            else:
                raise TypeError("`known_sources` seems to have wrong type. It "
                                "must be a tuple of ints or tuple of tuples "
                                "(of ints)")

        # checking the mask with the sources
        if source_mask[source_mask == 1].shape[0] > 50:
            msg = 'Input source mask is too crowded (check its validity)'
            raise RuntimeError(msg)

        soury, sourx = np.where(source_mask == 1)
        sources = []
        coor_ann = []
        arr_masked_sources = array.copy()
        centery, centerx = frame_center(array)
        for y, x in zip(soury, sourx):
            radd = dist(centery, centerx, int(y), int(x))
            if int(radd) < centery - np.ceil(fwhm):
                sources.append((y, x))

        for source in sources:
            y, x = source
            radd = dist(centery, centerx, int(y), int(x))
            anny, annx = get_annulus_segments(array, int(radd-fwhm),
                                              int(np.round(3 * fwhm)))[0]

            ciry, cirx = disk((y, x), int(np.ceil(fwhm)))
            # masking the sources positions (using the MAD of pixels in annulus)
            arr_masked_sources[ciry, cirx] = mad(array[anny, annx])

            # S/Ns of annulus without the sources
            coor_ann = [(x, y) for (x, y) in zip(annx, anny) if (x, y) not in
                        zip(cirx, ciry)]
            res = pool_map(nproc, snr, arr_masked_sources, iterable(coor_ann),
                           fwhm, True, array2, use2alone, 
                           exclude_negative_lobes)
            res = np.array(res, dtype=object)
            yy_res = res[:, 0]
            xx_res = res[:, 1]
            snr_value = res[:, 4]
            snrmap_array[yy_res.astype('int'), xx_res.astype('int')] = snr_value
            coor_ann += coor_ann

        # S/Ns of the rest of the frame without the annulus
        coor_rest = [(x, y) for (x, y) in zip(xx, yy) if (x, y) not in coor_ann]
        res = pool_map(nproc, snr, array, iterable(coor_rest), fwhm, True,
                       array2, use2alone, exclude_negative_lobes)
        res = np.array(res, dtype=object)
        yy_res = res[:, 0]
        xx_res = res[:, 1]
        snr_value = res[:, 4]
        snrmap_array[yy_res.astype('int'), xx_res.astype('int')] = snr_value

    if plot:
        plot_frames(snrmap_array, colorbar=True, title='S/N map', **kwargs)

    if verbose:
        print("S/N map created using {} processes".format(nproc))
        timing(start_time)
    return snrmap_array


def _snr_approx(array, source_xy, fwhm, centery, centerx):
    """
    array - frame convolved with top hat kernel
    """
    sourcex, sourcey = source_xy
    rad = dist(centery, centerx, sourcey, sourcex)
    ind_aper = disk((sourcey, sourcex), fwhm/2.)
    # noise : STDDEV in convolved array of 1px wide annulus (while
    # masking the flux aperture) * correction of # of resolution elements
    ind_ann = circle_perimeter(int(centery), int(centerx), int(rad))
    array2 = array.copy()
    array2[ind_aper] = mad(array[ind_ann])  # mask
    n2 = (2 * np.pi * rad) / fwhm - 1
    noise = array2[ind_ann].std(ddof=1) * np.sqrt(1+(1/n2))
    # signal : central px minus the mean of the pxs (masked) in 1px annulus
    signal = array[sourcey, sourcex] - array2[ind_ann].mean()
    snr_value = signal / noise
    return sourcey, sourcex, snr_value


def snr(array, source_xy, fwhm, full_output=False, array2=None, use2alone=False,
        exclude_negative_lobes=False, plot=False, verbose=False):
    """
    Calculate the S/N (signal to noise ratio) of a test resolution element
    in a residual frame (e.g. post-processed with LOCI, PCA, etc). Implements
    the approach described in Mawet et al. 2014 on small sample statistics,
    where a student t-test (eq. 9) can be used to determine S/N (and contrast)
    in high contrast imaging. 3 extra possibilities compared to Mawet et al. 
    2014 (https://ui.adsabs.harvard.edu/abs/2014ApJ...792...97M/abstract):

        * possibility to provide a second array (e.g. obtained with opposite \
        derotation angles) to have more apertures for noise estimation;
            
        * possibility to exclude negative ADI lobes directly adjacent to the \
        tested xy location, to not bias the noise estimate;
        
        * possibility to use only the second array for the noise estimation \
        (useful for images containing a lot of disk/extended signals).
        
    *** DISCLAIMER ***
    Signal-to-noise ratio is not significance! For a conversion from snr to 
    n-sigma (i.e. the equivalent confidence level of a Gaussian n-sigma), use 
    the significance() function.    
    
    Parameters
    ----------
    array : numpy ndarray, 2d
        Post-processed frame where we want to measure S/N.
    source_xy : tuple of floats
        X and Y coordinates of the planet or test speckle.
    fwhm : float
        Size in pixels of the FWHM.
    full_output : bool, optional
        If True returns back the S/N value, the y, x input coordinates, noise
        and flux.
    array2 : None or numpy ndarray, 2d, optional
        Additional image (e.g. processed image with negative derotation angles) 
        enabling to have more apertures for noise estimation at each radial
        separation. Should have the same dimensions as array.
    use2alone : bool, opt
        Whether to use array2 alone to estimate the noise (can be useful to 
        estimate the S/N of extended disk features)
    exclude_negative_lobes : bool, opt
        Whether to include the adjacent aperture lobes to the tested location 
        or not. Can be set to True if the image shows significant neg lobes.
    plot : bool, optional
        Plots the frame and the apertures considered for clarity. 
    verbose: bool, optional
        Chooses whether to print some output or not.

    Returns
    -------
    [if full_output=True:]
    sourcey : numpy ndarray
        [full_output=True] Input coordinates (``source_xy``) in Y.
    sourcex : numpy ndarray
        [full_output=True] Input coordinates (``source_xy``) in X.
    f_source : float
        [full_output=True] Flux in test elemnt.
    fluxes : numpy ndarray
        [full_output=True] Background apertures fluxes.
    [always:]    
    snr_vale : float
        Value of the S/N for the given test resolution element.
    """
    check_array(array, dim=2, msg='array')
    if not isinstance(source_xy, tuple):
        raise TypeError("`source_xy` must be a tuple of floats")
    if array2 is not None:
        if not array2.shape == array.shape:
            raise TypeError('`array2` has not the same shape as input array')

    sourcex, sourcey = source_xy
    centery, centerx = frame_center(array)
    sep = dist(centery, centerx, float(sourcey), float(sourcex))

    if not sep > (fwhm/2)+1:
        raise RuntimeError('`source_xy` is too close to the frame center')

    sens = 'clock'  # counterclock

    angle = np.arcsin(fwhm/2./sep)*2
    number_apertures = int(np.floor(2*np.pi/angle))
    yy = np.zeros((number_apertures))
    xx = np.zeros((number_apertures))
    cosangle = np.cos(angle)
    sinangle = np.sin(angle)
    xx[0] = sourcex - centerx
    yy[0] = sourcey - centery
    for i in range(number_apertures-1):
        if sens == 'clock':
            xx[i+1] = cosangle*xx[i] + sinangle*yy[i] 
            yy[i+1] = cosangle*yy[i] - sinangle*xx[i] 
        elif sens == 'counterclock':
            xx[i+1] = cosangle*xx[i] - sinangle*yy[i] 
            yy[i+1] = cosangle*yy[i] + sinangle*xx[i]

    xx += centerx
    yy += centery
    rad = fwhm/2.
    if exclude_negative_lobes:
        xx = np.concatenate(([xx[0]], xx[2:-1]))
        yy = np.concatenate(([yy[0]], yy[2:-1]))

    apertures = photutils.CircularAperture(zip(xx, yy), r=rad)  # Coordinates (X,Y)
    fluxes = photutils.aperture_photometry(array, apertures, method='exact')
    fluxes = np.array(fluxes['aperture_sum'])

    if array2 is not None:
        fluxes2 = photutils.aperture_photometry(array2, apertures,
                                                method='exact')
        fluxes2 = np.array(fluxes2['aperture_sum'])
        if use2alone:
            fluxes = np.concatenate(([fluxes[0]], fluxes2[:]))
        else:
            fluxes = np.concatenate((fluxes, fluxes2))

    f_source = fluxes[0].copy()
    fluxes = fluxes[1:]
    n2 = fluxes.shape[0]
    backgr_apertures_std = fluxes.std(ddof=1)
    snr_vale = (f_source - fluxes.mean())/(backgr_apertures_std *
                                           np.sqrt(1+(1/n2)))

    if verbose:
        msg1 = 'S/N for the given pixel = {:.3f}'
        msg2 = 'Integrated flux in FWHM test aperture = {:.3f}'
        msg3 = 'Mean of background apertures integrated fluxes = {:.3f}'
        msg4 = 'Std-dev of background apertures integrated fluxes = {:.3f}'
        print(msg1.format(snr_vale))
        print(msg2.format(f_source))
        print(msg3.format(fluxes.mean()))
        print(msg4.format(backgr_apertures_std))

    if plot:
        _, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(array, origin='lower', interpolation='nearest', alpha=0.5, 
                  cmap='gray')
        for i in range(xx.shape[0]):
            # Circle takes coordinates as (X,Y)
            aper = plt.Circle((xx[i], yy[i]), radius=fwhm/2., color='r',
                              fill=False, alpha=0.8)
            ax.add_patch(aper)
            cent = plt.Circle((xx[i], yy[i]), radius=0.8, color='r', fill=True,
                              alpha=0.5)
            ax.add_patch(cent)
            aper_source = plt.Circle((sourcex, sourcey), radius=0.7,
                                     color='b', fill=True, alpha=0.5)
            ax.add_patch(aper_source)
        ax.grid(False)
        plt.show()

    if full_output:
        return sourcey, sourcex, f_source, fluxes, snr_vale
    else:
        return snr_vale



def significance(snr, rad, fwhm, student_to_gauss=True):
    """ Converts a S/N ratio (measured as in Mawet et al. 2014) into the 
    equivalent gaussian significance, i.e. the n-sigma with the same confidence 
    level as the S/N at the given separation.
     
     
    Parameters
    ----------
    snr : float or numpy array
        SNR value(s)
    rad : float or numpy array
        Radial separation(s) from the star in pixels. If an array, it should be
        the same shape as snr and provide the radial separation corresponding
        to each snr measurement.
    fwhm : float
        Full Width Half Maximum of the PSF.
    student_to_gauss : bool, optional
        Whether the conversion is from Student SNR to Gaussian significance. If 
        False, will assume the opposite: Gaussian significance to Student SNR.
    
    Returns
    -------
    sigma : float
        Gaussian significance in terms of n-sigma
    
    """
    
    if student_to_gauss:
        sigma = norm.ppf(t.cdf(snr,(rad/fwhm)*2*np.pi-2))
        if t.cdf(snr,(rad/fwhm)*2*np.pi-2)==1.0:
            print("Warning high S/N! cdf>0.9999999999999999 is rounded to 1")
            print(r"Returning 8.2 instead of inf. Quote detection>8.2 $\sigma$")
            return 8.2
    else:
        sigma = t.ppf(norm.cdf(snr), (rad/fwhm)*2*np.pi-2)
        
    return sigma      
            

def frame_report(array, fwhm, source_xy=None, verbose=True):
    """ Gets information from given frame: Integrated flux in aperture, S/N of
    central pixel (either ``source_xy`` or max value), mean S/N in aperture.

    Parameters
    ----------
    array : numpy ndarray
        2d array or input frame.
    fwhm : float
        Size of the FWHM in pixels.
    source_xy : tuple of floats or list (of tuples of floats)
        X and Y coordinates of the center(s) of the source(s).
    verbose : bool, optional
        If True prints to stdout the frame info.

    Returns
    -------
    source_xy : tuple of floats or list (of tuples of floats)
        X and Y coordinates of the center(s) of the source(s).
    obj_flux : list of floats
        Integrated flux in aperture.
    snr_centpx : list of floats
        S/N of the ``source_xy`` pixels.
    meansnr_pixels : list of floats
        Mean S/N of pixels in 1xFWHM apertures centered on ``source_xy``.
    """
    if array.ndim != 2:
        raise TypeError('Array is not 2d.')

    obj_flux = []
    meansnr_pixels = []
    snr_centpx = []

    if source_xy is not None:
        if isinstance(source_xy, (list, tuple)):
            if not isinstance(source_xy[0], tuple):
                source_xy = [source_xy]
        else:
            raise TypeError("`source_xy` must be a tuple of floats or tuple "
                            "of tuples")

        for xy in source_xy:
            x, y = xy
            if verbose:
                print(sep)
                print('Coords of chosen px (X,Y) = {:.1f}, {:.1f}'.format(x, y))

            # we get integrated flux on aperture with diameter=1FWHM
            aper = photutils.CircularAperture((x, y), r=fwhm / 2.)
            obj_flux_i = photutils.aperture_photometry(array, aper,
                                                       method='exact')
            obj_flux_i = obj_flux_i['aperture_sum'][0]

            # we get the mean and stddev of SNRs on aperture
            yy, xx = disk((y, x), fwhm / 2)
            snr_pixels_i = [snr(array, (x_, y_), fwhm, plot=False,
                                verbose=False) for y_, x_ in zip(yy, xx)]
            meansnr_i = np.mean(snr_pixels_i)
            stdsnr_i = np.std(snr_pixels_i,ddof=1)
            pxsnr_i = snr(array, (x, y), fwhm, plot=False, verbose=False)

            obj_flux.append(obj_flux_i)
            meansnr_pixels.append(meansnr_i)
            snr_centpx.append(pxsnr_i)

            if verbose:
                msg0 = 'Flux in a centered 1xFWHM circular aperture = {:.3f}'
                print(msg0.format(obj_flux_i))
                print('Central pixel S/N = {:.3f}'.format(pxsnr_i))
                print(sep)
                print('Inside a centered 1xFWHM circular aperture:')
                msg1 = 'Mean S/N (shifting the aperture center) = {:.3f}'
                print(msg1.format(meansnr_i))
                msg2 = 'Max S/N (shifting the aperture center) = {:.3f}'
                print(msg2.format(np.max(snr_pixels_i)))
                msg3 = 'stddev S/N (shifting the aperture center) = {:.3f}'
                print(msg3.format(stdsnr_i))
                print('')

    else:
        y, x = np.where(array == array.max())
        y, x = y[0], x[0]  # assuming there is only one max, taking 1st if clump
        source_xy = (x, y)
        if verbose:
            print(sep)
            print('Coords of Max px (X,Y) = {:.1f}, {:.1f}'.format(x, y))

        # we get integrated flux on aperture with diameter=1FWHM
        aper = photutils.CircularAperture((x, y), r=fwhm / 2.)
        obj_flux_i = photutils.aperture_photometry(array, aper, method='exact')
        obj_flux_i = obj_flux_i['aperture_sum'][0]

        # we get the mean and stddev of SNRs on aperture
        yy, xx = disk((y, x), fwhm / 2.)
        snr_pixels_i = [snr(array, (x_, y_), fwhm, plot=False,
                            verbose=False) for y_, x_ in zip(yy, xx)]
        meansnr_pixels = np.mean(snr_pixels_i)
        stdsnr_i = np.std(snr_pixels_i,ddof=1)
        pxsnr_i = snr(array, (x, y), fwhm, plot=False, verbose=False)

        obj_flux.append(obj_flux_i)
        snr_centpx.append(pxsnr_i)

        if verbose:
            msg0 = 'Flux in a centered 1xFWHM circular aperture = {:.3f}'
            print(msg0.format(obj_flux_i))
            print('Central pixel S/N = {:.3f}'.format(pxsnr_i))
            print(sep)
            print('Inside a centered 1xFWHM circular aperture:')
            msg1 = 'Mean S/N (shifting the aperture center) = {:.3f}'
            print(msg1.format(meansnr_pixels))
            msg2 = 'Max S/N (shifting the aperture center) = {:.3f}'
            print(msg2.format(np.max(snr_pixels_i)))
            msg3 = 'stddev S/N (shifting the aperture center) = {:.3f}'
            print(msg3.format(stdsnr_i))
            print(sep)

    return source_xy, obj_flux, snr_centpx, meansnr_pixels
