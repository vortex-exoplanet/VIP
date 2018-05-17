#! /usr/bin/env python

"""
Module with S/N calculation functions.
"""

from __future__ import division, print_function

__author__ = 'Carlos Alberto Gomez Gonzalez, O. Absil @ ULg'
__all__ = ['snr_ss',
           'snr_peakstddev',
           'snrmap',
           'snrmap_fast']

import numpy as np
import itertools as itt
import photutils
from skimage import draw
from matplotlib import pyplot as plt
from astropy.convolution import convolve, Tophat2DKernel
from astropy.stats import median_absolute_deviation as mad
from multiprocessing import Pool, cpu_count
from ..conf.utils_conf import eval_func_tuple as EFT
from ..conf import time_ini, timing
from ..var import get_annulus, frame_center, dist, pp_subplots


def snrmap(array, fwhm, plot=False, mode='sss', source_mask=None, nproc=None,
           save_plot=None, plot_title=None, verbose=True):
    """Parallel implementation of the S/N map generation function. Applies the
    S/N function (small samples penalty) at each pixel.
    
    Parameters
    ----------
    array : array_like
        Input frame.
    fwhm : float
        Size in pixels of the FWHM.
    plot : bool, optional
        If True plots the S/N map. False by default.
    mode : {'sss', 'peakstddev'}, string optional
        'sss' uses the approach with the small sample statistics penalty and
        'peakstddev' uses the peak(aperture)/std(annulus) version.
    source_mask : array_like, optional
        If exists, it takes into account existing sources. The mask is a ones
        2d array, with the same size as the input frame. The centers of the 
        known sources have a zero value.
    nproc : int or None
        Number of processes for parallel computing.
    save_plot : string
        If provided, the S/N map is saved to this path.
    plot_title : string
        If provided, the S/N map plot is titled.
    verbose: bool, optional
        Whether to print timing or not.
    
    Returns
    -------
    snrmap : array_like
        Frame with the same size as the input frame with each pixel.
    """
    if verbose:
        start_time = time_ini()
    if array.ndim != 2:
        raise TypeError('Input array is not a 2d array or image.')
    if plot:  plt.close('snr')
        
    sizey, sizex = array.shape
    snrmap = np.zeros_like(array)
    width = min(sizey, sizex) / 2 - 1.5 * fwhm
    mask = get_annulus(array, (fwhm / 2) + 1, width)
    mask = np.ma.make_mask(mask)
    yy, xx = np.where(mask)
    coords = zip(xx, yy)
        
    if nproc is None:
        nproc = cpu_count() // 2        # Hyper-threading doubles the # of cores
    
    if mode == 'sss':
        func = snr_ss
    elif mode == 'peakstddev':
        func = snr_peakstddev
    else:
        raise TypeError('\nMode not recognized.')
    
    if source_mask is None:
        pool = Pool(processes=nproc)                                        
        res = pool.map(EFT, zip(itt.repeat(func),itt.repeat(array), coords,
                                itt.repeat(fwhm), itt.repeat(True)))
        res = np.array(res)
        pool.close()
        yy = res[:, 0]
        xx = res[:, 1]
        snr = res[:, 2]
        snrmap[yy.astype('int'), xx.astype('int')] = snr
    else:
        # checking the mask with the sources
        if array.shape != source_mask.shape:
            raise RuntimeError('Source mask has wrong size.')
        if source_mask[source_mask == 0].shape[0] == 0:
            msg = 'Input source mask is empty.'
            raise RuntimeError(msg)
        if source_mask[source_mask == 0].shape[0] > 20:
            msg = 'Input source mask is too crowded. Check its validity.'
            raise RuntimeError(msg)
        
        soury, sourx = np.where(source_mask == 0)
        sources = []
        ciry = []; cirx = []; anny = []; annx = []
        array_sources = array.copy()
        centery, centerx = frame_center(array)
        for (y,x) in zip(soury,sourx):
            radd = dist(centery, centerx, y, x)
            if int(radd) < centery - np.ceil(fwhm):
                sources.append((y,x))
        
        for source in sources:
            y, x = source        
            radd = dist(centery, centerx, y, x)
            tempay, tempax = get_annulus(array, int(radd-fwhm), 
                                         int(np.ceil(2*fwhm)),
                                         output_indices=True)
            tempcy, tempcx = draw.circle(y, x, int(np.ceil(1*fwhm)))
            # masking the source position (using the MAD of pixels in annulus)
            array_sources[tempcy, tempcx] = mad(array[tempay, tempax])
            ciry += list(tempcy)
            cirx += list(tempcx)
            anny += list(tempay)
            annx += list(tempax)

        # coordinates of annulus without the sources
        coor_ann = [(y,x) for (y,x) in zip(anny, annx)
                    if (y,x) not in zip(ciry, cirx)]

        # coordinates of the rest of the frame without the annulus
        coor_rest = [(y,x) for (y,x) in zip(yy, xx) if (y,x) not in coor_ann]
        
        pool1 = Pool(processes=nproc)
        res = pool1.map(EFT, zip(itt.repeat(func),itt.repeat(array), coor_rest,
                                 itt.repeat(fwhm), itt.repeat(True)))
        res = np.array(res)
        pool1.close()
        yy = res[:,0]
        xx = res[:,1]
        snr = res[:,2]
        snrmap[yy.astype('int'), xx.astype('int')] = snr
        
        pool2 = Pool(processes=nproc)
        res = pool2.map(EFT, zip(itt.repeat(func), itt.repeat(array_sources),
                                 coor_ann, itt.repeat(fwhm), itt.repeat(True)))
        res = np.array(res)
        pool2.close()
        yy = res[:,0]
        xx = res[:,1]
        snr = res[:,2]
        snrmap[yy.astype('int'), xx.astype('int')] = snr
    
    if plot:  
        pp_subplots(snrmap, colorb=True, title='S/N map')

    # Option to save snrmap in angular scale, using Keck NIRC2's ~0.01 pixel
    # scale. In this case, set plot = False
    elif save_plot is not None:
        pp_subplots(snrmap, colorb=True, title=plot_title, save=save_plot,
                    vmin=-1, vmax=5, angscale=True, getfig = True)

    if verbose:
        print("S/N map created using {} processes.".format(nproc))
        timing(start_time)
    return snrmap
   
   
def snrmap_fast(array, fwhm, nproc=None, plot=False, verbose=True):
    """ Approximated S/N map generation. To be used as a quick proxy of the
    S/N map generated using the small samples statistics definition.
    
    Parameters
    ----------
    array : array_like
        Input frame.
    fwhm : float
        Size in pixels of the FWHM.
    nproc : int or None
        Number of processes for parallel computing.
    plot : bool, optional
        If True plots the S/N map.
    verbose: bool, optional
        Whether to print timing or not.
    
    Returns
    -------
    snrmap : array_like
        Frame with the same size as the input frame with each pixel.
    """       
    if verbose:
        start_time = time_ini()
    if array.ndim != 2:
        raise TypeError('Input array is not a 2d array or image.')
    
    cy, cx = frame_center(array)
    tophat_kernel = Tophat2DKernel(fwhm/2)
    array = convolve(array, tophat_kernel)
            
    sizey, sizex = array.shape
    snrmap = np.zeros_like(array)
    width = min(sizey,sizex)/2 - 1.5*fwhm    
    mask = get_annulus(array, (fwhm/2)+1, width-1)
    mask = np.ma.make_mask(mask)
    yy, xx = np.where(mask)
    coords = [(x, y) for (x, y) in zip(xx, yy)]
    
    if nproc is None:  
        nproc = cpu_count() // 2        # Hyper-threading doubles the # of cores
    
    if nproc == 1:
        for y,x in zip(yy, xx):
            snrmap[y,x] = _snr_approx(array, (x, y), fwhm, cy, cx)[2]
    elif nproc > 1:
        pool = Pool(processes=nproc)                                        
        res = pool.map(EFT, zip(itt.repeat(_snr_approx), itt.repeat(array),
                                coords,itt.repeat(fwhm), itt.repeat(cy),
                                itt.repeat(cx)))
        pool.close()
        res = np.array(res)
        yy = res[:, 0]
        xx = res[:, 1]
        snr = res[:, 2]
        snrmap[yy.astype(int), xx.astype(int)] = snr
        
    if plot:
        pp_subplots(snrmap, colorb=True, title='SNRmap')
     
    if verbose:    
        print("S/N map created using {} processes.".format(nproc))
        timing(start_time)
    return snrmap


def _snr_approx(array, source_xy, fwhm, centery, centerx):
    """
    array - frame convolved with top hat kernel 
    """
    sourcex, sourcey = source_xy
    rad = dist(centery, centerx, sourcey, sourcex)
    ind_aper = draw.circle(sourcey, sourcex, fwhm/2.)
    # noise : STDDEV in convolved array of 1px wide annulus (while 
    # masking the flux aperture) * correction of # of resolution elements
    ind_ann = draw.circle_perimeter(int(centery), int(centerx), int(rad))
    array2 = array.copy()
    array2[ind_aper] = array[ind_ann].mean()   # quick-n-dirty mask
    n2 = (2*np.pi*rad)/fwhm - 1
    noise = array2[ind_ann].std()*np.sqrt(1+(1/n2))
    # signal : central px minus the mean of the pxs (masked) in 1px annulus
    signal = array[sourcey, sourcex] - array2[ind_ann].mean()
    snr = signal / noise
    return sourcey, sourcex, snr
    

# Leave the order of parameters as it is, the same for both snr functions
# to be compatible with the snrmap parallel implementation
def snr_ss(array, source_xy, fwhm, out_coor=False, plot=False, verbose=False,
           full_output=False):
    """Calculates the S/N (signal to noise ratio) of a test resolution element
    in a residual frame (e.g. post-processed with LOCI, PCA, etc). Uses the
    approach described in Mawet et al. 2014 on small sample statistics, where a
    student t-test (eq. 9) can be used to determine S/N (and contrast) in high
    contrast imaging.
    
    Parameters
    ----------
    array : array_like, 2d
        Post-processed frame where we want to measure S/N.
    source_xy : tuple of floats
        X and Y coordinates of the planet or test speckle.
    fwhm : float
        Size in pixels of the FWHM.
    out_coor : bool, optional
        If True returns back the S/N value and the y, x input coordinates. In
        this case it overrides the full_output parameter.
    plot : bool, optional
        Plots the frame and the apertures considered for clarity. 
    verbose : bool, optional
        Chooses whether to print some output or not. 
    full_output : bool, optional
        If True returns back the S/N value, the y, x input coordinates, noise
        and flux.   
    
    Returns
    -------
    snr : float
        Value of the S/N for the given planet or test speckle.
    If ``full_output`` is True then the function returns:
    sourcey, sourcex, f_source, fluxes.std(), snr
    """
    if array.ndim != 2:
        raise TypeError('Input array is not a frame or 2d array')
    if out_coor and full_output:
        raise TypeError('One of the 2 must be False')
    
    sourcex, sourcey = source_xy 
    
    centery, centerx = frame_center(array)
    sep = dist(centery, centerx, sourcey, sourcex)

    if not sep > (fwhm/2)+1:
        raise RuntimeError('`source_xy` is too close to the frame center')
    
    sens = 'clock' #counterclock
        
    angle = np.arcsin(fwhm/2/sep)*2
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
            
    xx[:] += centerx
    yy[:] += centery 
    rad = fwhm/2.
    apertures = photutils.CircularAperture((xx, yy), r=rad)  # Coordinates (X,Y)
    fluxes = photutils.aperture_photometry(array, apertures, method='exact')

    fluxes = np.array(fluxes['aperture_sum'])
    f_source = fluxes[0].copy()
    fluxes = fluxes[1:]
    n2 = fluxes.shape[0]
    snr = (f_source - fluxes.mean())/(fluxes.std()*np.sqrt(1+(1/n2)))
    
    if verbose:
        msg1 = 'S/N for the given pixel = {:.3f}'
        msg2 = 'Integrated flux in FWHM test aperture = {:.3f}'
        msg3 = 'Mean of background apertures integrated fluxes = {:.3f}'
        msg4 = 'Std-dev of background apertures integrated fluxes = {:.3f}'
        print(msg1.format(snr))
        print(msg2.format(f_source))
        print(msg3.format(fluxes.mean()))
        print(msg4.format(fluxes.std()))
    
    if plot:
        _, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(array, origin='lower', interpolation='nearest', alpha=0.5,
                  cmap='gray')
        for i in range(xx.shape[0]):
            # Circle takes coordinates as (X,Y)
            aper = plt.Circle((xx[i], yy[i]), radius=fwhm/2, color='r', 
                              fill=False, alpha=0.8)                                       
            ax.add_patch(aper)
            cent = plt.Circle((xx[i], yy[i]), radius=0.8, color='r', fill=True,
                              alpha=0.5)
            ax.add_patch(cent)
            aper_source = plt.Circle((sourcex, sourcey), radius=0.7, 
                                     color='b', fill=True, alpha=0.5)                                       
            ax.add_patch(aper_source)
        ax.grid('off')
        plt.show()
    
    if out_coor:
        return sourcey, sourcex, snr
    if full_output:
        return sourcey, sourcex, f_source, fluxes.std(), snr
    else:
        return snr
    

def snr_peakstddev(array, source_xy, fwhm, out_coor=False, plot=False,
                   verbose=False):
    """Calculates the S/N (signal to noise ratio) of a single planet in a
    post-processed (e.g. by LOCI or PCA) frame. The signal is taken as the ratio 
    of pixel value of the planet (test speckle) and the noise computed as the 
    standard deviation of the pixels in an annulus at the same radial distance 
    from the center of the frame. The diameter of the signal aperture and the 
    annulus width is in both cases 1 FWHM ~ 1 lambda/D.
    
    Parameters
    ----------
    array : array_like, 2d
        Post-processed frame where we want to measure S/N.
    source_xy : tuple of floats
        X and Y coordinates of the planet or test speckle.
    fwhm : float
        Size in pixels of the FWHM.
    out_coor: bool, optional
        If True returns back the S/N value and the y, x input coordinates.
    plot : bool, optional
        Plots the frame and the apertures considered for clarity. 
    verbose: bool, optional
        Chooses whether to print some intermediate results or not.    
        
    Returns
    -------
    snr : float
        Value of the S/N for the given planet or test speckle.
    """     
    sourcex, sourcey = source_xy
    centery, centerx = frame_center(array)
    rad = dist(centery,centerx,sourcey,sourcex)  
    
    array = array + np.abs(array.min()) 
    inner_rad = np.round(rad) - fwhm/2
    an_coor = get_annulus(array, inner_rad, fwhm, output_indices=True)
    ap_coor = draw.circle(sourcey, sourcex, int(np.ceil(fwhm/2)))
    array2 = array.copy()
    array2[ap_coor] = array[an_coor].mean()   # we 'mask' the flux aperture
    stddev = array2[an_coor].std()
    peak = array[sourcey, sourcex] 
    snr = peak / stddev
    if verbose:
        msg = "S/N = {:.3f}, Peak px = {:.3f}, Noise = {:.3f}"
        print(msg.format(snr, peak, stddev))
    
    if plot:
        _, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(array, origin='lower', interpolation='nearest')
        circ = plt.Circle((centerx, centery), radius=inner_rad, color='r', 
                          fill=False) 
        ax.add_patch(circ)
        circ2 = plt.Circle((centerx, centery), radius=inner_rad+fwhm, color='r', 
                           fill=False) 
        ax.add_patch(circ2)
        aper = plt.Circle((sourcex, sourcey), radius=fwhm/2., color='b', 
                          fill=False)   # Coordinates (X,Y)
        ax.add_patch(aper)
        plt.show()
        plt.gray()
        
    if out_coor:
        return sourcey, sourcex, snr
    else:
        return snr