#! /usr/bin/env python

"""
Module with fake companion injection functions.
"""

from __future__ import division
from __future__ import print_function

__author__ = 'C. Gomez @ ULg'
__all__ = ['create_psf_template',
           'psf_norm',
           'inject_fcs_cube',
           'inject_fc_frame']

import numpy as np
import photutils
from ..preproc import cube_crop_frames, frame_shift, frame_crop
from ..var import frame_center, fit_2dgaussian, get_circle


def inject_fcs_cube(array, psf_template, angle_list, flevel, plsc, rad_dists, 
                    n_branches=1, theta=0, imlib='opencv', verbose=True):
    """ Injects fake companions in branches, at given radial distances.
    
    Parameters
    ----------
    array : array_like
        Input frame or cube.
    psf_template : array_like 
        2d array with the normalized psf template. It should have an odd shape.
        It's recommended to run the function psf_norm to get a proper PSF
        template.
    flevel : float
        Factor for controlling the brightness of the fake companions.
    plsc : float
        Value of the plsc in pixels.
    rad_dists : list or array 1d
        Vector of radial distances of fake companions in pixels.
    n_branches : int, optional
        Number of azimutal branches.
    theta : float, optional
        Angle in degrees for rotating the position of the first branch that by
        default is located at zero degrees. Theta counts counterclockwise from
        the positive x axis.
    imlib : {'opencv', 'ndimage-fourier', 'ndimage-interp'}, string optional
        Library or method used for image operations (shifts). Opencv is the
        default for being the fastest.
    verbose : {True, False}, bool optional
        If True prints out additional information. 
    
    Returns
    -------
    array_out : array_like
        Output array with the injected fake companions.
        
    """
    if not array.ndim==3: 
        raise TypeError('Array is not a cube or 3d array')
    
    ceny, cenx = frame_center(array[0])
    rad_dists = np.array(rad_dists)
    if not rad_dists[-1]<array[0].shape[0]/2.:
        msg = 'rad_dists last location is at the border (or outside) of the field'
        raise ValueError(msg)
    
    size_fc = psf_template.shape[0]
    nframes = array.shape[0]
    fc_fr = np.zeros_like(array[0], dtype=np.float64)  # TODO: why float64?
    n_fc_rad = rad_dists.shape[0]

    w = int(np.floor(size_fc/2.))
    # fcomp in the center of a zeros frame
    fc_fr[int(ceny-w):int(ceny+w+1), int(cenx-w):int(cenx+w+1)] = psf_template

    array_out = np.zeros_like(array)
    for fr in range(nframes):
        tmp = np.zeros_like(array[0])
        for branch in range(n_branches):
            ang = (branch * 2 * np.pi / n_branches) + np.deg2rad(theta)
            for i in range(n_fc_rad):
                rad = rad_dists[i]                                         
                y = rad * np.sin(ang - np.deg2rad(angle_list[fr]))
                x = rad * np.cos(ang - np.deg2rad(angle_list[fr]))
                tmp += frame_shift(fc_fr, y, x, imlib=imlib)*flevel
        array_out[fr] = array[fr] + tmp
    
    if verbose:
        for branch in range(n_branches):
            print('Branch '+str(branch+1)+':')
            for i in range(n_fc_rad):
                ang = (branch * 2 * np.pi / n_branches) + np.deg2rad(theta)
                posy = rad_dists[i] * np.sin(ang) + ceny
                posx = rad_dists[i] * np.cos(ang) + cenx
                rad_arcs = rad_dists[i]*plsc
                msg ='\t(X,Y)=({:.2f}, {:.2f}) at {:.2f} arcsec ({:.2f} pxs)'
                print(msg.format(posx, posy, rad_arcs, rad_dists[i]))
        
    return array_out


def inject_fc_frame(array, array_fc, pos_y, pos_x, flux):
    """ Injects a fake companion in a single frame at given coordinates.
    """
    if not array.ndim==2:
        raise TypeError('Array is not a frame or 2d array.')
    size_fc = array_fc.shape[0]
    ceny, cenx = frame_center(array)
    fc_fr = np.zeros_like(array, dtype=np.float64)
    w = int(np.floor(size_fc/2.))
    # fcomp in the center of a zeros frame
    fc_fr[ceny-w:ceny+w+1, cenx-w:cenx+w+1] = array_fc   
    array_out = array + frame_shift(fc_fr, pos_y-ceny, pos_x-cenx)*flux
    return array_out


def create_psf_template(array, size, fwhm=4, verbose=True, collapse='mean'):
    """ Creates a psf template from a cube of non-saturated off-axis frames of
    the star by taking the mean and normalizing the psf flux.
    
    Parameters
    ----------
    array : array_like, 3d
        Input cube.
    size : int
        Size of the squared subimage.
    fwhm: float, optional
        The size of the Full Width Half Maximum in pixel.
    verbose : {True,False}, bool optional
        Whether to print to stdout information about file opening, cropping and
        completion of the psf template.
    collapse : {'mean','median'}, string optional
        Defines the way the frames are collapsed.
        
    Returns
    -------
    psf_normd : array_like
        Normalized PSF.
    """
    if not array.ndim==3:
        raise TypeError('Array is not a cube or 3d array.')
    
    n = array.shape[0]
    psf = cube_crop_frames(array, size=size, verbose=verbose)
    if collapse=='mean':
        psf = np.mean(psf, axis=0)
    elif collapse=='median':
        psf = np.median(psf, axis=0)
    else:
        raise TypeError('Collapse mode not recognized.')
    
    psf_normd = psf_norm(psf, size=size, fwhm=fwhm)
    
    if verbose:  
        print("Done scaled PSF template from the average of", n,"frames.")
    return psf_normd


def psf_norm(array, fwhm=4, size=None, threshold=None, mask_core=None,
             full_output=False, verbose=False):
    """ Scales a PSF, so the 1*FWHM aperture flux equals 1.
    
    Parameters
    ----------
    array: array_like
        The psf 2d array.
    fwhm: float, optional
        The the Full Width Half Maximum in pixels.
    size : int or None, optional
        If int it will correspond to the size of the squared subimage to be 
        cropped form the psf array.
    threshold : None of float, optional
        Sets to zero small values, trying to leave only the core of the PSF.
    mask_core : None of float, optional
        Sets the radius of a circular aperture for the core of the PSF, 
        everything else will be set to zero.
        
    Returns
    -------
    psf_norm: array_like
        The normalized psf.

    """
    if size is not None:  
        psfs = frame_crop(array, min(int(size), array.shape[0]), verbose=False)
    else:
        psfs = array.copy()
        # If frame size is even we drop last row and last column
        if psfs.shape[0]%2==0:
            psfs = psfs[:-1,:]
        if psfs.shape[1]%2==0:
            psfs = psfs[:,:-1]
    
    # we check if the psf is centered and fix it if needed
    cy, cx = frame_center(psfs, verbose=False)
    if cy!=np.where(psfs==psfs.max())[0] or cx!=np.where(psfs==psfs.max())[1]:
        # first we find the centroid and put it in the center of the array 
        centroidy, centroidx = fit_2dgaussian(psfs, fwhmx=fwhm, fwhmy=fwhm)
        shiftx, shifty = centroidx - cx, centroidy - cy
        psfs = frame_shift(psfs, -shifty, -shiftx)
        for _ in range(2):
            centroidy, centroidx = fit_2dgaussian(psfs, fwhmx=fwhm, fwhmy=fwhm)
            cy, cx = frame_center(psfs, verbose=False)
            shiftx, shifty = centroidx - cx, centroidy - cy
            psfs = frame_shift(psfs, -shifty, -shiftx)
    
    # we check whether the flux is normalized and fix it if needed
    fwhm_aper = photutils.CircularAperture((frame_center(psfs)), fwhm/2.)
    fwhm_aper_phot = photutils.aperture_photometry(psfs, fwhm_aper,
                                                   method='exact')
    fwhm_flux = np.array(fwhm_aper_phot['aperture_sum'])
    if verbose:
        print("Flux in 1xFWHM aperture: {}".format(fwhm_flux))

    if fwhm_flux>1.1 or fwhm_flux<0.9:
        psf_norm_array = psfs/np.array(fwhm_aper_phot['aperture_sum'])
    else:
        psf_norm_array = psfs
    
    if threshold is not None:
        psf_norm_array[np.where(psf_norm_array < threshold)] = 0
    
    if mask_core is not None:
        psf_norm_array = get_circle(psf_norm_array, radius=mask_core)

    if full_output:
        return psf_norm_array, fwhm_flux
    else:
        return psf_norm_array


