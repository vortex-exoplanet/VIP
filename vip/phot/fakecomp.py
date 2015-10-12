#! /usr/bin/env python

"""
Module with fake companion injection functions.
"""

from __future__ import division

__author__ = 'C. Gomez @ ULg'
__all__ = ['create_psf_template',
           'psf_norm',
           'inject_fcs_cube',
           'inject_fc_frame']

import numpy as np
import photutils
from ..calib import cube_crop_frames, frame_shift
from ..var import frame_center


def inject_fcs_cube(array, psf_template, angle_list, flevel, plsc, rad_arcs, 
                    n_branches=1, theta=0):
    """ Injects fake companions in branches, at given radial distances.
    
    Parameters
    ----------
    array : array_like
        Input frame or cube.
    psf_template : array_like 
        2d array with the psf fake companion template. Must have an odd shape.
    flevel : float
        Factor for controlling the brightness of the fake companion.
    plsc : float
        Value of the plsc in pixels.
    rad_arcs : list
        Vector of radial distances of fake companions [arcsec].
    n_branches : int, optional
        Number of azimutal branches.
    theta : float, optional
        Angle in degrees for rotating the position of the first branch that by
        default is located at zero degrees. Theta counts counterclockwise.
        
    Returns
    -------
    array_out : array_like
        Output array with the injected fake companions.
        
    """
    if not array.ndim==3: 
        raise TypeError('Array is not a cube or 3d array')
    
    size_fc = psf_template.shape[0]
    nframes = array.shape[0]
    ceny, cenx = frame_center(array[0])
    fc_fr = np.zeros_like(array[0], dtype=np.float64)
    n_fc_rad = len(rad_arcs)
    array_fc = psf_template.copy()
    # just to make it look cleaner
    array_fc[np.where(array_fc < 0.001)] = 0                                    
    
    w = int(np.floor(size_fc/2.))
    # fcomp in the center of a zeros frame
    fc_fr[ceny-w:ceny+w+1, cenx-w:cenx+w+1] = array_fc

    array_out = np.zeros_like(array)
    for fr in xrange(nframes):                                                  
        tmp = np.zeros_like(array[0])
        for branch in xrange(n_branches):
            ang = branch * 2 * np.pi + np.deg2rad(theta) / n_branches
            for i in xrange(n_fc_rad):
                rad = rad_arcs[i]/plsc                                          
                y = rad * np.sin(ang - np.deg2rad(angle_list[fr]))
                x = rad * np.cos(ang - np.deg2rad(angle_list[fr]))
                tmp = tmp + frame_shift(fc_fr, y, x)*flevel
        array_out[fr] = array[fr] + tmp
        
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


def create_psf_template(array, size, fwhm=5, verbose=True):
    """ Creates a psf template from a cube of non-saturated off-axis frames of
    the star by taking the mean and normalizing the psf flux.
    
    Parameters
    ----------
    array : array_like, 3d
        Input cube.
    size : int
        Size of the squared subimage.
    fwhm: float
        The size of the Full Width Half Maximum in pixel.
    
    """
    if not array.ndim==3:
        raise TypeError('Array is not a cube or 3d array.')
    
    n = array.shape[0]
    psf = cube_crop_frames(array, size=size)
    psf = np.mean(psf, axis=0)
    
    psf_normd = psf_norm(psf, size, fwhm, verbose)
    
    if verbose:  
        print "Done scaled PSF template from the average of", n,"frames."
    return psf_normd


def psf_norm(psf_path, size, fwhm, verbose=True):
    """
    Scale the psf, so the 1*FWHM aperture flux equals 1
    
    Parameters
    ----------
    psf_path: str
        The relative path to the psf fits image.
    size : int
        Size of the squared subimage.
    fwhm: float
        The size of the Full Width Half Maximum in pixel.
        
    Returns
    -------
    out: numpy.array
        The scaled psf.

    """
    psfs = open_fits(psf_path, verbose=verbose)

    psfs = frame_crop(psfs, size ) 
    fwhm_aper = photutils.CircularAperture((frame_center(psfs)), fwhm/2.)
    fwhm_aper_phot = photutils.aperture_photometry(psfs, fwhm_aper)
    
    return psfs/np.array(fwhm_aper_phot['aperture_sum'])
