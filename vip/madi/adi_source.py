#! /usr/bin/env python

"""
Module with ADI algorithm (median psf subtraction).
Carlos A. Gomez / ULg
"""

from __future__ import division 

__author__ = 'C. Gomez @ ULg'
__all__ = ['adi']

import numpy as np
from ..conf import timeInit, timing, LBT, VLT_NACO
from ..var import get_annulus
from ..calib import cube_derotate
from ..pca.pca_local import define_annuli, get_fwhm

def adi(array, angle_list, fwhm=None, instrument=None, radius_int=0, asize=2, 
        delta_rot=1, mode='simple', full_output=False, verbose=True):
    """ Algorithm based on Marois et al. 2006 on Angular Differential Imaging.   
    First the median frame is subtracted, then the median of the four closest 
    frames taking into account the pa_threshold (field rotation).
    
    Parameters
    ----------
    array : array_like, 3d
        Input cube.
    angle_list : array_like, 1d
        Corresponding parallactic angle for each frame.
    fwhm : float
        Known size of the FHWM in pixels to be used instead of the instrument 
        default.
    instrument: {'naco27, 'lmircam'}, optional
        Defines the type of dataset. For cubes without proper headers.
    radius_int : int, optional
        The radius of the innermost annulus. By default is 0, if >0 then the 
        central circular area is discarded.
    asize : int, optional
        The size of the annuli, in FWHM. Default is 2.
    delta_rot : int, optional
        Factor for increasing the parallactic angle threshold, expressed in FWHM.
        Default is 1 (excludes 1 FHWM on each side of the considered frame).
    mode : {"simple","annular"}, str optional
        In "simple" mode only the median frame is subtracted, in "annular" mode
        also the 4 closest frames given a PA threshold (annulus-wise) are 
        subtracted.
    full_output: boolean, optional
        Whether to return the final median combined image only or with other 
        intermediate arrays. 
    verbose : {True, False}, bool optional
        If True prints to stdout intermediate info.
        
    Returns
    -------
    frame : array_like, 2d
        Median combination of the de-rotated cube.
    If full_output is True:  
    cube_out : array_like, 3d
        The cube of residuals.
    cube_der : array_like, 3d
        The derotated cube of residuals.
         
    """
    def find_indices(angle_list, frame, thr):  
        """ Returns the indices to be left in frames library for optimized ADI.
        To find a more pythonic way to do this!
        """
        n = angle_list.shape[0]
        index_prev = 0 
        index_foll = frame                                  
        for i in xrange(0, frame):
            if np.abs(angle_list[frame]-angle_list[i]) < thr:
                index_prev = i
                break
            else:
                index_prev += 1
        for k in xrange(frame, n):
            if np.abs(angle_list[k]-angle_list[frame]) > thr:
                index_foll = k
                break
            else:
                index_foll += 1

        ind1 = index_prev-2
        if ind1<0: ind1=0
        ind2 = index_prev
        ind3 = index_foll
        ind4 = index_foll+2
        if ind4>n: ind4=n
        return np.array(range(ind1,ind2)+range(ind3,ind4))
    
    if not array.ndim == 3:
        raise TypeError('Input array is not a cube or 3d array.')
    if not array.shape[0] == angle_list.shape[0]:
        raise TypeError('Input vector or parallactic angles has wrong length.')
        
    n, y, _ = array.shape
     
    if verbose:  start_time = timeInit()
        
    #***************************************************************************
    # The median frame (basic psf reference) is first subtracted from each frame.
    #***************************************************************************      
    ref_psf = np.median(array, axis=0)
    array = array - ref_psf
    
    if mode=='simple':
        cube_out = array
        if verbose:  print 'Median psf reference subtracted'
    
    elif mode=='annular':   
        if not fwhm:  fwhm = get_fwhm(instrument) 
        annulus_width = int(asize * fwhm)                            # equal size for all annuli
        n_annuli = int(np.floor((y/2-radius_int)/annulus_width))    
        if verbose:  print 'N annuli =', n_annuli, ', FWHM =', fwhm, '\n'
        #***********************************************************************
        # The annuli are built, and the corresponding PA thresholds for frame 
        # rejection are calculated. The PA rejection is calculated at center of 
        # the annulus.
        #***********************************************************************
        cube_out = np.zeros_like(array) 
        for ann in xrange(n_annuli):
            pa_threshold,inner_radius,ann_center = define_annuli(angle_list, 
                                                                 ann, n_annuli, 
                                                                 fwhm, radius_int, 
                                                                 annulus_width, 
                                                                 delta_rot,
                                                                 verbose) 
    
            indices = get_annulus(array[0], inner_radius, annulus_width, 
                                  output_indices=True)
            yy = indices[0]
            xx = indices[1]
            
            matrix = array[:, yy, xx]    # shape [nframes x npx_annulus]
            
            #*******************************************************************
            # A second optimized psf reference is subtracted from each frame. 
            # For each frame we find 4 frames, given enough field rotation 
            # (PA thresh), to construct this optimized psf reference.
            #*******************************************************************
            for frame in xrange(n):                                                 
                if pa_threshold != 0:
                    indices_left = find_indices(angle_list, frame, pa_threshold)
                    matrix_disc = matrix[indices_left]
                    #print indices_left
                else:
                    matrix_disc = matrix
            
                ref_psf_opt = np.median(matrix_disc, axis=0)
                curr_frame = matrix[frame]
                subtracted = curr_frame - ref_psf_opt
                cube_out[frame][yy, xx] = subtracted
        if verbose:  print 'Optimized median psf reference subtracted'
        
    else:
        raise RuntimeError('Mode not recognized')
    
    cube_der, frame = cube_derotate(cube_out, angle_list)
    if verbose:
        print 'Done derotating and combining'
        timing(start_time)
    if full_output:
        return cube_out, cube_der, frame 
    else:
        return frame 


  