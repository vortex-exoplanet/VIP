#! /usr/bin/env python

"""
Module with ADI algorithm (median psf subtraction).
Carlos A. Gomez / ULg
"""

from __future__ import division 
from __future__ import print_function

__author__ = 'C. Gomez @ ULg'
__all__ = ['adi']

import numpy as np
from ..conf import time_ini, timing
from ..var import get_annulus, mask_circle
from ..preproc import cube_derotate, cube_collapse, check_PA_vector
from ..pca.pca_local import define_annuli


def adi(cube, angle_list, fwhm=4, radius_int=0, asize=2, delta_rot=1, 
        mode='fullfr', nframes=4, collapse='median', full_output=False, 
        verbose=True):
    """ Algorithm based on Marois et al. 2006 on Angular Differential Imaging.   
    First the median frame is subtracted, then the median of the four closest 
    frames taking into account the pa_threshold (field rotation).
    
    Parameters
    ----------
    cube : array_like, 3d
        Input cube.
    angle_list : array_like, 1d
        Corresponding parallactic angle for each frame.
    fwhm : float
        Known size of the FHWM in pixels to be used. Default is 4.
    radius_int : int, optional
        The radius of the innermost annulus. By default is 0, if >0 then the 
        central circular area is discarded.
    asize : int, optional
        The size of the annuli, in FWHM. Default is 2.
    delta_rot : int, optional
        Factor for increasing the parallactic angle threshold, expressed in FWHM.
        Default is 1 (excludes 1 FHWM on each side of the considered frame).
    mode : {"fullfr","annular"}, str optional
        In "simple" mode only the median frame is subtracted, in "annular" mode
        also the 4 closest frames given a PA threshold (annulus-wise) are 
        subtracted.
    nframes : even int optional
        Number of frames to be used for building the optimized reference PSF 
        when working in annular mode. 
    collapse : {'median', 'mean', 'sum', 'trimmean'}, str optional
        Sets the way of collapsing the frames for producing a final image.
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
    def find_indices(angle_list, frame, thr, nframes):  
        """ Returns the indices to be left in frames library for optimized ADI.
        TO-DO: find a pythonic way to do this!
        """
        n = angle_list.shape[0]
        index_prev = 0 
        index_foll = frame                                  
        for i in range(0, frame):
            if np.abs(angle_list[frame]-angle_list[i]) < thr:
                index_prev = i
                break
            else:
                index_prev += 1
        for k in range(frame, n):
            if np.abs(angle_list[k]-angle_list[frame]) > thr:
                index_foll = k
                break
            else:
                index_foll += 1

        window = int(nframes/2)
        ind1 = index_prev-window
        ind1 = max(ind1, 0)
        ind2 = index_prev
        ind3 = index_foll
        ind4 = index_foll+window
        ind4 = min(ind4, n)
        indices = np.array(range(ind1,ind2)+range(ind3,ind4))
        #print ind1, ind2, ind3, ind4, indices
        return indices
    
    #***************************************************************************
    array = cube
    
    if not array.ndim == 3:
        raise TypeError('Input array is not a cube or 3d array.')
    if not array.shape[0] == angle_list.shape[0]:
        raise TypeError('Input vector or parallactic angles has wrong length.')
    if not nframes%2==0:
        raise TypeError('nframes argument must be even value.')
    
    n, y, _ = array.shape
     
    if verbose:  start_time = time_ini()
    
    angle_list = check_PA_vector(angle_list)
    
    #***************************************************************************
    # The median frame (basic psf reference) is first subtracted from each frame.
    #***************************************************************************      
    ref_psf = np.median(array, axis=0)
    array = array - ref_psf
    
    if mode=='fullfr':
        if radius_int>0:
            cube_out = mask_circle(array, radius_int)
        else:
            cube_out = array
        if verbose:  print('Median psf reference subtracted')
    
    elif mode=='annular':   
        annulus_width = int(asize * fwhm)                            # equal size for all annuli
        n_annuli = int(np.floor((y/2-radius_int)/annulus_width))    
        if verbose:  print('N annuli =', n_annuli, ', FWHM =', fwhm, '\n')
        #***********************************************************************
        # The annuli are built, and the corresponding PA thresholds for frame 
        # rejection are calculated. The PA rejection is calculated at center of 
        # the annulus.
        #***********************************************************************
        cube_out = np.zeros_like(array) 
        for ann in range(n_annuli):
            pa_threshold,inner_radius,_= define_annuli(angle_list, ann, n_annuli, 
                                                       fwhm, radius_int, 
                                                       annulus_width, delta_rot,
                                                       verbose) 
    
            indices = get_annulus(array[0], inner_radius, annulus_width, 
                                  output_indices=True)
            yy = indices[0]
            xx = indices[1]
            
            matrix = array[:, yy, xx]    # shape [nframes x npx_annulus]
            
            #*******************************************************************
            # A second optimized psf reference is subtracted from each frame. 
            # For each frame we find *nframes*, depending on the PA threshold, 
            # to construct this optimized psf reference.
            #*******************************************************************
            for frame in range(n):
                if pa_threshold != 0:
                    indices_left = find_indices(angle_list, frame, pa_threshold, 
                                                nframes)
                    matrix_disc = matrix[indices_left]
                else:
                    matrix_disc = matrix
            
                ref_psf_opt = np.median(matrix_disc, axis=0)
                curr_frame = matrix[frame]
                subtracted = curr_frame - ref_psf_opt
                cube_out[frame][yy, xx] = subtracted
        if verbose:  print('Optimized median psf reference subtracted')
        
    else:
        raise RuntimeError('Mode not recognized')
    
    cube_der = cube_derotate(cube_out, angle_list)
    frame = cube_collapse(cube_der, mode=collapse)
    if verbose:
        print('Done derotating and combining')
        timing(start_time)
    if full_output:
        return cube_out, cube_der, frame 
    else:
        return frame 

