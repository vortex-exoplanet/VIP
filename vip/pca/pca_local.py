#! /usr/bin/env python

"""
Module with local smart pca (annulus-wise) serial and parallel implementations.
"""

from __future__ import division 

__author__ = 'C. Gomez @ ULg'
__all__ = ['annular_pca', 
           'subannular_pca', 
           'subannular_pca_parallel']

# TODO: to move here function scale_cube()

import numpy as np
import itertools as itt
from multiprocessing import Pool, cpu_count
from ..calib import cube_derotate, check_PA_vector
from ..conf import timeInit, timing, eval_func_tuple, VLT_NACO, LBT 
from ..var import get_annulus_quad
from ..pca.utils import svd_wrapper, reshape_matrix
from ..var import get_annulus


def annular_pca(array, angle_list, radius_int=0, asize=2, delta_rot=1, ncomp=1,
                svd_mode='randsvd', instrument=None, fwhm=None, center=True, 
                full_output=False, verbose=True, debug=False):
    """ Smart PCA (annular version) algorithm. On each annulus we discard 
    reference images taking into account the parallactic angle threshold.
     
    Parameters
    ----------
    array : array_like, 3d
        Input cube.
    angle_list : array_like, 1d
        Corresponding parallactic angle for each frame.
    radius_int : int, optional
        The radius of the innermost annulus. By default is 0, if >0 then the 
        central circular area is discarded.
    asize : int, optional
        The size of the annuli, in FWHM. Default is 2.
    delta_rot : int, optional
        Factor for increasing the parallactic angle threshold, expressed in FWHM.
        Default is 1 (excludes 1 FHWM on each side of the considered frame).
    ncomp : None or int, optional
        How many PCs are kept. If none it will be automatically determined.
    svd_mode : {randsvd, eigen, lapack, arpack, opencv}, optional
        Switch for different ways of computing the SVD and principal components.
    full_output: boolean, optional
        Whether to return the final median combined image only or with other 
        intermediate arrays.  
    instrument: {'naco27, 'lmircam'}, optional
        Defines the type of dataset. For cubes without proper headers.
    fwhm : float
        Known size of the FHWM in pixels to be used instead of the instrument 
        default.
    center : {True,False}, bool optional
        Whether to center the data or not.
    verbose : {True, False}, bool optional
        If True prints to stdout intermediate info. 
    debug : {False, True}, bool optional
        Whether to output some intermediate information.
        
    Returns
    -------
    frame : array_like, 2d    
        Median combination of the de-rotated cube.
    If full_output is True:  
    array_out : array_like, 3d 
        Cube of residuals.
    array_der : array_like, 3d
        Cube residuals after de-rotation.
     
    """
    if not array.ndim == 3:
        raise TypeError('Input array is not a cube or 3d array')
    if not array.shape[0] == angle_list.shape[0]:
        raise TypeError('Input vector or parallactic angles has wrong length')
     
    n, y, x = array.shape
    if not fwhm:  fwhm = get_fwhm(instrument)
     
    if verbose:  start_time = timeInit()
    
    angle_list = check_PA_vector(angle_list)
    
    if not ncomp: auto_ncomp = True
    else: auto_ncomp = False    

    annulus_width = int(asize * fwhm)                                           # equal size for all annuli
    n_annuli = int(np.floor((y/2-radius_int)/annulus_width))    
    if verbose:
        msg = '# annuli = {:}, Ann width = {:}, FWHM = {:.3f}\n'
        print msg.format(n_annuli, annulus_width, fwhm) 
        print 'PCA will be done locally per annulus.\n'
        
    #***************************************************************************
    # The annuli are built, and the corresponding PA thresholds for frame 
    # rejection are calculated. The PA rejection is calculated at center of the 
    # annulus.
    #***************************************************************************
    matrix_final = np.zeros_like(array) 
    for ann in xrange(n_annuli):
        pa_threshold, inner_radius, ann_center = define_annuli(angle_list, ann, 
                                                               n_annuli, 
                                                               fwhm, radius_int, 
                                                               annulus_width, 
                                                               delta_rot,
                                                               verbose)
        indices = get_annulus(array[0], inner_radius, annulus_width, 
                              output_indices=True)
        yy = indices[0]
        xx = indices[1]

        #***********************************************************************
        # We make the matrix of the given annulus. Centering the matrix if needed
        # removing temporal mean. 
        #***********************************************************************        
        data_all = array[:, yy, xx]                                             # shape [nframes x npx_annulus]                       
        if center:  data_all = data_all - data_all.mean(axis=0)
        
        #***********************************************************************
        # For each frame we find the frames (rows) to be rejected depending on 
        # the radial distance from the center.
        #***********************************************************************
        for frame in xrange(n):                                                 # for each frame 
            if pa_threshold != 0:
                #indices_left = find_indices(angle_list, frame, pa_threshold, False)
                if ann_center > fwhm*10:        ### TBD: fwhm*10
                    indices_left = find_indices(angle_list, frame, pa_threshold, True)
                else:
                    indices_left = find_indices(angle_list, frame, pa_threshold, False)
                
                data_ref = data_all[indices_left]
                
                if data_ref.shape[0] <= 2:
                    msg = 'No frames or too few frames left in the PCA reference frames library.'
                    raise RuntimeError(msg)
            else:
                data_ref = data_all
             
            if center:  
                data = data_ref - data_ref.mean(axis=0)
            else:
                data = data_ref                                
            
            curr_frame = data_all[frame]                                        # current frame
                         
            #*******************************************************************
            # If ncomp=None is calculated for current annular quadrant
            #*******************************************************************
            if auto_ncomp and frame==0:
                ncomp = get_ncomp(data, svd_mode, debug) 
                        
            #*******************************************************************
            # Performing PCA according to "mode" flag. 'data' is the matrix used 
            # in the SVD where each frame is a row, and the number of columns is
            # the number of pixels in a given annulus. 
            #*******************************************************************       
            V = svd_wrapper(data, svd_mode, ncomp, debug=False, verbose=False)
            transformed = np.dot(curr_frame, V.T)
            reconstructed = np.dot(transformed.T, V)                            # reference psf
            residuals = curr_frame - reconstructed     
            matrix_final[frame][yy, xx] = residuals                             
              
        if verbose:
            print 'Done PCA with {:} for current annulus'.format(svd_mode)
            timing(start_time)

    #***************************************************************************
    # Cube is reshaped.
    #***************************************************************************
    array_out = reshape_matrix(matrix_final, y, x)                 
     
    #***************************************************************************
    # Cube is derotated according to the parallactic angle and median combined.
    #***************************************************************************
    array_der, frame = cube_derotate(array_out, angle_list)
    if verbose:
        print 'Done derotating and combining.'
        timing(start_time)
    if full_output:
        return array_out, array_der, frame 
    else:
        return frame               


def subannular_pca(array, angle_list, radius_int=0, asize=1, delta_rot=1, 
                   ncomp=1, svd_mode='randsvd', instrument=None, fwhm=None, 
                   center=True, full_output=False, verbose=True, debug=False):
    """ Smart PCA (subannular version) algorithm. The PCA is computed locally 
    in each quadrant of each annulus. On each annulus we discard reference 
    images taking into account the parallactic angle threshold. 
     
    Parameters
    ----------
    array : array_like, 3d
        Input cube.
    angle_list : array_like, 1d
        Corresponding parallactic angle for each frame.
    radius_int : int, optional
        The radius of the innermost annulus. By default is 0, if >0 then the 
        central circular area is discarded.
    asize : int, optional
        The size of the annuli, in FWHM. Default is 3.
    delta_rot : int, optional
        Factor for increasing the parallactic angle threshold, expressed in FWHM.
        Default is 1 (excludes 1 FHWM on each side of the considered frame).
    ncomp : int, optional
        How many PCs are kept. If none it will be automatically determined.
    svd_mode : {randsvd, eigen, lapack, arpack, opencv}, optional
        Switch for different ways of computing the SVD and principal components.
    full_output: boolean, optional
        Whether to return the final median combined image only or with other 
        intermediate arrays.  
    instrument: {'naco27, 'lmircam'}, optional
        Defines the type of dataset. For cubes without proper headers.
    fwhm : float
        Know size of the FHWM in pixels to be used instead of the instrument 
    center : {True,False}, bool optional
        Whether to center the data or not.
        default.
    verbose : {True, False}, bool optional
        If True prints to stdout intermediate info. 
    debug : {False, True}, bool optional
        Whether to output some intermediate information.
     
    Returns
    -------
    frame : array_like, 2d    
        Median combination of the de-rotated cube.
    If full_output is True:  
    array_out : array_like, 3d 
        Cube of residuals.
    array_der : array_like, 3d
        Cube residuals after de-rotation.
     
    """
    if not array.ndim == 3:
        raise TypeError('Input array is not a cube or 3d array.')
    if not array.shape[0] == angle_list.shape[0]:
        raise TypeError('Input vector or parallactic angles has wrong length.')
     
    n, y, _ = array.shape
    if not fwhm:  fwhm = get_fwhm(instrument)
     
    if verbose:  start_time = timeInit()
    
    annulus_width = int(asize * fwhm)                                           # equal size for all annuli
    n_annuli = int(np.floor((y/2-radius_int)/annulus_width))    
    if verbose:
        msg = '# annuli = {:}, Ann width = {:}, FWHM = {:.3f}\n'
        print msg.format(n_annuli, annulus_width, fwhm) 
        print 'PCA will be done locally per annulus and per quadrant.\n'
     
    if not ncomp: auto_ncomp = True
    else: auto_ncomp = False
     
    #***************************************************************************
    # The annuli are built, and the corresponding PA thresholds for frame 
    # rejection are calculated. The PA rejection is calculated at center of the 
    # annulus.
    #***************************************************************************
    cube_out = np.zeros_like(array)
    for ann in xrange(n_annuli):
        pa_threshold, inner_radius, ann_center = define_annuli(angle_list, ann, 
                                                               n_annuli, 
                                                               fwhm, radius_int, 
                                                               annulus_width, 
                                                               delta_rot,
                                                               verbose) 
        indices = get_annulus_quad(array[0], inner_radius, annulus_width)
         
        #***********************************************************************
        # We arrange the PCA matrix for each annular quadrant and center if 
        # needed (removal of temporal mean).
        #***********************************************************************
        for quadrant in xrange(4):
            yy = indices[quadrant][0]
            xx = indices[quadrant][1]
            matrix_quad = array[:, yy, xx]                                      # shape [nframes x npx_quad] 
 
            if center:  matrix_quad = matrix_quad - matrix_quad.mean(axis=0)
             
            #*******************************************************************
            # For each frame we find the frames to be rejected depending on the 
            # radial distance from the center.
            #*******************************************************************
            for frame in xrange(n):                                             
                if pa_threshold != 0:
                    if ann_center > fwhm*10:                                    ### TBD: fwhm*10
                        indices_left = find_indices(angle_list, frame, 
                                                    pa_threshold, True)
                    else:
                        indices_left = find_indices(angle_list, frame, 
                                                    pa_threshold, False)
                      
                    data_ref = matrix_quad[indices_left]
                     
                    if data_ref.shape[0] <= 10:
                        msg = 'Too few frames left in the PCA library.'
                        raise RuntimeError(msg)
                else:
                    data_ref = matrix_quad
                               
                if center:
                    data = data_ref - data_ref.mean(axis=0)                     # removing temporal mean
                else:
                    data = data_ref
                                            
                curr_frame = matrix_quad[frame]                                 # current frame
                             
                #**************************************************************
                # If ncomp=None is calculated for current annular quadrant
                #**************************************************************
                if auto_ncomp and frame==0:
                    ncomp = get_ncomp(data, svd_mode, debug)
                 
                #***************************************************************
                # Performing SVD/PCA according to "mode" flag. 'data' is the 
                # matrix for feeding the SVD.
                #***************************************************************
                V = svd_wrapper(data, svd_mode, ncomp, debug=False, verbose=False)
                 
                transformed = np.dot(curr_frame, V.T)
                reconstructed = np.dot(transformed.T, V)                        # reference psf
                residuals = curr_frame - reconstructed     
                cube_out[frame][yy, xx] = residuals                            
         
        if verbose:
            print 'Done PCA with {:} for current annulus'.format(svd_mode)
            timing(start_time)      
         
    #***************************************************************************
    # Cube is derotated according to the parallactic angle and median combined.
    #***************************************************************************
    cube_der, frame = cube_derotate(cube_out, angle_list)
    if verbose:
        print 'Done derotating and combining.'
        timing(start_time)
    if full_output:
        return cube_out, cube_der, frame 
    else:
        return frame 


def subannular_pca_parallel(array, angle_list, radius_int=0, asize=1, 
                            delta_rot=1, ncomp=1, instrument=None, fwhm=None, 
                            center=True, nproc=None, svd_mode='arpack', 
                            full_output=False, verbose=True, debug=False):
    """ Local PCA (subannular version) parallel algorithm. The PCA is computed 
    locally in each quadrant of each annulus. On each annulus we discard 
    reference images taking into account the parallactic angle threshold. 
    
    This algorithm is meant for machines with several cores. It has been
    tested on a Linux and OSX. The OSX accelerate library, which comes by
    default in every OSX system, is broken for multiprocessing. Avoid using
    it unless you have compiled python against other linear algebra library.
    On linux with the default LAPACK/BLAS libraries it succesfully distributes
    the processes among all the existing cores.
    
    Parameters
    ----------
    array : array_like, 3d
        Input cube.
    angle_list : array_like, 1d
        Corresponding parallactic angle for each frame.
    radius_int : int, optional
        The radius of the innermost annulus. By default is 0, if >0 then the 
        central circular area is discarded.
    asize : int, optional
        The size of the annuli, in FWHM. Default is 3.
    delta_rot : int, optional
        Factor for increasing the parallactic angle threshold, expressed in FWHM.
        Default is 1 (excludes 1 FHWM on each side of the considered frame).
    ncomp : int, optional
        How many PCs are kept. If none it will be automatically determined.
    svd_mode : {lapack, randsvd, eigen, arpack, opencv}, str optional
        Switch for different ways of computing the SVD and principal components.
    instrument: {'naco27, 'lmircam'}, optional
        Defines the type of dataset. For cubes without proper headers.
    fwhm : float
        Know size of the FHWM in pixels to be used instead of the instrument 
    center : {True,False}, bool optional
        Whether to center the data or not.
    nproc : int, optional
        Number of processes for parallel computing. If None the number of 
        processes will be set to (cpu_count()/2). 
    full_output: boolean, optional
        Whether to return the final median combined image only or with other 
        intermediate arrays.  
    verbose : {True, False}, bool optional
        If True prints to stdout intermediate info. 
    debug : {False, True}, bool optional
        Whether to output some intermediate information.
    
    Returns
    -------
    frame : array_like, 2d    
        Median combination of the de-rotated cube.
    If full_output is True:  
    array_out : array_like, 3d 
        Cube of residuals.
    array_der : array_like, 3d
        Cube residuals after de-rotation.
    
    """
    #### TODO: check LAPACK library that numpy is using and raise error
    
    if not nproc:   # Hyper-threading "duplicates" the cores -> cpu_count/2
        nproc = (cpu_count()/2)
    
    if not array.ndim == 3:
        raise TypeError('Input array is not a cube or 3d array.')
    if not array.shape[0] == angle_list.shape[0]:
        raise TypeError('Input vector or parallactic angles has wrong length.')
     
    n, y, _ = array.shape
    if not fwhm:  fwhm = get_fwhm(instrument)
     
    if verbose:  start_time = timeInit()
    
    annulus_width = int(asize * fwhm)                                           # equal size for all annuli
    n_annuli = int(np.floor((y/2-radius_int)/annulus_width))    
    if verbose:
        msg = '# annuli = {:}, Ann width = {:}, FWHM = {:.3f}\n'
        print msg.format(n_annuli, annulus_width, fwhm) 
        print 'PCA will be done locally per annulus and per quadrant.\n'
     
    if not ncomp: auto_ncomp = True
    else: auto_ncomp = False
    
    #***************************************************************************
    # The annuli are built, and the corresponding PA thresholds for frame 
    # rejection are calculated. The PA rejection is calculated at center of the 
    # annulus
    #***************************************************************************
    cube_out = np.zeros_like(array)
    for ann in xrange(n_annuli):
        pa_threshold, inner_radius, ann_center = define_annuli(angle_list, ann, 
                                                               n_annuli, 
                                                               fwhm, radius_int, 
                                                               annulus_width, 
                                                               delta_rot,
                                                               verbose)  
        indices = get_annulus_quad(array[0], inner_radius, annulus_width)
        
        #***********************************************************************
        # PCA matrix is created for each annular quadrant and centered if needed
        #***********************************************************************
        for quadrant in xrange(4):
            yy = indices[quadrant][0]
            xx = indices[quadrant][1]
            matrix_quad = array[:, yy, xx]                                      # shape [nframes x npx_quad] 

            if center:  matrix_quad = matrix_quad - matrix_quad.mean(axis=0)

            #*******************************************************************
            # If ncomp=None is calculated for current annular quadrant
            #*******************************************************************
            # noise minimization for # pcs definition ### TODO: to finish
            if auto_ncomp:                                                      
                ncomp = get_ncomp(matrix_quad, svd_mode, debug)
                        
            #*******************************************************************
            # A multiprocessing pool is created to process frames in parallel.
            # SVD/PCA is done in do_pca_patch function. 
            #*******************************************************************            
            pool = Pool(processes=nproc)
            res = pool.map(eval_func_tuple, itt.izip(itt.repeat(do_pca_patch), 
                                                     itt.repeat(matrix_quad),
                                                     range(n),
                                                     itt.repeat(angle_list),
                                                     itt.repeat(fwhm),
                                                     itt.repeat(pa_threshold),
                                                     itt.repeat(center),
                                                     itt.repeat(ann_center),
                                                     itt.repeat(svd_mode),
                                                     itt.repeat(ncomp)))
            residuals = np.array(res)
            pool.close()
            for fr in range(n):
                cube_out[fr][yy, xx] = residuals[fr]
        
        if verbose:
            print 'Done PCA with {:} for current annulus'.format(svd_mode)
            timing(start_time)      
        
    #***************************************************************************
    # Cube is derotated according to the parallactic angle and median combined
    #***************************************************************************
    cube_der, frame = cube_derotate(cube_out, angle_list)
    if verbose:
        print 'Done derotating and combining.'
        timing(start_time)
    if full_output:
        return cube_out, cube_der, frame 
    else:
        return frame 
    
    
    
### Secondary functions ********************************************************
    
def get_fwhm(instrument):
    """ Defines the FWHM for a given instrument based on its parameters defined
    in a dictionary in vortex/cpar/param.py.                                    ### TODO: check final location
    """
    if instrument == 'naco27':                                                  
        if 'fwhm' in VLT_NACO:
            fwhm = VLT_NACO['fwhm']
        else:
            tel_diam = VLT_NACO['diam']
            fwhm = VLT_NACO['lambdal']/tel_diam*206265/VLT_NACO['plsc'] 
    elif instrument == 'lmircam':
        if 'fwhm' in LBT:
            fwhm = LBT['fwhm']
        else:
            fwhm = LBT['lambdal']/LBT['diam']*206265/LBT['plsc']                    
    elif not instrument:
        msg = 'One of parameters \'fwhm\' or \'instrument\' must be given'
        raise RuntimeError(msg)
    else:
        raise RuntimeError('Instrument not recognized')   
    return fwhm 


def define_annuli(angle_list, ann, n_annuli, fwhm, radius_int, annulus_width, 
                  delta_rot, verbose):
    """ Function that defines the annuli geometry using the input parameters.
    Returns the parallactic angle threshold, the inner radius and the annulus
    center for each annulus.
    """
    if ann == n_annuli-1:
        inner_radius = radius_int + (ann*annulus_width-1)
    else:                                                                                         
        inner_radius = radius_int + ann*annulus_width
    ann_center = (inner_radius+(annulus_width/2.0))
    pa_threshold = delta_rot * (fwhm/ann_center) / np.pi*180
     
    mid_range = (np.abs(angle_list[-1]) - np.abs(angle_list[0]))/2
    if pa_threshold >= mid_range - mid_range * 0.1:
        new_pa_th = float(mid_range - mid_range * 0.1)
        if verbose:
            msg = 'PA threshold {:.2f} is too big, will be set to {:.2f}'
            print msg.format(pa_threshold, new_pa_th)
        pa_threshold = new_pa_th
                         
    if verbose:
        msg2 = 'Annulus {:}, PA thresh = {:.2f}, Inn radius = {:.2f}, Ann center = {:.2f} '
        print msg2.format(int(ann+1),pa_threshold,inner_radius, ann_center) 
    return pa_threshold, inner_radius, ann_center


def find_indices(angle_list, frame, thr, truncate):  
    """ Returns the indices to be left in pca library.  
    
    # TODO: find a more pythonic way to to this. There must be a more elegant 
    way of dealing with the PA array for finding the needed indices. 
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
    
    half1 = range(0,index_prev)
    half2 = range(index_foll,n)

    if truncate:
        thr = int(n/2)                                                          ### TBD: leaving the n/2 closest frames   
        q = int(thr/2)
        if frame < thr: 
            half1 = range(max(0,index_prev-q),index_prev)
            half2 = range(index_foll,min(index_foll+thr-len(half1),n))
        else:
            half2 = range(index_foll,n)
            half1 = range(max(0,index_prev-thr+len(half2)),index_prev)
    #print half1, half2
    return np.array(half1+half2)


def get_ncomp(data, mode, debug):                                               ### TODO: redifine every m frames?
    """ Defines the number of principal components automatically for each zone,
    annulus or quadrant by minimizing the pixel noise (as the pixel standard
    deviation in the residuals) decay once per zone (for frame 0).              
    """
    ncomp = 0              
    #full_std = np.std(data, axis=0).mean()
    #full_var = np.var(data, axis=0).sum()
    #orig_px_noise = np.mean(np.std(data, axis=1))
    px_noise = []
    px_noise_decay = 1
    #px_noise_i = 1
    #while px_noise_i > input_px_noise*0.05:
    while px_noise_decay > 10e-5:
        ncomp += 1
        V = svd_wrapper(data, mode, ncomp, debug=False, 
                        verbose=False)
        transformed = np.dot(data, V.T)
        reconstructed = np.dot(transformed, V)                  # reference psf
        residuals = data - reconstructed  
        # noise (std of median frame) to be lower than a given thr(?)
         
        px_noise.append(np.std(np.median(residuals, axis=0)))         
        #px_noise_i = px_noise[-1]
        if ncomp>1: px_noise_decay = px_noise[-2] - px_noise[-1]
        #print 'ncomp {:} {:.4f} {:.4f}'.format(ncomp, px_noise[-1], px_noise_decay)
    if debug: print 'ncomp', ncomp
    return ncomp


def do_pca_patch(matrix_quad, frame, angle_list, fwhm, pa_threshold, center,
                 ann_center, svd_mode, ncomp):
    """
    Does the SVD/PCA for each frame patch (small matrix). For each frame we 
    find the frames to be rejected depending on the radial distance from the 
    center.
    """
    if pa_threshold != 0:
        if ann_center > fwhm*10:                            ### TBD: fwhm*10
            indices_left = find_indices(angle_list, frame, pa_threshold, True)
        else:
            indices_left = find_indices(angle_list, frame, pa_threshold, False)
         
        data_ref = matrix_quad[indices_left]
        
        if data_ref.shape[0] <= 10:
            msg = 'Too few frames left in the PCA library.'
            raise RuntimeError(msg)
    else:
        data_ref = matrix_quad
                  
    if center:                                          # removing temporal mean
        data = data_ref - data_ref.mean(axis=0)                     
    else:  
        data = data_ref
        
    curr_frame = matrix_quad[frame]                     # current frame
    
    # Performing SVD/PCA according to "svd_mode" flag
    V = svd_wrapper(data, svd_mode, ncomp, debug=False, verbose=False)          
    
    transformed = np.dot(curr_frame, V.T)
    reconstructed = np.dot(transformed.T, V)                        
    residuals = curr_frame - reconstructed     
    return residuals   


