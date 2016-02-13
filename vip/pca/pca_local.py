#! /usr/bin/env python

"""
Module with local smart pca (annulus-wise) serial and parallel implementations.
"""

from __future__ import division 

__author__ = 'C. Gomez @ ULg'
__all__ = ['pca_adi_annular_quad',
           'pca_rdi_annular']

import numpy as np
import itertools as itt
from scipy import stats
from multiprocessing import Pool, cpu_count
from sklearn.preprocessing import scale
from ..calib import cube_derotate, check_PA_vector
from ..conf import timeInit, timing
from ..conf import eval_func_tuple as EFT 
from ..var import get_annulus_quad
from ..pca.utils import svd_wrapper
from ..stats import descriptive_stats
from ..var import get_annulus



def pca_rdi_annular(array, angle_list, array_ref, radius_int=0, asize=1, 
                    ncomp=1, svd_mode='randsvd', min_corr=0.9, fwhm=4, 
                    full_output=False, verbose=True, debug=False):
    """ Annular PCA with Reference Library + Correlation + standardization
    
    In the case of having a large number of reference images, e.g. for a survey 
    on a single instrument, we can afford a better selection of the library by 
    constraining the correlation with the median of the science dataset and by 
    working on an annulus-wise way. As with other local PCA algorithms in VIP
    the number of principal components can be automatically adjusted by the
    algorithm by minmizing the residuals in the given patch (a la LOCI). 
    
    Parameters
    ----------
    array : array_like, 3d
        Input science cube.
    angle_list : array_like, 1d
        Corresponding parallactic angle for each frame.
    array_ref : array_like, 3d
        Reference library cube. For Reference Star Differential Imaging.
    radius_int : int, optional
        The radius of the innermost annulus. By default is 0, if >0 then the 
        central circular area is discarded.
    asize : int, optional
        The size of the annuli, in FWHM. Default is 3.
    ncomp : int, optional
        How many PCs are kept. If none it will be automatically determined.
    svd_mode : {randsvd, eigen, lapack, arpack, opencv}, optional
        Switch for different ways of computing the SVD and principal components.
    min_corr : int, optional
        Level of linear correlation between the library patches and the median 
        of the science. Deafult is 0.9.
    fwhm : float, optional
        Known size of the FHWM in pixels to be used. Deafult is 4.
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
    def define_annuli(angle_list, ann, n_annuli, fwhm, radius_int, annulus_width, 
                      verbose):
        """ Defining the annuli """
        if ann == n_annuli-1:
            inner_radius = radius_int + (ann*annulus_width-1)
        else:                                                                                         
            inner_radius = radius_int + ann*annulus_width
        ann_center = (inner_radius+(annulus_width/2.0))
        
        if verbose:
            msg2 = 'Annulus {:}, Inn radius = {:.2f}, Ann center = {:.2f} '
            print msg2.format(int(ann+1),inner_radius, ann_center) 
        return inner_radius, ann_center
    
    def fr_ref_correlation(vector, matrix):
        """ Getting the correlations """
        lista = []
        for i in xrange(matrix.shape[0]):
            pears, _ = stats.pearsonr(vector, matrix[i])
            lista.append(pears)
        
        return lista

    def do_pca_annulus(ncomp, matrix, svd_mode, noise_error, data_ref):
        """ Actual PCA for the annulus """
        #V = svd_wrapper(data_ref, svd_mode, ncomp, debug=False, verbose=False)
        V = get_eigenvectors(ncomp, matrix, svd_mode, noise_error=noise_error, 
                             data_ref=data_ref, debug=False)
        # new variables as linear combinations of the original variables in 
        # matrix.T with coefficientes from EV
        transformed = np.dot(V, matrix.T) 
        reconstructed = np.dot(V.T, transformed)
        residuals = matrix - reconstructed.T  
        return residuals, V.shape[0]

    #---------------------------------------------------------------------------

    if not array.ndim == 3:
        raise TypeError('Input array is not a cube or 3d array.')
    if not array.shape[0] == angle_list.shape[0]:
        raise TypeError('Input vector or parallactic angles has wrong length.')
    
    n, y, _ = array.shape
    if verbose:  start_time = timeInit()
    
    angle_list = check_PA_vector(angle_list)
    
    annulus_width = int(asize * fwhm)                                           # equal size for all annuli
    n_annuli = int(np.floor((y/2-radius_int)/annulus_width))    
    if verbose:
        msg = '# annuli = {:}, Ann width = {:}, FWHM = {:.3f}\n'
        print msg.format(n_annuli, annulus_width, fwhm) 
        print 'PCA will be done locally per annulus and per quadrant.\n'
     
    cube_out = np.zeros_like(array)
    for ann in xrange(n_annuli):
        inner_radius, _ = define_annuli(angle_list, ann, n_annuli, fwhm, 
                                        radius_int, annulus_width, verbose) 
        indices = get_annulus(array[0], inner_radius, annulus_width,
                              output_indices=True)
        yy = indices[0]
        xx = indices[1]
                    
        matrix = array[:, yy, xx]                 # shape [nframes x npx_ann] 
        matrix_ref = array_ref[:, yy, xx]
        
        corr = fr_ref_correlation(np.median(matrix, axis=0), matrix_ref)
        indcorr = np.where(np.abs(corr)>=min_corr)
        data_ref = matrix_ref[indcorr]
        nfrslib = data_ref.shape[0]
                
        if nfrslib<5:
            msg = 'Too few frames left (<5) fullfil the given correlation level.'
            msg += 'Try decreasing it'
            raise RuntimeError(msg)

        matrix = scale(matrix, with_mean=True, with_std=True)
        data_ref = scale(data_ref, with_mean=True, with_std=True)        
        
        residuals, ncomps = do_pca_annulus(ncomp, matrix, svd_mode, 10e-3, data_ref)  
        cube_out[:, yy, xx] = residuals  
            
        if verbose:
            print '# frames in LIB = {}'.format(nfrslib)
            print '# PCs = {}'.format(ncomps)
            print 'Done PCA with {:} for current annulus'.format(svd_mode)
            timing(start_time)      
         
    cube_der, frame = cube_derotate(cube_out, angle_list)
    if verbose:
        print 'Done derotating and combining.'
        timing(start_time)
    if full_output:
        return cube_out, cube_der, frame 
    else:
        return frame           


def pca_adi_annular_quad(array, angle_list, radius_int=0, fwhm=4, asize=3, 
                         delta_rot=1, ncomp=1, svd_mode='randsvd', nproc=1,
                         min_frames_pca=10, tol=1e-1, center=True, 
                         full_output=False, verbose=True, debug=False):
    """ Smart PCA (quadrants of annulus version) algorithm. The PCA is computed 
    locally in each quadrant of each annulus. On each annulus we discard 
    reference images taking into account the parallactic angle threshold. 
     
    Depending on parameter *nproc* the algorithm can work with several cores. 
    It's been tested on a Linux and OSX. The ACCELERATE library for linear 
    algebra calcularions, which comes by default in every OSX system, is broken 
    for multiprocessing. Avoid using this function unless you have compiled 
    Python against other linear algebra library. An easy fix is to install 
    latest ANACONDA (2.5 or later) distribution which ships MKL library 
    (replacing the problematic ACCELERATE). On linux with the default 
    LAPACK/BLAS libraries it successfully distributes the processes among all 
    the existing cores. 
    
    Parameters
    ----------
    array : array_like, 3d
        Input cube.
    angle_list : array_like, 1d
        Corresponding parallactic angle for each frame.
    radius_int : int, optional
        The radius of the innermost annulus. By default is 0, if >0 then the 
        central circular area is discarded.
    fwhm : float, optional
        Known size of the FHWM in pixels to be used. Deafult is 4.
    asize : int, optional
        The size of the annuli, in FWHM. Default is 3.
    delta_rot : int, optional
        Factor for increasing the parallactic angle threshold, expressed in FWHM.
        Default is 1 (excludes 1 FHWM on each side of the considered frame).
    ncomp : int, optional
        How many PCs are kept. If none it will be automatically determined.
    svd_mode : {randsvd, eigen, lapack, arpack, opencv}, optional
        Switch for different ways of computing the SVD and principal components.
    nproc : None or int, optional
        Number of processes for parallel computing. If None the number of 
        processes will be set to (cpu_count()/2). By default the algorithm works
        in single-process mode.
    min_frames_pca : int, optional 
        Minimum number of frames in the PCA reference library. Be careful, when
        min_frames_pca <= ncomp, then for certain frames the subtracted low-rank
        approximation is not optimal (getting a 10 PCs out of 2 frames is not
        possible so the maximum number of PCs is used = 2). In practice the 
        resulting frame may be more noisy. It is recommended to decrease 
        delta_rot and have enough frames in the libraries to allow getting 
        ncomp PCs.    
    tol : float, optional
        Stopping criterion for choosing the number of PCs when *ncomp* is None.
        Lower values will lead to smaller residuals and more PCs.
    full_output: boolean, optional
        Whether to return the final median combined image only or with other 
        intermediate arrays.  
    center : {True,False}, bool optional
        Whether to center (mean subtract) the data or not.
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
     
    if verbose:  start_time = timeInit()
    
    angle_list = check_PA_vector(angle_list)
    
    annulus_width = int(asize * fwhm)                # equal size for all annuli
    n_annuli = int(np.floor((y/2-radius_int)/annulus_width))    
    if verbose:
        msg = '# annuli = {:}, Ann width = {:}, FWHM = {:.3f}'
        print msg.format(n_annuli, annulus_width, fwhm) 
        print
        print 'PCA will be done locally per annulus and per quadrant'
        print
     
    if nproc is None:   # Hyper-threading "duplicates" the cores -> cpu_count/2
        nproc = (cpu_count()/2)
    
    #***************************************************************************
    # The annuli are built, and the corresponding PA thresholds for frame 
    # rejection are calculated (at the center of the annulus)
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
         
        #if ncomp is None and verbose:  print '# PCs info for each quadrant:' 
        #***********************************************************************
        # PCA matrix is created for each annular quadrant and centered if needed
        #***********************************************************************
        for quadrant in xrange(4):
            yy = indices[quadrant][0]
            xx = indices[quadrant][1]
            matrix_quad = array[:, yy, xx]          # shape [nframes x npx_quad] 
 
            if center:  matrix_quad = matrix_quad - matrix_quad.mean(axis=0)

            #*******************************************************************
            # For each frame we call the subfunction do_pca_patch that will 
            # do PCA on the small matrix, where some frames are discarded 
            # according to the PA threshold, and return the residuals
            #*******************************************************************
            ncomps = []
            nfrslib = []          
            if nproc==1:
                for frame in xrange(n):    
                    res = do_pca_patch(matrix_quad, frame, angle_list, fwhm,
                                       pa_threshold, center, ann_center, 
                                       svd_mode, ncomp, min_frames_pca, tol,
                                       debug)
                    residuals = res[0]
                    ncomps.append(res[1])
                    nfrslib.append(res[2])
                    cube_out[frame][yy, xx] = residuals 
            elif nproc>1:
                #***************************************************************
                # A multiprocessing pool is created to process the frames in a 
                # parallel way. SVD/PCA is done in do_pca_patch function
                #***************************************************************            
                pool = Pool(processes=int(nproc))
                res = pool.map(EFT, itt.izip(itt.repeat(do_pca_patch), 
                                             itt.repeat(matrix_quad),
                                             range(n), itt.repeat(angle_list),
                                             itt.repeat(fwhm),
                                             itt.repeat(pa_threshold),
                                             itt.repeat(center),
                                             itt.repeat(ann_center),
                                             itt.repeat(svd_mode),
                                             itt.repeat(ncomp),
                                             itt.repeat(min_frames_pca),
                                             itt.repeat(tol),
                                             itt.repeat(debug)))
                res = np.array(res)
                residuals = np.array(res[:,0])
                ncomps = res[:,1]
                nfrslib = res[:,2]
                pool.close()
                for frame in xrange(n):
                    cube_out[frame][yy, xx] = residuals[frame]                          

            # number of frames in library printed for each annular quadrant
            if verbose:
                descriptive_stats(nfrslib, verbose=verbose, label='Size LIB: ')
            # number of PCs printed for each annular quadrant     
            if ncomp is None and verbose:  
                descriptive_stats(ncomps, verbose=verbose, label='Numb PCs: ')
            
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
    
    
### Help functions ********************************************************
    
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
     
    mid_range = np.abs(np.amax(angle_list) - np.amin(angle_list))/2
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
    
    # TODO: find a more pythonic way to to this!
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
    
    # This truncation is done on the annuli after 5*FWHM and the goal is to 
    # keep min(num_frames/2, 100) in the library after discarding those based on
    # the PA threshold
    if truncate:
        thr = min(int(n/2), 100)                                                # TODO: 100 is optimal? new parameter? 
        if frame < thr: 
            half1 = range(max(0,index_prev-int(thr/2)), index_prev)
            half2 = range(index_foll, min(index_foll+thr-len(half1),n))
        else:
            half2 = range(index_foll, min(n, int(thr/2+index_foll)))
            half1 = range(max(0,index_prev-thr+len(half2)), index_prev)
    return np.array(half1+half2)


def do_pca_patch(matrix, frame, angle_list, fwhm, pa_threshold, center,
                 ann_center, svd_mode, ncomp, min_frames_pca, tol, debug):
    """
    Does the SVD/PCA for each frame patch (small matrix). For each frame we 
    find the frames to be rejected depending on the amount of rotation. The
    library is also truncated on the other end (frames too far or which have 
    rotated more) which are more decorrelated to keep the computational cost 
    lower. This truncation is done on the annuli after 5*FWHM and the goal is
    to keep min(num_frames/2, 100) in the library. 
    """
    if pa_threshold != 0:
        if ann_center > fwhm*5:                                                 # TODO: 5*FWHM optimal? new parameter?
            indices_left = find_indices(angle_list, frame, pa_threshold, True)
        else:
            indices_left = find_indices(angle_list, frame, pa_threshold, False)
         
        data_ref = matrix[indices_left]
        
        if data_ref.shape[0] <= min_frames_pca:
            msg = 'Too few frames left in the PCA library. '
            msg += 'Try decreasing either delta_rot or min_frames_pca.'
            raise RuntimeError(msg)
    else:
        data_ref = matrix
                  
    if center:                                          # removing temporal mean
        data = data_ref - data_ref.mean(axis=0)                     
    else:  
        data = data_ref
        
    curr_frame = matrix[frame]                     # current frame
    
    V = get_eigenvectors(ncomp, data, svd_mode, noise_error=tol, debug=False)        
    
    transformed = np.dot(curr_frame, V.T)
    reconstructed = np.dot(transformed.T, V)                        
    residuals = curr_frame - reconstructed     
    return residuals, V.shape[0], data_ref.shape[0]   


# Also used in pca_rdi_annular -------------------------------------------------
def get_eigenvectors(ncomp, data, svd_mode, noise_error=1e-3, max_evs=200, 
                     data_ref=None, debug=False):
    """ Choosing the size of the PCA truncation by Minimizing the residuals
    when ncomp set to None.
    """
    if data_ref is None:
        data_ref = data
    
    if ncomp is None:
        # Defines the number of PCs automatically for each zone (quadrant) by 
        # minimizing the pixel noise (as the pixel STDDEV of the residuals) 
        # decay once per zone         
        ncomp = 0              
        #full_std = np.std(data, axis=0).mean()
        #full_var = np.var(data, axis=0).sum()
        #orig_px_noise = np.mean(np.std(data, axis=1))
        px_noise = []
        px_noise_decay = 1
        # The eigenvectors (SVD/PCA) are obtained once    
        V_big = svd_wrapper(data_ref, svd_mode, min(data_ref.shape[0], max_evs),
                            False, False)
        # noise (stddev of residuals) to be lower than a given thr              
        while px_noise_decay >= noise_error:
            ncomp += 1
            V = V_big[:ncomp]
            transformed = np.dot(data, V.T)
            reconstructed = np.dot(transformed, V)                  
            residuals = data - reconstructed  
            px_noise.append(np.std((residuals)))         
            if ncomp>1: px_noise_decay = px_noise[-2] - px_noise[-1]
            #print 'ncomp {:} {:.4f} {:.4f}'.format(ncomp,px_noise[-1],px_noise_decay)
        
        if debug: print 'ncomp', ncomp
        
    else:
        # Performing SVD/PCA according to "svd_mode" flag
        V = svd_wrapper(data_ref, svd_mode, ncomp, debug=False, verbose=False)   
        
    return V
