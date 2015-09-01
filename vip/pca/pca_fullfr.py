#! /usr/bin/env python

"""
PCA algorithm performed on full frame.
"""

from __future__ import division 

__author__ = 'C. Gomez @ ULg, V. Christiaens @ U.Chile/ULg'
__all__ = ['pca',
           'pca_optimize_snr',
           'scale_cube']

import pdb
import copy
import numpy as np
from skimage import draw
from .utils import svd_wrapper, prepare_matrix, reshape_matrix
from ..calib import (scale_cube, cube_derotate, cube_rescaling, check_PA_vector,
                     check_scal_vector)
from ..conf import timing, timeInit
from ..var import frame_center
from .. import phot
from .pca_local import annular_pca, subannular_pca


def pca(cube, var_list,svd_mode='randsvd', ncomp=1, center='temporal', 
        radius_int=None, full_output=False, verbose=True, debug=False,
        variation='adi'):
    """ Algorithm where the PSF and the quasi-static speckle pattern is modeled 
    through PCA, using all the frames in the cube (as in KLIP or pynpoint).
    Several SVD methods are explored.
    Important: it is assumed that the input cube is already centered with the 
    star on its middle pixel.
    
    Parameters
    ----------
    cube : array_like, 3d
        Input cube.
    var_list : array_like, 1d
        Vector of parallactic angle (if variation = 'adi') or scaling factor 
        (if variation = 'ifs'). 
        Note: in case of ifs data, the scaling factor of each channel of 
        wavelength lambda_zz is approximately: lambda_ref/lambda_zz, where 
        lambda_ref is some ref wavelength (e.g. the longest one in the cube).
        See Pueyo et al. 2015 for a more precise way of computing them.
    svd_mode : {randsvd, eigen, lapack, arpack, opencv}, optional
        Switch for different ways of computing the SVD and selected PCs.
    ncomp : int, optional
        How many PCs are kept. 
    center : {None, 'spatial', 'temporal', 'global'}, optional
        Whether to remove the mean of the data or not and which one. 
    radius_int : Int
        If 0, no masking is done. If an integer > 1 then this value is the
        radius of the circular mask. 
    full_output: boolean, optional
        Whether to return the final median combined image only or with other 
        intermediate arrays.  
    verbose : {True, False}, bool optional
        If True prints intermediate info and timing. 
    debug : {False, True}, bool optional
        Whether to print debug information or not.
    variation: {adi, ifs}, optional
        Choose the variation present in the cube to disentangle speckles from 
        planets.
    
    Returns
    -------
    frame : array_like, 2d    
        Median combination of the de-rotated/de-scaled cube.
    If full_output is True:  
    pcs : array_like, 3d
        Cube with the selected principal components.
    recon : array_like, 3d
        Reconstruction of frames using the selected PCs.
    residual_res : array_like, 3d 
        Cube of residuals.
    residual_res_var : array_like, 3d
        Cube of residuals after de-rotation/de-scaling.
        
    """
    if not cube.ndim==3:
        raise TypeError('Input array is not a cube or 3d array')
    n, y_in, x_in = cube.shape

    if verbose: start_time = timeInit()
 
    if variation == 'ifs':
        var_list = check_scal_vector(var_list)
        array,_,y,x,cy,cx = scale_cube(cube,var_list)
    elif variation == 'adi':
        var_list = check_PA_vector(var_list)
        array = cube
        y, x = y_in, x_in
    else:
        raise ValueError("Please choose a valid variation: 'ifs' or 'adi'")


    if ncomp > n:
        ncomp = n
        print 'Number of PCs too high, set to maximum ({:})'.format(n)
    
    mask_center_px = radius_int
    if radius_int == 0: mask_center_px = None
    matrix_pca = prepare_matrix(array, center, mask_center_px, verbose) 
    V = svd_wrapper(matrix_pca, svd_mode, ncomp, debug, verbose)
    if verbose: timing(start_time)
    transformed = np.dot(V, matrix_pca.T)
    reconstructed = np.dot(transformed.T, V)
    residuals = matrix_pca - reconstructed
    residual_res = reshape_matrix(residuals,y,x)
        
    if variation == 'ifs':
        if full_output:
            residual_res_var,frame,_,_,_,_ = scale_cube(residual_res,
                                                        var_list,
                                                        full_output=full_output,
                                                        inverse=True,y_in=y_in,
                                                        x_in=x_in)
        else:
            frame = scale_cube(residual_res,var_list,full_output=full_output,
                               inverse=True,y_in=y_in,x_in=x_in)
    elif variation == 'adi':
        residual_res_var, frame = cube_derotate(residual_res, var_list)

    if verbose:
        print 'Done derotating and combining'
        timing(start_time)
        
    if full_output:
        pcs = reshape_matrix(V, y, x)
        recon = reshape_matrix(reconstructed, y, x)
        return pcs, recon, residual_res, residual_res_var, frame
    else:
        return frame
    

    
def pca_optimize_snr(cube, var_list, y, x, fwhm_in, svd_mode='randsvd', 
                     radius_int=5, min_snr=0, verbose=True, full_output=False, 
                     debug=False,variation='adi',pca_method=pca,strict=True,**kwargs):
    """ Optimizes the number of principal components by doing a simple grid 
    search measuring the SNR for a given position in the frame. The mean SNR 
    in a 1*FWHM circular aperture centered on the given pixel is the metric
    used instead of the given pixel's SNR (which may lead to false maximums).
    Important: it is assumed that the input cube is already centered with the 
    star on its middle pixel.
    
    Parameters
    ----------
    cube : array_like, 3d
        Input cube.
    var_list : array_like, 1d
        Vector of parallactic angle (if variation = 'adi') or scaling factor 
        (if variation = 'ifs'). 
        Note: in case of ifs data, the scaling factor of each channel of 
        wavelength lambda_zz is approximately: lambda_ref/lambda_zz, where 
        lambda_ref is some ref wavelength (e.g. the longest one in the cube).
        See Pueyo et al. 2015 for a more precise way of computing them.
    y, x : int
        Y and X coordinates of the pixel where the source is located and whose
        SNR is going to be maximized.
    fwhm : float
        Size of the PSF's FWHM in pixels. 
    svd_mode : {randsvd, eigen, lapack, arpack, opencv}, optional
        Switch for different ways of computing the SVD and selected PCs.
    radius_int : Int
        If 0, no masking is done. If an integer > 1 then this value is the
        radius of the circular mask.  
    min_snr : float
        Value for the minimum acceptable SNR. Setting this value higher will 
        reduce the steps.
    verbose : {True, False}, bool optional
        If True prints intermediate info and timing.
    full_output : {False, True} bool optional
        Whether to return the optimal number of PCs alone or along with the 
        final PCA processed frame.
    debug : {False, True}, bool optional
        Whether to print debug information or not.
    variation: {adi, ifs}, optional
        Choose the variation present in the cube to disentangle speckles from 
        planets.
    pca_method: {pca,annular_pca,subannular_pca}, optional
        Choose which pca algorithm to use: full frame (pca), annular 
        (annular_pca), or by portions of annulus (subannular_pca)
    strict: bool, {True, False}, optional
        Whether the algorithm should raise an error if the SNR is too low at the
        given position. If False, it returns opt_npc = 1, and (if full_ouput) 
        the pca final_fr obtained with pc = 1.

    Returns
    -------
    opt_npc : int
        Optimal number of PCs for given source.
    If full_output is True, the final processed frame is also returned:
    finalfr, opt_npc: array, int
    
    """
    if not cube.ndim==3:
        raise TypeError('Input array is not a cube or 3d array')
    
    if verbose: start_time = timeInit()

    def get_snr(cube, var_list, y, x, svd_mode, fwhm_in, ncomp, variation):
        frame = pca_method(cube, var_list, ncomp=ncomp, full_output=False, 
                           verbose=False, radius_int=radius_int, 
                           svd_mode=svd_mode,variation=variation,**kwargs)
        yy, xx = draw.circle(y, x, fwhm_in/2.)
        snr_pixels = [phot.snr_student(frame, y_, x_, fwhm_in, plot=False, 
                                       verbose=False) for y_, x_ in zip(yy, xx)]
        return np.mean(snr_pixels)

    n = cube.shape[0]
    nsteps = 0
    step1 = int(np.sqrt(n))
#    step1 = int(n/np.percentile(range(n), 10))
    snrlist = []
    pclist = []
    if debug:  print 'Step 1st grid:', step1
    for pc in range(1, n+1, step1):
        snr = get_snr(cube, var_list, y, x, svd_mode, fwhm_in, ncomp=pc,
                      variation=variation)
        nsteps += 1
        if nsteps>1 and snr<min_snr:  break
        snrlist.append(snr)
        pclist.append(pc)
        if debug:  print '{} {:.3f}'.format(pc, snr)
    argm = np.argmax(snrlist)
    argm_a = argm-1
    argm_b = argm+1
    if argm == 0: argm_a = argm
    if argm == len(snrlist)-1: argm_b = argm
   
    snrlist2 = []
    pclist2 = []
    
    if debug:
        print 'Finished stage 1', argm, pclist[argm], 
        print pclist[argm-1], pclist[argm+1]+1
        print
    
    if debug:  print 'Step 2nd grid:', 1
    for pc in range(pclist[argm_a], pclist[argm_b]+1, 1):
        snr = get_snr(cube, var_list, y, x, svd_mode, fwhm_in, ncomp=pc,
                      variation=variation)
        nsteps += 1
        if snr<min_snr:  break
        snrlist2.append(snr)
        pclist2.append(pc)
        if debug:  print '{} {:.3f}'.format(pc, snr)
    if len(snrlist2)==0:
        if strict:
            msg = 'SNR too low at given position. Optimization failed'
            raise RuntimeError(msg)
        else:
            print "Warning: Optimization failed; SNR too low!"
            print "opt_npc set to 1 and opt_snr set to ", min_snr
            snrlist2.append(min_snr)
            pclist2.append(1)
    
    argm2 = np.argmax(snrlist2)    
    
    if debug:  
        print 'Finished stage 2', argm2, pclist2[argm2]
        print '# of SVDs', nsteps
        print
    
    opt_npc = pclist2[argm2]
    if verbose:
        msg = 'Optimal # of PCs = {} for mean SNR = {:.3f}'
        print msg.format(opt_npc, snrlist2[argm2])
        timing(start_time)
        
    finalfr = pca_method(cube, var_list, ncomp=opt_npc, full_output=False, 
                         verbose=False, radius_int=radius_int, 
                         svd_mode=svd_mode, variation=variation,**kwargs)
    _ = phot.frame_quick_report(finalfr, fwhm_in, y, x, verbose=verbose)
    
    if full_output:
        return finalfr, opt_npc, snrlist2[argm2]
    else:
        return opt_npc
