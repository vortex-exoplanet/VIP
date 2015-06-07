#! /usr/bin/env python

"""
PCA algorithm performed on full frame.
"""

from __future__ import division 

__author__ = 'C. Gomez @ ULg'
__all__ = ['pca',
           'pca_optimize_snr']

import numpy as np
from skimage import draw
from .utils import svd_wrapper, prepare_matrix, reshape_matrix
from ..calib import cube_derotate
from ..conf import timing, timeInit
from .. import phot


def pca(cube, angle_list, svd_mode='randsvd', ncomp=1, center='temporal', 
        mask_center_px=None, full_output=False, verbose=True, debug=False):
    """ Algorithm where the PSF and the quasi-static speckle pattern is modeled 
    through PCA, using all the frames in the cube (as in KLIP or pynpoint).
    Several SVD methods are explored.
    
    Parameters
    ----------
    cube : array_like, 3d
        Input cube.
    angle_list : array_like, 1d
        Corresponding parallactic angle for each frame.
    svd_mode : {randsvd, eigen, lapack, arpack, opencv}, optional
        Switch for different ways of computing the SVD and selected PCs.
    ncomp : int, optional
        How many PCs are kept. 
    center : {None, 'spatial', 'temporal', 'global'}, optional
        Whether to remove the mean of the data or not and which one. 
    mask_center_px : None or int
        If None, no masking is done. If an integer > 1 then this value is the
        radius of the circular mask. 
    full_output: boolean, optional
        Whether to return the final median combined image only or with other 
        intermediate arrays.  
    verbose : {True, False}, bool optional
        If True prints intermediate info and timing. 
    debug : {False, True}, bool optional
        Whether to print debug information or not.
    
    Returns
    -------
    frame : array_like, 2d    
        Median combination of the de-rotated cube.
    If full_output is True:  
    pcs : array_like, 3d
        Cube with the selected principal components.
    recon : array_like, 3d
        Reconstruction of frames using the selected PCs.
    residuals_res : array_like, 3d 
        Cube of residuals.
    residuals_res_der : array_like, 3d
        Cube of residuals after de-rotation.
        
    """
    array = cube
    if not array.ndim==3:
        raise TypeError('Input array is not a cube or 3d array')
    n, y, x = array.shape
    
    if verbose: start_time = timeInit()
     
    if ncomp > n:
        ncomp = n
        print 'Number of PCs too high, set to maximum ({:})'.format(n)
    
    matrix_pca = prepare_matrix(array, center, mask_center_px, verbose)             
    V = svd_wrapper(matrix_pca, svd_mode, ncomp, debug, verbose)
    if verbose: timing(start_time)
    transformed = np.dot(V, matrix_pca.T)
    reconstructed = np.dot(transformed.T, V)
    residuals = matrix_pca - reconstructed
    residuals_res = reshape_matrix(residuals,y,x)
            
    residuals_res_der, frame = cube_derotate(residuals_res, angle_list)
    if verbose:
        print 'Done derotating and combining'
        timing(start_time)
        
    if full_output:
        pcs = reshape_matrix(V, y, x)
        recon = reshape_matrix(reconstructed, y, x)
        return pcs, recon, residuals_res, residuals_res_der, frame
    else:
        return frame
    
    
def pca_optimize_snr(cube, angle_list, y, x, fwhm, svd_mode='randsvd', 
                     mask_center_px=5, min_snr=0, verbose=True, 
                     output_frame=False, debug=False):
    """ Optimizes the number of principal components by doing a simple grid 
    search measuring the SNR for a given position in the frame. The mean SNR 
    in a 1*FWHM circular aperture centered on the given pixel is the metric
    used instead of the given pixel's SNR (which may lead to false maximums).
    
    Parameters
    ----------
    cube : array_like, 3d
        Input cube.
    angle_list : array_like, 1d
        Corresponding parallactic angle for each frame.
    y, x : int
        Y and X coordinates of the pixel where the source is located and whose
        SNR is going to be maximized.
    fwhm : float 
        Size of the PSF's FWHM in pixels. 
    svd_mode : {randsvd, eigen, lapack, arpack, opencv}, optional
        Switch for different ways of computing the SVD and selected PCs.
    mask_center_px : None or int
        If None, no masking is done. If an integer > 1 then this value is the
        radius of the circular mask.  
    min_snr : float
        Value for the minimum acceptable SNR. Setting this value higher will 
        reduce the steps.
    verbose : {True, False}, bool optional
        If True prints intermediate info and timing.
    output_frame : {False, True} bool optional
        Whether to return the final PCA processed frame with the optimal number
        of PCs or not.
    debug : {False, True}, bool optional
        Whether to print debug information or not.

    Returns
    -------
    opt_npc : int
        Optimal number of PCs for given source.
    If output_frame is True, the final processed frame is returned.
    
    """
    array = cube
    if not array.ndim==3:
        raise TypeError('Input array is not a cube or 3d array')
    
    if verbose: start_time = timeInit()
    
    def get_snr(cube, angle_list, y, x, svd_mode, fwhm, ncomp):
        frame = pca(cube, angle_list, ncomp=ncomp, full_output=False, 
                    verbose=False, mask_center_px=mask_center_px, 
                    svd_mode=svd_mode)
        yy, xx = draw.circle(y, x, fwhm/2.)
        snr_pixels = [phot.snr_student(frame, y_, x_, fwhm, plot=False, 
                                       verbose=False) for y_, x_ in zip(yy, xx)]
        return np.mean(snr_pixels)

    n = array.shape[0]
    nsteps = 0
    step1 = int(np.percentile(range(n), 10))
    snrlist = []
    pclist = []
    counter = 0
    if debug:  print 'Step 1st grid:', step1
    for pc in range(1, n+1, step1):
        snr = get_snr(cube, angle_list, y, x, svd_mode, fwhm, ncomp=pc)
        nsteps += 1
        if nsteps>1 and snr<min_snr:  break
        if nsteps>1 and snr<snrlist[-1]:  counter += 1
        snrlist.append(snr)
        pclist.append(pc)
        if debug:  print '{} {:.3f}'.format(pc, snr)
        if counter==3:  break 
    argm = np.argmax(snrlist)
    if argm==0:  return 1
    
    snrlist2 = []
    pclist2 = []
    
    if debug:
        print 'Finished stage 1', argm, pclist[argm], 
        print pclist[argm-1], pclist[argm+1]+1
        print
    
    if debug:  print 'Step 2nd grid:', 1
    for pc in range(pclist[argm-1], pclist[argm+1]+1, 1):
        snr = get_snr(cube, angle_list, y, x, svd_mode, fwhm, ncomp=pc)
        nsteps += 1
        if snr<min_snr:  break
        snrlist2.append(snr)
        pclist2.append(pc)
        if debug:  print '{} {:.3f}'.format(pc, snr)
    if len(snrlist2)==0:  
        msg = 'SNR too low at given position. Optimization failed'
        if verbose:  print msg
        return 1
    
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
        
    finalfr = pca(cube, angle_list, ncomp=opt_npc, full_output=False, 
                  verbose=False, mask_center_px=mask_center_px, 
                  svd_mode=svd_mode)    
    _ = phot.frame_quick_report(finalfr, fwhm, y, x, verbose=verbose)
    
    if output_frame:
        return finalfr
    else:
        return opt_npc


