#! /usr/bin/env python

"""
PCA algorithm performed on full frame for ADI or RDI.
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
from ..var import frame_center, dist
from .. import phot
from .utils import pca_annulus


def pca(cube, angle_list, cube_ref=None, svd_mode='randsvd', ncomp=1, 
        scaling='standard', mask_center_px=None, full_output=False, verbose=True, 
        debug=False):
    """ Algorithm where the reference PSF and the quasi-static speckle pattern 
    is modeled using Principal Component Analysis. PCA can be done on a matrix
    with all the frames in the ADI cube or a matrix of frames from a reference 
    star (RDI). Several SVD libraries can be used.
    
    References
    ----------
    KLIP: http://arxiv.org/abs/1207.4197
    pynpoint: http://arxiv.org/abs/1207.6637
    
    Parameters
    ----------
    cube : array_like, 3d
        Input cube.
    angle_list : array_like, 1d
        Corresponding parallactic angle for each frame.
    cube_ref : array_like, 3d, optional
        Reference library cube. For Reference Star Differential Imaging.
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
    if not cube.ndim==3:
        raise TypeError('Input array is not a cube or 3d array')
    n, y, x = cube.shape
    
    if cube_ref is not None:
        n, _, _ = cube_ref.shape
    
    if verbose: start_time = timeInit()
     
    if ncomp > n:
        ncomp = 10
        print 'Number of PCs too high, using instead {:} PCs.'.format(ncomp)
    
    matrix = prepare_matrix(cube, scaling, mask_center_px, verbose)
    if cube_ref is not None:
        ref_lib = prepare_matrix(cube_ref, scaling, mask_center_px, verbose)
    else:
        ref_lib = matrix           
    V = svd_wrapper(ref_lib, svd_mode, ncomp, debug, verbose)
    if verbose: timing(start_time)
    transformed = np.dot(V, matrix.T)
    reconstructed = np.dot(transformed.T, V)
    residuals = matrix - reconstructed
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
    
    
    
def pca_optimize_snr(cube, angle_list, y, x, fwhm, mode='full', 
                     annulus_width=2, svd_mode='randsvd', mask_center_px=5,
                     fmerit='px', min_snr=0, verbose=True, full_output=False, 
                     debug=False):
    """ Optimizes the number of principal components by doing a simple grid 
    search measuring the SNR for a given position in the frame. The metric
    used could be the given pixel's SNR, the maximun SNR in a fwhm circular
    aperture centred on the given coordinates or the mean SNR in the same
    circular aperture. They yield slightly different results.
    
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
    mode : {'full', 'annular'}, optional
        Mode for PCA processing (full-frame or just in an annulus).
    annulus_width : float, optional
        Width in pixels of the annulus in the case of the "annular" mode. 
    svd_mode : {randsvd, eigen, lapack, arpack, opencv}, optional
        Switch for different ways of computing the SVD and selected PCs.
    mask_center_px : None or int
        If None, no masking is done. If an integer > 1 then this value is the
        radius of the circular mask.  
    fmerit : {'px', 'max', 'mean'}
        The metric to be maximized. 'px' is the given pixel's SNR, 'max' the 
        maximum SNR in a FWHM circular aperture centered on the given coordinates 
        and 'mean' is the mean SNR in the same circular aperture.  
    min_snr : float
        Value for the minimum acceptable SNR. Setting this value higher will 
        reduce the steps.
    verbose : {True, False}, bool optional
        If True prints intermediate info and timing.
    full_output : {False, True} bool optional
        Whether to return the final PCA processed frame with the optimal number
        of PCs or not.
    debug : {False, True}, bool optional
        Whether to print debug information or not.

    Returns
    -------
    opt_npc : int
        Optimal number of PCs for given source.
    If full_output is True, the final processed frame is returned along with 
    the optimal number of principal components.
    """    
    def get_snr(cube, angle_list, y, x, mode, svd_mode, fwhm, ncomp, fmerit):
        if mode=='full':
            frame = pca(cube, angle_list, ncomp=ncomp, full_output=False, 
                        verbose=False, mask_center_px=mask_center_px, 
                        svd_mode=svd_mode)
        elif mode=='annular':
            y_cent, x_cent = frame_center(cube[0])
            annulus_radius = dist(y_cent, x_cent, y, x)
            frame = pca_annulus(cube, angle_list, ncomp, annulus_width, 
                                annulus_radius)
        else:
            raise RuntimeError('Wrong mode.')            
        
        if fmerit=='max':
            yy, xx = draw.circle(y, x, fwhm/2.)
            snr_pixels = [phot.snr_ss(frame, y_, x_, fwhm, plot=False, 
                                      verbose=False) for y_, x_ in zip(yy, xx)]
            return np.max(snr_pixels)
        elif fmerit=='px':
            return phot.snr_ss(frame, y, x, fwhm, plot=False, verbose=False)
        elif fmerit=='mean':
            yy, xx = draw.circle(y, x, fwhm/2.)
            snr_pixels = [phot.snr_ss(frame, y_, x_, fwhm, plot=False, 
                                      verbose=False) for y_, x_ in zip(yy, xx)]                                      
            return np.mean(snr_pixels)
    
    def grid(cube, angle_list, y, x, mode, svd_mode, fwhm, fmerit, step, inti, 
             intf, debug):
        nsteps = 0
        #n = cube.shape[0]
        snrlist = []
        pclist = []
        counter = 0
        if debug:  print 'Step current grid:', step
        for pc in range(inti, intf+1, step):
            snr = get_snr(cube, angle_list, y, x, mode, svd_mode, fwhm, 
                          pc, fmerit)
            if nsteps>2 and snr<min_snr:  
                print 'SNR too small'
                break
            if nsteps>1 and snr<snrlist[-1]:  counter += 1
            #if len(snrlist)>8 and np.all(np.diff(np.array(snrlist))[:-8] < 0 ) :
            #    print 'SNR decreasing'
            snrlist.append(snr)
            pclist.append(pc)
            nsteps += 1
            if debug: 
                print '{} {:.3f}'.format(pc, snr)
            if counter==3:  
                break 
        argm = np.argmax(snrlist)
        
        if len(pclist)==2: pclist.append(pclist[-1]+1)
    
        if debug:
            print 'Finished current stage' 
            print 'Interval for next grid: ', pclist[argm-1], 'to', pclist[argm+1]+1
            print
        
        if argm==0:  
            return 1, pclist, nsteps
        else:  
            return argm, pclist, nsteps
    
    #----------------------------------------------------
    if not cube.ndim==3:
        raise TypeError('Input array is not a cube or 3d array')
    
    if verbose: start_time = timeInit()
    n = cube.shape[0]
    
    # Up to min(n, 150) principal components. More isn't very realistic for any
    # ADI dataset I've tried. 
    argm, pclist, nsteps = grid(cube, angle_list, y, x, mode, svd_mode, fwhm, 
                                fmerit, 20, 1, min(150, n), debug)
    
    argm2, pclist2, nsteps2 = grid(cube, angle_list, y, x, mode, svd_mode, fwhm,
                                   fmerit, 10, pclist[argm-1], pclist[argm+1], 
                                   debug)
    
    argm3, pclist3, nsteps3 = grid(cube, angle_list, y, x, mode, svd_mode, fwhm, 
                                   fmerit, 1, pclist2[argm2-1], pclist2[argm2+1], 
                                   debug)
    
    if debug:  
        print '# of SVDs', nsteps+nsteps2+nsteps3
        print
    
    opt_npc = pclist3[argm3]
    if verbose:
        msg = 'Optimal # of PCs = {}'
        print msg.format(opt_npc)
        print
        timing(start_time)
        
    finalfr = pca(cube, angle_list, ncomp=opt_npc, full_output=False, 
                  verbose=False, mask_center_px=mask_center_px, 
                  svd_mode=svd_mode)    
    _ = phot.frame_quick_report(finalfr, fwhm, y, x, verbose=verbose)
    
    if full_output:
        return finalfr
    else:
        return opt_npc, finalfr


