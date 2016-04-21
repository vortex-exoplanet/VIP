#! /usr/bin/env python

"""
PCA algorithm performed on full frame for ADI, RDI or SDI (IFS data).
"""

from __future__ import division 

__author__ = 'C. Gomez @ ULg'
__all__ = ['pca',
           'pca_optimize_snr']

import numpy as np
import pandas as pd
import pyprind
from skimage import draw
from matplotlib import pyplot as plt
from .utils import svd_wrapper, prepare_matrix, reshape_matrix
from ..calib import (cube_derotate, cube_collapse, check_PA_vector, 
                     check_scal_vector)
from ..conf import timing, timeInit
from ..var import frame_center, dist
from ..stats import descriptive_stats
from .. import phot
from .utils import pca_annulus, scale_cube_for_pca
from .pca_local import find_indices, compute_pa_thresh

import warnings
warnings.filterwarnings("ignore", category=Warning)


def pca(cube, angle_list, cube_ref=None, scale_list=None, ncomp=1, ncomp2=1,
        svd_mode='lapack', scaling=None, mask_center_px=None, source_xy=None,
        delta_rot=1, fwhm=4, collapse='median', full_output=False, verbose=True, 
        debug=False):
    """ Algorithm where the reference PSF and the quasi-static speckle pattern 
    are modeled using Principal Component Analysis. Depending on the input
    parameters this PCA function can work in ADI, RDI or SDI (IFS data) mode.
    
    ADI: 
    if neither a reference cube or a scaling vector are provided, the target 
    cube itself is used to learn the PCs and to obtain a low-rank approximation 
    reference PSF (star + speckles).
    
    RDI + ADI: 
    if a reference cube is provided (triggered by *cube_ref* parameter), its 
    PCs are used to project the target frames and to obtain the reference PSF 
    (star + speckles).
    
    SDI (IFS data):
    if a scaling vector is provided (triggered by *scale_list* parameter) and
    the cube is a 3d array, its assumed it contains 1 frame at multiple 
    spectral channels. The frames are re-scaled to match the longest wavelenght.
    Then PCA is applied on this re-scaled cube where the planet will move 
    radially.
    
    SDI (IFS data) + ADI:
    if a scaling vector is provided (triggered by *scale_list* parameter) and
    the cube is a 4d array [# channels, # adi-frames, Y, X], its assumed it 
    contains several multi-spectral ADI frames. A double PCA is performed, first
    on each ADI multi-spectral frame (using *ncomp* PCs), then using each ADI
    residual to exploit the rotation (using *ncomp2* PCs). 
    
    Several SVD libraries can be used with almost (randsvd stands for randomized 
    SVD) the same result but different computing time.
    
    
    References
    ----------
    KLIP: http://arxiv.org/abs/1207.4197
    pynpoint: http://arxiv.org/abs/1207.6637
    IFS data: http://arxiv.org/abs/1409.6388
    
    Parameters
    ----------
    cube : array_like, 3d
        Input cube.
    angle_list : array_like, 1d
        Corresponding parallactic angle for each frame.    
    cube_ref : array_like, 3d, optional
        Reference library cube. For Reference Star Differential Imaging.
    scale_list : 
        Scaling factors in case of IFS data. Normally, the scaling factors are
        the central channel wavelength divided by the longest wavelength in the
        cube. More thorough approaches can be used to get the scaling factors.
    ncomp : int, optional
        How many PCs are used as a lower-dimensional subspace to project the
        target frames. In ADI ncomp is the number of PCs from the target data,
        in RDI ncomp is the number of PCs from the reference data and in IFS 
        ncomp is the number of PCs from the library of spectral channels. 
    ncomp2 : int, optional
        How many PCs are used for IFS+ADI datacubes in the second stage PCA. 
        ncomp2 goes up to the number of multi-spectral frames. 
    svd_mode : {'lapack', 'eigen', 'randsvd', 'arpack'}, str optional
        Switch for different ways of computing the SVD and selected PCs.
    scaling : {None, 'temp-mean', 'spat-mean', 'temp-standard', 'spat-standard'}
        With None, no scaling is performed on the input data before SVD. With 
        "temp-mean" then temporal px-wise mean subtraction is done, with 
        "spat-mean" then the spatial mean is subtracted, with "temp-standard" 
        temporal mean centering plus scaling to unit variance is done and with
        "spat-standard" spatial mean centering plus scaling to unit variance is
        performed.  
    mask_center_px : None or int
        If None, no masking is done. If an integer > 1 then this value is the
        radius of the circular mask. 
    source_xy : tuple of int, optional 
        For ADI PCA, this triggers a frame rejection in the PCA library. 
        source_xy are the coordinates X,Y of the center of the annulus where the
        PA criterion will be used to reject frames from the library. 
    fwhm : float, optional
        Known size of the FHWM in pixels to be used. Default value is 4.
    delta_rot : int, optional
        Factor for increasing the parallactic angle threshold, expressed in FWHM.
        Default is 1 (excludes 1 FHWM on each side of the considered frame).
    collapse : {'median', 'mean', 'sum', 'trimmean'}, str optional
        Sets the way of collapsing the frames for producing a final image.
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
        Median combination of the de-rotated/re-scaled residuals cube.
    
    If full_output is True then it returns: return pcs, recon, residuals_cube, 
    residuals_cube_ and frame. The PCs are not returned when a PA rejection 
    criterion is applied (when *source_xy* is entered). 
    pcs : array_like, 3d
        Cube with the selected principal components.
    recon : array_like, 3d
        Reconstruction of frames using the selected PCs.
    residuals_res : array_like, 3d 
        Cube of residuals.
    residuals_res_der : array_like, 3d
        Cube of residuals after de-rotation/re-scaling.
        
    In the case of IFS+ADI data the it returns: residuals_cube_channels, 
    residuals_cube_channels_ and frame
    residuals_cube_channels : array_like
        Cube with the residuals of the first stage PCA.
    residuals_cube_channels_ : array_like
        Cube with the residuals of the second stage PCA after de-rotation.        
    """
    #***************************************************************************
    # Helping function
    #***************************************************************************
    def subtract_projection(cube, cube_ref, ncomp, scaling, mask_center_px, 
                            debug, svd_mode, verbose, full_output, indices=None,
                            frame=None):
        """ Subtracts the reference PSF after PCA projection. Returns the cube
        of residuals.
        """
        _, y, x = cube.shape
        if indices is not None and frame is not None:
            matrix = prepare_matrix(cube, scaling, mask_center_px, False)
        else:
            matrix = prepare_matrix(cube, scaling, mask_center_px, verbose)
              
        if cube_ref is not None:
            ref_lib = prepare_matrix(cube_ref, scaling, mask_center_px, verbose)
        else:
            ref_lib = matrix    
        
        if indices is not None and frame is not None:                           # one row (frame) at a time
            ref_lib = ref_lib[indices]
            if ref_lib.shape[0] <= 10:
                msg = 'Too few frames left in the PCA library (<10). '
                msg += 'Try decreasing the parameter delta_rot'
                raise RuntimeError(msg)
            curr_frame = matrix[frame]                     # current frame
    
            V = svd_wrapper(ref_lib, svd_mode, ncomp, False, False)    
            transformed = np.dot(curr_frame, V.T)
            reconstructed = np.dot(transformed.T, V)                        
            residuals = curr_frame - reconstructed 
            if full_output:
                return ref_lib.shape[0], residuals, reconstructed
            else:
                return ref_lib.shape[0], residuals
        else:                                                                   # the whole matrix
            V = svd_wrapper(ref_lib, svd_mode, ncomp, debug, verbose)  
                  
            if verbose: timing(start_time)
            transformed = np.dot(V, matrix.T)
            reconstructed = np.dot(transformed.T, V)
            residuals = matrix - reconstructed
            residuals_res = reshape_matrix(residuals,y,x)
            if full_output:
                return residuals_res, reconstructed, V
            else:
                return residuals_res
    
    #***************************************************************************
    # Validation of input parameters
    #***************************************************************************
    if not cube.ndim>2:
        raise TypeError('Input array is not a 3d or 4d array')
    if not cube.shape[0] == angle_list.shape[0]:
        msg ='Angle list vector has wrong length. It must equal the number of \
        frames in the cube.'
        raise TypeError(msg)
    if source_xy is not None and delta_rot is None or fwhm is None:
        msg = 'Delta_rot or fwhm parameters missing. They are needed for the ' 
        msg += 'PA-based rejection of frames from the library'  
        raise TypeError(msg)
    if cube_ref is not None:
        if not cube_ref.ndim==3:
            raise TypeError('Input reference array is not a cube or 3d array')
        if not cube_ref.shape[1]==cube.shape[1]:
            msg = 'Frames in reference cube and target cube have different size'
            raise TypeError(msg)
        if scale_list is not None:
            raise RuntimeError('RDI + SDI (IFS) is not a valid mode')
        n, y, x = cube_ref.shape
    else:
        if scale_list is not None:
            if cube.ndim==3:
                z, y_in, x_in = cube.shape
            if cube.ndim==4:
                z, n, y_in, x_in = cube.shape
        else:
            n, y, x = cube.shape
    if angle_list is None and scale_list is None:
        msg = 'Either the angles list of scale factors list must be provided'
        raise ValueError(msg)
    if scale_list is not None and np.array(scale_list).ndim>1:
        raise TypeError('Wrong scaling factors list. Must be a vector')
    
    if verbose: start_time = timeInit()
    
    if angle_list is not None:  angle_list = check_PA_vector(angle_list)

    #***************************************************************************
    # scale_list triggers SDI(IFS)
    #***************************************************************************
    if scale_list is not None:
        if ncomp > z:
            ncomp = min(10, z)
            msg = 'Number of PCs too high (max PCs={}), using instead {:} PCs.'
            print msg.format(z, ncomp)
        scale_list = check_scal_vector(scale_list)
        
        #***********************************************************************
        # RDI (IFS): case of 3d cube with multiple spectral channels
        #***********************************************************************
        if cube.ndim==3:
            if verbose:  
                print '{:} spectral channels in IFS cube'.format(z)
            # cube has been re-scaled to have the planets moving radially
            cube, _, y, x, _, _ = scale_cube_for_pca(cube, scale_list)
            residuals_result = subtract_projection(cube, None, ncomp, scaling, 
                                                   mask_center_px,debug,svd_mode, 
                                                   verbose, full_output)
            if full_output:
                residuals_cube = residuals_result[0]
                reconstructed = residuals_result[1] 
                V = residuals_result[2]
                pcs = reshape_matrix(V, y, x)
                recon = reshape_matrix(reconstructed, y, x)
                residuals_cube_,frame,_,_,_,_ = scale_cube_for_pca(residuals_cube, 
                                                        scale_list,
                                                        full_output=full_output,
                                                        inverse=True, y_in=y_in, 
                                                        x_in=x_in)
            else:
                residuals_cube = residuals_result
                frame = scale_cube_for_pca(residuals_cube, scale_list,
                                           full_output=full_output,
                                           inverse=True, y_in=y_in, x_in=x_in)
            if verbose:
                print 'Done re-scaling and combining'
                timing(start_time)
        
        #***********************************************************************
        # RDI (IFS) + ADI: cube with multiple spectral channels + rotation
        # shape of cube: [# channels, # adi-frames, Y, X]
        #***********************************************************************
        elif cube.ndim==4 and angle_list is not None:
            if verbose:  
                print '{:} spectral channels in IFS cube'.format(z)
                print '{:} ADI frames in all channels'.format(n)
            residuals_cube_channels = np.zeros((n, y_in, x_in))
            
            bar = pyprind.ProgBar(n, stream=1, 
                                  title='Looping through ADI frames')
            for i in xrange(n):
                cube_res, _, y, x, _, _ = scale_cube_for_pca(cube[:,i,:,:], 
                                                             scale_list)
                residuals_result = subtract_projection(cube_res, None, ncomp,
                                                       scaling, mask_center_px, 
                                                       debug, svd_mode, False, 
                                                       full_output)
                if full_output:
                    residuals_cube = residuals_result[0]
                    _,frame,_,_,_,_ = scale_cube_for_pca(residuals_cube, 
                                                         scale_list,
                                                         full_output=full_output,
                                                         inverse=True, y_in=y_in, 
                                                         x_in=x_in)
                else:
                    residuals_cube = residuals_result
                    frame = scale_cube_for_pca(residuals_cube, scale_list,
                                               full_output=full_output,
                                               inverse=True,y_in=y_in,x_in=x_in)
                
                residuals_cube_channels[i] = frame
                bar.update()
            
            # de-rotation of the PCA processed channels
            if ncomp2 > n:
                ncomp2 = min(10, n)
                msg = 'Number of PCs too high (max PCs={}), using instead {:} PCs.'
                print msg.format(n, ncomp)
            res_ifs_adi = subtract_projection(residuals_cube_channels, None, 
                                              ncomp2, scaling, mask_center_px, 
                                              debug, svd_mode, False, 
                                              full_output)
            residuals_cube_channels_ = cube_derotate(res_ifs_adi, angle_list)
            frame = cube_collapse(residuals_cube_channels_, mode=collapse)
            if verbose:
                msg = 'Done PCA per ADI multi-spectral frame, de-rotating and '
                msg += 'combining'
                print msg
                timing(start_time)

    #***************************************************************************
    # cube_ref triggers RDI+ADI
    #***************************************************************************
    elif cube_ref is not None:
        if ncomp > n:
            ncomp = min(ncomp,n)
            msg = 'Number of PCs too high (max PCs={}), using instead {:} PCs.'
            print msg.format(n, ncomp)
        residuals_result = subtract_projection(cube, cube_ref, ncomp, scaling, 
                                               mask_center_px, debug, svd_mode, 
                                               verbose, full_output)
        if full_output: 
            residuals_cube = residuals_result[0]
            reconstructed = residuals_result[1] 
            V = residuals_result[2]
            pcs = reshape_matrix(V, y, x)
            recon = reshape_matrix(reconstructed, y, x)
        else:
            residuals_cube = residuals_result
        residuals_cube_ = cube_derotate(residuals_cube, angle_list)
        frame = cube_collapse(residuals_cube_, mode=collapse)
        
        if verbose:
            print 'Done de-rotating and combining'
            timing(start_time)
            
    #***************************************************************************
    # normal ADI PCA 
    #***************************************************************************     
    else:
        if ncomp > n:
            ncomp = min(ncomp,n)
            msg = 'Number of PCs too high (max PCs={}), using instead {:} PCs.'
            print msg.format(n, ncomp)
        
        if source_xy is None:
            residuals_result = subtract_projection(cube, None, ncomp, scaling, 
                                                   mask_center_px, debug, 
                                                   svd_mode,verbose,full_output)
            if full_output: 
                residuals_cube = residuals_result[0]
                reconstructed = residuals_result[1] 
                V = residuals_result[2]
                pcs = reshape_matrix(V, y, x)
                recon = reshape_matrix(reconstructed, y, x)
            else:
                residuals_cube = residuals_result
        else:
            nfrslib = []
            residuals_cube = np.zeros_like(cube)
            recon_cube = np.zeros_like(cube)
            yc, xc = frame_center(cube[0], False)
            x1, y1 = source_xy
            ann_center = dist(yc, xc, y1, x1)
            pa_thr = compute_pa_thresh(ann_center, fwhm, delta_rot)         
            mid_range = np.abs(np.amax(angle_list) - np.amin(angle_list))/2
            if pa_thr >= mid_range - mid_range * 0.1:
                new_pa_th = float(mid_range - mid_range * 0.1)
                if verbose:
                    msg = 'PA threshold {:.2f} is too big, will be set to {:.2f}'
                    print msg.format(pa_thr, new_pa_th)
                pa_thr = new_pa_th     
            
            for frame in xrange(n):
                if ann_center > fwhm*3:                                         # TODO: 3 optimal value? new parameter?
                    ind = find_indices(angle_list, frame, pa_thr, True)
                else:
                    ind = find_indices(angle_list, frame, pa_thr, False)
                
                res_result = subtract_projection(cube, None, ncomp, scaling, 
                                                 mask_center_px, debug, 
                                                 svd_mode,verbose,full_output,
                                                 ind, frame)
                if full_output:
                    nfrslib.append(res_result[0])
                    residual_frame = res_result[1]
                    recon_frame = res_result[2]
                    residuals_cube[frame] = residual_frame.reshape(cube[0].shape) 
                    recon_cube[frame] = recon_frame.reshape(cube[0].shape) 
                else:
                    nfrslib.append(res_result[0])
                    residual_frame = res_result[1]
                    residuals_cube[frame] = residual_frame.reshape(cube[0].shape) 
            
            # number of frames in library printed for each annular quadrant
            if verbose:
                descriptive_stats(nfrslib, verbose=verbose, label='Size LIB: ')
            
             
        residuals_cube_ = cube_derotate(residuals_cube, angle_list)
        frame = cube_collapse(residuals_cube_, mode=collapse)
        if verbose:
            print 'Done de-rotating and combining'
            timing(start_time)
        
    if full_output and cube.ndim<4:
        if source_xy is not None:
            return recon_cube, residuals_cube, residuals_cube_, frame
        else:
            return pcs, recon, residuals_cube, residuals_cube_, frame
    elif full_output and cube.ndim==4:
        return residuals_cube_channels, residuals_cube_channels_, frame
    else:
        return frame
    
    
    
def pca_optimize_snr(cube, angle_list, (source_xy), fwhm, cube_ref=None,
                     mode='full', annulus_width=2, range_pcs=None, 
                     svd_mode='lapack', scaling=None, mask_center_px=None, 
                     fmerit='px', min_snr=0, collapse='median', verbose=True, 
                     full_output=False, debug=False, plot=True):
    """ Optimizes the number of principal components by doing a simple grid 
    search measuring the SNR for a given position in the frame (ADI, RDI). 
    The metric used could be the given pixel's SNR, the maximum SNR in a FWHM 
    circular aperture centered on the given coordinates or the mean SNR in the 
    same circular aperture. They yield slightly different results.
    
    Parameters
    ----------
    cube : array_like, 3d
        Input cube.
    angle_list : array_like, 1d
        Corresponding parallactic angle for each frame.
    source_xy : tuple of floats
        X and Y coordinates of the pixel where the source is located and whose
        SNR is going to be maximized.
    fwhm : float 
        Size of the PSF's FWHM in pixels.
    cube_ref : array_like, 3d, optional
        Reference library cube. For Reference Star Differential Imaging. 
    mode : {'full', 'annular'}, optional
        Mode for PCA processing (full-frame or just in an annulus).
    annulus_width : float, optional
        Width in pixels of the annulus in the case of the "annular" mode. 
    range_pcs : tuple, optional
        The interval of PCs to be tried. If None then the algorithm will find
        a clever way to sample from 1 to 200 PCs. If a range is entered (as 
        (PC_INI, PC_MAX)) a sequential grid will be evaluated between PC_INI
        and PC_MAX with step of 1. If a range is entered (as 
        (PC_INI, PC_MAX, STEP)) a grid will be evaluated between PC_INI and 
        PC_MAX with the given STEP.          
    svd_mode : {'lapack', 'randsvd', 'eigen', 'arpack'}, optional
        Switch for different ways of computing the SVD and selected PCs.
    scaling : {None, 'temp-mean', 'spat-mean', 'temp-standard', 'spat-standard'}
        With None, no scaling is performed on the input data before SVD. With 
        "temp-mean" then temporal px-wise mean subtraction is done, with 
        "spat-mean" then the spatial mean is subtracted, with "temp-standard" 
        temporal mean centering plus scaling to unit variance is done and with
        "spat-standard" spatial mean centering plus scaling to unit variance is
        performed.  
    mask_center_px : None or int, optional
        If None, no masking is done. If an integer > 1 then this value is the
        radius of the circular mask.  
    fmerit : {'px', 'max', 'mean'}
        The function of merit to be maximized. 'px' is *source_xy* pixel's SNR, 
        'max' the maximum SNR in a FWHM circular aperture centered on 
        *source_xy* and 'mean' is the mean SNR in the same circular aperture.  
    min_snr : float
        Value for the minimum acceptable SNR. Setting this value higher will 
        reduce the steps.
    collapse : {'median', 'mean', 'sum', 'trimmean'}, str optional
        Sets the way of collapsing the frames for producing a final image.
    verbose : {True, False}, bool optional
        If True prints intermediate info and timing.
    full_output : {False, True} bool optional
        If True it returns the optimal number of PCs, the final PCA frame for 
        the optimal PCs and a cube with all the final frames for each number 
        of PC that was tried.
    debug : {False, True}, bool optional
        Whether to print debug information or not.
    plot : {True, False}, optional
        Whether to plot the SNR and flux as functions of PCs and final PCA 
        frame or not.

    Returns
    -------
    opt_npc : int
        Optimal number of PCs for given source.
    If full_output is True, the final processed frame, and a cube with all the
    PCA frames are returned along with the optimal number of PCs.
    """    
    def truncate_svd(matrix, angle_list, ncomp, V):    
        """ One SVD computation. Only for full-frame"""        
        transformed = np.dot(V[:ncomp], matrix.T)
        reconstructed = np.dot(transformed.T, V[:ncomp])
        residuals = matrix - reconstructed
        frsize = np.sqrt(matrix.shape[1])                                       # only for square frames 
        residuals_res = reshape_matrix(residuals, frsize, frsize)
        residuals_res_der = cube_derotate(residuals_res, angle_list)
        frame = cube_collapse(residuals_res_der, mode=collapse)
        return frame
    
    def get_snr(matrix, angle_list, cube_ref, y, x, mode, V, fwhm, ncomp, 
                fmerit, full_output):
        if mode=='full':
            frame = truncate_svd(matrix, angle_list, ncomp, V)                  
        elif mode=='annular':
            y_cent, x_cent = frame_center(cube[0])
            annulus_radius = dist(y_cent, x_cent, y, x)
            frame = pca_annulus(cube, angle_list, ncomp, annulus_width, 
                                annulus_radius, cube_ref)
        else:
            raise RuntimeError('Wrong mode.')            
        
        if fmerit=='max':
            yy, xx = draw.circle(y, x, fwhm/2.)
            res = [phot.snr_ss(frame, (x_,y_), fwhm, plot=False, verbose=False, 
                               full_output=True) for y_, x_ in zip(yy, xx)]
            snr_pixels = np.array(res)[:,-1]
            fluxes = np.array(res)[:,2]
            argm = np.argmax(snr_pixels)
            if full_output:
                # integrated fluxes for the max snr
                return np.max(snr_pixels), fluxes[argm], frame
            else:
                return np.max(snr_pixels), fluxes[argm]
        elif fmerit=='px':
            res = phot.snr_ss(frame, (x,y), fwhm, plot=False, verbose=False,
                              full_output=True)
            snrpx = res[-1]
            fluxpx = np.array(res)[2]
            if full_output:
                # integrated fluxes for the given px
                return snrpx, fluxpx, frame
            else:
                return snrpx, fluxpx
        elif fmerit=='mean':
            yy, xx = draw.circle(y, x, fwhm/2.)
            res = [phot.snr_ss(frame, (x_,y_), fwhm, plot=False, verbose=False, 
                               full_output=True) for y_, x_ in zip(yy, xx)]  
            snr_pixels = np.array(res)[:,-1]
            fluxes = np.array(res)[:,2]
            if full_output:
                # mean of the integrated fluxes (shifting the aperture)
                return np.mean(snr_pixels), np.mean(fluxes), frame
            else:                         
                return np.mean(snr_pixels), np.mean(fluxes)
    
    def grid(matrix, angle_list, y, x, mode, V, fwhm, fmerit, step, inti, intf, 
             debug, full_output, truncate=True):
        nsteps = 0
        #n = cube.shape[0]
        snrlist = []
        pclist = []
        fluxlist = []
        if full_output:  frlist = []
        counter = 0
        if debug:  
            print 'Step current grid:', step
            print 'PCs | SNR'
        for pc in range(inti, intf+1, step):
            if full_output:
                snr, flux, frame = get_snr(matrix, angle_list, cube_ref, y, x, 
                                           mode, V, fwhm, pc, fmerit, full_output) 
            else:
                snr, flux = get_snr(matrix, angle_list, cube_ref, y, x, mode, V, 
                                    fwhm, pc, fmerit, full_output)
            if np.isnan(snr):  snr=0
            if nsteps>1 and snr<snrlist[-1]:  counter += 1
            snrlist.append(snr)
            pclist.append(pc)
            fluxlist.append(flux)
            if full_output:  frlist.append(frame)
            nsteps += 1
            if truncate and nsteps>2 and snr<min_snr:  
                if debug:  print 'SNR too small'
                break
            if debug:  print '{} {:.3f}'.format(pc, snr)
            if truncate and counter==5:  break 
        argm = np.argmax(snrlist)
        
        if len(pclist)==2: pclist.append(pclist[-1]+1)
    
        if debug:
            print 'Finished current stage' 
            try:
                pclist[argm+1]
                print 'Interval for next grid: ', pclist[argm-1], 'to', pclist[argm+1]
            except:
                print 'The optimal SNR seems to be outside of the given PC range'
            print
        
        if argm==0:  argm = 1 
        if full_output:
            return argm, pclist, snrlist, fluxlist, frlist
        else: 
            return argm, pclist, snrlist, fluxlist
    
    #---------------------------------------------------------------------------
    if not cube.ndim==3:
        raise TypeError('Input array is not a cube or 3d array')
    
    if verbose: start_time = timeInit()
    n = cube.shape[0]
    x, y = source_xy 
    
    if range_pcs is not None:
        if len(range_pcs)==2:
            pcmin, pcmax = range_pcs
            pcmax = min(pcmax, n)
            step = 1
        elif len(range_pcs)==3:
            pcmin, pcmax, step = range_pcs
            pcmax = min(pcmax, n)
        else:
            msg = 'Range_pcs tuple must be entered as (PC_INI, PC_MAX, STEP) '
            msg += 'or (PC_INI, PC_MAX)'
            raise TypeError(msg)
    else:
        pcmin = 1
        pcmax = 200
        pcmax = min(pcmax, n)
    
    # Getting pcmax principal components (once)
    matrix = prepare_matrix(cube, scaling, mask_center_px, verbose=False)
    if cube_ref is not None:
        ref_lib = prepare_matrix(cube_ref, scaling, mask_center_px, 
                                 verbose=False)
    else:
        ref_lib = matrix    
    V = svd_wrapper(ref_lib, svd_mode, pcmax, False, verbose)
    
    # sequential grid
    if range_pcs is not None:
        grid1 = grid(matrix, angle_list, y, x, mode, V, fwhm, fmerit, step, 
                     pcmin, pcmax, debug, full_output, False)
        if full_output:  argm, pclist, snrlist, fluxlist, frlist = grid1
        else:  argm, pclist, snrlist, fluxlist = grid1
        
        opt_npc = pclist[argm]    
        if verbose:
            print 'Number of steps', len(pclist)
            msg = 'Optimal number of PCs = {}, for SNR={}'
            print msg.format(opt_npc, snrlist[argm])
            print
            timing(start_time)
        
        if full_output: 
            cubeout = np.array((frlist))

        # Plot of SNR as function of PCs  
        if plot:    
            plt.figure(figsize=(8,4))
            ax1 = plt.subplot(211)     
            ax1.plot(pclist, snrlist, '-', alpha=0.5)
            ax1.plot(pclist, snrlist, 'o', alpha=0.5, color='blue')
            ax1.set_xlim(np.array(pclist).min(), np.array(pclist).max())
            ax1.set_ylim(0, np.array(snrlist).max()+1)
            ax1.set_ylabel('SNR')
            ax1.minorticks_on()
            ax1.grid('on', 'major', linestyle='solid', alpha=0.4)
            
            ax2 = plt.subplot(212)
            ax2.plot(pclist, fluxlist, '-', alpha=0.5, color='green')
            ax2.plot(pclist, fluxlist, 'o', alpha=0.5, color='green')
            ax2.set_xlim(np.array(pclist).min(), np.array(pclist).max())
            ax2.set_ylim(0, np.array(fluxlist).max()+1)
            ax2.set_xlabel('Principal components')
            ax2.set_ylabel('Flux in FWHM ap. [ADUs]')
            ax2.minorticks_on()
            ax2.grid('on', 'major', linestyle='solid', alpha=0.4)
            print
            
    # automatic "clever" grid
    else:
        grid1 = grid(matrix, angle_list, y, x, mode, V, fwhm, fmerit, 
                     int(pcmax*0.1), pcmin, pcmax, debug, full_output)
        if full_output:  argm, pclist, snrlist, fluxlist, frlist1 = grid1
        else:  argm, pclist, snrlist, fluxlist = grid1
        
        grid2 = grid(matrix, angle_list, y, x, mode, V, fwhm, fmerit, 
                     int(pcmax*0.05), pclist[argm-1], pclist[argm+1], debug, 
                     full_output)
        if full_output:  argm2, pclist2, snrlist2, fluxlist2, frlist2 = grid2
        else:  argm2, pclist2, snrlist2, fluxlist2  = grid2
        
        grid3 = grid(matrix, angle_list, y, x, mode, V, fwhm, fmerit, 1, 
                     pclist2[argm2-1], pclist2[argm2+1], debug, full_output, 
                     False)
        if full_output:  _, pclist3, snrlist3, fluxlist3, frlist3 = grid3
        else:  _, pclist3, snrlist3, fluxlist3 = grid3
        
        argm = np.argmax(snrlist3)
        opt_npc = pclist3[argm]    
        dfr = pd.DataFrame(np.array((pclist+pclist2+pclist3, 
                                     snrlist+snrlist2+snrlist3,
                                     fluxlist+fluxlist2+fluxlist3)).T)  
        dfrs = dfr.sort(columns=0)
        dfrsrd = dfrs.drop_duplicates()
        ind = np.array(dfrsrd.index)    
        
        if verbose:
            print 'Number of evaluated steps', ind.shape[0]
            msg = 'Optimal number of PCs = {}, for SNR={}'
            print msg.format(opt_npc, snrlist3[argm])
            print
            timing(start_time)
        
        if full_output: 
            cubefrs = np.array((frlist1+frlist2+frlist3))
            cubeout = cubefrs[ind]
    
        # Plot of SNR as function of PCs  
        if plot:   
            plt.figure(figsize=(8,4))   
            ax1 = plt.subplot(211)  
            ax1.plot(np.array(dfrsrd.loc[:,0]), np.array(dfrsrd.loc[:,1]), '-', 
                     alpha=0.5)
            ax1.plot(np.array(dfrsrd.loc[:,0]), np.array(dfrsrd.loc[:,1]), 'o',  
                     alpha=0.5, color='blue')
            ax1.set_xlim(np.array(dfrsrd.loc[:,0]).min(), np.array(dfrsrd.loc[:,0]).max())
            ax1.set_ylim(0, np.array(dfrsrd.loc[:,1]).max()+1)
            #ax1.set_xlabel('')
            ax1.set_ylabel('S/N')
            ax1.minorticks_on()
            ax1.grid('on', 'major', linestyle='solid', alpha=0.4)
            
            ax2 = plt.subplot(212)
            ax2.plot(np.array(dfrsrd.loc[:,0]), np.array(dfrsrd.loc[:,2]), '-', 
                     alpha=0.5, color='green')
            ax2.plot(np.array(dfrsrd.loc[:,0]), np.array(dfrsrd.loc[:,2]), 'o', 
                     alpha=0.5, color='green')
            ax2.set_xlim(np.array(pclist).min(), np.array(pclist).max())
            ax2.set_ylim(0, np.array(fluxlist).max()+1)
            ax2.set_xlabel('Principal components')
            ax2.set_ylabel('Flux in FWHM ap. [ADUs]')
            ax2.minorticks_on()
            ax2.grid('on', 'major', linestyle='solid', alpha=0.4)
            print
    
    finalfr = pca(cube, angle_list, cube_ref, ncomp=opt_npc, svd_mode=svd_mode,  
                  mask_center_px=mask_center_px, scaling=scaling, verbose=False)  
    _ = phot.frame_quick_report(finalfr, fwhm, (x,y), verbose=verbose)
    
    if full_output:
        return opt_npc, finalfr, cubeout
    else:
        return opt_npc


