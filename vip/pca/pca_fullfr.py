#! /usr/bin/env python

"""
PCA algorithm performed on full frame for ADI, RDI or SDI (IFS data).
"""

from __future__ import division 

__author__ = 'C. Gomez @ ULg'
__all__ = ['pca',
           'pca_optimize_snr']

import numpy as np
import pyprind
from skimage import draw
from .utils import svd_wrapper, prepare_matrix, reshape_matrix
from ..calib import cube_derotate, check_PA_vector, check_scal_vector
from ..conf import timing, timeInit
from ..var import frame_center, dist
from .. import phot
from .utils import pca_annulus, scale_cube_for_pca


def pca(cube, angle_list, cube_ref=None, scale_list=None, ncomp=1, ncomp2=1,
        svd_mode='randsvd', scaling='mean', mask_center_px=None, 
        full_output=False, verbose=True, debug=False):
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
    svd_mode : {'randsvd', 'eigen', 'lapack', 'arpack', 'opencv'}, str optional
        Switch for different ways of computing the SVD and selected PCs.
    scaling : {'mean', 'standard', None}, optional
        If "mean" then temporal px-wise mean subtraction is done, if "standard" 
        then mean centering plus scaling to unit variance is done. With None, no
        scaling is performed on the input data before SVD.
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
        Median combination of the de-rotated/re-scaled residuals cube.
    
    If full_output is True then it returns: return pcs, recon, residuals_cube, 
    residuals_cube_ and frame.
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
                            debug, svd_mode, verbose, full_output):
        """ Subtracts the reference PSF after PCA projection. Returns the cube
        of residuals.
        """
        _, y, x = cube.shape
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
        if full_output:
            return residuals_res, reconstructed, V
        else:
            return residuals_res
    
    #***************************************************************************
    # Validation of input parameters
    #***************************************************************************
    if not cube.ndim>2:
        raise TypeError('Input array is not a 3d or 4d array')
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
                ncomp = min(10, n)
                msg = 'Number of PCs too high (max PCs={}), using instead {:} PCs.'
                print msg.format(n, ncomp)
            res_ifs_adi = subtract_projection(residuals_cube_channels, None, 
                                              ncomp2, scaling, mask_center_px, 
                                              debug, svd_mode, False, 
                                              full_output)
            residuals_cube_channels_, frame = cube_derotate(res_ifs_adi, 
                                                            angle_list)
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
        residuals_cube_, frame = cube_derotate(residuals_cube, angle_list)
        if verbose:
            print 'Done de-rotating and combining'
            timing(start_time)
            
    #***************************************************************************
    # normal ADI 
    #***************************************************************************     
    else:
        if ncomp > n:
            ncomp = min(ncomp,n)
            msg = 'Number of PCs too high (max PCs={}), using instead {:} PCs.'
            print msg.format(n, ncomp)
        residuals_result = subtract_projection(cube, None, ncomp, scaling, 
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
        residuals_cube_, frame = cube_derotate(residuals_cube, angle_list)
        if verbose:
            print 'Done de-rotating and combining'
            timing(start_time)
        
    if full_output and cube.ndim<4:
        return pcs, recon, residuals_cube, residuals_cube_, frame
    elif full_output and cube.ndim==4:
        return residuals_cube_channels, residuals_cube_channels_, frame
    else:
        return frame
    
    
    
def pca_optimize_snr(cube, angle_list, y, x, fwhm, mode='full', 
                     annulus_width=2, svd_mode='randsvd', mask_center_px=5,
                     fmerit='px', min_snr=0, verbose=True, full_output=False, 
                     debug=False):
    """ Optimizes the number of principal components by doing a simple grid 
    search measuring the SNR for a given position in the frame (for ADI). 
    The metric used could be the given pixel's SNR, the maximun SNR in a FWHM 
    circular aperture centred on the given coordinates or the mean SNR in the 
    same circular aperture. They yield slightly different results.
    
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
        return opt_npc, finalfr
    else:
        return opt_npc


