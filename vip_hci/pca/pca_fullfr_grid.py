#! /usr/bin/env python

"""
Full-frame PCA S/N optimization. It computes a grid search by truncating the
projection matrix.
"""

from __future__ import division, print_function

__author__ = 'Carlos Alberto Gomez Gonzalez'
__all__ = ['pca_optimize_snr']

import numpy as np
import pandas as pd
from skimage import draw
from matplotlib import pyplot as plt
from .svd import svd_wrapper
from .utils_pca import pca_annulus
from ..preproc import cube_derotate, cube_collapse
from ..conf import timing, time_ini
from ..conf.utils_conf import vip_figsize
from ..var import frame_center, dist, prepare_matrix, reshape_matrix
from .pca_fullfr import pca


def pca_optimize_snr(cube, angle_list, source_xy, fwhm, cube_ref=None,
                     mode='fullfr', annulus_width=20, range_pcs=None,
                     svd_mode='lapack', scaling=None, mask_center_px=None, 
                     fmerit='px', min_snr=0, imlib='opencv',
                     interpolation='lanczos4', collapse='median', verbose=True,
                     full_output=False, debug=False, plot=True, save_plot=None,
                     plot_title=None):
    """ Optimizes the number of principal components by doing a simple grid 
    search measuring the S/N for a given position in the frame (ADI, RDI).
    The metric used could be the given pixel's S/N, the maximum S/N in a FWHM
    circular aperture centered on the given coordinates or the mean S/N in the
    same circular aperture. It's useful for computing at once a cube of final
    PCA frames with a grid of values for the # of PCs (set ``full_output`` to
    True).
    
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
    mode : {'fullfr', 'annular'}, optional
        Mode for PCA processing (full-frame or just in an annulus). There is a
        catch: the optimal number of PCs in full-frame may not coincide with the
        one in annular mode. This is due to the fact that the annulus matrix is
        smaller (less noisy, probably not containing the central star) and also
        its intrinsic rank (smaller that in the full frame case).
    annulus_width : float, optional
        Width in pixels of the annulus in the case of the "annular" mode. 
    range_pcs : tuple, optional
        The interval of PCs to be tried. If None then the algorithm will find
        a clever way to sample from 1 to 200 PCs. If a range is entered (as 
        (PC_INI, PC_MAX)) a sequential grid will be evaluated between PC_INI
        and PC_MAX with step of 1. If a range is entered (as 
        (PC_INI, PC_MAX, STEP)) a grid will be evaluated between PC_INI and 
        PC_MAX with the given STEP.          
    svd_mode : {'lapack', 'arpack', 'eigen', 'randsvd', 'cupy', 'eigencupy', 'randcupy'}, str
        Switch for the SVD method/library to be used. ``lapack`` uses the LAPACK 
        linear algebra library through Numpy and it is the most conventional way 
        of computing the SVD (deterministic result computed on CPU). ``arpack`` 
        uses the ARPACK Fortran libraries accessible through Scipy (computation
        on CPU). ``eigen`` computes the singular vectors through the 
        eigendecomposition of the covariance M.M' (computation on CPU).
        ``randsvd`` uses the randomized_svd algorithm implemented in Sklearn 
        (computation on CPU). ``cupy`` uses the Cupy library for GPU computation
        of the SVD as in the LAPACK version. ``eigencupy`` offers the same 
        method as with the ``eigen`` option but on GPU (through Cupy). 
        ``randcupy`` is an adaptation of the randomized_svd algorith, where all 
        the computations are done on a GPU. 
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
    imlib : str, optional
        See the documentation of the ``vip_hci.preproc.frame_rotate`` function.
    interpolation : str, optional
        See the documentation of the ``vip_hci.preproc.frame_rotate`` function.
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
    save_plot: string
        If provided, the pc optimization plot will be saved to that path.
    plot_title: string
        If provided, the plot is titled

    Returns
    -------
    opt_npc : int
        Optimal number of PCs for given source.
    If full_output is True, the optimal number of PCs, the final processed
    frame, and a cube with all the PCA frames are returned.
    """
    from .. import metrics

    def truncate_svd_get_finframe(matrix, angle_list, ncomp, V):
        """ Projection, subtraction, derotation plus combination in one frame.
        Only for full-frame"""
        transformed = np.dot(V[:ncomp], matrix.T)
        reconstructed = np.dot(transformed.T, V[:ncomp])
        residuals = matrix - reconstructed
        frsize = int(np.sqrt(matrix.shape[1]))          # only for square frames
        residuals_res = reshape_matrix(residuals, frsize, frsize)
        residuals_res_der = cube_derotate(residuals_res, angle_list,
                                          imlib=imlib,
                                          interpolation=interpolation)
        frame = cube_collapse(residuals_res_der, mode=collapse)
        return frame

    def truncate_svd_get_finframe_ann(matrix, indices, angle_list, ncomp, V):
        """ Projection, subtraction, derotation plus combination in one frame.
        Only for annular mode"""
        transformed = np.dot(V[:ncomp], matrix.T)
        reconstructed = np.dot(transformed.T, V[:ncomp])
        residuals_ann = matrix - reconstructed
        residuals_res = np.zeros_like(cube)
        residuals_res[:,indices[0],indices[1]] = residuals_ann
        residuals_res_der = cube_derotate(residuals_res, angle_list,
                                          imlib=imlib,
                                          interpolation=interpolation)
        frame = cube_collapse(residuals_res_der, mode=collapse)
        return frame

    def get_snr(matrix, angle_list, y, x, mode, V, fwhm, ncomp, fmerit,
                full_output):
        if mode == 'fullfr':
            frame = truncate_svd_get_finframe(matrix, angle_list, ncomp, V)
        elif mode == 'annular':
            frame = truncate_svd_get_finframe_ann(matrix, annind, angle_list,
                                                  ncomp, V)
        else:
            raise RuntimeError('Wrong mode. Choose either full or annular')
        
        if fmerit == 'max':
            yy, xx = draw.circle(y, x, fwhm/2.)
            res = [metrics.snr_ss(frame, (x_, y_), fwhm, plot=False, verbose=False,
                                  full_output=True) for y_, x_ in zip(yy, xx)]
            snr_pixels = np.array(res)[:,-1]
            fluxes = np.array(res)[:,2]
            argm = np.argmax(snr_pixels)
            if full_output:
                # integrated fluxes for the max snr
                return np.max(snr_pixels), fluxes[argm], frame
            else:
                return np.max(snr_pixels), fluxes[argm]
        elif fmerit == 'px':
            res = metrics.snr_ss(frame, (x, y), fwhm, plot=False, verbose=False,
                                 full_output=True)
            snrpx = res[-1]
            fluxpx = np.array(res)[2]
            if full_output:
                # integrated fluxes for the given px
                return snrpx, fluxpx, frame
            else:
                return snrpx, fluxpx
        elif fmerit == 'mean':
            yy, xx = draw.circle(y, x, fwhm/2.)
            res = [metrics.snr_ss(frame, (x_, y_), fwhm, plot=False, verbose=False,
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
        snrlist = []
        pclist = []
        fluxlist = []
        if full_output:
            frlist = []
        counter = 0
        if debug:  
            print('Step current grid:', step)
            print('PCs | SNR')
        for pc in range(inti, intf+1, step):
            if full_output:
                snr, flux, frame = get_snr(matrix, angle_list, y, x, mode, V,
                                           fwhm, pc, fmerit, full_output)
            else:
                snr, flux = get_snr(matrix, angle_list, y, x, mode, V, fwhm, pc,
                                    fmerit, full_output)
            if np.isnan(snr):
                snr = 0
            if nsteps > 1 and snr < snrlist[-1]:
                counter += 1
            snrlist.append(snr)
            pclist.append(pc)
            fluxlist.append(flux)
            if full_output:
                frlist.append(frame)
            nsteps += 1
            if truncate and nsteps > 2 and snr < min_snr:
                if debug:
                    print('SNR too small')
                break
            if debug:
                print('{} {:.3f}'.format(pc, snr))
            if truncate and counter == 5:
                break
        argm = np.argmax(snrlist)
        
        if len(pclist) == 2:
            pclist.append(pclist[-1]+1)
    
        if debug:
            print('Finished current stage')
            try:
                pclist[argm+1]
                print('Interval for next grid: ', pclist[argm-1], 'to',
                      pclist[argm+1])
            except:
                print('The optimal SNR seems to be outside of the given '
                      'PC range')
            print()
        
        if argm == 0:
            argm = 1
        if full_output:
            return argm, pclist, snrlist, fluxlist, frlist
        else: 
            return argm, pclist, snrlist, fluxlist
    
    #---------------------------------------------------------------------------
    if cube.ndim != 3:
        raise TypeError('Input array is not a cube or 3d array')
    
    if verbose:
        start_time = time_ini()
    n = cube.shape[0]
    x, y = source_xy 
    
    if range_pcs is not None:
        if len(range_pcs) == 2:
            pcmin, pcmax = range_pcs
            pcmax = min(pcmax, n)
            step = 1
        elif len(range_pcs) == 3:
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
    
    # Getting `pcmax` principal components a single time
    if mode == 'fullfr':
        matrix = prepare_matrix(cube, scaling, mask_center_px, verbose=False)
        if cube_ref is not None:
            ref_lib = prepare_matrix(cube_ref, scaling, mask_center_px,
                                     verbose=False)
        else:
            ref_lib = matrix

    elif mode == 'annular':
        y_cent, x_cent = frame_center(cube[0])
        ann_radius = dist(y_cent, x_cent, y, x)
        matrix, annind = prepare_matrix(cube, scaling, None, mode='annular',
                                        annulus_radius=ann_radius,
                                        annulus_width=annulus_width,
                                        verbose=False)
        if cube_ref is not None:
            ref_lib, _ = prepare_matrix(cube_ref, scaling, mask_center_px,
                                     mode='annular', annulus_radius=ann_radius,
                                     annulus_width=annulus_width, verbose=False)
        else:
            ref_lib = matrix

    else:
        raise RuntimeError('Wrong mode. Choose either fullfr or annular')

    V = svd_wrapper(ref_lib, svd_mode, pcmax, False, verbose)

    # sequential grid
    if range_pcs is not None:
        grid1 = grid(matrix, angle_list, y, x, mode, V, fwhm, fmerit, step, 
                     pcmin, pcmax, debug, full_output, False)
        if full_output:
            argm, pclist, snrlist, fluxlist, frlist = grid1
        else:
            argm, pclist, snrlist, fluxlist = grid1
        
        opt_npc = pclist[argm]    
        if verbose:
            print('Number of steps', len(pclist))
            msg = 'Optimal number of PCs = {}, for SNR={:.3f}'
            print(msg.format(opt_npc, snrlist[argm]))
            print()
            timing(start_time)
        
        if full_output: 
            cubeout = np.array((frlist))

        # Plot of SNR as function of PCs  
        if plot:    
            plt.figure(figsize=vip_figsize)
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
            print()
            
    # automatic "clever" grid
    else:
        grid1 = grid(matrix, angle_list, y, x, mode, V, fwhm, fmerit, 
                     max(int(pcmax*0.1),1), pcmin, pcmax, debug, full_output)
        if full_output:
            argm, pclist, snrlist, fluxlist, frlist1 = grid1
        else:
            argm, pclist, snrlist, fluxlist = grid1
        
        grid2 = grid(matrix, angle_list, y, x, mode, V, fwhm, fmerit, 
                     max(int(pcmax*0.05),1), pclist[argm-1], pclist[argm+1],
                     debug, full_output)
        if full_output:
            argm2, pclist2, snrlist2, fluxlist2, frlist2 = grid2
        else:
            argm2, pclist2, snrlist2, fluxlist2 = grid2
        
        grid3 = grid(matrix, angle_list, y, x, mode, V, fwhm, fmerit, 1, 
                     pclist2[argm2-1], pclist2[argm2+1], debug, full_output, 
                     False)
        if full_output:
            _, pclist3, snrlist3, fluxlist3, frlist3 = grid3
        else:
            _, pclist3, snrlist3, fluxlist3 = grid3
        
        argm = np.argmax(snrlist3)
        opt_npc = pclist3[argm]    
        dfr = pd.DataFrame(np.array((pclist+pclist2+pclist3, 
                                     snrlist+snrlist2+snrlist3,
                                     fluxlist+fluxlist2+fluxlist3)).T)  
        dfrs = dfr.sort_values(0)
        dfrsrd = dfrs.drop_duplicates()
        ind = np.array(dfrsrd.index)    
        
        if verbose:
            print('Number of evaluated steps', ind.shape[0])
            msg = 'Optimal number of PCs = {}, for SNR={:.3f}'
            print(msg.format(opt_npc, snrlist3[argm]), '\n')
            timing(start_time)
        
        if full_output: 
            cubefrs = np.array((frlist1+frlist2+frlist3))
            cubeout = cubefrs[ind]
    
        # Plot of SNR as function of PCs  
        if plot:   
            alpha = 0.4
            lw = 2
            plt.figure(figsize=vip_figsize)
            ax1 = plt.subplot(211)  
            ax1.plot(np.array(dfrsrd.loc[:,0]), np.array(dfrsrd.loc[:,1]), '-', 
                     alpha=alpha, color='blue', lw=lw)
            ax1.plot(np.array(dfrsrd.loc[:,0]), np.array(dfrsrd.loc[:,1]), 'o',  
                     alpha=alpha/2, color='blue')
            ax1.set_xlim(np.array(dfrsrd.loc[:,0]).min(),
                         np.array(dfrsrd.loc[:,0]).max())
            ax1.set_ylim(0, np.array(dfrsrd.loc[:,1]).max()+1)
            ax1.set_ylabel('S/N')
            ax1.minorticks_on()
            ax1.grid('on', 'major', linestyle='solid', alpha=0.2)
            if plot_title is not None:
                ax1.set_title('Optimal pc: {} for {}'.format(opt_npc,
                                                             plot_title))
            
            ax2 = plt.subplot(212)
            ax2.plot(np.array(dfrsrd.loc[:,0]), np.array(dfrsrd.loc[:,2]), '-', 
                     alpha=alpha, color='green', lw=lw)
            ax2.plot(np.array(dfrsrd.loc[:,0]), np.array(dfrsrd.loc[:,2]), 'o', 
                     alpha=alpha/2, color='green')
            ax2.set_xlim(np.array(pclist).min(), np.array(pclist).max())
            #ax2.set_ylim(0, np.array(fluxlist).max()+1)
            ax2.set_xlabel('Principal components')
            ax2.set_ylabel('Flux in FWHM aperture')
            ax2.minorticks_on()
            ax2.set_yscale('log')
            ax2.grid('on', 'major', linestyle='solid', alpha=0.2)
            print()
    
    # Optionally, save the contrast curve
    if save_plot is not None:
        plt.savefig(save_plot, dpi=100, bbox_inches='tight')

    if mode == 'fullfr':
        finalfr = pca(cube, angle_list, cube_ref=cube_ref, ncomp=opt_npc,
                      svd_mode=svd_mode, mask_center_px=mask_center_px,
                      scaling=scaling, imlib=imlib, interpolation=interpolation,
                      collapse=collapse, verbose=False)
    elif mode == 'annular':
        finalfr = pca_annulus(cube, angle_list, ncomp=opt_npc,
                              annulus_width=annulus_width, r_guess=ann_radius,
                              cube_ref=cube_ref, svd_mode=svd_mode,
                              scaling=scaling, collapse=collapse, imlib=imlib,
                              interpolation=interpolation)

    _ = metrics.frame_quick_report(finalfr, fwhm, (x, y), verbose=verbose)
    
    if full_output:
        return opt_npc, finalfr, cubeout
    else:
        return opt_npc

    
