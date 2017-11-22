#! /usr/bin/env python

"""
LLSG (Gomez Gonzalez et al. 2016)
"""
from __future__ import division
from __future__ import print_function

__author__ = 'C. Gomez @ ULg'
__all__ = ['llsg']

import numpy as np
from scipy.linalg import qr
import itertools as itt
from multiprocessing import Pool, cpu_count
from astropy.stats import median_absolute_deviation as mad
from ..conf import time_ini, timing
from ..preproc import cube_derotate, cube_collapse
from ..var import get_annulus_quad, frame_filter_lowpass
from ..pca.svd import svd_wrapper
from ..pca.pca_local import define_annuli
from .thresholding import thresholding
from ..conf import eval_func_tuple as EFT 



def llsg(cube, angle_list, fwhm, rank=10, thresh=1, max_iter=10, 
         low_rank_mode='svd', thresh_mode='soft', nproc=1, radius_int=None, 
         random_seed=None, collapse='median', low_pass=False, full_output=False, 
         verbose=True, debug=False):
    """ 
    Local Low-rank plus Sparse plus Gaussian-noise decomposition (LLSG) as 
    described in Gomez Gonzalez et al. 2016. This first version of our algorithm 
    aims at decomposing ADI cubes into three terms L+S+G (low-rank, sparse and 
    Gaussian noise). Separating the noise from the S component (where the moving 
    planet should stay) allow us to increase the SNR of potential planets.
    
    The three tunable parameters are the *rank* or expected rank of the L
    component, the *thresh* or threshold for encouraging sparsity in the S 
    component and *max_iter* which sets the number of iterations. The rest of
    parameters can be tuned at the users own risk (do it if you know what you're
    doing).  
    
    Parameters
    ----------
    cube : array_like, 3d
        Input ADI cube.
    angle_list : array_like, 1d
        Corresponding parallactic angle for each frame.
    fwhm : float
        Known size of the FHWM in pixels to be used. 
    rank : int, optional
        Expected rank of the L component.
    thresh : float, optional
        Factor that scales the thresholding step in the algorithm.
    max_iter : int, optional
        Sets the number of iterations.
    low_rank_mode : {'svd', 'brp'}, optional 
        Sets the method of solving the L update.
    thresh_mode : {'soft', 'hard'}, optional
        Sets the type of thresholding.
    nproc : None or int, optional
        Number of processes for parallel computing. If None the number of 
        processes will be set to (cpu_count()/2). By default the algorithm works
        in single-process mode.
    radius_int : int, optional
        The radius of the innermost annulus. By default is 0, if >0 then the 
        central circular area is discarded.
    random_seed : int or None, optional
        Controls the seed for the Pseudo Random Number generator. 
    collapse : {'median', 'mean', 'sum', 'trimmean'}, str optional
        Sets the way of collapsing the frames for producing a final image.  
    low_pass : {False, True}, bool optional
        If True it performs a low_pass gaussian filter with kernel_size=fwhm on
        the final image.
    full_output: boolean, optional
        Whether to return the final median combined image only or with other 
        intermediate arrays.  
    verbose : {True, False}, bool optional
        If True prints to stdout intermediate info. 
    debug : {False, True}, bool optional
        Whether to output some intermediate information.
        
    Returns
    -------
    frame_s : array_like, 2d
        Final frame (from the S component) after rotation and median-combination.
    
    If *full_output* is True, the following intermediate arrays are returned:
    L_array_der, S_array_der, G_array_der, frame_l, frame_s, frame_g
        
    """
    asize=2
    
    if verbose:  start_time = time_ini()
    n,y,x = cube.shape
    
    if not radius_int:  radius_int = 0
    
    if nproc is None:   # Hyper-threading "duplicates" the cores -> cpu_count/2
        nproc = (cpu_count()/2)
    
    # ceil instead of taking integer part
    annulus_width = int(np.ceil(asize * fwhm))                                  # equal size for all annuli
    n_annuli = int(np.floor((y/2-radius_int)/annulus_width))    
    if verbose:
        msg = '{:} annuli, Ann width = {:}, FWHM = {:.3f}\n'
        print(msg.format(n_annuli, annulus_width, fwhm))
        
    matrix_final_s = np.zeros_like(cube) 
    if full_output: 
        matrix_final_l = np.zeros_like(cube)  
        matrix_final_g = np.zeros_like(cube) 
    # The annuli are built
    for ann in range(n_annuli):
        _, inner_radius, _ = define_annuli(angle_list, ann, n_annuli, fwhm, 
                                           radius_int, annulus_width, 0, False)
        indices = get_annulus_quad(cube[0], inner_radius, annulus_width)
        
        if nproc==1:
            for quadrant in range(4):
                yy = indices[quadrant][0]
                xx = indices[quadrant][1]
        
                data_all = cube[:, yy, xx]      # shape [nframes x npx_annquad]                       
                
                patch = patch_rlrps(data_all, rank, low_rank_mode, thresh, 
                                    thresh_mode, max_iter, random_seed, 
                                    debug=debug, full_output=full_output)
                if full_output:
                    matrix_final_l[:, yy, xx] = patch[0]
                    matrix_final_s[:, yy, xx] = patch[1]
                    matrix_final_g[:, yy, xx] = patch[2]
                else:
                    matrix_final_s[:, yy, xx] = patch
        # TODO: Avoid copying the whole cube for each process, less efficient.
        # TODO: Better parallelization scheme
        elif nproc>1:                                                           
            pool = Pool(processes=int(nproc))                                   
            res = pool.map(EFT, itt.izip(itt.repeat(_llsg_subf), 
                                         itt.repeat(cube),
                                         itt.repeat(indices),
                                         range(4), itt.repeat(rank),
                                         itt.repeat(low_rank_mode),
                                         itt.repeat(thresh),
                                         itt.repeat(thresh_mode),
                                         itt.repeat(max_iter),
                                         itt.repeat(random_seed),
                                         itt.repeat(debug),
                                         itt.repeat(full_output)))
            res = np.array(res)
            pool.close()
            patch = res[:,0] 
            yy = res[:,1]
            xx = res[:,2]
            quadrant = res[:,3]
            for q in range(4):
                if full_output:
                    matrix_final_l[:, yy[q], xx[q]] = patch[q][0]
                    matrix_final_s[:, yy[q], xx[q]] = patch[q][1]
                    matrix_final_g[:, yy[q], xx[q]] = patch[q][2]
                else:
                    matrix_final_s[:, yy[q], xx[q]] = patch[q]
        
    if full_output:       
        S_array_der = cube_derotate(matrix_final_s, angle_list)
        frame_s = cube_collapse(S_array_der, mode=collapse)
        L_array_der = cube_derotate(matrix_final_l, angle_list)
        frame_l = cube_collapse(L_array_der, mode=collapse)
        G_array_der = cube_derotate(matrix_final_g, angle_list)
        frame_g = cube_collapse(G_array_der, mode=collapse)
    else:
        S_array_der = cube_derotate(matrix_final_s, angle_list)              
        frame_s = cube_collapse(S_array_der, mode=collapse)
        
    if verbose:  timing(start_time) 
    
    if full_output:
        return L_array_der, S_array_der, G_array_der, frame_l, frame_s, frame_g
    else:
        if low_pass:
            return frame_filter_lowpass(frame_s, 'gauss', fwhm_size=fwhm)
        else:
            return frame_s
    


def patch_rlrps(array, rank, low_rank_mode, thresh, thresh_mode, max_iter, 
                random_seed, debug=False, full_output=False):
    """ Patch decomposition based on GoDec/SSGoDec (Zhou & Tao 2011) """           
    ### Initializing L and S
    L = array
    S = np.zeros_like(L)
    random_state = np.random.RandomState(random_seed)
    itr = 0    
    power = 0
    
    while itr<=max_iter:          
        ### Updating L
        if low_rank_mode=='brp':
            Y2 = random_state.randn(L.shape[1], rank)
            for _ in range(power+1):
                Y1 = np.dot(L, Y2)
                Y2 = np.dot(L.T, Y1)
            Q, _ = qr(Y2, mode='economic')    
            Lnew = np.dot(np.dot(L, Q), Q.T)    
        
        elif low_rank_mode=='svd':
            PC = svd_wrapper(L, 'randsvd', rank, False, False)
            Lnew = np.dot(np.dot(L, PC.T), PC)
        
        else:
            raise RuntimeError('Wrong Low Rank estimation mode')

        ### Updating S
        T = L - Lnew + S
        if itr==0:
            threshold = np.sqrt(mad(T.ravel()))*thresh
            if debug:  print('threshold level = {:.3f}'.format(threshold))
                
        S = thresholding(T, threshold, thresh_mode)   
        T -= S
        L = Lnew + T
        itr += 1
    
    if full_output:
        return L, S, array-L-S 
    else:
        return S


    
def _llsg_subf(cube, indices, quadrant, rank, low_rank_mode, thresh, thresh_mode, 
               max_iter, random_seed, debug, full_output):
    """ Sub-function for parallel processing of the quadrants. We build the 
    matrix to be decomposed for patch
    """
    yy = indices[quadrant][0]
    xx = indices[quadrant][1]

    data_all = cube[:, yy, xx]      # shape [nframes x npx_annquad] 
    patch = patch_rlrps(data_all, rank, low_rank_mode, thresh, thresh_mode, 
                       max_iter, random_seed, debug=debug, 
                       full_output=full_output)
    return patch, yy, xx, quadrant


