#! /usr/bin/env python

"""
Module with PSF reference approximation using Non-negative matrix factorization 
for ADI and RDI data, in full frames.
"""

__author__ = 'Carlos Alberto Gomez Gonzalez, Valentin Christiaens'
__all__ = ['nmf']

import numpy as np
from sklearn.decomposition import NMF
from ..preproc import cube_derotate, cube_collapse
from ..preproc.derotation import _compute_pa_thresh, _find_indices_adi
from ..var import (prepare_matrix, reshape_matrix, frame_center, dist, 
                   matrix_scaling, mask_circle)
from ..config import timing, time_ini


def nmf(cube, angle_list, cube_ref=None, ncomp=1, scaling=None, max_iter=10000,
        random_state=None, mask_center_px=None, source_xy=None, delta_rot=1, 
        fwhm=4, init_svd='nndsvd', collapse='median', full_output=False, 
        verbose=True, cube_sig=None, handle_neg='mask', nmf_args={},
        **rot_options):
    """ Non Negative Matrix Factorization for ADI sequences. Alternative to the
    full-frame ADI-PCA processing that does not rely on SVD or ED for obtaining
    a low-rank approximation of the datacube. This function embeds the 
    scikit-learn NMF algorithm solved through either the coordinate descent or 
    the multiplicative update method.
    
    Parameters
    ----------
    cube : numpy ndarray, 3d
        Input cube.
    angle_list : numpy ndarray, 1d
        Corresponding parallactic angle for each frame.   
    cube_ref : numpy ndarray, 3d, optional
        Reference library cube. For Reference Star Differential Imaging.
    ncomp : int, optional
        How many components are used as for low-rank approximation of the 
        datacube.
    scaling : {None, 'temp-mean', 'spat-mean', 'temp-standard', 'spat-standard'}
        With None, no scaling is performed on the input data before SVD. With 
        "temp-mean" then temporal px-wise mean subtraction is done, with 
        "spat-mean" then the spatial mean is subtracted, with "temp-standard" 
        temporal mean centering plus scaling to unit variance is done and with
        "spat-standard" spatial mean centering plus scaling to unit variance is
        performed.  
    max_iter : int optional
        The number of iterations for the coordinate descent solver.
    random_state : int or None, optional
        Controls the seed for the Pseudo Random Number generator.
    mask_center_px : None or int
        If None, no masking is done. If an integer > 1 then this value is the
        radius of the circular mask.
    source_xy : tuple of int, optional
        For ADI-PCA, this triggers a frame rejection in the PCA library, with
        ``source_xy`` as the coordinates X,Y of the center of the annulus where
        the PA criterion is estimated. When ``ncomp`` is a tuple, a PCA grid is
        computed and the S/Ns (mean value in a 1xFWHM circular aperture) of the
        given (X,Y) coordinates are computed.
    delta_rot : int, optional
        Factor for tunning the parallactic angle threshold, expressed in FWHM.
        Default is 1 (excludes 1xFHWM on each side of the considered frame).
    fwhm : float, optional
        Known size of the FHWM in pixels to be used. Default value is 4. 
    init_svd: str, optional {'nnsvd','nnsvda','random'}
        Method used to initialize the iterative procedure to find H and W.
        'nndsvd': non-negative double SVD recommended for sparseness
        'nndsvda': NNDSVD where zeros are filled with the average of cube; 
        recommended when sparsity is not desired
        'random': random initial non-negative matrix
    collapse : {'median', 'mean', 'sum', 'trimmean'}, str optional
        Sets the way of collapsing the frames for producing a final image.
    full_output: boolean, optional
        Whether to return the final median combined image only or with other 
        intermediate arrays.  
    verbose : {True, False}, bool optional
        If True prints intermediate info and timing. 
    handle_neg: str, opt {'subtr_min','mask','null'}
        Determines how to handle negative values: mask them, set them to zero,
        or subtract the minimum value in the arrays. Note: 'mask' or 'null' 
        may leave significant artefacts after derotation of residual cube
        => those options should be used carefully (e.g. with proper treatment
        of masked values in non-derotated cube of residuals).
    nmf_args : dictionary, optional
        Additional arguments for scikit-learn NMF algorithm. See:
        https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html   
    rot_options: dictionary, optional
        Dictionary with optional keyword values for "nproc", "imlib", 
        "interpolation, "border_mode", "mask_val",  "edge_blend", 
        "interp_zeros", "ker" (see documentation of 
        ``vip_hci.preproc.frame_rotate``)   
             
    Returns
    -------
    If full_output is False the final frame is returned. If True the algorithm
    returns the reshaped NMF components, the reconstructed cube, the residuals,
    the residuals derotated and the final frame.     
    
    """
    array = cube.copy()
    if verbose:  
        start_time = time_ini()
    n, y, x = array.shape
    
    matrix_ref=None
    matrix_sig=None
    
    # how to handle negative values
    if handle_neg == 'mask':
        if mask_center_px:
            array = mask_circle(array, mask_center_px)
        if cube_sig is not None:
            yy, xx = np.where(np.amin(array-np.abs(cube_sig),axis=0)>0)
        else:
            yy, xx = np.where(np.amin(array,axis=0)>0)
        H_tmp = np.zeros([ncomp,y,x])
        if len(yy)>0:
            matrix = array[:,yy,xx]
            matrix = matrix_scaling(matrix, scaling)
            if cube_ref is not None:
                matrix_ref = cube_ref[:,yy,xx]
                matrix_ref = matrix_scaling(matrix_ref, scaling)
            if cube_sig is not None:
                matrix_sig = cube_sig[:,yy,xx]              
        else:
            raise ValueError("Remove frame(s) with negative values")
    else:
        if handle_neg=='null':
            if cube_sig is not None:
                array[np.where(array-cube_sig<0)]=0
                cube_sig[np.where(array-cube_sig<0)]=0
            else:
                array[np.where(array<0)]=0
            
        elif handle_neg=='subtr_min':
            if cube_sig is not None:
                array -= np.amin(array-cube_sig)
            else:
                array -= np.amin(array)
        else:
            raise ValueError("Mode to handle neg. pixels not recognized")
    
        matrix = prepare_matrix(array, scaling, mask_center_px, mode='fullfr',
                                verbose=verbose)
        if cube_ref is not None:
            matrix_ref = prepare_matrix(cube_ref, scaling, mask_center_px,
                                        mode='fullfr', verbose=verbose)
        if cube_sig is not None:
            matrix_sig = prepare_matrix(cube_sig, scaling, mask_center_px, 
                                        mode='fullfr', verbose=verbose)
    

    if cube_sig is not None:
        # derotate
        residuals_cube = cube_sig.copy()
    else:
        residuals_cube = np.zeros_like(array)
    
    
    if source_xy is None:
        residuals = _project_subtract(matrix, matrix_ref, ncomp, scaling,
                                      mask_center_px, verbose, full_output, 
                                      matrix_sig=matrix_sig, max_iter=max_iter, 
                                      random_state=random_state, 
                                      init_svd=init_svd, **nmf_args)
        if verbose:
            timing(start_time)
        if full_output:
            #reconstructed = residuals[1]
            H = residuals[2]
            reconstructed = residuals[1]
            residuals = residuals[0]
        recon_cube = residuals_cube.copy()
        if handle_neg=='mask':
            for fr in range(n):
                residuals_cube[fr][yy, xx] = residuals[fr]
            if full_output:
                for fr in range(n):
                    recon_cube[fr][yy, xx] = reconstructed[fr]
                for pp in range(ncomp):
                    H_tmp[pp][yy,xx] = H[pp]
                H = H_tmp
        else:
            for fr in range(n):
                residuals_cube[fr] = residuals[fr].reshape((y, x))
            if full_output:
                recon_cube = reshape_matrix(reconstructed, y,x)
                H = H.reshape(ncomp,y,x)
    else:
        if delta_rot is None or fwhm is None:
            msg = 'Delta_rot or fwhm parameters missing. Needed for the'
            msg += 'PA-based rejection of frames from the library'
            raise TypeError(msg)
        recon_cube = np.zeros_like(cube)
        yc, xc = frame_center(cube[0], False)
        x1, y1 = source_xy
        ann_center = dist(yc, xc, y1, x1)
        pa_thr = _compute_pa_thresh(ann_center, fwhm, delta_rot)
        mid_range = np.abs(np.amax(angle_list) - np.amin(angle_list))/2
        if pa_thr >= mid_range - mid_range * 0.1:
            new_pa_th = float(mid_range - mid_range * 0.1)
            if verbose:
                msg = 'PA threshold {:.2f} is too big, will be set to '
                msg += '{:.2f}'
                print(msg.format(pa_thr, new_pa_th))
            pa_thr = new_pa_th

        for fr in range(n):
            ind = _find_indices_adi(angle_list, fr, pa_thr)
            res_result = _project_subtract(matrix, matrix_ref, ncomp, scaling,
                                           mask_center_px, verbose, 
                                           full_output, ind, fr, 
                                           matrix_sig=matrix_sig, 
                                           max_iter=max_iter, 
                                           random_state=random_state, 
                                           init_svd=init_svd, **nmf_args)
            # ! Instead of reshaping, fill frame using get_annulus?
            if full_output:
                residuals = res_result[0]
                recon_frame = res_result[1]
                H = res_result[2]
                if handle_neg=='mask':
                    recon_cube[fr][yy,xx] = recon_frame
                else:
                    recon_cube[fr] = recon_frame.reshape((y, x))
            else:
                residuals = res_result
            if handle_neg=='mask':
                residuals_cube[fr][yy, xx] = residuals
                if fr == n-1:
                    for pp in range(ncomp):
                        H_tmp[pp][yy, xx] = H[pp]
                    H = H_tmp
            else:
                residuals_cube[fr] = residuals.reshape((y, x))
                if fr == n-1:
                    H = H.reshape(ncomp,y,x)
                
    if verbose:  
        print('Done NMF with sklearn.NMF.')
        timing(start_time)
            
    residuals_cube_ = cube_derotate(residuals_cube, angle_list, **rot_options)
    frame = cube_collapse(residuals_cube_, mode=collapse)
    
    if verbose:  
        print('Done derotating and combining.')
        timing(start_time)
    if full_output:
        return (H, recon_cube, residuals_cube, residuals_cube_, frame)
    else:
        return frame
    
    
def _project_subtract(matrix, matrix_ref, ncomp, scaling, mask_center_px, 
                      verbose, full_output, indices=None, frame=None, 
                      matrix_sig=None, max_iter=100, random_state=None, 
                      init_svd='nndsvd', **kwargs):
    """
    PCA projection and model PSF subtraction. Used as a helping function by
    each of the PCA modes (ADI, ADI+RDI, ADI+mSDI).

    Parameters
    ----------
    cube : numpy ndarray
        Input cube.
    cube_ref : numpy ndarray
        Reference cube.
    ncomp : int
        Number of principal components.
    scaling : str
        Scaling of pixel values. See ``pca`` docstrings.
    mask_center_px : int
        Masking out a centered circular aperture.
    verbose : bool
        Verbosity.
    full_output : bool
        Whether to return intermediate arrays or not.
    indices : list
        Indices to be used to discard frames (a rotation threshold is used).
    frame : int
        Index of the current frame (when indices is a list and a rotation
        threshold was applied).
    cube_sig: numpy ndarray, opt
        Cube with estimate of significant authentic signals. If provided, this
        will be subtracted from both the cube and the PCA library, before 
        projecting the cube onto the principal components.
        
    Returns
    -------
    ref_lib_shape : int
        [indices is not None, frame is not None] Number of
        rows in the reference library for the given frame.
    residuals: numpy ndarray
        Residuals, returned in every case.
    reconstructed : numpy ndarray
        [full_output=True] The reconstructed array.
    V : numpy ndarray
        [full_output=True, indices is None, frame is None] The right singular
        vectors of the input matrix, as returned by ``svd/svd_wrapper()``
    """    
    if matrix_sig is None:
        matrix_emp = matrix.copy()
    else:
        matrix_emp = matrix-matrix_sig
        
    if matrix_ref is not None:
        ref_lib = matrix_ref
    elif indices is not None and frame is not None:
        ref_lib = matrix_emp[indices].copy()
    else:
        ref_lib = matrix_emp.copy()

    # to avoid bug, just consider positive values
    if np.median(ref_lib)<0:
        raise ValueError("Mostly negative values in the cube")
    else:
        ref_lib[np.where(ref_lib<0)] = 0

    solver = 'mu'
    # if init_svd != 'nndsvd':
    #     solver = 'mu'
    # else:
    #     solver = 'cd'
    mod = NMF(n_components=ncomp, solver=solver, init=init_svd, 
              max_iter=max_iter, random_state=random_state, **kwargs)   
    # a rotation threshold is used (frames are processed one by one)
    if indices is not None and frame is not None:
        if ref_lib.shape[0] <= 10:
            raise RuntimeError('Less than 10 frames left in the PCA library'
                               ', Try decreasing the parameter delta_rot')
        curr_frame = matrix[frame]  # current frame
        curr_frame_emp = matrix_emp[frame]
        # H [ncomp, n_pixels]
        H = mod.fit(ref_lib).components_
        # W: coefficients [1, ncomp]
        W = mod.transform(curr_frame_emp[np.newaxis,...])
        #V = svd_wrapper(ref_lib, svd_mode, ncomp, False)
        #transformed = np.dot(curr_frame_emp, V.T)
        #reconstructed = np.dot(transformed.T, V)
        reconstructed = np.dot(W, H)
        residuals = curr_frame - reconstructed
        if full_output:
            return residuals, reconstructed, H
        else:
            return residuals

    # the whole matrix is processed at once
    else:      
        # H [ncomp, n_pixels]: Non-negative components of the data
        #if cube_ref is not None:
        H = mod.fit(ref_lib).components_
        #else:
        #    H = mod.fit(matrix).components_          
        
        # W: coefficients [n_frames, ncomp]
        W = mod.transform(matrix_emp)
            
        reconstructed = np.dot(W, H)
        residuals = matrix - reconstructed
        
        # V = svd_wrapper(ref_lib, svd_mode, ncomp, verbose)
        # transformed = np.dot(V, matrix_emp.T)
        # reconstructed = np.dot(transformed.T, V)
        # residuals = matrix - reconstructed
        # residuals_res = reshape_matrix(residuals, y, x)
        if full_output:
            return residuals, reconstructed, H
        else:
            return residuals