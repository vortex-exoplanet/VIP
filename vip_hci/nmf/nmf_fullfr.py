#! /usr/bin/env python

"""
Module with NMF algorithm for ADI.
"""

__author__ = 'Carlos Alberto Gomez Gonzalez, Valentin Christiaens'
__all__ = ['nmf']

import numpy as np
from sklearn.decomposition import NMF
from ..preproc import cube_derotate, cube_collapse
from ..preproc.derotation import _compute_pa_thresh, _find_indices_adi
from ..var import prepare_matrix, reshape_matrix, frame_center, dist
from ..conf import timing, time_ini


def nmf(cube, angle_list, cube_ref=None, ncomp=1, scaling=None, max_iter=100,
        random_state=None, mask_center_px=None, source_xy=None, delta_rot=1, 
        fwhm=4, imlib='opencv', interpolation='lanczos4', collapse='median', 
        full_output=False, verbose=True, cube_sig=None, **kwargs):
    """ Non Negative Matrix Factorization for ADI sequences. Alternative to the
    full-frame ADI-PCA processing that does not rely on SVD or ED for obtaining
    a low-rank approximation of the datacube. This function embeds the 
    scikit-learn NMF algorithm solved through coordinate descent method. 
    
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
    imlib : str, optional
        See the documentation of the ``vip_hci.preproc.frame_rotate`` function.
    interpolation : str, optional
        See the documentation of the ``vip_hci.preproc.frame_rotate`` function.
    collapse : {'median', 'mean', 'sum', 'trimmean'}, str optional
        Sets the way of collapsing the frames for producing a final image.
    full_output: boolean, optional
        Whether to return the final median combined image only or with other 
        intermediate arrays.  
    verbose : {True, False}, bool optional
        If True prints intermediate info and timing. 
    kwargs 
        Additional arguments for scikit-learn NMF algorithm.             
             
    Returns
    -------
    If full_output is False the final frame is returned. If True the algorithm
    returns the reshaped NMF components, the reconstructed cube, the residuals,
    the residuals derotated and the final frame.     
    
    """
    array = cube
    if verbose:  
        start_time = time_ini()
    n, y, x = array.shape
    
    matrix = prepare_matrix(cube, scaling, mask_center_px, mode='fullfr',
                            verbose=verbose)
    matrix += np.abs(matrix.min())
    if cube_ref is not None:
        matrix_ref = prepare_matrix(cube_ref, scaling, mask_center_px,
                                    mode='fullfr', verbose=verbose)
        matrix_ref += np.abs(matrix_ref.min())         
    elif source_xy is None or delta_rot is None:
        matrix_ref = matrix.copy()
    else:
        # rotation threshold applied and NMF repeated for each frame on different matrices
        if delta_rot is None or fwhm is None:
            msg = 'Delta_rot or fwhm parameters missing. Needed for the'
            msg += 'PA-based rejection of frames from the library'
            raise TypeError(msg)
        residuals_cube = np.zeros_like(cube)
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

        for frame in range(n):
            if ann_center > fwhm * 3:  # TODO: 3 optimal value? new par?
                ind = _find_indices_adi(angle_list, frame, pa_thr,
                                        truncate=True)
            else:
                ind = _find_indices_adi(angle_list, frame, pa_thr)
                
    
    if source_xy is None:
        residuals_result = _project_subtract(cube, None, ncomp, scaling,
                                             mask_center_px, verbose, 
                                             full_output, cube_sig=cube_sig, 
                                             max_iter=max_iter, 
                                             random_state=random_state, 
                                             **kwargs)
        if verbose:
            timing(start_time)
        if full_output:
            residuals_cube = residuals_result[0]
            reconstructed = residuals_result[1]
            H = residuals_result[2]
            recon_cube = reshape_matrix(reconstructed, y, x)
        else:
            residuals_cube = residuals_result
    else:
        if delta_rot is None or fwhm is None:
            msg = 'Delta_rot or fwhm parameters missing. Needed for the'
            msg += 'PA-based rejection of frames from the library'
            raise TypeError(msg)
        residuals_cube = np.zeros_like(cube)
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

        for frame in range(n):
            if ann_center > fwhm * 3:  # TODO: 3 optimal value? new par?
                ind = _find_indices_adi(angle_list, frame, pa_thr,
                                        truncate=True)
            else:
                ind = _find_indices_adi(angle_list, frame, pa_thr)

            res_result = _project_subtract(cube, None, ncomp, scaling,
                                           mask_center_px, verbose, 
                                           full_output, ind, frame, 
                                           cube_sig=cube_sig, max_iter=max_iter, 
                                           random_state=random_state, **kwargs)
            if full_output:
                residual_frame = res_result[0]
                recon_frame = res_result[1]
                H = residuals_result[2]
                residuals_cube[frame] = residual_frame.reshape((y, x))
                recon_cube[frame] = recon_frame.reshape((y, x))
            else:
                residual_frame = res_result
                residuals_cube[frame] = residual_frame.reshape((y, x))
     
    if verbose:  
        print('Done NMF with sklearn.NMF.')
        timing(start_time)
            
    residuals_cube_ = cube_derotate(residuals_cube, angle_list, imlib=imlib,
                              interpolation=interpolation)
    frame = cube_collapse(residuals_cube_, mode=collapse)
    
    if verbose:  
        print('Done derotating and combining.')
        timing(start_time)
    if full_output:
        return (H.reshape(ncomp,y,x), recon_cube, residuals_cube,
                residuals_cube_, frame)
    else:
        return frame
    
    
def _project_subtract(cube, cube_ref, ncomp, scaling, mask_center_px, verbose, 
                      full_output, indices=None, frame=None, cube_sig=None,
                      max_iter=100, random_state=None, **kwargs):
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
    n, y, x = cube.shape
    # if cube_sig is not None:
    #     cube_emp = cube-cube_sig
    # else:
    #     cube_emp = None
    if indices is not None and frame is not None:
        matrix = prepare_matrix(cube, scaling, mask_center_px,
                                mode='fullfr', verbose=False)
    else:
        matrix = prepare_matrix(cube, scaling, mask_center_px,
                                mode='fullfr', verbose=verbose)
    if cube_sig is None:
        matrix_emp = matrix.copy()
    else:
        nfr = cube_sig.shape[0]
        cube_sig = np.reshape(cube_sig, (nfr, -1))
        matrix_emp = matrix-cube_sig

    if cube_ref is not None:
        ref_lib = prepare_matrix(cube_ref, scaling, mask_center_px,
                                 mode='fullfr', verbose=verbose)
    else:
        ref_lib = matrix_emp

    mod = NMF(n_components=ncomp, alpha=0, solver='cd', init='nndsvd', 
              max_iter=max_iter, random_state=random_state, **kwargs)   
    # a rotation threshold is used (frames are processed one by one)
    if indices is not None and frame is not None:
        ref_lib = ref_lib[indices]
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
            return H, residuals, reconstructed
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
        W = mod.transform(cube)
            
        reconstructed = np.dot(W, H)
        residuals = matrix - reconstructed
        
        array_out = np.zeros_like(cube)
        for i in range(n):
            array_out[i] = residuals[i].reshape(y,x)
        
        # V = svd_wrapper(ref_lib, svd_mode, ncomp, verbose)
        # transformed = np.dot(V, matrix_emp.T)
        # reconstructed = np.dot(transformed.T, V)
        # residuals = matrix - reconstructed
        # residuals_res = reshape_matrix(residuals, y, x)
        if full_output:
            return array_out, reconstructed, H
        else:
            return array_out


