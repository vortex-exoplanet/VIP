#! /usr/bin/env python

"""
Full-frame PCA algorithm for ADI, ADI+RDI and ADI+mSDI (IFS data) cubes.
"""

from __future__ import division, print_function

__author__ = 'Carlos Alberto Gomez Gonzalez'
__all__ = ['pca']

import numpy as np
from multiprocessing import cpu_count
from .svd import svd_wrapper
from ..preproc.derotation import _find_indices_adi, _compute_pa_thresh
from ..preproc import cube_rescaling_wavelengths as scwave
from ..preproc import (cube_derotate, cube_collapse, check_pa_vector,
                       check_scal_vector, cube_crop_frames)
from ..conf import timing, time_ini, check_enough_memory, Progressbar
from ..conf.utils_conf import pool_map, fixed
from ..var import frame_center, dist, prepare_matrix, reshape_matrix
from ..stats import descriptive_stats


def pca(cube, angle_list, cube_ref=None, scale_list=None, ncomp=1, ncomp2=1,
        svd_mode='lapack', scaling=None, adimsdi='double', mask_center_px=None,
        source_xy=None, delta_rot=1, fwhm=4, imlib='opencv',
        interpolation='lanczos4', collapse='median', check_mem=True,
        crop_ifs=True, nproc=1, full_output=False, verbose=True, debug=False):
    """ Algorithm where the reference PSF and the quasi-static speckle pattern
    are modeled using Principal Component Analysis. Depending on the input
    parameters this PCA function can work in ADI, RDI or SDI (IFS data) mode.
    
    ADI: If neither a reference cube nor a scaling vector are provided, the
    target cube itself is used to learn the PCs and to obtain a low-rank
    approximation model PSF (star + speckles).
    
    ADI + RDI: if a reference cube is provided (``cube_ref``), its PCs are used
    to reconstruct the target frames to obtain the model PSF (star + speckles).
    
    ADI + SDI (IFS data): if a scaling vector is provided (``scale_list``) and
    the cube is a 4d array [# channels, # adi-frames, Y, X], its assumed it 
    contains several multi-spectral ADI frames. A single or two stages PCA can
    be performed, depending on ``adimsdi``.
    
    Parameters
    ----------
    cube : array_like, 3d or 4d
        Input cube (ADI or ADI+mSDI).
    angle_list : array_like, 1d
        Corresponding parallactic angle for each frame.    
    cube_ref : array_like, 3d, optional
        Reference library cube. For Reference Star Differential Imaging.
    scale_list : 
        Scaling factors in case of IFS data (ADI+mSDI cube). Usually, the
        scaling factors are the central channel wavelength divided by the
        shortest wavelength in the cube (more thorough approaches can be used
        to get the scaling factors). This scaling factors are used to re-scale
        the spectral channels and align the speckles.
    ncomp : int, optional
        How many PCs are used as a lower-dimensional subspace to project the
        target frames. For an ADI cube, ``ncomp`` is the number of PCs extracted
        from ``cube``. For the RDI case, when ``cube`` and ``cube_ref`` are
        provided, ``ncomp`` is the number of PCs obtained from ``cube_ref``.
        For an ADI+mSDI cube (e.g. SPHERE/IFS), if ``adimsdi`` is ``double``
        then ``ncomp`` is the number of PCs obtained from each multi-spectral
        frame (if ``ncomp`` is None then this stage will be skipped and the
        spectral channels will be combined without subtraction). If ``adimsdi``
        is ``single``, then ``ncomp`` is the number of PCs obtained from the
        whole set of frames (n_channels * n_adiframes).
    ncomp2 : int, optional
        Only used for ADI+mSDI cubes, when ``adimsdi`` is set to ``double``.
        ``ncomp2`` sets the number of PCs used in the second PCA stage (ADI
        fashion, using the residuals of the first stage). If None then the
        second PCA stage is skipped and the residuals are de-rotated and
        combined.
    mode : {'lapack', 'arpack', 'eigen', 'randsvd', 'cupy', 'eigencupy',
            'randcupy', 'pytorch', 'eigenpytorch', 'randpytorch'}, str optional
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
        ``randcupy`` is an adaptation of the randomized_svd algorithm, where all
        the computations are done on a GPU (through Cupy). ``pytorch`` uses the
        Pytorch library for GPU computation of the SVD. ``eigenpytorch`` offers
        the same method as with the ``eigen`` option but on GPU (through
        Pytorch). ``randpytorch`` is an adaptation of the randomized_svd
        algorithm, where all the linear algebra computations are done on a GPU
        (through Pytorch).
    scaling : {None, 'temp-mean', 'spat-mean', 'temp-standard', 'spat-standard'}
        With None, no scaling is performed on the input data before SVD. With 
        "temp-mean" then temporal px-wise mean subtraction is done, with 
        "spat-mean" then the spatial mean is subtracted, with "temp-standard" 
        temporal mean centering plus scaling to unit variance is done and with
        "spat-standard" spatial mean centering plus scaling to unit variance is
        performed.
    adimsdi : {'double', 'single'}, str optional
        In the case ``cube`` is a 4d array, ``adimsdi`` determines whether a
        single or double pass PCA is going to be computed. In the ``single``
        case, the multi-spectral frames are rescaled wrt the largest wavelength
        to align the speckles and all the frames are processed with a single
        PCA low-rank approximation. In the ``double`` case, a firt stage is run
        on the rescaled spectral frames, and a second PCA frame is run on the
        residuals in an ADI fashion.
    mask_center_px : None or int
        If None, no masking is done. If an integer > 1 then this value is the
        radius of the circular mask. 
    source_xy : tuple of int, optional 
        For ADI PCA, this triggers a frame rejection in the PCA library. 
        source_xy are the coordinates X,Y of the center of the annulus where the
        PA criterion will be used to reject frames from the library. 
    fwhm : float, optional
        Known size of the FHWM in pixels to be used. Default value is 4.
    imlib : str, optional
        See the documentation of the ``vip_hci.preproc.frame_rotate`` function.
    interpolation : str, optional
        See the documentation of the ``vip_hci.preproc.frame_rotate`` function.
    delta_rot : int, optional
        Factor for tunning the parallactic angle threshold, expressed in FWHM.
        Default is 1 (excludes 1xFHWM on each side of the considered frame).
    collapse : {'median', 'mean', 'sum', 'trimmean'}, str optional
        Sets the way of collapsing the frames for producing a final image.
    check_mem : bool, optional
        If True, it check that the input cube(s) are smaller than the available
        system memory.
    crop_ifs: bool, optional
        If True and the data are to be reduced with ADI+SDI(IFS) in a single step,
        this will crop the cube at the moment of frame rescaling in wavelength. 
        This is recommended for large FOVs such as the one of SPHERE, but can 
        remove significant amount of information close to the edge of small FOVs 
        (e.g. SINFONI).
    nproc : None or int, optional
        Number of processes for parallel computing. If None the number of
        processes will be set to (cpu_count()/2).
    full_output: bool, optional
        Whether to return the final median combined image only or with other 
        intermediate arrays.  
    verbose : bool, optional
        If True prints intermediate info and timing. 
    debug : bool, optional
        Whether to print debug information or not.
    
    Returns
    -------
    frame : array_like, 2d    
        Median combination of the de-rotated/re-scaled residuals cube.
    
    If full_output is True, and depending on the type of cube (ADI or ADI+mSDI),
    then several arrays will be returned, such as the residuals, de-rotated
    residuals, principal components

    References
    ----------
    The full-frame ADI-PCA implementation is based on Soummer et al. 2012
    (http://arxiv.org/abs/1207.4197) and Amara & Quanz 2012
    (http://arxiv.org/abs/1207.6637).
    """
    if not cube.ndim > 2:
        raise TypeError('Input array is not a 3d or 4d array')

    if check_mem:
        input_bytes = cube.nbytes
        if cube_ref is not None:
            input_bytes += cube_ref.nbytes
        if not check_enough_memory(input_bytes, 1.5, False):
            msgerr = 'Input cubes are larger than available system memory. '
            msgerr += 'Set check_mem=False to override this memory check or '
            msgerr += 'use the incremental PCA (for ADI)'
            raise RuntimeError(msgerr)

    start_time = time_ini(verbose)

    angle_list = check_pa_vector(angle_list)

    if nproc is None:
        nproc = cpu_count() // 2        # Hyper-threading doubles the # of cores

    # ADI + mSDI. Shape of cube: (n_channels, n_adi_frames, y, x)
    if cube.ndim == 4:
        if adimsdi == 'double':
            res_pca = _adimsdi_doublepca(cube, angle_list, scale_list, ncomp,
                                         ncomp2, scaling, mask_center_px, debug,
                                         svd_mode, imlib, interpolation,
                                         collapse, verbose, start_time,
                                         full_output, nproc)
            residuals_cube_channels, residuals_cube_channels_, frame = res_pca
        elif adimsdi == 'single':
            res_pca = _adimsdi_singlepca(cube, angle_list, scale_list, ncomp,
                                         scaling, mask_center_px, debug,
                                         svd_mode, imlib, interpolation,
                                         collapse, verbose, start_time,
                                         crop_ifs, full_output)
            cube_allfr_residuals, cube_adi_residuals, frame = res_pca
        else:
            raise ValueError('`Adimsdi` mode not recognized')

    # ADI + RDI
    elif cube.ndim == 3 and cube_ref is not None:
        res_pca = _adi_rdi_pca(cube, cube_ref, angle_list, ncomp, scaling,
                               mask_center_px, debug, svd_mode, imlib,
                               interpolation, collapse, verbose, full_output,
                               start_time)
        pcs, recon, residuals_cube, residuals_cube_, frame = res_pca

    # ADI. Shape of cube: (n_adi_frames, y, x)
    elif cube.ndim == 3 and cube_ref is None:
        res_pca = _adi_pca(cube, angle_list, ncomp, source_xy, delta_rot, fwhm,
                           scaling, mask_center_px, debug, svd_mode, imlib,
                           interpolation, collapse, verbose, start_time, True)

        if source_xy is not None:
            recon_cube, residuals_cube, residuals_cube_, frame = res_pca
        else:
            pcs, recon, residuals_cube, residuals_cube_, frame = res_pca

    else:
        msg = 'Only ADI, ADI+RDI and ADI+mSDI observing techniques are '
        msg += 'supported'
        raise RuntimeError(msg)

    if cube.ndim == 3:
        if full_output:
            if source_xy is not None:
                return recon_cube, residuals_cube, residuals_cube_, frame
            else:
                return pcs, recon, residuals_cube, residuals_cube_, frame
        else:
            return frame
    elif cube.ndim == 4:
        if full_output:
            if adimsdi == 'double':
                return residuals_cube_channels, residuals_cube_channels_, frame
            elif adimsdi == 'single':
                return cube_allfr_residuals, cube_adi_residuals, frame
        else:
            return frame


def _subtr_proj_fullfr(cube, cube_ref, ncomp, scaling, mask_center_px, debug,
                       svd_mode, verbose, full_output, indices=None,
                       frame=None):
    """ PCA projection and model PSF subtraction. Returns the cube of residuals.
    """
    _, y, x = cube.shape
    if indices is not None and frame is not None:
        matrix = prepare_matrix(cube, scaling, mask_center_px, mode='fullfr',
                                verbose=False)
    else:
        matrix = prepare_matrix(cube, scaling, mask_center_px, mode='fullfr',
                                verbose=verbose)

    if cube_ref is not None:
        ref_lib = prepare_matrix(cube_ref, scaling, mask_center_px,
                                 mode='fullfr', verbose=verbose)
    else:
        ref_lib = matrix

    if indices is not None and frame is not None:  # one row (frame) at a time
        ref_lib = ref_lib[indices]
        if ref_lib.shape[0] <= 10:
            msg = 'Too few frames left in the PCA library (<10). '
            msg += 'Try decreasing the parameter delta_rot'
            raise RuntimeError(msg)
        curr_frame = matrix[frame]  # current frame

        V = svd_wrapper(ref_lib, svd_mode, ncomp, False, False)
        transformed = np.dot(curr_frame, V.T)
        reconstructed = np.dot(transformed.T, V)
        residuals = curr_frame - reconstructed
        if full_output:
            return ref_lib.shape[0], residuals, reconstructed
        else:
            return ref_lib.shape[0], residuals
    else:  # the whole matrix
        V = svd_wrapper(ref_lib, svd_mode, ncomp, debug, verbose)
        transformed = np.dot(V, matrix.T)
        reconstructed = np.dot(transformed.T, V)
        residuals = matrix - reconstructed
        residuals_res = reshape_matrix(residuals, y, x)
        if full_output:
            return residuals_res, reconstructed, V
        else:
            return residuals_res


def _adi_pca(cube, angle_list, ncomp, source_xy, delta_rot, fwhm, scaling,
             mask_center_px, debug, svd_mode, imlib, interpolation, collapse,
             verbose, start_time, full_output):
    """ Handles the ADI PCA post-processing.
    """
    n, y, x = cube.shape

    if not n == angle_list.shape[0]:
        msg = "Angle list vector has wrong length. It must equal the "
        msg += "number of frames in the cube"
        raise ValueError(msg)

    if ncomp > n:
        ncomp = min(ncomp, n)
        msg = 'Number of PCs too high (max PCs={}), using {} PCs instead.'
        print(msg.format(n, ncomp))

    if source_xy is None:
        residuals_result = _subtr_proj_fullfr(cube, None, ncomp, scaling,
                                              mask_center_px, debug,
                                              svd_mode, verbose, full_output)
        if verbose:
            timing(start_time)
        if full_output:
            residuals_cube = residuals_result[0]
            reconstructed = residuals_result[1]
            V = residuals_result[2]
            pcs = reshape_matrix(V, y, x)
            recon = reshape_matrix(reconstructed, y, x)
        else:
            residuals_cube = residuals_result
    else:
        if delta_rot is None or fwhm is None:
            msg = 'Delta_rot or fwhm parameters missing. Needed for the'
            msg += 'PA-based rejection of frames from the library'
            raise TypeError(msg)
        nfrslib = []
        residuals_cube = np.zeros_like(cube)
        recon_cube = np.zeros_like(cube)
        yc, xc = frame_center(cube[0], False)
        x1, y1 = source_xy
        ann_center = dist(yc, xc, y1, x1)
        pa_thr = _compute_pa_thresh(ann_center, fwhm, delta_rot)
        mid_range = np.abs(np.amax(angle_list) - np.amin(angle_list)) / 2
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

            res_result = _subtr_proj_fullfr(cube, None, ncomp, scaling,
                                            mask_center_px, debug, svd_mode,
                                            verbose, full_output, ind,
                                            frame)
            if full_output:
                nfrslib.append(res_result[0])
                residual_frame = res_result[1]
                recon_frame = res_result[2]
                residuals_cube[frame] = residual_frame.reshape((y, x))
                recon_cube[frame] = recon_frame.reshape((y, x))
            else:
                nfrslib.append(res_result[0])
                residual_frame = res_result[1]
                residuals_cube[frame] = residual_frame.reshape((y, x))

        # number of frames in library printed for each annular quadrant
        if verbose:
            descriptive_stats(nfrslib, verbose=verbose, label='Size LIB: ')

    residuals_cube_ = cube_derotate(residuals_cube, angle_list, imlib=imlib,
                                    interpolation=interpolation)
    frame = cube_collapse(residuals_cube_, mode=collapse)
    if verbose:
        print('Done de-rotating and combining')
        timing(start_time)
    if source_xy is not None:
        return recon_cube, residuals_cube, residuals_cube_, frame
    else:
        return pcs, recon, residuals_cube, residuals_cube_, frame


def _adimsdi_singlepca(cube, angle_list, scale_list, ncomp, scaling,
                       mask_center_px, debug, svd_mode, imlib, interpolation,
                       collapse, verbose, start_time, crop_ifs, full_output):
    """ Handles the full-frame ADI+mSDI single PCA post-processing.
    """
    z, n, y_in, x_in = cube.shape

    if ncomp is None:
        raise ValueError('`Ncomp` must be provided (positive integer value)')

    if not angle_list.shape[0] == n:
        msg = "Angle list vector has wrong length. It must equal the number"
        msg += " frames in the cube"
        raise ValueError(msg)

    if scale_list is None:
        raise ValueError('Scaling factors vector must be provided')
    else:
        if np.array(scale_list).ndim > 1:
            raise ValueError('Scaling factors vector is not 1d')
        if not scale_list.shape[0] == cube.shape[0]:
            raise ValueError('Scaling factors vector has wrong length')

    scale_list = check_scal_vector(scale_list)
    big_cube = []

    if verbose:
        print('Rescaling the spectral channels to align the speckles')
    for i in Progressbar(range(n), verbose=verbose):
        cube_resc = scwave(cube[:, i, :, :], scale_list)[0]
        if crop_ifs:
            cube_resc = cube_crop_frames(cube_resc, size=y_in, verbose=False)
        big_cube.append(cube_resc)

    big_cube = np.array(big_cube)
    if not crop_ifs:
        _, y_in, x_in = cube_resc.shape
    big_cube = big_cube.reshape(z * n, y_in, x_in)

    if verbose:
        timing(start_time)
        print('{} total frames'.format(n * z))
        print('Performing single PCA')
    res_cube = _subtr_proj_fullfr(big_cube, None, ncomp, scaling,
                                  mask_center_px, debug, svd_mode, False, False)

    if verbose:
        timing(start_time)

    resadi_cube = np.zeros((n, y_in, x_in))
    if verbose:
        print('Descaling the spectral channels')
    for i in Progressbar(range(n), verbose=verbose):
        frame_i = scwave(res_cube[i * z:(i+1) * z, :, :], scale_list,
                         full_output=full_output, inverse=True, y_in=y_in,
                         x_in=x_in, collapse=collapse)
        resadi_cube[i] = frame_i

    if verbose:
        print('De-rotating and combining residuals')
        timing(start_time)
    der_res = cube_derotate(resadi_cube, angle_list, imlib=imlib,
                            interpolation=interpolation)
    frame = cube_collapse(der_res, mode=collapse)

    cube_allfr_residuals = res_cube
    cube_adi_residuals = resadi_cube
    return cube_allfr_residuals, cube_adi_residuals, frame


def _adimsdi_doublepca(cube, angle_list, scale_list, ncomp, ncomp2, scaling,
                       mask_center_px, debug, svd_mode, imlib, interpolation,
                       collapse, verbose, start_time, full_output, nproc):
    """  Handles the full-frame ADI+mSDI double PCA post-processing.
    """
    z, n, y_in, x_in = cube.shape

    global ARRAY
    ARRAY = cube

    if not angle_list.shape[0] == n:
        msg = "Angle list vector has wrong length. It must equal the number"
        msg += " frames in the cube"
        raise ValueError(msg)

    if scale_list is None:
        raise ValueError('Scaling factors vector must be provided')
    else:
        if np.array(scale_list).ndim > 1:
            raise ValueError('Scaling factors vector is not 1d')
        if not scale_list.shape[0] == cube.shape[0]:
            raise ValueError('Scaling factors vector has wrong length')

    scale_list = check_scal_vector(scale_list)

    if verbose:
        print('{} spectral channels in IFS cube'.format(z))
        if ncomp is None:
            print('Combining multi-spectral frames (skipping PCA)')
        else:
            print('First PCA stage exploiting spectral variability')

    if ncomp is not None and ncomp > z:
        ncomp = min(ncomp, z)
        msg = 'Number of PCs too high (max PCs={}), using {} PCs instead'
        print(msg.format(z, ncomp))

    res = pool_map(nproc, _adimsdi_doublepca_ifs, fixed(range(n)), ncomp,
                   scale_list, scaling, mask_center_px, debug, svd_mode,
                   full_output, collapse, verbose=verbose)
    residuals_cube_channels = np.array(res)

    if verbose:
        timing(start_time)

    # de-rotation of the PCA processed channels, ADI fashion
    if ncomp2 is None:
        if verbose:
            print('{} ADI frames'.format(n))
            print('De-rotating and combining frames (skipping PCA)')
        residuals_cube_channels_ = cube_derotate(residuals_cube_channels,
                                                 angle_list, imlib=imlib,
                                                 interpolation=interpolation)
        frame = cube_collapse(residuals_cube_channels_, mode=collapse)
        if verbose:
            timing(start_time)
    else:
        if ncomp2 > n:
            ncomp2 = min(ncomp2, n)
            msg = 'Number of PCs too high (max PCs={}), using {} PCs '
            msg += 'instead'
            print(msg.format(n, ncomp2))
        if verbose:
            print('{} ADI frames'.format(n))
            msg = 'Second PCA stage exploiting rotational variability'
            print(msg)
        res_ifs_adi = _subtr_proj_fullfr(residuals_cube_channels, None,
                                         ncomp2, scaling, mask_center_px,
                                         debug, svd_mode, False,
                                         full_output)
        if verbose:
            print('De-rotating and combining residuals')
        der_res = cube_derotate(res_ifs_adi, angle_list, imlib=imlib,
                                interpolation=interpolation)
        residuals_cube_channels_ = der_res
        frame = cube_collapse(residuals_cube_channels_, mode=collapse)
        if verbose:
            timing(start_time)
    return residuals_cube_channels, residuals_cube_channels_, frame


def _adimsdi_doublepca_ifs(fr, ncomp, scale_list, scaling, mask_center_px,
                           debug, svd_mode, full_output, collapse):
    """
    """
    z, n, y_in, x_in = ARRAY.shape
    multispec_fr = ARRAY[:, fr, :, :]

    if ncomp is None:
        frame_i = cube_collapse(multispec_fr, mode=collapse)
    else:
        cube_resc = scwave(multispec_fr, scale_list)[0]
        res = _subtr_proj_fullfr(cube_resc, None, ncomp, scaling,
                                 mask_center_px, debug, svd_mode, False,
                                 full_output)
        if full_output:
            res = res[0]
        frame_i = scwave(res, scale_list, full_output=full_output,
                         inverse=True, y_in=y_in, x_in=x_in, collapse=collapse)
    return frame_i


def _adi_rdi_pca(cube, cube_ref, angle_list, ncomp, scaling, mask_center_px,
                 debug, svd_mode, imlib, interpolation, collapse, verbose,
                 full_output, start_time):
    """ Handles the ADI+RDI post-processing.
    """
    n, y, x = cube.shape
    if not cube_ref.ndim == 3:
        msg = 'Input reference array is not a cube or 3d array'
        raise ValueError(msg)
    if not cube_ref.shape[1] == y:
        msg = 'Reference and target frames have different shape'
        raise TypeError(msg)

    if ncomp > n:
        ncomp = min(ncomp, n)
        msg = 'Number of PCs too high (max PCs={}), using {} PCs instead.'
        print(msg.format(n, ncomp))
    residuals_result = _subtr_proj_fullfr(cube, cube_ref, ncomp, scaling,
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
    residuals_cube_ = cube_derotate(residuals_cube, angle_list, imlib=imlib,
                                    interpolation=interpolation)
    frame = cube_collapse(residuals_cube_, mode=collapse)

    if verbose:
        print('Done de-rotating and combining')
        timing(start_time)
    return pcs, recon, residuals_cube, residuals_cube_, frame


