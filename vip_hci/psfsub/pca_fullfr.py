#! /usr/bin/env python

"""
Full-frame PCA algorithm for ADI, ADI+RDI and ADI+mSDI (IFS data) cubes:

- *Full-frame PCA*, using the whole cube as the PCA reference library in the
  case of ADI or ADI+mSDI (ISF cube), or a sequence of reference frames
  (reference star) in the case of RDI. For ADI a big data matrix NxP, where N
  is the number of frames and P the number of pixels in a frame is created. Then
  PCA is done through eigen-decomposition of the covariance matrix (~$DD^T$) or
  the SVD of the data matrix. SVD can be calculated using different libraries
  including the fast randomized SVD (Halko et al. 2009).

- *Full-frame incremental PCA* for big (larger than available memory) cubes.

"""

__author__ = 'Carlos Alberto Gomez Gonzalez'
__all__ = ['pca']

import numpy as np
from multiprocessing import cpu_count
from .svd import svd_wrapper, SVDecomposer
from .utils_pca import pca_incremental, pca_grid
from ..preproc.derotation import _find_indices_adi, _compute_pa_thresh
from ..preproc import cube_rescaling_wavelengths as scwave
from ..preproc import (cube_derotate, cube_collapse, check_pa_vector,
                       check_scal_vector, cube_crop_frames, 
                       cube_subtract_sky_pca)
from ..config import (timing, time_ini, check_enough_memory, Progressbar,
                    check_array)
from ..config.utils_conf import pool_map, iterable
from ..var import (frame_center, dist, prepare_matrix, reshape_matrix,
                   cube_filter_lowpass, mask_circle)
from ..stats import descriptive_stats


def pca(cube, angle_list, cube_ref=None, scale_list=None, ncomp=1,
        svd_mode='lapack', scaling=None, mask_center_px=None, source_xy=None,
        delta_rot=1, fwhm=4, adimsdi='single', crop_ifs=True, imlib='vip-fft',
        imlib2='vip-fft', interpolation='lanczos4', collapse='median', 
        ifs_collapse_range='all', mask_rdi=None, check_memory=True, batch=None, 
        nproc=1, full_output=False, verbose=True, weights=None, conv=False,
        cube_sig=None, **rot_options):
    """ Algorithm where the reference PSF and the quasi-static speckle pattern
    are modeled using Principal Component Analysis. Depending on the input
    parameters this PCA function can work in ADI, RDI or SDI (IFS data) mode.

    ADI: the target ``cube`` itself is used to learn the PCs and to obtain a
    low-rank approximation model PSF (star + speckles). Both `cube_ref`` and
    ``scale_list`` must be None. The full-frame ADI-PCA implementation is based
    on Soummer et al. 2012 (http://arxiv.org/abs/1207.4197) and Amara & Quanz
    2012 (http://arxiv.org/abs/1207.6637). If ``batch`` is provided then the
    cube if processed with incremental PCA as described in Gomez Gonzalez et al.
    2017 (https://arxiv.org/abs/1705.06184).

    ADI + RDI: if a reference cube is provided (``cube_ref``), its PCs are used
    to reconstruct the target frames to obtain the model PSF (star + speckles).

    ADI + SDI (IFS data): if a scaling vector is provided (``scale_list``) and
    the cube is a 4d array [# channels, # adi-frames, Y, X], its assumed it
    contains several multi-spectral ADI frames. A single or two stages PCA can
    be performed, depending on ``adimsdi``.

    Parameters
    ----------
    cube : str or numpy ndarray, 3d or 4d
        Input cube (ADI or ADI+mSDI). If a string is given, it must correspond
        to the path to the fits file to be opened in memmap mode (for PCA
        incremental of ADI 3d cubes).
    angle_list : numpy ndarray, 1d
        Corresponding parallactic angle for each frame.
    cube_ref : numpy ndarray, 3d, optional
        Reference library cube. For Reference Star Differential Imaging.
    scale_list : numpy ndarray, 1d, optional
        Scaling factors in case of IFS data (ADI+mSDI cube). Usually, the
        scaling factors are the central channel wavelength divided by the
        shortest wavelength in the cube (more thorough approaches can be used
        to get the scaling factors). This scaling factors are used to re-scale
        the spectral channels and align the speckles.
    ncomp : int, float or tuple of int/None, optional
        How many PCs are used as a lower-dimensional subspace to project the
        target frames.

        * ADI (``cube`` is a 3d array): if an int is provided, ``ncomp`` is the
          number of PCs extracted from ``cube`` itself. If ``ncomp`` is a float
          in the interval (0, 1] then it corresponds to the desired cumulative
          explained variance ratio (the corresponding number of components is
          estimated). If ``ncomp`` is a tuple of two integers, then it
          corresponds to an interval of PCs in which final residual frames are
          computed (optionally, if a tuple of 3 integers is passed, the third
          value is the step). When``source_xy`` is not None, then the S/Ns
          (mean value in a 1xFWHM circular aperture) of the given
          (X,Y) coordinates are computed.

        * ADI+RDI (``cube`` and ``cube_ref`` are 3d arrays): ``ncomp`` is the
          number of PCs obtained from ``cube_ref``. If ``ncomp`` is a tuple,
          then it corresponds to an interval of PCs (obtained from ``cube_ref``)
          in which final residual frames are computed. If ``source_xy`` is not
          None, then the S/Ns (mean value in a 1xFWHM circular aperture) of the
          given (X,Y) coordinates are computed.

        * ADI+mSDI (``cube`` is a 4d array and ``adimsdi="single"``): ``ncomp``
          is the number of PCs obtained from the whole set of frames
          (n_channels * n_adiframes). If ``ncomp`` is a float in the interval
          (0, 1] then it corresponds to the desired CEVR, and the corresponding
          number of components will be estimated. If ``ncomp`` is a tuple, then
          it corresponds to an interval of PCs in which final residual frames
          are computed. If ``source_xy`` is not None, then the S/Ns (mean value
          in a 1xFWHM circular aperture) of the given (X,Y) coordinates are
          computed.

        * ADI+mSDI  (``cube`` is a 4d array and ``adimsdi="double"``): ``ncomp``
          must be a tuple, where the first value is the number of PCs obtained
          from each multi-spectral frame (if None then this stage will be
          skipped and the spectral channels will be combined without
          subtraction); the second value sets the number of PCs used in the
          second PCA stage, ADI-like using the residuals of the first stage (if
          None then the second PCA stage is skipped and the residuals are
          de-rotated and combined).

    svd_mode : {'lapack', 'arpack', 'eigen', 'randsvd', 'cupy', 'eigencupy',
        'randcupy', 'pytorch', 'eigenpytorch', 'randpytorch'}, str optional
        Switch for the SVD method/library to be used.

        ``lapack``: uses the LAPACK linear algebra library through Numpy
        and it is the most conventional way of computing the SVD
        (deterministic result computed on CPU).

        ``arpack``: uses the ARPACK Fortran libraries accessible through
        Scipy (computation on CPU).

        ``eigen``: computes the singular vectors through the
        eigendecomposition of the covariance M.M' (computation on CPU).

        ``randsvd``: uses the randomized_svd algorithm implemented in
        Sklearn (computation on CPU).

        ``cupy``: uses the Cupy library for GPU computation of the SVD as in
        the LAPACK version. `

        `eigencupy``: offers the same method as with the ``eigen`` option
        but on GPU (through Cupy).

        ``randcupy``: is an adaptation of the randomized_svd algorithm,
        where all the computations are done on a GPU (through Cupy). `

        `pytorch``: uses the Pytorch library for GPU computation of the SVD.

        ``eigenpytorch``: offers the same method as with the ``eigen``
        option but on GPU (through Pytorch).

        ``randpytorch``: is an adaptation of the randomized_svd algorithm,
        where all the linear algebra computations are done on a GPU
        (through Pytorch).

    scaling : {None, "temp-mean", spat-mean", "temp-standard",
        "spat-standard"}, None or str optional
        Pixel-wise scaling mode using ``sklearn.preprocessing.scale``
        function. If set to None, the input matrix is left untouched. Otherwise:

        ``temp-mean``: temporal px-wise mean is subtracted.

        ``spat-mean``: spatial mean is subtracted.

        ``temp-standard``: temporal mean centering plus scaling pixel values
        to unit variance. HIGHLY RECOMMENDED FOR ASDI AND RDI CASES!

        ``spat-standard``: spatial mean centering plus scaling pixel values
        to unit variance.

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
        Factor for tuning the parallactic angle threshold, expressed in FWHM.
        Default is 1 (excludes 1xFHWM on each side of the considered frame).
    fwhm : float, optional
        Known size of the FHWM in pixels to be used. Default value is 4. 
    adimsdi : {'single', 'double'}, str optional
        Changes the way the 4d cubes (ADI+mSDI) are processed. Basically it
        determines whether a single or double pass PCA is going to be computed.

        ``single``: the multi-spectral frames are rescaled wrt the largest
        wavelength to align the speckles and all the frames (n_channels *
        n_adiframes) are processed with a single PCA low-rank approximation.

        ``double``: a first stage is run on the rescaled spectral frames, and a
        second PCA frame is run on the residuals in an ADI fashion.

    crop_ifs: bool, optional
        [adimsdi='single'] If True cube is cropped at the moment of frame
        rescaling in wavelength. This is recommended for large FOVs such as the
        one of SPHERE, but can remove significant amount of information close to
        the edge of small FOVs (e.g. SINFONI).
    imlib : str, optional
        See the documentation of the ``vip_hci.preproc.frame_rotate`` function.
    imlib2 : str, optional
        See the documentation of the ``vip_hci.preproc.cube_rescaling_wavelengths`` 
        function.
    interpolation : str, optional
        See the documentation of the ``vip_hci.preproc.frame_rotate`` function.
    collapse : {'median', 'mean', 'sum', 'trimmean'}, str optional
        Sets the way of collapsing the frames for producing a final image.
    ifs_collapse_range: str 'all' or tuple of 2 int
        If a tuple, it should contain the first and last channels where the mSDI 
        residual channels will be collapsed (by default collapses all channels).
    mask_rdi: 2d numpy array, opt
        If provided, this binary mask will be used either in RDI mode or in 
        ADI+mSDI (2 steps) mode. The projection coefficients for the principal 
        components will be found considering the area covered by the mask 
        (useful to avoid self-subtraction in presence of bright disc signal)
    check_memory : bool, optional
        If True, it checks that the input cube is smaller than the available
        system memory.
    batch : None, int or float, optional
        When it is not None, it triggers the incremental PCA (for ADI and
        ADI+mSDI cubes). If an int is given, it corresponds to the number of
        frames in each sequential mini-batch. If a float (0, 1] is given, it
        corresponds to the size of the batch is computed wrt the available
        memory in the system.
    nproc : None or int, optional
        Number of processes for parallel computing. If None the number of
        processes will be set to (cpu_count()/2). Defaults to ``nproc=1``.
    full_output: bool, optional
        Whether to return the final median combined image only or with other
        intermediate arrays.
    verbose : bool, optional
        If True prints intermediate info and timing.
    weights: 1d numpy array or list, optional
        Weights to be applied for a weighted mean. Need to be provided if 
        collapse mode is 'wmean'.
    cube_sig: numpy ndarray, opt
        Cube with estimate of significant authentic signals. If provided, this
        will subtracted before projecting cube onto reference cube.
    rot_options: dictionary, optional
        Dictionary with optional keyword values for "border_mode", "mask_val",  
        "edge_blend", "interp_zeros", "ker" (see documentation of 
        ``vip_hci.preproc.frame_rotate``)
        
    Returns
    -------
    frame : numpy ndarray
        2D array, median combination of the de-rotated/re-scaled residuals cube.
        Always returned.
    pcs : numpy ndarray
        [full_output=True, source_xy=None] Principal components. Valid for
        ADI cubes (3D). This is also returned when ``batch`` is not None
        (incremental PCA).
    recon_cube, recon : numpy ndarray
        [full_output=True] Reconstructed cube. Valid for ADI cubes (3D).
    residuals_cube : numpy ndarray
        [full_output=True] Residuals cube. Valid for ADI cubes (3D).
    residuals_cube_ : numpy ndarray
        [full_output=True] Derotated residuals cube. Valid for ADI cubes (3D).
    residuals_cube_channels : numpy ndarray
        [full_output=True, adimsdi='double'] Residuals for each multi-spectral
        cube. Valid for ADI+mSDI (4D) cubes.
    residuals_cube_channels_ : numpy ndarray
        [full_output=True, adimsdi='double'] Derotated final residuals. Valid
        for ADI+mSDI (4D) cubes.
    cube_allfr_residuals : numpy ndarray
        [full_output=True, adimsdi='single']  Residuals cube (of the big cube
        with channels and time processed together). Valid for ADI+mSDI (4D)
        cubes.
    cube_adi_residuals : numpy ndarray
        [full_output=True, adimsdi='single'] Residuals cube (of the big cube
        with channels and time processed together) after de-scaling the wls.
        Valid for ADI+mSDI (4D).
    medians : numpy ndarray
        [full_output=True, source_xy=None] This is also returned when ``batch``
        is not None (incremental PCA).
    final_residuals_cube : numpy ndarray
        [ncomp is tuple] The residual final PCA frames for a grid a PCs.


    """
    start_time = time_ini(verbose)

    if batch is None:
        check_array(cube, (3, 4), msg='cube')
    else:
        if not isinstance(cube, (str, np.ndarray)):
            raise TypeError('`cube` must be a numpy (3d or 4d) array or a str '
                            'with the full path on disk')

    # checking memory (if in-memory numpy array is provided)
    if not isinstance(cube, str):
        input_bytes = cube_ref.nbytes if cube_ref is not None else cube.nbytes
        mem_msg = 'Set check_memory=False to override this memory check or ' \
                  'set `batch` to run incremental PCA (valid for ADI or ' \
                  'ADI+mSDI single-pass)'
        check_enough_memory(input_bytes, 1.0, raise_error=check_memory,
                            error_msg=mem_msg, verbose=verbose)

    if nproc is None:
        nproc = cpu_count() // 2        # Hyper-threading doubles the # of cores

    # ADI + mSDI. Shape of cube: (n_channels, n_adi_frames, y, x)
    if scale_list is not None: #isinstance(cube, np.ndarray) and cube.ndim == 4:
        if adimsdi == 'double':
            res_pca = _adimsdi_doublepca(cube, angle_list, scale_list, ncomp,
                                         scaling, mask_center_px, svd_mode,
                                         imlib, imlib2, interpolation, collapse, 
                                         ifs_collapse_range, verbose, start_time, 
                                         nproc, weights, fwhm, conv, mask_rdi,
                                         cube_sig, **rot_options)
            residuals_cube_channels, residuals_cube_channels_, frame = res_pca
        elif adimsdi == 'single':
            res_pca = _adimsdi_singlepca(cube, angle_list, scale_list, ncomp,
                                         fwhm, source_xy, scaling,
                                         mask_center_px, svd_mode, imlib,
                                         imlib2, interpolation, collapse, 
                                         ifs_collapse_range, verbose, start_time, 
                                         nproc, crop_ifs, batch, full_output=True,
                                         weights=weights, **rot_options)
            if isinstance(ncomp, (int, float)):
                cube_allfr_residuals, cube_adi_residuals, frame = res_pca
            elif isinstance(ncomp, tuple):
                if source_xy is None:
                    final_residuals_cube, pclist = res_pca
                else:
                    final_residuals_cube, frame, table, _ = res_pca
        else:
            raise ValueError('`adimsdi` mode not recognized')

    # ADI + RDI
    elif cube_ref is not None:
        res_pca = _adi_rdi_pca(cube, cube_ref, angle_list, ncomp, scaling,
                               mask_center_px, svd_mode, imlib, interpolation,
                               collapse, verbose, start_time, nproc, weights, 
                               mask_rdi, cube_sig, **rot_options)
        pcs, recon, residuals_cube, residuals_cube_, frame = res_pca


    # ADI. Shape of cube: (n_adi_frames, y, x)
    elif cube_ref is None:
        res_pca = _adi_pca(cube, angle_list, ncomp, batch, source_xy, delta_rot,
                           fwhm, scaling, mask_center_px, svd_mode, imlib,
                           interpolation, collapse, verbose, start_time, nproc,
                           True, weights, cube_sig, **rot_options)

        if batch is None:
            if source_xy is not None:
                # PCA grid, computing S/Ns
                if isinstance(ncomp, tuple):
                    if full_output:
                        final_residuals_cube, frame, table, _ = res_pca
                    else:
                        # returning only the optimal residual
                        final_residuals_cube = res_pca[1]
                # full-frame PCA with rotation threshold
                else:
                    recon_cube, residuals_cube, residuals_cube_, frame = res_pca
            else:
                # PCA grid
                if isinstance(ncomp, tuple):
                    final_residuals_cube, pclist = res_pca
                # full-frame standard PCA
                else:
                    pcs, recon, residuals_cube, residuals_cube_, frame = res_pca
        # full-frame incremental PCA
        else:
            frame, _, pcs, medians = res_pca

    else:
        raise RuntimeError('Only ADI, ADI+RDI and ADI+mSDI observing techniques'
                           ' are supported')

    # --------------------------------------------------------------------------
    # Returns for each case (ADI, ADI+RDI and ADI+mSDI) and combination of
    # parameters: full_output, source_xy, batch, ncomp
    # --------------------------------------------------------------------------
    if isinstance(cube, np.ndarray) and cube.ndim == 4:
        # ADI+mSDI double-pass PCA
        if adimsdi == 'double':
            if full_output:
                return frame, residuals_cube_channels, residuals_cube_channels_
            else:
                return frame

        elif adimsdi == 'single':
            # ADI+mSDI single-pass PCA
            if isinstance(ncomp, (float, int)):
                if full_output:
                    return frame, cube_allfr_residuals, cube_adi_residuals
                else:
                    return frame
            # ADI+mSDI single-pass PCA grid
            elif isinstance(ncomp, tuple):
                if source_xy is None and full_output:
                    return final_residuals_cube, pclist
                elif source_xy is None and not full_output:
                    return final_residuals_cube
                elif source_xy is not None and full_output:
                    return final_residuals_cube, frame, table
                elif source_xy is not None and not full_output:
                    return frame

    # ADI and ADI+RDI
    else:
        if (cube_ref is not None or source_xy is None) and full_output:
            # incremental PCA
            if batch is not None:
                return frame, pcs, medians
            else:
                # PCA grid
                if isinstance(ncomp, tuple):
                    return final_residuals_cube, pclist
                # full-frame standard PCA or ADI+RDI
                else:
                    return frame, pcs, recon, residuals_cube, residuals_cube_
        elif source_xy is not None and full_output:
            # PCA grid, computing S/Ns
            if isinstance(ncomp, tuple):
                return final_residuals_cube, frame, table
            # full-frame PCA with rotation threshold
            else:
                return frame, recon_cube, residuals_cube, residuals_cube_
        elif not full_output:
            # PCA grid
            if isinstance(ncomp, tuple):
                return final_residuals_cube
            # full-frame standard PCA or ADI+RDI
            else:
                return frame
        

def _adi_pca(cube, angle_list, ncomp, batch, source_xy, delta_rot, fwhm,
             scaling, mask_center_px, svd_mode, imlib, interpolation, collapse,
             verbose, start_time, nproc, full_output, weights=None, 
             cube_sig=None, **rot_options):
    """ Handles the ADI PCA post-processing.
    """
    # Full/Single ADI processing, incremental PCA
    if batch is not None:
        result = pca_incremental(cube, angle_list, batch=batch, ncomp=ncomp,
                                 collapse=collapse, verbose=verbose,
                                 full_output=full_output, start_time=start_time,
                                 weights=weights, nproc=nproc, imlib=imlib, 
                                 interpolation=interpolation, **rot_options)
        return result

    else:
        # Full/Single ADI processing
        n, y, x = cube.shape

        angle_list = check_pa_vector(angle_list)
        if not n == angle_list.shape[0]:
            raise ValueError("`angle_list` vector has wrong length. It must "
                             "equal the number of frames in the cube")

        if not isinstance(ncomp, (int, float, tuple)):
            raise TypeError("`ncomp` must be an int, float or a tuple in the "
                            "ADI case")

        if isinstance(ncomp, (int, float)):
            if isinstance(ncomp, int) and ncomp > n:
                ncomp = min(ncomp, n)
                print('Number of PCs too high (max PCs={}), using {} PCs '
                      'instead.'.format(n, ncomp))

            if source_xy is None:
                residuals_result = _project_subtract(cube, None, ncomp, scaling,
                                                     mask_center_px, svd_mode,
                                                     verbose, full_output,
                                                     cube_sig=cube_sig)
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

            # A rotation threshold is applied
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
                                                   mask_center_px, svd_mode,
                                                   verbose, full_output, ind,
                                                   frame, cube_sig=cube_sig)
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
                    descriptive_stats(nfrslib, verbose=verbose,
                                      label='Size LIB: ')

            residuals_cube_ = cube_derotate(residuals_cube, angle_list,
                                            nproc=nproc, imlib=imlib,
                                            interpolation=interpolation,
                                            **rot_options)
            frame = cube_collapse(residuals_cube_, mode=collapse, w=weights)
            if verbose:
                print('Done de-rotating and combining')
                timing(start_time)
            if source_xy is not None:
                return recon_cube, residuals_cube, residuals_cube_, frame
            else:
                return pcs, recon, residuals_cube, residuals_cube_, frame

        # When ncomp is a tuple, pca_grid is called
        else:
            gridre = pca_grid(cube, angle_list, fwhm, range_pcs=ncomp,
                              source_xy=source_xy, cube_ref=None, mode='fullfr',
                              svd_mode=svd_mode, scaling=scaling,
                              mask_center_px=mask_center_px, fmerit='mean',
                              collapse=collapse, verbose=verbose,
                              full_output=full_output, debug=False,
                              plot=verbose, start_time=start_time, 
                              weights=weights, nproc=nproc, imlib=imlib, 
                              interpolation=interpolation, **rot_options)
            return gridre


def _adimsdi_singlepca(cube, angle_list, scale_list, ncomp, fwhm, source_xy,
                       scaling, mask_center_px, svd_mode, imlib, imlib2, 
                       interpolation, collapse, ifs_collapse_range, verbose, 
                       start_time, nproc, crop_ifs, batch, full_output, 
                       weights=None, **rot_options):
    """ Handles the full-frame ADI+mSDI single PCA post-processing.
    """
    z, n, y_in, x_in = cube.shape

    angle_list = check_pa_vector(angle_list)
    if not angle_list.shape[0] == n:
        msg = "Angle list vector has wrong length. It must equal the number"
        msg += " frames in the cube"
        raise ValueError(msg)

    if scale_list is None:
        raise ValueError('`scale_list` must be provided')
    else:
        check_array(scale_list, dim=1, msg='scale_list')
        if not scale_list.shape[0] == z:
            raise ValueError('`scale_list` has wrong length')

    scale_list = check_scal_vector(scale_list)
    big_cube = []

    if verbose:
        print('Rescaling the spectral channels to align the speckles')
    for i in Progressbar(range(n), verbose=verbose):
        cube_resc = scwave(cube[:, i, :, :], scale_list, imlib=imlib2,
                           interpolation=interpolation)[0]
        if crop_ifs:
            cube_resc = cube_crop_frames(cube_resc, size=y_in, verbose=False)
        big_cube.append(cube_resc)

    big_cube = np.array(big_cube)
    big_cube = big_cube.reshape(z * n, big_cube.shape[2], big_cube.shape[3])

    if verbose:
        timing(start_time)
        print('{} total frames'.format(n * z))
        print('Performing single-pass PCA')

    if isinstance(ncomp, (int, float)):
        # When ncomp is a int and batch is not None, incremental ADI-PCA is run
        if batch is not None:
            res_cube = pca_incremental(big_cube, angle_list, batch, ncomp, 
                                       collapse, verbose, return_residuals=True, 
                                       start_time=start_time, weights=weights,
                                       nproc=nproc, imlib=imlib, 
                                       interpolation=interpolation, 
                                       **rot_options)
        # When ncomp is a int/float and batch is None, standard ADI-PCA is run
        else:
            res_cube = _project_subtract(big_cube, None, ncomp, scaling,
                                         mask_center_px, svd_mode, verbose,
                                         False)

        if verbose:
            timing(start_time)

        resadi_cube = np.zeros((n, y_in, x_in))

        if verbose:
            print('Descaling the spectral channels')
        if ifs_collapse_range == 'all':
            idx_ini = 0
            idx_fin = z
        else:
            idx_ini = ifs_collapse_range[0]
            idx_fin = ifs_collapse_range[1]
            
        for i in Progressbar(range(n), verbose=verbose):
            frame_i = scwave(res_cube[i*z+idx_ini:i*z+idx_fin, :, :], 
                             scale_list[idx_ini:idx_fin], full_output=False, 
                             inverse=True, y_in=y_in, x_in=x_in, imlib=imlib2,
                             interpolation=interpolation, collapse=collapse)
            resadi_cube[i] = frame_i

        if verbose:
            print('De-rotating and combining residuals')
            timing(start_time)
        der_res = cube_derotate(resadi_cube, angle_list, nproc=nproc, 
                                imlib=imlib, interpolation=interpolation,
                                **rot_options)
        frame = cube_collapse(der_res, mode=collapse, w=weights)
        cube_allfr_residuals = res_cube
        cube_adi_residuals = resadi_cube
        return cube_allfr_residuals, cube_adi_residuals, frame

    # When ncomp is a tuple, pca_grid is called
    elif isinstance(ncomp, tuple):
        gridre = pca_grid(big_cube, angle_list, fwhm, range_pcs=ncomp,
                          source_xy=source_xy, cube_ref=None, mode='fullfr',
                          svd_mode=svd_mode, scaling=scaling,
                          mask_center_px=mask_center_px, fmerit='mean',
                          collapse=collapse, 
                          ifs_collapse_range=ifs_collapse_range, 
                          verbose=verbose, full_output=full_output, debug=False,
                          plot=verbose, start_time=start_time,
                          scale_list=scale_list, initial_4dshape=cube.shape,
                          weights=weights, nproc=nproc, imlib=imlib, 
                          interpolation=interpolation, **rot_options)
        return gridre

    else:
        raise TypeError("`ncomp` must be an int, float or a tuple for "
                        "single-pass PCA")


def _adimsdi_doublepca(cube, angle_list, scale_list, ncomp, scaling,
                       mask_center_px, svd_mode, imlib, imlib2, interpolation,
                       collapse, ifs_collapse_range, verbose, start_time, nproc,
                       weights=None, fwhm=4, conv=False, mask_rdi=None, 
                       cube_sig=None, **rot_options):
    """
    Handle the full-frame ADI+mSDI double PCA post-processing.

    """
    z, n, y_in, x_in = cube.shape

    global ARRAY
    ARRAY = cube  # to be passed to _adimsdi_doublepca_ifs

    if not isinstance(ncomp, tuple):
        raise TypeError("`ncomp` must be a tuple when a double pass PCA"
                        " is performed")
    else:
        ncomp_ifs, ncomp_adi = ncomp

    angle_list = check_pa_vector(angle_list)
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
        if ncomp_ifs is None:
            print('Combining multi-spectral frames (skipping PCA)')
        else:
            print('First PCA stage exploiting spectral variability')

    if ncomp_ifs is not None and ncomp_ifs > z:
        ncomp_ifs = min(ncomp_ifs, z)
        msg = 'Number of PCs too high (max PCs={}), using {} PCs instead'
        print(msg.format(z, ncomp_ifs))

    res = pool_map(nproc, _adimsdi_doublepca_ifs, iterable(range(n)), ncomp_ifs,
                   scale_list, scaling, mask_center_px, svd_mode, imlib2, 
                   interpolation, collapse, ifs_collapse_range, fwhm, conv,
                   mask_rdi)
    residuals_cube_channels = np.array(res)

    if verbose:
        timing(start_time)

    # de-rotation of the PCA processed channels, ADI fashion
    if ncomp_adi is None:
        if verbose:
            print('{} ADI frames'.format(n))
            print('De-rotating and combining frames (skipping PCA)')
        residuals_cube_channels_ = cube_derotate(residuals_cube_channels,
                                                 angle_list, nproc=nproc,
                                                 imlib=imlib,
                                                 interpolation=interpolation,
                                                 **rot_options)
        frame = cube_collapse(residuals_cube_channels_, mode=collapse, 
                              w=weights)
        if verbose:
            timing(start_time)
    else:
        if ncomp_adi > n:
            ncomp_adi = n
            print('Number of PCs too high, using  maximum of {} PCs '
                  'instead'.format(n))
        if verbose:
            print('{} ADI frames'.format(n))
            print('Second PCA stage exploiting rotational variability')

        res_ifs_adi = _project_subtract(residuals_cube_channels, None,
                                        ncomp_adi, scaling, mask_center_px,
                                        svd_mode, verbose=False,
                                        full_output=False, cube_sig=cube_sig)
        if verbose:
            print('De-rotating and combining residuals')
        der_res = cube_derotate(res_ifs_adi, angle_list, nproc=nproc, 
                                imlib=imlib, interpolation=interpolation,
                                **rot_options)
        residuals_cube_channels_ = der_res
        frame = cube_collapse(residuals_cube_channels_, mode=collapse, 
                              w=weights)
        if verbose:
            timing(start_time)
    return residuals_cube_channels, residuals_cube_channels_, frame


def _adimsdi_doublepca_ifs(fr, ncomp, scale_list, scaling, mask_center_px,
                           svd_mode, imlib, interpolation, collapse, 
                           ifs_collapse_range, fwhm, conv, mask_rdi=None):
    """
    Called by _adimsdi_doublepca with pool_map.
    """
    global ARRAY

    z, n, y_in, x_in = ARRAY.shape
    multispec_fr = ARRAY[:, fr, :, :]

    if ifs_collapse_range == 'all':
        idx_ini = 0
        idx_fin = z
    else:
        idx_ini = ifs_collapse_range[0]
        idx_fin = ifs_collapse_range[1]

    if ncomp is None:
        frame_i = cube_collapse(multispec_fr[idx_ini:idx_fin])
    else:
        cube_resc = scwave(multispec_fr, scale_list, imlib=imlib, 
                           interpolation=interpolation)[0]
        if conv:
            # convolve all frames with the same kernel
            cube_resc = cube_filter_lowpass(cube_resc, mode='gauss', 
                                            fwhm_size=fwhm, verbose=False)
        if mask_rdi is None:
            residuals = _project_subtract(cube_resc, None, ncomp, scaling,
                                          mask_center_px, svd_mode, 
                                          verbose=False, full_output=False)
        else:
            residuals = np.zeros_like(cube_resc)
            for i in range(z):
                cube_tmp = np.array([cube_resc[i]])
                cube_ref = np.array([cube_resc[j] for j in range(z) if j!=i])
                residuals[i] = cube_subtract_sky_pca(cube_tmp, cube_ref, 
                                                     mask_rdi, ncomp=ncomp, 
                                                     full_output=False)        

        frame_i = scwave(residuals[idx_ini:idx_fin], scale_list[idx_ini:idx_fin], 
                         full_output=False, inverse=True, y_in=y_in, x_in=x_in,
                         imlib=imlib, interpolation=interpolation, 
                         collapse=collapse)
        if mask_center_px:
            frame_i = mask_circle(frame_i, mask_center_px)

    return frame_i


def _adi_rdi_pca(cube, cube_ref, angle_list, ncomp, scaling, mask_center_px,
                 svd_mode, imlib, interpolation, collapse, verbose, start_time,
                 nproc, weights=None, mask_rdi=None, cube_sig=None, 
                 **rot_options):
    """ Handles the ADI+RDI post-processing.
    """
    n, y, x = cube.shape
    n_ref, y_ref, x_ref = cube_ref.shape
    angle_list = check_pa_vector(angle_list)
    if not isinstance(ncomp, int):
        raise TypeError("`ncomp` must be an int in the ADI+RDI case")
    if ncomp > n_ref:
        msg = 'Requested number of PCs ({}) higher than the number of frames '+\
              'in the reference cube ({}); using the latter instead.'
        print(msg.format(ncomp,n_ref))
        ncomp = n_ref

    if not cube_ref.ndim == 3:
        msg = 'Input reference array is not a cube or 3d array'
        raise ValueError(msg)
    if not y_ref == y and x_ref == x:
        msg = 'Reference and target frames have different shape'
        raise TypeError(msg)

    if mask_rdi is None:
        residuals_result = _project_subtract(cube, cube_ref, ncomp, scaling,
                                             mask_center_px, svd_mode, verbose,
                                             True, cube_sig=cube_sig)
        residuals_cube = residuals_result[0]
        reconstructed = residuals_result[1]
        V = residuals_result[2]
        pcs = reshape_matrix(V, y, x)
        recon = reshape_matrix(reconstructed, y, x)
    else:
        residuals_result = cube_subtract_sky_pca(cube, cube_ref, mask_rdi,
                                                 ncomp=ncomp, full_output=True)
        residuals_cube = residuals_result[0]
        pcs = residuals_result[2]
        recon = residuals_result[-1]
        
    residuals_cube_ = cube_derotate(residuals_cube, angle_list, nproc=nproc,
                                    imlib=imlib, interpolation=interpolation,
                                    **rot_options)
    frame = cube_collapse(residuals_cube_, mode=collapse, w=weights)
    if mask_center_px:
        frame = mask_circle(frame, mask_center_px)

    if verbose:
        print('Done de-rotating and combining')
        timing(start_time)

    return pcs, recon, residuals_cube, residuals_cube_, frame


def _project_subtract(cube, cube_ref, ncomp, scaling, mask_center_px,
                      svd_mode, verbose, full_output, indices=None, frame=None,
                      cube_sig=None):
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
    svd_mode : str
        Mode for SVD computation. See ``pca`` docstrings.
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
    _, y, x = cube.shape
    if isinstance(ncomp, int):
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

        # a rotation threshold is used (frames are processed one by one)
        if indices is not None and frame is not None:
            ref_lib = ref_lib[indices]
            if ref_lib.shape[0] <= 10:
                raise RuntimeError('Less than 10 frames left in the PCA library'
                                   ', Try decreasing the parameter delta_rot')
            curr_frame = matrix[frame]  # current frame
            curr_frame_emp = matrix_emp[frame]
            V = svd_wrapper(ref_lib, svd_mode, ncomp, False)
            transformed = np.dot(curr_frame_emp, V.T)
            reconstructed = np.dot(transformed.T, V)
            residuals = curr_frame - reconstructed
            if full_output:
                return ref_lib.shape[0], residuals, reconstructed
            else:
                return ref_lib.shape[0], residuals

        # the whole matrix is processed at once
        else:
            V = svd_wrapper(ref_lib, svd_mode, ncomp, verbose)
            transformed = np.dot(V, matrix_emp.T)
            reconstructed = np.dot(transformed.T, V)
            residuals = matrix - reconstructed
            residuals_res = reshape_matrix(residuals, y, x)
            if full_output:
                return residuals_res, reconstructed, V
            else:
                return residuals_res

    elif isinstance(ncomp, float):
        if not 1 > ncomp > 0:
            raise ValueError("when `ncomp` if float, it must lie in the "
                             "interval (0,1]")

        svdecomp = SVDecomposer(cube, mode='fullfr', svd_mode=svd_mode,
                                scaling=scaling, verbose=verbose)
        _ = svdecomp.get_cevr(plot=False)
        # in this case ncomp is the desired CEVR
        cevr = ncomp
        ncomp = svdecomp.cevr_to_ncomp(cevr)
        V = svdecomp.v[:ncomp]
        transformed = np.dot(V, svdecomp.matrix.T)
        reconstructed = np.dot(transformed.T, V)
        residuals = svdecomp.matrix - reconstructed
        residuals_res = reshape_matrix(residuals, y, x)

        if verbose and isinstance(cevr, float):
            print("Components used : {}".format(V.shape[0]))

        if full_output:
            return residuals_res, reconstructed, V
        else:
            return residuals_res