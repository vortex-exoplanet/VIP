#! /usr/bin/env python

"""
Module with local/smart PCA (annulus or patch-wise in a multi-processing
fashion) model PSF subtraction for ADI, ADI+SDI (IFS) and ADI+RDI datasets.
"""

__author__ = 'Carlos Alberto Gomez Gonzalez'
__all__ = ['pca_annular']

import numpy as np
from multiprocessing import cpu_count
from ..preproc import (cube_derotate, cube_collapse, check_pa_vector,
                       check_scal_vector)
from ..preproc import cube_rescaling_wavelengths as scwave
from ..preproc.derotation import _find_indices_adi, _define_annuli
from ..preproc.rescaling import _find_indices_sdi
from ..config import time_ini, timing
from ..config.utils_conf import pool_map, iterable
from ..var import get_annulus_segments, matrix_scaling
from ..stats import descriptive_stats
from .svd import get_eigenvectors

def pca_annular(cube, angle_list, cube_ref=None, scale_list=None, radius_int=0,
                fwhm=4, asize=4, n_segments=1, delta_rot=(0.1, 1),
                delta_sep=(0.1, 1), ncomp=1, svd_mode='lapack', nproc=1,
                min_frames_lib=2, max_frames_lib=200, tol=1e-1, scaling=None,
                imlib='vip-fft', interpolation='lanczos4', collapse='median',
                ifs_collapse_range='all', full_output=False, verbose=True,
                weights=None, cube_sig=None, **rot_options):
    """ PCA model PSF subtraction for ADI, ADI+RDI or ADI+mSDI (IFS) data. The
    PCA model is computed locally in each annulus (or annular sectors according
    to ``n_segments``). For each sector we discard reference frames taking into
    account a parallactic angle threshold (``delta_rot``) and optionally a
    radial movement threshold (``delta_sep``) for 4d cubes.

    For ADI+RDI data, it computes the principal components from the reference
    library/cube, forcing pixel-wise temporal standardization. The number of
    principal components can be automatically adjusted by the algorithm by
    minimizing the residuals inside each patch/region.

    Parameters
    ----------
    cube : numpy ndarray, 3d or 4d
        Input cube.
    angle_list : numpy ndarray, 1d
        Corresponding parallactic angle for each frame.
    cube_ref : numpy ndarray, 3d, optional
        Reference library cube. For Reference Star Differential Imaging.
    scale_list : numpy ndarray, 1d
        Scaling factors in case of IFS data (ADI+mSDI cube). Usually, the
        scaling factors are the central channel wavelength divided by the
        shortest wavelength in the cube (more thorough approaches can be used
        to get the scaling factors). This scaling factors are used to re-scale
        the spectral channels and align the speckles.
    radius_int : int, optional
        The radius of the innermost annulus. By default is 0, if >0 then the
        central circular region is discarded.
    fwhm : float, optional
        Size of the FHWM in pixels. Default is 4.
    asize : float, optional
        The size of the annuli, in pixels.
    n_segments : int or list of ints or 'auto', optional
        The number of segments for each annulus. When a single integer is given
        it is used for all annuli. When set to 'auto', the number of segments is
        automatically determined for every annulus, based on the annulus width.
    delta_rot : float or tuple of floats, optional
        Factor for adjusting the parallactic angle threshold, expressed in
        FWHM. Default is 1 (excludes 1 FHWM on each side of the considered
        frame). If a tuple of two floats is provided, they are used as the lower
        and upper intervals for the threshold (grows linearly as a function of
        the separation).
    delta_sep : float or tuple of floats, optional
        The threshold separation in terms of the mean FWHM (for ADI+mSDI data).
        If a tuple of two values is provided, they are used as the lower and
        upper intervals for the threshold (grows as a function of the
        separation).
    ncomp : 'auto', int, tuple, 1d numpy array or tuple, optional
        How many PCs are used as a lower-dimensional subspace to project the
        target (sectors of) frames. Depends on the dimensionality of `cube`.

        * ADI and ADI+RDI case: if a single integer is provided, then the same
          number of PCs will be subtracted at each separation (annulus). If a
          tuple is provided, then a different number of PCs will be used for
          each annulus (starting with the innermost one). If ``ncomp`` is set to
          ``auto`` then the number of PCs are calculated for each region/patch
          automatically.

        * ADI+mSDI case: ``ncomp`` must be a tuple (two integers) with the
          number of PCs obtained from each multi-spectral frame (for each
          sector) and the number of PCs used in the second PCA stage (ADI
          fashion, using the residuals of the first stage). If None then the
          second PCA stage is skipped and the residuals are de-rotated and
          combined.

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

    nproc : None or int, optional
        Number of processes for parallel computing. If None the number of
        processes will be set to (cpu_count()/2).
    min_frames_lib : int, optional
        Minimum number of frames in the PCA reference library.
    max_frames_lib : int, optional
        Maximum number of frames in the PCA reference library. The more 
        distant/decorrelated frames are removed from the library.
    tol : float, optional
        Stopping criterion for choosing the number of PCs when ``ncomp``
        is None. Lower values will lead to smaller residuals and more PCs.
    scaling : {None, "temp-mean", spat-mean", "temp-standard",
        "spat-standard"}, None or str optional
        Pixel-wise scaling mode using ``sklearn.preprocessing.scale``
        function. If set to None, the input matrix is left untouched. Otherwise:

        ``temp-mean``: temporal px-wise mean is subtracted.

        ``spat-mean``: spatial mean is subtracted.

        ``temp-standard``: temporal mean centering plus scaling pixel values
        to unit variance. HIGHLY RECOMMENDED FOR ASDI AND RDI CASES.

        ``spat-standard``: spatial mean centering plus scaling pixel values
        to unit variance.

    imlib : str, optional
        See the documentation of the ``vip_hci.preproc.frame_rotate`` function.
    interpolation : str, optional
        See the documentation of the ``vip_hci.preproc.frame_rotate`` function.
    collapse : {'median', 'mean', 'sum', 'trimmean'}, str optional
        Sets the way of collapsing the frames for producing a final image.
    ifs_collapse_range: str 'all' or tuple of 2 int
        If a tuple, it should contain the first and last channels where the mSDI
        residual channels will be collapsed (by default collapses all channels).
    full_output: boolean, optional
        Whether to return the final median combined image only or with other
        intermediate arrays.
    verbose : bool, optional
        If True prints to stdout intermediate info. If set to 2, the number of
        frames in library and number of PCs are printed for each annular quadrant.
    weights: 1d numpy array or list, optional
        Weights to be applied for a weighted mean. Need to be provided if
        collapse mode is 'wmean'.
    cube_sig: numpy ndarray, opt
        Cube with estimate of significant authentic signals. If provided, this
        will be subtracted before projecting cube onto reference cube.
    rot_options: dictionary, optional
        Dictionary with optional keyword values for "border_mode", "mask_val",  
        "edge_blend", "interp_zeros", "ker" (see documentation of 
        ``vip_hci.preproc.frame_rotate``)
        
    Returns
    -------
    - If full_output is False:
    frame : numpy ndarray, 2d
        Median combination of the de-rotated cube.
    - If full_output is True:
    array_out : numpy ndarray, 3d
        Cube of residuals.
    array_der : numpy ndarray, 3d
        Cube residuals after de-rotation.
    frame : numpy ndarray, 2d
        Median combination of the de-rotated cube.
    """
    if verbose:
        global start_time
        start_time = time_ini()

    # ADI or ADI+RDI data
    if cube.ndim == 3:
        res = _pca_adi_rdi(cube, angle_list, radius_int, fwhm, asize,
                           n_segments, delta_rot, ncomp, svd_mode, nproc,
                           min_frames_lib, max_frames_lib, tol, scaling, imlib,
                           interpolation, collapse, True, verbose, cube_ref,
                           weights, cube_sig, **rot_options)

        cube_out, cube_der, frame = res
        if full_output:
            return cube_out, cube_der, frame
        else:
            return frame

    # ADI+mSDI (IFS) datacubes
    elif cube.ndim == 4:
        global ARRAY
        ARRAY = cube

        z, n, y_in, x_in = cube.shape
        fwhm = int(np.round(np.mean(fwhm)))
        n_annuli = int((y_in / 2 - radius_int) / asize)

        if scale_list is None:
            raise ValueError('Scaling factors vector must be provided')
        else:
            if np.array(scale_list).ndim > 1:
                raise ValueError('Scaling factors vector is not 1d')
            if not scale_list.shape[0] == z:
                raise ValueError('Scaling factors vector has wrong length')

        if not isinstance(ncomp, tuple):
            raise TypeError("`ncomp` must be a tuple of two integers when "
                            "`cube` is a 4d array")
        else:
            ncomp2 = ncomp[1]
            ncomp = ncomp[0]

        if verbose:
            print('First PCA subtraction exploiting the spectral variability')
            print('{} spectral channels per IFS frame'.format(z))
            print('N annuli = {}, mean FWHM = {:.3f}'.format(n_annuli, fwhm))

        res = pool_map(nproc, _pca_sdi_fr, iterable(range(n)), scale_list,
                       radius_int, fwhm, asize, n_segments, delta_sep, ncomp,
                       svd_mode, tol, scaling, imlib, interpolation, collapse,
                       ifs_collapse_range, verbose=verbose)
        residuals_cube_channels = np.array(res)

        # Exploiting rotational variability
        if verbose:
            timing(start_time)
            print('{} ADI frames'.format(n))

        if ncomp2 is None:
            if verbose:
                print('Skipping the second PCA subtraction')

            cube_out = residuals_cube_channels
            cube_der = cube_derotate(cube_out, angle_list, nproc=nproc,
                                     imlib=imlib, interpolation=interpolation, 
                                     **rot_options)
            frame = cube_collapse(cube_der, mode=collapse, w=weights)

        else:
            if verbose:
                print('Second PCA subtraction exploiting the angular '
                      'variability')

            res = _pca_adi_rdi(residuals_cube_channels, angle_list, radius_int,
                               fwhm, asize, n_segments, delta_rot, ncomp2,
                               svd_mode, nproc, min_frames_lib, max_frames_lib,
                               tol, scaling, imlib, interpolation, collapse,
                               full_output, verbose, None, weights, cube_sig,
                               **rot_options)
            if full_output:
                cube_out, cube_der, frame = res
            else:
                frame = res

        if full_output:
            return cube_out, cube_der, frame
        else:
            return frame

    else:
        raise TypeError('Input array is not a cube or 3d array')


################################################################################
# Functions encapsulating portions of the main algorithm
################################################################################


def _pca_sdi_fr(fr, wl, radius_int, fwhm, asize, n_segments, delta_sep, ncomp, 
                svd_mode, tol, scaling, imlib, interpolation, collapse,
                ifs_collapse_range):
    """ Optimized PCA subtraction on a multi-spectral frame (IFS data).
    """
    z, n, y_in, x_in = ARRAY.shape

    scale_list = check_scal_vector(wl)
    # rescaled cube, aligning speckles
    multispec_fr = scwave(ARRAY[:, fr, :, :], scale_list,
                          imlib=imlib, interpolation=interpolation)[0]

    # Exploiting spectral variability (radial movement)
    fwhm = int(np.round(np.mean(fwhm)))
    n_annuli = int((y_in / 2 - radius_int) / asize)

    if isinstance(n_segments, int):
        n_segments = [n_segments for _ in range(n_annuli)]
    elif n_segments == 'auto':
        n_segments = list()
        n_segments.append(2)  # for first annulus
        n_segments.append(3)  # for second annulus
        ld = 2 * np.tan(360 / 4 / 2) * asize
        for i in range(2, n_annuli):  # rest of annuli
            radius = i * asize
            ang = np.rad2deg(2 * np.arctan(ld / (2 * radius)))
            n_segments.append(int(np.ceil(360 / ang)))

    cube_res = np.zeros_like(multispec_fr)    # shape (z, resc_y, resc_x)

    if isinstance(delta_sep, tuple):
        delta_sep_vec = np.linspace(delta_sep[0], delta_sep[1], n_annuli)
    else:
        delta_sep_vec = [delta_sep] * n_annuli

    for ann in range(n_annuli):
        if ann == n_annuli - 1:
            inner_radius = radius_int + (ann * asize - 1)
        else:
            inner_radius = radius_int + ann * asize
        ann_center = inner_radius + (asize / 2)

        indices = get_annulus_segments(multispec_fr[0], inner_radius, asize,
                                       n_segments[ann])
        # Library matrix is created for each segment and scaled if needed
        for seg in range(n_segments[ann]):
            yy = indices[seg][0]
            xx = indices[seg][1]
            matrix = multispec_fr[:, yy, xx]  # shape (z, npx_annsegm)
            matrix = matrix_scaling(matrix, scaling)

            for j in range(z):
                indices_left = _find_indices_sdi(wl, ann_center, j,
                                                 fwhm, delta_sep_vec[ann])
                matrix_ref = matrix[indices_left]
                curr_frame = matrix[j]  # current frame
                V = get_eigenvectors(ncomp, matrix_ref, svd_mode,
                                     noise_error=tol, debug=False, 
                                     scaling=scaling)
                transformed = np.dot(curr_frame, V.T)
                reconstructed = np.dot(transformed.T, V)
                residuals = curr_frame - reconstructed
                # return residuals, V.shape[0], matrix_ref.shape[0]
                cube_res[j, yy, xx] = residuals

    if ifs_collapse_range == 'all':
        idx_ini = 0
        idx_fin = z
    else:
        idx_ini = ifs_collapse_range[0]
        idx_fin = ifs_collapse_range[1]
    frame_desc = scwave(cube_res[idx_ini:idx_fin], scale_list[idx_ini:idx_fin],
                        full_output=False, inverse=True,
                        y_in=y_in, x_in=x_in, imlib=imlib,
                        interpolation=interpolation, collapse=collapse)
    return frame_desc


def _pca_adi_rdi(cube, angle_list, radius_int=0, fwhm=4, asize=2, n_segments=1,
                 delta_rot=1, ncomp=1, svd_mode='lapack', nproc=None,
                 min_frames_lib=2, max_frames_lib=200, tol=1e-1, scaling=None,
                 imlib='vip-fft', interpolation='lanczos4', collapse='median',
                 full_output=False, verbose=1, cube_ref=None, weights=None,
                 cube_sig=None, **rot_options):
    """ PCA exploiting angular variability (ADI fashion).
    """
    array = cube
    if array.ndim != 3:
        raise TypeError('Input array is not a cube or 3d array')
    if array.shape[0] != angle_list.shape[0]:
        raise TypeError('Input vector or parallactic angles has wrong length')

    n, y, _ = array.shape

    angle_list = check_pa_vector(angle_list)
    n_annuli = int((y / 2 - radius_int) / asize)

    if isinstance(delta_rot, tuple):
        delta_rot = np.linspace(delta_rot[0], delta_rot[1], num=n_annuli)
    elif isinstance(delta_rot, (int, float)):
        delta_rot = [delta_rot] * n_annuli

    if isinstance(n_segments, int):
        n_segments = [n_segments for _ in range(n_annuli)]
    elif n_segments == 'auto':
        n_segments = list()
        n_segments.append(2)  # for first annulus
        n_segments.append(3)  # for second annulus
        ld = 2 * np.tan(360 / 4 / 2) * asize
        for i in range(2, n_annuli):  # rest of annuli
            radius = i * asize
            ang = np.rad2deg(2 * np.arctan(ld / (2 * radius)))
            n_segments.append(int(np.ceil(360 / ang)))

    if verbose:
        msg = 'N annuli = {}, FWHM = {:.3f}'
        print(msg.format(n_annuli, fwhm))
        print('PCA per annulus (or annular sectors):')

    if nproc is None:   # Hyper-threading "duplicates" the cores -> cpu_count/2
        nproc = cpu_count() // 2

    # The annuli are built, and the corresponding PA thresholds for frame
    # rejection are calculated (at the center of the annulus)
    cube_out = np.zeros_like(array)
    for ann in range(n_annuli):
        if isinstance(ncomp, tuple) or isinstance(ncomp, np.ndarray):
            if len(ncomp) == n_annuli:
                ncompann = ncomp[ann]
            else:
                raise TypeError('If `ncomp` is a tuple, it must match the '
                                'number of annuli')
        else:
            ncompann = ncomp

        n_segments_ann = n_segments[ann]
        res_ann_par = _define_annuli(angle_list, ann, n_annuli, fwhm,
                                     radius_int, asize, delta_rot[ann],
                                     n_segments_ann, verbose, True)
        pa_thr, inner_radius, ann_center = res_ann_par
        indices = get_annulus_segments(array[0], inner_radius, asize,
                                       n_segments_ann)
        # Library matrix is created for each segment and scaled if needed
        for j in range(n_segments_ann):
            yy = indices[j][0]
            xx = indices[j][1]
            matrix_segm = array[:, yy, xx]  # shape [nframes x npx_segment]
            matrix_segm = matrix_scaling(matrix_segm, scaling)
            if cube_ref is not None:
                matrix_segm_ref = cube_ref[:, yy, xx]
                matrix_segm_ref = matrix_scaling(matrix_segm_ref, scaling)
            else:
                matrix_segm_ref = None
            if cube_sig is not None:
                matrix_sig_segm = cube_sig[:, yy, xx]
            else:
                matrix_sig_segm = None

            res = pool_map(nproc, do_pca_patch, matrix_segm, iterable(range(n)),
                           angle_list, fwhm, pa_thr, ann_center, svd_mode,
                           ncompann, min_frames_lib, max_frames_lib, tol,
                           matrix_segm_ref, matrix_sig_segm)

            res = np.array(res, dtype=object)
            residuals = np.array(res[:, 0])
            ncomps = res[:, 1]
            nfrslib = res[:, 2]
            for fr in range(n):
                cube_out[fr][yy, xx] = residuals[fr]

            # number of frames in library printed for each annular quadrant
            # number of PCs printed for each annular quadrant
            if verbose == 2:
                descriptive_stats(nfrslib, verbose=verbose,
                                  label='\tLIBsize: ')
                descriptive_stats(ncomps, verbose=verbose, label='\tNum PCs: ')

        if verbose == 1:
            print('Done PCA with {} for current annulus'.format(svd_mode))
            timing(start_time)

    # Cube is derotated according to the parallactic angle and collapsed
    cube_der = cube_derotate(cube_out, angle_list, nproc=nproc, imlib=imlib,
                             interpolation=interpolation, **rot_options)
    frame = cube_collapse(cube_der, mode=collapse, w=weights)
    if verbose:
        print('Done derotating and combining.')
        timing(start_time)
    if full_output:
        return cube_out, cube_der, frame
    else:
        return frame


def do_pca_patch(matrix, frame, angle_list, fwhm, pa_threshold, ann_center,
                 svd_mode, ncomp, min_frames_lib, max_frames_lib, tol,
                 matrix_ref, matrix_sig_segm):
    """ Does the SVD/PCA for each frame patch (small matrix). For each frame we
    find the frames to be rejected depending on the amount of rotation. The
    library is also truncated on the other end (frames too far or which have
    rotated more) which are more decorrelated to keep the computational cost
    lower. This truncation is done on the annuli after 10*FWHM and the goal is
    to keep min(num_frames/2, 200) in the library.
    """

    if pa_threshold != 0:
        # if ann_center > fwhm*10:
        indices_left = _find_indices_adi(angle_list, frame, pa_threshold, 
                                         truncate=True, 
                                         max_frames=max_frames_lib)
        # else:
        #    indices_left = _find_indices_adi(angle_list, frame,
        #                                     pa_threshold, truncate=False)
        msg = 'Too few frames left in the PCA library. '
        msg += 'Accepted indices length ({:.0f}) less than {:.0f}. '
        msg += 'Try decreasing either delta_rot or min_frames_lib.'
        try:
            if matrix_sig_segm is not None:
                data_ref = matrix[indices_left]-matrix_sig_segm[indices_left]
            else:
                data_ref = matrix[indices_left]
        except IndexError:
            if matrix_ref is None:
                raise RuntimeError(msg.format(0, min_frames_lib))
            data_ref = None

        if data_ref.shape[0] < min_frames_lib and matrix_ref is None:
            raise RuntimeError(msg.format(len(indices_left), min_frames_lib))
    if matrix_ref is not None:
        #data_ref = None
    #if matrix_ref is not None:
        # Stacking the ref and the target ref (pa thresh) libraries
        if data_ref is not None:
            data_ref = np.vstack((matrix_ref, data_ref))
        else:
            data_ref = matrix_ref
    elif pa_threshold == 0:
        if matrix_sig_segm is not None:
            data_ref = matrix-matrix_sig_segm
        else:
            data_ref = matrix

    curr_frame = matrix[frame]  # current frame
    if matrix_sig_segm is not None:
        curr_frame_emp = matrix[frame]-matrix_sig_segm[frame]
    else:
        curr_frame_emp = curr_frame
    V = get_eigenvectors(ncomp, data_ref, svd_mode, noise_error=tol)
    transformed = np.dot(curr_frame_emp, V.T)
    reconstructed = np.dot(transformed.T, V)
    residuals = curr_frame - reconstructed
    return residuals, V.shape[0], data_ref.shape[0]
