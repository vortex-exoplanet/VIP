#! /usr/bin/env python

"""
Module with local/smart PCA (annulus or patch-wise) model PSF subtraction for
ADI, ADI+SDI (IFS) and ADI+RDI datasets. This implementation make use of
Python multiprocessing capabilities.
"""

from __future__ import division, print_function

__author__ = 'Carlos Alberto Gomez Gonzalez'
__all__ = ['pca_annular',
           'pca_rdi_annular']

import numpy as np
from scipy import stats
from multiprocessing import cpu_count
from ..preproc import (cube_derotate, cube_collapse, check_pa_vector,
                       check_scal_vector)
from ..preproc import cube_rescaling_wavelengths as scwave
from ..preproc.derotation import _find_indices_adi, _define_annuli
from ..preproc.rescaling import _find_indices_sdi
from ..conf import time_ini, timing
from ..conf.utils_conf import pool_map, fixed
from ..var import get_annulus_segments, matrix_scaling, get_annulus
from ..stats import descriptive_stats
from .svd import get_eigenvectors


def pca_annular(cube, angle_list, scale_list=None, radius_int=0, fwhm=4,
                asize=4, n_segments=1, delta_rot=1, delta_sep=(0.1, 1), ncomp=1,
                ncomp2=1, svd_mode='lapack', nproc=1, min_frames_lib=2,
                max_frames_lib=200, tol=1e-1, scaling=None, imlib='opencv',
                interpolation='lanczos4', collapse='median', full_output=False,
                verbose=True):
    """ PCA model PSF subtraction for ADI and ADI + mSDI (IFS) data. The PCA
    model is computed locally in each annulus (or annular sectors according to
    ``n_segments``). For each sector we discard reference frames taking into
    account a parallactic angle threshold (``delta_rot``) and a radial movement
    threshold (``delta_sep``).

    Parameters
    ----------
    cube : array_like, 3d or 4d
        Input cube.
    angle_list : array_like, 1d
        Corresponding parallactic angle for each frame.
    scale_list :
        Scaling factors in case of IFS data (ADI+mSDI cube). Usually, the
        scaling factors are the central channel wavelength divided by the
        shortest wavelength in the cube (more thorough approaches can be used
        to get the scaling factors). This scaling factors are used to re-scale
        the spectral channels and align the speckles.
    radius_int : int, optional
        The radius of the innermost annulus. By default is 0, if >0 then the
        central circular area is discarded.
    fwhm : float, optional
        Known size of the FHWM in pixels to be used. Default is 4.
    asize : float, optional
        The size of the annuli, in pixels.
    n_segments : int or list of ints or 'auto', optional
        The number of segments for each annulus. When a single integer is given
        it is used for all annuli. When set to 'auto', the number of segments is
        automatically determined for every annulus, based on the annulus width.
    delta_rot : float, optional
        Factor for increasing the parallactic angle threshold, expressed in
        FWHM. Default is 1 (excludes 1 FHWM on each side of the considered
        frame). According to Absil+13, a slightly better contrast can be reached
        for the innermost annuli if we consider a ``delta_rot`` condition as
        small as 0.1 lambda/D. This is because at very small separation, the
        effect of speckle correlation is more significant than self-subtraction.
    delta_sep : float or tuple of floats, optional
        The threshold separation in terms of the mean FWHM (for ADI+mSDI data).
        If a tuple of two values is provided, they are used as the lower and
        upper intervals for the threshold (grows as a function of the
        separation).
    ncomp : int or list or 1d numpy array, optional
        How many PCs are used as a lower-dimensional subspace to project the
        target (sectors of) frames. If ``auto`` it will be automatically
        determined. If ``cube`` is a 3d array (ADI), ``ncomp`` can be a list,
        in which case a different number of PCs will be used for each annulus
        (starting with the innermost one). If ``cube`` is a 4d array, then
        ``ncomp`` is the number of PCs obtained from each multi-spectral frame
        (for each sector).
    ncomp2 : int, optional
        Only used for ADI+mSDI (4d) cubes. ``ncomp2`` sets the number of PCs
        used in the second PCA stage (ADI fashion, using the residuals of the
        first stage). If None then the second PCA stage is skipped and the
        residuals are de-rotated and combined.
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
    nproc : None or int, optional
        Number of processes for parallel computing. If None the number of
        processes will be set to (cpu_count()/2).
    min_frames_lib : int, optional
        Minimum number of frames in the PCA reference library.
    max_frames_lib : int, optional
        Maximum number of frames in the PCA reference library for annuli beyond
        10*FWHM. The more distant/decorrelated frames are removed from the
        library.
    tol : float, optional
        Stopping criterion for choosing the number of PCs when ``ncomp``
        is None. Lower values will lead to smaller residuals and more PCs.
    scaling : {None, 'temp-mean', 'spat-mean', 'temp-standard', 'spat-standard'}
        With None, no scaling is performed on the input data before SVD. With
        "temp-mean" then temporal px-wise mean subtraction is done, with
        "spat-mean" then the spatial mean is subtracted, with "temp-standard"
        temporal mean centering plus scaling to unit variance is done and with
        "spat-standard" spatial mean centering plus scaling to unit variance is
        performed.
    imlib : str, optional
        See the documentation of the ``vip_hci.preproc.frame_rotate`` function.
    interpolation : str, optional
        See the documentation of the ``vip_hci.preproc.frame_rotate`` function.
    collapse : {'median', 'mean', 'sum', 'trimmean'}, str optional
        Sets the way of collapsing the frames for producing a final image.
    full_output: boolean, optional
        Whether to return the final median combined image only or with other
        intermediate arrays.
    verbose : bool, optional
        If True prints to stdout intermediate info.

    Returns
    -------
    frame : array_like, 2d
        Median combination of the de-rotated cube.
    If full_output is True:
    array_out : array_like, 3d
        Cube of residuals.
    array_der : array_like, 3d
        Cube residuals after de-rotation.

    """
    if verbose:
        global start_time
        start_time = time_ini()

    # ADI datacube
    if cube.ndim == 3:
        res = _pca_adi_ann(cube, angle_list, radius_int, fwhm, asize,
                           n_segments, delta_rot, ncomp, svd_mode, nproc,
                           min_frames_lib, max_frames_lib, tol, scaling, imlib,
                           interpolation, collapse, full_output, verbose)

        if verbose:
            print('Done derotating and combining.')
            timing(start_time)
        if full_output:
            cube_out, cube_der, frame = res
            return cube_out, cube_der, frame
        else:
            return res

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

        if verbose:
            print('First PCA subtraction exploiting the spectral variability')
            print('{} spectral channels per IFS frame'.format(z))
            print('N annuli = {}, mean FWHM = {:.3f}'.format(n_annuli, fwhm))

        res = pool_map(nproc, _pca_sdi_fr, fixed(range(n)), scale_list,
                       radius_int, fwhm, asize, n_segments, delta_sep, ncomp,
                       svd_mode, tol, scaling, imlib, interpolation, collapse,
                       verbose=verbose)
        residuals_cube_channels = np.array(res)

        # Exploiting rotational variability
        if verbose:
            timing(start_time)
            print('{} ADI frames'.format(n))

        if ncomp2 is None:
            if verbose:
                msg = 'Skipping the second PCA subtraction'
                print(msg)

            cube_out = residuals_cube_channels
            cube_der = cube_derotate(cube_out, angle_list, imlib=imlib,
                                     interpolation=interpolation)
            frame = cube_collapse(cube_der, mode=collapse)

        else:
            if verbose:
                msg = 'Second PCA subtraction exploiting the angular '
                msg += 'variability'
                print(msg)

            res = _pca_adi_ann(residuals_cube_channels, angle_list, radius_int,
                               fwhm, asize, n_segments, delta_rot, ncomp2,
                               svd_mode, nproc, min_frames_lib, max_frames_lib,
                               tol, scaling, imlib, interpolation, collapse,
                               full_output, verbose)
            if full_output:
                cube_out, cube_der, frame = res
            else:
                frame = res

        if verbose:
            print('Done derotating and combining.')
            timing(start_time)
        if full_output:
            return cube_out, cube_der, frame
        else:
            return frame

    else:
        raise TypeError('Input array is not a cube or 3d array')


def pca_rdi_annular(cube, angle_list, cube_ref, radius_int=0, asize=1, ncomp=1,
                    svd_mode='lapack', min_corr=0.9, fwhm=4,
                    scaling='temp-standard', imlib='opencv',
                    interpolation='lanczos4', collapse='median',
                    full_output=False, verbose=True):
    """ Annular PCA with Reference Library + Correlation + standardization

    In the case of having a large number of reference images, e.g. for a survey
    on a single instrument, we can afford a better selection of the library by
    constraining the correlation with the median of the science dataset and by
    working on an annulus-wise way. As with other local PCA algorithms in VIP
    the number of principal components can be automatically adjusted by the
    algorithm by minmizing the residuals in the given patch (a la LOCI).

    Parameters
    ----------
    cube : array_like, 3d
        Input science cube.
    angle_list : array_like, 1d
        Corresponding parallactic angle for each frame.
    cube_ref : array_like, 3d
        Reference library cube. For Reference Star Differential Imaging.
    radius_int : int, optional
        The radius of the innermost annulus. By default is 0, if >0 then the
        central circular area is discarded.
    asize : float, optional
        The size of the annuli, in FWHM. Default is 3.
    ncomp : int, optional
        How many PCs are kept. If none it will be automatically determined.
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
    min_corr : int, optional
        Level of linear correlation between the library patches and the median
        of the science. Deafult is 0.9.
    fwhm : float, optional
        Known size of the FHWM in pixels to be used. Deafult is 4.
    scaling : {None, 'temp-mean', 'spat-mean', 'temp-standard', 'spat-standard'}
        With None, no scaling is performed on the input data before SVD. With
        "temp-mean" then temporal px-wise mean subtraction is done, with
        "spat-mean" then the spatial mean is subtracted, with "temp-standard"
        temporal mean centering plus scaling to unit variance is done and with
        "spat-standard" spatial mean centering plus scaling to unit variance is
        performed.
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
        If True prints to stdout intermediate info.

    Returns
    -------
    frame : array_like, 2d
        Median combination of the de-rotated cube.
    If full_output is True:
    array_out : array_like, 3d
        Cube of residuals.
    array_der : array_like, 3d
        Cube residuals after de-rotation.

    """
    def define_annuli(angle_list, ann, n_annuli, fwhm, radius_int,
                      annulus_width,
                      verbose):
        """ Defining the annuli """
        if ann == n_annuli - 1:
            inner_radius = radius_int + (ann * annulus_width - 1)
        else:
            inner_radius = radius_int + ann * annulus_width
        ann_center = (inner_radius + (annulus_width / 2.0))

        if verbose:
            msg2 = 'Annulus {}, Inn radius = {:.2f}, Ann center = {:.2f} '
            print(msg2.format(int(ann + 1), inner_radius, ann_center))
        return inner_radius, ann_center

    def fr_ref_correlation(vector, matrix):
        """ Getting the correlations """
        lista = []
        for i in range(matrix.shape[0]):
            pears, _ = stats.pearsonr(vector, matrix[i])
            lista.append(pears)

        return lista

    def do_pca_annulus(ncomp, matrix, svd_mode, noise_error, data_ref):
        """ PCA for given annulus """
        V = get_eigenvectors(ncomp, matrix, svd_mode,
                             noise_error=noise_error,
                             data_ref=data_ref, debug=False)
        # new variables as linear combinations of the original variables in
        # matrix.T with coefficientes from EV
        transformed = np.dot(V, matrix.T)
        reconstructed = np.dot(V.T, transformed)
        residuals = matrix - reconstructed.T
        return residuals, V.shape[0]

    #---------------------------------------------------------------------------
    array = cube
    array_ref = cube_ref
    if array.ndim != 3:
        raise TypeError('Input array is not a cube or 3d array.')
    if array.shape[0] != angle_list.shape[0]:
        raise TypeError(
            'Input vector or parallactic angles has wrong length.')

    n, y, _ = array.shape
    if verbose:  start_time = time_ini()

    angle_list = check_pa_vector(angle_list)

    annulus_width = asize * fwhm  # equal size for all annuli
    n_annuli = int(np.floor((y / 2 - radius_int) / annulus_width))
    if verbose:
        msg = '# annuli = {}, Ann width = {}, FWHM = {:.3f}\n'
        print(msg.format(n_annuli, annulus_width, fwhm))
        print('PCA will be done locally per annulus and per quadrant.\n')

    cube_out = np.zeros_like(array)
    for ann in range(n_annuli):
        inner_radius, _ = define_annuli(angle_list, ann, n_annuli, fwhm,
                                        radius_int, annulus_width, verbose)
        indices = get_annulus(array[0], inner_radius, annulus_width,
                              output_indices=True)
        yy = indices[0]
        xx = indices[1]

        matrix = array[:, yy, xx]  # shape [nframes x npx_ann]
        matrix_ref = array_ref[:, yy, xx]

        corr = fr_ref_correlation(np.median(matrix, axis=0), matrix_ref)
        indcorr = np.where(np.abs(corr) >= min_corr)
        data_ref = matrix_ref[indcorr]
        nfrslib = data_ref.shape[0]

        if nfrslib < 5:
            msg = 'Too few frames left (<5) fulfill the given correlation'
            msg += ' level. Try decreasing it'
            raise RuntimeError(msg)

        matrix = matrix_scaling(matrix, scaling)
        data_ref = matrix_scaling(data_ref, scaling)

        residuals, ncomps = do_pca_annulus(ncomp, matrix, svd_mode, 10e-3,
                                           data_ref)
        cube_out[:, yy, xx] = residuals

        if verbose in [1, 2]:
            print('# frames in LIB = {}'.format(nfrslib))
            print('# PCs = {}'.format(ncomps))
            print('Done PCA with {} for current annulus'.format(svd_mode))
            timing(start_time)

    cube_der = cube_derotate(cube_out, angle_list, imlib=imlib,
                             interpolation=interpolation)
    frame = cube_collapse(cube_der, mode=collapse)
    if verbose:
        print('Done derotating and combining.')
        timing(start_time)
    if full_output:
        return cube_out, cube_der, frame
    else:
        return frame


################################################################################
# Help functions (encapsulating portions of the main algorithm)
################################################################################

def _pca_sdi_fr(fr, wl, radius_int, fwhm, asize, n_segments, delta_sep,
                ncomp, svd_mode, tol, scaling, imlib, interpolation, collapse):
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
                                     noise_error=tol, debug=False)
                transformed = np.dot(curr_frame, V.T)
                reconstructed = np.dot(transformed.T, V)
                residuals = curr_frame - reconstructed
                # return residuals, V.shape[0], matrix_ref.shape[0]
                cube_res[j, yy, xx] = residuals

    frame_desc = scwave(cube_res, scale_list, full_output=False, inverse=True,
                        y_in=y_in, x_in=x_in, imlib=imlib,
                        interpolation=interpolation, collapse=collapse)
    return frame_desc


def _pca_adi_ann(cube, angle_list, radius_int=0, fwhm=4, asize=2, n_segments=1,
                 delta_rot=1, ncomp=1, svd_mode='lapack', nproc=None,
                 min_frames_lib=2, max_frames_lib=200, tol=1e-1, scaling=None,
                 imlib='opencv', interpolation='lanczos4', collapse='median',
                 full_output=False, verbose=1):
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
        msg = '# annuli = {}, Ann width = {}, FWHM = {:.3f}'
        print(msg.format(n_annuli, asize, fwhm))
        print('PCA per annulus (or annular sectors):')

    if nproc is None:   # Hyper-threading "duplicates" the cores -> cpu_count/2
        nproc = cpu_count() // 2

    # The annuli are built, and the corresponding PA thresholds for frame
    # rejection are calculated (at the center of the annulus)
    cube_out = np.zeros_like(array)
    for ann in range(n_annuli):
        if isinstance(ncomp, list) or isinstance(ncomp, np.ndarray):
            if len(ncomp) == n_annuli:
                ncompann = ncomp[ann]
            else:
                msge = 'If ncomp is a list, it must match the number of annuli'
                raise TypeError(msge)
        else:
            ncompann = ncomp

        n_segments_ann = n_segments[ann]
        res_ann_par = _define_annuli(angle_list, ann, n_annuli, fwhm,
                                     radius_int, asize, delta_rot,
                                     n_segments_ann, verbose)
        pa_thr, inner_radius, ann_center = res_ann_par
        indices = get_annulus_segments(array[0], inner_radius, asize,
                                       n_segments_ann)
        # Library matrix is created for each segment and scaled if needed
        for j in range(n_segments_ann):
            yy = indices[j][0]
            xx = indices[j][1]
            matrix_segm = array[:, yy, xx]  # shape [nframes x npx_segment]
            matrix_segm = matrix_scaling(matrix_segm, scaling)

            res = pool_map(nproc, do_pca_patch, matrix_segm, fixed(range(n)),
                           angle_list, fwhm, pa_thr, ann_center, svd_mode,
                           ncompann, min_frames_lib, max_frames_lib, tol,
                           verbose=False)

            res = np.array(res)
            residuals = np.array(res[:, 0])
            ncomps = res[:, 1]
            nfrslib = res[:, 2]
            for fr in range(n):
                cube_out[fr][yy, xx] = residuals[fr]

            # number of frames in library printed for each annular quadrant
            # number of PCs printed for each annular quadrant
            if verbose == 2:
                descriptive_stats(nfrslib, verbose=verbose, label='\tLIBsize: ')
                descriptive_stats(ncomps, verbose=verbose, label='\tNum PCs: ')

        if verbose == 2:
            print('Done PCA with {} for current annulus'.format(svd_mode))
        if verbose:
            timing(start_time)

    # Cube is derotated according to the parallactic angle and collapsed
    cube_der = cube_derotate(cube_out, angle_list, imlib=imlib,
                             interpolation=interpolation)
    frame = cube_collapse(cube_der, mode=collapse)
    if verbose:
        print('Done derotating and combining.')
        timing(start_time)
    if full_output:
        return cube_out, cube_der, frame
    else:
        return frame


def do_pca_patch(matrix, frame, angle_list, fwhm, pa_threshold, ann_center,
                 svd_mode, ncomp, min_frames_lib, max_frames_lib, tol):
    """ Does the SVD/PCA for each frame patch (small matrix). For each frame we
    find the frames to be rejected depending on the amount of rotation. The
    library is also truncated on the other end (frames too far or which have
    rotated more) which are more decorrelated to keep the computational cost
    lower. This truncation is done on the annuli after 10*FWHM and the goal is
    to keep min(num_frames/2, 200) in the library.
    """
    if pa_threshold != 0:
        if ann_center > fwhm*10:    # TODO: 10*FWHM optimal? new parameter?
            indices_left = _find_indices_adi(angle_list, frame, pa_threshold,
                                             truncate=True,
                                             max_frames=max_frames_lib)
        else:
            indices_left = _find_indices_adi(angle_list, frame, pa_threshold,
                                             truncate=False)

        data_ref = matrix[indices_left]

        if data_ref.shape[0] <= min_frames_lib:
            msg = 'Too few frames left in the PCA library. '
            msg += 'Try decreasing either delta_rot or min_frames_lib.'
            raise RuntimeError(msg)
    else:
        data_ref = matrix

    data = data_ref
    curr_frame = matrix[frame]                     # current frame

    V = get_eigenvectors(ncomp, data, svd_mode, noise_error=tol, debug=False)

    transformed = np.dot(curr_frame, V.T)
    reconstructed = np.dot(transformed.T, V)
    residuals = curr_frame - reconstructed
    return residuals, V.shape[0], data_ref.shape[0]




