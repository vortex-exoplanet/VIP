#! /usr/bin/env python

"""
Module with helping functions for PCA.
"""

__author__ = 'Carlos Alberto Gomez Gonzalez'
__all__ = ['pca_annulus',
           'pca_grid']

import numpy as np
from sklearn.decomposition import IncrementalPCA
from pandas import DataFrame
from skimage.draw import disk
from matplotlib import pyplot as plt
from ..fits import open_fits
from ..preproc import cube_rescaling_wavelengths as scwave
from .svd import svd_wrapper
from ..preproc import cube_derotate, cube_collapse, check_pa_vector
from ..config import timing, time_ini, check_array, get_available_memory
from ..config.utils_conf import vip_figsize, vip_figdpi
from ..var import frame_center, dist, prepare_matrix, reshape_matrix, get_circle


def pca_grid(cube, angle_list, fwhm=None, range_pcs=None, source_xy=None,
             cube_ref=None, mode='fullfr', annulus_width=20, svd_mode='lapack',
             scaling=None, mask_center_px=None, fmerit='mean', 
             collapse='median', ifs_collapse_range='all', verbose=True, 
             full_output=False, debug=False, plot=True, save_plot=None, 
             start_time=None, scale_list=None, initial_4dshape=None, 
             weights=None, **rot_options):
    """
    Compute a grid, depending on ``range_pcs``, of residual PCA frames out of a
    3d ADI cube (or a reference cube). If ``source_xy`` is provided, the number
    of principal components are optimized by measuring the S/N at this location
    on the frame (ADI, RDI). The metric used, set by ``fmerit``, could be the
    given pixel's S/N, the maximum S/N in a FWHM circular aperture centered on
    the given coordinates or the mean S/N in the same circular aperture.

    Parameters
    ----------
    cube : numpy ndarray, 3d
        Input cube.
    angle_list : numpy ndarray, 1d
        Corresponding parallactic angle for each frame.
    fwhm : None or float, optional
        Size of the FWHM in pixels, used for computing S/Ns when ``source_xy``
        is passed.
    range_pcs : None or tuple, optional
        The interval of PCs to be tried. If a ``range_pcs`` is entered as
        ``[PC_INI, PC_MAX]`` a sequential grid will be evaluated between
        ``PC_INI`` and ``PC_MAX`` with step of 1. If a ``range_pcs`` is entered
        as ``[PC_INI, PC_MAX, STEP]`` a grid will be evaluated between
        ``PC_INI`` and ``PC_MAX`` with the given ``STEP``. If ``range_pcs`` is
        None, ``PC_INI=1``, ``PC_MAX=n_frames-1`` and ``STEP=1``, which will
        result in longer running time.
    source_xy : None or tuple of floats
        X and Y coordinates of the pixel where the source is located and whose
        SNR is going to be maximized.
    cube_ref : numpy ndarray, 3d, optional
        Reference library cube. For Reference Star Differential Imaging.
    mode : {'fullfr', 'annular'}, optional
        Mode for PCA processing (full-frame or just in an annulus). There is a
        catch: the optimal number of PCs in full-frame may not coincide with the
        one in annular mode. This is due to the fact that the annulus matrix is
        smaller (less noisy, probably not containing the central star) and also
        its intrinsic rank (smaller that in the full frame case).
    annulus_width : float, optional
        Width in pixels of the annulus in the case of the "annular" mode.
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
        to unit variance.

        ``spat-standard``: spatial mean centering plus scaling pixel values
        to unit variance.

    mask_center_px : None or int, optional
        If None, no masking is done. If an integer > 1 then this value is the
        radius of the circular mask.
    fmerit : {'px', 'max', 'mean'}
        The function of merit to be maximized. 'px' is *source_xy* pixel's SNR,
        'max' the maximum SNR in a FWHM circular aperture centered on
        ``source_xy`` and 'mean' is the mean SNR in the same circular aperture.
    collapse : {'median', 'mean', 'sum', 'trimmean'}, str optional
        Sets the way of collapsing the frames for producing a final image.
    ifs_collapse_range: str 'all' or tuple of 2 int
        If a tuple, it should contain the first and last channels where the mSDI 
        residual channels will be collapsed (by default collapses all channels).
    verbose : bool, optional
        If True prints intermediate info and timing.
    full_output : bool, optional
        If True it returns the optimal number of PCs, the final PCA frame for
        the optimal PCs and a cube with all the final frames for each number
        of PC that was tried.
    debug : bool, bool optional
        Whether to print debug information or not.
    plot : bool, optional
        Whether to plot the SNR and flux as functions of PCs and final PCA
        frame or not.
    save_plot: string
        If provided, the pc optimization plot will be saved to that path.
    start_time : None or datetime.datetime, optional
        Used when embedding this function in the main ``pca`` function. The
        object datetime.datetime is the global starting time. If None, it
        initiates its own counter.
    scale_list : None or numpy ndarray, optional
        Scaling factors in case of IFS data (ADI+mSDI cube). They will be used
        to descale the spectral cubes and obtain the right residual frames,
        assuming ``cube`` is a 4d ADI+mSDI cube turned into 3d.
    initial_4dshape : None or tuple, optional
        Shape of the initial ADI+mSDI cube.
    weights: 1d numpy array or list, optional
        Weights to be applied for a weighted mean. Need to be provided if 
        collapse mode is 'wmean'.
    rot_options: dictionary, optional
        Dictionary with optional keyword values for "nproc", "imlib", 
        "interpolation", "border_mode", "mask_val",  "edge_blend", 
        "interp_zeros", "ker" (see documentation of 
        ``vip_hci.preproc.frame_rotate``)
        
    Returns
    -------
    cubeout : numpy ndarray
        3D array with the residuals frames.
    pclist : list
        [full_output=True, source_xy is None] The list of PCs at which the
        residual frames are computed.
    finalfr : numpy ndarray
        [source_xy is not None] Residual frame with the highest S/N for
        ``source_xy``.
    df : pandas Dataframe
        [source_xy is not None]  Dataframe with the pcs, measured fluxed and
        S/Ns for ``source_xy``.
    opt_npc : int
        [source_xy is not None] Optimal number of PCs for ``source_xy``.

    """
    from ..metrics import snr, frame_report

    def truncate_svd_get_finframe(matrix, angle_list, ncomp, V):
        """ Projection, subtraction, derotation plus combination in one frame.
        Only for full-frame"""
        transformed = np.dot(V[:ncomp], matrix.T)
        reconstructed = np.dot(transformed.T, V[:ncomp])
        residuals = matrix - reconstructed
        frsize = int(np.sqrt(matrix.shape[1]))  # only for square frames
        residuals_res = reshape_matrix(residuals, frsize, frsize)

        # For the case of ADI+mSDI data (assuming crop_ifs=True), we descale
        # and collapse each spectral residuals cube
        if scale_list is not None and initial_4dshape is not None:
            print("Descaling the spectral channels and obtaining a final frame")
            z, n_adi, y_in, x_in = initial_4dshape
            residuals_reshaped = np.zeros((n_adi, y_in, y_in))

            if ifs_collapse_range == 'all':
                idx_ini = 0
                idx_fin = z
            else:
                idx_ini = ifs_collapse_range[0]
                idx_fin = ifs_collapse_range[1]

            for i in range(n_adi):
                frame_i = scwave(residuals_res[i*z+idx_ini:i*z+idx_fin, :, :],
                                 scale_list[idx_ini:idx_fin], full_output=False, 
                                 inverse=True, y_in=y_in, x_in=x_in, 
                                 collapse=collapse)
                residuals_reshaped[i] = frame_i
        else:
            residuals_reshaped = residuals_res

        residuals_res_der = cube_derotate(residuals_reshaped, angle_list, 
                                          **rot_options)
        res_frame = cube_collapse(residuals_res_der, mode=collapse, w=weights)
        return res_frame

    def truncate_svd_get_finframe_ann(matrix, indices, angle_list, ncomp, V):
        """ Projection, subtraction, derotation plus combination in one frame.
        Only for annular mode"""
        transformed = np.dot(V[:ncomp], matrix.T)
        reconstructed = np.dot(transformed.T, V[:ncomp])
        residuals_ann = matrix - reconstructed
        residuals_res = np.zeros_like(cube)
        residuals_res[:, indices[0], indices[1]] = residuals_ann
        residuals_res_der = cube_derotate(residuals_res, angle_list, 
                                          **rot_options)
        res_frame = cube_collapse(residuals_res_der, mode=collapse, w=weights)
        return res_frame

    def get_snr(frame, y, x, fwhm, fmerit):
        """
        """
        if fmerit == 'max':
            yy, xx = disk((y, x), fwhm / 2.)
            res = [snr(frame, (x_, y_), fwhm, plot=False, verbose=False,
                       full_output=True)
                   for y_, x_ in zip(yy, xx)]
            snr_pixels = np.array(res, dtype=object)[:, -1]
            fluxes = np.array(res, dtype=object)[:, 2]
            argm = np.argmax(snr_pixels)
            # integrated fluxes for the max snr
            return np.max(snr_pixels), fluxes[argm]

        elif fmerit == 'px':
            res = snr(frame, (x, y), fwhm, plot=False, verbose=False,
                      full_output=True)
            snrpx = res[-1]
            fluxpx = np.array(res, dtype=object)[2]
            # integrated fluxes for the given px
            return snrpx, fluxpx

        elif fmerit == 'mean':
            yy, xx = disk((y, x), fwhm / 2.)
            res = [snr(frame, (x_, y_), fwhm, plot=False, verbose=False,
                       full_output=True) for y_, x_
                   in zip(yy, xx)]
            snr_pixels = np.array(res, dtype=object)[:, -1]
            fluxes = np.array(res, dtype=object)[:, 2]
            # mean of the integrated fluxes (shifting the aperture)
            return np.mean(snr_pixels), np.mean(fluxes)

    # --------------------------------------------------------------------------
    check_array(cube, dim=3, msg='cube')

    if start_time is None:
        start_time = time_ini(verbose)
    n = cube.shape[0]

    if source_xy is not None:
        x, y = source_xy
    else:
        x = None
        y = None

    if range_pcs is None:
        pcmin = 1
        pcmax = n - 1
        step = 1
    elif len(range_pcs) == 2:
        pcmin, pcmax = range_pcs
        pcmax = min(pcmax, n)
        step = 1
    elif len(range_pcs) == 3:
        pcmin, pcmax, step = range_pcs
        pcmax = min(pcmax, n)
    else:
        raise TypeError('`range_pcs` must be None or a tuple, corresponding to '
                        '(PC_INI, PC_MAX) or (PC_INI, PC_MAX, STEP)')

    # Getting `pcmax` principal components once
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
        inrad = int(ann_radius - annulus_width / 2.)
        outrad = int(ann_radius + annulus_width / 2.)
        matrix, annind = prepare_matrix(cube, scaling, None, mode='annular',
                                        inner_radius=inrad, outer_radius=outrad,
                                        verbose=False)
        if cube_ref is not None:
            ref_lib, _ = prepare_matrix(cube_ref, scaling, mask_center_px,
                                        'annular', inner_radius=inrad,
                                        outer_radius=outrad, verbose=False)
        else:
            ref_lib = matrix

    else:
        raise RuntimeError('Wrong mode. Choose either fullfr or annular')

    V = svd_wrapper(ref_lib, svd_mode, pcmax, verbose)

    if verbose:
        timing(start_time)

    snrlist = []
    pclist = []
    fluxlist = []
    frlist = []
    counter = 0
    for pc in range(pcmin, pcmax + 1, step):
        if mode == 'fullfr':
            frame = truncate_svd_get_finframe(matrix, angle_list, pc, V)
        elif mode == 'annular':
            frame = truncate_svd_get_finframe_ann(matrix, annind,
                                                  angle_list, pc, V)
        else:
            raise RuntimeError('Wrong mode. Choose either full or annular')

        if x is not None and y is not None and fwhm is not None:
            snr_value, flux = get_snr(frame, y, x, fwhm, fmerit)
            if np.isnan(snr_value):
                snr_value = 0
            snrlist.append(snr_value)
            fluxlist.append(flux)

        counter += 1
        pclist.append(pc)
        frlist.append(frame)

    cubeout = np.array((frlist))

    # measuring the S/Ns for the given position
    if x is not None and y is not None and fwhm is not None:
        argmax = np.argmax(snrlist)
        opt_npc = pclist[argmax]
        df = DataFrame({'PCs': pclist, 'S/Ns': snrlist, 'fluxes': fluxlist})
        if debug:
            print(df, '\n')

        if verbose:
            print('Number of steps', len(pclist))
            msg = 'Optimal number of PCs = {}, for S/N={:.3f}'
            print(msg.format(opt_npc, snrlist[argmax]))

            # Plot of SNR and flux as function of PCs
            if plot:
                plt.figure(figsize=vip_figsize, dpi=vip_figdpi)
                ax1 = plt.subplot(211)
                ax1.plot(pclist, snrlist, '-', alpha=0.5, color='C0')
                ax1.plot(pclist, snrlist, 'o', alpha=0.5, color='C0')
                ax1.set_xlim(np.array(pclist).min(), np.array(pclist).max())
                ax1.set_ylim(min(snrlist), np.array(snrlist).max() + 1)
                ax1.set_ylabel('S/N')
                ax1.minorticks_on()
                ax1.grid('on', 'major', linestyle='solid', alpha=0.4)
                ax1.set_title('Optimal # PCs: {}'.format(opt_npc))

                ax2 = plt.subplot(212)
                ax2.plot(pclist, fluxlist, '-', alpha=0.5, color='C1')
                ax2.plot(pclist, fluxlist, 'o', alpha=0.5, color='C1')
                ax2.set_xlim(np.array(pclist).min(), np.array(pclist).max())
                ax2.set_ylim(min(fluxlist), np.array(fluxlist).max() + 1)
                ax2.set_xlabel('Principal components')
                ax2.set_ylabel('Flux in FWHM ap. [ADUs]')
                ax2.minorticks_on()
                ax2.grid('on', 'major', linestyle='solid', alpha=0.4)

            if save_plot is not None:
                plt.savefig(save_plot, dpi=100, bbox_inches='tight')

        finalfr = cubeout[argmax]
        _ = frame_report(finalfr, fwhm, (x, y), verbose=verbose)

        return cubeout, finalfr, df, opt_npc

    else:
        if verbose:
            print('Computed residual frames for PCs interval: '
                  '{}'.format(range_pcs))
            print('Number of steps', len(pclist))

    if verbose:
        timing(start_time)

    if full_output:
        return cubeout, pclist
    else:
        return cubeout


def pca_incremental(cube, angle_list, batch=0.25, ncomp=1, collapse='median', 
                    verbose=True, full_output=False, return_residuals=False, 
                    start_time=None, weights=None, **rot_options):
    """ Computes the full-frame PCA-ADI algorithm in batches, for processing
    fits files larger than the available system memory. It uses the incremental
    PCA algorithm from Sklearn. There is no ``scaling`` parameter as in other
    PCA algorithms in ``VIP``, but by default this implementation returns a
    temporally mean-centered frame ("temp-mean").

    Parameters
    ----------
    cube : str or numpy ndarray
        Input cube as numpy array or string with the path to the fits file to be
        opened in memmap mode.
    angle_list : str or numpy ndarray
        Corresponding parallactic angle for each frame.
    batch : int or float, optional
        When int it corresponds to the number of frames in each batch. If a
        float (0, 1] is passed then it is the size of the batch is computed wrt
        the available memory in the system.
    ncomp : int, optional
        How many PCs are used as a lower-dimensional subspace to project the
        target frames.
    collapse : {'median', 'mean', 'sum', 'trimmean'}, str optional
        Sets the way of collapsing the frames for producing a final image.
    rot_options: dictionary, optional
        Dictionary with optional keyword values for "nproc", "imlib", 
        "interpolation, "border_mode", "mask_val",  "edge_blend", 
        "interp_zeros", "ker" (see documentation of 
        ``vip_hci.preproc.frame_rotate``)
    verbose : {True, False}, bool optional
        If True prints intermediate info and timing.
    full_output : boolean, optional
        Whether to return the final median combined image only or with other
        intermediate arrays.
    return_residuals : bool, optional
        If True, only the cube of residuals is returned (before de-rotating).
    start_time : None or datetime.datetime, optional
        Used when embedding this function in the main ``pca`` function. The
        object datetime.datetime is the global starting time. If None, it
        initiates its own counter.
    weights: 1d numpy array or list, optional
        Weights to be applied for a weighted mean. Need to be provided if 
        collapse mode is 'wmean'.
        
    Returns
    -------
    frame : numpy ndarray
        [return_residuals=False] Final frame (2d array).
    ipca : scikit-learn model
        [full_output=True, return_residuals=False] The incremental PCA model of
        scikit-learn.
    pcs : numpy ndarray
        [full_output=True, return_residuals=False] Principal components reshaped
        into images.
    medians : numpy ndarray
        [full_output=True, return_residuals=False] The median of the derotated
        residuals for each batch.
    cube_residuals : numpy ndarray
        [return_residuals=True] Cube of residuals.


    """
    if start_time is None:
        start_time = time_ini(verbose)
        verbose_memcheck = True
    else:
        verbose_memcheck = False

    # checking cube and angle_list data types
    if not isinstance(cube, (np.ndarray, str)):
        raise TypeError('`cube` must be a str (full path on disk) or a numpy '
                        'array')
    if not isinstance(angle_list, (np.ndarray, str)):
        raise TypeError('`angle_list` must be a str (full path on disk) or a '
                        'numpy array')

    # opening data
    if isinstance(cube, str):
        # assuming the first HDULIST contains the datacube
        hdulist = open_fits(cube, n=0, return_memmap=True)
        cube = hdulist.data
    if not cube.ndim > 2:
        raise TypeError('Input array is not a 3d array')
    n_frames, y, x = cube.shape

    # checking angles length and ncomp
    if isinstance(angle_list, str):
        angle_list = open_fits(angle_list)
    angle_list = check_pa_vector(angle_list)
    if not n_frames == angle_list.shape[0] and not return_residuals:
        raise TypeError('`angle_list` vector has wrong length. It must be the '
                        'same as the number of frames in the cube')
    if not isinstance(ncomp, (int, float)):
        raise TypeError("`ncomp` must be an int or a float in the ADI case")
    if ncomp > n_frames:
        ncomp = min(ncomp, n_frames)
        msg = 'Number of PCs too high (max PCs={}), using {} PCs instead.'
        print(msg.format(n_frames, ncomp))

    # checking memory and determining batch size
    cube_size = cube.nbytes
    aval_mem = get_available_memory(verbose_memcheck)
    if isinstance(batch, int):      # the batch size in n_fr
        batch_size = batch
    elif isinstance(batch, float):  # the batch ratio wrt available memory
        if 1 > batch > 0:
            batch_size = min(int(n_frames * (batch * aval_mem) / cube_size),
                             n_frames)
    else:
        raise TypeError("`batch` must be an int or float")

    if verbose:
        msg1 = "Cube size = {:.3f} GB ({} frames)"
        print(msg1.format(cube_size / 1e9, n_frames))
        msg2 = "Batch size = {} frames ({:.3f} GB)\n"
        print(msg2.format(batch_size, cube[:batch_size].nbytes / 1e9))

    n_batches = n_frames // batch_size      # floor/int division
    remaining_frames = n_frames % batch_size
    if remaining_frames > 0:
        n_batches += 1

    # computing the PCA model for each batch
    ipca = IncrementalPCA(n_components=ncomp)

    for i in range(n_batches):
        intini = i * batch_size
        intfin = (i + 1) * batch_size
        batch = cube[intini:min(n_frames, intfin)]
        msg = 'Batch {}/{}\tshape: {}\tsize: {:.1f} MB'
        if verbose:
            print(msg.format(i+1, n_batches, batch.shape, batch.nbytes / 1e6))
        matrix = prepare_matrix(batch, verbose=False)
        ipca.partial_fit(matrix)

    if verbose:
        timing(start_time)

    # getting PCs and the mean in order to center each batch
    V = ipca.components_
    mean = ipca.mean_.reshape(y, x)

    if verbose:
        print('\nReconstructing and obtaining residuals')

    if return_residuals:
        cube_residuals = np.empty((n_frames, y, x))
    else:
        medians = []

    for i in range(n_batches):
        intini = i * batch_size
        intfin = (i + 1) * batch_size
        batch = cube[intini:min(n_frames, intfin)] - mean
        matrix = prepare_matrix(batch, verbose=False)
        reconst = np.dot(np.dot(matrix, V.T), V)
        resid = matrix - reconst
        resid_reshaped = resid.reshape(batch.shape)
        if return_residuals:
            cube_residuals[intini:intfin] = resid_reshaped
        else:
            resid_der = cube_derotate(resid_reshaped, angle_list[intini:intfin],
                                      **rot_options)
            medians.append(cube_collapse(resid_der, mode=collapse,w=weights))

    del matrix
    del batch

    if return_residuals:
        return cube_residuals

    else:
        medians = np.array(medians)
        frame = np.median(medians, axis=0)

        if verbose:
            timing(start_time)

        if full_output:
            pcs = reshape_matrix(V, y, x)
            return frame, ipca, pcs, medians
        else:
            return frame


def pca_annulus(cube, angs, ncomp, annulus_width, r_guess, cube_ref=None,
                svd_mode='lapack', scaling=None, collapse='median', 
                weights=None, **rot_options):
    """
    PCA process the cube only for an annulus of a given width and at a given
    radial distance to the frame center. It returns a PCA processed frame with 
    only non-zero values at the positions of the annulus.
    
    Parameters
    ----------
    cube : numpy ndarray
        The cube of fits images expressed as a numpy.array.
    angs : numpy ndarray
        The parallactic angles expressed as a numpy.array.
    ncomp : int
        The number of principal component.
    annulus_width : float
        The annulus width in pixel on which the PCA is performed.
    r_guess : float
        Radius of the annulus in pixels.
    cube_ref : numpy ndarray, 3d, optional
        Reference library cube. For Reference Star Differential Imaging.
    svd_mode : {'lapack', 'randsvd', 'eigen', 'arpack'}, str optional
        Switch for different ways of computing the SVD and selected PCs.
    scaling : {None, 'temp-mean', 'spat-mean', 'temp-standard', 'spat-standard'}
        With None, no scaling is performed on the input data before SVD. With
        "temp-mean" then temporal px-wise mean subtraction is done, with
        "spat-mean" then the spatial mean is subtracted, with "temp-standard"
        temporal mean centering plus scaling to unit variance is done and with
        "spat-standard" spatial mean centering plus scaling to unit variance is
        performed.
    collapse : {'median', 'mean', 'sum', 'trimmean', None}, str or None, optional
        Sets the way of collapsing the frames for producing a final image. If
        None then the cube of residuals is returned.
    weights: 1d numpy array or list, optional
        Weights to be applied for a weighted mean. Need to be provided if 
        collapse mode is 'wmean'.
    rot_options: dictionary, optional
        Dictionary with optional keyword values for "nproc", "imlib", 
        "interpolation, "border_mode", "mask_val",  "edge_blend", 
        "interp_zeros", "ker" (see documentation of 
        ``vip_hci.preproc.frame_rotate``)
        
    Returns
    -------
    Depending on ``collapse`` parameter a final collapsed frame or the cube of
    residuals is returned.
    """
    
    inrad = int(r_guess - annulus_width / 2.)
    outrad = int(r_guess + annulus_width / 2.)
    data, ind = prepare_matrix(cube, scaling, mode='annular', verbose=False,
                               inner_radius=inrad, outer_radius=outrad)
    yy, xx = ind

    if cube_ref is not None:
        data_svd, _ = prepare_matrix(cube_ref, scaling, mode='annular',
                                     verbose=False, inner_radius=inrad,
                                     outer_radius=outrad)
    else:
        data_svd = data
        
    V = svd_wrapper(data_svd, svd_mode, ncomp, verbose=False)
        
    transformed = np.dot(data, V.T)
    reconstructed = np.dot(transformed, V)                           
    residuals = data - reconstructed
    cube_zeros = np.zeros_like(cube)
    cube_zeros[:, yy, xx] = residuals

    if angs is not None:
        cube_res_der = cube_derotate(cube_zeros, angs, **rot_options)
        if collapse is not None:
            pca_frame = cube_collapse(cube_res_der, mode=collapse, w=weights)
            return pca_frame
        else:
            return cube_res_der

    else:
        if collapse is not None:
            pca_frame = cube_collapse(cube_zeros, mode=collapse, w=weights)
            return pca_frame
        else:
            return cube_zeros
            
            
def _compute_stim_map(cube_der):
    """
    Computes the STIM detection map. 
    
    Note: this is a duplicate of the STIM map routine in the metrics module 
    that is necessary to avoid circular imports (used in iterative PCA function)

    Parameters
    ----------
    cube_der : 3d numpy ndarray
        Input de-rotated cube, e.g. output 'residuals_cube_' from
        ``vip_hci.pca.pca``.

    Returns
    -------
    detection_map : 2d ndarray
        STIM detection map.
    """
    t, n, _ = cube_der.shape
    mu = np.mean(cube_der, axis=0)
    sigma = np.sqrt(np.var(cube_der, axis=0))
    detection_map = np.divide(mu, sigma, out=np.zeros_like(mu),
                              where=sigma != 0)
    return get_circle(detection_map, int(np.round(n/2.)))


def _compute_inverse_stim_map(cube, angle_list, **rot_options):
    """
    Computes the inverse STIM detection map, i.e. obtained with opposite 
    derotation angles.
    
    Note: this is a duplicate of the STIM map routine in the metrics module 
    that is necessary to avoid circular imports (used in iterative PCA function)

    Parameters
    ----------
    cube : 3d numpy ndarray
        Non de-rotated residuals from reduction algorithm, eg. output residuals
        from ``vip_hci.pca.pca``.
    angle_list : numpy ndarray, 1d
        Corresponding parallactic angle for each frame.   
    rot_options: dictionary, optional
        Dictionary with optional keyword values for "nproc", "imlib", 
        "interpolation, "border_mode", "mask_val",  "edge_blend", 
        "interp_zeros", "ker" (see documentation of 
        ``vip_hci.preproc.frame_rotate``)
        
    Returns
    -------
    inverse_stim_map : 2d ndarray
        Inverse STIM detection map.
    """
    t, n, _ = cube.shape
    cube_inv_der = cube_derotate(cube, -angle_list, **rot_options)
    inverse_stim_map = _compute_stim_map(cube_inv_der)
    return inverse_stim_map