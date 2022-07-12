#! /usr/bin/env python

"""
Module with local/smart PCA (annulus or patch-wise in a multi-processing
fashion) model PSF subtraction for ADI, ADI+SDI (IFS) and ADI+RDI datasets.

.. [PAI18]
   | Pairet et al. 2018
   | **Reference-less algorithm for circumstellar disks imaging**
   | *arXiv e-print:1812.01333*
   | `https://arxiv.org/abs/1812.01333
     <https://arxiv.org/abs/1812.01333>`_
     
     
.. [PAI21]
   | Pairet et al. 2021
   | **MAYONNAISE: a morphological components analysis pipeline for circumstellar discs and exoplanets imaging in the near-infrared**
   | *MNRAS, Volume 503, Issue 3, pp. 3724-3742*
   | `https://arxiv.org/abs/2008.05170
     <https://arxiv.org/abs/2008.05170>`_
   
"""

__author__ = 'Valentin Christiaens'
__all__ = ['pca_annular_it',
           'feves',
           'feves_auto']

from multiprocessing import cpu_count
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from ..config.utils_conf import pool_map, iterable
from .pca_fullfr import _interp2d_rad
from ..preproc import cube_derotate, cube_shift
from ..preproc.derotation import _define_annuli
from ..psfsub import nmf_annular, pca_annular
from ..metrics import throughput, stim_map, mask_sources, inverse_stim_map
from ..var import (get_annulus_segments, matrix_scaling, mask_circle,
                   frame_filter_lowpass, frame_center, dist_matrix)
# DELETE
from ..fits import write_fits

def pca_annular_it(cube, angle_list, cube_ref=None, ncomp=1, n_it=10, thr=1., 
                   thr_per_ann=True, thru_corr=False, n_neigh=0, strategy='ADI', 
                   psfn=None, n_br=6, radius_int=0, fwhm=4, asize=4, 
                   n_segments=1, delta_rot=(0.1, 1), svd_mode='lapack', nproc=1, 
                   min_frames_lib=2, max_frames_lib=200, tol=1e-1, scaling=None, 
                   imlib='vip-fft', interpolation='lanczos4', collapse='median', 
                   full_output=False, verbose=True, weights=None,
                   interp_order=2, rtol=1e-2, atol=1, **rot_options):
    """
    Iterative version of annular PCA.

    The algorithm finds significant disc or planet signal in the final PCA
    image, then subtracts it from the input cube (after rotation) just before
    (and only for) projection onto the principal components. This is repeated
    n_it times, which progressively reduces geometric biases in the image
    (e.g. negative side lobes for ADI, radial negative signatures for SDI).
    This is similar to the algorithm presented in Pairet et al. (2020).

    The same parameters as pca_annular() can be provided. There are two
    additional parameters related to the iterative algorithm: the number of
    iterations (n_it) and the threshold (thr) used for the identification of
    significant signals.

    Parameters
    ----------
    cube : numpy ndarray, 3d or 4d
        Input cube.
    angle_list : numpy ndarray, 1d
        Corresponding parallactic angle for each frame.
    cube_ref : numpy ndarray, 3d, optional
        Reference library cube. For Reference Star Differential Imaging.
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
    n_it: int, opt
        Number of iterations for the iterative PCA.
        n_neigh: int, opt
        If larger than 0, number of neighbouring pixels to the ones included
        as significant to be included in the mask. A larger than zero value can
        make the convergence faster but also bears the risk of including 
        non-significant signals.
    thr: float, opt
        Threshold used to identify significant signals in the final PCA image,
        iterartively. This threshold corresponds to the minimum intensity in
        the STIM map computed from PCA residuals (Pairet et al. 2019), as
        expressed in units of maximum intensity obtained in the inverse STIM
        map (i.e. obtained from using opposite derotation angles).
    thr_per_ann: bool, opt
        Whether the threshold should be calculated annulus per annulus
        (recommended).
    thru_corr: str, opt, {None, 'psf', 'map'}
        Whether to correct the significant signals by the algorithmic 
        throughput before subtraction to the original cube at the next
        iteration. If None, this is not performed. If 'psf', throughput is 
        estimated in a classical way with injected psfs. If 'map', the map of
        identified significant signals is used directly for an estimate of the
        2D throughput.
    convolve: bool, opt
        Whether to convolve the map of significant signals - this may enable to
        avoid sharp edges in final image.
    psfn: 2d numpy array, opt
        If either thru_corr or convolve is set to True, psfn should be a 
        normalised and centered unsaturated psf.
    n_br: int, opt
        Number of branches on which the fake planets are injected to compute 
        the throughput.
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
        Maximum number of frames in the PCA reference library for annuli beyond
        10*FWHM. The more distant/decorrelated frames are removed from the
        library.
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
        If True prints to stdout intermediate info.
    weights: 1d numpy array or list, optional
        Weights to be applied for a weighted mean. Need to be provided if
        collapse mode is 'wmean'.
    interp_order: int, opt
        Interpolation order for throughput vector. Only used if thru_corr set 
        to True.
    rtol: float, optional
        Relative tolerance threshold element-wise in the significant signal 
        image compared to the same image obtained either 1 or 2 iterations
        before, to consider convergence [more details in np.allclose].
    atol: float, optional
        Absolute tolerance threshold element-wise in the significant signal 
        image compared to the same image obtained either 1 or 2 iterations
        before, to consider convergence [more details in np.allclose].

    Returns
    -------
    frame : numpy ndarray
        2D array, median combination of the de-rotated/re-scaled residuals cube.
        Always returned. This is the final image obtained at the last iteration.
    it_cube: numpy ndarray
        [full_output=True] 3D array with final image from each iteration.
    pcs : numpy ndarray
        [full_output=True] Principal components from the last iteration
    residuals_cube : numpy ndarray
        [full_output=True] Residuals cube from the last iteration.
    residuals_cube_ : numpy ndarray
        [full_output=True] Derotated residuals cube from the last iteration.
    """

    def _find_significant_signals(residuals_cube, residuals_cube_, angle_list, 
                                  thr, mask=0, thr_per_ann=True, asize=4):
        # Identifies significant signals with STIM map (outside mask)
        stim = stim_map(residuals_cube_)
        inv_stim = inverse_stim_map(residuals_cube, angle_list)
        if mask is None:
            mask = 0
        if mask:
            #stim = mask_circle(stim, mask)
            inv_stim = mask_circle(inv_stim, mask)
        max_inv = np.amax(inv_stim)
        if max_inv <= 0:
            max_inv = np.amax(np.abs(stim))
        norm_stim = stim/max_inv
        if thr_per_ann:
            _, ny, nx = residuals_cube.shape
            n_ann = int((((ny-1)/2.)-mask)/asize)
            for aa in range(n_ann):
                asec = get_annulus_segments(inv_stim, mask+aa*asize, asize, 
                                            mode='mask')[0]
                max_inv = np.amax(asec)
                if max_inv <= 0:
                    max_inv = np.amax(np.abs(asec))
                norm_stim[np.where(asec!=0)] = stim[np.where(asec!=0)]/max_inv
        good_mask = np.zeros_like(stim)
        good_mask[np.where(norm_stim>thr)] = 1
        return good_mask, norm_stim, stim, inv_stim

    if thru_corr and psfn is None:
        msg = "psf should be provided for throughput correction or convolution"
        raise TypeError(msg)
        
    if strategy == 'ADI':
        ref_cube = None
    elif strategy == 'RDI' or strategy == 'RADI':
        if cube_ref is None:
            raise ValueError("cube_ref should be provided for RDI or RADI")
        ref_cube = cube_ref.copy() 
    else:
        raise ValueError("strategy not recognized: should be ADI, RDI or RADI")

    if nproc is None:
        nproc = int(cpu_count()/2)

    nframes = cube.shape[0]

    if imlib=='vip-fft' and cube.shape[-1]%2: # convert to even-size for FFT-based rotation
        cube = cube_shift(cube, 0.5, 0.5)
        if cube.ndim == 3:
            cube = cube[:, 1:, 1:]
        elif cube.ndim == 4:
            cube = cube[:, :, 1:, 1:]
            
    # 1. Get a first disc estimate, using PCA
    res = pca_annular(cube, angle_list, cube_ref=ref_cube, radius_int=radius_int, 
                      fwhm=fwhm, asize=asize, n_segments=n_segments, 
                      delta_rot=delta_rot, ncomp=ncomp, svd_mode=svd_mode,
                      nproc=nproc, min_frames_lib=min_frames_lib,
                      max_frames_lib=max_frames_lib, tol=tol, scaling=scaling,
                      imlib=imlib, interpolation=interpolation,
                      collapse=collapse, full_output=True, verbose=verbose, 
                      weights=weights, **rot_options)

    # 2. Identify significant signals with STIM map (outside mask)
    frame = res[-1].copy()
    it_cube = np.zeros([n_it+1, frame.shape[0], frame.shape[1]])
    it_cube_nd = np.zeros_like(it_cube)
    thru_2d_cube = np.zeros_like(it_cube)
    stim_cube = np.zeros_like(it_cube)
    stim_ori_cube = np.zeros_like(it_cube)
    stim_inv_cube = np.zeros_like(it_cube)
    it_cube[0] = res[-1].copy()
    it_cube_nd[0] = res[-1].copy()
    residuals_cube = res[0].copy()
    residuals_cube_ = res[1].copy()
    res = _find_significant_signals(residuals_cube, residuals_cube_, angle_list, 
                                    thr, mask=radius_int, 
                                    thr_per_ann=thr_per_ann, asize=asize)
    sig_mask, norm_stim, stim, inv_stim = res
    sig_image = frame.copy()
    sig_images = it_cube.copy()
    sig_image[np.where(1-sig_mask)] = 0
    sig_images[0] = sig_image
    stim_cube[0] = norm_stim
    stim_ori_cube[0] = stim
    stim_inv_cube[0] = inv_stim

    # 3.Loop, updating the reference cube before projection by subtracting the
    #   best disc estimate. This is done by providing sig_cube.
    for it in range(1, n_it+1):
        # rotate image before thresholding
        #write_fits("TMP_sig_image.fits",frame)
        # sometimes there's a cross at the center for PCA no_mask => blur it!
        # frame_mask = mask_circle(frame, radius=mask_tmp, fillwith=np.nan, 
        #                           mode='out')
        # frame_filt = frame_filter_lowpass(frame_mask, mode='gauss', fwhm_size=2, 
        #                                    iterate=False)
        # nonan_loc = np.where(np.isfinite(frame_mask))
        # frame[nonan_loc] = frame_filt[nonan_loc]
        # write_fits("TMP_sig_image_blur.fits",frame)
        # create and rotate sig cube
        sig_cube = np.repeat(frame[np.newaxis, :, :], nframes, axis=0)
        sig_cube = cube_derotate(sig_cube, -angle_list, imlib=imlib, 
                                 nproc=nproc, **rot_options)
        #write_fits("TMP_sig_cube.fits",sig_cube)
        # create and rotate binary mask
        mask_sig = np.zeros_like(sig_image)
        mask_sig[np.where(sig_image>0)] = 1
        
        #write_fits("TMP_derot_msig_image.fits",sig_image)
        sig_mcube = np.repeat(mask_sig[np.newaxis, :, :], nframes, axis=0)
        #write_fits("TMP_rot_msig_cube.fits",sig_mcube)
        sig_mcube = cube_derotate(sig_mcube, -angle_list, imlib='opencv', 
                                  interpolation='bilinear', nproc=nproc)
        sig_cube[np.where(sig_mcube<0.5)] = 0
        sig_cube = mask_circle(sig_cube, radius_int, 0)
        #write_fits("TMP_derot_msig_cube.fits",sig_mcube)
        #write_fits("TMP_derot_sig_cube_thr.fits",sig_cube)
        #pdb.set_trace()
        
        # sig_cube = np.repeat(sig_image[np.newaxis, :, :], nframes, axis=0)
        # sig_cube = cube_derotate(sig_cube, -angle_list, imlib=imlib,
        #                          interpolation=interpolation, nproc=nproc,
        #                          border_mode='constant')
        sig_cube[np.where(sig_cube<0)] = 0
        res_it = pca_annular(cube, angle_list, cube_ref=ref_cube, 
                          radius_int=radius_int, fwhm=fwhm, asize=asize, 
                          n_segments=n_segments,  delta_rot=delta_rot,
                          ncomp=ncomp, svd_mode=svd_mode, nproc=nproc,
                          min_frames_lib=min_frames_lib,
                          max_frames_lib=max_frames_lib, tol=tol,
                          scaling=scaling, imlib=imlib,
                          interpolation=interpolation, collapse=collapse,
                          full_output=True, verbose=verbose,
                          weights=weights, cube_sig=sig_cube, **rot_options)
        it_cube[it] = res_it[-1]
        frame = res_it[-1]
        residuals_cube = res_it[0].copy()
        residuals_cube_ = res_it[1].copy()

        # Scale cube and cube_ref if necessary before inv stim map calculation
        cube_tmp, cube_ref_tmp = _prepare_matrix_ann(cube, ref_cube, 
                                                     scaling, angle_list, fwhm, 
                                                     radius_int, asize, 
                                                     delta_rot, n_segments)

        res_nd = pca_annular(cube_tmp-sig_cube, angle_list, 
                             cube_ref=cube_ref_tmp, radius_int=radius_int, 
                             fwhm=fwhm, asize=asize, n_segments=n_segments, 
                             delta_rot=delta_rot, ncomp=ncomp,
                             svd_mode=svd_mode,  nproc=nproc,
                             min_frames_lib=min_frames_lib,
                             max_frames_lib=max_frames_lib, tol=tol,
                             scaling=scaling, imlib=imlib, 
                             interpolation=interpolation, collapse=collapse, 
                             full_output=True, verbose=False,
                             weights=weights, **rot_options)
        residuals_cube_nd = res_nd[0]
        frame_nd = res_nd[-1]
        res_sig = _find_significant_signals(residuals_cube_nd, residuals_cube_, 
                                            angle_list, thr, mask=radius_int,
                                            thr_per_ann=thr_per_ann, 
                                            asize=asize)
        sig_mask, norm_stim, stim, inv_stim = res_sig
        # expand the mask to consider signals within fwhm/2 of edges
        inv_sig_mask = np.ones_like(sig_mask)
        inv_sig_mask[np.where(sig_mask)] = 0
        if n_neigh > 0:
            inv_sig_mask = mask_sources(inv_sig_mask, n_neigh)
        if radius_int:
            inv_sig_mask = mask_circle(inv_sig_mask, radius_int, fillwith=1)
        sig_image = frame.copy()
        sig_image[np.where(inv_sig_mask)] = 0
        sig_image[np.where(sig_image<0)] = 0
        # correct by algo throughput if requested
        if thru_corr:
            thru_arr, rad_vec = throughput(cube_tmp-sig_cube, -angle_list,
                                           psf_template=psfn, fwhm=fwhm, 
                                           pxscale=1., algo=pca_annular, 
                                           nbranch=n_br, inner_rad=1, 
                                           imlib=imlib, verbose=False, 
                                           radius_int=int(fwhm), asize=asize, 
                                           ncomp=ncomp, n_segments=n_segments, 
                                           fc_snr=5, delta_rot=delta_rot, 
                                           scaling=scaling, 
                                           cube_ref=cube_ref_tmp, tol=tol, 
                                           svd_mode=svd_mode, nproc=nproc, 
                                           min_frames_lib=min_frames_lib, 
                                           max_frames_lib=max_frames_lib,
                                           interpolation=interpolation, 
                                           collapse=collapse)

            if interp_order is not None:
                # interpolating the throughput vector, spline order 2
                rad_samp = np.arange(int(np.ceil(rad_vec[0])), 
                                     int(np.floor(rad_vec[-1])),1)
                n_rad = len(rad_samp)
                thruput_interp = np.ones([n_br,n_rad])
                for bb in range(n_br):
                    f = InterpolatedUnivariateSpline(rad_vec, thru_arr[bb], 
                                                     k=interp_order)
                    thruput_interp[bb] = f(rad_samp)
            else:
                thruput_interp = thru_arr.copy()
                rad_samp = rad_vec.copy()
            #if thru_arr.ndim==1:
            #    thru_arr = thru_arr[np.newaxis,:]
            thru_2d = _interp2d_rad(thruput_interp, rad_samp, 
                                    cube_tmp.shape[-1], theta_0=0)
        # elif thru_corr == 'map':
        #     thru_2d, fr_nofc, fr_fc = throughput_ext(cube_tmp-sig_cube, 
        #                                              angle_list, sig_image, 
        #                                              fwhm=fwhm, pxscale=1., 
        #                                              algo=pca_annular, 
        #                                              imlib=imlib, #nbranch=n_br, 
        #                                              interpolation=interpolation, 
        #                                              #inner_rad=1, 
        #                                              verbose=False, 
        #                                              full_output=True, #fc_snr=5,
        #                                              radius_int=int(fwhm), 
        #                                              asize=asize, 
        #                                              ncomp=ncomp, 
        #                                              scale_list=scale_list,
        #                                              n_segments=n_segments, 
        #                                              delta_rot=delta_rot, 
        #                                              scaling=scaling, 
        #                                              delta_sep=delta_sep, 
        #                                              cube_ref=cube_ref_tmp, 
        #                                              tol=tol, svd_mode=svd_mode, 
        #                                              nproc=nproc, 
        #                                              min_frames_lib=min_frames_lib, 
        #                                              max_frames_lib=max_frames_lib,
        #                                              collapse=collapse, 
        #                                              ifs_collapse_range=ifs_collapse_range)
        #     # smooth the 2d array radially
        #     # interpolating the throughput vector, spline order 2
        #     cy, cx = frame_center(thru_2d)
        #     rad_vec = np.arange(int(fwhm), min(cy,cx), int(fwhm))
        #     rad_samp = np.arange(int(np.ceil(rad_vec[0])), 
        #                          int(np.floor(rad_vec[-1])),1)
        #     n_rad = len(rad_vec)
        #     n_rad_s = len(rad_samp)
        #     thruput_interp = np.ones([n_br,n_rad_s])
        #     thru_arr = np.ones([n_br,n_rad])
        #     thru_2d_tmp = thru_2d.copy()
        #     thru_2d_tmp[np.where(sig_image<=0)] = 0
        #     for rr in range(n_rad):
        #         tmp = get_annulus_segments(thru_2d_tmp, rad_vec[rr], 
        #                                    width=int(fwhm), nsegm=n_br, 
        #                                    theta_init=0, optim_scale_fact=1, 
        #                                    mode="val")
        #         for bb in range(n_br):
        #             tmp_tmp = tmp[bb].copy()
        #             if len(np.where(tmp_tmp>0)[0])>0:
        #                 thru_arr[bb,rr] = np.median(tmp_tmp[np.where(tmp_tmp>0)])
        #     for bb in range(n_br):
        #         f = InterpolatedUnivariateSpline(rad_vec, thru_arr[bb], k=2)
        #         thruput_interp[bb] = f(rad_samp)
        #     #if thru_arr.ndim==1:
        #     #    thru_arr = thru_arr[np.newaxis,:]
        #     thru_2d = _interp2d_rad(thruput_interp, rad_samp, 
        #                             cube_tmp.shape[-1], theta_0=0)            
        else:
            thru_2d=np.ones_like(sig_image)
        # convolve thru2d map with psf if requested
        # if convolve or thru_corr == 'map':
        #     thru_2d_tmp = thru_2d.copy()
        #     thru_2d_tmp[np.where(thru_2d_tmp==1)] = np.nan
        #     mean_thru = np.nanmean(thru_2d_tmp)
        #     thru_2d_tmp = ast_convolve(thru_2d_tmp, psfn)
        #     mean_conv_thru = np.nanmean(thru_2d_tmp)
        #     good_idx = np.where(thru_2d!=1)
        #     thru_2d[good_idx] = thru_2d_tmp[good_idx]
        #     if verbose:
        #         msg = "Mean throughput before/after conv = {:.2f}/{:.2f}"
        #         print(msg.format(mean_thru,mean_conv_thru))
        #     #sig_image = frame_filter_lowpass(sig_image, mode="psf", psf=psfn)
        sig_image/=thru_2d
        #tmp=frame.copy()
        frame[np.where(frame>0)] = frame[np.where(frame>0)]/thru_2d[np.where(frame>0)]
        it_cube[it] = frame
        it_cube_nd[it] = frame_nd
        sig_images[it] = sig_image
        thru_2d_cube[it] = thru_2d
        stim_cube[it] = norm_stim
        stim_ori_cube[it] = stim
        stim_inv_cube[it] = inv_stim

        # check if improvement compared to last iteration
        if it>1:
            cond1 = np.allclose(sig_image, sig_images[it-1], rtol=rtol, 
                                atol=atol)
            cond2 = np.allclose(sig_image, sig_images[it-2], rtol=rtol, 
                                atol=atol)
            if cond1 or cond2:
                if strategy == 'ADI' or strategy == 'RDI':
                    break
                if strategy == 'RADI':
                    # continue to iterate with ADI
                    strategy = 'ADI'
                    ref_cube=None
                    if verbose:
                        msg="After {:.0f} iterations, PCA-RDIann -> PCA-ADIann"
                        print(msg.format(it))

    # mask everything at the end
    if radius_int:
        frame = mask_circle(frame, radius_int)
        it_cube = mask_circle(it_cube, radius_int)
        residuals_cube = mask_circle(residuals_cube, radius_int)
        residuals_cube_ = mask_circle(residuals_cube_, radius_int)
        it_cube_nd = mask_circle(it_cube_nd, radius_int)

    if full_output:
        # if thru_corr == 'map':
        #     thru_2d = np.array([thru_2d,fr_nofc,fr_fc])
        return (frame, it_cube, sig_images, residuals_cube, residuals_cube_,
                thru_2d_cube, it_cube_nd, stim_cube, stim_ori_cube, 
                stim_inv_cube)
    else:
        return frame


def feves(cube, angle_list, cube_ref=None, ncomp=1, algo=pca_annular, n_it=2, 
          fwhm=4, buff=1, thr=1, thr_per_ann=False, n_frac=6, asizes=None, 
          n_segments=None, thru_corr=False, n_neigh=0, strategy='ADI', 
          psfn=None, n_br=6, radius_int=0, delta_rot=(0.1, 1),
          svd_mode='lapack', init_svd='nndsvda', nproc=1, min_frames_lib=2, 
          max_frames_lib=200, tol=1e-1, scaling=None, imlib='vip-fft', 
          interpolation='lanczos4', collapse='median', full_output=False, 
          verbose=True, weights=None, interp_order=2, rtol=1e-2, 
          atol=1, smooth=False, **rot_options):
    """
    Fractionation for Embedded Very young Exoplanet Search algorithm: Iterative
    PCA or NMF applied in progressively more fractionated image sections.

    The algorithm finds significant disc or planet signal in the final PCA
    image, then subtracts it from the input cube (after rotation) just before
    (and only for) projection onto the principal components. This is repeated
    n_it times, which progressively reduces geometric biases in the image
    (e.g. negative side lobes for ADI, radial negative signatures for SDI).
    This is similar to the algorithm presented in Pairet et al. (2020).

    The same parameters as pca_annular() can be provided. There are two
    additional parameters related to the iterative algorithm: the number of
    iterations (n_it) and the threshold (thr) used for the identification of
    significant signals.

    Parameters
    ----------
    cube : numpy ndarray, 3d or 4d
        Input cube.
    angle_list : numpy ndarray, 1d
        Corresponding parallactic angle for each frame.
    cube_ref : numpy ndarray, 3d, optional
        Reference library cube. For Reference Star Differential Imaging.
    ncomp : int or list of int or list of lists of int, optional
        How many PCs are used as a lower-dimensional subspace to project the
        target (sectors of) frames. Depends on the dimensionality of `cube`.
        If int: same ncomp used at all iterations.
        If a list: should have either the length of asizes (if the 
        latter is provided) or a length equal to n_frac (otherwise).
            - if its elements are int: ncomp used for each fractionation level
            - if its elements are lists: additional iterations with different
            ncomp will be done for each fractionation level.
    algo: function, opt, {vip_hci.pca.pca_annular, vip_hci.nmf.nmf_annular}
        Either PCA or NMF in concentric annuli, used iteratively.
    n_it: int, opt
        Number of iterations at each fractionation level.
    n_neigh: int, opt
        If larger than 0, number of neighbouring pixels to the ones included
        as significant to be included in the mask. A larger than zero value can
        make the convergence faster but also bears the risk of including 
        non-significant signals.
    buff: float, opt
        Radial buffer expressed in fwhm, used to smooth final image. The feves 
        algorithm will be repeated int(buff*fwhm), with radial increments of 
        one pixel for the inner mask size. The final results are then the 
        median of each of the int(buff*fwhm) results.
        If buff is set to 0, the feves algorithm is performed only once.
    thr: float, opt
        Threshold used to identify significant signals in the final PCA image,
        iterartively. This threshold corresponds to the minimum intensity in
        the STIM map computed from PCA residuals (Pairet et al. 2019), as
        expressed in units of maximum intensity obtained in the inverse STIM
        map (i.e. obtained from using opposite derotation angles).
    thr_per_ann: bool, opt
        Whether the threshold should be calculated annulus per annulus
        (recommended).
    n_frac: int, opt (between 1 and 7)
        Fractionation level. If asizes and n_elements are not provided 
        manually, the algorithm will consider the following automatic scheme,
        capped to the n_frac first elements:
            asizes =    [16,8,4,2,2,1,1] # in FWHM
            n_segments = [1,1,1,1,3,3,6] # azimuthal bins
        Note: if input cube is small (<35 FWHM in x and y), max n_frac will be
        smaller than 7, and the first entry(-ies) of the list will be skipped 
        until the annulus size fits the frames.
    thru_corr: str, opt, {None, 'psf', 'map'}
        Whether to correct the significant signals by the algorithmic 
        throughput before subtraction to the original cube at the next
        iteration. If None, this is not performed. If 'psf', throughput is 
        estimated in a classical way with injected psfs. If 'map', the map of
        identified significant signals is used directly for an estimate of the
        2D throughput.
    convolve: bool, opt
        Whether to convolve the map of significant signals - this may enable to
        avoid sharp edges in final image.
    psfn: 2d numpy array, opt
        If either thru_corr or convolve is set to True, psfn should be a 
        normalised and centered unsaturated psf.
    n_br: int, opt
        Number of branches on which the fake planets are injected to compute 
        the throughput.
    radius_int : int, optional
        The radius of the innermost annulus. By default is 0, if >0 then the
        central circular region is discarded.
    fwhm : float, optional
        Size of the FHWM in pixels. Default is 4.
    asizes : list, optional
        The size of the annuli at each round of fractionation, expressed in FWHM.
    n_segments : list, optional
        The number of segments for each annulus, at each round of fractionation.
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
    svd_mode : {'lapack', 'arpack', 'eigen', 'randsvd', 'cupy', 'eigencupy',
        'randcupy', 'pytorch', 'eigenpytorch', 'randpytorch'}, str optional
        Switch for the SVD method/library to be used (only used if the algo
        is set to pca_annular)

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
    init_svd: str, optional {'nnsvd','nnsvda','random'}
        Method used to initialize the iterative procedure to find H and W
        (only used if algo is set to nmf_annular).
        'nndsvd': non-negative double SVD recommended for sparseness
        'nndsvda': NNDSVD where zeros are filled with the average of cube; 
        recommended when sparsity is not desired
        'random': random initial non-negative matrix
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
        If True prints to stdout intermediate info.
    weights: 1d numpy array or list, optional
        Weights to be applied for a weighted mean. Need to be provided if
        collapse mode is 'wmean'.
    interp_order: int, opt
        Interpolation order for throughput vector. Only used if thru_corr set 
        to True.
    rtol: float, optional
        Relative tolerance threshold element-wise in the significant signal 
        image compared to the same image obtained either 1 or 2 iterations
        before, to consider convergence [more details in np.allclose].
    atol: float, optional
        Absolute tolerance threshold element-wise in the significant signal 
        image compared to the same image obtained either 1 or 2 iterations
        before, to consider convergence [more details in np.allclose].

    Returns
    -------
    frame : numpy ndarray
        2D array, median combination of the de-rotated/re-scaled residuals cube.
        Always returned. This is the final image obtained at the last iteration.
    it_cube: numpy ndarray
        [full_output=True] 3D array with final image from each iteration.
    pcs : numpy ndarray
        [full_output=True] Principal components from the last iteration
    residuals_cube : 3D or 4D numpy ndarray
        [full_output=True] Residuals cube from the last iteration. In the case
        of an RADI strategy, consists of a 4D array containingthe 3D residual 
        cubes at the end of the RDI and ADI parts, respectively.
    residuals_cube_ : 3D or 4D numpy ndarray
        [full_output=True] Derotated residuals cube from the last iteration. In 
        the case of an RADI strategy, consists of a 4D array containing the 3D 
        residual cubes at the end of the RDI and ADI parts, respectively.
    """

    if thru_corr and psfn is None:
        msg = "psf should be provided for throughput correction or convolution"
        raise TypeError(msg)
    
    if not buff:
        buffer = 1
    else:
        buffer = max(int(buff*fwhm),1)
        
    if nproc is None:
        nproc = int(cpu_count()/2)
        
    # select when to do mp depending on buffer
    if buffer < nproc:
        nproc_tmp = nproc
        nproc = 1
    else:
        nproc_tmp=1
    
    if strategy == 'ADI':
        ref_cube = None
    elif strategy == 'RDI' or strategy == 'RADI' or strategy=='ARDI':
        if cube_ref is None:
            raise ValueError("cube_ref should be provided for RDI, RADI or ARDI")
        ref_cube = cube_ref.copy() 
        if strategy=='RDI':
            delta_rot=cube.shape[1]*3/fwhm # forces doing RDI instead of ARDI
    else:
        raise ValueError("strategy not recognized: not ADI, RDI, RADI or ARDI")

    if asizes is not None and n_segments is not None:
        asz_def = asizes
        nsegm_def = n_segments
        n_frac = len(asizes)
        if n_frac != len(n_segments):
            raise ValueError("asizes and nsegments should have same lengths")
    else:
        msg = "asizes/nsegments not provided => auto-fractionation level {:.0f}"
        print(msg.format(n_frac))
        asz_def = [16,8,4,2,2,1,1]
        nsegm_def = [1,1,1,1,3,3,6]
        
    # adapt lists based on cube xy sizes
    cy, cx = frame_center(cube)
    asz_max = int((cy-radius_int-buffer)/fwhm)
    asizes = [a for a in asz_def if a < asz_max]
    n_segments = [n for i, n in enumerate(nsegm_def) if asz_def[i]<asz_max]
    if n_frac < 1 or n_frac > len(asizes):
        msg="set n_frac to a value between 1 and {:.0f}"
        raise ValueError(msg.format(len(asizes)))
    elif n_frac < len(asizes):
        asizes = asizes[:n_frac]
        n_segments = n_segments[:n_frac]
    
    # convert asizes to FWHM
    asizes = [int(a*fwhm) for a in asizes]
    
    # convert ncomp to a list of lists
    if isinstance(ncomp,int):
        ncomp = [[ncomp]]*len(asizes)
    elif isinstance(ncomp,list):
        if len(ncomp) != len(asizes):
            raise ValueError("ncomp should have same length as asizes")
        elif isinstance(ncomp[0],int):
            ncomp = [[ncomp[i]] for i in range(asizes)]
        elif not isinstance(ncomp[0],list):
            msg = "if ncomp is a list, its elements must be int or list of int"
            raise TypeError(msg)
    else:
        raise TypeError("ncomp can only be integer or list of int")
    
    # define empty lists
    master_cube = []
    master_sig_cube = []
    master_it_cube = []
    master_res_cube = []
    master_res_cube_ = []
    master_it_cube_nd = []
    master_stim_cube = []
    master_stim_ori_cube = []
    master_stim_inv_cube = []
    # 0. start the loop on the buffers (can be done in mp)
    if nproc>1:
        res = pool_map(nproc, _do_one_buff, iterable(range(buffer)), cube, 
                       angle_list, ref_cube, algo, n_it, thr, thr_per_ann, 
                       radius_int, fwhm, asizes, n_segments, n_neigh,
                       thru_corr, psfn, n_br, interp_order, strategy, delta_rot, 
                       ncomp, svd_mode, init_svd, min_frames_lib, 
                       max_frames_lib, tol, scaling, imlib, interpolation, 
                       collapse, atol, rtol, nproc_tmp, True, verbose, weights,
                       smooth, **rot_options)
        for bb in range(buffer):
            master_cube.append(res[bb][0])
            master_it_cube.append(res[bb][1])
            master_sig_cube.append(res[bb][2])
            master_res_cube.append(res[bb][3])
            master_res_cube_.append(res[bb][4])
            master_it_cube_nd.append(res[bb][6])
            master_stim_cube.append(res[bb][7])
            master_stim_ori_cube.append(res[bb][8])
            master_stim_inv_cube.append(res[bb][9])

    else:
        for bb in range(buffer):
            res = _do_one_buff(bb, cube, angle_list, ref_cube, algo, n_it, thr, 
                               thr_per_ann, radius_int, fwhm, asizes, 
                               n_segments, n_neigh, thru_corr, psfn,
                               n_br, interp_order, strategy, delta_rot, ncomp, 
                               svd_mode, init_svd, min_frames_lib,
                               max_frames_lib, tol, scaling, imlib, 
                               interpolation, collapse, atol, rtol, nproc_tmp, 
                               True, verbose, weights, smooth, **rot_options)
            master_cube.append(res[0])
            master_it_cube.append(res[1])
            master_sig_cube.append(res[2])
            master_res_cube.append(res[3])
            master_res_cube_.append(res[4])
            master_it_cube_nd.append(res[6])
            master_stim_cube.append(res[7])
            master_stim_ori_cube.append(res[8])
            master_stim_inv_cube.append(res[9])
    master_cube = np.array(master_cube)
    master_sig_cube = np.array(master_sig_cube)
    master_it_cube = np.array(master_it_cube)
    master_res_cube = np.array(master_res_cube)
    master_res_cube_ = np.array(master_res_cube_)
    master_it_cube_nd = np.array(master_it_cube_nd)
    master_stim_cube = np.array(master_stim_cube)
    master_stim_ori_cube = np.array(master_stim_ori_cube)
    master_stim_inv_cube = np.array(master_stim_inv_cube)
    
    master_frame = np.nanmedian(master_cube, axis=0)
    master_sig_cube = np.nanmedian(master_sig_cube, axis=0)
    master_it_cube = np.nanmedian(master_it_cube, axis=0)
    master_res_cube = np.nanmedian(master_res_cube, axis=0)
    master_res_cube_ = np.nanmedian(master_res_cube_, axis=0)
    master_it_cube_nd = np.nanmedian(master_it_cube_nd, axis=0)
    master_stim_cube = np.nanmedian(master_stim_cube, axis=0)
    master_stim_ori_cube = np.nanmedian(master_stim_ori_cube, axis=0)
    master_stim_inv_cube = np.nanmedian(master_stim_inv_cube, axis=0)

    # mask everything at the end
    if radius_int:
        master_frame = mask_circle(master_frame, radius_int, np.nan)
        master_cube = mask_circle(master_cube, radius_int, np.nan)
        master_it_cube = mask_circle(master_it_cube, radius_int, np.nan)
        master_res_cube = mask_circle(master_res_cube, radius_int, np.nan)
        master_res_cube_ = mask_circle(master_res_cube_, radius_int, np.nan)
        master_it_cube_nd = mask_circle(master_it_cube_nd, radius_int, np.nan)
    
    if full_output:
        return (master_frame, master_it_cube, master_sig_cube, master_res_cube, 
                master_res_cube_, master_it_cube_nd, master_stim_cube, 
                master_stim_ori_cube, master_stim_inv_cube, master_cube)
    else:
        return master_frame  
    

def feves_auto(cube, angle_list, cube_ref=None, ncomp=1, algo=pca_annular, 
               n_it='auto', blur=True, fwhm=4, buff=1, thr=0.5, thr_per_ann=False, 
               n_frac=6, asizes=None, n_segments=None, n_neigh=0, strategy='ADI', 
               n_br=6, radius_int=0, delta_rot=(0.1, 1), svd_mode='lapack', 
               init_svd='nndsvda', nproc=1, min_frames_lib=2, max_frames_lib=200, 
               tol=1e-1, scaling=None, imlib='vip-fft', interpolation='lanczos4', 
               collapse='median', full_output=False, verbose=True, weights=None, 
               interp_order=2, rtol=1e-2, atol=1e-2, transmission=None, 
               **rot_options):
    """
    Fractionation for Embedded Very young Exoplanet Search algorithm: Iterative
    PCA or NMF applied in progressively more fractionated image sections.

    The algorithm finds significant disc or planet signal in the final PCA
    image, then subtracts it from the input cube (after rotation) just before
    (and only for) projection onto the principal components. This is repeated
    n_it times, which progressively reduces geometric biases in the image
    (e.g. negative side lobes for ADI, radial negative signatures for SDI).
    This is similar to the algorithm presented in [PAI18] or GreeDS presented
    in [PAI21], with the exception that FEVES iteratively fractionate the field
    in smaller annular (sub)sections in order to improve the speckle modeling
    and suppression.

    The same parameters as pca_annular() can be provided. There are two
    additional parameters related to the iterative algorithm: the number of
    iterations (n_it) and the threshold (thr) used for the identification of
    significant signals.

    Parameters
    ----------
    cube : numpy ndarray, 3d or 4d
        Input cube.
    angle_list : numpy ndarray, 1d
        Corresponding parallactic angle for each frame.
    cube_ref : numpy ndarray, 3d, optional
        Reference library cube. For Reference Star Differential Imaging.
    ncomp : int or list of int or list of lists of int, or tuple, optional
        How many PCs are used as a lower-dimensional subspace to project the
        target (sectors of) frames. Depends on the dimensionality of `cube`.
        If int: same ncomp used at all iterations.
        If a list: should have either the length of asizes (if the 
        latter is provided) or a length equal to n_frac (otherwise).
        
            - if its elements are int: ncomp used for each fractionation level
            - if its elements are lists: additional iterations with different
            ncomp will be done for each fractionation level;
            - in case of 'RADI strategy, a tuple of 2 elements can be provided,
            the lagorithm will consider the same allowed input as above for 
            'RDI' and 'ADI' respectively.
            
    algo: function, opt, {vip_hci.pca.pca_annular, vip_hci.nmf.nmf_annular}
        Either PCA or NMF in concentric annuli, used iteratively.
    n_it: int, opt
        Number of iterations at each fractionation level.
    buff: float, opt
        Radial buffer expressed in fwhm, used to smooth final image. The feves 
        algorithm will be repeated int(buff*fwhm), with radial increments of 
        one pixel for the inner mask size. The final results are then the 
        median of each of the int(buff*fwhm) results.
        If buff is set to 0, the feves algorithm is performed only once.
    thr: float or tuple of floats, opt
        Threshold used to identify significant signals in the final PCA image,
        iterartively. This threshold corresponds to the minimum intensity in
        the STIM map computed from PCA residuals (Pairet et al. 2019), as
        expressed in units of maximum intensity obtained in the inverse STIM
        map (i.e. obtained from using opposite derotation angles).
    thr_per_ann: bool, opt
        Whether the threshold should be calculated annulus per annulus
        (recommended).
    n_frac: int, opt (between 1 and 7)
        Fractionation level. If asizes and n_elements are not provided 
        manually, the algorithm will consider the following automatic scheme,
        capped to the n_frac first elements:
            asizes =    [16,8,4,2,2,1,1] # in FWHM
            n_segments = [1,1,1,1,3,3,6] # azimuthal bins
        Note: if input cube is small (<35 FWHM in x and y), max n_frac will be
        smaller than 7, and the first entry(-ies) of the list will be skipped 
        until the annulus size fits the frames.
    blur: bool, opt
        Whether to convolve the map of significant signals with a Gaussian 
        kernel of fwhm/2 - this may help in the recovery of faint signals and 
        avoid getting sharp edges.
    strategy: {'ADI', 'RDI', 'RADI'}, str opt
        Strategy to be adopted for FEVES. 'RADI' corresponds to first RDI up to 
        the point where FEVES converges, and then ADI using the circumstellar 
        signals estimate at the lst iteration of FEVES-RDI (recommended when a 
        reference cube is available).
    n_br: int, opt
        Number of branches on which the fake planets are injected to compute 
        the throughput.
    radius_int : int, float or tuple of 2 floats, optional
        The radius of the innermost annulus. By default is 0, if >0 then the
        central circular region is discarded.
    fwhm : float, optional
        Size of the FHWM in pixels. Default is 4.
    asizes : list or tuple of 2 lists, optional
        The width of the annuli at each round of fractionation, expressed in 
        FWHM. Should be a single list of floats if strategy is ADI or ARDI.
    n_segments : list, optional
        The number of segments for each annulus, at each round of fractionation.
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
    svd_mode : {'lapack', 'arpack', 'eigen', 'randsvd', 'cupy', 'eigencupy',
        'randcupy', 'pytorch', 'eigenpytorch', 'randpytorch'}, str optional
        Switch for the SVD method/library to be used (only used if the algo
        is set to pca_annular)

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
    init_svd: str, optional {'nnsvd','nnsvda','random'}
        Method used to initialize the iterative procedure to find H and W
        (only used if algo is set to nmf_annular).
        'nndsvd': non-negative double SVD recommended for sparseness
        'nndsvda': NNDSVD where zeros are filled with the average of cube; 
        recommended when sparsity is not desired
        'random': random initial non-negative matrix
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
        If True prints to stdout intermediate info.
    weights: 1d numpy array or list, optional
        Weights to be applied for a weighted mean. Need to be provided if
        collapse mode is 'wmean'.
    interp_order: int, opt
        Interpolation order for throughput vector. Only used if thru_corr set 
        to True.
    rtol: float, optional
        Relative tolerance threshold element-wise in the significant signal 
        image compared to the same image obtained either 1 or 2 iterations
        before, to consider convergence [more details in np.allclose].
    atol: float, optional
        Absolute tolerance threshold element-wise in the significant signal 
        image compared to the same image obtained either 1 or 2 iterations
        before, to consider convergence [more details in np.allclose].
    transmission: numpy array, optional
        Radial transmission of the coronagraph, if any. Array with shape 
        2 x n_rad. The first column should contain the radial separation in 
        pixels, while the other column(s) are the corresponding off-axis 
        transmission (between 0 and 1). If provided, the final returned image
        will be corrected from transmission.

    Returns
    -------
    master_frame : numpy ndarray
        2D array, mean combination of the images obtained from the median 
        combination of de-rotated/re-scaled residuals cube with different
        inner mask values (radial buffers). Always returned. 
    it_cube: numpy ndarray
        [full_output=True] 3D array with final image from each iteration.
    sig_images: numpy ndarray
        [full_output=True] 3D array with significant signals identified at each
        iteration (used at the subsequent iteration).
    residuals_cube : numpy ndarray
        [full_output=True] Residuals cube from the last iteration.
    residuals_cube_ : numpy ndarray
        [full_output=True] Derotated residuals cube from the last iteration.
    it_cube_nd: numpy ndarray
        [full_output=True] 3D array with image obtained from the empty cube at 
        each iteration (i.e. the cube where circumstellar signals are 
        subtracted).
    stim_cube: numpy ndarray
        [full_output=True] 3D array with normalized STIM map calculated at each 
        iteration.
    stim_ori_cube : numpy ndarray
        [full_output=True] 3D array with STIM map calculated at each iteration.
    stim_inv_cube : numpy ndarray
        [full_output=True] 3D array with inverse STIM map calculated at each 
        iteration.
    master_cube: numpy ndarray
        [full_output=True] 3D array with the median-combined images obtained 
        with differnt radial buffers (before they are mean combined into 
        master_frame).
    """
    
    if not buff:
        buffer = 1
    else:
        buffer = max(int(buff*fwhm),1)
        
        
    if not isinstance(thr, (tuple,list)):
        thr = (thr, thr)
    if nproc is None:
        nproc = int(cpu_count()/2)
        
    if isinstance(radius_int, tuple):
        if len(radius_int) != 2:
            raise TypeError("If a tuple, radius_int should be of length 2")
    else:
        radius_int = (radius_int, radius_int)
        
    # select when to do mp depending on buffer
    if buffer < nproc:
        nproc_tmp = nproc
        nproc = 1
    else:
        nproc_tmp=1
    
    if strategy == 'ADI':
        ref_cube = None
    elif strategy == 'RDI' or strategy == 'RADI':
        if cube_ref is None:
            raise ValueError("cube_ref should be provided for RDI or RADI")
        ref_cube = cube_ref.copy() 
        if strategy=='RDI':
            delta_rot=cube.shape[1]*3/fwhm # forces doing RDI instead of ARDI
    else:
        raise ValueError("strategy not recognized: not ADI, RDI or RADI")

    if transmission is not None:
        t_nz = len(transmission)
        if t_nz != 2:
            msg = "If provided, transmission should have shape 2 x N"
            raise TypeError(msg)
        # if transmission doesn't have right format for interpolation, adapt it
        diag = np.sqrt(2)*cube.shape[-1]
        if transmission[0, 0] != 0 or transmission[0, -1] < diag:
            trans_rad_list = transmission[0].tolist()
            for j in range(t_nz-1):
                trans_list = transmission[j+1].tolist()
                # should have a zero point
                if transmission[0, 0] != 0:
                    if j == 0:
                        trans_rad_list = [0]+trans_rad_list
                    trans_list = [0]+trans_list
                # last point should be max possible distance between fc and star
                if transmission[0, -1] < np.sqrt(2)*cube.shape[-1]/2.:
                    if j == 0:
                        trans_rad_list = trans_rad_list+[diag]
                    trans_list = trans_list+[1]
                if j == 0:
                    ntransmission = np.zeros([t_nz, len(trans_rad_list)])
                    ntransmission[0] = trans_rad_list
                ntransmission[j+1] = trans_list
            transmission = ntransmission.copy()

    if asizes is not None and n_segments is not None:
        #asz_def = asizes
        #nsegm_def = n_segments
        if not isinstance(asizes, tuple):
            asizes = (asizes, asizes)
        if not isinstance(n_segments, tuple):
            n_segments = (n_segments, n_segments)
            
        for i in range(2):
            n_frac = len(asizes[i])
            if n_frac != len(n_segments[i]):
                raise ValueError("asizes and nsegments should have same lengths")
        # if n_frac < 1 or n_frac > len(asizes):
        #     msg="set n_frac to a value between 1 and {:.0f}"
        #     raise ValueError(msg.format(len(asizes)))
        # elif n_frac < len(asizes):
        #     asizes = asizes[:n_frac]
        #     n_segments = n_segments[:n_frac]
            
    else:
        msg = "asizes/nsegments not provided => auto-fractionation ({} levels)"
        print(msg.format(n_frac))
        #asz_def = [16,8,4,2,2,1,1]
        #nsegm_def = [1,1,1,1,3,3,6]
        
        # adapt lists based on cube xy sizes
        cy, cx = frame_center(cube)
        
        asizes = [None, None]
        n_segments = [None, None]
        for i in range(2):
            asz_max = np.floor((cy-radius_int[i]-buffer)/fwhm)
            if asz_max < 1:
                msg = "Input array size too small ({} px) - should be > {}px"
                raise ValueError(msg.format(cube.shape[-1], 
                                            int(2*(fwhm+radius_int[i]+buffer))))
                
            # asizes = [a for a in asz_def if a < asz_max]
            # n_segments = [n for i, n in enumerate(nsegm_def) if asz_def[i]<asz_max]
            asizes[i] = [asz_max]
            n_segments[i] = [1]
            while asizes[i][-1] > 2 and len(asizes[i])<n_frac:
                asizes[i].append(asizes[i][-1]/2.)
                n_segments[i].append(1)
            if len(asizes[i])<n_frac:
                asizes[i].append(asizes[i][-1])
                n_segments[i].append(3)
            if len(asizes[i])<n_frac:
                asizes[i].append(asizes[i][-1])
                n_segments[i].append(6)
            if len(asizes[i])<n_frac:
                msg = "WARNING - very small annular subsections requested. "
                msg += "A maximum fractionation level of {} will be used instead "
                msg += "the requested level of {}."
                print(msg.format(len(asizes[i]), n_frac))
                n_frac=len(asizes[i])
            if verbose and strategy == 'RADI':
                if i == 0:
                    lab_str = 'RDI'
                else:
                    lab_str = 'ADI'
                print("*** Annular widths to be used ({}): {} ***".format(lab_str, asizes[i]))
                msg = "*** Number of azimuthal subsections to be used ({}): {} ***"
                print(msg.format(lab_str, n_segments[i]))
            elif verbose and i == 0:
                print("*** Annular widths to be used ({}): {} ***".format(strategy, asizes[i]))
                msg = "*** Number of azimuthal subsections to be used ({}): {} ***"
                print(msg.format(strategy, n_segments[i]))
                
            # convert asizes to FWHM
            asizes[i] = [int(a*fwhm) for a in asizes[i]]
    
    # convert ncomp to a list of lists
    if isinstance(ncomp,int):
        ncomp = ([[ncomp]]*len(asizes[0]), [[ncomp]]*len(asizes[1]))
    elif isinstance(ncomp,list):
        if len(ncomp) != len(asizes[0]):
            raise ValueError("ncomp should have same length as asizes")
        elif isinstance(ncomp[0],int):
            ncomp = ([[ncomp[i]] for i in range(asizes[0])], 
                     [[ncomp[i]] for i in range(asizes[1])])
        elif isinstance(ncomp[0],list):
            ncomp = (ncomp, ncomp)           
        else:
            msg = "if ncomp is a list, its elements must be int or list of int"
            raise TypeError(msg)
    elif isinstance(ncomp,tuple):
        if len(ncomp) != 2:
            raise TypeError("If a tuple, ncompt should have length 2")
        ncomp = ([[ncomp[0]]]*len(asizes[0]), [[ncomp[1]]]*len(asizes[1]))
    else:
        raise TypeError("ncomp can only be integer or list of int")
    
    # define empty lists
    master_cube = []
    master_sig_cube = []
    master_it_cube = []
    master_res_cube = []
    master_res_cube_ = []
    master_it_cube_nd = []
    master_stim_cube = []
    master_stim_ori_cube = []
    master_stim_inv_cube = []
    # 0. start the loop on the buffers (can be done in mp)
    if nproc>1:
        res = pool_map(nproc, _do_one_buff, iterable(range(buffer)), cube, 
                       angle_list, ref_cube, algo, n_it, thr, thr_per_ann, 
                       radius_int, fwhm, asizes, n_segments, n_neigh,
                       False, None, n_br, interp_order, strategy, delta_rot, 
                       ncomp, svd_mode, init_svd, min_frames_lib, 
                       max_frames_lib, tol, scaling, imlib, interpolation, 
                       collapse, atol, rtol, nproc_tmp, True, verbose, weights,
                       blur, **rot_options)
        for bb in range(buffer):
            master_cube.append(res[bb][0])
            master_it_cube.append(res[bb][1])
            master_sig_cube.append(res[bb][2])
            master_res_cube.append(res[bb][3])
            master_res_cube_.append(res[bb][4])
            master_it_cube_nd.append(res[bb][6])
            master_stim_cube.append(res[bb][7])
            master_stim_ori_cube.append(res[bb][8])
            master_stim_inv_cube.append(res[bb][9])

    else:
        for bb in range(buffer):
            res = _do_one_buff(bb, cube, angle_list, ref_cube, algo, n_it, thr, 
                               thr_per_ann, radius_int, fwhm, asizes, 
                               n_segments, n_neigh, False, None,
                               n_br, interp_order, strategy, delta_rot, ncomp, 
                               svd_mode, init_svd, min_frames_lib,
                               max_frames_lib, tol, scaling, imlib, 
                               interpolation, collapse, atol, rtol, nproc_tmp, 
                               True, verbose, weights, blur, **rot_options)
            master_cube.append(res[0])
            master_it_cube.append(res[1])
            master_sig_cube.append(res[2])
            master_res_cube.append(res[3])
            master_res_cube_.append(res[4])
            master_it_cube_nd.append(res[6])
            master_stim_cube.append(res[7])
            master_stim_ori_cube.append(res[8])
            master_stim_inv_cube.append(res[9])
    master_cube = np.array(master_cube)
    master_sig_cube = np.array(master_sig_cube)
    master_it_cube = np.array(master_it_cube)
    master_res_cube = np.array(master_res_cube)
    master_res_cube_ = np.array(master_res_cube_)
    master_it_cube_nd = np.array(master_it_cube_nd)
    master_stim_cube = np.array(master_stim_cube)
    master_stim_ori_cube = np.array(master_stim_ori_cube)
    master_stim_inv_cube = np.array(master_stim_inv_cube)
    
    master_frame = np.nanmean(master_cube, axis=0)
    master_sig_cube = np.nanmean(master_sig_cube, axis=0)
    master_it_cube = np.nanmean(master_it_cube, axis=0)
    master_res_cube = np.nanmean(master_res_cube, axis=0)
    master_res_cube_ = np.nanmean(master_res_cube_, axis=0)
    master_it_cube_nd = np.nanmean(master_it_cube_nd, axis=0)
    master_stim_cube = np.nanmean(master_stim_cube, axis=0)
    master_stim_ori_cube = np.nanmean(master_stim_ori_cube, axis=0)
    master_stim_inv_cube = np.nanmean(master_stim_inv_cube, axis=0)

    # correct for transmission
    if transmission is not None:
        trans = transmission[1]
        radvec_trans = transmission[0]
        f2 = InterpolatedUnivariateSpline(radvec_trans, trans, k=1)
        
        cy, cx = frame_center(master_frame)
        rad_img = dist_matrix(master_frame.shape[-1], cx=cx, cy=cy)
        
        trans_interp = f2(rad_img)
        master_frame[np.where(trans_interp)] /= trans_interp[np.where(trans_interp)]

    # mask everything at the end
    if radius_int[1]:
        master_frame = mask_circle(master_frame, radius_int[1], np.nan)
        master_cube = mask_circle(master_cube, radius_int[1], np.nan)
        master_it_cube = mask_circle(master_it_cube, radius_int[1], np.nan)
        master_res_cube = mask_circle(master_res_cube, radius_int[1], np.nan)
        master_res_cube_ = mask_circle(master_res_cube_, radius_int[1], np.nan)
        master_it_cube_nd = mask_circle(master_it_cube_nd, radius_int[1], np.nan)
    
    if full_output:
        return (master_frame, master_it_cube, master_sig_cube, master_res_cube, 
                master_res_cube_, master_it_cube_nd, master_stim_cube, 
                master_stim_ori_cube, master_stim_inv_cube, master_cube)
    else:
        return master_frame  
    
    
def _do_one_buff(bb, cube, angle_list, ref_cube, algo, n_it, thr, thr_per_ann, 
                 rad_int, fwhm, asizes, n_segments, n_neigh, thru_corr, psfn, 
                 n_br, interp_order, strategy, delta_rot, ncomp, svd_mode, 
                 init_svd, min_frames_lib, max_frames_lib, tol, scaling, imlib, 
                 interpolation, collapse, atol, rtol, nproc, full_output, 
                 verbose, weights, smooth, **rot_options):
    
    def _gauss_interp_2d(array, fwhm_sz=2):
        nan_mask = array.copy()
        nan_mask[np.where(array==0)] = np.nan
        blurred_img = frame_filter_lowpass(nan_mask, mode='gauss', 
                                           fwhm_size=fwhm_sz, iterate=False)
        blurred_img[np.where(np.isnan(blurred_img))] = 0
        nan_mask[np.where(array==0)] = blurred_img[np.where(array==0)]
        
        return nan_mask
    
    def _gauss_interp_3d(array, fwhm_sz=2):
        bl_array = np.zeros_like(array)
        for i in range(array.shape[0]):
            bl_array[i] = _gauss_interp_2d(array[i], fwhm_sz) 
        return bl_array  
    
    def _blurring_2d(array, mask_center_sz, fwhm_sz=2):
        if mask_center_sz:
            frame_mask = mask_circle(array, radius=mask_center_sz+fwhm_sz, 
                                     fillwith=np.nan, mode='out')
            frame_mask2 = mask_circle(array, radius=mask_center_sz, 
                                      fillwith=np.nan, mode='out')
            frame_filt = frame_filter_lowpass(frame_mask, mode='gauss', 
                                              fwhm_size=fwhm_sz, 
                                              iterate=False)
            nonan_loc = np.where(np.isfinite(frame_mask2))
            array[nonan_loc] = frame_filt[nonan_loc]
        else:
            array = frame_filter_lowpass(array, mode='gauss', 
                                         fwhm_size=fwhm_sz, iterate=False)
        return array
    
    def _blurring_3d(array, mask_center_sz, fwhm_sz=2):
        bl_array = np.zeros_like(array)
        for i in range(array.shape[0]):
            bl_array[i] = _blurring_2d(array[i], mask_center_sz, fwhm_sz) 
        return bl_array
    
    if isinstance(rad_int, tuple):
        radius_int = rad_int[0]+bb
    else:
        radius_int = rad_int+bb
    
    nframes = cube.shape[0]
    n_frac = len(asizes[0])
    # 1. Get a first disc estimate, using PCA
    if algo == pca_annular:
        res = pca_annular(cube, angle_list, cube_ref=ref_cube, 
                          radius_int=radius_int, fwhm=fwhm, asize=asizes[0][0], 
                          n_segments=n_segments[0][0], delta_rot=delta_rot, 
                          ncomp=ncomp[0][0][0], svd_mode=svd_mode, nproc=nproc, 
                          min_frames_lib=min_frames_lib, 
                          max_frames_lib=max_frames_lib, tol=tol, 
                          scaling=scaling, imlib=imlib, 
                          interpolation=interpolation, collapse=collapse, 
                          full_output=True, verbose=verbose, weights=weights,
                          **rot_options)
    elif algo == nmf_annular:
        res = nmf_annular(cube, angle_list, cube_ref=ref_cube, 
                          radius_int=radius_int, fwhm=fwhm, asize=asizes[0][0], 
                          n_segments=n_segments[0][0], delta_rot=delta_rot, 
                          ncomp=ncomp[0][0][0], init_svd=init_svd, nproc=nproc,
                          min_frames_lib=min_frames_lib, 
                          max_frames_lib=max_frames_lib, scaling=scaling, 
                          imlib=imlib, interpolation=interpolation, 
                          collapse=collapse, full_output=True, verbose=verbose, 
                          weights=weights, **rot_options)
    else:
        msg = "algo not recognized, can only be pca_annular or nmf_annular"
        raise ValueError(msg)

    #n_it_frac = int(n_frac*n_it)
    n_it_frac = np.sum([len(ncomp[0][i])*n_it for i in range(n_frac)])
    if strategy == 'RADI':
        n_it_tot = int(2*n_it_frac)
    else:
        n_it_tot = n_it_frac
        
    # 2. Identify significant signals with STIM map (outside mask)
    frame = res[-1].copy()
    it_cube = np.zeros([n_it_tot, frame.shape[0], frame.shape[1]])
    it_cube_nd = np.zeros_like(it_cube)
    thru_2d_cube = np.zeros_like(it_cube)
    stim_cube = np.zeros_like(it_cube)
    stim_ori_cube = np.zeros_like(it_cube)
    stim_inv_cube = np.zeros_like(it_cube)
    it_cube[0] = res[-1].copy()
    #it_cube_nd[0] = res[-1].copy()
    
    if strategy == 'RADI':
        res_shape = [2]+list(res[0].shape)
        residuals_cube_tmp = np.zeros(res_shape)
        residuals_cube_tmp[0] = res[0].copy()
        residuals_cube = residuals_cube_tmp.copy()
        residuals_cube_tmp[0] = res[1].copy()
        residuals_cube_ = residuals_cube_tmp.copy()
        residuals_cube_tmp = None
    else:
        residuals_cube = res[0].copy()
        residuals_cube_ = res[1].copy()

    res = _find_significant_signals(res[0], res[1], angle_list, 
                                    thr[0], mask=radius_int, 
                                    thr_per_ann=thr_per_ann, asize=asizes[0][0])
    
    sig_mask, norm_stim, stim, inv_stim = res
    sig_image = frame.copy()
    sig_images = it_cube.copy()
    sig_nd_images = it_cube.copy()
    sig_image[np.where(1-sig_mask)] = 0
    sig_images[0] = sig_image
    stim_cube[0] = norm_stim
    stim_ori_cube[0] = stim
    stim_inv_cube[0] = inv_stim

    # 3.Loop, updating the reference cube before projection by subtracting the
    #   best disc estimate. This is done by providing sig_cube.
    ai = 0
    it = 0
    it_f = 0
    its = 0
    first_time = True # for cond4 to be repeated twice
    while it_f < n_it_frac:
        for pp, npc in enumerate(ncomp[ai][(it_f%n_it_frac)//n_it]):
            if it_f==0 and pp==0:
                it+=1
                it_f+=1
                continue # this is done out of the loop
            it_tmp = (it_f%n_it_frac)//n_it
            # if smooth:# and n_segments[it_f//n_it]>1:
            #     write_fits("TMP_sig_image_it{}.fits".format(it),frame)
            #     #frame = _blurring_2d(frame, None, fwhm_sz=fwhm/2)
            #     frame_s = _gauss_interp_2d(frame, fwhm_sz=fwhm/2)
            #     write_fits("TMP_sig_image_blur_it{}.fits".format(it),frame_s)
            # create and rotate sig cube
            sig_cube = np.repeat(frame[np.newaxis, :, :], nframes, axis=0)
            sig_cube = cube_derotate(sig_cube, -angle_list, imlib=imlib, 
                                     nproc=nproc, **rot_options)
            #write_fits("TMP_sig_cube.fits",sig_cube)
            # create and rotate binary mask
            mask_sig = np.zeros_like(sig_image)
            mask_sig[np.where(sig_image>0)] = 1
            
           # write_fits("TMP_derot_msig_image.fits",sig_image)
            sig_mcube = np.repeat(mask_sig[np.newaxis, :, :], nframes, axis=0)
            #write_fits("TMP_rot_msig_cube.fits",sig_mcube)
            sig_mcube = cube_derotate(sig_mcube, -angle_list, imlib='opencv', 
                                      interpolation='bilinear', nproc=nproc)
            sig_cube[np.where(sig_mcube<0.5)] = 0
            # sig_cube = mask_circle(sig_cube, radius_int, 0)
            # write_fits("TMP_derot_msig_cube.fits", sig_mcube)
            # write_fits("TMP_derot_sig_cube_thr.fits", sig_cube)
            # pdb.set_trace()
        
            # sig_cube = np.repeat(sig_image[np.newaxis, :, :], nframes, axis=0)
            # sig_cube = cube_derotate(sig_cube, -angle_list, imlib=imlib,
            #                          interpolation=interpolation, nproc=1,
            #                          border_mode='constant')
            sig_cube[np.where(sig_cube<0)] = 0
            
            # blur SIG estimate image
            if smooth:
                #write_fits("TMP_sig_cube_it{}.fits".format(it_f),sig_cube)
                sig_cube = _blurring_3d(sig_cube, None, fwhm_sz=fwhm/2)
                #sig_cube = _gauss_interp_3d(sig_cube, fwhm_sz=1)#fwhm/2)
                #write_fits("TMP_sig_cube_blur_it{}.fits".format(it_f),sig_cube)
            
            fit_cube = []
            fresiduals_cube = []
            fresiduals_cube_ = []
            fresiduals_cube_nd = []
            fresiduals_cube_nd_ = []
            fframe_nd = []
            
            if strategy == 'RADI' and isinstance(radius_int, tuple):
                asz_tmp = asizes[ai][it_tmp]-(radius_int[0]-radius_int[1])
            else:
                asz_tmp = asizes[ai][it_tmp]
            
            for tt in range(n_segments[ai][it_tmp]):
                th_i = tt*(360/n_segments[ai][it_tmp])/n_segments[ai][it_tmp]
                if algo == pca_annular:
                    res_it = pca_annular(cube, angle_list, cube_ref=ref_cube, 
                                         radius_int=radius_int, fwhm=fwhm, 
                                         asize=asz_tmp, 
                                         n_segments=n_segments[ai][it_tmp], 
                                         delta_rot=delta_rot, ncomp=npc, 
                                         svd_mode=svd_mode, nproc=nproc, 
                                         min_frames_lib=min_frames_lib,
                                         max_frames_lib=max_frames_lib, tol=tol,
                                         scaling=scaling, imlib=imlib,
                                         interpolation=interpolation, 
                                         collapse=collapse, full_output=True, 
                                         verbose=verbose, theta_init=th_i, 
                                         weights=weights, cube_sig=sig_cube, 
                                         **rot_options)
                else:
                    res_it = nmf_annular(cube, angle_list, cube_ref=ref_cube, 
                                         radius_int=radius_int, fwhm=fwhm, 
                                         asize=asz_tmp, 
                                         n_segments=n_segments[ai][it_tmp], 
                                         delta_rot=delta_rot, ncomp=npc, 
                                         init_svd=init_svd, nproc=nproc, 
                                         min_frames_lib=min_frames_lib,
                                         max_frames_lib=max_frames_lib, tol=tol,
                                         scaling=scaling, imlib=imlib,
                                         interpolation=interpolation, 
                                         collapse=collapse, full_output=True, 
                                         verbose=verbose, theta_init=th_i, 
                                         weights=weights, cube_sig=sig_cube, 
                                         **rot_options)
                fit_cube.append(res_it[-1])
                fresiduals_cube.append(res_it[0].copy())
                fresiduals_cube_.append(res_it[1].copy())
    
                # Scale cube and cube_ref if necessary before inv stim map calculation
                cube_tmp, cube_ref_tmp = _prepare_matrix_ann(cube, ref_cube, scaling, 
                                                             angle_list, fwhm, 
                                                             radius_int, 
                                                             asizes[ai][it_tmp], 
                                                             delta_rot, 
                                                             n_segments[ai][it_tmp],
                                                             theta_init=th_i)
                if algo == pca_annular:
                    res_nd = pca_annular(cube_tmp-sig_cube, angle_list, 
                                         cube_ref=cube_ref_tmp, 
                                         radius_int=radius_int, 
                                         fwhm=fwhm, asize=asz_tmp, 
                                         n_segments=n_segments[ai][it_tmp], 
                                         delta_rot=delta_rot, ncomp=npc,
                                         svd_mode=svd_mode, nproc=nproc,
                                         min_frames_lib=min_frames_lib,
                                         max_frames_lib=max_frames_lib, tol=tol,
                                         scaling=scaling, imlib=imlib, 
                                         interpolation=interpolation, 
                                         collapse=collapse, 
                                         full_output=True, verbose=False, 
                                         theta_init=th_i, weights=weights, 
                                         **rot_options)
                else:
                    res_nd = nmf_annular(cube_tmp-sig_cube, angle_list, 
                                         cube_ref=cube_ref_tmp, 
                                         radius_int=radius_int, 
                                         fwhm=fwhm, asize=asz_tmp, 
                                         n_segments=n_segments[ai][it_tmp], 
                                         delta_rot=delta_rot, ncomp=npc,
                                         init_svd=init_svd, nproc=nproc,
                                         min_frames_lib=min_frames_lib,
                                         max_frames_lib=max_frames_lib, tol=tol,
                                         scaling=scaling, imlib=imlib, 
                                         interpolation=interpolation, 
                                         collapse=collapse, 
                                         full_output=True, verbose=False, 
                                         theta_init=th_i, weights=weights, 
                                         **rot_options)                
                fresiduals_cube_nd.append(res_nd[0])
                fresiduals_cube_nd_.append(res_nd[1])
                fframe_nd.append(res_nd[-1])
                
            fit_cube = np.array(fit_cube)
            fresiduals_cube = np.array(fresiduals_cube)
            fresiduals_cube_ = np.array(fresiduals_cube_)
            fresiduals_cube_nd = np.array(fresiduals_cube_nd)
            fframe_nd = np.array(fframe_nd)
            
            it_cube[it] = np.median(fit_cube,axis=0)
            frame = it_cube[it].copy()
            if strategy == 'RADI' or its>0:
                residuals_cube[its] = np.median(fresiduals_cube,axis=0)
                residuals_cube_[its] = np.median(fresiduals_cube_,axis=0)
                res_tmp_ = residuals_cube_[its]
            else:
                residuals_cube = np.median(fresiduals_cube,axis=0)
                residuals_cube_ = np.median(fresiduals_cube_,axis=0)
                res_tmp_ = residuals_cube_
                
            residuals_cube_nd = np.median(fresiduals_cube_nd,axis=0)
            residuals_cube_nd_ = np.median(fresiduals_cube_nd_,axis=0)
            frame_nd = np.median(fframe_nd,axis=0)
            
            # if smooth:# and n_segments[it_f//n_it]>1:
            #     residuals_cube = _blurring_3d(residuals_cube, None, 
            #                                   fwhm_sz=fwhm/2)
            #     residuals_cube_ = cube_derotate(residuals_cube, angle_list, 
            #                                     imlib=imlib, nproc=nproc,
            #                                     **rot_options)
            #     frame = cube_collapse(residuals_cube_, collapse)
            
            # Find significant signals in results, using disk-subtracted non-derotated cube for calculation of inverse STIM
            res_sig = _find_significant_signals(residuals_cube_nd, 
                                                res_tmp_, angle_list, 
                                                thr[0], mask=radius_int,
                                                thr_per_ann=thr_per_ann, 
                                                asize=asz_tmp)
            sig_mask, norm_stim, stim, inv_stim = res_sig
            #path = "/Users/valentin/Documents/Postdoc/MWC758/NIRC2/20151024/3_postproc_vip_ssim/_autofeves_it_tests/TMP_v10/"
            #write_fits(path+"norm_stim_it{}.fits".format(it), norm_stim)
            
            # expand the mask to consider signals within fwhm/2 of edges
            inv_sig_mask = np.ones_like(sig_mask)
            inv_sig_mask[np.where(sig_mask)] = 0
            if n_neigh > 0:
                inv_sig_mask = mask_sources(inv_sig_mask, n_neigh)
            if radius_int:
                inv_sig_mask = mask_circle(inv_sig_mask, radius_int, 
                                           fillwith=1)
            sig_image = frame.copy()
            sig_image[np.where(inv_sig_mask)] = 0
            sig_image[np.where(sig_image<0)] = 0
            # correct by algo throughput if requested
            if thru_corr:
                if algo == pca_annular:
                    thru, rad_vec = throughput(cube_tmp-sig_cube, -angle_list,
                                               psf_template=psfn, fwhm=fwhm, 
                                               pxscale=1., algo=algo, 
                                               nbranch=n_br, inner_rad=1, 
                                               imlib=imlib, verbose=False, 
                                               radius_int=int(fwhm), ncomp=npc,  
                                               asize=asz_tmp,
                                               n_segments=n_segments[ai][it_tmp], 
                                               fc_snr=5, delta_rot=delta_rot, 
                                               scaling=scaling, 
                                               cube_ref=cube_ref_tmp, tol=tol, 
                                               svd_mode=svd_mode, nproc=nproc, 
                                               min_frames_lib=min_frames_lib, 
                                               max_frames_lib=max_frames_lib,
                                               interpolation=interpolation, 
                                               collapse=collapse)
                else:
                    thru, rad_vec = throughput(cube_tmp-sig_cube, -angle_list,
                                               psf_template=psfn, fwhm=fwhm, 
                                               pxscale=1., algo=algo, 
                                               nbranch=n_br, inner_rad=1, 
                                               imlib=imlib, verbose=False, 
                                               radius_int=int(fwhm), ncomp=npc, 
                                               asize=asz_tmp, 
                                               n_segments=n_segments[ai][it_tmp], 
                                               fc_snr=5, delta_rot=delta_rot, 
                                               scaling=scaling, 
                                               cube_ref=cube_ref_tmp, tol=tol, 
                                               init_svd=init_svd, nproc=nproc, 
                                               min_frames_lib=min_frames_lib, 
                                               max_frames_lib=max_frames_lib,
                                               interpolation=interpolation, 
                                               collapse=collapse)                
    
                if interp_order is not None:
                    # interpolating the throughput vector, spline order 2
                    rad_samp = np.arange(int(np.ceil(rad_vec[0])), 
                                         int(np.floor(rad_vec[-1])),1)
                    n_rad = len(rad_samp)
                    thruput_interp = np.ones([n_br,n_rad])
                    for bb in range(n_br):
                        f = InterpolatedUnivariateSpline(rad_vec, thru[bb], 
                                                         k=interp_order)
                        thruput_interp[bb] = f(rad_samp)
                else:
                    thruput_interp = thru.copy()
                    rad_samp = rad_vec.copy()
                #if thru_arr.ndim==1:
                #    thru_arr = thru_arr[np.newaxis,:]
                thru_2d = _interp2d_rad(thruput_interp, rad_samp, 
                                        cube_tmp.shape[-1], theta_0=0)         
            else:
                thru_2d=np.ones_like(sig_image)
    
            sig_image/=thru_2d
            #tmp=frame.copy()
                
            frame[np.where(frame>0)] = frame[np.where(frame>0)]/thru_2d[np.where(frame>0)]
            it_cube[it] = frame.copy()
            it_cube_nd[it] = frame_nd.copy()
            sig_images[it] = sig_image.copy()
            thru_2d_cube[it] = thru_2d.copy()
            stim_cube[it] = norm_stim.copy()
            stim_ori_cube[it] = stim.copy()
            stim_inv_cube[it] = inv_stim.copy()
    
            # Prepare next iteration?
            ## Check whether sig signals are still found in no-disk cube, if not move to next fractionation
            res_sig_nd = _find_significant_signals(residuals_cube_nd, 
                                                   residuals_cube_nd_, 
                                                   angle_list, thr[1], 
                                                   mask=radius_int,
                                                   thr_per_ann=thr_per_ann, 
                                                   asize=asz_tmp)
            norm_stim = res_sig_nd[1]
            # for improvement purpose write fist
            path = "/Users/valentin/Documents/Postdoc/MWC758/NIRC2/20151024/3_postproc_vip_ssim/_autofeves_it_tests/TMP_v18/"
            #write_fits(path+"norm_stim_nd_it{}.fits".format(it), norm_stim)
            if np.amax(np.abs(norm_stim))<thr[1]:
                no_sig_nd=True
            else:
                no_sig_nd=False
                # add significant signals from disk model to accelerate convergence (both neg and pos signals)
                sig_nd_mask = np.zeros_like(frame_nd)
                sig_nd_mask[np.where(np.abs(norm_stim)>thr[1])] = frame_nd[np.where(np.abs(norm_stim)>thr[1])]
                write_fits(path+"sig_nd_mask_it{}.fits".format(it), sig_nd_mask)
                if smooth:
                    sig_nd_mask = _blurring_2d(sig_nd_mask, None, fwhm_sz=fwhm/4)
                    write_fits(path+"sig_nd_mask_smooth_it{}.fits".format(it), 
                               sig_nd_mask)
                    # npix_sig = len(sig_nd_mask[np.where(sig_nd_mask>0)])
                    # # scale the image to better capture the true flux of significant signals after blurring
                    # if npix_sig>10:
                    #     # take percentile to reduce chance of scaling with outlier
                    #     num_scal = np.nanpercentile(sig_nd_mask[np.where(sig_nd_mask>0)], 
                    #                                 90)
                    #     denom_scal = np.nanpercentile(sig_nd_mask_b[np.where(sig_nd_mask_b>0)], 
                    #                                   90)
                    # else:
                    #     num_scal = np.nanmax(sig_nd_mask[np.where(sig_nd_mask>0)])
                    #     denom_scal = np.nanmax(sig_nd_mask_b[np.where(sig_nd_mask_b>0)])
                    # sig_nd_mask_b *= num_scal/denom_scal
                    # sig_nd_mask = sig_nd_mask_b
                #write_fits(path+"sig_nd_mask_smooth_sca_it{}.fits".format(it), sig_nd_mask)
                sig_nd_images[it] = sig_nd_mask
                frame += sig_nd_mask
    
            write_fits(path+"sig_image_it{}.fits".format(it), sig_image)
            ## check if improvement compared to last iteration
            if it>1:
                cond1 = np.allclose(sig_image, sig_images[it-1], rtol=rtol, 
                                    atol=atol)
                cond2 = np.allclose(sig_image, sig_images[it-2], rtol=rtol, 
                                    atol=atol)
                pos_med = np.nanmedian(sig_image[np.where(sig_image>0)])
                cond3 = np.allclose(sig_nd_images[it], sig_nd_images[it-1], 
                                    rtol=rtol, 
                                    atol=atol*pos_med)
                cond4 = np.allclose(sig_nd_images[it], sig_nd_images[it-2], 
                                    rtol=rtol, 
                                    atol=atol*pos_med)
                cond5 = (strategy == 'RADI' and it_f >= n_it_frac)
                # take mean of last 2 itereations (it's bouncing around optimum)
                if cond4 and first_time:
                    frame -= sig_nd_mask
                    frame += np.nanmean([sig_nd_images[it-1],sig_nd_images[it]],
                                        axis=0)
                    first_time=False
                else:
                    write_fits(path+"norm_stim_nd_it{}.fits".format(it), 
                               norm_stim)
                    write_fits(path+"frame_it{}.fits".format(it), 
                               frame)
                    write_fits(path+"frame_it_nd{}.fits".format(it), 
                               frame_nd)
                    write_fits(path+"sig_nd_mask_it{}.fits".format(it), 
                               sig_nd_mask)
                    if cond1 or cond2 or cond3 or cond4 or no_sig_nd or cond5:
                        print("CONDITIONS MET: ", cond1, cond2, cond3, cond4, 
                              no_sig_nd)
                        if not (strategy == 'RADI' and it_f >= n_it_frac):
                            it_f+= (n_it-(it_f%n_it))
                            it += 1
                            break
                        else:
                            ai+=1
                            write_fits(path+"residuals_cube_RDI{}.fits".format(it), 
                                       residuals_cube[its])
                            # continue to iterate with ADI
                            strategy = 'ADI'
                            first_time=True
                            ref_cube=None
                            if isinstance(rad_int, tuple):
                                radius_int = rad_int[1]+bb
                            if verbose:
                                msg="After {:.0f} iterations, PCA-RDIann -> PCA-ADIann"
                                print(msg.format(it))
                            it_f = 1 #ADDED
                            its += 1
                            it += 1
                            break #ADDED
            it+=1
            it_f+=1

    # mask everything at the end
    if radius_int:
        frame = mask_circle(frame, radius_int, np.nan)
        it_cube = mask_circle(it_cube, radius_int, np.nan)
        residuals_cube = mask_circle(residuals_cube, radius_int, np.nan)
        residuals_cube_ = mask_circle(residuals_cube_, radius_int, np.nan)
        it_cube_nd = mask_circle(it_cube_nd, radius_int, np.nan)

    if full_output:
        return (frame, it_cube, sig_images, residuals_cube, residuals_cube_,
                thru_2d_cube, it_cube_nd, stim_cube, stim_ori_cube, 
                stim_inv_cube)
    else:
        return frame
    

def _find_significant_signals(residuals_cube, residuals_cube_, angle_list, 
                              thr, mask=0, thr_per_ann=True, asize=4):
    # Identifies significant signals with STIM map (outside mask)
    stim = stim_map(residuals_cube_)
    inv_stim = inverse_stim_map(residuals_cube, angle_list)
    if mask is not None:
        stim = mask_circle(stim, mask)
        inv_stim = mask_circle(inv_stim, mask)
    max_inv = np.amax(inv_stim)
    if max_inv <= 0:
        max_inv = np.amax(np.abs(stim))
    norm_stim = stim/max_inv
    if thr_per_ann:
        _, ny, nx = residuals_cube.shape
        n_ann = int((((ny-1)/2.)-mask)/asize)
        for aa in range(n_ann):
            asec = get_annulus_segments(inv_stim, mask+aa*asize, asize, 
                                        mode='mask')[0]
            max_inv = np.amax(asec)
            if max_inv <= 0:
                max_inv = np.amax(np.abs(asec))
            norm_stim[np.where(asec!=0)] = stim[np.where(asec!=0)]/max_inv
    good_mask = np.zeros_like(stim)
    good_mask[np.where(norm_stim>thr)] = 1
    return good_mask, norm_stim, stim, inv_stim
    
    
def _prepare_matrix_ann(cube_tmp, ref_cube, scaling, angle_list, fwhm, 
                        radius_int, asize, delta_rot, n_segments, theta_init):
    if scaling is None:
        if ref_cube is None:
            return cube_tmp.copy(), None
        else:
            return cube_tmp.copy(), ref_cube.copy()
    cube_ref_tmp = None
    n, y, _ = cube_tmp.shape
    n_annuli = int((y / 2 - radius_int) / asize)
    
    if isinstance(delta_rot, tuple):
        delta_rot_tmp = np.linspace(delta_rot[0], delta_rot[1], 
                                    num=n_annuli)
    elif isinstance(delta_rot, (int, float)):
        delta_rot_tmp = [delta_rot] * n_annuli
    else:
        delta_rot_tmp = delta_rot
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
    for ann in range(n_annuli):
        n_segments_ann = n_segments[ann]
        res_ann_par = _define_annuli(angle_list, ann, n_annuli, fwhm,
                                     radius_int, asize, 
                                     delta_rot_tmp[ann], 
                                     n_segments_ann, False)
        pa_thr, inner_radius, ann_center = res_ann_par
        indices = get_annulus_segments(cube_tmp[0], inner_radius, asize,
                                       n_segments_ann, theta_init)
        # Library matrix is created for each segment and scaled if needed
        for j in range(n_segments_ann):
            yy = indices[j][0]
            xx = indices[j][1]
            # shape [nframes x npx_segment]
            matrix_segm = cube_tmp[:, yy, xx]
            matrix_segm = matrix_scaling(matrix_segm, scaling)
            for fr in range(n):
                cube_tmp[fr][yy, xx] = matrix_segm[fr]
            if ref_cube is not None:
                matrix_segm_ref = ref_cube[:, yy, xx]
                matrix_segm_ref = matrix_scaling(matrix_segm_ref,
                                                 scaling)
                for fr in range(ref_cube.shape[0]):
                    cube_ref_tmp[fr][yy, xx] = matrix_segm_ref[fr]
    return cube_tmp, cube_ref_tmp