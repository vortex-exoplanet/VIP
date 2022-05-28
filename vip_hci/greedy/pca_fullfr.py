#! /usr/bin/env python

"""
Full-frame PCA algorithm for ADI, ADI+RDI and ADI+mSDI (IFS data) cubes.
"""

__author__ = 'Valentin Christiaens'
__all__ = ['pca_it']

import inspect
import numpy as np
from scipy.interpolate import interp2d, InterpolatedUnivariateSpline
from ..config import time_ini, timing
from ..fm import cube_inject_companions, frame_inject_companion
from ..psfsub import nmf, pca
from ..preproc import cube_derotate, cube_collapse
from ..metrics import throughput, stim_map, mask_sources, inverse_stim_map
from ..var import (prepare_matrix, mask_circle, cart_to_pol, frame_center,
                   frame_filter_lowpass)


def pca_it(cube, angle_list, cube_ref=None, algo=pca, mode=None, ncomp=1, 
           ncomp_step=1, n_it=10, thr=1, thru_corr=False, n_neigh=0, 
           strategy='ADI', psfn=None, n_br=6, svd_mode='lapack', 
           init_svd='nndsvd', scaling=None, source_xy=None, mask_center_px=None, 
           delta_rot=1, fwhm=4, mask_rdi=None, imlib='vip-fft', 
           interpolation='lanczos4', collapse='median', nproc=1, 
           check_memory=True, full_output=False, verbose=True, weights=None, 
           smooth=False, rtol=1e-2, atol=1, **kwargs_nmf):
    """
    Iterative version of PCA. 
    
    The algorithm finds significant disc or planet signal in the final PCA 
    image, then subtracts it from the input cube (after rotation) just before 
    (and only for) projection onto the principal components. This is repeated
    n_it times, which progressively reduces geometric biases in the image 
    (e.g. negative side lobes for ADI).
    This is similar to the algorithm presented in Pairet et al. (2020).

    The same parameters as pca() can be provided, except 'batch'. There are two 
    additional parameters related to the iterative algorithm: the number of 
    iterations (n_it) and the threshold (thr) used for the identification of 
    significant signals.
    
    Note: The iterative PCA can only be used in ADI, RDI or R+ADI modes.

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
    algo: function, opt, {vip_hci.pca.pca, vip_hci.nmf.nmf}
        Either PCA or NMF, used iteratively.
    scale_list : numpy ndarray, 1d, optional
        Scaling factors in case of IFS data (ADI+mSDI cube). Usually, the
        scaling factors are the central channel wavelength divided by the
        shortest wavelength in the cube (more thorough approaches can be used
        to get the scaling factors). This scaling factors are used to re-scale
        the spectral channels and align the speckles.
    mode: str or None, opt {'Pairet18', 'Pairet21'}
        If None: runs with provided value of 'n_comp' for 'n_it' iterations, 
        and considering threshold 'thr'.
        If 'Pairet18': runs for n_comp iterations, with n_comp=1,...,n_comp, 
        at each iteration (if n_it is provided it is ignored). thr set to 0
        (ignored if provided).
        If 'Pairet21': runs with n_comp=1,...,n_comp, and n_it times for each 
        n_comp (i.e. outer loop on n_comp, inner loop on n_it). thr set to 0
        (ignored if provided).
    ncomp : int or tuple/list of int, optional
        How many PCs are used as a lower-dimensional subspace to project the
        target frames.
        - if mode is None:
            * ADI or RDI: if an int is provided, ``ncomp`` is the number of PCs 
            extracted from ``cube`` itself.
            * RADI: can be a list/tuple of 2 elements corresponding to the ncomp
              to be used for RDI and then ADI.
         - if mode is not None:
             ncomp should correspond to the maximum number of principal 
             components to be tested. The increment will be ncomp_step.
    ncomp_step: int, opt
        Incremental step for number of principal components - used when mode is 
        not None.
    n_it: int, opt
        Number of iterations for the iterative PCA.
        - if mode is None:
            total number of iterations
        - if mode is 'Pairet21' or 'Christiaens21':
            
    thr: float, opt
        Threshold used to identify significant signals in the final PCA image,
        iterartively. This threshold corresponds to the minimum intensity in 
        the STIM map computed from PCA residuals (Pairet et al. 2019), as 
        expressed in units of maximum intensity obtained in the inverse STIM 
        map (i.e. obtained from using opposite derotation angles).
    thru_corr: bool, opt
        Whether to correct the significant signals by the algorithmic 
        throughput before subtraction to the original cube at the next
        iteration. 
        Deprecated: If None, this is not performed. If 'psf', throughput is 
        estimated in a classical way with injected psfs. If 'map', the map of
        identified significant signals is used directly for an estimate of the
        2D throughput.
    n_neigh: int, opt
        If larger than 0, number of neighbouring pixels to the ones included
        as significant to be included in the mask. A larger than zero value can
        make the convergence faster but also bears the risk of including 
        non-significant signals.
    strategy: str, opt
        Whether to do iterative ADI only ('ADI'), iterative RDI only ('RDI'), 
        or iterative RDI and iterative ADI consecutivey ('RADI'). A reference 
        cube needs to be provided for 'RDI' and 'RADI'.
    psfn: 2d numpy array, opt
        If either thru_corr or convolve is set to True, psfn should be a 
        normalised and centered unsaturated psf.
    n_br: int, opt
        Number of branches on which the fake planets are injected to compute 
        the throughput.
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
    delta_rot : int or float, optional
        Factor for tunning the parallactic angle threshold, expressed in FWHM.
        Default is 1 (excludes 1xFHWM on each side of the considered frame).
    fwhm : float, optional
        Known size of the FHWM in pixels to be used. Default value is 4. 
    imlib : str, optional
        See the documentation of the ``vip_hci.preproc.frame_rotate`` function.
    imlib2 : str, optional
        See the documentation of the ``vip_hci.preproc.cube_rescaling_wavelengths`` 
        function.
    interpolation : str, optional
        See the documentation of the ``vip_hci.preproc.frame_rotate`` function.
    collapse : {'median', 'mean', 'sum', 'trimmean'}, str optional
        Sets the way of collapsing the frames for producing a final image.
    check_memory : bool, optional
        If True, it checks that the input cube is smaller than the available
        system memory.
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
    rtol: float, optional
        Relative tolerance threshold element-wise in the significant signal 
        image compared to the same image obtained either 1 or 2 iterations
        before, to consider convergence [more details in np.allclose].
    atol: float, optional
        Absolute tolerance threshold element-wise in the significant signal 
        image compared to the same image obtained either 1 or 2 iterations
        before, to consider convergence [more details in np.allclose].
    kwargs_nmf: 
        Optional arguments of nmf function, including 
        init_svd: {'nndsvd', 'nndsvda', 'random'} (default: 'nndsvd').
        See more options at:
        https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html

    Returns
    -------
    frame : numpy ndarray
        2D array, median combination of the de-rotated/re-scaled residuals cube.
        Always returned. This is the final image obtained at the last iteration.
    it_cube: numpy ndarray
        [full_output=True] 3D array with final image from each iteration. If
        thru_corr is set to True, these images are corrected from algorithmic 
        throughput (not the case of returned 'frame').
    pcs : numpy ndarray
        [full_output=True] Principal components from the last iteration
    residuals_cube : numpy ndarray
        [full_output=True] Residuals cube from the last iteration. 
    residuals_cube_ : numpy ndarray
        [full_output=True] Derotated residuals cube from the last iteration. 
    """
    def _find_significant_signals(residuals_cube, residuals_cube_, angle_list, 
                                  thr, mask=0):
        # Identifies significant signals with STIM map (outside mask)
        stim = stim_map(residuals_cube_)
        inv_stim = inverse_stim_map(residuals_cube, angle_list)
        if mask:
            #stim = mask_circle(stim, mask) # done outside the function
            inv_stim = mask_circle(inv_stim, mask)
        max_inv = np.amax(inv_stim)
        if max_inv == 0:
            max_inv = 1 #np.amin(stim[np.where(stim>0)])
        norm_stim = stim/max_inv
        good_mask = np.zeros_like(stim)
        good_mask[np.where(norm_stim>thr)] = 1
        return good_mask, norm_stim

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
    
    
    if thru_corr and psfn is None:
        msg = "psf should be provided for throughput correction or convolution"
        raise TypeError(msg)
    
    # test whether better
    mask_center_px_ori = mask_center_px
    # if mask_center_px_ori is None:
    #     mask_tmp = int(fwhm+1)
    # else:
    #     mask_tmp = mask_center_px_ori+1
    mask_center_px = None
        
    mask_rdi_tmp = None
    if strategy == 'ADI':
        ref_cube = None
        mask_rdi_tmp=mask_rdi
    elif strategy == 'RDI' or strategy == 'RADI':
        if mask_rdi is not None:
            mask_rdi_tmp = mask_rdi.copy()
        if cube_ref is None:
            raise ValueError("cube_ref should be provided for RDI or RADI")
        ref_cube = cube_ref.copy() 
    else:
        raise ValueError("strategy not recognized: should be ADI, RDI or RADI")
      
    if isinstance(ncomp, (float,int)):
        ncomp_list = [ncomp]
        if strategy == 'RADI':
            ncomp_list.append(ncomp)
    elif isinstance(ncomp, (tuple,list)):
        ncomp_list = ncomp
        if len(ncomp)==1:
            if strategy == 'RADI':
                ncomp_list.append(ncomp)            
        elif not len(ncomp)==2:
            raise ValueError("Length of npc list cannot be larger than 2")
    ncomp_tmp = ncomp_list[0]    
    nframes = cube.shape[0]
    
    if mode is not None:
        final_ncomp = list(range(1,ncomp_tmp+1,ncomp_step))
        if mode == 'Pairet18':
            n_it = ncomp_tmp
            final_ncomp = list(range(1,ncomp_tmp+1,ncomp_step))
            thr=0
        elif mode =='Pairet21' or 'Christiaens21':
            final_ncomp = []
            for npc in range(1,ncomp_tmp+1,ncomp_step):
                for ii in range(n_it):
                    final_ncomp.append(npc)
            n_it = len(final_ncomp)
            if mode =='Pairet21':
                thr=0
    else:
        final_ncomp = [ncomp_tmp]*n_it
    
                
    # 1. Get a first disc estimate, using PCA
    if algo == pca:
        res = pca(cube, angle_list, cube_ref=ref_cube, ncomp=final_ncomp[0], 
              svd_mode=svd_mode, scaling=scaling, mask_center_px=mask_center_px, 
              source_xy=source_xy, delta_rot=delta_rot, fwhm=fwhm, imlib=imlib, 
              interpolation=interpolation, collapse=collapse, 
              mask_rdi=mask_rdi_tmp, check_memory=check_memory, nproc=nproc, 
              full_output=True, verbose=verbose, weights=weights)
        frame = res[0]
        residuals_cube = res[-2]
        residuals_cube_ = res[-1]
    elif algo == nmf:
        cube[np.where(cube<0)]=0 # to avoid a bug
        res = nmf(cube, angle_list, cube_ref=ref_cube, ncomp=final_ncomp[0],
                  init_svd=init_svd, scaling=scaling, 
                  mask_center_px=mask_center_px, source_xy=source_xy, 
                  delta_rot=delta_rot, fwhm=fwhm, imlib=imlib, 
                  interpolation=interpolation, collapse=collapse, 
                  full_output=True, verbose=verbose, **kwargs_nmf)
        frame = res[-1]
        residuals_cube = res[-3]
        residuals_cube_ = res[-2]
    else:
        raise ValueError("algo not recognized. Choose pca or nmf.")
    ## derotate manually residuals cube after blurring
    
    # 2. Identify significant signals with STIM map (outside mask)
    it_cube = np.zeros([n_it,frame.shape[0],frame.shape[1]])
    it_cube_nd = np.zeros_like(it_cube)
    thru_2d_cube = np.zeros_like(it_cube)
    stim_cube = np.zeros_like(it_cube)
    it_cube[0] = frame.copy()
    it_cube_nd[0] = frame.copy()
    sig_mask, norm_stim = _find_significant_signals(residuals_cube, 
                                                    residuals_cube_, 
                                                    angle_list, thr, 
                                                    mask=mask_center_px_ori)
    sig_image = frame.copy()
    sig_images = np.zeros_like(it_cube)
    sig_image[np.where(1-sig_mask)] = 0
    sig_image[np.where(sig_image<0)] = 0
    sig_images[0] = sig_image.copy()
    stim_cube[0] = norm_stim.copy()
    mask_rdi_tmp=None # after first iteration do not use it any more
    
    # 3.Loop, updating the reference cube before projection by subtracting the
    #   best disc estimate. This is done by providing sig_cube.
    #from ..fits import write_fits
    #write_fits("TMP_res_cube.fits", residuals_cube)
    #write_fits("TMP_res_der_cube.fits", residuals_cube_)
    for it in range(1,n_it):
        # rotate image before thresholding
        #write_fits("TMP_sig_image.fits",frame)
        # sometimes there's a cross at the center for PCA no_mask => blur it!
        if smooth:
            frame = _blurring_2d(frame, None, fwhm_sz=2)
            #write_fits("TMP_sig_image_blur.fits",frame)
        # create and rotate sig cube
        sig_cube = np.repeat(frame[np.newaxis, :, :], nframes, axis=0)
        sig_cube = cube_derotate(sig_cube, -angle_list, imlib=imlib, 
                                 #edge_blend='interp+noise', 
                                 nproc=nproc)
        #write_fits("TMP_sig_cube.fits", sig_cube)
        # create and rotate binary mask
        mask_sig = np.zeros_like(sig_image)
        mask_sig[np.where(sig_image>0)] = 1
        
        #write_fits("TMP_derot_msig_image.fits",sig_image)
        sig_mcube = np.repeat(mask_sig[np.newaxis, :, :], nframes, axis=0)
        #write_fits("TMP_rot_msig_cube.fits",sig_mcube)
        sig_mcube = cube_derotate(sig_mcube, -angle_list, imlib='opencv', 
                                  interpolation='bilinear', nproc=nproc)
        sig_cube[np.where(sig_mcube<0.5)] = 0
        # UNCOMMENT FOLLOWING OR NOT?
        #sig_cube = mask_circle(sig_cube, mask_center_px_ori, 0)
        # write_fits("TMP_derot_msig_cube.fits",sig_mcube)
        # write_fits("TMP_derot_sig_cube_thr.fits",sig_cube)
        # pdb.set_trace()
        sig_cube[np.where(sig_cube<0)] = 0
        if algo == pca:
            res = pca(cube, angle_list, cube_ref=ref_cube, ncomp=final_ncomp[it],
                      svd_mode=svd_mode, scaling=scaling,
                      mask_center_px=mask_center_px, source_xy=source_xy, 
                      delta_rot=delta_rot, fwhm=fwhm, imlib=imlib, 
                      interpolation=interpolation, collapse=collapse, 
                      mask_rdi=mask_rdi_tmp, check_memory=check_memory, 
                      nproc=nproc, full_output=True, verbose=verbose, 
                      weights=weights, cube_sig=sig_cube)
            frame = res[0]
            residuals_cube = res[-2]
        elif algo == nmf:
            res = nmf(cube, angle_list, cube_ref=ref_cube, ncomp=final_ncomp[it],
                      init_svd=init_svd, scaling=scaling, 
                      mask_center_px=mask_center_px, source_xy=source_xy, 
                      delta_rot=delta_rot, fwhm=fwhm, imlib=imlib, 
                      interpolation=interpolation, collapse=collapse, 
                      full_output=True, verbose=verbose, cube_sig=sig_cube, 
                      **kwargs_nmf)
            frame = res[-1]
            residuals_cube = res[-3]
        it_cube[it] = frame.copy()
        
        # ADDED for smoothing
        if smooth:
            residuals_cube = _blurring_3d(residuals_cube, None, fwhm_sz=2)
            residuals_cube_ = cube_derotate(residuals_cube, angle_list, 
                                            imlib=imlib, nproc=nproc)
            frame = cube_collapse(residuals_cube_, collapse)
        
        # Scale cube and cube_ref if necessary
        cube_tmp = prepare_matrix(cube, scaling=scaling, 
                                  mask_center_px=mask_center_px, mode='fullfr',
                                  verbose=False)
        cube_tmp = np.reshape(cube_tmp, cube.shape)
        if ref_cube is not None:
            cube_ref_tmp = prepare_matrix(ref_cube, scaling=scaling, 
                                          mask_center_px=mask_center_px, 
                                          mode='fullfr', verbose=False)
            cube_ref_tmp = np.reshape(cube_ref_tmp, ref_cube.shape)
        else:
            cube_ref_tmp = None
        
        if algo == pca:
            res_nd = pca(cube_tmp-sig_cube, angle_list, cube_ref=cube_ref_tmp, 
                         ncomp=final_ncomp[it], svd_mode=svd_mode, scaling=None, 
                         mask_center_px=mask_center_px, source_xy=source_xy, 
                         delta_rot=delta_rot, fwhm=fwhm, imlib=imlib, 
                         interpolation=interpolation, collapse=collapse, 
                         mask_rdi=mask_rdi_tmp, check_memory=check_memory, 
                         nproc=nproc, full_output=True, verbose=verbose, 
                         weights=weights)
            residuals_cube_nd = res_nd[-2]
            frame_nd = res_nd[0]
        elif algo==nmf:
            res_nd = nmf(cube_tmp-sig_cube, angle_list, cube_ref=ref_cube, 
                         ncomp=final_ncomp[it], init_svd=init_svd, 
                         scaling=scaling, mask_center_px=mask_center_px, 
                         source_xy=source_xy, delta_rot=delta_rot, fwhm=fwhm, 
                         imlib=imlib, interpolation=interpolation, 
                         collapse=collapse, full_output=True, verbose=verbose, 
                         **kwargs_nmf)
            residuals_cube_nd = res_nd[-3]
            frame_nd = res_nd[-1]
        sig_mask_p=sig_mask.copy()
        sig_mask, norm_stim = _find_significant_signals(residuals_cube_nd, 
                                                        residuals_cube_, 
                                                        angle_list, thr, 
                                                        mask=mask_center_px_ori)
        # expand the mask to consider signals within fwhm/2 of edges
        inv_sig_mask = np.ones_like(sig_mask)
        inv_sig_mask[np.where(sig_mask)] = 0
        if n_neigh > 0:
            inv_sig_mask = mask_sources(inv_sig_mask, n_neigh)
        if mask_center_px_ori:
            inv_sig_mask = mask_circle(inv_sig_mask, mask_center_px_ori, 
                                       fillwith=1)
        sig_image = frame.copy()
        sig_image[np.where(inv_sig_mask)] = 0
        sig_image[np.where(sig_image<0)] = 0
        # correct by algo throughput if requested
        if thru_corr:# == 'psf':
            # cc = contrast_curve(cube_tmp-sig_cube, -angle_list,
            #                     psf_template=psfn, fwhm=fwhm, pxscale=1.,
            #                     starphot=1., algo=pca, nbranch=n_br, 
            #                     inner_rad=1, plot=False, imlib=imlib,
            #                     verbose=False, mask_center_px=int(fwhm), 
            #                     source_xy=source_xy, ncomp=ncomp,
            #                     delta_rot=delta_rot, cube_ref=cube_ref_tmp,
            #                     crop_ifs=crop_ifs, imlib2=imlib2, 
            #                     svd_mode=svd_mode, interpolation=interpolation, 
            #                     collapse=collapse, weights=weights, conv=conv)'
            #thru_arr = np.array(cc['throughput'])
            #rad_vec = np.array(cc['distance'])
            if algo == pca:
                thru_arr, rad_vec = throughput(cube_tmp-sig_cube, -angle_list,
                                               psf_template=psfn, fwhm=fwhm, 
                                               pxscale=1., algo=algo, 
                                               nbranch=n_br, inner_rad=1, 
                                               imlib=imlib, verbose=False, 
                                               fc_snr=5, 
                                               mask_center_px=int(fwhm), 
                                               source_xy=source_xy, ncomp=1,
                                               delta_rot=delta_rot, 
                                               cube_ref=cube_ref_tmp,
                                               svd_mode=svd_mode, 
                                               interpolation=interpolation, 
                                               collapse=collapse, 
                                               weights=weights, 
                                               mask_rdi=mask_rdi_tmp)
            elif algo == nmf:
                thru_arr, rad_vec = throughput(cube_tmp-sig_cube, -angle_list,
                                               psf_template=psfn, fwhm=fwhm, 
                                               pxscale=1., algo=algo, 
                                               nbranch=n_br, inner_rad=1, 
                                               imlib=imlib, verbose=False, 
                                               fc_snr=5, init_svd=init_svd, 
                                               mask_center_px=int(fwhm), 
                                               source_xy=source_xy, ncomp=1,
                                               delta_rot=delta_rot, 
                                               cube_ref=cube_ref_tmp, 
                                               interpolation=interpolation, 
                                               collapse=collapse, **kwargs_nmf)

            # interpolating the throughput vector, spline order 2
            rad_samp = np.arange(int(np.ceil(rad_vec[0])),
                                 int(np.floor(rad_vec[-1])),1)
            n_rad = len(rad_samp)
            thruput_interp = np.ones([n_br,n_rad])
            for bb in range(n_br):
                f = InterpolatedUnivariateSpline(rad_vec, thru_arr[bb], k=2)
                thruput_interp[bb] = f(rad_samp)
            #if thru_arr.ndim==1:
            #    thru_arr = thru_arr[np.newaxis,:]
            thru_2d = _interp2d_rad(thruput_interp, rad_samp, 
                                    cube_tmp.shape[-1], theta_0=0)
        # elif thru_corr == 'map':
        #     # cc = contrast_curve(cube_tmp-sig_cube, -angle_list,
        #     #                     psf_template=psfn, fwhm=fwhm, pxscale=1.,
        #     #                     starphot=1., algo=pca, nbranch=n_br, 
        #     #                     inner_rad=1, plot=False, imlib=imlib,
        #     #                     verbose=False, mask_center_px=int(fwhm), 
        #     #                     source_xy=source_xy, ncomp=ncomp,
        #     #                     delta_rot=delta_rot, cube_ref=cube_ref_tmp,
        #     #                     crop_ifs=crop_ifs, imlib2=imlib2, 
        #     #                     svd_mode=svd_mode, interpolation=interpolation, 
        #     #                     collapse=collapse, weights=weights, conv=conv)'
        #     #thru_arr = np.array(cc['throughput'])
        #     #rad_vec = np.array(cc['distance'])
        #     thru_2d, fr_nofc, fr_fc = throughput_ext(cube_tmp-sig_cube, 
        #                                              angle_list, sig_image, 
        #                                              fwhm=fwhm, pxscale=1., 
        #                                              algo=pca, imlib=imlib, #nbranch=n_br, 
        #                                              interpolation=interpolation, 
        #                                              #inner_rad=1, 
        #                                              verbose=False, 
        #                                              full_output=True, #fc_snr=5,
        #                                              mask_center_px=int(fwhm), 
        #                                              source_xy=source_xy, 
        #                                              ncomp=ncomp, 
        #                                              delta_rot=delta_rot, 
        #                                              cube_ref=cube_ref_tmp,
        #                                              crop_ifs=crop_ifs, 
        #                                              imlib2=imlib2, 
        #                                              svd_mode=svd_mode, 
        #                                              collapse=collapse, 
        #                                              weights=weights, conv=conv)
        #     thru_2d[np.where(thru_2d<=0)] = 1
        #     thru_2d[np.where(thru_2d>1)] = 1
        #     # smooth the 2d array radially
        #     # interpolating the throughput vector, spline order 2
        #     # cy, cx = frame_center(thru_2d)
        #     # rad_vec = np.arange(int(fwhm), min(cy,cx), int(fwhm))
        #     # rad_samp = np.arange(int(np.ceil(rad_vec[0])), 
        #     #                      int(np.floor(rad_vec[-1])),1)
        #     # n_rad = len(rad_vec)
        #     # n_rad_s = len(rad_samp)
        #     # thruput_interp = np.ones([n_br,n_rad_s])
        #     # thru_arr = np.ones([n_br,n_rad])
        #     # thru_2d_tmp = thru_2d.copy()
        #     # thru_2d_tmp[np.where(sig_image<=0)] = 0
        #     # for rr in range(n_rad):
        #     #     tmp = get_annulus_segments(thru_2d_tmp, rad_vec[rr], 
        #     #                                width=int(fwhm), nsegm=n_br, 
        #     #                                theta_init=0, optim_scale_fact=1, 
        #     #                                mode="val")
        #     #     for bb in range(n_br):
        #     #         tmp_tmp = tmp[bb].copy()
        #     #         if len(np.where(tmp_tmp>0)[0])>0:
        #     #             thru_arr[bb,rr] = np.median(tmp_tmp[np.where(tmp_tmp>0)])
        #     # for bb in range(n_br):
        #     #     f = InterpolatedUnivariateSpline(rad_vec, thru_arr[bb], k=2)
        #     #     thruput_interp[bb] = f(rad_samp)
        #     # #if thru_arr.ndim==1:
        #     # #    thru_arr = thru_arr[np.newaxis,:]
        #     # thru_2d = _interp2d_rad(thruput_interp, rad_samp, 
        #     #                         cube_tmp.shape[-1], theta_0=0)       
        else:
            thru_2d=np.ones_like(sig_image)
        # convolve thru2d map with psf if requested
        # if convolve and thru_corr:
        #     thru_2d_tmp = thru_2d.copy()
        #     thru_2d_tmp[np.where(inv_sig_mask)] = np.nan
        #     mean_thru = np.nanmean(thru_2d_tmp)
        #     thru_2d_tmp = ast_convolve(thru_2d_tmp, psfn)
        #     mean_conv_thru = np.nanmean(thru_2d_tmp)
        #     good_idx = np.where(1-inv_sig_mask)
        #     thru_2d[good_idx] = thru_2d_tmp[good_idx]
        #     if verbose:
        #         msg = "Mean throughput before/after conv = {:.2f}/{:.2f}"
        #         print(msg.format(mean_thru,mean_conv_thru))
        #     #sig_image = frame_filter_lowpass(sig_image, mode="psf", psf=psfn)
        sig_image/=thru_2d
        #frame[np.where(frame>0)] /= thru_2d[np.where(frame>0)]
        it_cube[it] = frame.copy()
        it_cube[it][np.where(frame>0)]/=thru_2d[np.where(frame>0)]
        it_cube_nd[it] = frame_nd.copy()
        sig_images[it] = sig_image.copy()
        thru_2d_cube[it] = thru_2d.copy()
        stim_cube[it] = norm_stim.copy()
        
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
                    ncomp_tmp = ncomp_list[1]
                    strategy = 'ADI'
                    ref_cube=None
                    if verbose:
                        msg="After {:.0f} iterations, PCA-RDI -> PCA-ADI."
                        print(msg.format(it))
    
    # mask everything last
    if mask_center_px_ori is not None:
        frame = mask_circle(frame, mask_center_px_ori)
        it_cube = mask_circle(it_cube, mask_center_px_ori)
        residuals_cube = mask_circle(residuals_cube, mask_center_px_ori)
        residuals_cube_ = mask_circle(residuals_cube_, mask_center_px_ori)
        it_cube_nd = mask_circle(it_cube_nd, mask_center_px_ori)
    
    if full_output:
        #if thru_corr == 'map':
        #    thru_2d = np.array([thru_2d,fr_nofc,fr_fc])
        return (frame, it_cube, sig_images, residuals_cube, residuals_cube_,
                thru_2d_cube, stim_cube, it_cube_nd)
    else:
        return frame
        

def _interp2d_rad(thruput_arr, radii, frame_sz, theta_0=0):
    """
    Converts 1d throughput curve into a 2d image.
    
    Parameters
    ----------
    thruput_arr : 2d numpy array
        Matrix of estimated throughput as a function of azimuth and radius.
    radii : 1d numpy array
        Vector of radii corresponding to the throughput
    frame_sz : int
        Size of the 2d image to be returned
    theta_0: float, opt
        Polar (trigonometric) angle of the first branch used for fake planet
        injections.

    Returns
    -------
    thru_2d : 2d numpy array
        2D throughput image.

    """
    n_br = thruput_arr.shape[0]
    # median over 3 azimuthally shifted thruput maps to smooth 2pi transition
    thru_2d_list = []
    for ii in range(4):
        thetas = np.linspace(theta_0, theta_0+360, n_br, endpoint=False)
        sh=ii*90
        sh_idx = np.where(thetas>theta_0+360-sh) 
        thetas[sh_idx] = thetas[sh_idx]-360
        
        f_thru = interp2d(radii, thetas, thruput_arr)
        
        yy, xx = np.ogrid[:frame_sz, :frame_sz]  
        cy, cx = (frame_sz-1)/2, (frame_sz-1)/2
        r, theta = cart_to_pol(xx, yy, cx, cy)
        theta[np.where(theta<np.amin(thetas))]+=360
        theta[np.where(theta>np.amax(thetas))]-=360
    
        thru_2d = np.zeros([frame_sz,frame_sz])
        for i in range(frame_sz):
            for j in range(frame_sz):
                thru_2d[i,j] = f_thru(r[i,j], theta[i,j])
        thru_2d[np.where(thru_2d<=0)] = 1
        thru_2d[np.where(thru_2d>1)] = 1
        thru_2d_list.append(thru_2d)
    
    final_thru_2d = np.median(np.array(thru_2d_list),axis=0)
    
    return final_thru_2d


def throughput_ext(cube, angle_list, sig_image, fwhm, pxscale, algo, #nbranch=1, 
                   #theta=0, inner_rad=1, fc_rad_sep=3, wedge=(0,360),  
                   imlib='opencv', interpolation='lanczos4', verbose=True, 
                   full_output=False, **algo_dict):
    """ Measures the throughput for chosen algorithm and input dataset (ADI or
    ADI+mSDI). The final throughput is the average of the same procedure
    measured in ``nbranch`` azimutally equidistant branches.

    Parameters
    ---------_
    cube : numpy ndarray
        The input cube, 3d (ADI data) or 4d array (IFS data), without fake
        companions.
    angle_list : numpy ndarray
        Vector with the parallactic angles.
    sig_image : numpy ndarray
        Frame containing the identified significant signal, with the rest of 
        the image masked to 0.
    fwhm: int or float or 1d array, optional
        The the Full Width Half Maximum in pixels. It can handle a different
        FWHM value for different wavelengths (IFS data).
    pxscale : float
        Plate scale in arcsec/px.
    algo : callable or function
        The post-processing algorithm, e.g. vip_hci.pca.pca. Third party Python
        algorithms can be plugged here. They must have the parameters: 'cube',
        'angle_list' and 'verbose'. Optionally a wrapper function can be used.
    #nbranch : int optional
    #    Number of branches on which to inject fakes companions. Each branch
    #    is tested individually.
    # theta : float, optional
    #     Angle in degrees for rotating the position of the first branch that by
    #     default is located at zero degrees. Theta counts counterclockwise from
    #     the positive x axis.
    inner_rad : int, optional
        Innermost radial distance to be considered in terms of FWHM.
    # fc_rad_sep : int optional
    #     Radial separation between the injected companions (in each of the
    #     patterns) in FWHM. Must be large enough to avoid overlapping. With the
    #     maximum possible value, a single fake companion will be injected per
    #     cube and algorithm post-processing (which greatly affects computation
    #     time).
    wedge : tuple of floats, optional
        Initial and Final angles for using a wedge. For example (-90,90) only
        considers the right side of an image.
    full_output : bool, optional
        If True returns intermediate arrays.
    imlib : str, optional
        See the documentation of the ``vip_hci.preproc.frame_shift`` function.
    interpolation : str, optional
        See the documentation of the ``vip_hci.preproc.frame_shift`` function.
    verbose : bool, optional
        If True prints out timing and information.
    **algo_dict
        Parameters of the post-processing algorithms must be passed here.

    Returns
    -------
    thruput : numpy ndarray
        2d array containing which ratio of flux is preserved after 
        post-processing for each original significant intensity pixels.

    If full_output is True then the function returns: thruput, frame_fc and 
    frame_nofc.
    frame_fc : numpy ndarray
        2d array with the PCA processed frame with significant signal.
    frame_nofc : numpy ndarray
        2d array, PCA processed frame without significant signal.

    """
    nbranch=1
    theta=0
    array = cube
    parangles = angle_list

    if array.ndim != 3 and array.ndim != 4:
        raise TypeError('The input array is not a 3d or 4d cube')
    else:
        if array.ndim == 3:
            if array.shape[0] != parangles.shape[0]:
                msg = 'Input parallactic angles vector has wrong length'
                raise TypeError(msg)
            if sig_image.ndim != 2:
                raise TypeError('Template PSF is not a frame or 2d array')
            #maxfcsep = int((array.shape[1]/2.)/fwhm)-1
            # if fc_rad_sep < 3 or fc_rad_sep > maxfcsep:
            #     msg = 'Too large separation between companions in the radial '
            #     msg += 'patterns. Should lie between 3 and {}'
            #     raise ValueError(msg.format(maxfcsep))

        elif array.ndim == 4:
            if array.shape[1] != parangles.shape[0]:
                msg = 'Input vector or parallactic angles has wrong length'
                raise TypeError(msg)
            if sig_image.ndim != 3:
                raise TypeError('Template PSF is not a frame, 3d array')
            if 'scale_list' not in algo_dict:
                raise ValueError('Vector of wavelength not found')
            else:
                if algo_dict['scale_list'].shape[0] != array.shape[0]:
                    raise TypeError('Input wavelength vector has wrong length')
                # if isinstance(fwhm, float) or isinstance(fwhm, int):
                #     maxfcsep = int((array.shape[2] / 2.) / fwhm) - 1
                # else:
                #     maxfcsep = int((array.shape[2] / 2.) / np.amin(fwhm)) - 1
                # if fc_rad_sep < 3 or fc_rad_sep > maxfcsep:
                #     msg = 'Too large separation between companions in the '
                #     msg += 'radial patterns. Should lie between 3 and {}'
                #     raise ValueError(msg.format(maxfcsep))

        if sig_image.shape[1] % 2 == 0:
            raise ValueError("Only odd-sized PSF is accepted")
        if not hasattr(algo, '__call__'):
            raise TypeError('Parameter `algo` must be a callable function')
        # if not isinstance(inner_rad, int):
        #     raise TypeError('inner_rad must be an integer')
        #angular_range = wedge[1] - wedge[0]
        # if nbranch > 1 and angular_range < 360:
        #     msg = 'Only a single branch is allowed when working on a wedge'
        #     raise RuntimeError(msg)

    if isinstance(fwhm, (np.ndarray,list)):
        fwhm_med = np.median(fwhm)
    else:
        fwhm_med = fwhm

    if verbose:
        start_time = time_ini()
    #***************************************************************************
    # Compute noise in concentric annuli on the "empty frame"
    argl = inspect.getargspec(algo).args
    if 'cube' in argl and 'angle_list' in argl and 'verbose' in argl:
        if 'fwhm' in argl:
            frame_nofc = algo(cube=array, angle_list=parangles, fwhm=fwhm_med,
                              verbose=False, **algo_dict)
            # if algo_dict.pop('scaling',None):
            #     new_algo_dict = algo_dict.copy()
            #     new_algo_dict['scaling'] = None
            #     frame_nofc_noscal = algo(cube=array, angle_list=parangles, 
            #                              fwhm=fwhm_med, verbose=False, 
            #                              **new_algo_dict)
            # else:
            #     frame_nofc_noscal = frame_nofc
        else:
            frame_nofc = algo(array, angle_list=parangles, verbose=False,
                              **algo_dict)
            # if algo_dict.pop('scaling',None):
            #     new_algo_dict = algo_dict.copy()
            #     new_algo_dict['scaling'] = None
            #     frame_nofc_noscal = algo(cube=array, angle_list=parangles,
            #                              verbose=False, **new_algo_dict)
            # else:
            #     frame_nofc_noscal = frame_nofc
                
    if verbose:
        msg1 = 'Cube without fake companions processed with {}'
        print(msg1.format(algo.__name__))
        timing(start_time)

    # noise, res_level, vector_radd = noise_per_annulus(frame_nofc, 
    #                                                   separation=fwhm_med,
    #                                                   fwhm=fwhm_med, 
    #                                                   wedge=wedge)
    # noise_noscal, _, _ = noise_per_annulus(frame_nofc_noscal, 
    #                                        separation=fwhm_med, fwhm=fwhm_med, 
    #                                        wedge=wedge)                                       
    # vector_radd = vector_radd[inner_rad-1:]
    # noise = noise[inner_rad-1:]
    # res_level = res_level[inner_rad-1:]
    # noise_noscal = noise_noscal[inner_rad-1:]
    if verbose:
        print('Measured annulus-wise noise in resulting frame')
        timing(start_time)

    # We crop the PSF and check if PSF has been normalized (so that flux in
    # 1*FWHM aperture = 1) and fix if needed
    new_psf_size = int(round(3 * fwhm_med))
    if new_psf_size % 2 == 0:
        new_psf_size += 1

    if cube.ndim == 3:
        n, y, x = array.shape
        psf_template = sig_image.copy() 
        #normalize_psf(psf_template, fwhm=fwhm, verbose=verbose,
        #size=min(new_psf_size,
        #psf_template.shape[1]))

        # Initialize the fake companions
        #angle_branch = angular_range / nbranch
        #thruput_arr = np.zeros((nbranch, sig_image.shape[0]))
        cy, cx = frame_center(array[0])

        # each branch is computed separately
        for br in range(nbranch):
            # each pattern is computed separately. For each one the companions
            # are separated by "fc_rad_sep * fwhm", interleaving the injections
            #for irad in range(fc_rad_sep):
            radvec = np.array([0])#vector_radd[irad::fc_rad_sep]
            cube_fc = array.copy()
            # filling map with small numbers
            fc_map = np.ones_like(array[0]) * 1e-6
            fcy = []
            fcx = []
            for i in range(radvec.shape[0]):
                flux = 1 #fc_snr * noise_noscal[irad + i * fc_rad_sep]
                cube_fc = cube_inject_companions(cube_fc, psf_template,
                                                 parangles, flux, pxscale,
                                                 rad_dists=[radvec[i]],
                                                 theta=#br*angle_branch +
                                                       theta,
                                                 imlib=imlib, verbose=False,
                                                 interpolation=
                                                    interpolation)
                y = cy + radvec[i] * np.sin(np.deg2rad(#br * angle_branch +
                                                       theta))
                x = cx + radvec[i] * np.cos(np.deg2rad(#br * angle_branch +
                                                       theta))
                fc_map = frame_inject_companion(fc_map, psf_template, y, x,
                                                flux, imlib, interpolation)
                fcy.append(y)
                fcx.append(x)

            if verbose:
                msg2 = 'Fake sig image injected in branch {} '
                #msg2 += '(pattern {}/{})'
                print(msg2.format(br+1))
                timing(start_time)

            #***************************************************************
            arg = inspect.getargspec(algo).args
            if 'cube' in arg and 'angle_list' in arg and 'verbose' in arg:
                if 'fwhm' in arg:
                    frame_fc = algo(cube=cube_fc, angle_list=parangles,
                                    fwhm=fwhm_med, verbose=False, **algo_dict)
                else:
                    frame_fc = algo(cube=cube_fc, angle_list=parangles,
                                    verbose=False, **algo_dict)
            else:
                msg = 'Input algorithm must have at least 3 parameters: '
                msg += 'cube, angle_list and verbose'
                raise ValueError(msg)

            if verbose:
                msg3 = 'Cube with fake companions processed with {}'
                msg3 += '\nMeasuring its annulus-wise throughput'
                print(msg3.format(algo.__name__))
                timing(start_time)

            #***************************************************************
            injected_flux = fc_map
            recovered_flux = frame_fc - frame_nofc
            thruput = np.ones_like(injected_flux)
            num = recovered_flux[np.where(injected_flux > 0)]
            denom = injected_flux[np.where(injected_flux > 0)]
            thruput[np.where(injected_flux > 0)] = num/denom
            thruput[np.where(thruput <= 0)] = 1
            thruput[np.where(thruput > 1)] = 1
            #thruput_arr[br] = thruput

    elif cube.ndim == 4:
        w, n, y, x = array.shape
        if isinstance(fwhm, (int, float)):
            fwhm = [fwhm] * w
        psf_template = sig_image.copy()
                       #normalize_psf(psf_template, fwhm=fwhm, verbose=verbose,
                       #              size=min(new_psf_size,
                       #                       psf_template.shape[1]))

        # Initialize the fake companions
        #angle_branch = angular_range / nbranch
        #thruput_arr = np.zeros((nbranch, sig_image.shape))
        cy, cx = frame_center(array[0, 0])

        # each branch is computed separately
        for br in range(nbranch):
            # each pattern is computed separately. For each pattern the
            # companions are separated by "fc_rad_sep * fwhm"
            # radius = vector_radd[irad::fc_rad_sep]
            #for irad in range(fc_rad_sep):
            radvec = np.array([0])#vector_radd[irad::fc_rad_sep]
            thetavec = range(int(theta), int(theta) + 360,
                             360 // len(radvec))
            cube_fc = array.copy()
            # filling map with small numbers
            fc_map = np.ones_like(array[:, 0]) * 1e-6
            fcy = []
            fcx = []
            for i in range(radvec.shape[0]):
                flux = 1 #fc_snr * noise_noscal[irad + i * fc_rad_sep]
                cube_fc = cube_inject_companions(cube_fc, psf_template,
                                                 parangles, flux, pxscale,
                                                 rad_dists=[radvec[i]],
                                                 theta=thetavec[i],
                                                 verbose=False)
                y = cy + radvec[i] * np.sin(np.deg2rad(#br * angle_branch +
                                                       thetavec[i]))
                x = cx + radvec[i] * np.cos(np.deg2rad(#br * angle_branch +
                                                       thetavec[i]))
                fc_map = frame_inject_companion(fc_map, psf_template, y, x,
                                                flux)
                fcy.append(y)
                fcx.append(x)

            if verbose:
                msg2 = 'Fake companions injected in branch {} '
                #msg2 += '(pattern {}/{})'
                print(msg2.format(br + 1))#, irad + 1, fc_rad_sep))
                timing(start_time)

            # **************************************************************
            arg = inspect.getargspec(algo).args
            if 'cube' in arg and 'angle_list' in arg and 'verbose' in arg:
                if 'fwhm' in arg:
                    frame_fc = algo(cube=cube_fc, angle_list=parangles,
                                    fwhm=fwhm_med, verbose=False, **algo_dict)
                else:
                    frame_fc = algo(cube=cube_fc, angle_list=parangles,
                                    verbose=False, **algo_dict)

            if verbose:
                msg3 = 'Cube with fake companions processed with {}'
                msg3 += '\nMeasuring its annulus-wise throughput'
                print(msg3.format(algo.__name__))
                timing(start_time)

            # **************************************************************
            injected_flux = np.mean(fc_map, axis=0)
            recovered_flux = frame_fc - frame_nofc
            thruput = np.ones_like(injected_flux)
            num = recovered_flux[np.where(injected_flux > 0)]
            denom = injected_flux[np.where(injected_flux > 0)]
            thruput[np.where(injected_flux > 0)] = num/denom
            thruput[np.where(thruput <= 0)] = 1
            thruput[np.where(thruput > 1)] = 1
            #thruput_arr[br] = thruput

    if verbose:
        msg = 'Finished measuring the throughput in {} branches'
        print(msg.format(nbranch))
        timing(start_time)

    if full_output:
        return thruput, frame_fc, frame_nofc
    else:
        return thruput