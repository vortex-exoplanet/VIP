#! /usr/bin/env python

"""
FEVES algorithm for iterative PCA/NMF in progressively fractionated images.
"""

__author__ = 'Valentin Christiaens'
__all__ = ['feves_opt',
           'greeds_opt']

try:
    # before skimage version '0.18' the function is skimage.measure.compare_ssim
    from skimage.measure import compare_ssim as ssim
except:
    # for skimage version '0.18' or above the function is skimage.metrics.structural_similarity
    from skimage.metrics import structural_similarity as ssim
from multiprocessing import cpu_count
import numpy as np
from scipy.ndimage import gaussian_filter
import os
from os.path import isdir, isfile
from .pca_local import feves
from .pca_fullfr import pca_it
from ..config.utils_conf import pool_map, iterable
from ..fits import write_fits, open_fits
from ..preproc import cube_crop_frames, frame_crop
from ..preproc.derotation import _define_annuli
from ..psfsub import nmf_annular, nmf, pca_annular, pca
from ..var import frame_center, get_annulus_segments


def feves_opt(cube, angle_list, cube_ref=None, fwhm=4, strategy='ADI', 
              algo=pca_annular, fm='med_stim', gt_image=None, nfrac_max=6, 
              ncomps=[1,2,3,4,5], nits=[1,2], thrs=[0,0.5], drots=[0,0.5,1], 
              buffer=1, fiducial_results=None, thru_corr=False, n_br=6, 
              psfn=None, starphot=1., plsc=1., svd_mode='lapack', 
              init_svd='nndsvd', scaling=None, mask_center_px=None, 
              imlib='opencv', interpolation='lanczos4', collapse='median', 
              check_memory=True, nproc=1, full_output=False, verbose=True, 
              weights=None, debug=False, path='', overwrite=True,  
              auto_crop=False, smooth=False, **rot_options):
    """
    Optimal version of Fractionation for Embedded Very young Exoplanet Search 
    algorithm: Iterative PCA or NMF applied in progressively more fractionated 
    image sections. 
    
    This method is relevant for stellar residuals varying temporally and to 
    some extent spatially.
    
    See complete description of the algorithm in Christiaens et al. (2022b):
    *** INSERT LINK ***    

    Note: The feves_opt routine can only be used in ADI or RDI modes.

    Parameters
    ----------
    cube : str or numpy ndarray, 3d or 4d
        Input cube (ADI or ADI+mSDI). If a string is given, it must correspond
        to the path to the fits file to be opened in memmap mode (for PCA
        incremental of ADI 3d cubes). Note: the algorithm is the most efficient
        if the provided cube is already cropped to ~35 FWHM across. This is 
        automatically done if auto_crop is set to True.
    angle_list : numpy ndarray, 1d
        Corresponding parallactic angle for each frame.
    cube_ref : numpy ndarray, 3d, optional
        Reference library cube. For Reference Star Differential Imaging.
    fwhm : float, optional
        Known size of the FHWM in pixels to be used. Default value is 4.
    strategy: str, opt
        Whether to do iterative ADI only ('ADI'), iterative RDI only ('RDI'), 
        or iterative RDI and iterative ADI consecutivey ('RADI'). A reference 
        cube needs to be provided for 'RDI' and 'RADI'.
    algo: function, opt, {vip_hci.pca.pca, vip_hci.nmf.nmf}
        Either PCA or NMF in concentric annuli, used iteratively.
    fm : str, optional {'med_stim','mean_stim','max_stim','perc**_stim','ssim', 
                        'fiducial'}
        Figure of merit to reconstruct the optimal image: median, mean, max or
        ** percentile (replace ** with a number between 0 and 100) of the STIM 
        value in each patch, respectively, OR the maximum structural 
        similarity index measure ('ssim') with a reference ground truth image,
        OR the results obtained with a fiducial cube.
    gt_image: 2d numpy a rray, optional
        If fm is set to 'ssim', this is the ground truth image to which the 
        processed FEVES images are compared to, when searching for the optimal 
        reduction parameters.
    nfrac_max: int, optional
        Maximum fractionation level. At the first iteration, the full frame is 
        considered, for each sugsequent iteration, the annular segments will be
        thinner and/or subdivided in azimuthal segments. The higher nfrac, the 
        smaller the segments in the last iteration. Max: 7.
        # 1. it. PCA in a single ~16FWHM-wide annulus or full frame
        # 2. it. PCA in ~8FWHM-wide annuli (2)
        # 3. it. PCA in ~4FWHM-wide annuli (4)
        # 4. it. PCA in ~2FWHM-wide annuli (8)
        # Loops on azimuthal segments:
        # 5. it. PCA in ~2FWHM-annuli and 3 az. segments [shift 3 times az.]
        # 6. it. PCA in ~2FWHM-annuli and 6 az. segments [shift 6 times az.]
    ncomps : list or tuple of int/None/float, optional
        Test list of number of PC values used as a lower-dimensional subspace 
        to project the target frames. The format of each test ncomp in the list 
        should be compatible with the pca.pca() function, that is:

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

    nits: list of int, opt
        Maximum number of iterations for the iterative PCA. Note that if no 
        improvement is found from one iteration to the next, the algorithm 
        will stop automatically.
    thrs: list of float, tuple of floats or tuple or tuples/lists, opt
        List of tested thresholds. The threshold is used to (iteratively) 
        identify significant signals in the final PCA image obtained in each 
        regime. If a float, the same value will be used for both regimes. This 
        threshold is a minimum pixel intensity in the STIM map computed with 
        the PCA residual cube (Pairet et al. 2019), as expressed 
        in units of max. pixel intensity obtained in the inverse STIM map. 
        Recommended value: 1. But for discs with bright axisymmetric signals, a 
        lower value may be necessary.
    drots : list of float, optional
        Factor for adjusting the parallactic angle threshold, expressed in
        FWHM. Default is 1 (excludes 1 FHWM on each side of the considered
        frame). If a tuple of two floats is provided, they are used as the lower
        and upper intervals for the threshold (grows linearly as a function of
        the separation).
    buffer: float, option
        Buffer in terms of FWHM used to smooth the transition. In practice, the
        annular (resp. full-frame) PCA will be performed several times, with an 
        outer radius (resp. inner radius) ranging between (r_lim-buffer/2)*FWHM
        and (r_lim+buffer/2)*FWHM, per steps of 1 px.
    fiducial_results: 4d numpy array, optional
        If fm is set to fiducial, this 4d cube of images should contain, the 
        optimal nfrac (axis 0, index 0), optimal ncomp (axis 0, index 1), 
        optimal number of iterations (axis 0, index 2), optimal thr (axis 0, 
        index 3) and optimal delta rot (axis 0, index 4), for each buffer and 
        azimuthal shift (axis 1).
    thru_corr: bool, opt
        Whether to correct the significant signals by the algorithmic 
        throughput before subtraction to the original cube at the next
        iteration. 
        Deprecated: If None, this is not performed. If 'psf', throughput is 
        estimated in a classical way with injected psfs. If 'map', the map of
        identified significant signals is used directly for an estimate of the
        2D throughput.
    n_br: int, opt
        Number of branches on which the fake planets are injected to compute 
        the throughput.
    psfn: 2d numpy array, opt
        If thru_corr is set to True, psfn should be a normalised and centered 
        unsaturated psf.
    starphot: float, opt
        Integrated flux of the star in a 1FWHM aperture. Only relevant for
        accurate output contrast curves in full_output mode.
    plsc: float, opt
        Plate scale in arcsec/px. Only relevant for accurate output contrast 
        curves in full_output mode.
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
        radius of the circular mask. If buffer is larger than 0, the effective 
        mask_center_px will be increased accordingly with the outer radius of 
        the annulus, for the different reductions used for edge smoothing.
        
    source_xy : tuple of int, optional
        For ADI-PCA, this triggers a frame rejection in the PCA library, with
        ``source_xy`` as the coordinates X,Y of the center of the annulus where
        the PA criterion is estimated. When ``ncomp`` is a tuple, a PCA grid is
        computed and the S/Ns (mean value in a 1xFWHM circular aperture) of the
        given (X,Y) coordinates are computed.
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
    interpolation : str, optional
        See the documentation of the ``vip_hci.preproc.frame_rotate`` function.
    collapse : {'median', 'mean', 'sum', 'trimmean'}, str optional
        Sets the way of collapsing the frames for producing a final image.
    ifs_collapse_range: str 'all' or tuple of 2 int
        If a tuple, it should contain the first and last channels where the mSDI 
        residual channels will be collapsed (by default collapses all channels).
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
    debug: bool, opt
        Whether to save intermediate fits files in directory given by path.
    path: str, opt
        Where to save intermediate fits files if debug is True. 

    Returns
    -------
    master_frame: 2d numpy ndarray
        2D array obtained from combining images obtained at optimal contrast 
        at each radius.
    master_cube: 3d numpy ndarray
        [full_output=True] Master cube of all nanmedian-combined images used 
        to produce master_frame.
    drot_opt_arr: 1d numpy ndarray
        [full_output=True] Array containing the delta_rot values used to 
        produce each image of master_cube. 
    npc_opt_arr: 1d numpy ndarray
        [full_output=True] Array containing the optimal npc value used to 
        produce each image of master_cube. 
    nit_opt_arr: 1d numpy ndarray
        [full_output=True] Array containing the optimal number of iterations 
        used to produce each image of master_cube. 
    cc_ss_opt_arr: 1d numpy ndarray
        [full_output=True] Array containing the optimal contrast value achieved 
        at each radius (used to select optimal images), when significant signal
        is subtracted from the cube.
    cc_ws_opt_arr: 1d numpy ndarray
        [full_output=True] Array containing the optimal contrast value achieved 
        at each radius (used to select optimal images), when significant signal
        is NOT subtracted from the cube.    
    cc_rad_arr: 1d numpy ndarray
        [full_output=True] Array containing the radii where the contrasts are 
        achieved (used to select optimal images).       
    """
    # formatting
    
    if not isinstance(ncomps, (list,tuple,range)):
        raise TypeError("ncomps can only be list, tuple or range")
    if not isinstance(ncomps, list):
        ncomps = list(ncomps)
    if not isinstance(nits, (tuple,list,range)):
        raise TypeError("nits can only be list, tuple or range")
    if not isinstance(nits, list):
        nits=list(nits)
    if not isinstance(thrs, (tuple,list,range)):
        raise TypeError("thrs can only be list, tuple or range")
    if not isinstance(thrs, list):
        thrs=list(thrs)
    if not isinstance(drots, (tuple,list,range)):
        raise TypeError("drots can only be list, tuple or range")
    if not isinstance(drots, list):
        drots=list(drots)        
    if thru_corr and psfn is None:
        msg = "psf should be provided for throughput correction"
        raise TypeError(msg)

    if mask_center_px is None:
        mask_inner_sz = 0
    else:
        mask_inner_sz = mask_center_px
    
    if strategy == 'ADI':
        ref_cube = None
    elif strategy == 'RDI' or strategy == 'RADI':
        if cube_ref is None:
            raise ValueError("cube_ref should be provided for RDI or RADI")
        ref_cube = cube_ref.copy() 
    else:
        raise ValueError("strategy not recognized: should be ADI, RDI or RADI")
     
    if path:
        if not isdir(path):
            os.makedirs(path)
    if not buffer:
        buff = 1
    else:
        buff = max(int(buffer*fwhm),1)
    
    # crop if needed
    if ref_cube is None:
        ref_cube_tmp = None
    else:
        ref_cube_tmp = ref_cube.copy()
        
    if auto_crop:
        if ref_cube is not None:
            ref_sz = ref_cube_tmp.shape[-1]
        crop_sz = min(int(33*fwhm + 2*mask_inner_sz + 2*buffer), ref_sz)
        if not crop_sz%2:
            crop_sz+=1
        if cube.shape[-1] > crop_sz:
            cube_tmp = cube_crop_frames(cube, crop_sz)
        if ref_cube is not None:
            if ref_sz>crop_sz:
                ref_cube_tmp = cube_crop_frames(ref_cube_tmp, crop_sz)
    else:
        cube_tmp = cube.copy()
        
    if gt_image is not None:
        if gt_image.ndim !=2:
            raise TypeError("gt image should be 2d")
        cond1 = gt_image.shape[0] < cube_tmp.shape[1]
        cond2 = gt_image.shape[1] < cube_tmp.shape[2]
        cond3 = gt_image.shape[0]%2 != cube_tmp.shape[1]%2
        cond4 = gt_image.shape[1]%2 != cube_tmp.shape[2]%2
        if cond1 or cond2 or cond3 or cond4:
            if cond1 or cond2:
                msg = "gt image should be larger or equal to input cube frames"
            if cond3 or cond4: 
                msg= "gt image dims should have same parity as input cube dims"
            raise TypeError(msg)
        cond1 = gt_image.shape[0] > cube_tmp.shape[1]
        cond2 = gt_image.shape[1] > cube_tmp.shape[2]
        if cond1 or cond2:
            print("Provided gt image will be cropped to match input cube.")
            gt_image = frame_crop(gt_image, cube_tmp.shape[-1])

    nfrac_opt = []
    drot_opt = []
    thr_opt = []
    npc_opt = []
    nit_opt = []
    stim_opt = []
    master_crop_cube = []
    
    if nproc is None:
        nproc = int(cpu_count()/2)
    print("{:.0f} CPUs will be used".format(nproc))
    
    if nproc>buff:
        nproc_tmp = nproc
        nproc=1
    else:
        nproc_tmp=1
        
    if fiducial_results is not None:
        n_az = 1
        if nfrac_max == 5:
            n_az=3
        elif nfrac_max == 6:
            n_az=6
        fiducial_buff = [fiducial_results[:,i*n_az:(i+1)*n_az] for i in range(buff)]
    
    if nproc>1:
        res = pool_map(nproc, _opt_buff_feve, iterable(range(buff)), cube_tmp, 
                       angle_list, ref_cube_tmp, algo, ncomps, thrs, nits, 
                       strategy, drots, fwhm, fm, gt_image, nfrac_max,
                       thru_corr, psfn, n_br, starphot, plsc, mask_inner_sz, 
                       nproc_tmp, scaling, svd_mode, init_svd, imlib, 
                       interpolation, path, debug, overwrite, smooth, 
                       iterable(fiducial_buff), **rot_options)
        for bb in range(buff):
            nfrac_opt.extend(res[bb][0])
            drot_opt.extend(res[bb][1])
            thr_opt.extend(res[bb][2])
            npc_opt.extend(res[bb][3])
            nit_opt.extend(res[bb][4])
            stim_opt.extend(res[bb][5])
            master_crop_cube.extend(res[bb][6])
    else:
        for bb in range(buff):
            if fiducial_results is not None:
                fiducial_res = fiducial_results[:,bb*n_az:(bb+1)*n_az]
            else:
                fiducial_res = None
            res = _opt_buff_feve(bb, cube_tmp, angle_list, ref_cube_tmp, algo, 
                                 ncomps, thrs, nits, strategy, drots, fwhm, fm, 
                                 gt_image, nfrac_max, thru_corr, psfn, n_br, 
                                 starphot, plsc, mask_inner_sz, nproc_tmp,
                                 scaling, svd_mode, init_svd, imlib, 
                                 interpolation, path, debug, overwrite, smooth,
                                 fiducial_res, **rot_options)
            nfrac_opt.extend(res[0])
            drot_opt.extend(res[1])
            thr_opt.extend(res[2])
            npc_opt.extend(res[3])
            nit_opt.extend(res[4])
            stim_opt.extend(res[5])
            master_crop_cube.extend(res[6])           

    # 3. produce master final frame with np.nanmedian()
    ## place in a master cube with same dimensions as original cube
    ncombi = len(master_crop_cube)
    master_cube = np.zeros([ncombi, cube.shape[-2], cube.shape[-1]])
    cy, cx = frame_center(master_cube[0])
    for mm in range(ncombi):
        cy_tmp, cx_tmp = frame_center(master_crop_cube[mm])
        idx_y0 = int(cy-cy_tmp)
        idx_yN = int(cy+cy_tmp+1)
        idx_x0 = int(cx-cx_tmp)
        idx_xN = int(cx+cx_tmp+1)
        master_cube[mm,idx_y0:idx_yN,idx_x0:idx_xN] = master_crop_cube[mm]
    master_cube[np.where(master_cube==0)] = np.nan   
    master_frame= np.nanmedian(master_cube,axis=0)

    if full_output:
        nfrac_opt_arr = np.array(nfrac_opt)
        drot_opt_arr = np.array(drot_opt)
        thr_opt_arr = np.array(thr_opt)
        npc_opt_arr = np.array(npc_opt)
        nit_opt_arr= np.array(nit_opt)
        stim_opt_arr= np.array(stim_opt)
        return (master_frame, master_cube, nfrac_opt_arr, drot_opt_arr, 
                thr_opt_arr, npc_opt_arr, nit_opt_arr, stim_opt_arr)
    else:
        return master_frame
 
    
def _opt_buff_feve(bb, cube, angle_list, ref_cube=None, algo=pca_annular, 
                   ncomps=[1,2,5], thrs=[0,1], nits=[3,5,10], strategy='ADI', 
                   drots=[0,0.5], fwhm=4, fm='med_stim', gt_image=None, 
                   nfrac_max=6, thru_corr=False, psfn=None, n_br=6, starphot=1, 
                   plsc=1., in_mask_sz=0, nproc=1, scaling=None, 
                   svd_mode='lapack', init_svd='nndsvda', imlib='opencv', 
                   interpolation='lanczos4', path='', debug=False, 
                   overwrite=False, smooth=False, fiducial_results=None,
                   **rot_options):
    # select fm function
    if fm == 'med_stim':
        ffm = np.median
    elif fm == 'mean_stim':
        ffm = np.mean
    elif fm == 'max_stim':
        ffm = np.amax
    elif 'perc' in fm:
        ffm = np.percentile
        perc = int(fm[4:6])
    elif fm == 'ssim':
        if gt_image is None:
            msg = "For ssim figure of merit, provide a Ground Truth image"
            raise TypeError(msg)
        ffm = ssim
        # done in preparation of ssim:
        if not smooth:
            gt_image = gaussian_filter(gt_image, sigma=1.5)
    elif fm == 'fiducial':
        if fiducial_results is None:
            msg = "For 'fiducial' figure of merit, provide fiducial results"
            raise TypeError(msg)
    else:
        msg="fm not recognized. Should be 'med_stim', 'mean_stim' or 'max_stim'"
        raise ValueError(msg)
        
    nfrac_opt = []
    drot_opt = []
    thr_opt = []
    npc_opt = []
    nit_opt = []
    stim_opt = []
    master_crop_cube = []
    
    mask_sz = in_mask_sz+bb

    stim_bb = []
    bb_nfrac = []
    bb_drot = []
    bb_npc = []
    bb_thr = []
    bb_it = []
    bb_frames = []
    
    cy, cx = frame_center(cube[0])
    if algo == pca_annular:
        algo_lab = 'PCAann'
    elif algo == nmf_annular:
        algo_lab = 'NMFann'
    else:
        msg = "Algo not recognized, should be 'pca_annular' or 'nmf_annular'"
        raise ValueError(msg)
        
    if debug:
        fn_tmp = "TMP_{}-{}_{:.0f}bb_{:.0f}frac_{:.0f}nit_{:.1f}thr"
        fn_tmp += "_{:.1f}drot_{:.0f}npc"
        
    if fm == 'fiducial':
        nfrac_max = int(np.amax(fiducial_results[0]))
        
    if nproc == 1:
        ## loop over delta_rot_test
        for delta_rot_tmp in drots:
            ## loop over pcs
            tmp_imgs = np.zeros([len(ncomps), cube.shape[-2], cube.shape[-1]])
            for tt, thr_t in enumerate(thrs):
                for nit in nits:
                    fn = "TMP_{}-{}_{:.0f}bb_{:.0f}frac_{:.0f}nit"
                    fn+= "_{:.1f}thr_{:.1f}drot_imgs.fits"
                    filename_tmp = path+fn.format(algo_lab, strategy, bb, 
                                                  nfrac_max, nit, thr_t, 
                                                  delta_rot_tmp)
    
                    for pp, ncomp in enumerate(ncomps):
                        if fm == 'fiducial':
                            cond_dr = delta_rot_tmp in fiducial_results[4]
                            cond_thr = thr_t in fiducial_results[3]
                            cond_nit = nit in fiducial_results[2]
                            cond_npc = ncomp in fiducial_results[1]
                            necessary = cond_dr & cond_thr & cond_nit & cond_npc
                            if not necessary:
                                continue
                        if debug:
                            fn = path+fn_tmp.format(algo_lab, strategy, bb, 
                                                    nfrac_max, nit, thr_t, 
                                                    delta_rot_tmp, ncomp)
                        else:
                            fn=''
                        if not debug or not isfile(fn+"_norm_stim_cube.fits") or overwrite:
                            res = feves(cube, angle_list, cube_ref=ref_cube, 
                                        ncomp=ncomp, algo=algo, n_it=nit, fwhm=fwhm, 
                                        buff=0, thr=thr_t, n_frac=nfrac_max,
                                        asizes=None, n_segments=None,
                                        thru_corr=thru_corr, strategy=strategy, 
                                        psfn=psfn, n_br=n_br, radius_int=mask_sz, 
                                        delta_rot=delta_rot_tmp, svd_mode=svd_mode, 
                                        init_svd=init_svd, nproc=nproc,
                                        interpolation=interpolation,
                                        imlib=imlib, full_output=True,
                                        interp_order=None, smooth=smooth, 
                                        **rot_options)
                            it_cube_nd = res[-5]
                            norm_stim_cube = res[-4]
                            stim_cube = res[-3]
                            inv_stim_cube = res[-2]
    
                            tmp_imgs[pp] = res[0]
                            it_cube = res[1]
                            sig_cube = res[2]
                            if debug:
                                fn = path+fn_tmp.format(algo_lab, strategy, bb, 
                                                        nfrac_max, nit, thr_t,
                                                        delta_rot_tmp, ncomp)
                                write_fits(fn+"_it_cube.fits", it_cube)
                                write_fits(fn+"_sig_cube.fits", sig_cube)
                                write_fits(fn+"_norm_stim_cube.fits", 
                                           norm_stim_cube)
                                write_fits(fn+"_it_cube_nd.fits", it_cube_nd)
                                write_fits(fn+"_stim_cube.fits", stim_cube)
                                write_fits(fn+"_inv_stim_cube.fits", 
                                           inv_stim_cube)
                        else:
                            it_cube = open_fits(fn+"_it_cube.fits")
                            sig_cube = open_fits(fn+"_sig_cube.fits")
                            norm_stim_cube = open_fits(fn+"_norm_stim_cube.fits")
                            
                        for i in range(norm_stim_cube.shape[0]):
                            stim_bb.append(norm_stim_cube[i])
                            bb_frames.append(it_cube[i])
                        for i in range(nfrac_max):
                            for it in range(nit):
                                bb_nfrac.append(i)
                                bb_drot.append(delta_rot_tmp)
                                bb_npc.append(ncomp)
                                bb_thr.append(thr_t)
                                bb_it.append(nit)
                        
                    ### save if debug
                    if debug and not isfile(filename_tmp):
                        fn = "TMP_{}-{}_{:.0f}bb_{:.0f}frac_{:.0f}nit"
                        fn+="_{:.1f}thr_{:.1f}drot_imgs.fits"
                        write_fits(path+fn.format(algo_lab, strategy, bb, 
                                                  nfrac_max, nit, thr_t, 
                                                  delta_rot_tmp),
                                   tmp_imgs)
    #MP version of above
    else:
        # create list of params
        it_params = []
        for drot_i in drots:
            for tt, thr_i in enumerate(thrs):
                for nit_i in nits:
                    for pp, ncomp_i in enumerate(ncomps):
                        it_params.append((drot_i, thr_i, nit_i, ncomp_i))
                        
        pool_map(nproc, _feves_wrap, iterable(it_params), path, algo_lab, 
                 strategy, bb, nfrac_max, cube, angle_list, ref_cube, algo, 
                 fwhm, thru_corr, psfn, n_br, mask_sz, svd_mode, init_svd, 
                 interpolation, imlib, smooth, overwrite, fm, fiducial_results,
                 **rot_options)
                
        for it_param in it_params:
            delta_rot_tmp, thr_t, nit, ncomp = it_param
            if debug:
                fn = path+fn_tmp.format(algo_lab, strategy, bb, 
                                        nfrac_max, nit, thr_t, 
                                        delta_rot_tmp, ncomp)
            else:
                fn=''
            if isfile(fn+"_it_cube.fits"):
                it_cube = open_fits(fn+"_it_cube.fits")
                sig_cube = open_fits(fn+"_sig_cube.fits")
                norm_stim_cube = open_fits(fn+"_norm_stim_cube.fits")
                
                for i in range(norm_stim_cube.shape[0]):
                    stim_bb.append(norm_stim_cube[i])
                    bb_frames.append(it_cube[i])
                for i in range(nfrac_max):
                    for it in range(nit):
                        bb_nfrac.append(i)
                        bb_drot.append(delta_rot_tmp)
                        bb_npc.append(ncomp)
                        bb_thr.append(thr_t)
                        bb_it.append(nit)
                    
                    
    # FIND optimal params for each annular section
    stim_bb = np.array(stim_bb)
    bb_frames = np.array(bb_frames)
    if fm == 'ssim' and not smooth:
        # done in preparation of ssim:
        bb_frames_g = gaussian_filter(bb_frames, sigma=[0,1.5,1.5])
    else:
        bb_frames_g = bb_frames.copy()
    
    # Default: 1FWHM-wide ann, 6 azimuthal segments
    asize = int(fwhm)
    n_segments = 6
    theta_init = [i*(360/n_segments)/n_segments for i in range(n_segments)]
    n_annuli = int((cy - mask_sz) / asize)
    for i in range(n_segments): # loop on az shifts
        stim_tmp = np.zeros([stim_bb.shape[1],stim_bb.shape[2]])
        frame_tmp = np.zeros_like(stim_tmp) 
        frac_tmp = np.zeros_like(stim_tmp) 
        drot_tmp = np.zeros_like(stim_tmp)
        npc_tmp = np.zeros_like(stim_tmp)
        thr_tmp = np.zeros_like(stim_tmp)
        it_tmp = np.zeros_like(stim_tmp)
        for ann in range(n_annuli):
            res_ann_par = _define_annuli(angle_list, ann, n_annuli, fwhm,
                                         mask_sz, asize, 0, n_segments, 
                                         False, False)
            _, inner_radius, ann_center = res_ann_par

            # Library matrix is created for each segment and scaled if needed
            indices = get_annulus_segments(stim_bb[0], inner_radius, 
                                           asize, n_segments, 
                                           theta_init[i])
            for j in range(n_segments): # actual loop on segments
                yy = indices[j][0]
                xx = indices[j][1]
                if fm == 'fiducial':
                    # first check all values in annulus segment are equal
                    ann_match = True
                    for m in range(1,5):
                        if np.all(fiducial_results[m,i,yy,xx]!=fiducial_results[m,i,yy,xx][0]):
                            ann_match=False
                    if not ann_match:
                        import pdb
                        pdb.set_trace()
                        msg = "Tested annuli and in fiducial results do not match"
                        raise ValueError(msg)
                    for k in range(bb_frames.shape[0]):
                        c_dr = bb_drot[k] == fiducial_results[4,i,yy,xx][0]
                        c_thr = bb_thr[k] == fiducial_results[3,i,yy,xx][0]
                        c_nit = bb_it[k] == fiducial_results[2,i,yy,xx][0]
                        c_npc = bb_npc[k] == fiducial_results[1,i,yy,xx][0]
                        c_nfrac = bb_nfrac[k] == fiducial_results[0,i,yy,xx][0]
                        found_it = c_dr & c_thr & c_nit & c_npc & c_nfrac
                        if found_it:
                            idx_opt = k
                            break
                else:
                    if fm == 'ssim':
                        fm_stim = np.zeros(bb_frames.shape[0])
                        for k in range(bb_frames.shape[0]):
                            fm_stim[k] = ffm(gt_image[yy, xx],
                                             bb_frames_g[k, yy, xx], win_size=1, 
                                             gaussian_weights=False,
                                             use_sample_covariance=False)
                    elif 'perc' in fm:
                        fm_stim = ffm(stim_bb[:, yy, xx], perc, axis=1)
                    else:
                        fm_stim = ffm(stim_bb[:, yy, xx],axis=1)
                    idx_opt = np.argmax(fm_stim, axis=0)
                    
                stim_tmp[yy, xx] = stim_bb[idx_opt, yy, xx].copy()
                frame_tmp[yy, xx] = bb_frames[idx_opt, yy, xx].copy()
                frac_tmp[yy, xx] = bb_nfrac[idx_opt]
                drot_tmp[yy, xx] = bb_drot[idx_opt]                
                npc_tmp[yy, xx] = bb_npc[idx_opt]
                thr_tmp[yy, xx] = bb_thr[idx_opt]
                it_tmp[yy, xx] = bb_it[idx_opt]
    
        nfrac_opt.append(frac_tmp)
        drot_opt.append(drot_tmp)
        thr_opt.append(thr_tmp)
        npc_opt.append(npc_tmp)
        nit_opt.append(it_tmp)
        stim_opt.append(stim_tmp)
        master_crop_cube.append(frame_tmp)    
        
    return (nfrac_opt, drot_opt, thr_opt, npc_opt, nit_opt, stim_opt, 
            master_crop_cube)


def _feves_wrap(it_param, path, algo_lab, strategy, bb, nfrac_max, cube,
                angle_list, ref_cube, algo, fwhm, thru_corr, psfn, n_br, 
                mask_sz, svd_mode, init_svd, interpolation, imlib, 
                smooth, overwrite, fm, fiducial_results, **rot_options):
    
    delta_rot_tmp, thr_t, nit, ncomp = it_param
    
    fn_tmp = "TMP_{}-{}_{:.0f}bb_{:.0f}frac_{:.0f}nit_{:.1f}thr"
    fn_tmp += "_{:.1f}drot_{:.0f}npc"
    
    # filename_tmp = path+fn.format(algo_lab, strategy, bb, nfrac_max, 
    #                               nit, thr_t, delta_rot_tmp)

    #for pp, ncomp in enumerate(ncomps):

    fn = path+fn_tmp.format(algo_lab, strategy, bb, 
                            nfrac_max, nit, thr_t, 
                            delta_rot_tmp, ncomp)

    if fm == 'fiducial':
        cond_dr = delta_rot_tmp in fiducial_results[4]
        cond_thr = thr_t in fiducial_results[3]
        cond_nit = nit in fiducial_results[2]
        cond_npc = ncomp in fiducial_results[1]
        necessary = cond_dr & cond_thr & cond_nit & cond_npc
        if not necessary:
            return None

    #print(fn+"_norm_stim_cube.fits")
    if not isfile(fn+"_norm_stim_cube.fits") or overwrite:
        res = feves(cube, angle_list, cube_ref=ref_cube, ncomp=ncomp, algo=algo, 
                    n_it=nit, fwhm=fwhm, buff=0, thr=thr_t, n_frac=nfrac_max,
                    asizes=None, n_segments=None, thru_corr=thru_corr, 
                    strategy=strategy, psfn=psfn, n_br=n_br, radius_int=mask_sz, 
                    delta_rot=delta_rot_tmp, svd_mode=svd_mode, 
                    init_svd=init_svd, nproc=1, interpolation=interpolation,
                    imlib=imlib, full_output=True, interp_order=None, 
                    smooth=smooth, **rot_options)
        it_cube_nd = res[-5]
        norm_stim_cube = res[-4]
        stim_cube = res[-3]
        inv_stim_cube = res[-2]

        #tmp_imgs[pp] = res[0]
        it_cube = res[1]
        sig_cube = res[2]
        
        fn = path+fn_tmp.format(algo_lab, strategy, bb, 
                                nfrac_max, nit, thr_t,
                                delta_rot_tmp, ncomp)
        write_fits(fn+"_it_cube.fits", it_cube)
        write_fits(fn+"_sig_cube.fits", sig_cube)
        write_fits(fn+"_norm_stim_cube.fits", 
                   norm_stim_cube)
        write_fits(fn+"_it_cube_nd.fits", it_cube_nd)
        write_fits(fn+"_stim_cube.fits", stim_cube)
        write_fits(fn+"_inv_stim_cube.fits", 
                   inv_stim_cube)
        
    ### save if debug
    # if debug and not isfile(filename_tmp) or overwrite:
    #     fn = "TMP_{}-{}_{:.0f}bb_{:.0f}frac_{:.0f}nit"
    #     fn+="_{:.1f}thr_{:.1f}drot_imgs.fits"
    #     write_fits(path+fn.format(algo_lab, strategy, bb, nfrac_max,
    #                               nit, thr_t, delta_rot_tmp),
    #                tmp_imgs) 


def greeds_opt(cube, angle_list, cube_ref=None, fwhm=4, strategy='ADI', 
               algo=pca, fm='med_stim', fm_frac=False, gt_image=None, 
               ncomp_max=10, ncomp_step=1, nits=list(range(10)), thrs=[0,1], 
               drots=[0,1], source_xy=None, add_res=False, thru_corr=False,
               n_br=6, psfn=None, starphot=1., plsc=1., svd_mode='lapack', 
               init_svd='nndsvd', scaling=None, mask_center_px=None, 
               imlib='opencv', interpolation='lanczos4', collapse='median', 
               check_memory=True, nproc=1, full_output=False, verbose=True, 
               weights=None, debug=False, path='', overwrite=True, 
               auto_crop=False, smooth=False, **kwargs_nmf):
    """
    Optimal version of GreeDs (Pairet et al. 2021): iterative PCA or NMF in 
    full frames, with optimization of the parameters based on a chosen figure 
    of merit.
    
    This algorithm is faster than feves_opt, but may not perform as well on
    datasets acquired in temporally and spatially varying speckle.
    
    Note: greeds_opt can only be used in ADI or RDI modes.


    Parameters
    ----------
    cube : str or numpy ndarray, 3d or 4d
        Input cube (ADI or ADI+mSDI). If a string is given, it must correspond
        to the path to the fits file to be opened in memmap mode (for PCA
        incremental of ADI 3d cubes). Note: the algorithm is the most efficient
        if the provided cube is already cropped to ~35 FWHM across. This is 
        automatically done if auto_crop is set to True.
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
    fwhm : float, optional
        Known size of the FHWM in pixels to be used. Default value is 4.
    fm : str, optional {'med_stim','mean_stim','max_stim','perc**_stim','ssim'}
        Figure of merit to reconstruct the optimal image: median, mean, max or
        ** percentile (replace ** with a number between 0 and 100) of the STIM 
        value in each patch, respectively, OR the maximum structural 
        similarity index measure ('ssim') with a reference ground truth image.
    fm_frac: bool, opt
        Whether the figure of merit should be considered in annular segments 
        (True) or in full frame (False).
    gt_image: 2d numpy a rray, optional
        If fm is set to 'ssim', this is the ground truth image to which the 
        processed FEVES images are compared to, when searching for the optimal 
        reduction parameters.
    strategy: str, opt
        Whether to do iterative ADI only ('ADI'), iterative RDI only ('RDI'), 
        or iterative RDI and iterative ADI consecutivey ('RADI'). A reference 
        cube needs to be provided for 'RDI' and 'RADI'.
    ncomp_max : list of int, optional
        Maximum number of principal components to be tested. Values smaller
        than that will be tested per step of ncomp_step.
    ncomp_step: int, opt
        Step for the number of principal components to be tested.
    n_its: list of int, opt
        List for the number of iterations to be used in GreeDs.
    thr: float, tuple of floats or tuple or tuples/lists, opt
        Threshold used to (iteratively) identify significant signals in the 
        final PCA image obtained in each regime. If a float, the same value 
        will be used for both regimes. If a tuple, can either contain 1) two 
        floats corresponding to the resp. thresholds to be used for inner and 
        outer regions; or ii) two lists/tuples of float values which will be 
        tested. This threshold is a minimum pixel intensity in the STIM map 
        computed with the PCA residual cube (Pairet et al. 2019), as expressed 
        in units of max. pixel intensity obtained in the inverse STIM map. 
        Recommended value: 1. But for discs with bright axisymmetric signals, a 
        lower value may be necessary.
    thru_corr: bool, opt
        Whether to correct the significant signals by the algorithmic 
        throughput before subtraction to the original cube at the next
        iteration. 
        Deprecated: If None, this is not performed. If 'psf', throughput is 
        estimated in a classical way with injected psfs. If 'map', the map of
        identified significant signals is used directly for an estimate of the
        2D throughput.
    n_br: int, opt
        Number of branches on which the fake planets are injected to compute 
        the throughput.
    psfn: 2d numpy array, opt
        If thru_corr is set to True, psfn should be a normalised and centered 
        unsaturated psf.
    starphot: float, opt
        Integrated flux of the star in a 1FWHM aperture. Only relevant for
        accurate output contrast curves in full_output mode.
    plsc: float, opt
        Plate scale in arcsec/px. Only relevant for accurate output contrast 
        curves in full_output mode.
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
        radius of the circular mask. If buffer is larger than 0, the effective 
        mask_center_px will be increased accordingly with the outer radius of 
        the annulus, for the different reductions used for edge smoothing.
        
    source_xy : tuple of int, optional
        For ADI-PCA, this triggers a frame rejection in the PCA library, with
        ``source_xy`` as the coordinates X,Y of the center of the annulus where
        the PA criterion is estimated. When ``ncomp`` is a tuple, a PCA grid is
        computed and the S/Ns (mean value in a 1xFWHM circular aperture) of the
        given (X,Y) coordinates are computed.
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
    interpolation : str, optional
        See the documentation of the ``vip_hci.preproc.frame_rotate`` function.
    collapse : {'median', 'mean', 'sum', 'trimmean'}, str optional
        Sets the way of collapsing the frames for producing a final image.
    ifs_collapse_range: str 'all' or tuple of 2 int
        If a tuple, it should contain the first and last channels where the mSDI 
        residual channels will be collapsed (by default collapses all channels).
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
    debug: bool, opt
        Whether to save intermediate fits files in directory given by path.
    path: str, opt
        Where to save intermediate fits files if debug is True. 

    Returns
    -------
    master_frame: 2d numpy ndarray
        2D array obtained from combining images obtained at optimal contrast 
        at each radius.
    master_cube: 3d numpy ndarray
        [full_output=True] Master cube of all nanmedian-combined images used 
        to produce master_frame.
    drot_opt_arr: 1d numpy ndarray
        [full_output=True] Array containing the delta_rot values used to 
        produce each image of master_cube. 
    npc_opt_arr: 1d numpy ndarray
        [full_output=True] Array containing the optimal npc value used to 
        produce each image of master_cube. 
    nit_opt_arr: 1d numpy ndarray
        [full_output=True] Array containing the optimal number of iterations 
        used to produce each image of master_cube. 
    cc_ss_opt_arr: 1d numpy ndarray
        [full_output=True] Array containing the optimal contrast value achieved 
        at each radius (used to select optimal images), when significant signal
        is subtracted from the cube.
    cc_ws_opt_arr: 1d numpy ndarray
        [full_output=True] Array containing the optimal contrast value achieved 
        at each radius (used to select optimal images), when significant signal
        is NOT subtracted from the cube.    
    cc_rad_arr: 1d numpy ndarray
        [full_output=True] Array containing the radii where the contrasts are 
        achieved (used to select optimal images).       
    """
    # formatting
    
    if not isinstance(nits, (tuple,list,range)):
        raise TypeError("nits can only be list, tuple or range")
    if not isinstance(nits, list):
        nits=list(nits)
    if not isinstance(thrs, (tuple,list,range)):
        raise TypeError("thrs can only be list, tuple or range")
    if not isinstance(thrs, list):
        thrs=list(thrs)
    if not isinstance(drots, (tuple,list,range)):
        raise TypeError("drots can only be list, tuple or range")
    if not isinstance(drots, list):
        drots=list(drots)        

    if thru_corr and psfn is None:
        msg = "psf should be provided for throughput correction"
        raise TypeError(msg)

    if mask_center_px is None:
        mask_inner_sz = 0
    else:
        mask_inner_sz = mask_center_px
    
    if strategy == 'ADI':
        ref_cube = None
    elif strategy == 'RDI' or strategy == 'RADI':
        if cube_ref is None:
            raise ValueError("cube_ref should be provided for RDI or RADI")
        ref_cube = cube_ref.copy() 
    else:
        raise ValueError("strategy not recognized: should be ADI, RDI or RADI")
     
    if path:
        if not isdir(path):
            os.makedirs(path)
    
    # crop if needed
    if ref_cube is None:
        ref_cube_tmp = None
    else:
        ref_cube_tmp = ref_cube.copy()
        
    if auto_crop:
        if ref_cube is not None:
            ref_sz = ref_cube_tmp.shape[-1]
        crop_sz = min(int(33*fwhm + 2*mask_inner_sz), ref_sz)
        if not crop_sz%2:
            crop_sz+=1
        if cube.shape[-1] > crop_sz:
            cube_tmp = cube_crop_frames(cube, crop_sz)
        if ref_cube is not None:
            if ref_sz>crop_sz:
                ref_cube_tmp = cube_crop_frames(ref_cube_tmp, crop_sz)
    else:
        cube_tmp = cube.copy()
        
    if gt_image is not None:
        if gt_image.ndim !=2:
            raise TypeError("gt image should be 2d")
        cond1 = gt_image.shape[0] < cube_tmp.shape[1]
        cond2 = gt_image.shape[1] < cube_tmp.shape[2]
        cond3 = gt_image.shape[0]%2 != cube_tmp.shape[1]%2
        cond4 = gt_image.shape[1]%2 != cube_tmp.shape[2]%2
        if cond1 or cond2 or cond3 or cond4:
            if cond1 or cond2:
                msg = "gt image should be larger or equal to input cube frames"
            if cond3 or cond4: 
                msg= "gt image dims should have same parity as input cube dims"
            raise TypeError(msg)
        cond1 = gt_image.shape[0] > cube_tmp.shape[1]
        cond2 = gt_image.shape[1] > cube_tmp.shape[2]
        if cond1 or cond2:
            print("Provided gt image will be cropped to match input cube.")
            gt_image = frame_crop(gt_image, cube_tmp.shape[-1])
    
    if nproc is None:
        nproc = int(cpu_count()/2)
    print("{:.0f} CPUs will be used".format(nproc))
    
    res = _opt_greeds(cube_tmp, angle_list, ref_cube_tmp, algo, ncomp_max, 
                      ncomp_step, thrs, nits, strategy, drots, source_xy, fwhm, 
                      fm, fm_frac, gt_image, thru_corr, psfn, n_br, starphot, 
                      plsc, add_res, mask_inner_sz, nproc, scaling, svd_mode, 
                      init_svd, imlib, interpolation, collapse, path, debug, 
                      overwrite, smooth, **kwargs_nmf)
    drot_opt, thr_opt, npc_opt, nit_opt, nit_tr_opt, stim_opt, master_cube = res 
    master_frame= np.nanmedian(master_cube,axis=0)

    if full_output:
        return (master_frame, master_cube, drot_opt, thr_opt, npc_opt, nit_opt, 
                nit_tr_opt, stim_opt)
    else:
        return master_frame
        
    
def _opt_greeds(cube, angle_list, ref_cube=None, algo=pca, ncomp_max=10, 
                ncomp_step=1, thrs=[0,1], nits=[3,5,10], strategy='ADI', 
                drots=[0,1], source_xy=None, fwhm=4, fm='med_stim', 
                fm_frac=False, gt_image=None, thru_corr=False, psfn=None, 
                n_br=6, starphot=1, plsc=1., add_res=False, mask_sz=0, nproc=1, 
                scaling=None, svd_mode='lapack', init_svd='nndsvda', 
                imlib='opencv', interpolation='lanczos4', collapse='median', 
                path='', debug=False, overwrite=False, smooth=False, 
                **kwargs_nmf):
    # select fm function
    if fm == 'med_stim':
        ffm = np.median
    elif fm == 'mean_stim':
        ffm = np.mean
    elif fm == 'max_stim':
        ffm = np.amax
    elif 'perc' in fm:
        ffm = np.percentile
        perc = int(fm[4:6])
    elif fm == 'ssim':
        if gt_image is None:
            msg = "For ssim figure of merit, provide a Ground Truth image"
            raise TypeError(msg)
        ffm = ssim
        # done in preparation of ssim - OR NOT!
        #if not smooth:
            #gt_image = gaussian_filter(gt_image, sigma=1.5)
    else:
        msg="fm not recognized. Should be 'med_stim', 'mean_stim' or 'max_stim'"
        raise ValueError(msg)
        
    drot_opt = []
    thr_opt = []
    npc_opt = []
    nit_opt = []
    nit_tr_opt = []
    stim_opt = []
    master_crop_cube = []

    ## loop over delta_rot_test
    stim_bb = []
    bb_drot = []
    bb_npc = []
    bb_thr = []
    bb_it = []
    bb_tr_it = []
    bb_frames = []
    
    cy, cx = frame_center(cube[0])
    if algo == pca:
        algo_lab = 'PCA'
    elif algo == nmf:
        algo_lab = 'NMF'
    else:
        msg = "Algo not recognized, should be 'pca' or 'nmf'"
        raise ValueError(msg)
        
    if debug:
        fn_tmp = "TMP_{}-{}_{:.0f}nit_{:.1f}thr"
        fn_tmp += "_{:.1f}drot_{:.0f}npc"

    for delta_rot_tmp in drots:
        for tt, thr_t in enumerate(thrs):
            tmp_imgs = np.zeros([len(nits), cube.shape[-2], cube.shape[-1]])
            for ii, nit in enumerate(nits):
                fn = "TMP_{}-{}_{:.0f}nit"
                fn+="_{:.1f}thr_{:.1f}drot_imgs.fits"
                filename_tmp = path+fn.format(algo_lab, strategy,
                                              nit, thr_t, delta_rot_tmp)

                
                #for pp, ncomp in enumerate(test_npc):
                if debug:
                    fn = path+fn_tmp.format(algo_lab, strategy, nit, thr_t, 
                                            delta_rot_tmp, ncomp_max)
                else:
                    fn=''
                if not debug or not isfile(fn+"_norm_stim_cube.fits") or overwrite:
                    res = pca_it(cube, angle_list, cube_ref=ref_cube, 
                                 algo=algo, mode='Christiaens21', 
                                 ncomp=ncomp_max, ncomp_step=ncomp_step, 
                                 n_it=nit, thr=thr_t, thru_corr=thru_corr, 
                                 strategy=strategy, psfn=psfn, n_br=n_br, 
                                 svd_mode=svd_mode, init_svd=init_svd, 
                                 scaling=scaling, source_xy=source_xy, 
                                 mask_center_px=mask_sz, 
                                 delta_rot=delta_rot_tmp, fwhm=fwhm, 
                                 mask_rdi=None, imlib=imlib, 
                                 interpolation=interpolation, collapse=collapse, 
                                 nproc=nproc, check_memory=True, 
                                 full_output=True, verbose=False, 
                                 add_res=add_res, smooth=smooth, **kwargs_nmf)

                    it_cube_nd = res[-1]
                    norm_stim_cube = res[-2]
                    #stim_cube = res[-3]
                    #inv_stim_cube = res[-2]

                    tmp_imgs[ii] = res[0]
                    it_cube = res[1]
                    sig_cube = res[2]
                    if debug:
                        fn = path+fn_tmp.format(algo_lab, strategy, nit, 
                                                thr_t, delta_rot_tmp, 
                                                ncomp_max)
                        write_fits(fn+"_it_cube.fits", it_cube)
                        write_fits(fn+"_sig_cube.fits", sig_cube)
                        write_fits(fn+"_norm_stim_cube.fits", 
                                   norm_stim_cube)
                        write_fits(fn+"_it_cube_nd.fits", it_cube_nd)
                        #write_fits(fn+"_stim_cube.fits", stim_cube)
                        #write_fits(fn+"_inv_stim_cube.fits", 
                        #           inv_stim_cube)
                else:
                    it_cube = open_fits(fn+"_it_cube.fits")
                    sig_cube = open_fits(fn+"_sig_cube.fits")
                    norm_stim_cube = open_fits(fn+"_norm_stim_cube.fits")
                     
                for i in range(norm_stim_cube.shape[0]):
                    stim_bb.append(norm_stim_cube[i])
                    bb_frames.append(it_cube[i])
                    bb_tr_it.append(i)
                    bb_drot.append(delta_rot_tmp)
                    bb_thr.append(thr_t)
                    bb_it.append(nit)
                for npc in range(1, ncomp_max+1, ncomp_step):
                    for it in range(nit):
                         bb_npc.append(npc)
                         
            ### save if debug
            if debug and not isfile(filename_tmp):
                fn = "TMP_{}-{}_{:.0f}nit"
                fn+="_{:.1f}thr_{:.1f}drot_imgs.fits"
                write_fits(path+fn.format(algo_lab, strategy, nit, thr_t, 
                                          delta_rot_tmp),
                           tmp_imgs)
    
    # FIND optimal params for each annular section
    stim_bb = np.array(stim_bb)
    bb_frames = np.array(bb_frames)
    if fm == 'ssim' and not smooth:
        # done in preparation of ssim OR NOT:
        #bb_frames = gaussian_filter(bb_frames, sigma=[0,1.5,1.5])
        pass
    
    # Default: 1FWHM-wide ann, 6 azimuthal segments
    if fm_frac:
        asize = int(fwhm)
        n_segments = 6
        theta_init = [i*(360/n_segments)/n_segments for i in range(n_segments)]
        n_annuli = int((cy - mask_sz) / asize)
    else:
        asize = int((cy - mask_sz))-1
        n_segments = 1
        theta_init = [0]
        n_annuli = 1
    for i in range(n_segments): # loop on az shifts
        stim_tmp = np.zeros([stim_bb.shape[1],stim_bb.shape[2]])
        frame_tmp = np.zeros_like(stim_tmp)
        drot_tmp = np.zeros_like(stim_tmp)
        npc_tmp = np.zeros_like(stim_tmp)
        thr_tmp = np.zeros_like(stim_tmp)
        it_tmp = np.zeros_like(stim_tmp)
        tr_it_tmp = np.zeros_like(stim_tmp)
        for ann in range(n_annuli):
            res_ann_par = _define_annuli(angle_list, ann, n_annuli, fwhm,
                                         mask_sz, asize, 0, n_segments, False, 
                                         False)
            _, inner_radius, ann_center = res_ann_par

            # Library matrix is created for each segment and scaled if needed
            indices = get_annulus_segments(stim_bb[0], inner_radius, asize, 
                                           n_segments, theta_init[i])
            for j in range(n_segments): # actual loop on segments
                yy = indices[j][0]
                xx = indices[j][1]
                if fm == 'ssim':
                    fm_stim = np.zeros(bb_frames.shape[0])
                    for k in range(bb_frames.shape[0]):
                        fm_stim[k] = ffm(gt_image[yy, xx], bb_frames[k, yy, xx], 
                                         win_size=1, gaussian_weights=False,
                                         use_sample_covariance=False)
                elif 'perc' in fm:
                    fm_stim = ffm(stim_bb[:, yy, xx], perc, axis=1)
                else:
                    fm_stim = ffm(stim_bb[:, yy, xx], axis=1)
                idx_opt = np.argmax(fm_stim, axis=0)
                frame_tmp[yy, xx] = bb_frames[idx_opt, yy, xx].copy()
                drot_tmp[yy, xx] = bb_drot[idx_opt]                
                npc_tmp[yy, xx] = bb_npc[idx_opt]
                thr_tmp[yy, xx] = bb_thr[idx_opt]
                it_tmp[yy, xx] = bb_it[idx_opt]
                tr_it_tmp[yy, xx] = bb_tr_it[idx_opt]
                if fm == 'ssim':
                    stim_tmp[yy, xx] = fm_stim[idx_opt]
                else:
                    stim_tmp[yy, xx] = stim_bb[idx_opt, yy, xx].copy()
                
        drot_opt.append(drot_tmp)
        thr_opt.append(thr_tmp)
        npc_opt.append(npc_tmp)
        nit_opt.append(it_tmp)
        nit_tr_opt.append(tr_it_tmp)
        stim_opt.append(stim_tmp)
        master_crop_cube.append(frame_tmp)    
        
    drot_opt = np.array(drot_opt, dtype=float)
    thr_opt = np.array(thr_opt, dtype=float)
    npc_opt = np.array(npc_opt, dtype=float)
    nit_opt = np.array(nit_opt, dtype=float)
    nit_tr_opt = np.array(nit_tr_opt, dtype=float)
    stim_opt = np.array(stim_opt, dtype=float)
    master_crop_cube = np.array(master_crop_cube, dtype=float)
        
    return (drot_opt, thr_opt, npc_opt, nit_opt, nit_tr_opt, stim_opt, 
            master_crop_cube)
    