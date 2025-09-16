#! /usr/bin/env python
"""
Full-frame PCA algorithm for ADI, (ADI+)RDI and (ADI+)mSDI (IFS data) cubes.

Options :

- *Full-frame PCA*, using the whole cube as the PCA reference library in the
  case of ADI or ADI+mSDI (IFS cube), or a sequence of reference frames
  (reference star) in the case of RDI. For ADI a big data matrix NxP, where N
  is the number of frames and P the number of pixels in a frame is created. Then
  PCA is done through eigen-decomposition of the covariance matrix (~$DD^T$) or
  the SVD of the data matrix. SVD can be calculated using different libraries
  including the fast randomized SVD.

- *Full-frame incremental PCA* for big (larger than available memory) cubes.

.. [AMA12]
   | Amara & Quanz 2012
   | **PYNPOINT: an image processing package for finding exoplanets**
   | *MNRAS, Volume 427, Issue 1, pp. 948-955*
   | `https://arxiv.org/abs/1207.6637
     <https://arxiv.org/abs/1207.6637>`_

.. [CHR19]
   | Christiaens et al. 2019
   | **Separating extended disc features from the protoplanet in PDS 70 using
     VLT/SINFONI**
   | *MNRAS, Volume 486, Issue 4, pp. 5819-5837*
   | `https://arxiv.org/abs/1905.01860
     <https://arxiv.org/abs/1905.01860>`_

.. [HAL09]
   | Halko et al. 2009
   | **Finding structure with randomness: Probabilistic algorithms for
     constructing approximate matrix decompositions**
   | *arXiv e-prints*
   | `https://arxiv.org/abs/0909.4061
     <https://arxiv.org/abs/0909.4061>`_

.. [REN23]
   | Ren 2023
   | **Karhunen-Loève data imputation in high-contrast imaging**
   | *Astronomy & Astrophysics, Volume 679, p. 8*
   | `https://arxiv.org/abs/2308.16912
     <https://arxiv.org/abs/2308.16912>`_

"""

__author__ = "C.A. Gomez Gonzalez, V. Christiaens, T. Bédrine"
__all__ = ["pca", "PCA_Params", "get_pca_coeffs"]

import numpy as np
from multiprocessing import cpu_count
from typing import Tuple, Union, List
from dataclasses import dataclass
from enum import Enum
from .svd import svd_wrapper, SVDecomposer
from .utils_pca import pca_incremental, pca_grid
from ..config import (timing, time_ini, check_enough_memory, Progressbar,
                      check_array)
from ..config.paramenum import (
    SvdMode,
    Adimsdi,
    Interpolation,
    Imlib,
    Collapse,
    ALGO_KEY,
)
from ..config.utils_conf import pool_map, iterable
from ..config.utils_param import setup_parameters, separate_kwargs_dict
from ..preproc.derotation import _find_indices_adi, _compute_pa_thresh
from ..preproc import cube_rescaling_wavelengths as scwave
from ..preproc import (
    cube_derotate,
    cube_collapse,
    cube_subtract_sky_pca,
    check_pa_vector,
    check_scal_vector,
    cube_crop_frames,
)
from ..stats import descriptive_stats
from ..var import (
    frame_center,
    dist,
    prepare_matrix,
    reshape_matrix,
    frame_filter_lowpass,
    cube_filter_lowpass,
    mask_circle,
)


@dataclass
class PCA_Params:
    """
    Set of parameters for the PCA module.

    See function `pca` below for the documentation.
    """

    cube: np.ndarray = None
    angle_list: np.ndarray = None
    cube_ref: np.ndarray = None
    scale_list: np.ndarray = None
    ncomp: Union[Tuple, List, float, int] = 1
    svd_mode: Enum = SvdMode.LAPACK
    scaling: Enum = None
    mask_center_px: int = None
    source_xy: Tuple[int] = None
    delta_rot: int = None
    fwhm: float = 4
    adimsdi: Enum = Adimsdi.SINGLE
    crop_ifs: bool = True
    imlib: Enum = Imlib.VIPFFT
    imlib2: Enum = Imlib.VIPFFT
    interpolation: Enum = Interpolation.LANCZOS4
    collapse: Enum = Collapse.MEDIAN
    collapse_ifs: Enum = Collapse.MEAN
    ifs_collapse_range: Union[str, Tuple[int]] = "all"
    smooth: float = None
    smooth_first_pass: float = None
    mask_rdi: np.ndarray = None
    ref_strategy: str = 'RDI'  # {'RDI', 'ARDI', 'RSDI', 'ARSDI'}
    check_memory: bool = True
    batch: Union[int, float] = None
    nproc: int = 1
    full_output: bool = False
    verbose: bool = True
    weights: np.ndarray = None
    left_eigv: bool = False
    min_frames_pca: int = 10
    max_frames_pca: int = None
    cube_sig: np.ndarray = None
    med_of_npcs: bool = False


def pca(*all_args: List, **all_kwargs: dict):
    """Full-frame PCA algorithm applied to PSF subtraction.

    The reference PSF and the quasi-static speckle pattern are modeled using
    Principal Component Analysis. Depending on the input parameters this PCA
    function can work in ADI, RDI or mSDI (IFS data) mode.

    ADI: the target ``cube`` itself is used to learn the PCs and to obtain a
    low-rank approximation model PSF (star + speckles). Both `cube_ref`` and
    ``scale_list`` must be None. The full-frame ADI-PCA implementation is based
    on [AMA12]_ and [SOU12]_. If ``batch`` is provided then the cube is
    processed with incremental PCA as described in [GOM17]_.

    (ADI+)RDI: if a reference cube is provided (``cube_ref``), its PCs are used
    to reconstruct the target frames to obtain the model PSF (star + speckles).

    (ADI+)mSDI (IFS data): if a scaling vector is provided (``scale_list``) and
    the cube is a 4d array [# channels, # adi-frames, Y, X], it's assumed it
    contains several multi-spectral frames acquired in pupil-stabilized mode.
    A single or two stages PCA can be performed, depending on ``adimsdi``, as
    explained in [CHR19]_.

    Parameters
    ----------
    all_args: list, optional
        Positional arguments for the PCA algorithm. Full list of parameters
        below.
    all_kwargs: dictionary, optional
        Mix of keyword arguments that can initialize a PCA_Params and the
        optional 'rot_options' dictionary (with keyword values ``border_mode``,
        ``mask_val``, ``edge_blend``, ``interp_zeros``, ``ker``; see docstring
        of ``vip_hci.preproc.frame_rotate``). Can also contain a PCA_Params
        dictionary named `algo_params`.

    PCA parameters
    --------------
    cube : str or numpy ndarray, 3d or 4d
        Input cube (ADI or ADI+mSDI). If 4D, the first dimension should be
        spectral. If a string is given, it must correspond to the path to the
        fits file to be opened in memmap mode (incremental PCA-ADI of 3D cubes
        only).
    angle_list : numpy ndarray, 1d
        Vector of derotation angles to align North up in your images.
    cube_ref : 3d or 4d numpy ndarray, or list of 3D numpy ndarray, optional
        Reference library cube for Reference Star Differential Imaging. Should
        be 3D, except if input cube is 4D and no scale_list is provided,
        reference cube can then either be 4D or a list of 3D cubes (i.e.
        providing the reference cube for each individual spectral cube).
    scale_list : numpy ndarray, 1d, optional
        If provided, triggers mSDI reduction. These should be the scaling
        factors used to re-scale the spectral channels and align the speckles
        in case of IFS data (ADI+mSDI cube). Usually, the
        scaling factors are the last channel wavelength divided by the
        other wavelengths in the cube (more thorough approaches can be used
        to get the scaling factors, e.g. with
        ``vip_hci.preproc.find_scal_vector``).
    ncomp : int, float, tuple of int/None, or list, optional
        How many PCs are used as a lower-dimensional subspace to project the
        target frames.

        * ADI (``cube`` is a 3d array): if an int is provided, ``ncomp`` is the
        number of PCs extracted from ``cube`` itself. If ``ncomp`` is a float
        in the interval [0, 1] then it corresponds to the desired cumulative
        explained variance ratio (the corresponding number of components is
        estimated). If ``ncomp`` is a tuple of two integers, then it
        corresponds to an interval of PCs in which final residual frames are
        computed (optionally, if a tuple of 3 integers is passed, the third
        value is the step). If ``ncomp`` is a list of int, these will be used to
        calculate residual frames. When ``ncomp`` is a tuple or list, and
        ``source_xy`` is not None, then the S/Ns (mean value in a 1xFWHM
        circular aperture) of the given (X,Y) coordinates are computed.

        * ADI+RDI (``cube`` and ``cube_ref`` are 3d arrays): ``ncomp`` is the
        number of PCs obtained from ``cube_ref``. If ``ncomp`` is a tuple,
        then it corresponds to an interval of PCs (obtained from ``cube_ref``)
        in which final residual frames are computed. If ``ncomp`` is a list of
        int, these will be used to calculate residual frames. When ``ncomp`` is
        a tuple or list, and ``source_xy`` is not None, then the S/Ns (mean
        value in a 1xFWHM circular aperture) of the given (X,Y) coordinates are
        computed.

        * ADI or ADI+RDI (``cube`` is a 4d array): same input format allowed as
        above. If ``ncomp`` is a list with the same length as the number of
        channels, each element of the list will be used as ``ncomp`` value
        (be it int, float or tuple) for each spectral channel. If not a
        list or a list with a different length as the number of spectral
        channels, these will be tested for all spectral channels respectively.

        * ADI+mSDI (``cube`` is a 4d array and ``adimsdi="single"``): ``ncomp``
        is the number of PCs obtained from the whole set of frames
        (n_channels * n_adiframes). If ``ncomp`` is a float in the interval
        (0, 1] then it corresponds to the desired CEVR, and the corresponding
        number of components will be estimated. If ``ncomp`` is a tuple, then
        it corresponds to an interval of PCs in which final residual frames
        are computed. If ``ncomp`` is a list of int, these will be used to
        calculate residual frames. When ``ncomp`` is a tuple or list, and
        ``source_xy`` is not None, then the S/Ns (mean value in a 1xFWHM
        circular aperture) of the given (X,Y) coordinates are computed.

        * ADI+mSDI  (``cube`` is a 4d array and ``adimsdi="double"``): ``ncomp``
        must be a tuple, where the first value is the number of PCs obtained
        from each multi-spectral frame (if None then this stage will be
        skipped and the spectral channels will be combined without
        subtraction); the second value sets the number of PCs used in the
        second PCA stage, ADI-like using the residuals of the first stage (if
        None then the second PCA stage is skipped and the residuals are
        de-rotated and combined).

    svd_mode : Enum, see `vip_hci.config.paramenum.SvdMode`
        Switch for the SVD method/library to be used.
    scaling : Enum, or tuple of Enum, see `vip_hci.config.paramenum.Scaling`
        Pixel-wise scaling mode using ``sklearn.preprocessing.scale``
        function. If set to None, the input matrix is left untouched. In the
        case of PCA-SADI in 2 steps, this can be a tuple of 2 values,
        corresponding to the scaling for each of the 2 steps of PCA.
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
        Default is 1 (excludes 1xFWHM on each side of the considered frame).
    fwhm : float, list or 1d numpy array, optional
        Known size of the FWHM in pixels to be used. Default value is 4.
        Can be a list or 1d numpy array for a 4d input cube with no scale_list.
    adimsdi : Enum, see `vip_hci.config.paramenum.Adimsdi`
        Changes the way the 4d cubes (ADI+mSDI) are processed. Basically it
        determines whether a single or double pass PCA is going to be computed.
    crop_ifs: bool, optional
        [adimsdi='single'] If True cube is cropped at the moment of frame
        rescaling in wavelength. This is recommended for large FOVs such as the
        one of SPHERE, but can remove significant amount of information close to
        the edge of small FOVs (e.g. SINFONI).
    imlib : Enum, see `vip_hci.config.paramenum.Imlib`
        See the documentation of ``vip_hci.preproc.frame_rotate``.
    imlib2 : Enum, see `vip_hci.config.paramenum.Imlib`
        See the documentation of ``vip_hci.preproc.cube_rescaling_wavelengths``.
    interpolation : Enum, see `vip_hci.config.paramenum.Interpolation`
        See the documentation of the ``vip_hci.preproc.frame_rotate`` function.
    collapse : Enum, see `vip_hci.config.paramenum.Collapse`
        Sets how temporal residual frames should be combined to produce an
        ADI image.
    collapse_ifs : Enum, see `vip_hci.config.paramenum.Collapse`
        Sets how spectral residual frames should be combined to produce an
        mSDI image.
    ifs_collapse_range: str 'all' or tuple of 2 int
        If a tuple, it should contain the first and last channels where the mSDI
        residual channels will be collapsed (by default collapses all channels).
    smooth: float or None, optional
        Gaussian kernel size to use to smooth the images. None by default (no
        smoothing). Can be used when pca is used within NEGFC with the Hessian
        figure of merit.
    smooth_first_pass: float or None, optional
        [adimsdi='double'] For 4D cubes with requested PCA-SADI processing in 2
        steps, the Gaussian kernel size to use to smooth the images of the first
        pass before performing the second pass. None by default (no smoothing).
    mask_rdi: tuple of two numpy array or one signle 2d numpy array, opt
        If provided, binary mask(s) will be used either in RDI mode or in
        ADI+mSDI (2 steps) mode. If two masks are provided, they will the anchor
        and boat regions, respectively, following the denominations in [REN23]_.
        If only one mask is provided, it will be used as the anchor, and the
        boat images will not be masked (i.e., full frames used).
    ref_strategy: str, opt {'RDI', 'ARDI'}
        [cube_ref is not None] Indicates the strategy to be adopted when a
        reference cube is provided. By default, RDI is done - i.e. the science
        images are not used in the PCA library. If set to 'ARDI', the PCA
        library is made of both the science and reference images.
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
    left_eigv : bool, optional
        Whether to use rather left or right singularvectors
        This mode is not compatible with 'mask_rdi' and 'batch'
    min_frames_pca : int, optional
        Minimum number of frames required in the PCA library. An error is raised
        if less than such number of frames can be found.
    cube_sig: numpy ndarray, opt
        Cube with estimate of significant authentic signals. If provided, this
        will be subtracted before projecting considering the science cube as
        reference cube.
    med_of_npcs: bool, opt
        [ncomp is tuple or list] Whether to consider the median image of the
        list of images obtained with a list or tuple of ncomp values.

    Return
    -------
    final_residuals_cube : List of numpy ndarray
        [(ncomp is tuple or list) & (med_of_npcs=False or source_xy != None)]
        List of residual final PCA frames obtained for a grid of PC values.
    frame : numpy ndarray
        [(ncomp is scalar) or (source_xy != None)] 2D array, median combination
        of the de-rotated/re-scaled residuals cube.
        [(ncomp is tuple or list) & (med_of_npcs=True)] median of images
        obtained with different ncomp values.
    pcs : numpy ndarray
        [full_output=True, source_xy=None] Principal components. Valid for
        ADI cubes 3D or 4D (i.e. ``scale_list=None``). This is also returned
        when ``batch`` is not None (incremental PCA).
    recon_cube, recon : numpy ndarray
        [full_output=True] Reconstructed cube. Valid for ADI cubes 3D or 4D
        (i.e. ``scale_list=None``)
    residuals_cube : numpy ndarray
        [full_output=True] Residuals cube. Valid for ADI cubes 3D or 4D
        (i.e. ``scale_list=None``)
    residuals_cube_ : numpy ndarray
        [full_output=True] Derotated residuals cube. Valid for ADI cubes 3D or
        4D (i.e. ``scale_list=None``)
    residuals_cube_channels : numpy ndarray
        [full_output=True, adimsdi='double'] Residuals for each multi-spectral
        cube. Valid for ADI+mSDI (4D) cubes (when ``scale_list`` is provided)
    residuals_cube_channels_ : numpy ndarray
        [full_output=True, adimsdi='double'] Derotated final residuals. Valid
        for ADI+mSDI (4D) cubes (when ``scale_list`` is provided)
    cube_allfr_residuals : numpy ndarray
        [full_output=True, adimsdi='single']  Residuals cube (of the big cube
        with channels and time processed together). Valid for ADI+mSDI (4D)
        cubes (when ``scale_list`` is provided)
    cube_desc_residuals : numpy ndarray
        [full_output=True, adimsdi='single'] Residuals cube after de-scaling the
        spectral frames to their original scale. Valid for ADI+mSDI (4D) (when
        ``scale_list`` is provided).
    cube_adi_residuals : numpy ndarray
        [full_output=True, adimsdi='single'] Residuals cube after de-scaling the
        spectral frames to their original scale and collapsing the channels.
        Valid for ADI+mSDI (4D) (when ``scale_list`` is provided).
    ifs_adi_frames : numpy ndarray
        [full_output=True, 4D input cube, ``scale_list=None``] This is the cube
        of individual ADI reductions for each channel of the IFS cube.
    medians : numpy ndarray
        [full_output=True, source_xy=None, batch!=None] Median images of each
        batch, in incremental PCA, for 3D input cubes only.

    """
    # Separating the parameters of the ParamsObject from optional rot_options

    class_params, rot_options = separate_kwargs_dict(
        initial_kwargs=all_kwargs, parent_class=PCA_Params
    )

    # Extracting the object of parameters (if any)
    algo_params = None
    if ALGO_KEY in rot_options.keys():
        algo_params = rot_options[ALGO_KEY]
        del rot_options[ALGO_KEY]

    if algo_params is None:
        algo_params = PCA_Params(*all_args, **class_params)

    # by default, interpolate masked area before derotation if a mask is used
    if algo_params.mask_center_px and len(rot_options) == 0:
        rot_options['mask_val'] = 0
        rot_options['ker'] = 1
        rot_options['interp_zeros'] = True

    start_time = time_ini(algo_params.verbose)

    if algo_params.batch is None:
        check_array(algo_params.cube, (3, 4), msg="cube")
    else:
        if not isinstance(algo_params.cube, (str, np.ndarray)):
            raise TypeError(
                "`cube` must be a numpy (3d or 4d) array or a str "
                "with the full path on disk"
            )

    if algo_params.left_eigv:
        if (
            algo_params.batch is not None
            or algo_params.mask_rdi is not None
            or algo_params.cube_ref is not None
        ):
            raise NotImplementedError(
                "left_eigv is not compatible with 'mask_rdi' nor 'batch'"
            )

    # checking memory (if in-memory numpy array is provided)
    if not isinstance(algo_params.cube, str):
        input_bytes = (
            algo_params.cube_ref.nbytes
            if algo_params.cube_ref is not None
            else algo_params.cube.nbytes
        )
        mem_msg = (
            "Set check_memory=False to override this memory check or "
            "set `batch` to run incremental PCA (valid for ADI or "
            "ADI+mSDI single-pass)"
        )
        check_enough_memory(
            input_bytes,
            1.0,
            raise_error=algo_params.check_memory,
            error_msg=mem_msg,
            verbose=algo_params.verbose,
        )

    if algo_params.nproc is None:
        algo_params.nproc = cpu_count() // 2  # Hyper-threading doubles # cores

    # All possible outputs for any PCA usage must be pre-declared to None
    # Default possible outputs

    (
        frame,
        final_residuals_cube,
        pclist,
        pcs,
        medians,
        recon,
        residuals_cube,
        residuals_cube_,
    ) = (None for _ in range(8))

    # Full_output/cube dimension dependant variables

    (
        table,
        cube_allfr_residuals,
        cube_adi_residuals,
        residuals_cube_channels,
        residuals_cube_channels_,
        ifs_adi_frames,
    ) = (None for _ in range(6))

    # ADI + mSDI. Shape of cube: (n_channels, n_adi_frames, y, x)
    # isinstance(cube, np.ndarray) and cube.ndim == 4:
    if algo_params.scale_list is not None:
        add_params = {"start_time": start_time}
        if algo_params.cube_ref is not None:
            if algo_params.cube_ref.ndim != 4:
                msg = "Ref cube has wrong format for 4d input cube"
                raise TypeError(msg)
            if 'A' in algo_params.ref_strategy:  # e.g. 'ARSDI'
                add_params["ref_strategy"] = 'ARSDI'  # uniformize
                if algo_params.adimsdi == Adimsdi.SINGLE:
                    cube_ref = np.concatenate((algo_params.cube,
                                               algo_params.cube_ref), axis=1)
                    add_params["cube_ref"] = cube_ref
            else:
                add_params["ref_strategy"] = 'RSDI'

        if algo_params.adimsdi == Adimsdi.DOUBLE:
            func_params = setup_parameters(
                params_obj=algo_params, fkt=_adimsdi_doublepca, **add_params
            )
            res_pca = _adimsdi_doublepca(
                **func_params,
                **rot_options,
            )
            residuals_cube_channels, residuals_cube_channels_, frame = res_pca
        elif algo_params.adimsdi == Adimsdi.SINGLE:
            func_params = setup_parameters(
                params_obj=algo_params, fkt=_adimsdi_singlepca, **add_params
            )
            res_pca = _adimsdi_singlepca(
                **func_params,
                **rot_options,
            )
            if np.isscalar(algo_params.ncomp):
                cube_allfr_residuals, cube_desc_residuals = res_pca[:2]
                cube_adi_residuals, frame = res_pca[2:]
            elif isinstance(algo_params.ncomp, (tuple, list)):
                if algo_params.source_xy is None:
                    if algo_params.full_output:
                        final_residuals_cube, pclist = res_pca
                    else:
                        final_residuals_cube = res_pca
                else:
                    final_residuals_cube, frame, table, _ = res_pca
        else:
            raise ValueError("`adimsdi` mode not recognized")

    # 4D cube, but no mSDI desired
    elif algo_params.cube.ndim == 4:
        nch, nz, ny, nx = algo_params.cube.shape
        ifs_adi_frames = np.zeros([nch, ny, nx])
        if not isinstance(algo_params.ncomp, list):
            ncomp = [algo_params.ncomp] * nch
        elif len(algo_params.ncomp) != nch:
            nnpc = len(algo_params.ncomp)
            ifs_adi_frames = np.zeros([nch, nnpc, ny, nx])
            ncomp = [algo_params.ncomp] * nch
        else:
            ncomp = algo_params.ncomp
        if np.isscalar(algo_params.fwhm):
            algo_params.fwhm = [algo_params.fwhm] * nch

        pcs = []
        recon = []
        residuals_cube = []
        residuals_cube_ = []
        final_residuals_cube = []
        recon_cube = []
        medians = []
        table = []
        pclist = []
        grid_case = False

        # ADI or RDI
        for ch in range(nch):
            add_params = {
                "start_time": start_time,
                "cube": algo_params.cube[ch],
                "ncomp": ncomp[ch],  # algo_params.ncomp[ch],
                "fwhm": algo_params.fwhm[ch],
                "full_output": True,
            }

            # RDI
            if algo_params.cube_ref is not None:
                if algo_params.cube_ref[ch].ndim != 3:
                    msg = "Ref cube has wrong format for 4d input cube"
                    raise TypeError(msg)
                if algo_params.ref_strategy == 'RDI':
                    add_params["cube_ref"] = algo_params.cube_ref[ch]
                elif algo_params.ref_strategy == 'ARDI':
                    cube_ref = np.concatenate((algo_params.cube[ch],
                                               algo_params.cube_ref[ch]))
                    add_params["cube_ref"] = cube_ref
                else:
                    msg = "ref_strategy argument not recognized."
                    msg += "Should be 'RDI' or 'ARDI'"
                    raise TypeError(msg)

            func_params = setup_parameters(
                params_obj=algo_params, fkt=_adi_rdi_pca, **add_params
            )
            res_pca = _adi_rdi_pca(
                **func_params,
                **rot_options,
            )

            if algo_params.batch is None:
                if algo_params.source_xy is not None:
                    # PCA grid, computing S/Ns
                    if isinstance(ncomp[ch], (tuple, list)):
                        final_residuals_cube.append(res_pca[0])
                        ifs_adi_frames[ch] = res_pca[1]
                        table.append(res_pca[2])
                    # full-frame PCA with rotation threshold
                    else:
                        recon_cube.append(res_pca[0])
                        residuals_cube.append(res_pca[1])
                        residuals_cube_.append(res_pca[2])
                        ifs_adi_frames[ch] = res_pca[-1]
                else:
                    # PCA grid
                    if isinstance(ncomp[ch], (tuple, list)):
                        ifs_adi_frames[ch] = res_pca[0]
                        pclist.append(res_pca[1])
                        grid_case = True
                    # full-frame standard PCA
                    else:
                        pcs.append(res_pca[0])
                        recon.append(res_pca[1])
                        residuals_cube.append(res_pca[2])
                        residuals_cube_.append(res_pca[3])
                        ifs_adi_frames[ch] = res_pca[-1]
            # full-frame incremental PCA
            else:
                ifs_adi_frames[ch] = res_pca[0]
                pcs.append(res_pca[2])
                medians.append(res_pca[3])

        if grid_case:
            for i in range(len(ncomp[0])):
                frame = cube_collapse(ifs_adi_frames[:, i],
                                      mode=algo_params.collapse_ifs)
                final_residuals_cube.append(frame)
        else:
            frame = cube_collapse(ifs_adi_frames,
                                  mode=algo_params.collapse_ifs)

        # convert to numpy arrays when relevant
        if len(pcs) > 0:
            pcs = np.array(pcs)
        if len(recon) > 0:
            recon = np.array(recon)
        if len(residuals_cube) > 0:
            residuals_cube = np.array(residuals_cube)
        if len(residuals_cube_) > 0:
            residuals_cube_ = np.array(residuals_cube_)
        if len(final_residuals_cube) > 0:
            final_residuals_cube = np.array(final_residuals_cube)
        if len(recon_cube) > 0:
            recon_cube = np.array(recon_cube)
        if len(medians) > 0:
            medians = np.array(medians)

    # 3D RDI or ADI. Shape of cube: (n_adi_frames, y, x)
    else:
        add_params = {
            "start_time": start_time,
            "full_output": True,
        }

        if algo_params.cube_ref is not None and algo_params.batch is not None:
            raise ValueError("RDI not compatible with batch mode")
        elif algo_params.cube_ref is not None:
            if algo_params.ref_strategy == 'ARDI':
                algo_params.cube_ref = np.concatenate((algo_params.cube,
                                                       algo_params.cube_ref))
            elif algo_params.ref_strategy != 'RDI':
                msg = "ref_strategy argument not recognized."
                msg += "Should be 'RDI' or 'ARDI'"
                raise TypeError(msg)

        func_params = setup_parameters(params_obj=algo_params,
                                       fkt=_adi_rdi_pca, **add_params)

        res_pca = _adi_rdi_pca(**func_params, **rot_options)

        if algo_params.batch is None:
            if algo_params.source_xy is not None:
                # PCA grid, computing S/Ns
                if isinstance(algo_params.ncomp, (tuple, list)):
                    if algo_params.full_output:
                        final_residuals_cube, frame, table, _ = res_pca
                    else:
                        # returning only the optimal residual
                        frame = res_pca[1]
                # full-frame PCA with rotation threshold
                else:
                    recon_cube, residuals_cube, residuals_cube_, frame = res_pca
            else:
                # PCA grid
                if isinstance(algo_params.ncomp, (tuple, list)):
                    final_residuals_cube, pclist = res_pca
                # full-frame standard PCA
                else:
                    pcs, recon, residuals_cube, residuals_cube_, frame = res_pca
        # full-frame incremental PCA
        else:
            frame, _, pcs, medians = res_pca

    # else:
    #     raise RuntimeError(
    #        "Only ADI, ADI+RDI and ADI+mSDI observing techniques are supported"
    #     )

    # --------------------------------------------------------------------------
    # Returns for each case (ADI, ADI+RDI and ADI+mSDI) and combination of
    # parameters: full_output, source_xy, batch, ncomp
    # --------------------------------------------------------------------------
    # If requested (except when source_xy is not None), return median image
    # cond_s = algo_params.source_xy is None
    if final_residuals_cube is not None and algo_params.med_of_npcs:
        final_residuals_cube = np.median(final_residuals_cube, axis=0)

    isarr = isinstance(algo_params.cube, np.ndarray)
    if isarr and algo_params.scale_list is not None:
        # ADI+mSDI double-pass PCA
        if algo_params.adimsdi == Adimsdi.DOUBLE:
            if algo_params.full_output:
                return frame, residuals_cube_channels, residuals_cube_channels_
            else:
                return frame

        elif algo_params.adimsdi == Adimsdi.SINGLE:
            # ADI+mSDI single-pass PCA
            if np.isscalar(algo_params.ncomp):
                if algo_params.full_output:
                    return (frame, cube_allfr_residuals, cube_desc_residuals,
                            cube_adi_residuals)
                else:
                    return frame
            # ADI+mSDI single-pass PCA grid
            elif isinstance(algo_params.ncomp, (tuple, list)):
                if algo_params.source_xy is None:
                    if algo_params.full_output:
                        return final_residuals_cube, pclist
                    else:
                        return final_residuals_cube
                else:
                    if algo_params.full_output:
                        return final_residuals_cube, frame, table
                    else:
                        return frame
            else:
                msg = "ncomp value should only be a float, an int or a tuple of"
                msg += f" those, not a {type(algo_params.ncomp)}."
                raise ValueError(msg)
        else:
            msg = f"ADIMSDI value should only be {Adimsdi.SINGLE} or"
            msg += f" {Adimsdi.DOUBLE}."
            raise ValueError(msg)

    # ADI and ADI+RDI (3D or 4D)
    elif isinstance(algo_params.cube, str) or algo_params.scale_list is None:
        if algo_params.source_xy is None and algo_params.full_output:
            # incremental PCA
            if algo_params.batch is not None:
                final_res = [frame, pcs, medians]
            else:
                # PCA grid
                if isinstance(algo_params.ncomp, (tuple, list)):
                    final_res = [final_residuals_cube, pclist]
                # full-frame standard PCA or ADI+RDI
                else:
                    final_res = [frame, pcs, recon, residuals_cube,
                                 residuals_cube_]
            if algo_params.cube.ndim == 4:
                final_res.append(ifs_adi_frames)
            return tuple(final_res)
        elif algo_params.source_xy is not None and algo_params.full_output:
            # PCA grid, computing S/Ns
            if isinstance(algo_params.ncomp, (tuple, list)):
                final_res = [final_residuals_cube, frame, table]
            # full-frame PCA with rotation threshold
            else:
                final_res = [frame, recon_cube, residuals_cube, residuals_cube_]
            if algo_params.cube.ndim == 4:
                final_res.append(ifs_adi_frames)
            return tuple(final_res)
        elif algo_params.source_xy is not None:
            return frame
        elif not algo_params.full_output:
            # PCA grid
            if isinstance(algo_params.ncomp, (tuple, list)):
                return final_residuals_cube
            # full-frame standard PCA or ADI+RDI
            else:
                return frame

    else:
        msg = "cube value should only be a str or a numpy.ndarray, not a "
        msg += f"{type(algo_params.cube)}."
        raise ValueError(msg)


def _adi_rdi_pca(
    cube,
    cube_ref,
    angle_list,
    ncomp,
    batch,
    source_xy,
    delta_rot,
    fwhm,
    scaling,
    mask_center_px,
    svd_mode,
    imlib,
    interpolation,
    collapse,
    verbose,
    start_time,
    nproc,
    full_output,
    weights=None,
    mask_rdi=None,
    cube_sig=None,
    left_eigv=False,
    min_frames_pca=10,
    max_frames_pca=None,
    smooth=None,
    **rot_options,
):
    """Handle the ADI or ADI+RDI PCA post-processing."""
    (
        frame,
        pcs,
        recon,
        residuals_cube,
        residuals_cube_,
    ) = (None for _ in range(5))
    # Full/Single ADI processing, incremental PCA
    if batch is not None:
        result = pca_incremental(
            cube,
            angle_list,
            batch=batch,
            ncomp=ncomp,
            collapse=collapse,
            verbose=verbose,
            full_output=full_output,
            start_time=start_time,
            weights=weights,
            nproc=nproc,
            imlib=imlib,
            interpolation=interpolation,
            **rot_options,
        )
        return result

    else:
        # Full/Single ADI processing
        n, y, x = cube.shape

        angle_list = check_pa_vector(angle_list)
        if not n == angle_list.shape[0]:
            raise ValueError(
                "`angle_list` vector has wrong length. It must "
                "equal the number of frames in the cube"
            )

        if not np.isscalar(ncomp) and not isinstance(ncomp, (tuple, list)):
            msg = "`ncomp` must be an int, float, tuple or list in the ADI case"
            raise TypeError(msg)

        if np.isscalar(ncomp):
            if cube_ref is not None:
                nref = cube_ref.shape[0]
            else:
                nref = n
            if isinstance(ncomp, int) and ncomp > nref:
                ncomp = min(ncomp, nref)
                print(
                    "Number of PCs too high (max PCs={}), using {} PCs "
                    "instead.".format(nref, ncomp)
                )
            elif ncomp <= 0:
                msg = "Number of PCs too low. It should be > 0."
                raise ValueError(msg)
            if mask_rdi is None:
                if source_xy is None:
                    residuals_result = _project_subtract(
                        cube,
                        cube_ref,
                        ncomp,
                        scaling,
                        mask_center_px,
                        svd_mode,
                        verbose,
                        full_output,
                        cube_sig=cube_sig,
                        left_eigv=left_eigv,
                    )
                    if verbose:
                        timing(start_time)
                    if full_output:
                        residuals_cube = residuals_result[0]
                        reconstructed = residuals_result[1]
                        V = residuals_result[2]
                        pcs = reshape_matrix(V, y, x) if not left_eigv else V.T
                        recon = reshape_matrix(reconstructed, y, x)
                    else:
                        residuals_cube = residuals_result

                # A rotation threshold is applied
                else:
                    if delta_rot is None or fwhm is None:
                        msg = "Delta_rot or fwhm parameters missing. Needed for"
                        msg += "PA-based rejection of frames from the library"
                        raise TypeError(msg)
                    nfrslib = []
                    residuals_cube = np.zeros_like(cube)
                    recon_cube = np.zeros_like(cube)
                    yc, xc = frame_center(cube[0], False)
                    x1, y1 = source_xy
                    ann_center = dist(yc, xc, y1, x1)
                    pa_thr = _compute_pa_thresh(ann_center, fwhm, delta_rot)
                    max_fr = max_frames_pca
                    if max_frames_pca is not None:
                        truncate = True
                    else:
                        truncate = False

                    for frame in range(n):
                        ind = _find_indices_adi(angle_list, frame, pa_thr,
                                                truncate=truncate,
                                                max_frames=max_fr)

                        res_result = _project_subtract(
                            cube,
                            cube_ref,
                            ncomp,
                            scaling,
                            mask_center_px,
                            svd_mode,
                            verbose,
                            full_output,
                            ind,
                            frame,
                            cube_sig=cube_sig,
                            left_eigv=left_eigv,
                            min_frames_pca=min_frames_pca,
                        )
                        if full_output:
                            nfrslib.append(res_result[0])
                            residual_frame = res_result[1]
                            recon_frame = res_result[2]
                            residuals_cube[frame] = residual_frame.reshape((y,
                                                                            x))
                            recon_cube[frame] = recon_frame.reshape((y, x))
                        else:
                            nfrslib.append(res_result[0])
                            residual_frame = res_result[1]
                            residuals_cube[frame] = residual_frame.reshape((y,
                                                                            x))

                    # number of frames in library printed for each ann. quadrant
                    if verbose:
                        descriptive_stats(nfrslib, verbose=verbose,
                                          label="Size LIB: ")
            else:
                residuals_result = cube_subtract_sky_pca(
                    cube, cube_ref, mask_rdi, ncomp=ncomp, full_output=True
                )
                residuals_cube = residuals_result[0]
                pcs = residuals_result[2]
                recon = residuals_result[-1]
            residuals_cube_ = cube_derotate(
                residuals_cube,
                angle_list,
                nproc=nproc,
                imlib=imlib,
                interpolation=interpolation,
                **rot_options,
            )
            frame = cube_collapse(residuals_cube_, mode=collapse, w=weights)
            if smooth is not None:
                frame = frame_filter_lowpass(frame, mode='gauss',
                                             fwhm_size=smooth)
            if mask_center_px:
                residuals_cube_ = mask_circle(residuals_cube_, mask_center_px)
                frame = mask_circle(frame, mask_center_px)
            if verbose:
                print("Done de-rotating and combining")
                timing(start_time)
            if source_xy is not None:
                if full_output:
                    return (recon_cube,
                            residuals_cube,
                            residuals_cube_,
                            frame)
                else:
                    return frame
            else:
                if full_output:
                    return (pcs,
                            recon,
                            residuals_cube,
                            residuals_cube_,
                            frame)
                else:
                    return frame

        # When ncomp is a tuple, pca_grid is called
        else:
            gridre = pca_grid(
                cube,
                angle_list,
                fwhm,
                range_pcs=ncomp,
                source_xy=source_xy,
                cube_ref=cube_ref,
                mode="fullfr",
                svd_mode=svd_mode,
                scaling=scaling,
                mask_center_px=mask_center_px,
                fmerit="mean",
                collapse=collapse,
                verbose=verbose,
                full_output=full_output,
                debug=False,
                plot=verbose,
                start_time=start_time,
                weights=weights,
                nproc=nproc,
                imlib=imlib,
                interpolation=interpolation,
                **rot_options,
            )
            return gridre


def _adimsdi_singlepca(
    cube,
    cube_ref,
    angle_list,
    scale_list,
    ncomp,
    fwhm,
    source_xy,
    scaling,
    mask_center_px,
    svd_mode,
    imlib,
    imlib2,
    interpolation,
    collapse,
    collapse_ifs,
    ifs_collapse_range,
    verbose,
    start_time,
    nproc,
    crop_ifs,
    batch,
    full_output,
    weights=None,
    left_eigv=False,
    min_frames_pca=10,
    ref_strategy='RSDI',
    **rot_options,
):
    """Handle the full-frame ADI+mSDI single PCA post-processing."""
    z, n, y_in, x_in = cube.shape

    angle_list = check_pa_vector(angle_list)
    if not angle_list.shape[0] == n:
        msg = "Angle list vector has wrong length. It must equal the number"
        msg += " frames in the cube"
        raise ValueError(msg)

    if scale_list is None:
        raise ValueError("`scale_list` must be provided")
    else:
        check_array(scale_list, dim=1, msg="scale_list")
        if not scale_list.shape[0] == z:
            raise ValueError("`scale_list` has wrong length")

    # scale_list = check_scal_vector(scale_list)
    big_cube = []

    if verbose:
        print("Rescaling the spectral channels to align the speckles")
    for i in Progressbar(range(n), verbose=verbose):
        cube_resc = scwave(cube[:, i, :, :], scale_list, imlib=imlib2,
                           interpolation=interpolation)[0]
        if crop_ifs:
            cube_resc = cube_crop_frames(cube_resc, size=y_in, verbose=False)
        big_cube.append(cube_resc)

    big_cube = np.array(big_cube)
    big_cube = big_cube.reshape(z * n, big_cube.shape[2], big_cube.shape[3])

    # Do the same with reference cube if provided
    if cube_ref is not None:
        z, nr, y_in, x_in = cube_ref.shape
        big_cube_ref = []
        if verbose:
            msg = "Rescaling the spectral channels of the reference cube.."
            print(msg)
        for i in Progressbar(range(nr), verbose=verbose):
            cube_resc = scwave(cube_ref[:, i, :, :], scale_list, imlib=imlib2,
                               interpolation=interpolation)[0]
            if crop_ifs:
                cube_resc = cube_crop_frames(cube_resc, size=y_in,
                                             verbose=False)
            big_cube_ref.append(cube_resc)

        big_cube_ref = np.array(big_cube_ref)
        big_cube_ref = big_cube_ref.reshape(z * nr, big_cube_ref.shape[2],
                                            big_cube_ref.shape[3])
    else:
        big_cube_ref = None

    if verbose:
        timing(start_time)
        print("{} total frames".format(n * z))
        print("Performing single-pass PCA")

    if np.isscalar(ncomp):
        # When ncomp is a int and batch is not None, incremental ADI-PCA is run
        if batch is not None:
            res_cube = pca_incremental(
                big_cube,
                angle_list,
                batch,
                ncomp,
                collapse,
                verbose,
                return_residuals=True,
                start_time=start_time,
                weights=weights,
                nproc=nproc,
                imlib=imlib,
                interpolation=interpolation,
                **rot_options,
            )
        # When ncomp is a int/float and batch is None, standard ADI-PCA is run
        else:
            res_cube = _project_subtract(
                big_cube,
                big_cube_ref,
                ncomp,
                scaling,
                mask_center_px,
                svd_mode,
                verbose,
                False,
                left_eigv=left_eigv,
                min_frames_pca=min_frames_pca,
            )

        if verbose:
            timing(start_time)

        resadi_cube = np.zeros((n, y_in, x_in))

        if verbose:
            print("Descaling the spectral channels")
        if ifs_collapse_range == "all":
            idx_ini = 0
            idx_fin = z
        else:
            idx_ini = ifs_collapse_range[0]
            idx_fin = ifs_collapse_range[1]

        cube_desc_residuals = np.zeros_like(cube)

        for i in Progressbar(range(n), verbose=verbose):
            res_i = scwave(
                res_cube[i * z + idx_ini:i * z + idx_fin],
                scale_list[idx_ini:idx_fin],
                full_output=True,
                inverse=True,
                y_in=y_in,
                x_in=x_in,
                imlib=imlib2,
                interpolation=interpolation,
                collapse=collapse_ifs,
            )
            cube_desc_residuals[:, i] = res_i[0]
            resadi_cube[i] = res_i[1]

        if verbose:
            print("De-rotating and combining residuals")
            timing(start_time)
        der_res = cube_derotate(
            resadi_cube,
            angle_list,
            nproc=nproc,
            imlib=imlib,
            interpolation=interpolation,
            **rot_options,
        )
        if mask_center_px:
            der_res = mask_circle(der_res, mask_center_px)
        frame = cube_collapse(der_res, mode=collapse, w=weights)
        cube_allfr_residuals = res_cube
        cube_adi_residuals = resadi_cube
        return (cube_allfr_residuals, cube_desc_residuals, cube_adi_residuals,
                frame)

    # When ncomp is a tuple, pca_grid is called
    elif isinstance(ncomp, (tuple, list)):
        gridre = pca_grid(
            big_cube,
            angle_list,
            fwhm,
            range_pcs=ncomp,
            source_xy=source_xy,
            cube_ref=None,
            mode="fullfr",
            svd_mode=svd_mode,
            scaling=scaling,
            mask_center_px=mask_center_px,
            fmerit="mean",
            collapse=collapse,
            ifs_collapse_range=ifs_collapse_range,
            verbose=verbose,
            full_output=full_output,
            debug=False,
            plot=verbose,
            start_time=start_time,
            scale_list=scale_list,
            initial_4dshape=cube.shape,
            weights=weights,
            nproc=nproc,
            imlib=imlib,
            interpolation=interpolation,
            **rot_options,
        )
        return gridre

    else:
        raise TypeError(
            "`ncomp` must be an int, float, tuple or list for single-pass PCA"
        )


def _adimsdi_doublepca(
    cube,
    cube_ref,
    angle_list,
    scale_list,
    ncomp,
    scaling,
    mask_center_px,
    svd_mode,
    imlib,
    imlib2,
    interpolation,
    collapse,
    collapse_ifs,
    ifs_collapse_range,
    smooth_first_pass,
    verbose,
    start_time,
    nproc,
    weights=None,
    source_xy=None,
    delta_rot=None,
    fwhm=4,
    min_frames_pca=10,
    max_frames_pca=None,
    mask_rdi=None,
    cube_sig=None,
    left_eigv=False,
    ref_strategy='RSDI',
    **rot_options,
):
    """Handle the full-frame ADI+mSDI double PCA post-processing."""
    z, n, y_in, x_in = cube.shape

    if cube_ref is not None:
        cube = np.concatenate((cube, cube_ref), axis=1)
        nr = cube_ref.shape[1]
    else:
        nr = 0

    global ARRAY
    ARRAY = cube  # to be passed to _adimsdi_doublepca_ifs

    if not isinstance(ncomp, tuple):
        raise TypeError(
            "`ncomp` must be a tuple when a double pass PCA" " is performed"
        )
    else:
        ncomp_ifs, ncomp_adi = ncomp

    angle_list = check_pa_vector(angle_list)
    if not angle_list.shape[0] == n:
        msg = "Angle list vector has wrong length. It must equal the number"
        msg += " frames in the cube"
        raise ValueError(msg)

    if scale_list is None:
        raise ValueError("Scaling factors vector must be provided")
    else:
        if np.array(scale_list).ndim > 1:
            raise ValueError("Scaling factors vector is not 1d")
        if not scale_list.shape[0] == cube.shape[0]:
            raise ValueError("Scaling factors vector has wrong length")
    # scale_list = check_scal_vector(scale_list)

    if type(scaling) is not tuple:
        scaling = (scaling, scaling)

    if verbose:
        print("{} spectral channels in IFS cube".format(z))
        if ncomp_ifs is None:
            print("Combining multi-spectral frames (skipping PCA)")
        else:
            print("First PCA stage exploiting spectral variability")

    if ncomp_ifs is not None and ncomp_ifs > z:
        ncomp_ifs = min(ncomp_ifs, z)
        msg = "Number of PCs too high (max PCs={}), using {} PCs instead"
        print(msg.format(z, ncomp_ifs))

    res = pool_map(
        nproc,
        _adimsdi_doublepca_ifs,
        iterable(range(n+nr)),
        ncomp_ifs,
        scale_list,
        scaling[0],
        mask_center_px,
        svd_mode,
        imlib2,
        interpolation,
        collapse_ifs,
        ifs_collapse_range,
        fwhm,
        mask_rdi,
        left_eigv,
    )
    res_cube_channels = np.array(res)

    if verbose:
        timing(start_time)

    if smooth_first_pass is not None:
        res_cube_channels = cube_filter_lowpass(res_cube_channels,
                                                mode='gauss',
                                                fwhm_size=smooth_first_pass,
                                                verbose=False)

    # de-rotation of the PCA processed channels, ADI fashion
    if ncomp_adi is None:
        if verbose:
            print("{} ADI frames".format(n))
            print("De-rotating and combining frames (skipping PCA)")
        residuals_cube_channels_ = cube_derotate(
            res_cube_channels[:n],
            angle_list,
            nproc=nproc,
            imlib=imlib,
            interpolation=interpolation,
            **rot_options,
        )
        frame = cube_collapse(residuals_cube_channels_, mode=collapse,
                              w=weights)
        if verbose:
            timing(start_time)
    else:
        if ncomp_adi > n+nr:
            ncomp_adi = n+nr
            msg = "Number of PCs too high, using  maximum of {} PCs instead"
            print(msg.format(n))
        if verbose:
            print("{} ADI frames".format(n))
            if nr:
                print("+ {} reference frames".format(nr))
            print("Second PCA stage exploiting rotational variability")

        if source_xy is None:
            if 'A' in ref_strategy or cube_ref is None:  # e.g. 'ARSDI'
                res_ifs_adi = _project_subtract(
                    res_cube_channels,
                    None,
                    ncomp_adi,
                    scaling[1],
                    mask_center_px,
                    svd_mode,
                    verbose,
                    False,
                    cube_sig=cube_sig,
                    left_eigv=left_eigv,
                )
            else:  # 'RSDI'
                res_ifs_adi = _project_subtract(
                    res_cube_channels[:n],
                    res_cube_channels[n:],  # ref cube
                    ncomp_adi,
                    scaling[1],
                    mask_center_px,
                    svd_mode,
                    verbose,
                    False,
                    cube_sig=cube_sig,
                    left_eigv=left_eigv,
                )
            if verbose:
                timing(start_time)
        # A rotation threshold is applied
        else:
            if delta_rot is None or fwhm is None:
                msg = "Delta_rot or fwhm parameters missing. Needed for"
                msg += "PA-based rejection of frames from the library"
                raise TypeError(msg)
            yc, xc = frame_center(cube[0], False)
            x1, y1 = source_xy
            ann_center = dist(yc, xc, y1, x1)
            pa_thr = _compute_pa_thresh(ann_center, fwhm, delta_rot)

            res_ifs_adi = np.zeros_like(res_cube_channels)
            max_fr = max_frames_pca
            if max_frames_pca is not None:
                truncate = True
            else:
                truncate = False

            for frame in range(n):
                ind = _find_indices_adi(angle_list, frame, pa_thr,
                                        truncate=truncate, max_frames=max_fr)

                res_result = _project_subtract(
                    res_cube_channels[:n],
                    res_cube_channels[n:],
                    ncomp_adi,
                    scaling[1],
                    mask_center_px,
                    svd_mode,
                    verbose,
                    False,
                    ind,
                    frame,
                    cube_sig=cube_sig,
                    left_eigv=left_eigv,
                    min_frames_pca=min_frames_pca,
                )
                res_ifs_adi[frame] = res_result[-1].reshape((y_in, x_in))

        if verbose:
            print("De-rotating and combining residuals")
        residuals_cube_channels_ = cube_derotate(res_ifs_adi[:n], angle_list,
                                                 nproc=nproc, imlib=imlib,
                                                 interpolation=interpolation,
                                                 **rot_options)
        frame = cube_collapse(residuals_cube_channels_, mode=collapse,
                              w=weights)
        if verbose:
            timing(start_time)
    return res_cube_channels, residuals_cube_channels_, frame


def _adimsdi_doublepca_ifs(
    fr,
    ncomp,
    scale_list,
    scaling,
    mask_center_px,
    svd_mode,
    imlib,
    interpolation,
    collapse,
    ifs_collapse_range,
    fwhm,
    mask_rdi=None,
    left_eigv=False,
):
    """Call by _adimsdi_doublepca with pool_map."""
    global ARRAY

    z, n, y_in, x_in = ARRAY.shape
    multispec_fr = ARRAY[:, fr, :, :]

    if ifs_collapse_range == "all":
        idx_ini = 0
        idx_fin = z
    else:
        idx_ini = ifs_collapse_range[0]
        idx_fin = ifs_collapse_range[1]

    if ncomp is None:
        frame_i = cube_collapse(multispec_fr[idx_ini:idx_fin])
    else:
        cube_resc = scwave(
            multispec_fr, scale_list, imlib=imlib, interpolation=interpolation
        )[0]

        if mask_rdi is None:
            residuals = _project_subtract(
                cube_resc,
                None,
                ncomp,
                scaling,
                mask_center_px,
                svd_mode,
                verbose=False,
                full_output=False,
                left_eigv=left_eigv,
            )
        else:
            residuals = np.zeros_like(cube_resc)
            for i in range(z):
                cube_tmp = np.array([cube_resc[i]])
                cube_ref = np.array([cube_resc[j] for j in range(z) if j != i])
                residuals[i] = cube_subtract_sky_pca(
                    cube_tmp, cube_ref, mask_rdi, ncomp=ncomp, full_output=False
                )
        frame_i = scwave(
            residuals[idx_ini:idx_fin],
            scale_list[idx_ini:idx_fin],
            full_output=False,
            inverse=True,
            y_in=y_in,
            x_in=x_in,
            imlib=imlib,
            interpolation=interpolation,
            collapse=collapse,
        )
        if mask_center_px:
            frame_i = mask_circle(frame_i, mask_center_px)

    return frame_i


# def _adi_rdi_pca(
#     cube,
#     cube_ref,
#     angle_list,
#     ncomp,
#     source_xy,
#     delta_rot,
#     fwhm,
#     scaling,
#     mask_center_px,
#     svd_mode,
#     imlib,
#     interpolation,
#     collapse,
#     verbose,
#     start_time,
#     nproc,
#     weights=None,
#     mask_rdi=None,
#     cube_sig=None,
#     left_eigv=False,
#     **rot_options,
# ):
#     """Handle the ADI+RDI post-processing."""
#     n, y, x = cube.shape
#     n_ref, y_ref, x_ref = cube_ref.shape
#     angle_list = check_pa_vector(angle_list)
#     if not isinstance(ncomp, int):
#         raise TypeError("`ncomp` must be an int in the ADI+RDI case")
#     if ncomp > n_ref:
#         msg = (
#             "Requested number of PCs ({}) higher than the number of frames "
#             + "in the reference cube ({}); using the latter instead."
#         )
#         print(msg.format(ncomp, n_ref))
#         ncomp = n_ref

#     if not cube_ref.ndim == 3:
#         msg = "Input reference array is not a cube or 3d array"
#         raise ValueError(msg)
#     if not y_ref == y and x_ref == x:
#         msg = "Reference and target frames have different shape"
#         raise TypeError(msg)

#     if mask_rdi is None:
#         if source_xy is None:
#             residuals_result = _project_subtract(
#                 cube,
#                 cube_ref,
#                 ncomp,
#                 scaling,
#                 mask_center_px,
#                 svd_mode,
#                 verbose,
#                 True,
#                 cube_sig=cube_sig,
#                 left_eigv=left_eigv,
#             )
#             residuals_cube = residuals_result[0]
#             reconstructed = residuals_result[1]
#             V = residuals_result[2]
#             pcs = reshape_matrix(V, y, x) if not left_eigv else V.T
#             recon = reshape_matrix(reconstructed, y, x)
#         # A rotation threshold is applied
#         else:
#             if delta_rot is None or fwhm is None:
#                 msg = "Delta_rot or fwhm parameters missing. Needed for the"
#                 msg += "PA-based rejection of frames from the library"
#                 raise TypeError(msg)
#             nfrslib = []
#             residuals_cube = np.zeros_like(cube)
#             recon_cube = np.zeros_like(cube)
#             yc, xc = frame_center(cube[0], False)
#             x1, y1 = source_xy
#             ann_center = dist(yc, xc, y1, x1)
#             pa_thr = _compute_pa_thresh(ann_center, fwhm, delta_rot)
#             mid_range = np.abs(np.max(angle_list) - np.min(angle_list)) / 2
#             if pa_thr >= mid_range - mid_range * 0.1:
#                 new_pa_th = float(mid_range - mid_range * 0.1)
#                 if verbose:
#                     msg = "PA threshold {:.2f} is too big, will be set to "
#                     msg += "{:.2f}"
#                     print(msg.format(pa_thr, new_pa_th))
#                 pa_thr = new_pa_th

#             for frame in range(n):
#                 if ann_center > fwhm * 3:  # TODO: 3 optimal value? new par?
#                     ind = _find_indices_adi(
#                         angle_list, frame, pa_thr, truncate=True
#                     )
#                 else:
#                     ind = _find_indices_adi(angle_list, frame, pa_thr)

#                 res_result = _project_subtract(
#                     cube,
#                     cube_ref,
#                     ncomp,
#                     scaling,
#                     mask_center_px,
#                     svd_mode,
#                     verbose,
#                     full_output,
#                     ind,
#                     frame,
#                     cube_sig=cube_sig,
#                     left_eigv=left_eigv,
#                     min_frames_pca=min_frames_pca,
#                 )
#                 if full_output:
#                     nfrslib.append(res_result[0])
#                     residual_frame = res_result[1]
#                     recon_frame = res_result[2]
#                     residuals_cube[frame] = residual_frame.reshape((y, x))
#                     recon_cube[frame] = recon_frame.reshape((y, x))
#                 else:
#                     nfrslib.append(res_result[0])
#                     residual_frame = res_result[1]
#                     residuals_cube[frame] = residual_frame.reshape((y, x))

#             # number of frames in library printed for each annular quadrant
#             if verbose:
#                 descriptive_stats(nfrslib, verbose=verbose,
#                                   label="Size LIB: ")
#     else:
#         residuals_result = cube_subtract_sky_pca(
#             cube, cube_ref, mask_rdi, ncomp=ncomp, full_output=True
#         )
#         residuals_cube = residuals_result[0]
#         pcs = residuals_result[2]
#         recon = residuals_result[-1]

#     residuals_cube_ = cube_derotate(
#         residuals_cube,
#         angle_list,
#         nproc=nproc,
#         imlib=imlib,
#         interpolation=interpolation,
#         **rot_options,
#     )
#     frame = cube_collapse(residuals_cube_, mode=collapse, w=weights)
#     if mask_center_px:
#         frame = mask_circle(frame, mask_center_px)

#     if verbose:
#         print("Done de-rotating and combining")
#         timing(start_time)

#     return pcs, recon, residuals_cube, residuals_cube_, frame


def _project_subtract(
    cube,
    cube_ref,
    ncomp,
    scaling,
    mask_center_px,
    svd_mode,
    verbose,
    full_output,
    indices=None,
    frame=None,
    cube_sig=None,
    left_eigv=False,
    min_frames_pca=10,
):
    """
    PCA projection and model PSF subtraction.

    Used as a helping function by each of the PCA modes (ADI, ADI+RDI,
    ADI+mSDI).

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
    left_eigv : bool, optional
        Whether to use rather left or right singularvectors
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
        [full_output=True, indices is None, frame is None]
        The right singular vectors of the input matrix, as returned by
        ``svd/svd_wrapper()``
    """
    _, y, x = cube.shape

    if not isinstance(ncomp, (int, np.int_, float, np.float16, np.float32,
                              np.float64)):
        raise TypeError("Type not recognized for ncomp, should be int or float")

    # if a cevr is provided instead of an actual ncomp, first calculate it
    if isinstance(ncomp, (float, np.float16, np.float32, np.float64)):
        if not 1 > ncomp > 0:
            raise ValueError(
                "if `ncomp` is float, it must lie in the " "interval (0,1]"
            )

        svdecomp = SVDecomposer(cube, mode="fullfr", svd_mode=svd_mode,
                                scaling=scaling, verbose=verbose)
        _ = svdecomp.get_cevr(plot=False)
        # in this case ncomp is the desired CEVR
        cevr = ncomp
        ncomp = svdecomp.cevr_to_ncomp(cevr)
        if verbose:
            print("Components used : {}".format(ncomp))

    #  if isinstance(ncomp, (int, np.int_)):
    if indices is not None and frame is not None:
        matrix = prepare_matrix(
            cube, scaling, mask_center_px, mode="fullfr", verbose=False
        )
    elif left_eigv:
        matrix = prepare_matrix(cube, scaling, mask_center_px,
                                mode="fullfr", verbose=verbose,
                                discard_mask_pix=True)
    else:
        matrix = prepare_matrix(
            cube, scaling, mask_center_px, mode="fullfr", verbose=verbose
        )
    if cube_sig is None:
        matrix_emp = matrix.copy()
    else:
        if left_eigv:
            matrix_sig = prepare_matrix(cube_sig, scaling, mask_center_px,
                                        mode="fullfr", verbose=verbose,
                                        discard_mask_pix=True)
        else:
            nfr = cube_sig.shape[0]
            matrix_sig = np.reshape(cube_sig, (nfr, -1))
        matrix_emp = matrix - matrix_sig

    if cube_ref is not None:
        if left_eigv:
            matrix_ref = prepare_matrix(cube_sig, scaling, mask_center_px,
                                        mode="fullfr", verbose=verbose,
                                        discard_mask_pix=True)
        else:
            matrix_ref = prepare_matrix(cube_ref, scaling, mask_center_px,
                                        mode="fullfr", verbose=verbose)

    # check whether indices are well defined (i.e. not empty)
    msg = "{} frames comply to delta_rot condition < less than "
    msg1 = msg + "min_frames_pca ({}). Try decreasing delta_rot or "
    msg1 += "min_frames_pca"
    msg2 = msg + "ncomp ({}). Try decreasing the parameter delta_rot or "
    msg2 += "ncomp"
    if indices is not None and frame is not None:
        try:
            ref_lib = matrix_emp[indices]
        except IndexError:
            indices = None
        if cube_ref is None and indices is None:
            raise RuntimeError(msg1.format(0, min_frames_pca))

    # a rotation threshold is used (frames are processed one by one)
    if indices is not None and frame is not None:
        if cube_ref is not None:
            ref_lib = np.concatenate((ref_lib, matrix_ref))
        if ref_lib.shape[0] < min_frames_pca:
            raise RuntimeError(msg1.format(ref_lib.shape[0],
                                           min_frames_pca))
        if ref_lib.shape[0] < ncomp:
            raise RuntimeError(msg2.format(ref_lib.shape[0], ncomp))
        curr_frame = matrix[frame]  # current frame
        curr_frame_emp = matrix_emp[frame]
        if left_eigv:
            V = svd_wrapper(ref_lib, svd_mode, ncomp, False,
                            left_eigv=left_eigv)
            transformed = np.dot(curr_frame_emp.T, V)
            reconstructed = np.dot(V, transformed.T)
        else:
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
        if cube_ref is not None:
            ref_lib = matrix_ref
        else:
            ref_lib = matrix_emp
        if left_eigv:
            V = svd_wrapper(ref_lib, svd_mode, ncomp, verbose,
                            left_eigv=left_eigv)
            transformed = np.dot(matrix_emp.T, V)
            reconstructed = np.dot(V, transformed.T)
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


def get_pca_coeffs(cube, pcs, ncomp, scaling=None, mask_center_px=None,
                   verbose=True):
    """Return the weights (coefficients) of each PC to create the PCA model for\
    each image of the input cube.

    Parameters
    ----------
    cube : 3d np.ndarray
        DESCRIPTION.
    pcs : 2d np ndarray
        DESCRIPTION.
    ncomp : int
        How many PCs are used as a lower-dimensional subspace to project the
        target frames.

        * ADI (``cube`` is a 3d array): if an int is provided, ``ncomp`` is the
        number of PCs extracted from ``cube`` itself. If ``ncomp`` is a float
        in the interval [0, 1] then it corresponds to the desired cumulative
        explained variance ratio (the corresponding number of components is
        estimated). If ``ncomp`` is a tuple of two integers, then it
        corresponds to an interval of PCs in which final residual frames are
        computed (optionally, if a tuple of 3 integers is passed, the third
        value is the step). If ``ncomp`` is a list of int, these will be used to
        calculate residual frames. When ``ncomp`` is a tuple or list, and
        ``source_xy`` is not None, then the S/Ns (mean value in a 1xFWHM
        circular aperture) of the given (X,Y) coordinates are computed.

        * ADI+RDI (``cube`` and ``cube_ref`` are 3d arrays): ``ncomp`` is the
        number of PCs obtained from ``cube_ref``. If ``ncomp`` is a tuple,
        then it corresponds to an interval of PCs (obtained from ``cube_ref``)
        in which final residual frames are computed. If ``ncomp`` is a list of
        int, these will be used to calculate residual frames. When ``ncomp`` is
        a tuple or list, and ``source_xy`` is not None, then the S/Ns (mean
        value in a 1xFWHM circular aperture) of the given (X,Y) coordinates are
        computed.
    scaling : Enum, or tuple of Enum, see `vip_hci.config.paramenum.Scaling`
        Pixel-wise scaling mode using ``sklearn.preprocessing.scale``
        function. If set to None, the input matrix is left untouched. In the
        case of PCA-SADI in 2 steps, this can be a tuple of 2 values,
        corresponding to the scaling for each of the 2 steps of PCA.
    mask_center_px : None or int
        If None, no masking is done. If an integer > 1 then this value is the
        radius of the circular mask.
    verbose : bool, optional
        If True prints intermediate info.

    Returns
    -------
    coeffs: : numpy ndarray
        Weights of each PC to create the PCA model for each image of the cube.

    """
    z, y, x = np.shape(cube)
    matrix = prepare_matrix(cube, scaling=scaling,
                            mask_center_px=mask_center_px, mode='fullfr',
                            verbose=verbose)
    V = pcs.reshape(ncomp, -1)
    coeffs = np.dot(V, matrix.T)

    return coeffs
