#! /usr/bin/env python
"""Module for the post-processing PCA algorithm."""

__author__ = "Thomas Bédrine"
__all__ = [
    "PCABuilder",
]

from typing import Tuple, Optional, Union, List
from dataclasses import dataclass, field

import numpy as np
from pandas import DataFrame
from dataclass_builder import dataclass_builder

from .dataset import Dataset
from .postproc import PostProc
from ..psfsub import pca, pca_annular, pca_grid, pca_annulus
from ..config.utils_conf import algo_calculates_decorator as calculates


# TODO : separate the various cases of PCA usage (basics, optnpc finding, others ?)
# TODO : work on a significance computation function (maybe in PostProc)
@dataclass
class PPPCA(PostProc):
    """
    Post-processing PCA algorithm, compatible with various options.

    Depending on what mode you need, parameters may vary. Check the list below to
    ensure which arguments are required. Currently, four variations of the PCA can be
    called :
        - full-frame PCA
        - annular PCA
        - grid PCA
        - single annulus PCA.
    Some parameters are common to several variations.

    Common parameters
    -----------------
    ncomp : int, float or tuple of int/None, or list, optional
        How many PCs are used as a lower-dimensional subspace to project the
        target frames (see documentation of ``vip_hci.psfsub.pca_fullfr`` and
        ``vip_hci.psfsub.pca_local`` for information on the various modes).
    svd_mode : {'lapack', 'arpack', 'eigen', 'randsvd', 'cupy', 'eigencupy',
        'randcupy', 'pytorch', 'eigenpytorch', 'randpytorch'}, str optional
        Switch for the SVD method/library to be used. See the documentation
        of ``vip_hci.psfsub.pca_fullfr`` or ``vip_hci.psfsub.pca_local``.
    scaling : {None, "temp-mean", spat-mean", "temp-standard",
        "spat-standard"}, None or str optional
        Pixel-wise scaling mode using ``sklearn.preprocessing.scale`` function.
        If set to None, the input matrix is left untouched. See the documentation
        of ``vip_hci.psfsub.pca_fullfr`` or ``vip_hci.psfsub.pca_local``.
    delta_rot : int, float or tuple of floats, optional
        Factor for tuning the parallactic angle threshold, expressed in FWHM.
        Used in different ways by full-frame and annular modes, be sure to take a
        look at their respective documentation.
    imlib : str, optional
        See the documentation of ``vip_hci.preproc.frame_rotate``.
    interpolation : str, optional
        See the documentation of the ``vip_hci.preproc.frame_rotate`` function.
    collapse : {'median', 'mean', 'sum', 'trimmean'}, str optional
        Sets how temporal residual frames should be combined to produce an
        ADI image.
    collapse_ifs : {'median', 'mean', 'sum', 'trimmean'}, str optional
        Sets how spectral residual frames should be combined to produce an
        mSDI image.
    ifs_collapse_range: str 'all' or tuple of 2 int
        If a tuple, it should contain the first and last channels where the mSDI
        residual channels will be collapsed (by default collapses all channels).
    annulus_width : float, optional
        Width in pixels of the annulus in the case of the "single annulus" or "grid"
        mode.
    weights: 1d numpy array or list, optional
        Weights to be applied for a weighted mean. Need to be provided if
        collapse mode is 'wmean'.
    cube_sig: numpy ndarray, opt
        Cube with estimate of significant authentic signals. If provided, this
        will subtracted before projecting cube onto reference cube.

    Full-frame parameters
    ---------------------
    imlib2 : str, optional
        See the documentation of ``vip_hci.preproc.cube_rescaling_wavelengths``.
    mask_center_px : None or int
        If None, no masking is done. If an integer > 1 then this value is the
        radius of the circular mask.
    source_xy : tuple of int, optional
        For ADI-PCA, this triggers a frame rejection in the PCA library, with
        ``source_xy`` as the coordinates X,Y of the center of the annulus where
        the PA criterion is estimated. When ``ncomp`` is a tuple, a PCA grid is
        computed and the S/Ns (mean value in a 1xFWHM circular aperture) of the
        given (X,Y) coordinates are computed.
    adimsdi : {'single', 'double'}, str optional
        Changes the way the 4d cubes (ADI+mSDI) are processed. Basically it
        determines whether a single or double pass PCA is going to be computed.

        * ``single``: the multi-spectral frames are rescaled wrt the largest
          wavelength to align the speckles and all the frames (n_channels *
          n_adiframes) are processed with a single PCA low-rank approximation.

        * ``double``: a first stage is run on the rescaled spectral frames, and
          a second PCA frame is run on the residuals in an ADI fashion.

    crop_ifs: bool, optional
        [adimsdi='single'] If True cube is cropped at the moment of frame
        rescaling in wavelength. This is recommended for large FOVs such as the
        one of SPHERE, but can remove significant amount of information close
        to the edge of small FOVs (e.g. SINFONI).
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

    Annular parameters
    ------------------
    radius_int : int, optional
        The radius of the innermost annulus. By default is 0, if >0 then the
        central circular region is discarded.
    asize : float, optional
        The size of the annuli, in pixels.
    n_segments : int or list of ints or 'auto', optional
        The number of segments for each annulus. When a single integer is given
        it is used for all annuli. When set to 'auto', the number of segments is
        automatically determined for every annulus, based on the annulus width.
    delta_sep : float or tuple of floats, optional
        The threshold separation in terms of the mean FWHM (for ADI+mSDI data).
        If a tuple of two values is provided, they are used as the lower and
        upper intervals for the threshold (grows as a function of the
        separation).
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
    theta_init : int
        Initial azimuth [degrees] of the first segment, counting from the
        positive x-axis counterclockwise (irrelevant if n_segments=1).

    Grid parameters
    ---------------
    range_pcs : None or tuple, optional
        The interval of PCs to be tried. Refer to ``vip_hci.psfsub.pca_grid`` for more
        information.
    mode : {'fullfr', 'annular'}, optional
        Mode for PCA processing (full-frame or just in an annulus).
    fmerit : {'px', 'max', 'mean'}
        The function of merit to be maximized. 'px' is *source_xy* pixel's SNR,
        'max' the maximum SNR in a FWHM circular aperture centered on
        ``source_xy`` and 'mean' is the mean SNR in the same circular aperture.
    plot : bool, optional
        Whether to plot the SNR and flux as functions of PCs and final PCA
        frame or not.
    save_plot: string
        If provided, the pc optimization plot will be saved to that path.
    initial_4dshape : None or tuple, optional
        Shape of the initial ADI+mSDI cube.
    exclude_negative_lobes : bool, opt
        Whether to include the adjacent aperture lobes to the tested location
        or not. Can be set to True if the image shows significant neg lobes.

    Single annulus parameters
    -------------------------
    r_guess : float
        Radius of the annulus in pixels.

    """

    # Common parameters
    ncomp: int = 1
    svd_mode: str = "lapack"
    scaling: str = None
    delta_rot: Union[int, Tuple[float]] = 1
    imlib: str = "vip-fft"
    interpolation: str = "lanczos4"
    collapse: str = "median"
    collapse_ifs: str = "mean"
    ifs_collapse_range: str = "all"
    weights: List = None
    annulus_width: int = 20
    _algo_name: List[str] = field(
        default_factory=lambda: [
            "pca",
            "pca_annular",
            "pca_grid",
            "pca_annulus",
        ]
    )
    cube_sig: np.ndarray = None
    cube_residuals: np.ndarray = None
    cube_residuals_der: np.ndarray = None
    # Full-frame parameters
    mask_center_px: int = None
    source_xy: Tuple[int] = None
    adimsdi: str = "double"
    crop_ifs: bool = True
    imlib2: str = "vip-fft"
    mask_rdi: np.ndarray = None
    check_mem: bool = True
    conv: bool = False
    batch: Union[float, int] = None
    cube_reconstructed: np.ndarray = None
    pcs: np.ndarray = None
    cube_residuals_per_channel: np.ndarray = None
    cube_residuals_per_channel_der: np.ndarray = None
    cube_residuals_resc: np.ndarray = None
    # Annular parameters
    radius_int: int = 0
    asize: int = 4
    n_segments: int = 1
    delta_sep: Union[float, Tuple[float]] = (0.1, 1)
    nproc: int = 1
    min_frames_lib: int = 2
    max_frames_lib: int = 200
    tol: float = 1e-1
    theta_init: float = 0
    # Grid parameters
    range_pcs: Tuple[int] = None
    mode: str = "fullfr"
    fmerit: str = "mean"
    plot: bool = True
    save_plot: str = None
    exclude_negative_lobes: bool = False
    initial_4dshape: Tuple = None
    dataframe: DataFrame = None
    pc_list: List = None
    opt_number_pc: int = None
    # Single annulus parameters
    r_guess: float = None

    # TODO: write test
    @calculates(
        "frame_final",
        "cube_reconstructed",
        "cube_residuals",
        "cube_residuals_der",
        "pcs",
        "cube_residuals_per_channel",
        "cube_residuals_per_channel_der",
        "cube_residuals_resc",
        "dataframe",
        "pc_list",
        "opt_number_pc",
    )
    def run(
        self,
        runmode: Optional[str] = "fullframe",
        dataset: Optional[Dataset] = None,
        nproc: Optional[int] = 1,
        verbose: Optional[bool] = True,
        full_output: Optional[bool] = True,
        **rot_options: Optional[dict]
    ):
        """
        Run the post-processing PCA algorithm for model PSF subtraction.

        Depending on the desired mode - full-frame or annular - parameters used will
        diverge, and calculated attributes may vary as well. In full-frame case :

            3D case:
                cube_reconstructed
                cube_residuals
                cube_residuals_der
            3D case, source_xy is None:
                cube_residuals
                pcs
            4D case, adimsdi="double":
                cube_residuals_per_channel
                cube_residuals_per_channel_der
            4D case, adimsdi="single":
                cube_residuals
                cube_residuals_resc

        Parameters
        ----------
        runmode : {'fullframe', 'annular', 'grid', 'annulus'}, optional
            Mode of execution for the PCA.
        dataset : Dataset, optional
            Dataset to process. If not provided, ``self.dataset`` is used (as
            set when initializing this object).
        nproc : int, optional
        verbose : bool, optional
            If True prints to stdout intermediate info.
        full_output: boolean, optional
            Whether to return the final median combined image only or with
            other intermediate arrays.
        rot_options: dictionary, optional
            Dictionary with optional keyword values for "border_mode", "mask_val",
            "edge_blend", "interp_zeros", "ker" (see documentation of
            ``vip_hci.preproc.frame_rotate``)

        """
        self._update_dataset(dataset)

        # Fullframe mode
        if runmode == "fullframe":
            if self.source_xy is not None and self.dataset.fwhm is None:
                raise ValueError("`fwhm` has not been set")

            # TODO : review the wavelengths attribute to be a scale_list instead
            add_params = {
                "cube": self.dataset.cube,
                "angle_list": self.dataset.angles,
                "cube_ref": self.dataset.cuberef,
                "fwhm": self.dataset.fwhm,
                "scale_list": self.dataset.wavelengths,
                "nproc": nproc,
                "full_output": full_output,
            }

            func_params = self._setup_parameters(fkt=pca, **add_params)

            res = pca(**func_params, **rot_options)

            if self.dataset.cube.ndim == 3:
                if self.source_xy is not None:
                    frame, recon_cube, res_cube, res_cube_der = res
                    self.cube_reconstructed = recon_cube
                    self.cube_residuals = res_cube
                    self.cube_residuals_der = res_cube_der
                    self.frame_final = frame
                else:
                    frame, pcs, recon_cube, res_cube, res_cube_der = res
                    self.pcs = pcs
                    self.cube_reconstructed = recon_cube
                    self.cube_residuals = res_cube
                    self.cube_residuals_der = res_cube_der
                    self.frame_final = frame
            elif self.dataset.cube.ndim == 4:
                if self.adimsdi == "double":
                    frame, res_cube_chan, res_cube_chan_der = res
                    self.cube_residuals_per_channel = res_cube_chan
                    self.cube_residuals_per_channel_der = res_cube_chan_der
                    self.frame_final = frame
                elif self.adimsdi == "single":
                    frame, cube_allfr_res, cube_adi_res = res
                    self.cube_residuals = cube_allfr_res
                    self.cube_residuals_resc = cube_adi_res
                    self.frame_final = frame

            if self.results is not None:
                self.results.register_session(
                    params=func_params,
                    frame=self.frame_final,
                    algo_name=self._algo_name[0],
                )

        # Annular mode
        elif runmode == "annular":
            if self.dataset.fwhm is None:
                raise ValueError("`fwhm` has not been set")

            if self.nproc is None:
                self.nproc = nproc

            add_params = {
                "cube": self.dataset.cube,
                "angle_list": self.dataset.angles,
                "cube_ref": self.dataset.cuberef,
                "fwhm": self.dataset.fwhm,
                "scale_list": self.dataset.wavelengths,
                "full_output": full_output,
                "verbose": verbose,
            }

            func_params = self._setup_parameters(fkt=pca_annular, **add_params)

            res = pca_annular(**func_params, **rot_options)

            self.cube_residuals, self.cube_residuals_der, self.frame_final = res

            if self.results is not None:
                self.results.register_session(
                    params=func_params,
                    frame=self.frame_final,
                    algo_name=self._algo_name[1],
                )

        # Grid mode
        elif runmode == "grid":
            if self.dataset.fwhm is None:
                raise ValueError("`fwhm` has not been set")

            add_params = {
                "cube": self.dataset.cube,
                "angle_list": self.dataset.angles,
                "cube_ref": self.dataset.cuberef,
                "fwhm": self.dataset.fwhm,
                "scale_list": self.dataset.wavelengths,
                "full_output": full_output,
                "verbose": verbose,
            }

            func_params = self._setup_parameters(fkt=pca_grid, **add_params)

            res = pca_grid(**func_params, **rot_options)

            (
                self.cube_residuals,
                self.pc_list,
                self.frame_final,
                self.dataframe,
                self.opt_number_pc,
            ) = res

            if self.results is not None:
                self.results.register_session(
                    params=func_params,
                    frame=self.frame_final,
                    algo_name=self._algo_name[2],
                )

        # Annulus mode
        else:
            if self.dataset.fwhm is None:
                raise ValueError("`fwhm` has not been set")

            add_params = {
                "cube": self.dataset.cube,
                "angs": self.dataset.angles,
                "cube_ref": self.dataset.cuberef,
            }

            func_params = self._setup_parameters(fkt=pca_annulus, **add_params)

            res = pca_annulus(**func_params, **rot_options)

            self.frame_final = res

            if self.results is not None:
                self.results.register_session(
                    params=func_params,
                    frame=self.frame_final,
                    algo_name=self._algo_name[3],
                )


PCABuilder = dataclass_builder(PPPCA)
