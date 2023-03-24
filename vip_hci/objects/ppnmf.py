#! /usr/bin/env python
"""Module for the post-processing non-negative matrix factorization algorithm."""

__author__ = "Thomas Bédrine"
__all__ = [
    "NMFBuilder",
]

from typing import Tuple, Optional, List
from dataclasses import dataclass, field

import numpy as np
from dataclass_builder import dataclass_builder

from .dataset import Dataset
from .postproc import PostProc
from ..psfsub import nmf, nmf_annular
from ..config.utils_conf import algo_calculates_decorator as calculates


# TODO : write test
# TODO : update PPNMF doc to include 'nndsvdar' in init_svd section, weights too
@dataclass
class PPNMF(PostProc):
    """
    Post-processing full-frame non-negative matrix factorization algorithm.

    Parameters
    ----------
    ncomp : int, optional
        How many components are used as for low-rank approximation of the
        datacube.
    scaling : {None, 'temp-mean', 'spat-mean', 'temp-standard', 'spat-standard'}
        With None, no scaling is performed on the input data before SVD. With
        "temp-mean" then temporal px-wise mean subtraction is done, with
        "spat-mean" then the spatial mean is subtracted, with "temp-standard"
        temporal mean centering plus scaling to unit variance is done and with
        "spat-standard" spatial mean centering plus scaling to unit variance is
        performed.
    max_iter : int optional
        The number of iterations for the coordinate descent solver.
    random_state : int or None, optional
        Controls the seed for the Pseudo Random Number generator.
    mask_center_px : None or int
        If None, no masking is done. If an integer > 1 then this value is the
        radius of the circular mask.
    source_xy : tuple of int, optional
        For ADI-PCA, this triggers a frame rejection in the PCA library, with
        ``source_xy`` as the coordinates X,Y of the center of the annulus where
        the PA criterion is estimated. When ``ncomp`` is a tuple, a PCA grid is
        computed and the S/Ns (mean value in a 1xFWHM circular aperture) of the
        given (X,Y) coordinates are computed.
    delta_rot : array  (int and float/tuple of floats), optional
        Factor for tunning the parallactic angle threshold, expressed in FWHM.
        The int value is used for the full frame case, while the float goes for
        the annular case. Default is 1 (excludes 1xFHWM on each side of the
        considered frame). If a tuple of two floats is provided, they are used as
        the lower and upper intervals for the threshold (grows linearly as a
        function of the separation). !!! Important: this is used even if a reference
        cube is provided for RDI. This is to allow ARDI (PCA library built from both
        science and reference cubes). If you want to do pure RDI, set delta_rot
        to an arbitrarily high value such that the condition is never fulfilled
        for science frames to make it in the PCA library.
    init_svd: str, optional {'nnsvd','nnsvda','random'}
        Method used to initialize the iterative procedure to find H and W.
        'nndsvd': non-negative double SVD recommended for sparseness
        'nndsvda': NNDSVD where zeros are filled with the average of cube;
        recommended when sparsity is not desired
        'random': random initial non-negative matrix
    collapse : {'median', 'mean', 'sum', 'trimmean'}, str optional
        Sets the way of collapsing the frames for producing a final image.
    full_output: boolean, optional
        Whether to return the final median combined image only or with other
        intermediate arrays.
    verbose : {True, False}, bool optional
        If True prints intermediate info and timing.
    handle_neg: str, opt {'subtr_min','mask','null'}
        Determines how to handle negative values: mask them, set them to zero,
        or subtract the minimum value in the arrays. Note: 'mask' or 'null'
        may leave significant artefacts after derotation of residual cube
        => those options should be used carefully (e.g. with proper treatment
        of masked values in non-derotated cube of residuals).
    nmf_args : dictionary, optional
        Additional arguments for scikit-learn NMF algorithm. See:
        https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html

    """

    ncomp: int = 1
    scaling: str = None
    max_iter: int = 10000
    random_state: int = None
    mask_center_px: int = None
    source_xy: Tuple[int] = None
    delta_rot: int = 1
    delta_rot_ann: Tuple[float] = (0.1, 1)
    init_svd: str = "nndsvd"
    collapse: str = "median"
    verbose: bool = True
    cube_sig = None
    handle_neg: str = "mask"
    nmf_args: dict = field(default_factory=lambda: {})
    radius_int: int = 0
    asize: int = 4
    n_segments: int = 1
    min_frames_lib: int = 2
    max_frames_lib: int = 200
    imlib: str = "vip-fft"
    interpolation: str = "lanczos4"
    theta_init: int = 0
    weights: List = None
    _algo_name: List[str] = field(default_factory=lambda: ["nmf", "nmf_annular"])
    nmf_reshaped: np.ndarray = None
    cube_residuals: np.ndarray = None
    cube_residuals_der: np.ndarray = None
    cube_residuals_resc: np.ndarray = None

    @calculates(
        "nmf_reshaped",
        "cube_recon",
        "cube_residuals",
        "cube_residuals_der",
        "frame_final",
    )
    def run(
        self,
        runmode: Optional[str] = "fullframe",
        dataset: Optional[Dataset] = None,
        nproc: Optional[int] = 1,
        full_output: Optional[bool] = True,
        verbose: Optional[bool] = True,
        **rot_options: Optional[dict]
    ):
        """
        Run the post-processing NMF algorithm for model PSF subtraction.

        Parameters
        ----------
        runmode : {'fullframe', 'annular'}
            Defines which version of NMF to run between full frame and annular.
        dataset : Dataset object
            An Dataset object to be processed.
        nproc : None or int, optional
            Number of processes for parallel computing. If None the number of
            processes will be set to cpu_count()/2.
        full_output: bool, optional
            Whether to return the final median combined image only or with other
            intermediate arrays.
        verbose : bool, optional
            If True prints to stdout intermediate info.
        rot_options: dictionary, optional
            Dictionary with optional keyword values for "border_mode", "mask_val",
            "edge_blend", "interp_zeros", "ker" (see documentation of
            ``vip_hci.preproc.frame_rotate``).

        """
        self._update_dataset(dataset)

        if self.dataset.fwhm is None:
            raise ValueError("`fwhm` has not been set")

        add_params = {
            "cube": self.dataset.cube,
            "angle_list": self.dataset.angles,
            "fwhm": self.dataset.fwhm,
            "cube_ref": self.dataset.cuberef,
            "nproc": nproc,
            "full_output": full_output,
            "verbose": verbose,
        }

        if runmode == "fullframe":
            func_params = self._setup_parameters(fkt=nmf, **add_params)
            res = nmf(**func_params, **rot_options)

            (
                self.nmf_reshaped,
                self.cube_recon,
                self.cube_residuals,
                self.cube_residuals_der,
                self.frame_final,
            ) = res

            if self.results is not None:
                self.results.register_session(
                    params=func_params,
                    frame=self.frame_final,
                    algo_name=self._algo_name[0],
                )
        else:
            func_params = self._setup_parameters(fkt=nmf_annular, **add_params)
            res = nmf_annular(**func_params, **rot_options)

            (
                self.cube_residuals,
                self.cube_residuals_der,
                self.cube_recon,
                self.nmf_reshaped,
                self.frame_final,
            ) = res

            if self.results is not None:
                self.results.register_session(
                    params=func_params,
                    frame=self.frame_final,
                    algo_name=self._algo_name[1],
                )


NMFBuilder = dataclass_builder(PPNMF)
