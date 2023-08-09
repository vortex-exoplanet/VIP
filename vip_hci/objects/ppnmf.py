#! /usr/bin/env python
"""Module for the post-processing non-negative matrix factorization algorithm."""

__author__ = "Thomas Bédrine"
__all__ = ["NMFBuilder", "PPNMF"]

from typing import Optional, List, Tuple, Union
from dataclasses import dataclass, field

import numpy as np
from dataclass_builder import dataclass_builder

from .dataset import Dataset
from .postproc import PostProc
from ..psfsub import nmf, nmf_annular, NMF_Params, NMF_ANNULAR_Params
from ..config.utils_conf import algo_calculates_decorator as calculates

DELTA_ANN_DEFAULT = (0.1, 1)
DELTA_FF_DEFAULT = 1


# TODO: update PPNMF doc to include 'nndsvdar' in init_svd section, weights too
@dataclass
class PPNMF(PostProc, NMF_ANNULAR_Params, NMF_Params):
    """
    Post-processing full-frame non-negative matrix factorization algorithm.

    Parameters
    ----------
    delta_rot: int or float or tuple of floats, optional
        Factor for tunning the parallactic angle threshold, expressed in FWHM.
        Due to this parameter being shared between annular and fullframe versions of
        NMF, the annular version is forcefully set by default. If the fullframe
        version is selected, the default value is changed to the fullframe value.
    full_output: bool, optional
        Whether to return the final median combined image only or with other
        intermediate arrays.
    _algo_name: str, optional
        Name of the algorithm wrapped by the object.

    """

    delta_rot: Union[int, float, Tuple[float]] = DELTA_ANN_DEFAULT
    full_output: bool = True
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
        verbose: Optional[bool] = None,
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
        self.snr_map = None
        self._update_dataset(dataset)

        if self.dataset.fwhm is None:
            raise ValueError("`fwhm` has not been set")

        self._explicit_dataset()
        if nproc is not None:
            self.nproc = nproc

        if verbose is not None:
            self.verbose = verbose

        all_params = {"algo_params": self, **rot_options}

        if runmode == "fullframe":
            # Annular NMF gives the default delta_rot, fullframe delta_rot must be int
            if not isinstance(self.delta_rot, float):
                self.delta_rot = DELTA_FF_DEFAULT

            params_dict = self._create_parameters_dict(NMF_Params)
            res = nmf(**all_params)

            (
                self.nmf_reshaped,
                self.cube_recon,
                self.cube_residuals,
                self.cube_residuals_der,
                self.frame_final,
            ) = res

            if self.results is not None:
                self.results.register_session(
                    params=params_dict,
                    frame=self.frame_final,
                    algo_name=self._algo_name[0],
                )

            # Putting back the annular delta_rot default value
            self.delta_rot = DELTA_ANN_DEFAULT

        else:
            params_dict = self._create_parameters_dict(NMF_ANNULAR_Params)
            res = nmf_annular(**all_params)

            (
                self.cube_residuals,
                self.cube_residuals_der,
                self.cube_recon,
                self.nmf_reshaped,
                self.frame_final,
            ) = res

            if self.results is not None:
                self.results.register_session(
                    params=params_dict,
                    frame=self.frame_final,
                    algo_name=self._algo_name[1],
                )


NMFBuilder = dataclass_builder(PPNMF)
