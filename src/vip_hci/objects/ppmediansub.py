#! /usr/bin/env python
"""Module for the post-processing median subtraction algorithm."""

__author__ = "Thomas Bédrine"
__all__ = ["MedianBuilder", "PPMedianSub"]

from typing import Optional
from dataclasses import dataclass

import numpy as np
from dataclass_builder import dataclass_builder

from .dataset import Dataset
from .postproc import PostProc
from ..psfsub import median_sub, MEDIAN_SUB_Params
from ..config.utils_conf import algo_calculates_decorator as calculates


@dataclass
class PPMedianSub(PostProc, MEDIAN_SUB_Params):
    """
    Object used as a wrapper for the ``vip_hci.psfsub.median_sub``.

    Gets its parameters from the MedsubParams dataclass.

    Parameters
    ----------
    full_output: bool, optional
        Whether to return the final median combined image only or with other
        intermediate arrays.
    _algo_name: str, optional
        Name of the algorithm wrapped by the object.

    """

    full_output: bool = True
    _algo_name: str = "median_sub"
    cube_residuals: np.ndarray = None
    cube_residuals_der: np.ndarray = None

    # TODO: write test
    @calculates("cube_residuals", "cube_residuals_der", "frame_final")
    def run(
        self,
        dataset: Optional[Dataset] = None,
        nproc: Optional[int] = None,
        full_output: Optional[bool] = None,
        **rot_options: Optional[dict]
    ) -> None:
        """
        Run the post-processing median subtraction algorithm for model PSF subtraction.

        Parameters
        ----------
        results : PPResult object, optional
            Container for the results of the algorithm. May hold the parameters used,
            as well as the ``frame_final`` (and the ``snr_map`` if generated).
        dataset : Dataset object, optional
            An Dataset object to be processed.
        nproc : None or int, optional
            Number of processes for parallel computing. If None the number of
            processes will be set to cpu_count()/2. By default the algorithm works
            in single-process mode.
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

        if self.mode == "annular" and self.dataset.fwhm is None:
            raise ValueError("`fwhm` has not been set")

        if nproc is not None:
            self.nproc = nproc

        if full_output is not None:
            self.full_output = full_output

        self._explicit_dataset()

        params_dict = self._create_parameters_dict(MEDIAN_SUB_Params)

        all_params = {"algo_params": self, **rot_options}

        res = median_sub(**all_params)

        self.cube_residuals, self.cube_residuals_der, self.frame_final = res

        if self.results is not None:
            self.results.register_session(
                params=params_dict,
                frame=self.frame_final,
                algo_name=self._algo_name,
            )


MedianBuilder = dataclass_builder(PPMedianSub)
