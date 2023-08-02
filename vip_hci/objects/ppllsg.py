#! /usr/bin/env python
"""Module for the post-processing LLSG algorithm."""

__author__ = "Thomas Bédrine"
__all__ = ["LLSGBuilder", "PPLLSG"]

from typing import Optional
from dataclasses import dataclass

import numpy as np
from dataclass_builder import dataclass_builder

from .dataset import Dataset
from .postproc import PostProc
from ..psfsub import llsg, LLSG_Params
from ..config.utils_conf import algo_calculates_decorator as calculates


@dataclass
class PPLLSG(PostProc, LLSG_Params):
    """
    Post-processing LLSG algorithm.

    Parameters
    ----------
    full_output: bool, optional
        Whether to return the final median combined image only or with other
        intermediate arrays.
    _algo_name: str, optional
        Name of the algorithm wrapped by the object.
    """

    full_output: bool = True
    _algo_name: str = "llsg"
    frame_l: np.ndarray = None
    frame_s: np.ndarray = None
    frame_g: np.ndarray = None

    # TODO : write test
    @calculates("frame_final", "frame_l", "frame_s", "frame_g")
    def run(
        self,
        dataset: Optional[Dataset] = None,
        nproc: Optional[int] = None,
        full_output: Optional[bool] = True,
        **rot_options: Optional[dict]
    ):
        """
        Run the post-processing LLSG algorithm for model PSF subtraction.

        Parameters
        ----------
        dataset : Dataset, optional
            Dataset to process. If not provided, ``self.dataset`` is used (as
            set when initializing this object).
        nproc : int, optional
        full_output: boolean, optional
            Whether to return the final median combined image only or with
            other intermediate arrays.
        rot_options: dictionary, optional
            Dictionary with optional keyword values for "border_mode", "mask_val",
            "edge_blend", "interp_zeros", "ker" (see documentation of
            ``vip_hci.preproc.frame_rotate``)

        """
        self.snr_map = None
        self._update_dataset(dataset)
        self._explicit_dataset()

        if nproc is not None:
            self.nproc = nproc

        if full_output is not None:
            self.full_output = full_output

        if self.dataset.fwhm is None:
            raise ValueError("`fwhm` has not been set")

        params_dict = self._create_parameters_dict(LLSG_Params)

        all_params = {"algo_params": self, **rot_options}

        res = llsg(**all_params)
        self.frame_l = res[3]
        self.frame_s = res[4]
        self.frame_g = res[5]

        self.frame_final = self.frame_s

        if self.results is not None:
            self.results.register_session(
                params=params_dict, frame=self.frame_final, algo_name=self._algo_name
            )


LLSGBuilder = dataclass_builder(PPLLSG)
