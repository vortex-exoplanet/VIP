#! /usr/bin/env python
"""Module for the post-processing LOCI algorithm."""

__author__ = "Thomas Bédrine"
__all__ = ["LOCIBuilder", "PPLOCI"]


from typing import Optional
from dataclasses import dataclass

import numpy as np
from dataclass_builder import dataclass_builder

from .dataset import Dataset
from .postproc import PostProc
from ..psfsub import xloci, XLOCI_Params
from ..config.utils_conf import algo_calculates_decorator as calculates


@dataclass
class PPLOCI(PostProc, XLOCI_Params):
    """
    Post-processing LOCI algorithm.

    Parameters
    ----------
    full_output: bool, optional
        Whether to return the final median combined image only or with other
        intermediate arrays.
    _algo_name: str, optional
        Name of the algorithm wrapped by the object.

    """

    full_output: bool = True
    _algo_name: str = "xloci"
    cube_res: np.ndarray = None
    cube_der: np.ndarray = None

    # TODO: write test
    @calculates("frame_final", "cube_res", "cube_der")
    def run(
        self,
        dataset: Optional[Dataset] = None,
        nproc: Optional[int] = None,
        **rot_options: Optional[dict]
    ):
        """
        Run the post-processing LOCI algorithm for model PSF subtraction.

        Parameters
        ----------
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
        self.snr_map = None
        self._update_dataset(dataset)

        if self.dataset.fwhm is None:
            raise ValueError("`fwhm` has not been set")

        self._explicit_dataset()

        if nproc is not None:
            self.nproc = nproc

        params_dict = self._create_parameters_dict(XLOCI_Params)

        all_params = {"algo_params": self, **rot_options}
        res = xloci(**all_params)

        self.cube_res, self.cube_der, self.frame_final = res

        if self.results is not None:
            self.results.register_session(
                frame=self.frame_final, params=params_dict, algo_name=self._algo_name
            )


LOCIBuilder = dataclass_builder(PPLOCI)
