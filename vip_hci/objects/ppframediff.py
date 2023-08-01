#! /usr/bin/env python
"""Module for the post-processing frame differencing algorithm."""

__author__ = "Thomas Bédrine"
__all__ = ["FrameDiffBuilder", "PPFrameDiff"]

from dataclasses import dataclass
from typing import Optional

from dataclass_builder import dataclass_builder

from .dataset import Dataset
from .postproc import PostProc
from ..psfsub import frame_diff, FRAME_DIFF_Params
from ..config.utils_conf import algo_calculates_decorator as calculates


@dataclass
class PPFrameDiff(PostProc, FRAME_DIFF_Params):
    """
    Post-processing frame differencing algorithm.

    Parameters
    ----------
    _algo_name: str, optional
        Name of the algorithm wrapped by the object.

    """

    _algo_name: str = "frame_diff"

    # TODO: write test
    @calculates("frame_final")
    def run(
        self,
        dataset: Optional[Dataset] = None,
        nproc: Optional[int] = 1,
        full_output: Optional[bool] = True,
        **rot_options: Optional[dict]
    ):
        """
        Run the post-processing median subtraction algorithm for model PSF subtraction.

        Parameters
        ----------
        dataset : Dataset object
            A Dataset object to be processed.
        nproc : None or int, optional
            Number of processes for parallel computing. If None the number of
            processes will be set to cpu_count()/2. By default the algorithm works
            in single-process mode.
        full_output: bool, optional
            Whether to return the final median combined image only or with other
            intermediate arrays.
        rot_options: dictionary, optional
            Dictionary with optional keyword values for "border_mode", "mask_val",
            "edge_blend", "interp_zeros", "ker" (see documentation of
            ``vip_hci.preproc.frame_rotate``).

        """
        self.snr_map = None
        self._update_dataset(dataset)

        if self.dataset.fwhm is None:
            raise ValueError("`fwhm` has not been set")

        if nproc is not None:
            self.nproc = nproc

        if full_output is not None:
            self.full_output = full_output

        self._explicit_dataset()
        params_dict = self._create_parameters_dict(FRAME_DIFF_Params)

        all_params = {"algo_params": self, **rot_options}

        res = frame_diff(**all_params)

        self.frame_final = res

        if self.results is not None:
            self.results.register_session(
                params=params_dict,
                frame=self.frame_final,
                algo_name=self._algo_name,
            )


FrameDiffBuilder = dataclass_builder(PPFrameDiff)
