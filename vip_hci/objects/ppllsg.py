#! /usr/bin/env python
"""Module for the post-processing LLSG algorithm."""

__author__ = "Thomas Bédrine"
__all__ = [
    "LLSGBuilder",
]

from typing import Optional
from dataclasses import dataclass

import numpy as np
from dataclass_builder import dataclass_builder

from .dataset import Dataset
from .postproc import PostProc
from ..psfsub import llsg
from ..config.utils_conf import algo_calculates_decorator as calculates


@dataclass
class PPLLSG(PostProc):
    """
    Post-processing LLSG algorithm.

    Parameters
    ----------
        rank : int, optional
            Expected rank of the L component.
        thresh : float, optional
            Factor that scales the thresholding step in the algorithm.
        max_iter : int, optional
            Sets the number of iterations.
        low_rank_ref :
            If True the first estimation of the L component is obtained from the
            remaining segments in the same annulus.
        low_rank_mode : {'svd', 'brp'}, optional
            Sets the method of solving the L update.
        auto_rank_mode : {'noise', 'cevr'}, str optional
            If ``rank`` is None, then ``auto_rank_mode`` sets the way that the
            ``rank`` is determined: the noise minimization or the cumulative
            explained variance ratio (when 'svd' is used).
        residuals_tol : float, optional
            The value of the noise decay to be used when ``rank`` is None and
            ``auto_rank_mode`` is set to ``noise``.
        cevr : float, optional
            Float value in the range [0,1] for selecting the cumulative explained
            variance ratio to choose the rank automatically (if ``rank`` is None).
        thresh_mode : {'soft', 'hard'}, optional
            Sets the type of thresholding.
        nproc : None or int, optional
            Number of processes for parallel computing. If None the number of
            processes will be set to cpu_count()/2. By default the algorithm works
            in single-process mode.
        asize : int or None, optional
            If ``asize`` is None then each annulus will have a width of ``2*asize``.
            If an integer then it is the width in pixels of each annulus.
        n_segments : int or list of ints, optional
            The number of segments for each annulus. When a single integer is given
            it is used for all annuli.
        azimuth_overlap : int or None, optional
            Sets the amount of azimuthal averaging.
        radius_int : int, optional
            The radius of the innermost annulus. By default is 0, if >0 then the
            central circular area is discarded.
        random_seed : int or None, optional
            Controls the seed for the Pseudo Random Number generator.
        high_pass : odd int or None, optional
            If set to an odd integer <=7, a high-pass filter is applied to the
            frames. The ``vip_hci.var.frame_filter_highpass`` is applied twice,
            first with the mode ``median-subt`` and a large window, and then with
            ``laplacian-conv`` and a kernel size equal to ``high_pass``. 5 is an
            optimal value when ``fwhm`` is ~4.
        collapse : {'median', 'mean', 'sum', 'trimmean'}, str optional
            Sets the way of collapsing the frames for producing a final image.

    """

    rank: int = 10
    thresh: float = 1
    max_iter: int = 10
    low_rank_ref: bool = False
    low_rank_mode: str = "svd"
    auto_rank_mode: str = "noise"
    residuals_tol: float = 1e-1
    cevr: float = 0.9
    thresh_mode: str = "soft"
    nproc: int = 1
    asize: int = None
    n_segments: int = 4
    azimuth_overlap: int = None
    radius_int: int = None
    random_seed: int = None
    high_pass: int = None
    collapse: str = "median"
    _algo_name: str = "llsg"
    frame_l: np.ndarray = None
    frame_s: np.ndarray = None
    frame_g: np.ndarray = None

    # TODO : write test
    @calculates("frame_final", "frame_l", "frame_s", "frame_g")
    def run(
        self,
        dataset: Optional[Dataset] = None,
        verbose: Optional[bool] = True,
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

        if self.dataset.fwhm is None:
            raise ValueError("`fwhm` has not been set")

        add_params = {
            "cube": self.dataset.cube,
            "angle_list": self.dataset.angles,
            "fwhm": self.dataset.fwhm,
            "full_output": full_output,
            "verbose": verbose,
        }

        func_params = self._setup_parameters(fkt=llsg, **add_params)

        res = llsg(**func_params, **rot_options)
        self.frame_l = res[3]
        self.frame_s = res[4]
        self.frame_g = res[5]

        self.frame_final = self.frame_s

        if self.results is not None:
            self.results.register_session(
                params=func_params, frame=self.frame_final, algo_name=self._algo_name
            )


LLSGBuilder = dataclass_builder(PPLLSG)
