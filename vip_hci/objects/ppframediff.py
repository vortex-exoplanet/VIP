#! /usr/bin/env python
"""Module for the post-processing frame differencing algorithm."""

__author__ = "Thomas Bédrine"
__all__ = [
    "FrameDiffBuilder",
]

from dataclasses import dataclass
from typing import Optional

from dataclass_builder import dataclass_builder

from .dataset import Dataset
from .postproc import PostProc
from ..psfsub import frame_diff
from ..config.utils_conf import algo_calculates_decorator as calculates


@dataclass
class PPFrameDiff(PostProc):
    """
    Post-processing frame differencing algorithm.

    Parameters
    ----------
    dataset : Dataset object, optional
        A Dataset object to be processed.
    metric : str, optional
        Distance metric to be used ('cityblock', 'cosine', 'euclidean', 'l1',
        'l2', 'manhattan', 'correlation', etc). It uses the scikit-learn
        function ``sklearn.metrics.pairwise.pairwise_distances`` (check its
        documentation).
    dist_threshold : int, optional
        Indices with a distance larger than ``dist_threshold`` percentile will
        initially discarded.
    n_similar : None or int, optional
        If a postive integer value is given, then a median combination of
        ``n_similar`` frames will be used instead of the most similar one.
    delta_rot : float, optional
        Minimum parallactic angle distance between the pairs.
    radius_int : int, optional
        The radius of the innermost annulus. By default is 0, if >0 then the
        central circular area is discarded.
    asize : int, optional
        The size of the annuli, in pixels.
    ncomp : None or int, optional
        If a positive integer value is given, then the annulus-wise PCA low-rank
        approximation with ``ncomp`` principal components will be subtracted.
        The pairwise subtraction will be performed on these residuals.
    nproc : None or int, optional
        Number of processes for parallel computing. If None the number of
        processes will be set to cpu_count()/2. By default the algorithm works
        in single-process mode.
    imlib : str, optional
        See description in vip.preproc.frame_rotate()
    interpolation : str, optional
        See description in vip.preproc.frame_rotate()
    collapse: str, optional
        What to do with derotated residual cube? See options of
        vip.preproc.cube_collapse()
    verbose : bool, optional
        If True prints to stdout intermediate info.

    """

    metric: str = "manhattan"
    dist_threshold: int = 50
    n_similar: int = None
    delta_rot: float = 0.5
    radius_int: int = 2
    asize: int = 4
    ncomp: int = None
    imlib: str = "vip-fft"
    interpolation: str = "lanczos4"
    collapse: str = "median"
    nproc: int = 1
    verbose: bool = True
    _algo_name: str = "frame_diff"

    # TODO: write test
    @calculates("frame_final")
    def run(
        self,
        dataset: Optional[Dataset] = None,
        nproc: Optional[int] = 1,
        full_output: Optional[bool] = True,
        verbose: Optional[bool] = True,
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
            "nproc": nproc,
            "full_output": full_output,
        }

        func_params = self._setup_parameters(fkt=frame_diff, **add_params)

        res = frame_diff(**func_params, **rot_options)

        self.frame_final = res

        if self.results is not None:
            self.results.register_session(
                params=func_params, frame=self.frame_final, algo_name=self._algo_name
            )


FrameDiffBuilder = dataclass_builder(PPFrameDiff)
