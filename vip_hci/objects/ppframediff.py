#! /usr/bin/env python
"""Module for the post-processing frame differencing algorithm."""

__author__ = "Thomas Bédrine"
__all__ = [
    "PPFrameDiff",
]

from .postproc import PostProc
from ..psfsub import frame_diff
from ..config.utils_conf import algo_calculates_decorator as calculates


class PPFrameDiff(PostProc):
    """Post-processing frame differencing algorithm."""

    def __init__(
        self,
        dataset=None,
        metric="manhattan",
        dist_threshold=50,
        n_similar=None,
        delta_rot=0.5,
        radius_int=2,
        asize=4,
        ncomp=None,
        imlib="vip-fft",
        interpolation="lanczos4",
        collapse="median",
        nproc=1,
        verbose=True,
    ):
        """
        Set up the frame differencing algorithm parameters.

        Parameters
        ----------
        dataset : Dataset object
            A Dataset object to be processed.
        metric : str, optional
            Distance metric to be used ('cityblock', 'cosine', 'euclidean', 'l1',
            'l2', 'manhattan', 'correlation', etc). It uses the scikit-learn
            function ``sklearn.metrics.pairwise.pairwise_distances`` (check its
            documentation).
        dist_threshold : int
            Indices with a distance larger than ``dist_threshold`` percentile will
            initially discarded.
        n_similar : None or int, optional
            If a postive integer value is given, then a median combination of
            ``n_similar`` frames will be used instead of the most similar one.
        delta_rot : int
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
        imlib : str, opt
            See description in vip.preproc.frame_rotate()
        interpolation : str, opt
            See description in vip.preproc.frame_rotate()
        collapse: str, opt
            What to do with derotated residual cube? See options of
            vip.preproc.cube_collapse()
        verbose : bool, optional
            If True prints to stdout intermediate info.

        """
        super(PPFrameDiff, self).__init__(locals())

    @calculates("frame_final")
    def run(self, dataset=None, nproc=1, full_output=True, verbose=True, **rot_options):
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
        dataset = self._get_dataset(dataset, verbose)

        if dataset.fwhm is None:
            raise ValueError("`fwhm` has not been set")

        add_params = {
            "cube": dataset.cube,
            "angle_list": dataset.angles,
            "fwhm": dataset.fwhm,
            "nproc": nproc,
            "full_output": full_output,
        }

        func_params = self._setup_parameters(fkt=frame_diff, **add_params)

        res = frame_diff(**func_params, **rot_options)

        self.frame_final = res
