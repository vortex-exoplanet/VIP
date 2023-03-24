#! /usr/bin/env python
"""Module for the post-processing LOCI algorithm."""

__author__ = "Thomas Bédrine"
__all__ = ["LOCIBuilder"]


from typing import Tuple, Optional
from dataclasses import dataclass

import numpy as np
from dataclass_builder import dataclass_builder

from .dataset import Dataset
from .postproc import PostProc
from ..psfsub import xloci
from ..config.utils_conf import algo_calculates_decorator as calculates


@dataclass
class PPLOCI(PostProc):
    """
    Post-processing LOCI algorithm.

    Parameters
    ----------
    metric : str, optional
        Distance metric to be used ('cityblock', 'cosine', 'euclidean', 'l1',
        'l2', 'manhattan', 'correlation', etc). It uses the scikit-learn
        function ``sklearn.metrics.pairwise.pairwise_distances`` (check its
        documentation).
    dist_threshold : int, optional
        Indices with a distance larger than ``dist_threshold`` percentile will
        initially discarded. 100 by default.
    delta_rot : float or tuple of floats, optional
        Factor for adjusting the parallactic angle threshold, expressed in
        FWHM. Default is 1 (excludes 1 FHWM on each side of the considered
        frame). If a tuple of two floats is provided, they are used as the lower
        and upper intervals for the threshold (grows linearly as a function of
        the separation).
    delta_sep : float or tuple of floats, optional
        The threshold separation in terms of the mean FWHM (for ADI+mSDI data).
        If a tuple of two values is provided, they are used as the lower and
        upper intervals for the threshold (grows as a function of the
        separation).
    radius_int : int, optional
        The radius of the innermost annulus. By default is 0, if >0 then the
        central circular region is discarded.
    asize : int, optional
        The size of the annuli, in pixels.
    n_segments : int or list of int or 'auto', optional
        The number of segments for each annulus. When a single integer is given
        it is used for all annuli. When set to 'auto', the number of segments is
        automatically determined for every annulus, based on the annulus width.
    nproc : None or int, optional
        Number of processes for parallel computing. If None the number of
        processes will be set to cpu_count()/2. By default the algorithm works
        in single-process mode.
    solver : {'lstsq', 'nnls'}, str optional
        Choosing the solver of the least squares problem. ``lstsq`` uses the
        standard scipy least squares solver. ``nnls`` uses the scipy
        non-negative least-squares solver.
    tol : float, optional
        Valid when ``solver`` is set to lstsq. Sets the cutoff for 'small'
        singular values; used to determine effective rank of a. Singular values
        smaller than ``tol * largest_singular_value`` are considered zero.
        Smaller values of ``tol`` lead to smaller residuals (more aggressive
        subtraction).
    optim_scale_fact : float, optional
        If >1, the least-squares optimization is performed on a larger segment,
        similar to LOCI. The optimization segments share the same inner radius,
        mean angular position and angular width as their corresponding
        subtraction segments.
    adimsdi : {'skipadi', 'double'}, str optional
        Changes the way the 4d cubes (ADI+mSDI) are processed.

        ``skipadi``: the multi-spectral frames are rescaled wrt the largest
        wavelength to align the speckles and the least-squares model is
        subtracted on each spectral cube separately.

        ``double``: a first subtraction is done on the rescaled spectral frames
        (as in the ``skipadi`` case). Then the residuals are processed again in
        an ADI fashion.

    imlib : str, optional
        See the documentation of the ``vip_hci.preproc.frame_rotate`` function.
    interpolation : str, optional
        See the documentation of the ``vip_hci.preproc.frame_rotate`` function.
    collapse : {'median', 'mean', 'sum', 'trimmean'}, str optional
        Sets the way of collapsing the frames for producing a final image.

    """

    metric: str = "manhattan"
    dist_threshold: int = 90
    delta_rot: Tuple[float] = (0.1, 1)
    delta_sep: Tuple[float] = (0.1, 1)
    radius_int: int = 0
    asize: int = 4
    n_segments: int = 4
    solver: str = "lstsq"
    nproc: int = None
    tol: float = 1e-2
    optim_scale_fact: int = 2
    adimsdi: str = "skipadi"
    imlib: str = "vip-fft"
    interpolation: str = "lanczos4"
    collapse: str = "median"
    _algo_name: str = "xloci"
    cube_res: np.ndarray = None
    cube_der: np.ndarray = None

    # TODO: write test
    @calculates("frame_final", "cube_res", "cube_der")
    def run(
        self,
        dataset: Optional[Dataset] = None,
        nproc: Optional[int] = 1,
        verbose: Optional[bool] = True,
        full_output: Optional[bool] = True,
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
        self._update_dataset(dataset)

        if self.dataset.fwhm is None:
            raise ValueError("`fwhm` has not been set")

        add_params = {
            "cube": self.dataset.cube,
            "angle_list": self.dataset.angles,
            "fwhm": self.dataset.fwhm,
            "scale_list": self.dataset.wavelengths,
            "nproc": nproc,
            "full_output": full_output,
        }

        func_params = self._setup_parameters(fkt=xloci, **add_params)

        res = xloci(**func_params, **rot_options)

        self.cube_res, self.cube_der, self.frame_final = res

        if self.results is not None:
            self.results.register_session(
                frame=self.frame_final, params=func_params, algo_name=self._algo_name
            )


LOCIBuilder = dataclass_builder(PPLOCI)
