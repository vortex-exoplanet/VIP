#! /usr/bin/env python
"""Module for the post-processing median subtraction algorithm."""

__author__ = "Thomas Bédrine"
__all__ = ["MedianBuilder"]

from typing import Tuple, Optional
from dataclasses import dataclass

import numpy as np
from dataclass_builder import dataclass_builder

from .dataset import Dataset
from .postproc import PostProc
from ..psfsub import median_sub
from ..config.utils_conf import algo_calculates_decorator as calculates


@dataclass
class PPMedianSub(PostProc):
    """
    Dataclass used for the storage and typing of the PPMedianSub object.

    Parameters
    ----------
    flux_sc_list : numpy ndarray, 1d
        In the case of IFS data (ADI+SDI), this is the list of flux scaling
        factors applied to each spectral frame after geometrical rescaling.
        These should be set to either the ratio of stellar fluxes between the
        last spectral channel and the other channels, or to the second output
        of `preproc.find_scal_vector` (when using 2 free parameters). If not
        provided, the algorithm will still work, but with a lower efficiency
        at subtracting the stellar halo.
    radius_int : int, optional
        The radius of the innermost annulus. By default is 0, if >0 then the
        central circular area is discarded.
    asize : int, optional
        The size of the annuli, in pixels.
    delta_rot : float, optional
        Factor for increasing the parallactic angle threshold, expressed in
        FWHM. Default is 1 (excludes 1 FHWM on each side of the considered
        frame).
    delta_sep : float or tuple of floats, optional
        The threshold separation in terms of the mean FWHM (for ADI+mSDI data).
        If a tuple of two values is provided, they are used as the lower and
        upper intervals for the threshold (grows as a function of the
        separation).
    mode : {'fullfr', 'annular'}, str optional
        In ``fullfr`` mode only the median frame is subtracted, in ``annular``
        mode also the 4 closest frames given a PA threshold (annulus-wise) are
        subtracted.
    nframes : int or None, optional
        Number of frames (even value) to be used for building the optimized
        reference PSF when working in ``annular`` mode. None by default, which
        means that all frames, excluding the thresholded ones, are used.
    sdi_only: bool, optional
        In the case of IFS data (ADI+SDI), whether to perform median-SDI, or
        median-ASDI (default).
    imlib : str, optional
        See the documentation of the ``vip_hci.preproc.frame_rotate`` function.
    interpolation : str, optional
        See the documentation of the ``vip_hci.preproc.frame_rotate`` function.
    collapse : {'median', 'mean', 'sum', 'trimmean'}, str optional
        Sets the way of collapsing the frames for producing a final image.
    verbose : bool, optional
        If True prints to stdout intermediate info.

    """

    flux_sc_list: np.ndarray = None
    radius_int: int = 0
    asize: int = 4
    delta_rot: float = 1
    delta_sep: Tuple[float] = (0.1, 1)
    mode: str = "fullfr"
    nframes: int = 4
    sdi_only: bool = False
    imlib: str = "vip-fft"
    interpolation: str = "lanczos4"
    collapse: str = "median"
    verbose: bool = True
    _algo_name: str = "median_sub"
    cube_residuals: np.ndarray = None
    cube_residuals_der: np.ndarray = None

    # TODO: write test
    @calculates("cube_residuals", "cube_residuals_der", "frame_final")
    def run(
        self,
        dataset: Optional[Dataset] = None,
        nproc: Optional[int] = 1,
        full_output: Optional[bool] = True,
        verbose: Optional[bool] = True,
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
        self._update_dataset(dataset)

        if self.mode == "annular" and self.dataset.fwhm is None:
            raise ValueError("`fwhm` has not been set")

        add_params = {
            "cube": self.dataset.cube,
            "angle_list": self.dataset.angles,
            "fwhm": self.dataset.fwhm,
            "scale_list": self.dataset.wavelengths,
            "nproc": nproc,
            "full_output": full_output,
        }

        func_params = self._setup_parameters(fkt=median_sub, **add_params)

        res = median_sub(**func_params, **rot_options)

        self.cube_residuals, self.cube_residuals_der, self.frame_final = res

        if self.results is not None:
            self.results.register_session(
                params=func_params, frame=self.frame_final, algo_name=self._algo_name
            )


MedianBuilder = dataclass_builder(PPMedianSub)
