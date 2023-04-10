#! /usr/bin/env python
"""Module for the post-processing FMMF algorithm."""

__author__ = "Thomas Bédrine"
__all__ = [
    "FMMFBuilder",
]

from typing import Optional
from dataclasses import dataclass, field

import numpy as np
from dataclass_builder import dataclass_builder

from .dataset import Dataset
from .postproc import PostProc
from ..invprob import fmmf
from ..config.utils_conf import algo_calculates_decorator as calculates


@dataclass
class PPFMMF(PostProc):
    """
    Post-processing forward model matching filter algorithm.

    Parameters
    ----------
    min_r : int, optional
        Center radius of the first annulus considered in the FMMF detection
        map estimation. The radius should be larger than half
        the value of the 'crop' parameter . Default is None which
        corresponds to one FWHM.
    max_r : int, optional
        Center radius of the last annulus considered in the FMMF detection
        map estimation. The radius should be smaller or equal to half the
        size of the image minus half the value of the 'crop' parameter.
        Default is None which corresponds to half the size of the image
        minus half the value of the 'crop' parameter.
    model: {'KLIP', 'LOCI'}, optional
        Selected PSF-subtraction technique for the computation of the FMMF
        detection map. FMMF work either with KLIP or LOCI. Default is 'KLIP'.
    var: {'FR', 'FM', 'TE'}, optional
        Model used for the residual noise variance estimation used in the
        matched filtering (maximum likelihood estimation of the flux and SNR).
        Three different approaches are proposed:

        * 'FR': consider the pixels in the selected annulus with a width equal
          to asize but separately for every frame.
        * 'FM': consider the pixels in the selected annulus with a width
          equal to asize but separately for every frame. Apply a mask one FWHM
          on the selected pixel and its surrounding.
        * 'TE': rely on the method developped in PACO to estimate the
          residual noise variance (take the pixels in a region of one FWHM
          arround the selected pixel, considering every frame in the
          derotated cube of residuals except for the selected frame)
    param: dict, optional
        Dictionnary regrouping the parameters used by the KLIP (ncomp and
        delta_rot) or LOCI (tolerance and delta_rot) PSF-subtraction
        technique:

        * ncomp : int, optional. Number of components used for the low-rank
          approximation of the speckle field. Default is 20.
        * tolerance: float, optional. Tolerance level for the approximation of
          the speckle field via a linear combination of the reference images in
          the LOCI algorithm. Default is 5e-3.
        * delta_rot : float, optional. Factor for tunning the parallactic angle
          threshold, expressed in FWHM. Default is 0.5 (excludes 0.5xFHWM on each
          side of the considered frame).
    crop: int, optional
        Part of the PSF template considered in the estimation of the FMMF
        detection map. Default is 5.
    imlib : str, optional
        Parameter used for the derotation of the residual cube. See the
        documentation of the ``vip_hci.preproc.frame_rotate`` function.
    interpolation : str, optional
        Parameter used for the derotation of the residual cube. See the
        documentation of the ``vip_hci.preproc.frame_rotate`` function.
    nproc : int or None, optional
        Number of processes for parallel computing. By default ('nproc=1')
        the algorithm works in single-process mode. If set to None, nproc
        is automatically set to half the number of available CPUs.

    """

    min_r: int = None
    max_r: int = None
    model: str = "KLIP"
    var: str = "FR"
    param: dict = field(
        default_factory=lambda: {"ncomp": 20, "tolerance": 5e-3, "delta_rot": 0.5}
    )
    crop: int = 5
    imlib: str = "vip-fft"
    interpolation: str = "lanczos4"
    nproc: int = 1
    _algo_name: str = "fmmf"
    snr_map: np.ndarray = None

    # TODO: write test
    @calculates("frame_final", "snr_map")
    def run(
        self,
        dataset: Optional[Dataset] = None,
        verbose: Optional[bool] = True,
    ):
        """
        Run the post-processing FMMF algorithm for model PSF subtraction.

        Parameters
        ----------
        dataset : Dataset object
            An Dataset object to be processed.
        model: {'KLIP', 'LOCI'}, optional
            If you want to change the default model. See documentation above for more
            information.
        nproc : None or int, optional
            Number of processes for parallel computing. If None the number of
            processes will be set to cpu_count()/2. By default the algorithm works
            in single-process mode.
        verbose : bool, optional
            If True prints to stdout intermediate info.

        """
        self._update_dataset(dataset)

        if self.dataset.fwhm is None:
            raise ValueError("`fwhm` has not been set")

        add_params = {
            "cube": self.dataset.cube,
            "pa": self.dataset.angles,
            "psf": self.dataset.psf,
            "fwhm": self.dataset.fwhm,
            "scale_list": self.dataset.wavelengths,
        }

        func_params = self._setup_parameters(fkt=fmmf, **add_params)

        res = fmmf(**func_params)

        self.frame_final, self.snr_map = res

        if self.results is not None:
            self.results.register_session(
                params=func_params,
                frame=self.frame_final,
                snr_map=self.snr_map,
                algo_name=self._algo_name,
            )

    def make_snrmap(self):
        """
        Do nothing, snr_map is already generated by ``fmmf``.

        This function must overload the inherited one to avoid misusage.

        """


FMMFBuilder = dataclass_builder(PPFMMF)
