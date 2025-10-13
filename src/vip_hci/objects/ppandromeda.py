#! /usr/bin/env python
"""Module for the post-processing ANDROMEDA algorithm."""

__author__ = "Thomas Bédrine, Carlos Alberto Gomez Gonzalez, Ralf Farkas"
__all__ = ["AndroBuilder", "PPAndromeda"]

from typing import Optional
from dataclasses import dataclass

import numpy as np
from dataclass_builder import dataclass_builder

from .dataset import Dataset
from .postproc import PostProc
from ..invprob import andromeda, ANDROMEDA_Params
from ..config.utils_conf import algo_calculates_decorator as calculates


@dataclass
class PPAndromeda(PostProc, ANDROMEDA_Params):
    """Post-processing ANDROMEDA algorithm."""

    _algo_name: str = "andromeda"
    contrast_map: np.ndarray = None
    likelihood_map: np.ndarray = None
    snr_map: np.ndarray = None
    snr_map_notnorm: np.ndarray = None
    stdcontrast_map: np.ndarray = None
    stdcontrast_map_notnorm: np.ndarray = None
    ext_radius: int = None
    detection_map: np.ndarray = None

    @calculates(
        "frame_final",
        "contrast_map",
        "likelihood_map",
        "snr_map",
        "stdcontrast_map",
        "snr_map_notnorm",
        "stdcontrast_map_notnorm",
        "ext_radius",
        "detection_map",
    )
    def run(
        self,
        dataset: Optional[Dataset] = None,
        nproc: Optional[int] = None,
    ):
        """
        Run the ANDROMEDA algorithm for model PSF subtraction.

        Parameters
        ----------
        dataset : Dataset, optional
            Dataset to process. If not provided, ``self.dataset`` is used (as
            set when initializing this object).
        nproc : int, optional
            Number of processes to use.
        verbose : bool, optional
            Print some parameter values for control.

        """
        self.snr_map = None
        self._update_dataset(dataset)

        if self.dataset.fwhm is None:
            raise ValueError("`fwhm` has not been set")

        self._explicit_dataset()

        if nproc is not None:
            self.nproc = nproc

        params_dict = self._create_parameters_dict(ANDROMEDA_Params)
        all_params = {"algo_params": self}
        res = andromeda(**all_params)

        self.contrast_map = res[0]
        self.likelihood_map = res[5]
        self.ext_radius = res[6]

        # normalized/not normalized depending on nsmooth:
        self.snr_map = res[2]
        self.stdcontrast_map = res[4]
        if self.nsmooth_snr != 0:
            self.snr_map_notnorm = res[1]
            self.stdcontrast_map_notnorm = res[3]

        # general attributes:
        self.frame_final = self.contrast_map
        self.detection_map = self.snr_map

        if self.results is not None:
            self.results.register_session(
                params=params_dict,
                frame=self.frame_final,
                algo_name=self._algo_name,
                snr_map=self.snr_map,
            )


AndroBuilder = dataclass_builder(PPAndromeda)
