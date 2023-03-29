#! /usr/bin/env python
"""Module for the post-processing ANDROMEDA algorithm."""

__author__ = "Thomas Bédrine, Carlos Alberto Gomez Gonzalez, Ralf Farkas"
__all__ = [
    "AndroBuilder",
]

from typing import Optional
from dataclasses import dataclass

import numpy as np
from dataclass_builder import dataclass_builder

from .dataset import Dataset
from .postproc import PostProc
from ..invprob import andromeda
from ..config.utils_conf import algo_calculates_decorator as calculates


@dataclass
class PPAndromeda(PostProc):
    r"""
    Post-processing ANDROMEDA algorithm.

    Parameters
    ----------
    oversampling_fact : float
        Oversampling factor for the wavelength corresponding to the filter used
        to obtain ``cube`` (defined as the ratio between the wavelength of
        the filter and the Shannon wavelength). Usually above 1 and below 3.
        Note that in ANDROMEDA everything is coded in lambda/D unit so this is
        an important parameter. See Note for example calculation.
        IDL parameter: ``OVERSAMPLING_1_INPUT``
    filtering_fraction : float, optional
        Strength of the high-pass filter. If set to ``1``, no high-pass filter
        is used.
        IDL parameter: ``FILTERING_FRACTION_INPUT``
    min_sep : float, optional
        Angular separation is assured to be above ``min_sep*lambda/D``.
        IDL parameter: ``MINIMUM_SEPARATION_INPUT``
    annuli_width : float, optional
        Annuli width on which the subtraction are performed. The same for all
        annuli.
        IDL parameter: ``ANNULI_WIDTH_INPUT``
    roa : float, optional
        Ratio of the optimization area. The optimization annulus area is defined
        by ``roa * annuli_width``.
        ``roa`` is forced to ``1`` when ``opt_method="no"`` is chosen.
        IDL parameter: ``RATIO_OPT_AREA_INPUT``
    opt_method : {'no', 'total', 'lsq', 'robust'}, optional
        Method used to balance for the flux difference that exists between the
        two subtracted annuli in an optimal way during ADI.
        IDL parameter: ``OPT_METHOD_ANG_INPUT``
    nsmooth_snr : int, optional
        Number of pixels over which the radial robust standard deviation profile
        of the SNR map is smoothed to provide a global trend for the SNR map
        normalization. For ``nsmooth_snr=0`` the SNR map normalization is
        disabled.
        IDL parameter: ``NSMOOTH_SNR_INPUT``
    iwa : float or None, optional
        Inner working angle / inner radius of the first annulus taken into
        account, expressed in ``lambda/D``. If ``None``, it is chosen
        automatically between the values ``0.5``, ``4`` or ``0.25``.
        IDL parameter: ``IWA_INPUT``
    owa : float, optional
        Outer working angle / **inner** radius of the last annulus, expressed in
        ``lambda/D``. If ``None``, the value is automatically chosen based on
        the frame size.
        IDL parameter: ``OWA_INPUT``
    precision : int, optional
        Number of shifts applied to the PSF. Passed to
        ``calc_psf_shift_subpix`` , which then creates a 4D cube with shape
        (precision+1, precision+1, N, N).
        IDL parameter: ``PRECISION_INPUT``
    fast : float or bool, optional
        Size of the annuli from which the speckle noise should not be dominant
        anymore, in multiples of ``lambda/D``. If ``True``, a value of
        ``20 lambda/D`` is used, ``False`` (the default) disables the fast mode
        entirely. Above this threshold, the annuli width is set to
        ``4*annuli_width``.
        IDL parameter: ``FAST``
    homogeneous_variance : bool, optional
        If set, variance is treated as homogeneous and is calculated as a mean
        of variance in each position through time.
        IDL parameter: ``HOMOGENEOUS_VARIANCE_INPUT``
    ditimg : float, optional
        DIT for images (in sec)
        IDL Parameter: ``DITIMG_INPUT``
    ditpsf : float or None, optional
        DIT for PSF (in sec)
        IDL Parameter: ``DITPSF_INPUT``
        If set to ``None``, the value of ``ditimg`` is used.
    tnd : float, optional
        Neutral Density Transmission.
        IDL parameter: ``TND_INPUT``
    total : bool, optional
        ``total=True`` is the old behaviour (normalizing the PSF to its sum).
        IDL parameter: ``TOTAL`` (was ``MAX`` in previous releases).
    multiply_gamma : bool, optional
        Use gamma for signature computation too.
        IDL parameter: ``MULTIPLY_GAMMA_INPUT``
    verbose : bool, optional
        Print some parameter values for control.
        IDL parameter: ``VERBOSE``

    """

    oversampling_fact: float = None
    filtering_fraction: float = 0.25
    min_sep: float = 0.5
    annuli_width: float = 1.0
    roa: float = 2.0
    opt_method: str = "lsq"
    nsmooth_snr: int = 18
    iwa: float = None
    owa: float = None
    precision: int = 50
    fast: bool = False
    homogeneous_variance: bool = True
    ditimg: float = 1.0
    ditpsf: float = None
    tnd: float = 1.0
    total: bool = False
    multiply_gamma: bool = True
    verbose: bool = True
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
        nproc: Optional[int] = 1,
        verbose: Optional[bool] = True,
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
        self._update_dataset(dataset)

        if self.dataset.fwhm is None:
            raise ValueError("`fwhm` has not been set")

        add_params = {
            "cube": self.dataset.cube,
            "angles": self.dataset.angles,
            "psf": self.dataset.psf,
            "fwhm": self.dataset.fwhm,
            "nproc": nproc,
        }

        func_params = self._setup_parameters(fkt=andromeda, **add_params)
        res = andromeda(**func_params)

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
                params=func_params,
                frame=self.frame_final,
                algo_name=self._algo_name,
                snr_map=self.snr_map,
            )


AndroBuilder = dataclass_builder(PPAndromeda)
