"""
Implementation of the ANDROMEDA algorithm from [MUG09]_ / [CAN15]_.

Based on ANDROMEDA v3.1 from 28/06/2018.

.. [MUG09]
   | Mugnier et al, 2009
   | **Optimal method for exoplanet detection by angular differential imaging**
   | *J. Opt. Soc. Am. A, 26(6), 1326-1334*
   | `doi:10.1364/JOSAA.26.001326 <http://doi.org/10.1364/JOSAA.26.001326>`_

.. [CAN15]
   | Cantalloube et al, 2015
   | **Direct exoplanet detection and characterization using the ANDROMEDA
     method: Performance on VLT/NaCo data**
   | *A&A, Volume 582, p. 89*
   | `https://arxiv.org/abs/1508.06406
     <https://arxiv.org/abs/1508.06406>`_

"""

__author__ = "Thomas Bédrine, Ralf Farkas"
__all__ = ["andromeda", "ANDROMEDA_Params"]

import numpy as np
from dataclasses import dataclass
from enum import Enum

from typing import Union, List

from ..config.utils_param import setup_parameters, separate_kwargs_dict
from ..config.utils_conf import pool_map, iterable
from ..config.paramenum import OptMethod, ALGO_KEY
from ..var.filters import frame_filter_highpass, cube_filter_highpass
from ..var import dist_matrix

from .utils_andro import (
    calc_psf_shift_subpix,
    fitaffine,
    idl_round,
    idl_where,
    robust_std,
    subpixel_shift,
)

global CUBE


@dataclass
class ANDROMEDA_Params:
    """
    Set of parameters for the ANDROMEDA algorithm.

    See function `andromeda` below for the documentation.
    """

    cube: np.ndarray = None
    oversampling_fact: float = None
    angle_list: np.ndarray = None
    psf: np.ndarray = None
    filtering_fraction: float = 0.25
    min_sep: float = 0.5
    annuli_width: float = 1.0
    roa: float = 2
    opt_method: Enum = OptMethod.LSQ
    nsmooth_snr: int = 18
    iwa: float = None
    owa: float = None
    precision: int = 50
    fast: Union[float, bool] = False
    homogeneous_variance: bool = True
    ditimg: float = 1.0
    ditpsf: float = None
    tnd: float = 1.0
    total: bool = False
    multiply_gamma: bool = True
    nproc: int = 1
    verbose: bool = False


def andromeda(*all_args: List, **all_kwargs: dict):
    r"""
    Exoplanet detection in ADI sequences by maximum-likelihood approach.

    This is as implemented in [CAN15]_, itself inspired by the framework
    presented in [MUG09]_.

    Parameters
    ----------
    all_args: list, optional
        Positionnal arguments for the andromeda algorithm. Full list of
        parameters below.
    all_kwargs: dictionary, optional
        Mix of keyword arguments that can initialize a AndroParams or a
        AndroParams itself.

    Andromeda parameters
    ----------
    cube : 3d numpy ndarray
        Input cube.
        IDL parameter: ``IMAGES_1_INPUT``
    oversampling_fact : float
        Oversampling factor for the wavelength corresponding to the filter used
        to obtain ``cube`` (defined as the ratio between the wavelength of
        the filter and the Shannon wavelength). Usually above 1 and below 3.
        Note that in ANDROMEDA everything is coded in lambda/D unit so this is
        an important parameter. See Note for example calculation.
        IDL parameter: ``OVERSAMPLING_1_INPUT``
    angle_list : numpy ndarray
        List of derotation angles associated with each frame in ``cube``. Note
        that, compared to the IDL version, the PA convention is different: If
        you would pass ``[1,2,3]`` to the IDL version (parallactic angles), you
        should pass ``[-1,-2, -3]`` (derotation angles) to this function to
        obtain the same results.
        IDL parameter: ``- ANGLES_INPUT``
    psf : 2d numpy ndarray
        The experimental PSF used to model the planet signature in the
        subtracted images. This PSF is usually a non-coronographic or saturated
        observation of the target star.
        IDL parameter: ``PSF_PLANET_INPUT``
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
    nproc : int, optional
        Number of processes to use.
    verbose : bool, optional
        Print some parameter values for control.
        IDL parameter: ``VERBOSE``

    Returns
    -------
    contrast : 2d ndarray
        Calculated contrast map.
        (IDL return value)
    snr : 2d ndarray
        Signal to noise ratio map (defined as the estimated contrast divided by
        the estimated standard deviation of the contrast).
        IDL parameter: ``SNR_OUTPUT``
    snr_norm : 2d ndarray
        IDL parameter: ``SNR_NORM_OUTPUT``
    stdcontrast : 2d ndarray
        Map of the estimated standard deviation of the contrast.
        IDL parameter: `STDDEVCONTRAST_OUTPUT`` (previously
        ``STDEVFLUX_OUTPUT``)
    stdcontrast_norm : 2d ndarray
        Map of the estimated normalized standard deviation of the contrast.
        IDL parameter: ``STDDEVCONTRAST_NORM_OUTPUT``
    likelihood : 2d ndarray
        likelihood
        IDL parameter: ``LIKELIHOOD_OUTPUT``
    ext_radius : float
        Edge of the SNR map. Slightly decreased due to the normalization
        procedure. Useful to a posteriori reject potential companions that are
        too close to the edge to be analyzed.
        IDL parameter: ``EXT_RADIUS_OUTPUT``

    Notes
    -----
    IDL outputs:

    - SNR_OUTPUT
    - SNR_NORM_OUTPUT
    - LIKELIHOOD_OUTPUT
    - STDDEVCONTRAST_OUTPUT (was STDEVFLUX_OUTPUT)
    - STDDEVCONTRAST_NORM_OUTPUT

    The following IDL parameters were not implemented:

        - SDI-related parameters
            - IMAGES_2_INPUT
            - OVERSAMPLING_2_INPUT
            - OPT_METHOD_SPEC_INPUT
        - ROTOFF_INPUT
        - recentering (should be done in VIP before):
            - COORD_CENTRE_1_INPUT
            - COORD_CENTRE_2_INPUT
        - debug/expert testing testing
            - INDEX_NEG_INPUT
            - INDEX_POS_INPUT
            - ANNULI_LIMITS_INPUT
        - other
            - DISPLAY
            - VERSION
            - HELP
        - return parameters
            - IMAGES_1_CENTRED_OUTPUT
            - IMAGES_2_RESCALED_OUTPUT
            - VARIANCE_1_CENTRED_OUTPUT
            - VARIANCE_2_RESCALED_OUTPUT
            - GAMMA_INFO_OUTPUT
        - variances (VARIANCE_1_INPUT, VARIANCE_2_INPUT)

    Note
    ----
    The oversampling factor can be computed as:\
    :math:`oversampling = plsc_{NYQ} / plsc`
    , where:\
        :math:`plsc = 12.25` [mas/px] for SPHERE/IRDIS\
        :math:`plsc_{NYQ} = (0.5*lambda/diam_tel)/pi*180*3600*1e3` [mas/px]\
        lambda = 3.8e-6  : Imaging wavelength [m]\
        diam_tel = 8.0   : Telescope diameter [m]

    """
    class_params, other_options = separate_kwargs_dict(
        initial_kwargs=all_kwargs, parent_class=ANDROMEDA_Params
    )

    # Extracting the object of parameters (if any)
    algo_params = None
    if ALGO_KEY in other_options.keys():
        algo_params = other_options[ALGO_KEY]
        del other_options[ALGO_KEY]

    if algo_params is None:
        algo_params = ANDROMEDA_Params(*all_args, **class_params)

    def info(msg, *fmt, **kwfmt):
        if algo_params.verbose:
            print(msg.format(*fmt, **kwfmt))

    def info2(msg, *fmt, **kwfmt):
        if algo_params.verbose == 2:
            print(msg.format(*fmt, **kwfmt))

    global CUBE  # assigned after high-pass filter

    # ===== verify input

    # the andromeda algorithm handles PAs differently from the other algos in
    # VIP. This normalizes the API:
    algo_params.angle_list = -algo_params.angle_list

    andro_cube = np.zeros_like(algo_params.cube)

    if andro_cube.shape[-1] % 2 == 1:
        # shift and crop
        for idx, img in enumerate(algo_params.cube):
            andro_cube[idx] = subpixel_shift(img, 0.5, 0.5)
        andro_cube = andro_cube[:, 1:, 1:]
    else:
        # shifting due to new VIP convention for even-sized images
        for idx, img in enumerate(algo_params.cube):
            andro_cube[idx] = subpixel_shift(img, -0.5, -0.5)

    if algo_params.psf.shape[0] % 2 == 1:
        # shift and crop
        algo_params.psf = subpixel_shift(algo_params.psf, 0.5, 0.5)
        algo_params.psf = algo_params.psf[1:, 1:]
    else:
        # shifting due to new VIP convention for even-sized images
        algo_params.psf = subpixel_shift(algo_params.psf, -0.5, -0.5)

    if algo_params.filtering_fraction > 1 or algo_params.filtering_fraction < 0:
        raise ValueError("``filtering_fraction`` must be between 0 and 1")

    frames, npix, _ = andro_cube.shape
    npixpsf, _ = algo_params.psf.shape

    # ===== set default parameters:

    if algo_params.opt_method != "no":
        if algo_params.roa < 1:
            raise ValueError(
                "The optimization to subtraction area ``roa`` " "must be >= 1"
            )

    else:
        algo_params.roa = 1

    if algo_params.iwa is None:
        for test_iwa in [0.5, 4, 0.25]:
            # keep first IWA which produces frame pairs
            test_ang = 2 * np.arcsin(algo_params.min_sep /
                                     (2 * test_iwa)) * 180 / np.pi
            test_id, _, _ = create_indices(
                algo_params.angle_list, angmin=test_ang)
            if test_id is not None:  # pairs found
                break

        algo_params.iwa = test_iwa
        info("iwa automatically set to {}*lambda/D", algo_params.iwa)

    if algo_params.owa is None:
        algo_params.owa = (npix / 2 - npixpsf / 2) / \
            (2 * algo_params.oversampling_fact)
        info("owa automatically set to {} (based on frame size)",
             algo_params.owa)
    else:
        # radius of the last annulus taken into account for process [lambda/D]:
        algo_params.owa -= (npixpsf / 2) / (2 * algo_params.oversampling_fact)

    if algo_params.owa <= algo_params.iwa - algo_params.annuli_width:
        raise ValueError("You must increase `owa` or decrease `iwa`")

    if algo_params.fast is False:
        pass
    elif algo_params.fast is True:  # IDL: IF fast EQ 1.0
        algo_params.fast = 20  # [lambda/D]
        if algo_params.owa > algo_params.fast:
            dmean = algo_params.fast
        else:
            algo_params.fast = 0
        if algo_params.iwa > algo_params.fast:
            dmean = algo_params.owa
    else:
        if algo_params.owa > algo_params.fast:
            dmean = algo_params.fast
        else:
            algo_params.fast = 0

    if algo_params.fast:
        info(
            "annuli_width is set to {} from {} lambda/D",
            4 * algo_params.annuli_width,
            dmean,
        )

    # contrast maps:
    if algo_params.ditpsf is None:
        algo_params.ditpsf = algo_params.ditimg

    if np.asarray(algo_params.tnd).ndim == 0:  # int or float
        info2("Throughput map: Homogeneous transmission: {}%",
              algo_params.tnd * 100)
    else:  # TODO: test if really 2d map?
        info2("Throughput map: Inhomogeneous 2D throughput map given.")

    if algo_params.nsmooth_snr != 0 and algo_params.nsmooth_snr < 2:
        raise ValueError("`nsmooth_snr` must be >= 2")

    # ===== info output
    if algo_params.filtering_fraction == 1:
        info("No high-pass pre-filtering of the images!")

    # ===== initialize output

    flux = np.zeros_like(andro_cube[0])
    snr = np.zeros_like(andro_cube[0])
    likelihood = np.zeros_like(andro_cube[0])
    stdflux = np.zeros_like(andro_cube[0])

    # ===== pre-processing

    # normalization...
    if algo_params.total:
        psf_scale_factor = np.sum(algo_params.psf)
    else:
        psf_scale_factor = np.max(algo_params.psf)

    # creates new array in memory (prevent overwriting of input parameters)
    algo_params.psf = algo_params.psf / psf_scale_factor

    # ...and spatial filterin on the PSF:
    if algo_params.filtering_fraction != 1:
        algo_params.psf = frame_filter_highpass(
            algo_params.psf, "hann", hann_cutoff=algo_params.filtering_fraction
        )

    # library of all different PSF positions
    psf_cube = calc_psf_shift_subpix(
        algo_params.psf, precision=algo_params.precision)

    # spatial filtering of the preprocessed image-cubes:
    if algo_params.filtering_fraction != 1:
        if algo_params.verbose:
            print(
                "Pre-processing filtering of the images and the PSF: "
                "done! F={}".format(algo_params.filtering_fraction)
            )
        andro_cube = cube_filter_highpass(
            andro_cube,
            mode="hann",
            hann_cutoff=algo_params.filtering_fraction,
            verbose=algo_params.verbose,
        )

    CUBE = andro_cube

    # definition of the width of each annuli (to perform ADI)
    dmin = algo_params.iwa  # size of the lowest annuli, in lambda/D
    dmax = algo_params.owa  # size of the greatest annuli, in lambda/D
    if algo_params.fast:
        first_distarray = (
            dmin
            + np.arange(
                int(np.round(np.abs(dmean - dmin - 1)) /
                    algo_params.annuli_width + 1),
                dtype=float,
            )
            * algo_params.annuli_width
        )
        second_distarray = (
            dmean
            + dmin
            - 1
            + np.arange(
                int(np.round(dmax - dmean) / (4 * algo_params.annuli_width) + 1),
                dtype=float,
            )
            * 4
            * algo_params.annuli_width
        )
        distarray_lambdaonD = np.hstack([first_distarray, second_distarray])
        if algo_params.iwa > algo_params.fast:
            distarray_lambdaonD = first_distarray
        if distarray_lambdaonD[-1] > dmax:
            distarray_lambdaonD[-1] = dmax

        annuli_limits = (
            algo_params.oversampling_fact * 2 * distarray_lambdaonD
        )  # in pixels

    else:
        distarray_lambdaonD = (
            dmin
            + np.arange(
                int(np.round(dmax - dmin) / algo_params.annuli_width + 1), dtype=float
            )
            * algo_params.annuli_width
        )
        distarray_lambdaonD[-1] = dmax
        annuli_limits = np.floor(
            algo_params.oversampling_fact * 2 * distarray_lambdaonD
        ).astype(int)

    while dmax * (2 * algo_params.oversampling_fact) < annuli_limits[-1]:
        # remove last element:
        annuli_limits = annuli_limits[:-1]  # view, not a copy!

    annuli_number = len(annuli_limits) - 1

    infomsg = "Using these user parameters, {} annuli will be processed, from a "
    infomsg += "separation of {} to {} pixels."
    info(infomsg, annuli_number, annuli_limits[0], annuli_limits[-1])

    # ===== main loop

    add_params = {
        "i": iterable(range(annuli_number)[::-1]),
        "annuli_limits": annuli_limits,
        "psf_cube": psf_cube,
    }

    func_params = setup_parameters(
        params_obj=algo_params,
        fkt=_process_annulus,
        as_list=True,
        show_params=False,
        **add_params,
    )

    res_all = pool_map(
        algo_params.nproc,
        _process_annulus,
        # start with outer annuli, they take longer:
        *func_params,
        msg="annulus",
        leave=False,
        verbose=False,
    )

    for res in res_all:
        if res is None:
            continue

        flux += res[0]
        snr += res[1]
        likelihood += res[2]
        stdflux += res[3]

    # translating into contrast:
    # flux_factor: float or 2d array, depending on tnd
    factor = 1 / psf_scale_factor
    flux_factor = factor * algo_params.tnd * \
        (algo_params.ditpsf / algo_params.ditimg)
    if algo_params.verbose:
        print("[34m", "psf_scale_factor:", psf_scale_factor, "[0m")
        print("[34m", "tnd:", algo_params.tnd, "[0m")
        print("[34m", "ditpsf:", algo_params.ditpsf, "[0m")
        print("[34m", "ditimg:", algo_params.ditimg, "[0m")
        print("[34m", "flux_factor:", flux_factor, "[0m")

    # post-processing of the output:
    if algo_params.nsmooth_snr != 0:
        if algo_params.verbose:
            print("Normalizing SNR...")

        # normalize snr map by its radial robust std:
        snr_norm, snr_std = normalize_snr(
            snr, nsmooth_snr=algo_params.nsmooth_snr, fast=algo_params.fast
        )

        # normalization of the std of the flux (same way):
        stdflux_norm = np.zeros((npix, npix))
        zone = snr_std != 0
        stdflux_norm[zone] = stdflux[zone] * snr_std[zone]

        ext_radius = annuli_limits[annuli_number - 1] / (
            2 * algo_params.oversampling_fact
        )

        # TODO: return value handling should be improved.

        return (
            flux * flux_factor,  # IDL RETURN
            snr,  # snr_output
            snr_norm,  # snr_norm_output
            stdflux * flux_factor,  # IDL stddevcontrast_output
            stdflux_norm * flux_factor,  # IDL stddevcontrast_norm_output
            likelihood,  # IDL likelihood_output
            ext_radius,
        )  # IDL ext_radius_output, [lambda/D]

        # previous return values:
        # return flux, snr_norm, likelihood, stdflux_norm, ext_radius
    else:
        ext_radius = np.floor(annuli_limits[annuli_number]) / (
            2 * algo_params.oversampling_fact
        )

        return (
            flux * flux_factor,  # IDL RETURN
            snr,  # snr_output
            snr,  # snr_norm_output
            stdflux * flux_factor,  # IDL stddevcontrast_output
            stdflux * flux_factor,  # IDL stddevcontrast_norm_output
            likelihood,  # IDL likelihood_output
            ext_radius,
        )  # IDL ext_radius_output [lambda/D]


def _process_annulus(
    i,
    annuli_limits,
    roa,
    min_sep,
    oversampling_fact,
    angle_list,
    opt_method,
    multiply_gamma,
    psf_cube,
    homogeneous_variance,
    verbose=False,
):
    """
    Process one single annulus, with diff_images and andromeda_core.

    Parameters
    ----------
    i : int
        Number of the annulus
    **kwargs

    Returns
    -------
    res : tuple
        The result of ``andromeda_core``, on the specific annulus.

    """
    global CUBE

    rhomin = annuli_limits[i]
    rhomax = annuli_limits[i + 1]
    rhomax_opt = np.sqrt(roa * rhomax**2 - (roa - 1) * rhomin**2)

    # compute indices from min_sep
    if verbose:
        print("  Pairing frames...")
    min_sep_pix = min_sep * oversampling_fact * 2
    angmin = 2 * np.arcsin(min_sep_pix / (2 * rhomin)) * 180 / np.pi
    index_neg, index_pos, indices_not_used = create_indices(angle_list, angmin)

    if len(indices_not_used) != 0:
        if verbose:
            print(
                "  WARNING: {} frame(s) cannot be used because it wasn't "
                "possible to find any other frame to couple with them. "
                "Their indices are: {}".format(
                    len(indices_not_used), indices_not_used)
            )
        max_sep_pix = (
            2 * rhomin *
            np.sin(np.deg2rad((max(angle_list) - min(angle_list)) / 4))
        )
        max_sep_ld = max_sep_pix / (2 * oversampling_fact)

        if verbose:
            msg = "  For all frames to be used in this annulus, the minimum"
            msg += " separation must be set at most to {} *lambda/D "
            msg += "(corresponding to {} pixels)."
            print(msg.format(max_sep_ld, max_sep_pix))

    if index_neg is None:
        if verbose:
            msg = "  Warning: No couples found for this distance. "
            msg += "Skipping annulus..."
            print(msg)

        return None

    # ===== angular differences
    if verbose:
        print("  Performing angular difference...")

    res = diff_images(
        cube_pos=CUBE[index_pos],
        cube_neg=CUBE[index_neg],
        rint=rhomin,
        rext=rhomax_opt,
        opt_method=opt_method,
    )
    cube_diff, gamma, gamma_prime = res

    if not multiply_gamma:
        # reset gamma & gamma_prime to 1 (they were returned by diff_images)
        gamma = np.ones_like(gamma)
        gamma_prime = np.ones_like(gamma_prime)

    # TODO: gamma_info_output etc not implemented

    # ;Gamma_affine:
    # gamma_info_output[0,0,i] = min(gamma_output_ang[*,0])
    # gamma_info_output[1,0,i] = max(gamma_output_ang[*,0])
    # gamma_info_output[2,0,i] = mean(gamma_output_ang[*,0])
    # gamma_info_output[3,0,i] = median(gamma_output_ang[*,0])
    # gamma_info_output[4,0,i] = variance(gamma_output_ang[*,0])
    # ;Gamma_prime:
    # gamma_info_output[0,1,i] = min(gamma_output_ang[*,1])
    # gamma_info_output[1,1,i] = max(gamma_output_ang[*,1])
    # gamma_info_output[2,1,i] = mean(gamma_output_ang[*,1])
    # gamma_info_output[3,1,i] = median(gamma_output_ang[*,1])
    # gamma_info_output[4,1,i] = variance(gamma_output_ang[*,1])
    #
    #
    # -> they are returned, no further modification from here on.

    # launch andromeda core (:859)
    if verbose:
        print("  Matching...")
    res = andromeda_core(
        diffcube=cube_diff,
        index_neg=index_neg,
        index_pos=index_pos,
        angle_list=angle_list,
        psf_cube=psf_cube,
        homogeneous_variance=homogeneous_variance,
        rhomin=rhomin,
        rhomax=rhomax,
        gamma=gamma,
        verbose=verbose,
    )
    # TODO: ANDROMEDA v3.1r2 calls `ANDROMEDA_CORE` with `/WITHOUT_GAMMA_INPUT`.
    return res  # (flux, snr, likelihood, stdflux)


def andromeda_core(
    diffcube,
    index_neg,
    index_pos,
    angle_list,
    psf_cube,
    rhomin,
    rhomax,
    gamma=None,
    homogeneous_variance=True,
    verbose=False,
):
    """
    Core engine of ANDROMEDA.

    Estimates the flux distribution in the observation field from differential
    images built from different field rotation angles.

    Parameters
    ----------
    diffcube : 3d ndarray
        Differential image cube, set of ``npairs`` differential images. Shape
        ``(npairs, npix, npix)``.
        IDL parameter: ``DIFF_IMAGES_INPUT``
    index_neg : 1d ndarray
    index_pos : 1d ndarray
    angle_list : 1d ndarray
        IDL parameter: ``ANGLES_INPUT``
    psf_cube : 4d ndarray
        IDL parameter: ``PSFCUBE_INPUT``
    rhomin : float
        IDL parameter: ``RHOMIN_INPUT``
    rhomax : float
        is ceiled for the pixel-for-loop.
        IDL parameter: ``RHOMAX_INPUT``
    gamma
        IDL parameter: ``GAMMA_INPUT[*, 0]``
    homogeneous_variance: bool, optional
        IDL parameter: ``HOMOGENEOUS_VARIANCE_INPUT``
    verbose : bool, optional
        print more.

    Returns
    -------
    flux : 2d ndarray
        IDL return value
    snr : 2d ndarray
        IDL output parameter: ``SNR_OUTPUT``
    likelihood : 2d ndarray
        IDL output parameter: ``LIKELIHOOD_OUTPUT``
    stdflux : 2d ndarray
        IDL output parameter: ``STDEVFLUX_OUTPUT``

    Notes
    -----
    - IDL 15/05/2018: add a check if there is only one couple and hence
      weights_diff_2D = 1.

    Differences from IDL implementation
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    Upper case parameters/functions refer to the IDL ANDROMEDA implementation.

    - IDL ANDROMEDA accepts ``WITHOUT_GAMMA_INPUT`` (boolean, for test) and
      ``GAMMA_INPUT`` ("tuple" of ``gamma`` and ``gamma_prime``). The
      ``gamma_prime`` part of ``GAMMA_INPUT`` is never used inside
      ``ANDROMEDA_CORE``. Instead of these parameters, the python implementation
      accepts one single ``gamma`` parameter.
    - IDL's ``kmax`` was renamed to ``npairs``.
    - **not implemented parameters**:
        - The ``POSITIVITY`` parameter is not used any more in ANDROMEDA,
          and maybe removed in the future. It was removed in the python
          implementation.
        - ``GOOD_PIXELS_INPUT``
            - This is a mask, applied to IDL's ``weight_cut`` and
              ``weighted_diff_images``. It is functional in ``ANDROMEDA_CORE``,
              but not exposed through the ``ANDROMEDA`` function.
        - ``MASK_INPUT``
            - similar to ``GOOD_PIXELS_INPUT``, but applied to IDL's
            ``select_pixels`` (which controlls which pixels are processed). It
            is not exposed to ``ANDROMEDA``.
        - ``WEIGHTS_DIFF_INPUT``
            - "(optional input) cube of inverse-of-variance maps. If it is not
              given the variance is treated as constant in time and computed
              empirically for each spatial position."
            - in the python implementation, the variance is **always** treated
              as constant in time.
            - note: ``WEIGHTS_DIFF_INPUT`` is obtained as ``WEIGHTS_OUTPUT``
              from ``DIFF_IMAGES``.
        - ``PATTERN_OUTPUT``
            - this is just an empty ``DBLARR(npix, npix, kmax)``

    """
    npairs, npix, _ = diffcube.shape
    npixpsf = psf_cube.shape[2]  # shape: (p+1, p+1, x, y)
    precision = psf_cube.shape[0] - 1

    # ===== verify + sanitize input
    if npix % 2 == 1:
        raise ValueError("size of the cube is odd!")
    if npixpsf % 2 == 1:
        raise ValueError("PSF has odd pixel size!")

    if gamma is None:
        if verbose:
            msg = "    ANDROMEDA_CORE: The scaling factor is not taken into "
            msg += "account to build the model!"
            print(msg)

    # calculate variance
    if npairs == 1:
        variance_diff_2d = 1
    else:
        variance_diff_2d = (diffcube**2).sum(0) / npairs - (
            diffcube.sum(0) / npairs
        ) ** 2

    # calculate weights from variance
    if homogeneous_variance:
        varmean = np.mean(variance_diff_2d)  # idlwrap.mean
        weights_diff_2d = np.zeros((npix, npix)) + 1 / varmean
        if verbose:
            msg = "    ANDROMEDA_CORE: Variance is considered homogeneous, mean"
            msg += " {:.3f}".format(varmean)
            print(msg)
    else:
        weights_diff_2d = variance_diff_2d > 0
        weights_diff_2d /= variance_diff_2d + (variance_diff_2d == 0)
        if verbose:
            msg = "    ANDROMEDA_CORE: Variance is taken equal to the "
            msg += "empirical variance in each pixel (inhomogeneous, but "
            msg += "constant in time)"
            print(msg)

    wd_images = diffcube * weights_diff_2d

    # create annuli
    d = dist_matrix(npix)
    select_pixels = (d > rhomin) & (d < rhomax)

    if verbose:
        msg = "    ANDROMEDA_CORE: working with {} differential images, radius "
        msg += "{} to {}".format(npairs, rhomin, rhomax)
        print(msg)

    # definition of the expected pattern (if a planet is present)
    numerator = np.zeros((npix, npix))
    denominator = np.ones((npix, npix))

    parang = np.array(
        [angle_list[index_neg], angle_list[index_pos]]) * np.pi / 180
    # shape (2,npairs) -> array([[1, 2, 3],
    #                             [4, 5, 6]])   (for npairs=3)
    # IDL: dimension = SIZE =  _, npairs,2, _, _

    for j in range(
        npix // 2 - np.ceil(rhomax).astype(int), npix // 2 +
        np.ceil(rhomax).astype(int)
    ):
        for i in range(
            npix // 2 - np.ceil(rhomax).astype(int),
            npix // 2 + np.ceil(rhomax).astype(int),
        ):  # same ranges!
            # IDL: scans in different direction!
            if select_pixels[j, i]:
                # distance to center of rotation, in x
                x0 = i - (npix / 2 - 0.5)
                # distance to center of rotation, in y
                y0 = j - (npix / 2 - 0.5)

                decalx = x0 * np.cos(parang) - y0 * np.sin(parang)  # (2,npairs)
                decaly = y0 * np.cos(parang) + x0 * np.sin(parang)  # (2,npairs)

                tbr = decalx - np.floor(decalx).astype(int)
                subp_x = (idl_round(tbr) * precision).astype(int)  # (2,npairs)
                tbr = decaly - np.floor(decaly).astype(int)
                subp_y = (idl_round(tbr) * precision).astype(int)  # (2,npairs)

                # compute, for each k and for both positive and negative indices
                # the coordinates of the squares in which the psf will be placed
                # lef, bot, ... have shape (2,npairs)
                lef = npix // 2 + np.floor(decalx).astype(int) - npixpsf // 2
                bot = npix // 2 + np.floor(decaly).astype(int) - npixpsf // 2
                rig = npix // 2 + \
                    np.floor(decalx).astype(int) + npixpsf // 2 - 1
                top = npix // 2 + \
                    np.floor(decaly).astype(int) + npixpsf // 2 - 1

                # now select the minimum of the two, to compute the area to be
                # cut (the smallest rectangle which contains both psf's)
                px_xmin = np.minimum(lef[0], lef[1])
                px_xmax = np.maximum(rig[0], rig[1])
                px_ymin = np.minimum(bot[0], bot[1])
                px_ymax = np.maximum(top[0], top[1])

                # computation of planet patterns
                num_part = 0
                den_part = 0

                for k in range(npairs):
                    # this is the innermost loop, performed MANY times
                    patt_pos = np.zeros(
                        (px_ymax[k] - px_ymin[k] + 1,
                         px_xmax[k] - px_xmin[k] + 1)
                    )
                    patt_neg = np.zeros(
                        (px_ymax[k] - px_ymin[k] + 1,
                         px_xmax[k] - px_xmin[k] + 1)
                    )

                    # put the positive psf in the right place
                    y0 = bot[1, k] - px_ymin[k]
                    yN = bot[1, k] - px_ymin[k] + npixpsf
                    x0 = lef[1, k] - px_xmin[k]
                    xN = lef[1, k] - px_xmin[k] + npixpsf
                    patt_pos[y0:yN, x0:xN] = psf_cube[subp_y[1, k], subp_x[1, k]]
                    # TODO: should add a +1 somewhere??

                    # same for the negative psf, with a multiplication by gamma!
                    y0 = bot[0, k] - px_ymin[k]
                    yN = bot[0, k] - px_ymin[k] + npixpsf
                    x0 = lef[0, k] - px_xmin[k]
                    xN = lef[0, k] - px_xmin[k] + npixpsf
                    patt_neg[y0:yN, x0:xN] = psf_cube[subp_y[0, k], subp_x[0, k]]
                    # TODO: should add a +1 somewhere??

                    # subtraction between the two
                    if gamma is None:
                        pc = patt_pos - patt_neg
                    else:
                        pc = patt_pos - patt_neg * gamma[k]

                    # compare current (2D) map of small rectangle of weights:
                    if npairs == 1:
                        weight_cut = weights_diff_2d
                    else:
                        weight_cut = weights_diff_2d[
                            px_ymin[k]: px_ymax[k] + 1, px_xmin[k]: px_xmax[k] + 1
                        ]

                    num_part += np.sum(
                        pc
                        * wd_images[
                            k, px_ymin[k]: px_ymax[k] + 1, px_xmin[k]: px_xmax[k] + 1
                        ]
                    )

                    den_part += np.sum(pc**2 * weight_cut)

                numerator[j, i] = num_part
                denominator[j, i] = den_part

    # computation of estimated flux for current assumed planet position:
    flux = numerator / denominator

    # computation of snr map:
    snr = numerator / np.sqrt(denominator)

    # computation of likelihood map:
    likelihood = 0.5 * snr**2

    # computation of the standard deviation on the estimated flux
    stdflux = flux / (snr + (snr == 0))
    # TODO: 0 values are replaced by 1, but
    # small values like 0.1 are kept. Is this
    # the right approach?

    return flux, snr, likelihood, stdflux


def create_indices(angle_list, angmin, verbose=True):
    """
    Compute the couples of indices to satisfy the minimum separation ``angmin``.

    Given a monotonic array of ``angle_list``, this function computes and returns
    the couples of indices of the array for which the separation is the closest
    to the value ``angmin``, by using the highest possible number of angles, all
    if possible.


    Parameters
    ----------
    angle_list : 1d numpy ndarray
        ndarray containing the angles associated to each image. The array should
        be monotonic
    angmin : float
        The minimum acceptable difference between two angles of a couple.
    verbose : bool, optional
        Show warning if no couples can be found.

    Returns
    -------
    indices_neg, indices_pos : ndarrays or None
        The couples of indices, so that ``index_pos[0]`` should be paired with
        ``index_neg[0]`` and so on. Set to None if no couples can be found.
    indices_not_used : list
        The list of the frames which were not used. This list should preferably
        be empty.

    Notes
    -----
    - ``WASTE`` flag removed, instead this function returns ``indices_not_used``

    """
    # make array monotonic -> increasing
    if angle_list[-1] < angle_list[0]:
        angle_list = -angle_list

    good_angles = idl_where(angle_list - angle_list[0] >= angmin)

    if len(good_angles) == 0:
        if verbose:
            print(
                "Impossible to find any couple of angles! Try to "
                "reduce the IWA first, else you need to reduce the "
                "minimum separation."
            )
        return None, None, []

    indices_neg = [0]
    indices_pos = [good_angles[0]]
    indices_not_used = []

    for i in range(1, len(angle_list)):
        good_angles = idl_where((angle_list - angle_list[i] >= angmin))

        if len(good_angles) > 0:
            indices_neg.append(i)
            indices_pos.append(good_angles[0])
        else:  # search in other direction
            if i not in indices_pos:
                good_angles_back = idl_where(
                    (angle_list[i] - angle_list >= angmin))

                if len(good_angles_back) > 0:
                    indices_neg.append(i)
                    indices_pos.append(good_angles_back[-1])
                else:
                    # no new couple found
                    indices_not_used.append(i)

    return np.array(indices_neg), np.array(indices_pos), indices_not_used


def diff_images(
    cube_pos,
    cube_neg,
    rint,
    rext,
    opt_method="lsq",
    variance_pos=None,
    variance_neg=None,
    verbose=False,
):
    """
    Compute the optimized difference between two cubes of images.

    Parameters
    ----------
    cube_pos : 3d ndarray
        stack of square images (nimg x N x N)
    cube_neg : 3d ndarray
        stack of square images (nimg x N x N)
    rint : float
        inner radius of the optimization annulus (in pixels)
    rext : float
        outer radius of the optimization annulus (in pixels)
    opt_method : {'no', 'total', 'lsq', 'l1'}, optional
        Optimization for the immage difference. Numeric values kept for
        compatibility with the IDL version (e.g. calling both functions with the
        same parameters)

        ``"no"`` / ``1``
           corresponds to ``diff_images = i1 - gamma*i2`` and
           ``gamma = gamma_prime = 0``
        ``"total"`` / ``2``
           total ratio optimization. ``diff_images = i1 - gamma*i2`` and
           ``gamma = sum(i1*i2 / sum(i2**2))``, ``gamma_prime = 0``
        ``"lsq"`` / ``3``
           least-squares optimization. ``diff_images = i1 - gamma*i2``,
           ``gamma = sum(i1*i2)/sum(i2**2)``, ``gamma_prime = 0``
        ``"l1"`` / ``4``
           L1-affine optimization, using ``fitaffine`` function.
           ``diff_images = i1 - gamma * i2 - gamma_prime``
    verbose : bool, optional
        Prints some parameters, most notably the values of gamma for each
        difference

    Returns
    -------
    cube_diff
        cube with differences, shape (nimg x N x N)
    gamma, gamma_prime
        arrays containing the optimization coefficient gamma and gamma'. To
        be used to compute the correct planet signatures used by the ANDROMEDA
        algorithm.

    Note
    ----
    - ``GN_NO`` and ``GAIN`` keywords were never used in the IDL version, so
      they were not implemented.
    - VARIANCE_POS_INPUT, VARIANCE_NEG_INPUT, VARIANCE_TOT_OUTPUT,
      WEIGHTS_OUTPUT were removed
    - The numeric ``opt_method`` from the IDL version (``1`` for ``"no"``,
      etc.) are also accepted, but discouraged. Use the strings instead.

    """
    nimg, npix, _ = cube_pos.shape

    # initialize
    cube_diff = np.zeros_like(cube_pos)
    gamma = np.zeros(nimg)  # linear factor, per frame
    gamma_prime = np.zeros(nimg)  # affine factor. Only !=0 for 'l1' affine fit

    distarray = dist_matrix(npix)
    annulus = (distarray > rint) & (distarray <= rext)  # 2d True/False map

    if verbose:
        print("number of elements in annulus:", annulus.sum())

    # compute normalization factors
    if opt_method in ["no", 1]:
        # no renormalization
        msg = "    DIFF_IMAGES: no optimisation is being performed. Note that "
        msg += "keywords rint and rext will be ignored."
        print(msg)
        gamma += 1
    else:
        if verbose:
            msg = "  DIFF_IMAGES: optimization annulus limits: {:.1f} -> {:.1f}"
            print(msg.format(rint, rext))

        for i in range(nimg):
            if opt_method in ["total", 2]:
                sum1 = np.sum(cube_pos[i][annulus])
                sum2 = np.sum(cube_neg[i][annulus])
                gamma[i] = sum1 / sum2
            elif opt_method in ["lsq", 3]:
                sum1 = np.sum(cube_pos[i][annulus] * cube_neg[i][annulus])
                sum2 = np.sum(cube_neg[i][annulus] ** 2)
                gamma[i] = sum1 / sum2

                if verbose:
                    msg = "DIFF_IMAGES: Factor gamma_ls for difference #{}:{}"
                    print(msg.format(i + 1, gamma[i]))
            elif opt_method in ["l1", 4]:  # L1-affine optimization
                ann_pos = cube_pos[i][annulus]
                ann_neg = cube_neg[i][annulus]
                gamma[i], gamma_prime[i] = fitaffine(y=ann_pos, x=ann_neg)
                if verbose:
                    msg = "    DIFF_IMAGES: Factor gamma and gamma_prime for "
                    msg += "difference #{}/{}: {}, {}"
                    print(msg.format(i + 1, nimg, gamma[i], gamma_prime[i]))
            else:
                raise ValueError("opt_method '{}' unknown".format(opt_method))

    if verbose:
        msg = "    DIFF_IMAGES: median gamma={:.3f}, median gamma_prime={:.3f}"
        print(msg.format(np.median(gamma), np.median(gamma_prime)))

    # compute image differences
    for i in range(nimg):
        cube_diff[i] = cube_pos[i] - cube_neg[i] * gamma[i] - gamma_prime[i]

    return cube_diff, gamma, gamma_prime


def normalize_snr(
    snr,
    nsmooth_snr=1,
    iwa=None,
    owa=None,
    oversampling=None,
    fast=None,
    fit=False,
    show=False,
):
    """
    Normalize each pixels of the SNR map by the robust std of its annulus.

    The aim is to get rid of the decreasing trend from the center of the image
    to its edge in order to obtain a SNR map of mean 0 and of variance 1 as
    expected by the algorithm if the noise model (white) was right. Thanks to
    this operation, a constant threshold can be applied on the SNR map to
    perform automatic detection.

    Parameters
    ----------
    snr : 2d ndarray
        Square image/SNR-map to be normalized by its own radial
        robust standard deviation.
    nsmooth_snr : int [pixels], optional
        Number of pixel(s) over which the robust std radial profile is smoothed
        in the outer direction. (e.g. if ``nsmooth_snr=8``, the regarded
        annulus is smoothed w.r.t the 8 following adjacent pixel-annulus (at
        larger separation).
    iwa : float, optional
        Inner working angle in lambda/D. Radius of the smallest annulus
        processed by ANDROMEDA.
    owa : float, optional
        Outer working angle in lambda/D. Radius of the widest annulus processed
        by ANDROMEDA.
    oversampling : float or None, optional
    fast : bool
        (Can also be a non-zero int, as used inside ``andromeda``.)
    fit : bool, optional
        Use a 4D polynomial fit.
    show : bool, optional
        NOT IMPLEMENTED

    Returns
    -------
    snr_norm
        Normalized SNR map of mean 0 and variance 1.
    snr_std
        In order to calculate once for all the
        2D map of the SNR radial robust standard deviation,
        this variable records it.

    Note
    ----
    - in IDL ANDROMEDA, ``/FIT`` is disabled by default, so it was not (yet)
      implemented.

    """
    # ===== initialization
    nsnr = snr.shape[1]
    xcen = ycen = (nsnr - 1) / 2  # floats

    prof_snr = couronne_img(image=snr, xcen=xcen, ycen=ycen, verbose=False)
    # couronne_img, image_input=snr_input, xcen_input=xcen , ycen_input=ycen, $
    #               intenmoy_output=prof_snr, /SILENT

    it_nosmoo = np.zeros(nsnr // 2)  # TODO: check even/odd frames
    it_robust = np.zeros(nsnr // 2)
    imaz_robust = np.zeros_like(snr)

    # ===== defaults
    if owa is None or oversampling is None:
        # If no OWA input then just take the last non-zero value
        dmax = nsnr // 2
    else:
        dmax = np.ceil(owa * 2 * oversampling).astype(int)
        if dmax > nsnr / 2:
            dmax = nsnr // 2

    if iwa is None or oversampling is None:
        # If no IWA input then just take the first non-zero value
        for dm in range(nsnr // 2):  # TODO: floor/ceil?
            dmin = dm
            if snr[int(xcen + dm), int(ycen)] != 0:
                break
    else:
        dmin = np.round(iwa * 2 * oversampling).astype(int)

    # ===== build annulus
    tempo = dist_matrix(nsnr, xcen, ycen)  # 2D ndarray
    # IDL: DIST_CIRCLE, tempo, nsnr, xcen, ycen

    # ===== main calculations
    j = 0
    for i in range(dmin, dmax):
        if prof_snr[i] != 0:
            id = (tempo >= i) & (tempo <= i + nsmooth_snr)
            id2 = (tempo >= i - 0.5) & (tempo <= i + 0.5)
            id3 = (tempo >= i) & (tempo <= i + 1)

            it_nosmoo[i] = robust_std(snr[id3])
            it_robust[i] = robust_std(snr[id])

            if nsmooth_snr == 0:  # IDL: IF nn EQ 1.0
                imaz_robust[id3] = it_nosmoo[i]
            else:
                imaz_robust[id2] = it_robust[i]
        else:
            j = i
            break  # IDL: `GOTO, farzone`

    # IDL `farzone:`
    dfast = 450  # [px] for SPHERE-IRDIS data # TODO: add as function argument?
    dnozero = snr[int(ycen), int(xcen):].nonzero()[0][-1].item()

    if dnozero == dmax:
        id5 = (tempo >= (dnozero - nsmooth_snr - 1)) & (tempo <= nsnr / 2 - 1)
        for i in range(dnozero - nsmooth_snr - 1, nsnr // 2):
            it_robust[i] = robust_std(snr[id5])
            imaz_robust[id5] = it_robust[i]
    else:
        if fast and (dnozero >= dfast):  # IDL: IF KEYWORD_SET(fast)
            # TODO: can `fast` be 0? What would happen then?
            for i in range(dfast - nsmooth_snr - 1, nsnr // 2):
                id3 = (tempo >= i) & (tempo <= i + 1)
                it_robust[i] = it_robust[dnozero - nsmooth_snr - 1]
                imaz_robust[id3] = it_robust[dnozero - nsmooth_snr - 1]
        else:
            # find the first non-zero value:
            k = None
            for i in range(j - nsmooth_snr, dnozero):
                if prof_snr[i] != 0:
                    k = i
            if k is None:  # error handling not present in IDL version.
                import pdb

                pdb.set_trace()
                raise RuntimeError("prof_snr is zero!")

            for i in range(j - nsmooth_snr, k):
                id = (tempo >= i) & (tempo <= dnozero)
                id2 = (tempo >= i - 0.5) & (tempo <= i + 0.5)
                id3 = (tempo >= i) & (tempo <= i + 1)
                id4 = (tempo >= i) & (tempo <= k)
                if id3.sum() > 0:  # condition different from IDL version.
                    it_nosmoo[i] = robust_std(snr[id3])
                if id4.sum() > 0:
                    it_robust[i] = robust_std(snr[id4])

                if nsmooth_snr == 0:  # IDL: IF nn EQ 1.0
                    imaz_robust[id3] = it_nosmoo[i]
                else:
                    imaz_robust[id2] = it_robust[i]

    # using polynomial fit (4th order):
    # offset = 0
    if fit:
        raise NotImplementedError("`fit` parameter is not implemented!")

        # xfit = np.arange(int(j - dmin + offset)) + dmin + offset
        # y_nosmoo = it_nosmoo[int(dmin + offset): j-1]  # TODO: check ranges
        # ...

    # preview if asked:
    if show:
        raise NotImplementedError("`show` parameter is not implemented!")

        # xpix = np.arange(nsnr//2)
        # ...

    # normalize the SNR by its radial std:
    snr_norm = np.zeros((nsnr, nsnr))
    # because imaz_robust has zero value, select a zone:
    zone = imaz_robust != 0
    snr_norm[zone] = snr[zone] / imaz_robust[zone]

    snr_std = imaz_robust
    return snr_norm, snr_std


def couronne_img(image, xcen, ycen=None, lieu=None, step=0.5, rmax=None, verbose=False):
    """
    Provide intensity radial profiles of 2D images.

    Parameters
    ----------
    image : 2d ndarray
        Input image.
    xcen : float
        Center coordinates along the horizontal direction.
    ycen : float, optional
        Center coordinates along the vertical direction.
        Defaults to ``xcen`` if not provided.
    lieu : bool mask, optional
        Locations of the pixels to be removed (``False``) or kepts (``True``).
    step : float, optional
        Width of the regarded annulus.
    rmax : int, optional
        Maximal radius from the image center on which calculus are performed.
        Defaults to half of the ``image`` size (floored).
    verbose : bool, optional
        Show more output.

    Returns
    -------
    intenmoy : 1d ndarray
        Mean intensity per annulus. The only parameter needed for
        ``normalize_snr``.

    Note
    ----
    **Differences from the IDL version**

    - All output variables except ``intenmoy_output`` are not implemented, as
      they are not needed for ``normalize_snr``:
        - inten{site,med,min,max,var,rob,cumulee}_output
        - imaz_{med,var,stddev,robust}_output
    - ``xcen`` was made a required positional argument.

    """
    # ===== verify input
    if image.shape[0] != image.shape[1]:
        raise ValueError("`image` should be square")

    # ===== default values:
    if ycen is None:
        ycen = xcen

    if rmax is None:
        rmax = image.shape[0] // 2

    if lieu is None:
        lieu = np.ones_like(image, dtype=bool)  # `True` bool mask

    if verbose:
        msg = "Computation of azimuthal values from center to " "rmax={}"
        print(msg.format(rmax))

    intenmoy = np.zeros(rmax + 1)
    intenmoy[0] = image[int(ycen), int(xcen)]  # order?

    tempo = dist_matrix(image.shape[0], xcen, ycen)

    for i in range(1, rmax + 1):
        # boolean mask for annulus:
        mask = np.abs(tempo - i) <= step
        mask &= lieu
        if mask.sum() > 0:
            # check if we have matches. If `id` is full of `False`, we get a
            # RuntimeWarning: Mean of empty slice
            local = image[mask]  # 1D array
            intenmoy[i] = np.mean(local)

    return intenmoy
