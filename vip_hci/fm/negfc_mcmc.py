#! /usr/bin/env python
"""
Module with the MCMC (``emcee``) sampling for NEGFC parameter estimation.

.. [CHR21]
   | Christiaens et al. 2021
   | **A faint companion around CrA-9: protoplanet or obscured binary?**
   | *MNRAS, Volume 502, Issue 4, pp. 6117-6139*
   | `https://arxiv.org/abs/2102.10288
     <https://arxiv.org/abs/2102.10288>`_

.. [FOR13]
   | Foreman-Mackey et al. 2013
   | **emcee: The MCMC Hammer**
   | *PASP, Volume 125, Issue 925, p. 306*
   | `https://arxiv.org/abs/1202.3665
     <https://arxiv.org/abs/1202.3665>`_

.. [QUA15]
   | Quanz et al. 2015
   | **Confirmation and Characterization of the Protoplanet HD 100546 b—Direct
   Evidence for Gas Giant Planet Formation at 50 AU**
   | *Astronomy & Astrophysics, Volume 807, p. 64*
   | `https://arxiv.org/abs/1412.5173
     <https://arxiv.org/abs/1412.5173>`_

.. [WER17]
   | Wertz et al. 2017
   | **VLT/SPHERE robust astrometry of the HR8799 planets at milliarcsecond
   level accuracy. Orbital architecture analysis with PyAstrOFit**
   | *Astronomy & Astrophysics, Volume 598, p. 83*
   | `https://arxiv.org/abs/1610.04014
     <https://arxiv.org/abs/1610.04014>`_

.. [GOO10]
   | Goodman & Weare 2010
   | **Ensemble samplers with affine invariance**
   | *Comm. App. Math. Comp. Sci., Vol. 5, Issue 1, pp. 65-80.*
   | `https://ui.adsabs.harvard.edu/abs/2010CAMCS...5...65G
     <https://ui.adsabs.harvard.edu/abs/2010CAMCS...5...65G>`_

"""
from ..fits import write_fits
__author__ = 'O. Wertz, Carlos Alberto Gomez Gonzalez, V. Christiaens'
__all__ = ['mcmc_negfc_sampling',
           'chain_zero_truncated',
           'show_corner_plot',
           'show_walk_plot',
           'confidence']
import numpy as np
import os
import emcee
import multiprocessing
import inspect
import datetime
import corner
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
import pickle
from scipy.stats import norm
from ..fm import cube_inject_companions
from ..config import time_ini, timing
from ..config.utils_conf import sep
from ..psfsub import pca_annulus
from .negfc_fmerit import get_values_optimize, get_mu_and_sigma
from .utils_mcmc import gelman_rubin, autocorr_test
from .utils_negfc import find_nearest


def lnprior(param, bounds, force_rPA=False):
    """Define the prior log-function.

    Parameters
    ----------
    param: tuple
        The model parameters.
    bounds: list
        The bounds for each model parameter.
        Ex: bounds = [(10,20),(0,360),(0,5000)]
    force_rPA: bool, optional
        Whether to only search for optimal flux, provided (r,PA).

    Returns
    -------
    out: float.
        0 if all the model parameters satisfy the prior conditions defined here.
        -np.inf if at least one model parameters is out of bounds.

    """
    if not force_rPA:
        try:
            _ = param[0]
            _ = param[1]
            _ = param[2:]
        except TypeError:
            print('param must be a tuple, {} given'.format(type(param)))

    if not force_rPA:
        try:
            _ = bounds[0]
            _ = bounds[1]
            _ = bounds[2:]
        except TypeError:
            msg = 'bounds must be a list of tuple, {} given'.format(
                type(bounds))
            print(msg)

    cond = True
    for i in range(len(param)):
        if bounds[i][0] <= param[i] <= bounds[i][1]:
            pass
        else:
            cond = False

    if cond:
        return 0.0
    else:
        return -np.inf


def lnlike(param, cube, angs, psf_norm, fwhm, annulus_width, ncomp,
           aperture_radius, initial_state, cube_ref=None, svd_mode='lapack',
           scaling=None, algo=pca_annulus, delta_rot=1, fmerit='sum',
           imlib='vip-fft', interpolation='lanczos4', collapse='median',
           algo_options={}, weights=None, transmission=None, mu_sigma=True,
           sigma='spe+pho', force_rPA=False, debug=False):
    """Define the log-likelihood function.

    Parameters
    ----------
    param: tuple
        The model parameters, typically (r, theta, flux).
    cube: 3d or 4d numpy ndarray
        Input ADI or ADI+IFS cube.
    angs : 1d numpy ndarray
        Vector of derotation angles to align North up in your cube images.
    psf_norm: numpy.array
        The scaled psf expressed as a numpy.array.
    annulus_width: float
        The width of the annulus of interest in pixels.
    ncomp: int or None
        The number of principal components for PCA-based algorithms.
    fwhm : float
        The FHWM in pixels.
    aperture_radius: float
        The radius of the circular aperture in terms of the FWHM.
    initial_state: numpy.array
        The initial guess for the position and the flux of the planet.
    cube_ref : 3d or 4d numpy ndarray, or list of 3d ndarray, optional
        Reference library cube for Reference Star Differential Imaging. Should
        be 3d, except if the input cube is 4d, in which case it can either be a
        4d array or a list of 3d ndarray (reference cube for each wavelength).
    svd_mode : {'lapack', 'randsvd', 'eigen', 'arpack'}, str optional
        Switch for different ways of computing the SVD and selected PCs.
    scaling : {None, "temp-mean", spat-mean", "temp-standard",
        "spat-standard"}, None or str optional
        Pixel-wise scaling mode using ``sklearn.preprocessing.scale``
        function. If set to None, the input matrix is left untouched. Otherwise:

        * ``temp-mean``: temporal px-wise mean is subtracted.

        * ``spat-mean``: spatial mean is subtracted.

        * ``temp-standard``: temporal mean centering plus scaling pixel values
          to unit variance (temporally).

        * ``spat-standard``: spatial mean centering plus scaling pixel values
          to unit variance (spatially).

        DISCLAIMER: Using ``temp-mean`` or ``temp-standard`` scaling can improve
        the speckle subtraction for ASDI or (A)RDI reductions. Nonetheless, this
        involves a sort of c-ADI preprocessing, which (i) can be dangerous for
        datasets with low amount of rotation (strong self-subtraction), and (ii)
        should probably be referred to as ARDI (i.e. not RDI stricto sensu).
    algo: vip function, optional {pca_annulus, pca_annular}
        Post-processing algorithm used.
    delta_rot: float, optional
        If algo is set to pca_annular, delta_rot is the angular threshold used
        to select frames in the PCA library (see description of pca_annular).
    fmerit : {'sum', 'stddev', 'hessian'}, string optional
        If mu_sigma is not provided nor set to True, this parameter determines
        which figure of merit to be used:

            * ``sum``: minimizes the sum of absolute residual intensities in the
            aperture defined with `initial_state` and `aperture_radius`. More
            details in [WER17]_.

            * ``stddev``: minimizes the standard deviation of residual
            intensities in the aperture defined with `initial_state` and
            `aperture_radius`. More details in [WER17]_.

            * ``hessian``: minimizes the sum of absolute values of the
            determinant of the Hessian matrix calculated for each of the 4
            pixels encompassing the first guess location defined with
            `initial_state`. More details in [QUA15]_.

        From experience: ``sum`` is more robust for high SNR companions (but
        rather consider setting mu_sigma=True), while ``stddev`` tend to be more
        reliable in presence of strong residual speckle noise. ``hessian`` is
        expected to be more reliable in presence of extended signals around the
        companion location.
    imlib : str, optional
        See the documentation of the ``vip_hci.preproc.frame_shift`` function.
    interpolation : str, optional
        See the documentation of the ``vip_hci.preproc.frame_shift`` function.
    collapse : {'median', 'mean', 'sum', 'trimmean', 'wmean'}, str, optional
        Sets the way of collapsing the frames for producing a final image. If
        None then the cube of residuals is used when measuring the function of
        merit (instead of a single final frame).
    algo_options: dict, opt
        Dictionary with additional parameters related to the algorithm
        (e.g. tol, min_frames_lib, max_frames_lib). If 'algo' is not a vip
        routine, this dict should contain all necessary arguments apart from
        the cube and derotation angles. Note: arguments such as ncomp, svd_mode,
        scaling, imlib, interpolation or collapse can also be included in this
        dict (the latter are also kept as function arguments for consistency
        with older versions of vip).
    weights : 1d array, optional
        If provided, the negative fake companion fluxes will be scaled according
        to these weights before injection in the cube. Can reflect changes in
        the observing conditions throughout the sequence.
    transmission: numpy array, optional
        Radial transmission of the coronagraph, if any. Array with either
        2 x n_rad, 1+n_ch x n_rad columns. The first column should contain the
        radial separation in pixels, while the other column(s) are the
        corresponding off-axis transmission (between 0 and 1), for either all,
        or each spectral channel (only relevant for a 4D input cube).
    mu_sigma: tuple of 2 floats or None, opt
        If set to None: not used, and falls back to original version of the
        algorithm, using fmerit. Otherwise, should be a tuple of 2 elements,
        containing the mean and standard deviation of pixel intensities in an
        annulus centered on the location of the companion, excluding the area
        directly adjacent to the companion.
    sigma: str, opt
        Sets the type of noise to be included as sigma^2 in the log-probability
        expression. Choice between 'pho' for photon (Poisson) noise, 'spe' for
        residual (mostly whitened) speckle noise, or 'spe+pho' for both.
    force_rPA: bool, optional
        Whether to only search for optimal flux, provided (r,PA).
    debug: boolean
        If True, the cube is returned along with the likelihood log-function.

    Returns
    -------
    out: float
        The log of the likelihood.

    """
    # set imlib for rotation and shift
    if imlib == 'opencv':
        imlib_rot = imlib
        imlib_sh = imlib
    elif imlib == 'skimage' or imlib == 'ndimage-interp':
        imlib_rot = 'skimage'
        imlib_sh = 'ndimage-interp'
    elif imlib == 'vip-fft' or imlib == 'ndimage-fourier':
        imlib_rot = 'vip-fft'
        imlib_sh = 'ndimage-fourier'
    else:
        raise TypeError("Interpolation not recognized.")

    if force_rPA:
        r0 = initial_state[0]
        theta0 = initial_state[1]
        if len(param) > 1:
            flux = -np.array(param)
        else:
            flux = -param[0]
    else:
        r0 = param[0]
        theta0 = param[1]
        if len(param) > 3:
            flux = -np.array(param[2:])
        else:
            flux = -param[2]

    # Apply weights if any
    # if weights is None:
    norm_weights = None  # weights
    if weights is not None:
        #norm_weights = weights/np.sum(weights)
        if np.isscalar(flux):
            flux = flux*weights
        else:
            flux = np.outer(flux, weights)

    # Create the cube with the negative fake companion injected
    cube_negfc = cube_inject_companions(cube, psf_norm, angs, flevel=flux,
                                        rad_dists=[r0], n_branches=1,
                                        theta=theta0, imlib=imlib_sh,
                                        interpolation=interpolation,
                                        transmission=transmission,
                                        verbose=False)
    # Perform PCA and extract the zone of interest
    values = get_values_optimize(cube_negfc, angs, ncomp, annulus_width,
                                 aperture_radius, fwhm, initial_state[0],
                                 initial_state[1], cube_ref=cube_ref,
                                 svd_mode=svd_mode, scaling=scaling,
                                 algo=algo, delta_rot=delta_rot, imlib=imlib_rot,
                                 interpolation=interpolation, collapse=collapse,
                                 algo_options=algo_options,
                                 weights=norm_weights)

    if isinstance(mu_sigma, tuple):
        mu = mu_sigma[0]
        sigma2 = mu_sigma[1]**2
        num = np.power(mu-values, 2)
        denom = 0
        if 'spe' in sigma:
            denom += sigma2
        if 'pho' in sigma:
            denom += np.abs(values-mu)
        lnlikelihood = -0.5 * np.sum(num/denom)
    else:
        mu = mu_sigma
        # old version - delete?
        if fmerit == 'sum':
            lnlikelihood = -0.5 * np.sum(np.abs(values-mu))
        elif fmerit == 'stddev':
            values = values[values != 0]
            lnlikelihood = -np.std(values, ddof=1)*values.size
        else:
            raise RuntimeError('fmerit choice not recognized.')

    if debug:
        return lnlikelihood, cube_negfc
    else:
        return lnlikelihood


def lnprob(param, bounds, cube, angs, psf_norm, fwhm, annulus_width, ncomp,
           aperture_radius, initial_state, cube_ref=None, svd_mode='lapack',
           scaling=None, algo=pca_annulus, delta_rot=1, fmerit='sum',
           imlib='vip-fft', interpolation='lanczos4', collapse='median',
           algo_options={}, weights=None, transmission=None, mu_sigma=True,
           sigma='spe+pho', force_rPA=False, display=False):
    """Define the log-probability function as the sum between the prior and\
    log-likelihood funtions.

    Parameters
    ----------
    param: tuple
        The model parameters.
    bounds: list
        The bounds for each model parameter.
        Ex: bounds = [(10,20),(0,360),(0,5000)]
    cube: 3d or 4d numpy ndarray
        Input ADI or ADI+IFS cube.
    angs : 1d numpy ndarray
        Vector of derotation angles to align North up in your cube images.
    psfn: numpy 2D or 3D array
        Normalised PSF template used for negative fake companion injection.
        The PSF must be centered and the flux in a 1xFWHM aperture must equal 1
        (use ``vip_hci.metrics.normalize_psf``).
        If the input cube is 3D and a 3D array is provided, the first dimension
        must match for both cubes. This can be useful if the star was
        unsaturated and conditions were variable.
        If the input cube is 4D, psfn must be either 3D or 4D. In either cases,
        the first dimension(s) must match those of the input cube.
    fwhm : float
        The FHWM in pixels.
    annulus_width: float
        The width in pixel of the annulus on wich the PCA is performed.
    ncomp: int or None
        The number of principal components for PCA-based algorithms.
    aperture_radius: float
        The radius of the circular aperture in FWHM.
    initial_state: numpy.array
        The initial guess for the position and the flux of the planet.
    cube_ref : 3d or 4d numpy ndarray, or list of 3d ndarray, optional
        Reference library cube for Reference Star Differential Imaging. Should
        be 3d, except if the input cube is 4d, in which case it can either be a
        4d array or a list of 3d ndarray (reference cube for each wavelength).
    svd_mode : {'lapack', 'randsvd', 'eigen', 'arpack'}, str optional
        Switch for different ways of computing the SVD and selected PCs.
    scaling : {None, "temp-mean", spat-mean", "temp-standard",
        "spat-standard"}, None or str optional
        Pixel-wise scaling mode using ``sklearn.preprocessing.scale``
        function. If set to None, the input matrix is left untouched. Otherwise:

        * ``temp-mean``: temporal px-wise mean is subtracted.

        * ``spat-mean``: spatial mean is subtracted.

        * ``temp-standard``: temporal mean centering plus scaling pixel values
          to unit variance (temporally).

        * ``spat-standard``: spatial mean centering plus scaling pixel values
          to unit variance (spatially).

        DISCLAIMER: Using ``temp-mean`` or ``temp-standard`` scaling can improve
        the speckle subtraction for ASDI or (A)RDI reductions. Nonetheless, this
        involves a sort of c-ADI preprocessing, which (i) can be dangerous for
        datasets with low amount of rotation (strong self-subtraction), and (ii)
        should probably be referred to as ARDI (i.e. not RDI stricto sensu).
    fmerit : {'sum', 'stddev', 'hessian'}, string optional
        If mu_sigma is not provided nor set to True, this parameter determines
        which figure of merit to be used:

            * ``sum``: minimizes the sum of absolute residual intensities in the
            aperture defined with `initial_state` and `aperture_radius`. More
            details in [WER17]_.

            * ``stddev``: minimizes the standard deviation of residual
            intensities in the aperture defined with `initial_state` and
            `aperture_radius`. More details in [WER17]_.

            * ``hessian``: minimizes the sum of absolute values of the
            determinant of the Hessian matrix calculated for each of the 4
            pixels encompassing the first guess location defined with
            `initial_state`. More details in [QUA15]_.

        From experience: ``sum`` is more robust for high SNR companions (but
        rather consider setting mu_sigma=True), while ``stddev`` tend to be more
        reliable in presence of strong residual speckle noise. ``hessian`` is
        expected to be more reliable in presence of extended signals around the
        companion location.
    imlib : str, optional
        See the documentation of the ``vip_hci.preproc.frame_rotate`` function.
    interpolation : str, optional
        See the documentation of the ``vip_hci.preproc.frame_rotate`` function.
    algo_options, : dict, opt
        Dictionary with additional parameters related to the algorithm
        (e.g. tol, min_frames_lib, max_frames_lib). If 'algo' is not a vip
        routine, this dict should contain all necessary arguments apart from
        the cube and derotation angles. Note: arguments such as ncomp, svd_mode,
        scaling, imlib, interpolation or collapse can also be included in this
        dict (the latter are also kept as function arguments for consistency
        with older versions of vip).
    collapse : {'median', 'mean', 'sum', 'trimmean', 'wmean'}, str, optional
        Sets the way of collapsing the frames for producing a final image. If
        None then the cube of residuals is used when measuring the function of
        merit (instead of a single final frame).
    weights : 1d array, optional
        If provided, the negative fake companion fluxes will be scaled according
        to these weights before injection in the cube. Can reflect changes in
        the observing conditions throughout the sequence.
    transmission: numpy array, optional
        Radial transmission of the coronagraph, if any. Array with either
        2 x n_rad, 1+n_ch x n_rad columns. The first column should contain the
        radial separation in pixels, while the other column(s) are the
        corresponding off-axis transmission (between 0 and 1), for either all,
        or each spectral channel (only relevant for a 4D input cube).
    mu_sigma: tuple of 2 floats or None, opt
        If set to None: not used, and falls back to original version of the
        algorithm, using fmerit. Otherwise, should be a tuple of 2 elements,
        containing the mean and standard deviation of pixel intensities in an
        annulus centered on the location of the companion, excluding the area
        directly adjacent to the companion.
    sigma: str, opt
        Sets the type of noise to be included as sigma^2 in the log-probability
        expression. Choice between 'pho' for photon (Poisson) noise, 'spe' for
        residual (mostly whitened) speckle noise, or 'spe+pho' for both.
    force_rPA: bool, optional
        Whether to only search for optimal flux, provided (r,PA).
    display: boolean
        If True, the cube is displayed with ds9.

    Returns
    -------
    out: float
        The probability log-function.

    """
    if initial_state is None:
        initial_state = param

    lp = lnprior(param, bounds, force_rPA)

    if np.isinf(lp):
        return -np.inf

    return lp + lnlike(param, cube, angs, psf_norm, fwhm, annulus_width,
                       ncomp, aperture_radius, initial_state, cube_ref,
                       svd_mode, scaling, algo, delta_rot, fmerit, imlib,
                       interpolation, collapse, algo_options, weights,
                       transmission, mu_sigma, sigma, force_rPA)


def mcmc_negfc_sampling(cube, angs, psfn, initial_state, algo=pca_annulus,
                        ncomp=1, annulus_width=8, aperture_radius=1, fwhm=4,
                        mu_sigma=True, sigma='spe+pho', force_rPA=False,
                        fmerit='sum', cube_ref=None, svd_mode='lapack',
                        scaling=None, delta_rot=1, imlib='vip-fft',
                        interpolation='lanczos4', collapse='median',
                        algo_options={}, wedge=None, weights=None,
                        transmission=None, nwalkers=100, bounds=None, a=2.0,
                        burnin=0.3, rhat_threshold=1.01, rhat_count_threshold=1,
                        niteration_min=10, niteration_limit=10000,
                        niteration_supp=0, check_maxgap=20, conv_test='ac',
                        ac_c=50, ac_count_thr=3, nproc=1, output_dir='results/',
                        output_file=None, display=False, verbosity=0,
                        save=False):
    r"""Run an affine invariant mcmc sampling algorithm to determine position
    and flux of a companion using the 'Negative Fake Companion' technique.

    The result of this procedure is a chain with the samples from the posterior
    distributions of each of the 3 parameters. This technique can be summarized
    as follows:
    1) We inject a negative fake companion (one candidate) at a given position
    and characterized by a given flux, both close to the expected values.
    2) We run PCA on an full annulus which pass through the initial guess,
    regardless of the position of the candidate.
    3) We extract the intensity values of all the pixels contained in a
    circular aperture centered on the initial guess.
    4) We calculate a function of merit :math:`\chi^2` (see below).
    The steps 1) to 4) are then looped. At each iteration, the candidate model
    parameters are defined by the ``emcee`` Affine Invariant algorithm [FOR13]_.

    There are different possibilities for the figure of merit (step 4):
        - ``mu_sigma=None``; ``fmerit='sum'`` (as in [WER17]_):\
          :math:`\chi^2 = \sum(\|I_j\|)`
        - ``mu_sigma=None``; ``fmerit='stddev'`` (likely more appropriate than
          'sum' when residual speckle noise is still significant):\
          :math:`\chi^2 = N \sigma_{I_j}(values,ddof=1)*values.size`
        - ``mu_sigma=True`` or a tuple (as in [CHR21]_, new default):\
          :math:`\chi^2 = \sum\frac{I_j- \mu)^2}{\sigma^2}`

    where :math:`j \in {1,...,N}` with N the total number of pixels
    contained in the circular aperture, :math:`\sigma_{I_j}` is the standard
    deviation of :math:`I_j` values, and :math:`\mu` is the mean pixel
    intensity in a truncated annulus at the radius of the companion candidate
    (i.e. excluding the cc region).
    See description of ``mu_sigma`` and ``sigma`` for more details on
    :math:`\sigma` considered in the last equation.

    Speed tricks:
        - crop your input cube to a size such as to just include the annulus on
          which the PCA is performed;
        - set ``imlib='opencv'`` (much faster image rotation, at the expense
          of flux conservation);
        - increase ``nproc`` (if your machine allows);
        - reduce ``ac_c`` (or increase ``rhat_threshold`` if ``conv_test='gb'``)
          for a faster convergence);
        - reduce ``niteration_limit`` to force the sampler to stop even if it
          has not reached convergence.

    Parameters
    ----------
    cube: 3d or 4d numpy ndarray
        Input ADI or ADI+IFS cube.
    angs : 1d numpy ndarray
        Vector of derotation angles to align North up in your cube images.
    psfn: numpy 2D or 3D array
        Normalised PSF template used for negative fake companion injection.
        The PSF must be centered and the flux in a 1xFWHM aperture must equal 1
        (use ``vip_hci.metrics.normalize_psf``).
        If the input cube is 3D and a 3D array is provided, the first dimension
        must match for both cubes. This can be useful if the star was
        unsaturated and conditions were variable.
        If the input cube is 4D, psfn must be either 3D or 4D. In either cases,
        the first dimension(s) must match those of the input cube.
    initial_state: tuple or 1d numpy ndarray
        The first guess for the position and flux(es) of the planet, in that
        order. In the case of a 3D input cube, it should have a length of 3,
        while in the case of a 4D input cube, the length should be 2 + number
        of spectral channels (one flux estimate per wavelength). Each walker
        will start in a small ball around this preferred position.
    algo : python routine, optional
        Post-processing algorithm used to model and subtract the star. First
        2 arguments must be input cube and derotation angles. Must return a
        post-processed 2d frame.
    ncomp: int or None, optional
        The number of principal components for PCA-based algorithms.
    annulus_width: float, optional
        The width in pixels of the annulus on which the PCA is performed.
    aperture_radius: float, optional
        The radius in FWHM of the circular aperture.
    fwhm : float
        The FHWM in pixels.
    mu_sigma: tuple of 2 floats or bool, opt
        If set to None: not used, and falls back to original version of the
        algorithm, using ``fmerit`` [WER17]_.
        If a tuple of 2 elements: should be the mean and standard deviation of
        pixel intensities in an annulus centered on the location of the
        companion candidate, excluding the area directly adjacent to the CC.
        If set to anything else, but None/False/tuple: will compute said mean
        and standard deviation automatically.
        These values will then be used in the log-probability of the MCMC.
    sigma: str, opt
        Sets the type of noise to be included as sigma^2 in the log-probability
        expression. Choice between 'pho' for photon (Poisson) noise, 'spe' for
        residual (mostly whitened) speckle noise, or 'spe+pho' for both.
    force_rPA: bool, optional
        Whether to only search for optimal flux, provided (r,PA).
    fmerit : {'sum', 'stddev', 'hessian'}, string optional
        If mu_sigma is not provided nor set to True, this parameter determines
        which figure of merit to be used:

            * ``sum``: minimizes the sum of absolute residual intensities in the
            aperture defined with `initial_state` and `aperture_radius`. More
            details in [WER17]_.

            * ``stddev``: minimizes the standard deviation of residual
            intensities in the aperture defined with `initial_state` and
            `aperture_radius`. More details in [WER17]_.

            * ``hessian``: minimizes the sum of absolute values of the
            determinant of the Hessian matrix calculated for each of the 4
            pixels encompassing the first guess location defined with
            `initial_state`. More details in [QUA15]_.

        From experience: ``sum`` is more robust for high SNR companions (but
        rather consider setting mu_sigma=True), while ``stddev`` tend to be more
        reliable in presence of strong residual speckle noise. ``hessian`` is
        expected to be more reliable in presence of extended signals around the
        companion location.

    cube_ref : 3d or 4d numpy ndarray, or list of 3d ndarray, optional
        Reference library cube for Reference Star Differential Imaging. Should
        be 3d, except if the input cube is 4d, in which case it can either be a
        4d array or a list of 3d ndarray (reference cube for each wavelength).
    svd_mode : {'lapack', 'randsvd', 'eigen', 'arpack'}, str optional
        Switch for different ways of computing the SVD and selected PCs.
        'randsvd' is not recommended for the negative fake companion technique.
    scaling : {None, "temp-mean", spat-mean", "temp-standard",
        "spat-standard"}, None or str optional
        Pixel-wise scaling mode using ``sklearn.preprocessing.scale``
        function. If set to None, the input matrix is left untouched. Otherwise:

        * ``temp-mean``: temporal px-wise mean is subtracted.

        * ``spat-mean``: spatial mean is subtracted.

        * ``temp-standard``: temporal mean centering plus scaling pixel values
          to unit variance (temporally).

        * ``spat-standard``: spatial mean centering plus scaling pixel values
          to unit variance (spatially).

        DISCLAIMER: Using ``temp-mean`` or ``temp-standard`` scaling can improve
        the speckle subtraction for ASDI or (A)RDI reductions. Nonetheless, this
        involves a sort of c-ADI preprocessing, which (i) can be dangerous for
        datasets with low amount of rotation (strong self-subtraction), and (ii)
        should probably be referred to as ARDI (i.e. not RDI stricto sensu).
    delta_rot: float, optional
        If algo is set to pca_annular, delta_rot is the angular threshold used
        to select frames in the PCA library (see description of pca_annular).
        Increases processing time.
    imlib : str, optional
        Imlib used for both image rotation and sub-px shift:
            - "opencv": will use it for both;
            - "skimage" or "ndimage-interp" will use scikit-image and
              scipy.ndimage for rotation and shift resp.;
            - "ndimage-fourier" or "vip-fft" will use Fourier transform based
              methods for both.
    interpolation : str, optional
        Interpolation order. See the documentation of the
        ``vip_hci.preproc.frame_rotate`` function. Note that the interpolation
        options are identical for rotation and shift within each of the 3 imlib
        cases above.
    collapse : {'median', 'mean', 'sum', 'trimmean', 'wmean'}, str, optional
        Sets the way of collapsing the frames for producing a final image. If
        None then the cube of residuals is used when measuring the function of
        merit (instead of a single final frame).
    algo_options: dict, opt
        Dictionary with additional parameters related to the algorithm
        (e.g. tol, min_frames_lib, max_frames_lib). If 'algo' is not a vip
        routine, this dict should contain all necessary arguments apart from
        the cube and derotation angles. Note: arguments such as ncomp, svd_mode,
        scaling, imlib, interpolation or collapse can also be included in this
        dict (the latter are also kept as function arguments for consistency
        with older versions of vip).
    wedge: tuple, opt
        Range in theta where the mean and standard deviation are computed in an
        annulus defined in the PCA image. If None, it will be calculated
        automatically based on initial guess and derotation angles to avoid
        negative ADI side lobes. If some disc signal is present elsewhere in
        the annulus, it is recommended to provide wedge manually. The provided
        range should be continuous and >0. E.g. provide (270, 370) to consider
        a PA range between [-90,+10].
    weights : 1d array, optional
        If provided, the negative fake companion fluxes will be scaled according
        to these weights before injection in the cube. Can reflect changes in
        the observing conditions throughout the sequence.
    transmission: numpy array, optional
        Radial transmission of the coronagraph, if any. Array with either
        2 x n_rad, 1+n_ch x n_rad columns. The first column should contain the
        radial separation in pixels, while the other column(s) are the
        corresponding off-axis transmission (between 0 and 1), for either all,
        or each spectral channel (only relevant for a 4D input cube).
    nwalkers: int optional
        The number of [GOO10]_ 'walkers'.
    bounds: numpy.array or list, default=None, optional
        The prior knowledge on the model parameters. If None, large bounds will
        be automatically estimated from the initial state.
    a: float, default=2.0
        The proposal scale parameter. See Note.
    burnin: float, default=0.3
        The fraction of a walker chain which is discarded. Note: only used for
        Gelman-Rubin convergence test - the chains are returned full.
    rhat_threshold: float, default=0.01
        The Gelman-Rubin threshold used for the test for nonconvergence.
    rhat_count_threshold: int, optional
        The Gelman-Rubin test must be satisfied 'rhat_count_threshold' times in
        a row before claiming that the chain has converged.
    conv_test: str, optional {'gb','ac'}
        Method to check for convergence:
            - 'gb' for gelman-rubin test
                (https://digitalassets.lib.berkeley.edu/sdtr/ucb/text/305.pdf)
            - 'ac' for autocorrelation analysis
                (https://emcee.readthedocs.io/en/stable/tutorials/autocorr/)
    ac_c: float, optional
        If the convergence test is made using the auto-correlation, this is the
        value of C such that tau/N < 1/C is the condition required for tau to be
        considered a reliable auto-correlation time estimate (for N number of
        samples). Recommended: C>50.
        More details here:
        https://emcee.readthedocs.io/en/stable/tutorials/autocorr/
    ac_count_thr: int, optional
        The auto-correlation test must be satisfied ac_count_thr times in a row
        before claiming that the chain has converged.
    niteration_min: int, optional
        Steps per walker lower bound. The simulation will run at least this
        number of steps per walker.
    niteration_limit: int, optional
        Steps per walker upper bound. If the simulation runs up to
        'niteration_limit' steps without having reached the convergence
        criterion, the run is stopped.
    niteration_supp: int, optional
        Number of iterations to run after having "reached the convergence".
    check_maxgap: int, optional
        Maximum number of steps per walker between two Gelman-Rubin test.
    nproc: int or None, optional
        The number of processes to use for parallelization. If None, will be set
        automatically to the number of CPUs available.
    output_dir: str, optional
        The name of the output directory which contains the output files in the
        case  ``save`` is True.
    output_file: str, optional
        The name of the output file which contains the MCMC results in the case
        ``save`` is True.
    display: bool, optional
        If True, the walk plot is displayed at each evaluation of the Gelman-
        Rubin test.
    verbosity: 0, 1, 2 or 3, optional
        Verbosity level. 0 for no output and 3 for full information.
        (only difference between 2 and 3 is that 3 also writes intermediate
        pickles containing the state of the chain at convergence tests; these
        can end up taking a lot of space).
    save: bool, optional
        If True, the MCMC results are pickled.

    Returns
    -------
    out : numpy.array
        The MCMC chain.

    Note
    ----
    The parameter ``a`` must be > 1. For more theoretical information
    concerning this parameter, see [GOO10]_.

    The parameter ``rhat_threshold`` can be a numpy.array with individual
    threshold value for each model parameter.

    """
    if verbosity > 0:
        start_time = time_ini()
        print("        MCMC sampler for the NEGFC technique       ")
        print(sep)

    # If required, one create the output folder.
    if save:
        output_file_tmp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        if output_dir[-1] == '/':
            output_dir = output_dir[:-1]
        try:
            os.makedirs(output_dir)
        except OSError as exc:
            if exc.errno == 17 and os.path.isdir(output_dir):
                # errno.EEXIST == 17 -> File exists
                pass
            else:
                raise

    if cube.ndim != 4 and cube.ndim != 3:
        raise ValueError('`cube` must be a 3D or 4D numpy array')

    if cube_ref is not None:
        if cube.ndim == 3:
            if cube_ref.ndim != 3:
                raise ValueError('`cube_ref` must be a 3D numpy array')
        elif cube.ndim == 4:
            if cube_ref[0].ndim != 3:
                msg = '`cube_ref` must be a 4D array or a list of 3D arrays'
                raise ValueError(msg)
    if weights is not None:
        if not len(weights) == cube.shape[-3]:
            msg = "Weights should have same length as temporal cube axis"
            raise TypeError(msg)
        norm_weights = None  # weights/np.sum(weights)
    else:
        norm_weights = weights

    if psfn.ndim > 2:
        if psfn.shape[0] != cube.shape[0]:
            msg = "If PSF is 3D or 4D, first dimension should match that of "
            msg += "input cube"
            raise TypeError(msg)

    if 'spe' not in sigma and 'pho' not in sigma:
        raise ValueError("sigma not recognized")

    # set imlib for rotation and shift
    if imlib == 'opencv':
        imlib_rot = imlib
    elif imlib == 'skimage' or imlib == 'ndimage-interp':
        imlib_rot = 'skimage'
    elif imlib == 'vip-fft' or imlib == 'ndimage-fourier':
        imlib_rot = 'vip-fft'
    else:
        raise TypeError("Interpolation not recognized.")

    if nproc is None:  # if the user has not provided nproc, determine the number of processes for Pool to use
        nproc = multiprocessing.cpu_count()

    # #########################################################################
    # Initialization of the variables
    # #########################################################################
    dim = len(initial_state)
    if force_rPA:
        dim -= 2
    itermin = niteration_min
    limit = niteration_limit
    supp = niteration_supp
    maxgap = check_maxgap
    initial_state = np.array(initial_state)
    if initial_state[1] == 0:
        initial_state[1] = 360  # for appropriate scaling of initial ball

    mu_sig = get_mu_and_sigma(cube, angs, ncomp, annulus_width, aperture_radius,
                              fwhm, initial_state[0], initial_state[1],
                              initial_state[2:], psfn, cube_ref=cube_ref,
                              wedge=wedge, svd_mode=svd_mode, scaling=scaling,
                              algo=algo, delta_rot=delta_rot, imlib=imlib_rot,
                              interpolation=interpolation, collapse=collapse,
                              weights=norm_weights, algo_options=algo_options)

    # Measure mu and sigma once in the annulus (instead of each MCMC step)
    if isinstance(mu_sigma, tuple):
        if len(mu_sigma) != 2:
            raise TypeError("if a tuple, mu_sigma should have 2 elements")

    elif mu_sigma:
        mu_sigma = mu_sig
        if verbosity > 0:
            msg = "The mean and stddev in the annulus at the radius of the "
            msg += "companion (excluding the PA area directly adjacent to it)"
            msg += " are {:.2f} and {:.2f} respectively."
            print(msg.format(mu_sigma[0], mu_sigma[1]))
    else:
        mu_sigma = mu_sig[0]  # just take mean

    if itermin > limit:
        itermin = 0

    fraction = 0.3
    geom = 0
    lastcheck = 0
    konvergence = np.inf
    rhat_count = 0
    ac_count = 0
    chain = np.empty([nwalkers, 1, dim])
    nIterations = limit + supp
    rhat = np.zeros(dim)
    stop = np.inf

    if bounds is None:
        bounds = []
        d0 = 0
        if not force_rPA:
            # angle subtended by aperture_radius/2 or fwhm at r=initial_state[0]
            dr = min(annulus_width/2, aperture_radius*fwhm/2)
            dth = 360./(2*np.pi*initial_state[0]/(aperture_radius*fwhm/2))
            bounds = [(initial_state[0] - dr, initial_state[0] + dr),  # radius
                      (initial_state[1] - dth, initial_state[1] + dth)]  # angle
            d0 = 2
        for i in range(dim-d0):
            bounds.append((0, 5 * initial_state[d0+i]))  # flux
    # size of ball of parameters for MCMC initialization
    scal = abs(bounds[0][0]-initial_state[0])/initial_state[0]
    for i in range(dim):
        for j in range(2):
            test_scal = abs(bounds[i][j]-initial_state[i])/initial_state[i]
            if test_scal < scal:
                scal = test_scal
    if force_rPA:
        init = initial_state[2:]
        if len(init) > 1:
            labels = []
            for i in range(len(init)):
                labels.append(r"$f{:.0f}$".format(i))
        else:
            labels = [r"$f$"]
    else:
        init = initial_state
        if len(init) > 3:
            labels = ["$r$", r"$\theta$"]
            for i in range(len(init)-2):
                labels.append(r"$f{:.0f}$".format(i))
        else:
            labels = ["$r$", r"$\theta$", "$f$"]
    pos = init*(1+np.random.normal(0, scal/50., (nwalkers, dim)))
    # divided by 7 to not have any walker initialized out of bounds

    if verbosity > 0:
        print('Beginning emcee Ensemble sampler...')

    # deactivate multithreading
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"

    avail_methods = multiprocessing.get_all_start_methods()
    if "forkserver" in avail_methods:
        multiprocessing.set_start_method("forkserver", force=True)  # faster, better
    else:
        multiprocessing.set_start_method("spawn", force=True)  # slower, but available on all platforms

    with multiprocessing.Pool(processes=nproc) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, dim, lnprob,
                                        pool=pool, moves=emcee.moves.StretchMove(a=a),
                                        args=([bounds, cube, angs, psfn,
                                              fwhm, annulus_width, ncomp,
                                              aperture_radius, initial_state,
                                              cube_ref, svd_mode, scaling, algo,
                                              delta_rot, fmerit, imlib,
                                              interpolation, collapse,
                                              algo_options, weights, transmission,
                                              mu_sigma, sigma, force_rPA]))

        if verbosity > 0:
            print('emcee Ensemble sampler successful', flush=True)
        start = datetime.datetime.now()

        # #########################################################################
        # Affine Invariant MCMC run
        # #########################################################################
        if verbosity > 1:
            print('\nStart of the MCMC run ...')
            print('Step  |  Duration/step (sec)  |  Remaining Estimated Time (sec)')

        for k, res in enumerate(sampler.sample(pos, iterations=nIterations)):
            elapsed = (datetime.datetime.now()-start).total_seconds()
            if verbosity > 1:
                if k == 0:
                    q = 0.5
                else:
                    q = 1
                print('{}\t\t{:.5f}\t\t\t{:.5f}'.format(k, elapsed * q,
                                                        elapsed * (limit-k-1) * q),
                      flush=True)

            start = datetime.datetime.now()

            # ---------------------------------------------------------------------
            # Store the state manually in order to handle with dynamical sized chain
            # ---------------------------------------------------------------------
            # Check if the size of the chain is long enough.
            s = chain.shape[1]
            if k+1 > s:     # if not, one doubles the chain length
                empty = np.zeros([nwalkers, 2*s, dim])
                chain = np.concatenate((chain, empty), axis=1)
            # Store the state of the chain
            chain[:, k] = res[0]

            # ---------------------------------------------------------------------
            # If k meets the criterion, one tests the non-convergence.
            # ---------------------------------------------------------------------
            criterion = int(np.amin([np.ceil(itermin*(1+fraction)**geom),
                                     lastcheck+np.floor(maxgap)]))
            if k == criterion:
                if verbosity > 1:
                    print('\n {} convergence test in progress...'.format(conv_test))

                geom += 1
                lastcheck = k
                if display:
                    show_walk_plot(chain, labels=labels)

                if save and verbosity == 3:
                    fname = '{d}/{f}_temp_k{k}'.format(d=output_dir,
                                                       f=output_file_tmp, k=k)
                    data = {'chain': sampler.chain,
                            'lnprob': sampler.get_log_prob(),
                            'AR': sampler.acceptance_fraction}
                    with open(fname, 'wb') as fileSave:
                        pickle.dump(data, fileSave)

                # We only test the rhat if we have reached the min # of steps
                if (k+1) >= itermin and konvergence == np.inf:
                    if conv_test == 'gb':
                        thr0 = int(np.floor(burnin*k))
                        thr1 = int(np.floor((1-burnin)*k*0.25))

                        # We calculate the rhat for each model parameter.
                        for j in range(dim):
                            part1 = chain[:, thr0:thr0 + thr1, j].reshape(-1)
                            part2 = chain[:, thr0 + 3 * thr1:thr0 + 4 * thr1, j
                                          ].reshape(-1)
                            series = np.vstack((part1, part2))
                            rhat[j] = gelman_rubin(series)
                        if verbosity > 0:
                            print('   r_hat = {}'.format(rhat))
                            cond = rhat <= rhat_threshold
                            print('   r_hat <= threshold = {} \n'.format(cond), flush=True)
                        # We test the rhat.
                        if (rhat <= rhat_threshold).all():
                            rhat_count += 1
                            if rhat_count < rhat_count_threshold:
                                if verbosity > 0:
                                    msg = "Gelman-Rubin test OK {}/{}"
                                    print(msg.format(rhat_count,
                                                     rhat_count_threshold))
                            elif rhat_count >= rhat_count_threshold:
                                if verbosity > 0:
                                    print('... ==> convergence reached')
                                konvergence = k
                                stop = konvergence + supp
                        else:
                            rhat_count = 0
                    elif conv_test == 'ac':
                        # We calculate the auto-corr test for each model parameter.
                        if save:
                            chain_name = "TMP_test_chain{:.0f}.fits".format(k)
                            write_fits(output_dir+'/'+chain_name, chain[:, :k])
                        for j in range(dim):
                            rhat[j] = autocorr_test(chain[:, :k, j])
                        thr = 1./ac_c
                        if verbosity > 0:
                            print('Auto-corr tau/N = {}'.format(rhat))
                            print('tau/N <= {} = {} \n'.format(thr, rhat < thr), flush=True)
                        if (rhat <= thr).all():
                            ac_count += 1
                            if verbosity > 0:
                                msg = "Auto-correlation test passed for all params!"
                                msg += "{}/{}".format(ac_count, ac_count_thr)
                                print(msg)
                            if ac_count >= ac_count_thr:
                                msg = '\n ... ==> convergence reached'
                                print(msg)
                                stop = k
                        else:
                            ac_count = 0
                    else:
                        raise ValueError('conv_test value not recognized')
                    # append the autocorrelation factor to file for easy reading
                    if save:
                        with open(output_dir + '/MCMC_results_tau.txt', 'a') as f:
                            f.write(str(rhat) + '\n')
            # We have reached the maximum number of steps for our Markov chain.
            if k+1 >= stop:
                if verbosity > 0:
                    print('We break the loop because we have reached convergence')
                break

    if k == nIterations-1:
        if verbosity > 0:
            print("We have reached the limit # of steps without convergence")

    if save:
        frame = inspect.currentframe()
        args, _, _, values = inspect.getargvalues(frame)
        input_parameters = {j: values[j] for j in args[1:]}

        output = {'chain': chain_zero_truncated(chain),
                  'input_parameters': input_parameters,
                  'AR': sampler.acceptance_fraction,
                  'lnprobability': sampler.get_log_prob()}

        if output_file is None:
            output_file = 'MCMC_results'
        with open(output_dir+'/'+output_file, 'wb') as fileSave:
            pickle.dump(output, fileSave)

        msg = "\nThe file MCMC_results has been stored in the folder {}"
        print(msg.format(output_dir+'/'))

    if verbosity > 0:
        timing(start_time)

    # reactivate multithreading
    ncpus = multiprocessing.cpu_count()
    os.environ["MKL_NUM_THREADS"] = str(ncpus)
    os.environ["NUMEXPR_NUM_THREADS"] = str(ncpus)
    os.environ["OMP_NUM_THREADS"] = str(ncpus)

    return chain_zero_truncated(chain)


def chain_zero_truncated(chain):
    """
    Return the Markov chain with the dimension: walkers x steps* x parameters,\
    where steps* is the last step before having 0 (not yet constructed chain).

    Parameters
    ----------
    chain: numpy.array
        The MCMC chain.

    Returns
    -------
    out: numpy.array
        The truncated MCMC chain, that is to say, the chain which only contains
        relevant information.
    """
    try:
        idxzero = np.where(chain[0, :, 0] == 0.0)[0][0]
    except BaseException:
        idxzero = chain.shape[1]
    return chain[:, 0:idxzero, :]


def show_walk_plot(chain, save=False, output_dir='', **kwargs):
    """
    Display or save figure showing the path of each walker during the MCMC run.

    Parameters
    ----------
    chain: numpy.array
        The Markov chain. The shape of chain must be nwalkers x length x dim.
        If a part of the chain is filled with zero values, the method will
        discard these steps.
    save: boolean, default: False
        If True, a pdf file is created.
    output_dir: str, optional
        The name of the output directory which contains the output files in the
        case ``save`` is True.
    kwargs:
        Additional attributes are passed to the matplotlib plot method.

    Returns
    -------
    Display the figure or create a pdf file named walk_plot.pdf in the working
    directory.

    """
    temp = np.where(chain[0, :, 0] == 0.0)[0]
    if len(temp) != 0:
        chain = chain[:, :temp[0], :]

    labels = kwargs.pop('labels', ["$r$", r"$\theta$", "$f$"])
    npar = len(labels)
    y = 2*npar
    if npar > 1:
        fig, axes = plt.subplots(npar, 1, sharex=True,
                                 figsize=kwargs.pop('figsize', (8, y)))
        axes[-1].set_xlabel(kwargs.pop('xlabel', 'step number'))
        axes[-1].set_xlim(kwargs.pop('xlim', [0, chain.shape[1]]))
        color = kwargs.pop('color', 'k')
        alpha = kwargs.pop('alpha', 0.4)
        for j in range(npar):
            axes[j].plot(chain[:, :, j].T, color=color, alpha=alpha, **kwargs)
            axes[j].yaxis.set_major_locator(MaxNLocator(5))
            axes[j].set_ylabel(labels[j])
    else:
        fig = plt.figure(figsize=kwargs.pop('figsize', (8, y)))
        plt.xlabel(kwargs.pop('xlabel', 'step number'))
        plt.xlim(kwargs.pop('xlim', [0, chain.shape[1]]))
        color = kwargs.pop('color', 'k')
        alpha = kwargs.pop('alpha', 0.4)
        plt.plot(chain[:, :, 0].T, color=color, alpha=alpha, **kwargs)
        plt.locator_params(axis='y', tight=True, nbins=5)
        plt.ylabel(labels[0])
    fig.tight_layout(h_pad=0)
    if save:
        plt.savefig(output_dir+'walk_plot.pdf')
        plt.close(fig)


def show_corner_plot(chain, burnin=0.5, save=False, output_dir='', **kwargs):
    """
    Display or save a figure showing the corner plot (pdfs + correlation plots).

    Parameters
    ----------
    chain: numpy.array
        The Markov chain. The shape of chain must be nwalkers x length x dim.
        If a part of the chain is filled with zero values, the method will
        discard these steps.
    burnin: float, default: 0
        The fraction of a walker chain we want to discard.
    save: boolean, default: False
        If True, a pdf file is created.
    output_dir: str, optional
        The name of the output directory which contains the output files in the
        case ``save`` is True.
    kwargs:
        Additional attributes passed to the corner.corner() method.

    Returns
    -------
    None
        (Displays the figure or create a pdf file named walk_plot.pdf in the
        working directory).

    Raises
    ------
    ImportError

    """
    if chain.shape[0] == 0:
        msg = "It seems the chain is empty. Have you already run the MCMC?"
        raise ValueError(msg)

    labels = kwargs.pop('labels', ["$r$", r"$\theta$", "$f$"])

    try:
        npar = len(labels)
        temp = np.where(chain[0, :, 0] == 0.0)[0]
        if len(temp) != 0:
            chain = chain[:, :temp[0], :]
        length = chain.shape[1]
        indburn = int(np.floor(burnin*(length-1)))
        chain = chain[:, indburn:length, :].reshape((-1, npar))
    except IndexError:
        pass

    fig = corner.corner(chain, labels=labels, **kwargs)

    if save:
        plt.savefig(output_dir+'corner_plot.pdf')
        plt.close(fig)


def confidence(isamples, cfd=68.27, bins=100, gaussian_fit=False, weights=None,
               verbose=True, save=False, output_dir='', force=False,
               output_file='confidence.txt', title=None, ndig=1, plsc=None,
               labels=['r', 'theta', 'f'], gt=None, **kwargs):
    r"""
    Determine the highly probable value for each model parameter, as well as
    the 1-sigma confidence interval.

    Parameters
    ----------
    isamples: numpy.array
        The independent samples for each model parameter.
    cfd: float, optional
        The confidence level given in percentage.
    bins: int, optional
        The number of bins used to sample the posterior distributions.
    gaussian_fit: boolean, optional
        If True, a gaussian fit is performed in order to determine
        (:math:`\mu, \sigma`).
    weights : (n, ) numpy ndarray or None, optional
        An array of weights for each sample.
    verbose: boolean, optional
        Display information in the shell.
    save: boolean, optional
        If "True", a txt file with the results is saved in the output
        repository.
    output_dir: str, optional
        If save is True, this is the full path to a directory where the results
        are saved.
    force: bool, optional
        If set to True, force the confidence interval estimate even if too
        many samples fall in a single bin (unreliable CI estimates). If False,
        an error message is raised if the percentile of samples falling in a
        single bin is larger than cfd, suggesting to increase number of bins.
    output_file: str, optional
        If save is True, name of the text file in which the results are saved.
    title: bool or str, optional
        If not None, will print labels and parameter values on top of each
        plot. If a string, will print that label in front of the parameter
        values.
    ndig: int or list of int, optional
        If title is not None, this sets the number of significant digits to be
        used when printing the best estimate for each parameter. If int, the
        same number of digits is used for all parameters. If a list of int, the
        length should match the number of parameters.
    plsc: float, optional
        If save is True, this is used to convert pixels to arcsec when writing
        results for r.
    labels: list of strings, optional
        Labels to be used for both plots and dictionaries. Length should match
        the number of free parameters.
    gt: list or 1d numpy array, optional
        Ground truth values (when known). If provided, will also be plotted for
        reference.

    Returns
    -------
    out: A 2 elements tuple with either:
        [gaussian_fit=False] i) the highly probable solutions (dictionary),
            ii) the respective confidence interval (dict.);
        [gaussian_fit=True] i) the center of the best-fit 1d Gaussian
            distributions (tuple of 3 floats), and ii) their standard deviation,
            for each parameter

    """
    try:
        l = isamples.shape[1]
        if l == 1:
            isamples = isamples[:, 0]
    except BaseException:
        l = 1

    if not l == len(labels):
        raise ValueError("Length of labels different to number of parameters")

    if gt is not None:
        if len(gt) != l:
            msg = "If provided, the length of ground truth values should match"
            msg += " number of parameters"
            raise TypeError(msg)
    if np.isscalar(ndig):
        ndig = [ndig]*l
    else:
        if len(ndig) != l:
            msg = "Length of ndig list different to number of parameters"
            raise ValueError(msg)

    pKey = labels
    label_file = labels

    confidenceInterval = {}
    val_max = {}

    if cfd == 100:
        cfd = 99.9

    #######################################
    #  Determine the confidence interval  #
    #######################################
    if gaussian_fit:
        mu = np.zeros(l)
        sigma = np.zeros_like(mu)

    if gaussian_fit:
        nrows = 2*max(int(np.ceil(l/4)), 1)
        fig, ax = plt.subplots(nrows, min(4, l), figsize=(12, 4*nrows))
    else:
        nrows = max(int(np.ceil(l/4)), 1)
        fig, ax = plt.subplots(nrows, min(4, l), figsize=(12, 4*nrows))

    for j in range(l):
        if nrows > 1:
            if l > 1:
                ax0_tmp = ax[j//4][j % 4]
                if gaussian_fit:
                    ax1_tmp = ax[nrows//2+j//4][j % 4]
            else:
                ax0_tmp = ax[j//4]
                if gaussian_fit:
                    ax1_tmp = ax[nrows//2+j//4]
        elif l > 1 and not gaussian_fit:
            ax0_tmp = ax[j]
        else:
            ax0_tmp = ax
        if l > 1:
            if gaussian_fit:
                n, bin_vertices, _ = ax0_tmp.hist(isamples[:, j], bins=bins,
                                                  weights=weights,
                                                  histtype='step',
                                                  edgecolor='gray')
            else:
                n, bin_vertices, _ = ax0_tmp.hist(isamples[:, j], bins=bins,
                                                  weights=weights,
                                                  histtype='step',
                                                  edgecolor='gray')
        else:
            if gaussian_fit:
                n, bin_vertices, _ = ax0_tmp.hist(isamples[:], bins=bins,
                                                  weights=weights,
                                                  histtype='step',
                                                  edgecolor='gray')
            else:
                n, bin_vertices, _ = ax0_tmp.hist(isamples[:], bins=bins,
                                                  weights=weights,
                                                  histtype='step',
                                                  edgecolor='gray')
        bins_width = np.mean(np.diff(bin_vertices))
        surface_total = np.sum(np.ones_like(n)*bins_width * n)
        n_arg_sort = np.argsort(n)[::-1]

        test = 0
        pourcentage = 0
        for k, jj in enumerate(n_arg_sort):
            test = test + bins_width*n[int(jj)]
            pourcentage = test/surface_total*100
            if pourcentage > cfd:
                if verbose:
                    msg = 'percentage for {}: {}%'
                    print(msg.format(label_file[j], pourcentage))
                break
        if k == 0:
            msg = "WARNING: Percentile reached in a single bin. "
            msg += "This may be due to outliers or a small sample."
            msg += "Uncertainties will be unreliable. Try one of these:"
            msg += "increase bins, or trim outliers, or decrease cfd."
            if force:
                raise ValueError(msg)
            else:
                print(msg)
        n_arg_min = int(n_arg_sort[:k+1].min())
        n_arg_max = int(n_arg_sort[:k+1].max())

        if n_arg_min == 0:
            n_arg_min += 1
        if n_arg_max == bins:
            n_arg_max -= 1

        val_max[pKey[j]] = bin_vertices[int(n_arg_sort[0])]+bins_width/2.
        confidenceInterval[pKey[j]] = np.array([bin_vertices[n_arg_min-1],
                                               bin_vertices[n_arg_max+1]]
                                               - val_max[pKey[j]])
        if title is not None:
            if isinstance(title, str):
                lab = title
            else:
                lab = pKey[j]
        if l > 1:
            arg = (isamples[:, j] >= bin_vertices[n_arg_min - 1]) * \
                  (isamples[:, j] <= bin_vertices[n_arg_max + 1])
            if gaussian_fit:
                ax0_tmp.hist(isamples[arg, j], bins=bin_vertices,
                             facecolor='gray', edgecolor='darkgray',
                             histtype='stepfilled', alpha=0.5)
                ax0_tmp.set_xlabel(labels[j])
                if j == 0:
                    ax0_tmp.set_ylabel('Counts')
                if title is not None:
                    fmt = "{{:.{0}f}}".format(ndig[j]).format
                    msg = r"${{{0}}}_{{{1}}}^{{+{2}}}$"
                    tit = msg.format(fmt(val_max[pKey[j]]),
                                     fmt(confidenceInterval[pKey[j]][0]),
                                     fmt(confidenceInterval[pKey[j]][1]))
                    if gt is not None:
                        tit += r" (gt: ${{{0}}}$)".format(fmt(gt[j]))
                    ax0_tmp.set_title("{0}: {1}".format(lab, tit), fontsize=10)
                if gt is not None:
                    x_close = find_nearest(bin_vertices, gt[j])
                    ax0_tmp.vlines(gt[j], 0, n[x_close], label='gt',
                                   linestyles='dashed', color='blue')
                    label = 'estimate'
                else:
                    label = None
                ax0_tmp.vlines(val_max[pKey[j]], 0, n[int(n_arg_sort[0])],
                               linestyles='dashed', color='red', label=label)

                mu[j], sigma[j] = norm.fit(isamples[:, j])
                n_fit, bins_fit = np.histogram(isamples[:, j], bins, density=1,
                                               weights=weights)
                ax1_tmp.hist(isamples[:, j], bins, density=1, weights=weights,
                             facecolor='gray', edgecolor='darkgray',
                             histtype='step')
                y = norm.pdf(bins_fit, mu[j], sigma[j])
                ax1_tmp.plot(bins_fit, y, 'g-', linewidth=2, alpha=0.7)

                ax1_tmp.set_xlabel(labels[j])
                if j == 0:
                    ax1_tmp.set_ylabel('Counts')

                if title is not None:
                    fmt = "{{:.{0}f}}".format(ndig[j]).format
                    msg = r"$\mu$ = {0}, $\sigma$ = {1}"
                    tit = msg.format(fmt(mu[j]), fmt(sigma[j]))
                    if gt is not None:
                        tit += r" (gt: ${{{0}}}$)".format(fmt(gt[j]))
                    ax1_tmp.set_title("{0}: {1}".format(lab, tit),
                                      fontsize=10)
                if gt is not None:
                    x_close = find_nearest(bins_fit, gt[j])
                    ax1_tmp.vlines(gt[j], 0, y[x_close], linestyles='dashed',
                                   color='blue', label='gt')
                    label = r'estimate ($\mu$)'
                else:
                    label = None

                ax1_tmp.vlines(mu[j], 0, np.amax(y), linestyles='dashed',
                               color='green', label=label)
                if gt is not None:
                    ax0_tmp.legend()
                    ax1_tmp.legend()

            else:
                ax0_tmp.hist(isamples[arg, j], bins=bin_vertices,
                             facecolor='gray', edgecolor='darkgray',
                             histtype='stepfilled', alpha=0.5)
                ax0_tmp.vlines(val_max[pKey[j]], 0, n[int(n_arg_sort[0])],
                               linestyles='dashed', color='red')
                ax0_tmp.set_xlabel(labels[j])
                if j == 0:
                    ax0_tmp.set_ylabel('Counts')

                if title is not None:
                    fmt = "{{:.{0}f}}".format(ndig[j]).format
                    msg = r"${{{0}}}_{{{1}}}^{{+{2}}}$"
                    tit = msg.format(fmt(val_max[pKey[j]]),
                                     fmt(confidenceInterval[pKey[j]][0]),
                                     fmt(confidenceInterval[pKey[j]][1]))
                    if gt is not None:
                        tit += r" (gt: ${{{0}}}$)".format(fmt(gt[j]))
                    ax0_tmp.set_title("{0}: {1}".format(lab, tit), fontsize=10)
                if gt is not None:
                    x_close = find_nearest(bin_vertices, gt[j])
                    ax0_tmp.vlines(gt[j], 0, n[x_close], label='gt',
                                   linestyles='dashed', color='blue')
                    label = 'estimate'
                else:
                    label = None
                ax0_tmp.vlines(val_max[pKey[j]], 0, n[int(n_arg_sort[0])],
                               linestyles='dashed', color='red', label=label)
                if gt is not None:
                    ax0_tmp.legend()

        else:
            arg = (isamples[:] >= bin_vertices[n_arg_min - 1]) * \
                (isamples[:] <= bin_vertices[n_arg_max + 1])
            if gaussian_fit:
                ax0_tmp.hist(isamples[arg], bins=bin_vertices,
                             facecolor='gray', edgecolor='darkgray',
                             histtype='stepfilled', alpha=0.5)

                ax0_tmp.set_xlabel(labels[j])
                if j == 0:
                    ax0_tmp.set_ylabel('Counts')

                if gt is not None:
                    x_close = find_nearest(bin_vertices, gt[j])
                    ax0_tmp.vlines(gt[j], 0, n[x_close], label='gt',
                                   linestyles='dashed', color='blue')
                    label = 'estimate'
                else:
                    label = None
                ax0_tmp.vlines(val_max[pKey[j]], 0, n[int(n_arg_sort[0])],
                               linestyles='dashed', color='red', label=label)

                if title is not None:
                    fmt = "{{:.{0}f}}".format(ndig[j]).format
                    msg = r"${{{0}}}_{{{1}}}^{{+{2}}}$"
                    tit = msg.format(fmt(val_max[pKey[j]]),
                                     fmt(confidenceInterval[pKey[j]][0]),
                                     fmt(confidenceInterval[pKey[j]][1]))
                    if gt is not None:
                        tit += r" (gt: ${{{0}}}$)".format(fmt(gt[j]))
                    ax0_tmp.set_title("{0}: {1}".format(lab, tit), fontsize=10)

                mu[j], sigma[j] = norm.fit(isamples[:])
                n_fit, bins_fit = np.histogram(isamples[:], bins, density=1,
                                               weights=weights)
                ax1_tmp.hist(isamples[:], bins, density=1, weights=weights,
                             facecolor='gray', edgecolor='darkgray',
                             histtype='step')
                y = norm.pdf(bins_fit, mu[j], sigma[j])
                ax1_tmp.plot(bins_fit, y, 'g-', linewidth=2, alpha=0.7)

                ax1_tmp.set_xlabel(labels[j])
                if j == 0:
                    ax1_tmp.set_ylabel('Counts')

                if title is not None:
                    fmt = "{{:.{0}f}}".format(ndig[j]).format
                    msg = r"$\mu$ = {{{0}}}, $\sigma$ = {{{1}}}"
                    tit = msg.format(fmt(mu[j]), fmt(sigma[j]))
                    if gt is not None:
                        tit += r" (gt: ${{{0}}}$)".format(fmt(gt[j]))
                    ax1_tmp.set_title("{0}: {1}".format(lab, tit),
                                      fontsize=10)

                if gt is not None:
                    x_close = find_nearest(bins_fit, gt[j])
                    ax1_tmp.vlines(gt[j], 0, y[x_close], label='gt',
                                   linestyles='dashed', color='blue')
                    label = r'estimate ($\mu$)'
                else:
                    label = None
                ax1_tmp.vlines(mu[j], 0, np.amax(y), linestyles='dashed',
                               color='green', label=label)
                if gt is not None:
                    ax0_tmp.legend()
                    ax1_tmp.legend()

            else:
                ax0_tmp.hist(isamples[arg], bins=bin_vertices, facecolor='gray',
                             edgecolor='darkgray', histtype='stepfilled',
                             alpha=0.5)
                ax0_tmp.set_xlabel(labels[j])
                if j == 0:
                    ax0_tmp.set_ylabel('Counts')

                if title is not None:
                    fmt = "{{:.{0}f}}".format(ndig[j]).format
                    msg = r"${{{0}}}_{{{1}}}^{{+{2}}}$"
                    tit = msg.format(fmt(val_max[pKey[j]]),
                                     fmt(confidenceInterval[pKey[j]][0]),
                                     fmt(confidenceInterval[pKey[j]][1]))
                    if gt is not None:
                        tit += r" (gt: ${{{0}}}$)".format(fmt(gt[j]))
                    ax0_tmp.set_title("{0}: {1}".format(lab, tit), fontsize=10)
                if gt is not None:
                    x_close = find_nearest(bin_vertices, gt[j])
                    ax0_tmp.vlines(gt[j], 0, n[x_close], label='gt',
                                   linestyles='dashed', color='blue')
                    label = 'estimate'
                else:
                    label = None
                ax0_tmp.vlines(val_max[pKey[j]], 0, n[int(n_arg_sort[0])],
                               linestyles='dashed', color='red', label=label)
                if gt is not None:
                    ax0_tmp.legend()

        plt.tight_layout(w_pad=0.1)

    if save:
        if gaussian_fit:
            plt.savefig(output_dir+'confi_hist_gaussfit.pdf')
        else:
            plt.savefig(output_dir+'confi_hist.pdf')

    if verbose:
        print('\n\nConfidence intervals:')
        for i, lab in enumerate(labels):
            print('{}: {} [{},{}]'.format(lab, val_max[lab],
                                          confidenceInterval[lab][0],
                                          confidenceInterval[lab][1]))
        if gaussian_fit:
            print()
            print('Gaussian fit results:')
            for i, lab in enumerate(labels):
                print('{}: {} +-{}'.format(lab, mu[i], sigma[i]))

    ############################################
    #  Write inference results in a text file  #
    ############################################
    if save:
        with open(output_dir+output_file, "w") as f:
            f.write('###########################\n')
            f.write('####   INFERENCE TEST   ###\n')
            f.write('###########################\n')
            f.write(' \n')
            f.write('Results of the MCMC fit\n')
            f.write('----------------------- \n')
            f.write(' \n')
            f.write('>> Position and flux of the planet (highly probable):\n')
            f.write('{} % confidence interval\n'.format(cfd))
            f.write(' \n')

            for i in range(l):
                confidenceMax = confidenceInterval[pKey[i]][1]
                confidenceMin = -confidenceInterval[pKey[i]][0]
                if i == 2 or l == 1:
                    text = '{}: \t\t\t{:.3f} \t-{:.3f} \t+{:.3f}\n'
                else:
                    text = '{}: \t\t\t{:.3f} \t\t-{:.3f} \t\t+{:.3f}\n'

                f.write(text.format(pKey[i], val_max[pKey[i]],
                                    confidenceMin, confidenceMax))
            if l > 1 and plsc is not None and 'r' in labels:
                f.write(' ')
                f.write('Platescale = {} mas\n'.format(plsc*1000))
                f.write('r (mas): \t\t{:.2f} \t\t-{:.2f} \t\t+{:.2f}\n'.format(
                    val_max[pKey[0]]*plsc*1000,
                    -confidenceInterval[pKey[0]][0]*plsc*1000,
                    confidenceInterval[pKey[0]][1]*plsc*1000))

    if gaussian_fit:
        return mu, sigma
    else:
        return val_max, confidenceInterval
