#! /usr/bin/env python
"""
Module with routines allowing for the estimation of the uncertainty on the
parameters of an imaged companion associated to residual speckle noise.
"""

__author__ = 'O. Wertz, C. A. Gomez Gonzalez, V. Christiaens'
__all__ = ['speckle_noise_uncertainty']

from multiprocessing import cpu_count
import numpy as np
import matplotlib.pyplot as plt

from ..config.utils_conf import pool_map, iterable  # eval_func_tuple
from ..fm import cube_inject_companions, cube_planet_free
from .negfc_simplex import firstguess_simplex
from .negfc_fmerit import get_mu_and_sigma
from .negfc_mcmc import confidence


def speckle_noise_uncertainty(cube, p_true, angle_range, derot_angles, algo,
                              psfn, fwhm, aperture_radius, opp_ang=False,
                              indep_ap=False, cube_ref=None, fmerit='sum',
                              algo_options={}, transmission=None, mu_sigma=None,
                              wedge=None, weights=None, force_rPA=False,
                              ndet=None, nproc=None, simplex_options=None,
                              bins=None, save=False, output=None, verbose=True,
                              full_output=True, plot=False, sigma_trim=None):
    """
    Step-by-step procedure used to determine the speckle noise uncertainty\
    associated to the parameters of a companion candidate.

    The steps 1 to 3 need to be performed for each angle.

    1) At the true planet radial distance and for a given angle, we \
    inject a fake companion in our planet-free cube.

    2) Then, using the negative fake companion method, we determine the \
    position and flux of the fake companion thanks to a Simplex \
    Nelder-Mead minimization.

    3) We calculate the offset between the true values of the position \
    and the flux of the fake companion, and those obtained from the \
    minimization. The results will be dependent on the angular \
    position of the fake companion.

    The resulting distribution of deviations is then used to infer the
    1-sigma uncertainty on each parameter by fitting a 1d-gaussian.

    Parameters
    ----------
    cube: 3d or 4d numpy array
        The original ADI or ADI+IFS cube.
    p_true: tuple or numpy array with 3 (or more) elements
        The radial separation, position angle (from x=0 axis) and flux
        associated to a given companion candidate for which the speckle
        uncertainty is to be evaluated. The planet will first be subtracted
        from the cube, then used for test injections. For a 4D input cube, the
        length of ``p_true`` should be equal to 2 (for r, theta) + the number
        of spectral channels (flux at each wavelength).
    angle_range: 1d numpy array
        Range of angles (counted from x=0 axis, counter-clockwise) at which the
        fake companions will be injected, in [0,360].
    derot_angles: 1d numpy array
        Derotation angles for ADI. Length should match input cube.
    algo: python routine
        Routine to be used to model and subtract the stellar PSF. From an input
        cube, derotation angles, and optional arguments, it should return a
        post-processed frame.
    psfn: 2d numpy array
        2d array with the normalized PSF template. The PSF image must be
        centered wrt to the array. Therefore, it is recommended to run the
        function ``metrics/normalize_psf()`` to generate a centered and
        flux-normalized PSF template.
    fwhm: float
        FWHM of the PSF in pixels.
    aperture_radius: float
        Radius of the apertures used for NEGFC, in terms of FWHM.
    opp_ang: bool, opt
        Whether to also use opposite derotation angles to double sample size.
        Uses the same angle range.
    indep_ap: bool, opt.
        Whether to only consider independent apertures. If yes, will supersede
        the range provided in angle_range, and only consider the first and last
        values, then fit as many non-overlapping apertures as possible.
        The empty cube will also be used with opposite derotation angles to
        double the number of independent apertures.
    algo_options: dict, opt.
        Options for algo. To be provided as a dictionary. Can include ncomp
        (for PCA), svd_mode, collapse, imlib, interpolation, scaling, delta_rot
    transmission: numpy array, optional
        Array with 2 columns. First column is the radial separation in pixels.
        Second column is the off-axis transmission (between 0 and 1) at the
        radial separation given in column 1.
    mu_sigma: tuple of 2 floats, bool or None, opt
        If set to None: not used, and falls back to original version of the
        algorithm, using fmerit.
        If a tuple of 2 elements: should be the mean and standard deviation of
        pixel intensities in an annulus centered on the location of the
        companion candidate, excluding the area directly adjacent to the CC.
        If set to anything else, but None/False/tuple: will compute said mean
        and standard deviation automatically.
    wedge: tuple, opt
        [Only used if mu_sigma is not None] Input for the
        ``vip_hci.fm.get_mu_and_sigma`` function. Range in theta where the mean
        and standard deviation are computed in an annulus defined in the PCA
        image. If None, it will be calculated automatically based on initial
        guess and derotation angles to avoid. If some disc signal is present
        elsewhere in the annulus, it is recommended to provide wedge manually.
        The provided range should be continuous and >0. E.g. provide (270, 370)
        to consider a PA range between [-90,+10].
    force_rPA: bool, optional
        Whether to only search for optimal flux, provided (r,PA).
    ndet: int or None, optional
        [only used if fmerit='hessian'] If not None, ndet should be the number
        of pixel(s) along x and y around the first guess position for which the
        determinant of the Hessian matrix is calculated. If odd, the pixel(s)
        around the closest integer coordinates will be considered. If even, the
        pixel(s) around the subpixel coordinates of the first guess location are
        considered. The figure of merit is the absolute sum of the determinants.
        If None, ndet is determined automatically to be max(1, round(fwhm/2)).
    nproc: int or None, optional
        The number of processes to use for parallelization. If None, will be set
        automatically to half the number of CPUs available.
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
    simplex_options: dict
        All the required simplex parameters, for instance {'tol':1e-08,
        'max_iter':200}
    bins: int or None, opt
        Number of bins for histogram of parameter deviations. If None, will be
        determined automatically based on number of injected fake companions.
    full_output: bool, optional
        Whether to return more outputs.
    output: str, optional
        The name of the output file (if save is True)
    save: bool, optional
        If True, the result are pickled.
    verbose: bool, optional
        If True, information is displayed in the shell.
    plot: bool, optional
        Whether to plot the gaussian fit to the distributions of parameter
        deviations (between retrieved and injected).
    sigma_trim: float, opt
        If provided, sigma threshold used to trim out outliers before
        considering a Gaussian fit to the histogram of residual deviations.

    Returns
    -------
    sp_unc: numpy ndarray of 3 elements
        Uncertainties on the radius, position angle and flux of the companion,
        respectively, associated to residual speckle noise. Only 1 element if
        force_rPA is set to True.
    mean_dev: numpy ndarray of 3 elements
        [full_output = True] Mean deviation for each of the 3 parameters
    p_simplex: numpy ndarray n_fc x 3
        [full_output = True] Parameters retrieved by the simplex for the
        injected fake companions; n_fc is the number of injected
    offset: numpy ndarray n_fc x 3
        [full_output = True] Deviations with respect to the values used for
        injection of the fake companions.
    chi2, nit, success: numpy ndarray of length n_fc
        [full_output = True] Outputs from the simplex function for the retrieval
        of the parameters of each injected companion: chi square value, number
        of iterations and whether the simplex converged, respectively.

    """
    if not nproc:   # Hyper-threading "duplicates" the cores -> cpu_count/2
        nproc = int(cpu_count()/2)

    if verbose:
        print('')
        print('#######################################################')
        print('###            SPECKLE NOISE DETERMINATION          ###')
        print('#######################################################')
        print('')

    if len(p_true) == 3:
        r_true, theta_true, f_true = p_true
        nch = 1
    elif len(p_true) > 3 and cube.ndim == 4 and cube.shape[0] == len(p_true)-2:
        r_true = p_true[0]
        theta_true = p_true[1]
        f_true = np.array(p_true[2:])
        nch = cube.shape[0]
    else:
        msg = "cube ndim ({}) and parameter length ({}) combo not accepted"
        raise TypeError(msg.format(cube.ndim, len(p_true)))

    if indep_ap:
        angle_span = angle_range[-1]-angle_range[0]
        n_ap = int(np.deg2rad(angle_span)*r_true/fwhm)
        delta_theta = angle_span/n_ap
        angle_range = np.linspace(angle_range[0]+delta_theta/2,
                                  angle_range[-1]+delta_theta/2, n_ap,
                                  endpoint=False)

    if angle_range[0] % 360 == angle_range[-1] % 360:
        angle_range = angle_range[:-1]

    if verbose:
        print('Number of steps: {}'.format(angle_range.shape[0]))
        print('')

    imlib = algo_options.get('imlib', 'vip-fft')
    interpolation = algo_options.get('interpolation', 'lanczos4')

    # FIRST SUBTRACT THE TRUE COMPANION CANDIDATE
    if len(p_true) == 3:
        planet_parameter = np.array([[r_true, theta_true, f_true]])
    else:
        planet_parameter = np.zeros([1, 3, nch])
        planet_parameter[0, 0, :] = r_true
        planet_parameter[0, 1, :] = theta_true
        planet_parameter[0, 2] = f_true
    cube_pf = cube_planet_free(planet_parameter, cube, derot_angles, psfn,
                               imlib=imlib, interpolation=interpolation,
                               transmission=transmission)

    # Measure mu and sigma once in the annulus (instead of each MCMC step)
    if isinstance(mu_sigma, tuple):
        if len(mu_sigma) != 2:
            raise TypeError("If a tuple, mu_sigma must have 2 elements")
    elif mu_sigma is not None:
        ncomp = algo_options.get('ncomp', 1)
        annulus_width = algo_options.get('annulus_width', int(fwhm))
        if weights is not None:
            if not len(weights) == cube.shape[0]:
                raise TypeError(
                    "Weights should have same length as cube axis 0")
            norm_weights = weights/np.sum(weights)
        else:
            norm_weights = weights
        mu_sigma = get_mu_and_sigma(cube, derot_angles, ncomp, annulus_width,
                                    aperture_radius, fwhm, r_true, theta_true,
                                    f_true, psfn, cube_ref=cube_ref,
                                    wedge=wedge, algo=algo,
                                    weights=norm_weights,
                                    algo_options=algo_options)

    res = pool_map(nproc, _estimate_speckle_one_angle, iterable(angle_range),
                   cube_pf, psfn, derot_angles, r_true, f_true, fwhm,
                   aperture_radius, cube_ref, fmerit, algo, algo_options,
                   transmission, mu_sigma, weights, force_rPA, ndet,
                   simplex_options, imlib, interpolation, verbose=verbose)
    residuals = np.array(res)

    if opp_ang:  # do opposite angles
        res = pool_map(nproc, _estimate_speckle_one_angle,
                       iterable(angle_range), cube_pf, psfn, -derot_angles,
                       r_true, f_true, fwhm, aperture_radius, cube_ref, fmerit,
                       algo, algo_options, transmission, mu_sigma, weights,
                       force_rPA, ndet, simplex_options, imlib, interpolation,
                       verbose=verbose)
        residuals2 = np.array(res)
        residuals = np.concatenate((residuals, residuals2))

    if verbose:
        print("residuals (offsets): ", residuals[:, nch+2], residuals[:, nch+3],
              residuals[:, nch+4])

    p_simp_stack = [residuals[:, 0], residuals[:, 1]]
    for ch in range(nch):
        p_simp_stack.append(residuals[:, 2+ch])
    p_simplex = np.transpose(np.vstack(p_simp_stack))
    p_off_stack = [residuals[:, nch+2], residuals[:, nch+3]]
    for ch in range(nch):
        p_off_stack.append(residuals[:, nch+4+ch])
    offset = np.transpose(np.vstack(p_off_stack))
    print(offset)
    chi2 = residuals[:, int(2*nch)+4]
    nit = residuals[:, int(2*nch)+5]
    success = residuals[:, int(2*nch)+6]

    if save:
        speckles = {'r_true': r_true,
                    'angle_range': angle_range,
                    'f_true': f_true,
                    'r_simplex': residuals[:, 0],
                    'theta_simplex': residuals[:, 1],
                    'f_simplex': residuals[:, 2:2+nch],
                    'offset': offset,
                    'chi2': chi2,
                    'nit': nit,
                    'success': success}

        if output is None:
            output = 'speckles_noise_result'

        from pickle import Pickler
        with open(output, 'wb') as fileSave:
            myPickler = Pickler(fileSave)
            myPickler.dump(speckles)

    # Calculate 1 sigma of distribution of deviations
    print(offset.shape)
    if force_rPA:
        offset = offset[:, 2:]
        print(offset.shape)
    if sigma_trim:
        std = np.std(offset, axis=0)
        trim_offset = []
        for i in range(offset.shape[0]):
            if np.all(np.abs(offset[i]) < sigma_trim*std):
                trim_offset.append(offset[i])
        offset = np.array(trim_offset)

    if bins is None:
        bins = int(offset.shape[0]/6)

    if force_rPA:
        labels = []
    else:
        labels = ['r', 'theta']

    if cube.ndim == 3:
        labels.append('f')
    else:
        for ch in range(nch):
            labels.append('f{}'.format(ch))

    mean_dev, sp_unc = confidence(offset, cfd=68.27, bins=bins,
                                  gaussian_fit=True, verbose=verbose,
                                  save=False, output_dir='', labels=labels,
                                  force=True, plot=verbose)
    if plot:
        plt.show()

    if full_output:
        return sp_unc, mean_dev, p_simplex, offset, chi2, nit, success
    else:
        return sp_unc


def _estimate_speckle_one_angle(angle, cube_pf, psfn, angs, r_true, f_true,
                                fwhm, aperture_radius, cube_ref, fmerit, algo,
                                algo_options, transmission, mu_sigma, weights,
                                force_rPA, ndet, simplex_options, imlib,
                                interpolation, verbose=True):

    if verbose:
        print('Process is running for angle: {:.2f}'.format(angle))

    cube_fc = cube_inject_companions(cube_pf, psfn, angs, flevel=f_true,
                                     rad_dists=[r_true], n_branches=1,
                                     theta=angle, transmission=transmission,
                                     imlib=imlib, interpolation=interpolation,
                                     verbose=False)

    if cube_pf.ndim == 4:
        p_ini = [r_true, angle]
        for f in f_true:
            p_ini.append(f)
        p_ini = tuple(p_ini)
    else:
        p_ini = (r_true, angle, f_true)

    ncomp = algo_options.get('ncomp', 1)
    annulus_width = algo_options.get('annulus_width', int(fwhm))
    delta_rot = algo_options.get('delta_rot', 1)

    res_simplex = firstguess_simplex(p_ini, cube_fc, angs,
                                     psfn, ncomp, fwhm, annulus_width,
                                     aperture_radius, cube_ref=cube_ref,
                                     fmerit=fmerit, algo=algo, delta_rot=delta_rot,
                                     algo_options=algo_options, imlib=imlib,
                                     interpolation=interpolation,
                                     transmission=transmission,
                                     mu_sigma=mu_sigma, weights=weights,
                                     force_rPA=force_rPA, ndet=ndet,
                                     options=simplex_options,
                                     verbose=False)
    res = []
    if cube_pf.ndim == 3:
        if force_rPA:
            simplex_res_f, = res_simplex.x
            simplex_res_r, simplex_res_PA = r_true, angle
        else:
            simplex_res_r, simplex_res_PA, simplex_res_f = res_simplex.x
        res.append(simplex_res_r)
        res.append(simplex_res_PA)
        res.append(simplex_res_f)
        offset_r = simplex_res_r - r_true
        offset_PA = simplex_res_PA - angle
        offset_f = simplex_res_f - f_true
        res.append(offset_r)
        res.append(offset_PA)
        res.append(offset_f)
    else:
        if force_rPA:
            simplex_res_f = np.array(res_simplex.x)
            simplex_res_r, simplex_res_PA = r_true, angle
        else:
            simplex_res = res_simplex.x
            simplex_res_r = simplex_res[0]
            simplex_res_PA = simplex_res[1]
            simplex_res_f = np.array(simplex_res[2:])
        res.append(simplex_res_r)
        res.append(simplex_res_PA)
        offset_r = simplex_res_r - r_true
        offset_PA = simplex_res_PA - angle
        offset_f = simplex_res_f - f_true
        for f in simplex_res_f:
            res.append(f)
        res.append(offset_r)
        res.append(offset_PA)
        for f in offset_f:
            res.append(f)

    chi2 = res_simplex.fun
    nit = res_simplex.nit
    success = res_simplex.success

    res.append(chi2)
    res.append(nit)
    res.append(success)

    res = tuple(res)

    return res
