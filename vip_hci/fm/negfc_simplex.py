#! /usr/bin/env python
"""
Module with simplex (Nelder-Mead) optimization for defining the flux and
position of a companion using the Negative Fake Companion.

"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

from ..config import time_ini
from ..config import timing
from ..config.utils_conf import sep
from ..psfsub import pca_annulus
from ..var import frame_center
from .negfc_fmerit import chisquare
from .negfc_fmerit import get_mu_and_sigma


__author__ = 'O. Wertz, C. A. Gomez Gonzalez, V. Christiaens'
__all__ = ['firstguess',
           'firstguess_from_coord']


def firstguess_from_coord(planet, center, cube, angs, psfn, fwhm, annulus_width,
                          aperture_radius, ncomp=1, cube_ref=None,
                          svd_mode='lapack', scaling=None, fmerit='sum',
                          imlib='skimage', interpolation='biquintic',
                          collapse='median', algo=pca_annulus, delta_rot=1,
                          algo_options={}, f_range=None, transmission=None,
                          mu_sigma=(0, 1), weights=None, ndet=None,
                          bin_spec=False, plot=False, verbose=True, save=False,
                          debug=False, full_output=False):
    """Determine a first guess for the flux of a companion at a given position\
    in the cube by doing a simple grid search evaluating the reduced chi2 using\
    the negative fake companion technique (i.e. the reduced chi2 is calculated\
    in the post-processed frame after subtraction of a negative fake companion).

    Parameters
    ----------
    planet: numpy.array
        The (x,y) position of the planet in the processed cube.
    center: numpy.array
        The (x,y) position of the cube center.
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
        The FWHM in pixels.
    annulus_width: int, optional
        The width in pixels of the annulus on which the PCA is done.
    aperture_radius: int, optional
        The radius of the circular aperture in terms of the FWHM.
    ncomp: int, optional
        The number of principal components, if the algorithm used is PCA.
    cube_ref : numpy ndarray, 3d, optional
        Reference library cube. For Reference Star Differential Imaging.
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
    collapse : {'median', 'mean', 'sum', 'trimmean', None}, str or None, opt
        Sets the way of collapsing the frames for producing a final image. If
        None then the cube of residuals is used when measuring the function of
        merit (instead of a single final frame).
    algo: python routine, opt {pca_annulus, pca_annular, pca, custom}
        Routine to be used to model and subtract the stellar PSF. From an input
        cube, derotation angles, and optional arguments, it should return a
        post-processed frame.
    delta_rot: float, optional
        If algo is set to pca_annular, delta_rot is the angular threshold used
        to select frames in the PCA library (see description of pca_annular).
    algo_options: dict, opt
        Dictionary with additional parameters for the pca algorithm (e.g. tol,
        min_frames_lib, max_frames_lib). Note: arguments such as svd_mode,
        scaling imlib, interpolation or collapse can also be included in this
        dict (the latter are also kept as function arguments for compatibility
        with older versions of vip).
    f_range: numpy.array, optional
        The range of tested flux values. If None, 30 values between 1e-1 and 1e4
        are tested.
    transmission: numpy array, optional
        Array with 2 columns. First column is the radial separation in pixels.
        Second column is the off-axis transmission (between 0 and 1) at the
        radial separation given in column 1.
    mu_sigma: tuple of 2 floats or None, opt
        If set to None: not used, and falls back to original version of the
        algorithm, using fmerit. Otherwise, should be a tuple of 2 elements,
        containing the mean and standard deviation of pixel intensities in an
        annulus centered on the location of the companion, excluding the area
        directly adjacent to the companion.
    weights : 1d array, optional
        If provided, the negative fake companion fluxes will be scaled according
        to these weights before injection in the cube. Can reflect changes in
        the observing conditions throughout the sequence.
    ndet: int or None, optional
        [only used if fmerit='hessian'] If not None, ndet should be the number
        of pixel(s) along x and y around the first guess position for which the
        determinant of the Hessian matrix is calculated. If odd, the pixel(s)
        around the closest integer coordinates will be considered. If even, the
        pixel(s) around the subpixel coordinates of the first guess location are
        considered. The figure of merit is the absolute sum of the determinants.
        If None, ndet is determined automatically to be max(1, round(fwhm/2)).
    bin_spec: bool, optional
        [only used if cube is 4D] Whether to collapse the spectral dimension
        (i.e. estimate a single binned flux) instead of estimating the flux in
        each spectral channel.
    plot: boolean, optional
        If True, the figure chi2 vs. flux is displayed.
    verbose: boolean
        If True, display intermediate info in the shell.
    save: boolean, optional
        If True, the figure chi2 vs. flux is saved as .pdf if plot is also True
    debug: bool, optional
        Whether to print details of the grid search
    full_output : bool, optional
        Whether to also return the range of fluxes tested and their chi2r
        values.

    Returns
    -------
    res : tuple
        The polar coordinates and the flux(es) of the companion.
    f_range: 1d numpy.array
        [full_output=True] The range of tested flux values.
    chi2r: 1d numpy.array
        [full_output=True] The chi2r values corresponding to tested flux values.
    """

    def _grid_search_f(r0, theta0, ch, cube, angs, psfn, fwhm, annulus_width,
                       aperture_radius, ncomp, cube_ref=None, svd_mode='lapack',
                       scaling=None, fmerit='sum', imlib='vip-fft',
                       interpolation='lanczos4', collapse='median',
                       algo=pca_annulus, delta_rot=1, algo_options={},
                       f_range=np.geomspace(1e-1, 1e4, 30), transmission=None,
                       mu_sigma=None, weights=None, ndet=None, bin_spec=False,
                       verbose=True, debug=False):

        chi2r = []
        if verbose:
            print('Step | flux    | chi2r')

        counter = 0
        for j, f_guess in enumerate(f_range):
            if cube.ndim == 3 or (cube.ndim == 4 and bin_spec):
                params = (r0, theta0, f_guess)
            elif ch is not None and cube.ndim == 4:
                params = [r0, theta0]
                fluxes = [0]*cube.shape[0]
                fluxes[ch] = f_guess
                params = tuple(params+fluxes)
            else:
                raise TypeError("If cube is 4d, channel index must be provided")
            chi2r.append(chisquare(params, cube, angs, psfn, fwhm,
                                   annulus_width, aperture_radius, (r0, theta0),
                                   ncomp, cube_ref, svd_mode, scaling, fmerit,
                                   collapse, algo, delta_rot, imlib,
                                   interpolation, algo_options, transmission,
                                   mu_sigma, weights, False, ndet, bin_spec,
                                   debug))
            if chi2r[j] > chi2r[j-1]:
                counter += 1
            if counter == 4:
                break
            if verbose:
                print('{}/{}   {:.3f}   {:.3f}'.format(j +
                      1, n, f_guess, chi2r[j]))

        return chi2r

    xy = planet-center
    r0 = np.sqrt(xy[0]**2 + xy[1]**2)
    theta0 = np.mod(np.arctan2(xy[1], xy[0]) / np.pi*180, 360)

    if f_range is not None:
        n = f_range.shape[0]
    else:
        n = 30
        f_range = np.geomspace(1e-1, 1e4, n)

    if cube.ndim == 3 or bin_spec:
        chi2r = _grid_search_f(r0, theta0, None, cube, angs, psfn, fwhm,
                               annulus_width, aperture_radius, ncomp,
                               cube_ref=cube_ref, svd_mode=svd_mode,
                               scaling=scaling, fmerit=fmerit, imlib=imlib,
                               interpolation=interpolation, collapse=collapse,
                               algo=algo, delta_rot=delta_rot,
                               algo_options=algo_options, f_range=f_range,
                               transmission=transmission, mu_sigma=mu_sigma,
                               weights=weights, ndet=ndet, bin_spec=bin_spec,
                               verbose=verbose, debug=debug)
        chi2r = np.array(chi2r)
        f0 = f_range[chi2r.argmin()]

        if plot:
            plt.figure(figsize=(8, 4))
            plt.title('$\\chi^2_{r}$ vs flux')
            plt.xlim(f_range[0], f_range[:chi2r.shape[0]].max())
            plt.ylim(chi2r.min()*0.9, chi2r.max()*1.1)
            plt.plot(f_range[:chi2r.shape[0]], chi2r, linestyle='-',
                     color='gray', marker='.', markerfacecolor='r',
                     markeredgecolor='r')
            plt.xlabel('flux')
            plt.ylabel(r'$\chi^2_r$')
            plt.grid('on')
        if save and plot:
            plt.savefig('chi2rVSflux.pdf')
        if plot:
            plt.show()

        res = (r0, theta0, f0)

    else:
        f0 = []
        chi2r = []
        if plot:
            plt.figure(figsize=(8, 4))
            plt.title('$\\chi^2_{r}$ vs flux')
            plt.xlabel('flux')
            plt.ylabel(r'$\chi^2_{r}$')
            plt.grid('on')

        for i in range(cube.shape[0]):
            if verbose:
                print('Processing spectral channel {}...'.format(i))
            chi2r_tmp = _grid_search_f(r0, theta0, i, cube, angs, psfn, fwhm,
                                       annulus_width, aperture_radius, ncomp,
                                       cube_ref=cube_ref, svd_mode=svd_mode,
                                       scaling=scaling, fmerit=fmerit,
                                       imlib=imlib, interpolation=interpolation,
                                       collapse=collapse, algo=algo,
                                       delta_rot=delta_rot,
                                       algo_options=algo_options,
                                       f_range=f_range,
                                       transmission=transmission,
                                       mu_sigma=mu_sigma, weights=weights,
                                       ndet=ndet, verbose=False, debug=False)
            chi2r.append(chi2r_tmp)
            chi2r_tmp = np.array(chi2r_tmp)
            f0.append(f_range[chi2r_tmp.argmin()])
            if verbose:
                msg = r'... optimal grid flux: {:.3f} ($\chi^2_r$ = {:.1f})'
                print(msg.format(f0[i], np.amin(chi2r_tmp)))

            if i == 0:
                min_chi2r = chi2r_tmp.min()
                max_chi2r = chi2r_tmp.max()
                fmax = f0[i]
            else:
                if min_chi2r > chi2r_tmp.min():
                    min_chi2r = chi2r_tmp.min()
                if max_chi2r < chi2r_tmp.max():
                    max_chi2r = chi2r_tmp.max()
                if fmax < f0[i]:
                    fmax = f0[i]

            if plot:
                plt.plot(f_range[:chi2r_tmp.shape[0]], chi2r_tmp, linestyle='-',
                         marker='.', markerfacecolor='r', markeredgecolor='r',
                         label='ch. {}'.format(i))

        if plot:
            plt.xlim(f_range[0], f_range[:chi2r_tmp.shape[0]].max())
            plt.ylim(min_chi2r*0.9, max_chi2r*1.1)
            plt.legend()
        if save and plot:
            plt.savefig('chi2rVSflux.pdf')
        if plot:
            plt.show()

        res = tuple([r0, theta0]+f0)

    if full_output:
        return res, f_range, chi2r
    else:
        return res


def firstguess_simplex(p, cube, angs, psfn, ncomp, fwhm, annulus_width,
                       aperture_radius, cube_ref=None, svd_mode='lapack',
                       scaling=None, fmerit='sum', imlib='skimage',
                       interpolation='biquintic', collapse='median',
                       algo=pca_annulus, delta_rot=1, algo_options={},
                       p_ini=None, transmission=None, mu_sigma=(0, 1),
                       weights=None, force_rPA=False, ndet=None, bin_spec=False,
                       options=None, verbose=False, **kwargs):
    """Determine the position of a companion using the negative fake companion\
    technique and a standard minimization algorithm (Default=Nelder-Mead).

    Parameters
    ----------
    p : np.array
        Estimate of the candidate position.
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
    ncomp: int or None
        The number of principal components to use, if the algorithm is PCA.
    fwhm : float
        The FWHM in pixels.
    annulus_width: int, optional
        The width in pixels of the annulus on which the PCA is done.
    aperture_radius: int, optional
        The radius of the circular aperture in terms of the FWHM.
    cube_ref : numpy ndarray, 3d, optional
        Reference library cube. For Reference Star Differential Imaging.
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
    collapse : {'median', 'mean', 'sum', 'trimmean', None}, str or None, opt
        Sets the way of collapsing the frames for producing a final image. If
        None then the cube of residuals is used when measuring the function of
        merit (instead of a single final frame).
    algo: python routine, opt {pca_annulus, pca_annular, pca, custom}
        Routine to be used to model and subtract the stellar PSF. From an input
        cube, derotation angles, and optional arguments, it should return a
        post-processed frame.
    delta_rot: float, optional
        If algo is set to pca_annular, delta_rot is the angular threshold used
        to select frames in the PCA library (see description of pca_annular).
    algo_options: dict, opt
        Dictionary with additional parameters for the pca algorithm (e.g. tol,
        min_frames_lib, max_frames_lib). Note: arguments such as svd_mode,
        scaling imlib, interpolation or collapse can also be included in this
        dict (the latter are also kept as function arguments for compatibility
        with older versions of vip).
    p_ini : np.array
        Position (r, theta) of the circular aperture center.
    transmission: numpy array, optional
        Array with 2 columns. First column is the radial separation in pixels.
        Second column is the off-axis transmission (between 0 and 1) at the
        radial separation given in column 1.
    mu_sigma: tuple of 2 floats or None, opt
        If set to None: not used, and falls back to original version of the
        algorithm, using fmerit. Otherwise, should be a tuple of 2 elements,
        containing the mean and standard deviation of pixel intensities in an
        annulus centered on the location of the companion, excluding the area
        directly adjacent to the companion.
    weights : 1d array, optional
        If provided, the negative fake companion fluxes will be scaled according
        to these weights before injection in the cube. Can reflect changes in
        the observing conditions throughout the sequence.
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
    bin_spec: bool, optional
        [only used if cube is 4D] Whether to collapse the spectral dimension
        (i.e. estimate a single binned flux) instead of estimating the flux in
        each spectral channel.
    options: dict, optional
        The scipy.optimize.minimize options.
    verbose : boolean, optional
        If True, additional information is printed out.
    **kwargs: optional
        Optional arguments to the scipy.optimize.minimize function

    Returns
    -------
    solu : scipy.optimize.minimize solution object
        The solution of the minimization algorithm.

    """
    if verbose:
        print('\nNelder-Mead minimization is running...')

    if p_ini is None:
        p_ini = p

    if force_rPA:
        p_t = p[2:]
        p_ini = (p[0], p[1])
    else:
        p_t = p
    solu = minimize(chisquare, p_t, args=(cube, angs, psfn, fwhm, annulus_width,
                                          aperture_radius, p_ini, ncomp,
                                          cube_ref, svd_mode, scaling, fmerit,
                                          collapse, algo, delta_rot, imlib,
                                          interpolation, algo_options,
                                          transmission, mu_sigma, weights,
                                          force_rPA, ndet, bin_spec),
                    method='Nelder-Mead', options=options, **kwargs)

    if verbose:
        print(solu)
    return solu


def firstguess(cube, angs, psfn, planets_xy_coord, ncomp=1, fwhm=4,
               annulus_width=4, aperture_radius=1, cube_ref=None,
               svd_mode='lapack', scaling=None, fmerit='sum', imlib='skimage',
               interpolation='biquintic', collapse='median', algo=pca_annulus,
               delta_rot=1, f_range=None, transmission=None, mu_sigma=True,
               wedge=None, weights=None, force_rPA=False, ndet=None,
               bin_spec=False, algo_options={}, simplex=True,
               simplex_options=None, plot=False, verbose=True, save=False):
    """Determine a first guess for the position and the flux of a planet using\
    the negative fake companion technique, as explained in [WER17]_.

    This first requires processing the cube without injecting any negative fake
    companion. Once planets or planet candidates are identified, their initial
    guess (x,y) coordinates can be provided to this function. A preliminary flux
    guess is then found for each planet by using the method
    ``firstguess_from_coord`` called within this function. Optionally, a Simplex
    Nelder_Mead minimization is used for a refined estimate of position and flux
    based on the preliminary guesses.

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
    planets_xy_coord: array or list
        The list of (x,y) positions of the planets.
    ncomp : int or 1d numpy array of int, optional
        The number of principal components to use, if the algorithm is PCA. If
        the input cube is 4D, ncomp can be a list of integers, with length
        matching the first dimension of the cube.
    fwhm : float, optional
        The FWHM in pixels.
    annulus_width: int, optional
        The width in pixels of the annulus on which the PCA is done.
    aperture_radius: int, optional
        The radius of the circular aperture in terms of the FWHM.
    cube_ref : numpy ndarray, 3d, optional
        Reference library cube. For Reference Star Differential Imaging.
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
    collapse : {'median', 'mean', 'sum', 'trimmean', None}, str or None, opt
        Sets the way of collapsing the frames for producing a final image. If
        None then the cube of residuals is used when measuring the function of
        merit (instead of a single final frame).
    algo: python routine, opt {pca_annulus, pca_annular, pca, custom}
        Routine to be used to model and subtract the stellar PSF. From an input
        cube, derotation angles, and optional arguments, it should return a
        post-processed frame.
    delta_rot: float, optional
        If algo is set to pca_annular, delta_rot is the angular threshold used
        to select frames in the PCA library (see description of pca_annular).
    f_range: numpy.array, optional
        The range of flux tested values. If None, 20 values between 0 and 5000
        are tested.
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
        Range in theta where the mean and standard deviation are computed in an
        annulus defined in the PCA image. If None, it will be calculated
        automatically based on initial guess and derotation angles to avoid.
        If some disc signal is present elsewhere in the annulus, it is
        recommended to provide wedge manually. The provided range should be
        continuous and >0. E.g. provide (270, 370) to consider a PA range
        between [-90,+10].
    weights : 1d array, optional
        If provided, the negative fake companion fluxes will be scaled according
        to these weights before injection in the cube. Can reflect changes in
        the observing conditions throughout the sequence.
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
    bin_spec: bool, optional
        [only used if cube is 4D] Whether to collapse the spectral dimension
        (i.e. estimate a single binned flux) instead of estimating the flux in
        each spectral channel.
    algo_options: dict, opt
        Dictionary with additional parameters for the pca algorithm (e.g. tol,
        min_frames_lib, max_frames_lib). Note: arguments such as svd_mode,
        scaling imlib, interpolation or collapse can also be included in this
        dict (the latter are also kept as function arguments for compatibility
        with older versions of vip).
    simplex: bool, optional
        If True, the Nelder-Mead minimization is performed after the flux grid
        search.
    simplex_options: dict, optional
        The scipy.optimize.minimize options.
    plot: boolean, optional
        If True, the figure chi2 vs. flux is displayed.
    verbose: bool, optional
        If True, display intermediate info in the shell.
    save: bool, optional
        If True, the figure chi2 vs. flux is saved.

    Returns
    -------
    out : tuple of 3+ elements
        The polar coordinates and the flux(es) of the companion.

    Note
    ----
    Polar angle is not the conventional NORTH-TO-EAST P.A., but the
    counter-clockwise angle measured from the positive x axis.
    """
    if cube.ndim != 3 and cube.ndim != 4:
        raise TypeError("Input cube is not 3D nor 4D")

    if verbose:
        start_time = time_ini()

    planets_xy_coord = np.array(planets_xy_coord)
    n_planet = planets_xy_coord.shape[0]
    center_xy_coord = np.array(frame_center(cube[0]))

    r_0 = np.zeros(n_planet)
    theta_0 = np.zeros_like(r_0)
    if cube.ndim == 3:
        f_0 = np.zeros_like(r_0)
    else:
        if psfn.ndim < 3:
            msg = "The normalized PSF should be 3D for a 4D input cube"
            raise TypeError(msg)
        if cube.ndim == 4 and bin_spec:
            f_0 = np.zeros_like(r_0)
        else:
            f_0 = np.zeros([n_planet, cube.shape[0]])

    if weights is not None:
        if not len(weights) == cube.shape[-3]:
            msg = "Weights should have same length as temporal cube axis"
            raise TypeError(msg)
        norm_weights = weights/np.sum(weights)
    else:
        norm_weights = weights

    for i_planet in range(n_planet):
        if verbose:
            print('\n'+sep)
            print('             Planet {}           '.format(i_planet))
            print(sep+'\n')
            msg2 = 'Planet {}: flux estimation at the position [{},{}], '
            msg2 += 'running ...'
            print(msg2.format(i_planet, planets_xy_coord[i_planet, 0],
                              planets_xy_coord[i_planet, 1]))
        # Measure mu and sigma once in the annulus (instead of each MCMC step)
        if isinstance(mu_sigma, tuple):
            if len(mu_sigma) != 2:
                raise TypeError("If a tuple, mu_sigma must have 2 elements")
        elif mu_sigma is not None:
            xy = planets_xy_coord[i_planet]-center_xy_coord
            r0 = np.sqrt(xy[0]**2 + xy[1]**2)
            theta0 = np.mod(np.arctan2(xy[1], xy[0]) / np.pi*180, 360)
            mu_sigma = get_mu_and_sigma(cube, angs, ncomp, annulus_width,
                                        aperture_radius, fwhm, r0,
                                        theta0, cube_ref=cube_ref,
                                        wedge=wedge, svd_mode=svd_mode,
                                        scaling=scaling, algo=algo,
                                        delta_rot=delta_rot, imlib=imlib,
                                        interpolation=interpolation,
                                        collapse=collapse, weights=norm_weights,
                                        algo_options=algo_options,
                                        bin_spec=bin_spec)

        res_init = firstguess_from_coord(planets_xy_coord[i_planet],
                                         center_xy_coord, cube, angs,
                                         psfn, fwhm, annulus_width,
                                         aperture_radius, ncomp,
                                         f_range=f_range, cube_ref=cube_ref,
                                         svd_mode=svd_mode, scaling=scaling,
                                         fmerit=fmerit, imlib=imlib,
                                         collapse=collapse, algo=algo,
                                         delta_rot=delta_rot,
                                         interpolation=interpolation,
                                         algo_options=algo_options,
                                         transmission=transmission,
                                         mu_sigma=mu_sigma, weights=weights,
                                         ndet=ndet, bin_spec=bin_spec,
                                         plot=plot, verbose=verbose, save=save)

        r_pre = res_init[0]
        theta_pre = res_init[1]
        f_pre = res_init[2:]

        if verbose:
            msg3a = 'Planet {}: preliminary position guess: (r, theta)=({:.1f},'
            msg3a += ' {:.1f})'
            print(msg3a.format(i_planet, r_pre, theta_pre))
            msg3b = 'Planet {}: preliminary flux guess: '.format(i_planet)
            for z in range(len(f_pre)):
                msg3b += '{:.2f}'.format(f_pre[z])
                if z < len(f_pre)-1:
                    msg3b += ', '
            print(msg3b)

        if simplex or force_rPA:
            if verbose:
                msg4 = 'Planet {}: Simplex Nelder-Mead minimization, '
                msg4 += 'running ...'
                print(msg4.format(i_planet))

            if simplex_options is None:
                simplex_options = {'xatol': 1e-6, 'fatol': 1e-6, 'maxiter': 800,
                                   'maxfev': 2000}

            res = firstguess_simplex(res_init, cube, angs, psfn, ncomp, fwhm,
                                     annulus_width, aperture_radius,
                                     cube_ref=cube_ref, svd_mode=svd_mode,
                                     scaling=scaling, fmerit=fmerit,
                                     imlib=imlib, interpolation=interpolation,
                                     collapse=collapse, algo=algo,
                                     delta_rot=delta_rot,
                                     algo_options=algo_options,
                                     transmission=transmission,
                                     mu_sigma=mu_sigma, weights=weights,
                                     force_rPA=force_rPA, ndet=ndet,
                                     bin_spec=bin_spec, options=simplex_options,
                                     verbose=False)
            if force_rPA:
                r_0[i_planet], theta_0[i_planet] = (r_pre, theta_pre)
                f_0[i_planet] = res.x[:]
            else:
                r_0[i_planet] = res.x[0]
                theta_0[i_planet] = res.x[1]
                if cube.ndim == 3 or (cube.ndim == 4 and bin_spec):
                    f_0[i_planet] = res.x[2]
                else:
                    f_0[i_planet] = res.x[2:]
            if verbose:
                msg5 = 'Planet {}: Success: {}, nit: {}, nfev: {}, chi2r: {}'
                print(msg5.format(i_planet, res.success, res.nit, res.nfev,
                                  res.fun))
                print('message: {}'.format(res.message))

        else:
            if verbose:
                msg4bis = 'Planet {}: Simplex Nelder-Mead minimization skipped.'
                print(msg4bis.format(i_planet))
            r_0[i_planet] = r_pre
            theta_0[i_planet] = theta_pre
            if cube.ndim == 3 or (cube.ndim == 4 and bin_spec):
                f_0[i_planet] = f_pre[0]
            else:
                f_0[i_planet] = f_pre

        if verbose:
            centy, centx = frame_center(cube[0])
            posy = r_0 * np.sin(np.deg2rad(theta_0[i_planet])) + centy
            posx = r_0 * np.cos(np.deg2rad(theta_0[i_planet])) + centx
            msg6 = 'Planet {} simplex result: (r, theta, '.format(i_planet)
            if cube.ndim == 3 or (cube.ndim == 4 and bin_spec):
                msg6 += 'f)=({:.3f}, {:.3f}, {:.3f})'.format(r_0[i_planet],
                                                             theta_0[i_planet],
                                                             f_0[i_planet])
            else:
                msg6b = '('
                for z in range(cube.shape[0]):
                    msg6 += 'f{}'.format(z)
                    msg6b += '{:.3f}'.format(f_0[i_planet, z])
                    if z < cube.shape[0]-1:
                        msg6 += ', '
                        msg6b += ', '
                msg6 += ')='
                msg6b += ')'
                msg6 += msg6b
            msg6 += ' at \n          (X,Y)=({:.2f}, {:.2f})'.format(posx[0],
                                                                    posy[0])
            print(msg6)

    if verbose:
        print('\n', sep, '\nDONE !\n', sep)
        timing(start_time)

    return r_0, theta_0, f_0
