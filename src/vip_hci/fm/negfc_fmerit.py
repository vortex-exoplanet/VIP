#! /usr/bin/env python
"""Module with function of merit definitions for the NEGFC optimization."""

__author__ = "O. Wertz, Carlos Alberto Gomez Gonzalez, Valentin Christiaens"
__all__ = ["get_mu_and_sigma"]

import numpy as np

from hciplot import plot_frames
from skimage.draw import disk
from ..fm import cube_inject_companions, cube_planet_free
from ..var import (frame_center, get_annular_wedge, cube_filter_highpass,
                   get_annulus_segments)
from ..psfsub import pca_annulus, pca_annular, nmf_annular, pca
from ..preproc import cube_crop_frames, frame_crop


def chisquare(
    modelParameters,
    cube,
    angs,
    psfs_norm,
    fwhm,
    annulus_width,
    aperture_radius,
    initialState,
    ncomp,
    cube_ref=None,
    svd_mode="lapack",
    scaling=None,
    fmerit="sum",
    collapse="median",
    algo=pca_annulus,
    delta_rot=1,
    imlib="vip-fft",
    interpolation="lanczos4",
    algo_options={},
    transmission=None,
    mu_sigma=(0, 1),
    weights=None,
    force_rPA=False,
    ndet=None,
    bin_spec=False,
    debug=False,
):
    r"""
    Calculate the reduced :math:`\chi^2`:
    .. math:: \chi^2_r = \frac{1}{N-Npar}\sum_{j=1}^{N} \frac{(I_j-\mu)^2}{\sigma^2}
    (mu_sigma is a tuple) or:
    .. math:: \chi^2_r = \frac{1}{N-Npar}\sum_{j=1}^{N} |I_j| (mu_sigma=None),
    where N is the number of pixels within a circular aperture centered on the
    first estimate of the planet position, Npar the number of parameters to be
    fitted (3 for a 3D input cube, 2+n_ch for a 4D input cube), and :math:`I_j`
    the j-th pixel intensity.

    Parameters
    ----------
    modelParameters: tuple
        The model parameters: (r, theta, flux) for a 3D input cube, or
        (r, theta, f1, ..., fN) for a 4D cube with N spectral channels.
    cube: 3d or 4d numpy ndarray
        Input ADI or ADI+IFS cube.
    angs : 1d numpy ndarray
        Vector of derotation angles to align North up in your cube images.
    psfs_norm: numpy.array
        The scaled psf expressed as a numpy.array.
    fwhm : float
        The FWHM in pixels.
    annulus_width: int, optional
        The width in pixels of the annulus on which the PCA is done.
    aperture_radius: int, optional
        The radius of the circular aperture in terms of the FWHM.
    initialState: numpy.array
        Position (r, theta) of the circular aperture center.
    ncomp: int or None
        The number of principal components for PCA-based algorithms.
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
    imlib : str, optional
        See the documentation of the ``vip_hci.preproc.frame_shift`` function.
    interpolation : str, optional
        See the documentation of the ``vip_hci.preproc.frame_shift`` function.
    algo_options: dict, opt
        Dictionary with additional parameters related to the algorithm
        (e.g. tol, min_frames_lib, max_frames_lib). If 'algo' is not a vip
        routine, this dict should contain all necessary arguments apart from
        the cube and derotation angles. Note: arguments such as ncomp, svd_mode,
        scaling, imlib, interpolation or collapse can also be included in this
        dict (the latter are also kept as function arguments for consistency
        with older versions of vip).
    transmission: numpy array, optional
        Array with 2 columns. First column is the radial separation in pixels.
        Second column is the off-axis transmission (between 0 and 1) at the
        radial separation given in column 1.
    mu_sigma: tuple of 2 floats or None, opt
        If set to None: not used, and falls back to original version of the
        algorithm, using fmerit.
        If set to anything but None: will compute the mean and standard
        deviation of pixel intensities in an annulus centered on the location
        of the companion, excluding the area directly adjacent to the companion.
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
    debug: bool, opt
        Whether to debug and plot the post-processed frame after injection of
        the negative fake companion.
    Returns
    -------
    out: float
        The reduced chi squared.

    """
    if cube.ndim == 3 or (cube.ndim == 4 and bin_spec):
        if force_rPA:
            r, theta = initialState
            flux_tmp = modelParameters[0]
        else:
            try:
                r, theta, flux_tmp = modelParameters
            except TypeError:
                msg = "modelParameters must be a tuple, {} was given"
                print(msg.format(type(modelParameters)))
    else:
        if force_rPA:
            r, theta = initialState
            flux_tmp = np.array(modelParameters)
        else:
            try:
                r = modelParameters[0]
                theta = modelParameters[1]
                flux_tmp = np.array(modelParameters[2:])
            except TypeError:
                msg = "modelParameters must be a tuple, {} was given"
                print(msg.format(type(modelParameters)))

    # set imlib for rotation and shift
    if imlib == "opencv":
        imlib_sh = imlib
        imlib_rot = imlib
    elif imlib == "skimage" or imlib == "ndimage-interp":
        imlib_sh = "ndimage-interp"
        imlib_rot = "skimage"
    elif imlib == "vip-fft" or imlib == "ndimage-fourier":
        imlib_sh = "ndimage-fourier"
        imlib_rot = "vip-fft"
    else:
        raise TypeError("Interpolation not recognized.")

    norm_weights = None
    if weights is None:
        flux = -flux_tmp
        # norm_weights=weights
    elif np.isscalar(flux_tmp):
        flux = -flux_tmp * weights
        # norm_weights=weights
        # norm_weights = weights/np.sum(weights)
    else:
        flux = -np.outer(flux_tmp, weights)
        # norm_weights=weights
        # norm_weights = weights/np.sum(weights)

    # Create the cube with the negative fake companion injected
    cube_negfc = cube_inject_companions(
        cube,
        psfs_norm,
        angs,
        flevel=flux,
        rad_dists=[r],
        n_branches=1,
        theta=theta,
        imlib=imlib_sh,
        interpolation=interpolation,
        transmission=transmission,
        verbose=False
    )

    # Perform PCA and extract the zone of interest
    full_output = (debug and collapse) or (fmerit == "hessian")

    res = get_values_optimize(
        cube_negfc,
        angs,
        ncomp,
        annulus_width,
        aperture_radius,
        fwhm,
        initialState[0],
        initialState[1],
        cube_ref=cube_ref,
        svd_mode=svd_mode,
        scaling=scaling,
        algo=algo,
        delta_rot=delta_rot,
        collapse=collapse,
        algo_options=algo_options,
        weights=norm_weights,
        imlib=imlib_rot,
        interpolation=interpolation,
        full_output=full_output,
    )

    if full_output:
        values, frpca = res
        if debug:
            plot_frames(frpca)
    else:
        values = res

    # Function of merit
    if mu_sigma is None:
        if fmerit == "sum":
            ddf = values.size - len(modelParameters)
            chi = np.nansum(np.abs(values)) / ddf
        elif fmerit == "stddev":
            values = values[values != 0]
            ddf = values.size - len(modelParameters)
            chi = np.nanstd(values) * values.size / ddf  # TODO: test std**2
        elif fmerit == "hessian":
            # number of Hessian determinants (i.e. of pixels) to consider
            if ndet is None:
                ndet = int(round(max(min(fwhm/2, r), 2)))
            elif not isinstance(ndet, int):
                raise TypeError("If provided, ndet should be an integer")

            # consider a sub-image in the post-processed image
            ny, nx = frpca.shape[-2:]
            cy, cx = frame_center(frpca)
            yi = cy+r*np.sin(np.deg2rad(theta))
            xi = cx+r*np.cos(np.deg2rad(theta))
            if ndet % 2:
                # odd crop
                yround, xround = int(np.round(yi)), int(np.round(xi))
            else:
                # even crop
                yround, xround = int(np.ceil(yi)), int(np.ceil(xi))
            crop_sz = ndet+4

            # check there is enough space around the location to crop
            spaces = [yround, xround, ny-yround, nx-xround]
            if crop_sz/2 > np.amin(spaces):
                msg = "Test location too close from image edge for Hessian "
                msg += "calculation. Consider larger input images."
                raise ValueError(msg)

            subim = frame_crop(frpca, crop_sz, xy=(xround, yround),
                               force=True, verbose=False)
            H = hessian(subim)
            dets = np.zeros([ndet, ndet])
            for i in range(ndet):
                for j in range(ndet):
                    dets[i, j] = np.linalg.det(H[:, :, 2+i, 2+j])
            chi = np.sum(np.abs(dets))
        else:
            raise RuntimeError("fmerit choice not recognized.")
    else:
        # true expression of a gaussian log probability
        mu = mu_sigma[0]
        sigma = mu_sigma[1]
        ddf = values.size - len(modelParameters)
        chi = np.sum(np.power(mu - values, 2) / sigma**2) / ddf

    return chi


def get_values_optimize(
    cube,
    angs,
    ncomp,
    annulus_width,
    aperture_radius,
    fwhm,
    r_guess,
    theta_guess,
    cube_ref=None,
    svd_mode="lapack",
    scaling=None,
    algo=pca_annulus,
    delta_rot=1,
    imlib="vip-fft",
    interpolation="lanczos4",
    collapse="median",
    algo_options={},
    weights=None,
    full_output=False,
):
    """Extract a PCA-ed annulus from the cube and returns the flux values of\
    the pixels included in a circular aperture centered at a given position.

    Parameters
    ----------
    cube: 3d or 4d numpy ndarray
        Input ADI or ADI+IFS cube.
    angs : 1d numpy ndarray
        Vector of derotation angles to align North up in your cube images.
    ncomp: int or None
        The number of principal components for PCA-based algorithms.
    annulus_width: float
        The width in pixels of the annulus on which the PCA is performed.
    aperture_radius: float
        The radius in fwhm of the circular aperture.
    fwhm: float
        Value of the FWHM of the PSF.
    r_guess: float
        The radial position of the center of the circular aperture. This
        parameter is NOT the radial position of the candidate associated to the
        Markov chain, but should be the fixed initial guess.
    theta_guess: float
        The angular position of the center of the circular aperture. This
        parameter is NOT the angular position of the candidate associated to the
        Markov chain, but should be the fixed initial guess.
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
    algo: python routine, opt {pca_annulus, pca_annular, pca, custom}
        Routine to be used to model and subtract the stellar PSF. From an input
        cube, derotation angles, and optional arguments, it should return a
        post-processed frame.
    delta_rot: float, optional
        If algo is set to pca_annular, delta_rot is the angular threshold used
        to select frames in the PCA library (see description of pca_annular).
    imlib : str, optional
        See the documentation of the ``vip_hci.preproc.frame_rotate`` function.
    interpolation : str, optional
        See the documentation of the ``vip_hci.preproc.frame_rotate`` function.
    collapse : {'median', 'mean', 'sum', 'trimmean', None}, str or None, opt
        Sets the way of collapsing the frames for producing a final image. If
        None then the cube of residuals is returned.
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
    full_output: boolean
        If True, the cube is returned along with the values.

    Returns
    -------
    values: numpy ndarray
        The pixel values in the circular aperture after the PCA process.
    res: numpy ndarray
        [full_output=True & collapse!= None] The post-processed image.

    """
    ceny_fr, cenx_fr = frame_center(cube[0])
    posy = r_guess * np.sin(np.deg2rad(theta_guess)) + ceny_fr
    posx = r_guess * np.cos(np.deg2rad(theta_guess)) + cenx_fr
    halfw = max(aperture_radius * fwhm, annulus_width / 2)

    # Checking annulus/aperture sizes. Assuming square frames
    msg = "The annulus and/or the circular aperture used by the NegFC falls "
    msg += "outside the FOV. Try increasing the size of your frames or "
    msg += "decreasing the annulus or aperture size. "
    msg += "r_guess: {:.1f}px; half xy dim: {:.1f}px; ".format(r_guess, cenx_fr)
    msg += "Aperture radius: {:.1f}px ".format(aperture_radius * fwhm)
    msg += "Annulus half width: {:.1f}px".format(annulus_width / 2)
    if r_guess > cenx_fr - halfw:  # or r_guess <= halfw:
        raise RuntimeError(msg)

    algo_opt_copy = algo_options.copy()
    ncomp = algo_opt_copy.pop("ncomp", ncomp)
    svd_mode = algo_opt_copy.pop("svd_mode", svd_mode)
    scaling = algo_opt_copy.pop("scaling", scaling)
    imlib = algo_opt_copy.pop("imlib", imlib)
    interpolation = algo_opt_copy.pop("interpolation", interpolation)
    collapse = algo_opt_copy.pop("collapse", collapse)
    collapse_ifs = algo_opt_copy.pop("collapse_ifs", "absmean")
    nproc = algo_opt_copy.pop("nproc", 1)
    verbose = algo_opt_copy.pop("verbose", False)
    if algo == pca:
        mask_rdi = algo_opt_copy.pop("mask_rdi", None)

    if algo == pca_annulus:
        res = pca_annulus(
            cube,
            angs,
            ncomp,
            annulus_width,
            r_guess,
            cube_ref,
            svd_mode,
            scaling,
            imlib=imlib,
            interpolation=interpolation,
            collapse=collapse,
            collapse_ifs=collapse_ifs,
            weights=weights,
            nproc=nproc,
            **algo_opt_copy,
        )

    elif algo == pca_annular or algo == nmf_annular:
        tol = algo_opt_copy.pop("tol", 1e-1)
        min_frames_lib = algo_opt_copy.pop("min_frames_lib", 2)
        max_frames_lib = algo_opt_copy.pop("max_frames_lib", 200)
        radius_int = max(1, int(np.floor(r_guess - annulus_width / 2)))
        radius_int = algo_opt_copy.pop("radius_int", radius_int)
        asize = algo_opt_copy.pop("asize", annulus_width)
        delta_rot = algo_opt_copy.pop("delta_rot", delta_rot)
        # crop cube to just be larger than annulus => FASTER PCA
        crop_sz = int(2 * np.ceil(radius_int + asize + 1))
        if not crop_sz % 2:
            crop_sz += 1
        if crop_sz < cube.shape[-2] and crop_sz < cube.shape[-1]:
            pad = int((cube.shape[-2] - crop_sz) / 2)
            crop_cube = cube_crop_frames(cube, crop_sz, verbose=False)
        else:
            crop_cube = cube
            pad = 0
        if algo == pca_annular:
            res_tmp = algo(
                cube=crop_cube,
                angle_list=angs,
                cube_ref=cube_ref,
                radius_int=radius_int,
                fwhm=fwhm,
                asize=asize,
                delta_rot=delta_rot,
                ncomp=ncomp,
                svd_mode=svd_mode,
                scaling=scaling,
                imlib=imlib,
                interpolation=interpolation,
                collapse=collapse,
                collapse_ifs=collapse_ifs,
                weights=weights,
                tol=tol,
                nproc=nproc,
                min_frames_lib=min_frames_lib,
                max_frames_lib=max_frames_lib,
                full_output=False,
                verbose=verbose,
                **algo_opt_copy,
            )
        else:
            res_tmp = algo(
                cube=crop_cube,
                angle_list=angs,
                cube_ref=cube_ref,
                radius_int=radius_int,
                fwhm=fwhm,
                asize=annulus_width,
                delta_rot=delta_rot,
                ncomp=ncomp,
                scaling=scaling,
                imlib=imlib,
                interpolation=interpolation,
                collapse=collapse,
                weights=weights,
                nproc=nproc,
                min_frames_lib=min_frames_lib,
                max_frames_lib=max_frames_lib,
                full_output=False,
                verbose=verbose,
                **algo_opt_copy,
            )
        # pad again now
        res = np.pad(res_tmp, pad, mode="constant", constant_values=0)

    elif algo == pca:
        scale_list = algo_opt_copy.pop("scale_list", None)
        ifs_collapse_range = algo_opt_copy.pop("ifs_collapse_range", "all")
        mask_rdi = algo_opt_copy.pop("mask_rdi", None)
        delta_rot = algo_opt_copy.pop("delta_rot", delta_rot)
        source_xy = algo_opt_copy.pop("source_xy", None)
        res = pca(
            cube=cube,
            angle_list=angs,
            cube_ref=cube_ref,
            scale_list=scale_list,
            ncomp=ncomp,
            svd_mode=svd_mode,
            scaling=scaling,
            delta_rot=delta_rot,
            source_xy=source_xy,
            fwhm=fwhm,
            imlib=imlib,
            interpolation=interpolation,
            collapse=collapse,
            collapse_ifs=collapse_ifs,
            ifs_collapse_range=ifs_collapse_range,
            nproc=nproc,
            weights=weights,
            mask_rdi=mask_rdi,
            verbose=verbose,
            **algo_opt_copy,
        )
    else:
        res = algo(cube=cube, angle_list=angs, **algo_options)

    indices = disk((posy, posx), radius=aperture_radius * fwhm)
    yy, xx = indices

    # also consider indices of the annulus for pca_annulus
    if algo == pca_annulus:
        fr_size = res.shape[-1]
        inner_rad = r_guess - annulus_width / 2
        yy_a, xx_a = get_annulus_segments(
            (fr_size, fr_size), inner_rad, annulus_width, nsegm=1
        )[0]
        # only consider overlapping indices
        yy_f = []
        xx_f = []
        for i in range(len(yy)):
            ind_y = np.where(yy_a == yy[i])
            for j in ind_y[0]:
                if xx[i] == xx_a[j]:
                    yy_f.append(yy[i])
                    xx_f.append(xx[i])
        yy = np.array(yy_f, dtype=int)
        xx = np.array(xx_f, dtype=int)

    if collapse is None:
        values = res[:, yy, xx].ravel()
    else:
        values = res[yy, xx].ravel()

    if full_output and collapse is not None:
        return values, res
    else:
        return values


def get_mu_and_sigma(cube, angs, ncomp, annulus_width, aperture_radius, fwhm,
                     r_guess, theta_guess, f_guess=None, psfn=None,
                     cube_ref=None, wedge=None, svd_mode="lapack", scaling=None,
                     algo=pca_annulus, delta_rot=1, imlib="vip-fft",
                     interpolation="lanczos4", collapse="median", weights=None,
                     algo_options={}, bin_spec=False, verbose=False):
    """Extract the mean and standard deviation of pixel intensities in an\
    annulus of the PCA-ADI image obtained with 'algo', in the part of a defined\
    wedge that is not overlapping with PA_pl+-delta_PA.

    Parameters
    ----------
    cube: numpy.array
        The cube of fits images expressed as a numpy.array.
    angs : 1d numpy ndarray
        Vector of derotation angles to align North up in your cube images.
    ncomp: int or None
        The number of principal components for PCA-based algorithms.
    annulus_width: float
        The width in pixels of the annulus on which the PCA is performed.
    aperture_radius: float
        The radius in fwhm of the circular aperture.
    fwhm: float
        Value of the FWHM of the PSF.
    r_guess: float
        The radial position of the center of the circular aperture. This
        parameter is NOT the radial position of the candidate associated to the
        Markov chain, but should be the fixed initial guess.
    theta_guess: float
        The angular position of the center of the circular aperture. This
        parameter is NOT the angular position of the candidate associated to the
        Markov chain, but should be the fixed initial guess.
    f_guess: float or 1d numpy array, optional
        The flux estimate for the companion.
    psfn: 2D or 3D numpy ndarray, optional
        Normalized psf used to remove the companion if f_guess is provided.
    cube_ref : numpy ndarray, 3d, optional
        Reference library cube. For Reference Star Differential Imaging.
    wedge: tuple, opt
        Range in theta where the mean and standard deviation are computed in an
        annulus defined in the PCA image. If None, it will be calculated
        automatically based on initial guess and derotation angles to avoid.
        If some disc signal is present elsewhere in the annulus, it is
        recommended to provide wedge manually. The provided range should be
        continuous and >0. E.g. provide (270, 370) to consider a PA range
        between [-90,+10].
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
    algo: python routine, opt {pca_annulus, pca_annular, pca, custom}
        Routine to be used to model and subtract the stellar PSF. From an input
        cube, derotation angles, and optional arguments, it should return a
        post-processed frame.
    delta_rot: float, optional
        If algo is set to pca_annular, delta_rot is the angular threshold used
        to select frames in the PCA library (see description of pca_annular).
    imlib : str, optional
        See the documentation of the ``vip_hci.preproc.frame_rotate`` function.
    interpolation : str, optional
        See the documentation of the ``vip_hci.preproc.frame_rotate`` function.
    collapse : {'median', 'mean', 'sum', 'trimmean', None}, str or None, opt
        Sets the way of collapsing the frames for producing a final image. If
        None then the cube of residuals is returned.
    weights: 1d numpy array or list, optional
        Weights to be applied for a weighted mean. Need to be provided if
        collapse mode is 'wmean'.
    algo_options: dict, opt
        Dictionary with additional parameters related to the algorithm
        (e.g. tol, min_frames_lib, max_frames_lib). If 'algo' is not a vip
        routine, this dict should contain all necessary arguments apart from
        the cube and derotation angles. Note: arguments such as ncomp, svd_mode,
        scaling, imlib, interpolation or collapse can also be included in this
        dict (the latter are also kept as function arguments for consistency
        with older versions of vip).
    bin_spec: bool, optional
        [only used if cube is 4D] Whether to collapse the spectral dimension
        (i.e. estimate a single binned flux) instead of estimating the flux in
        each spectral channel.

    Returns
    -------
    values: numpy.array
        The pixel values in the circular aperture after the PCA process.

    """
    if f_guess is not None and psfn is not None:
        if np.isscalar(f_guess):
            planet_parameter = (r_guess, theta_guess, f_guess)
        elif len(f_guess) == 1:
            planet_parameter = (r_guess, theta_guess, f_guess[0])
        else:
            r_all = [r_guess]*len(f_guess)
            theta_all = [r_guess]*len(f_guess)
            planet_parameter = np.array([r_all, theta_all, f_guess])
        array = cube_planet_free(planet_parameter, cube, angs, psfn,
                                 imlib=imlib, interpolation=interpolation,
                                 transmission=None)
    else:
        if verbose:
            msg = "WARNING: f_guess not provided. The companion will not be "
            msg += "removed from the cube before estimating mu and sigma. "
            msg += "A wedge will be used"
            print(msg)
        array = cube.copy()

    centy_fr, centx_fr = frame_center(array[0])
    halfw = max(aperture_radius * fwhm, annulus_width / 2)

    # Checking annulus/aperture sizes. Assuming square frames
    msg = "The annulus and/or the circular aperture used by the NegFC falls"
    msg += " outside the FOV. Try increasing the size of your frames or "
    msg += "decreasing the annulus or aperture size."
    msg += "rguess: {:.0f}px; centx_fr: {:.0f}px".format(r_guess, centx_fr)
    msg += "halfw: {:.0f}px".format(halfw)
    if r_guess > centx_fr - halfw:  # or r_guess <= halfw:
        raise RuntimeError(msg)

    # check if r_guess is less than fwhm
    if r_guess < fwhm:
        raise ValueError("r_guess should be greater than fwhm.")

    algo_opt_copy = algo_options.copy()
    ncomp = algo_opt_copy.pop("ncomp", ncomp)
    svd_mode = algo_opt_copy.pop("svd_mode", svd_mode)
    scaling = algo_opt_copy.pop("scaling", scaling)
    imlib = algo_opt_copy.pop("imlib", imlib)
    interpolation = algo_opt_copy.pop("interpolation", interpolation)
    collapse = algo_opt_copy.pop("collapse", collapse)

    radius_int = max(int(np.floor(r_guess - annulus_width / 2)), 0)
    radius_int = algo_opt_copy.pop("radius_int", radius_int)

    # not recommended, except if large-scale residual sky present (NIRC2-L')
    hp_filter = algo_opt_copy.pop("hp_filter", None)
    hp_kernel = algo_opt_copy.pop("hp_kernel", None)
    if hp_filter is not None:
        if "median" in hp_filter:
            array = cube_filter_highpass(array, mode=hp_filter,
                                         median_size=hp_kernel)
        elif "gauss" in hp_filter:
            array = cube_filter_highpass(array, mode=hp_filter,
                                         fwhm_size=hp_kernel)
        else:
            array = cube_filter_highpass(array, mode=hp_filter,
                                         kernel_size=hp_kernel)

    if algo == pca_annulus:
        pca_res = pca_annulus(
            array,
            angs,
            ncomp,
            annulus_width,
            r_guess,
            cube_ref,
            svd_mode,
            scaling,
            imlib=imlib,
            interpolation=interpolation,
            collapse=collapse,
            weights=weights,
            **algo_opt_copy,
        )
        if f_guess is not None and psfn is not None:
            pca_res_inv = pca_annulus(
                array,
                -angs,
                ncomp,
                annulus_width,
                r_guess,
                cube_ref,
                svd_mode,
                scaling,
                imlib=imlib,
                interpolation=interpolation,
                collapse=collapse,
                weights=weights,
                **algo_opt_copy,
            )

    elif algo == pca_annular or algo == nmf_annular:
        tol = algo_opt_copy.pop("tol", 1e-1)
        min_frames_lib = algo_opt_copy.pop("min_frames_lib", 2)
        max_frames_lib = algo_opt_copy.pop("max_frames_lib", 200)
        radius_int = max(1, int(np.floor(r_guess - annulus_width / 2)))
        radius_int = algo_opt_copy.pop("radius_int", radius_int)
        asize = algo_opt_copy.pop("asize", annulus_width)
        delta_rot = algo_opt_copy.pop("delta_rot", delta_rot)
        _ = algo_opt_copy.pop("verbose", verbose)
        # crop cube to just be larger than annulus => FASTER PCA
        crop_sz = int(2 * np.ceil(radius_int + asize + 1))
        if not crop_sz % 2:
            crop_sz += 1
        if crop_sz < cube.shape[-2] and crop_sz < cube.shape[-1]:
            pad = int((cube.shape[-2] - crop_sz) / 2)
            crop_cube = cube_crop_frames(cube, crop_sz, verbose=False)
        else:
            crop_cube = cube
            pad = 0
        if algo == pca_annular:
            res_tmp = algo(
                cube=crop_cube,
                angle_list=angs,
                cube_ref=cube_ref,
                radius_int=radius_int,
                fwhm=fwhm,
                asize=asize,
                delta_rot=delta_rot,
                ncomp=ncomp,
                svd_mode=svd_mode,
                scaling=scaling,
                imlib=imlib,
                interpolation=interpolation,
                collapse=collapse,
                weights=weights,
                tol=tol,
                min_frames_lib=min_frames_lib,
                max_frames_lib=max_frames_lib,
                full_output=False,
                verbose=False,
                **algo_opt_copy,
            )
        else:
            res_tmp = algo(
                cube=crop_cube,
                angle_list=angs,
                cube_ref=cube_ref,
                radius_int=radius_int,
                fwhm=fwhm,
                asize=annulus_width,
                delta_rot=delta_rot,
                ncomp=ncomp,
                scaling=scaling,
                imlib=imlib,
                interpolation=interpolation,
                collapse=collapse,
                weights=weights,
                min_frames_lib=min_frames_lib,
                max_frames_lib=max_frames_lib,
                full_output=False,
                verbose=False,
                **algo_opt_copy,
            )
        # pad again now
        pca_res = np.pad(res_tmp, pad, mode="constant", constant_values=0)

        if f_guess is not None and psfn is not None:
            pca_res_tinv = pca_annular(
                cube=crop_cube,
                angle_list=-angs,
                radius_int=radius_int,
                fwhm=fwhm,
                asize=annulus_width,
                delta_rot=delta_rot,
                ncomp=ncomp,
                svd_mode=svd_mode,
                scaling=scaling,
                imlib=imlib,
                interpolation=interpolation,
                collapse=collapse,
                tol=tol,
                min_frames_lib=min_frames_lib,
                max_frames_lib=max_frames_lib,
                full_output=False,
                verbose=False,
                weights=weights,
                **algo_opt_copy,
                )
            pca_res_inv = np.pad(pca_res_tinv, pad, mode="constant",
                                 constant_values=0)

    elif algo == pca:
        scale_list = algo_opt_copy.pop("scale_list", None)
        ifs_collapse_range = algo_opt_copy.pop("ifs_collapse_range", "all")
        nproc = algo_opt_copy.pop("nproc", 1)
        source_xy = algo_opt_copy.pop("source_xy", None)

        pca_res = pca(
            cube=array,
            angle_list=angs,
            cube_ref=cube_ref,
            scale_list=scale_list,
            ncomp=ncomp,
            svd_mode=svd_mode,
            scaling=scaling,
            delta_rot=delta_rot,
            source_xy=source_xy,
            imlib=imlib,
            interpolation=interpolation,
            collapse=collapse,
            ifs_collapse_range=ifs_collapse_range,
            nproc=nproc,
            weights=weights,
            verbose=False,
            **algo_opt_copy,
        )
        if f_guess is not None and psfn is not None:
            pca_res_inv = pca(
                cube=array,
                angle_list=-angs,
                cube_ref=cube_ref,
                scale_list=scale_list,
                ncomp=ncomp,
                svd_mode=svd_mode,
                scaling=scaling,
                delta_rot=delta_rot,
                source_xy=source_xy,
                imlib=imlib,
                interpolation=interpolation,
                collapse=collapse,
                ifs_collapse_range=ifs_collapse_range,
                nproc=nproc,
                weights=weights,
                verbose=False,
                **algo_opt_copy,
            )

    else:
        algo_args = algo_options
        pca_res = algo(cube=array, angle_list=angs, **algo_args)
        if f_guess is not None and psfn is not None:
            pca_res_inv = algo(cube=array, angle_list=-angs, **algo_args)

    if f_guess is not None and psfn is not None:
        if wedge is None:
            wedge = (0, 360)
        else:
            wedge = wedge
    elif wedge is None:
        delta_theta = np.amax(angs) - np.amin(angs)
        if delta_theta > 120:
            delta_theta = 120  # if too much rotation, be less conservative

        theta_ini = (theta_guess + delta_theta) % 360
        theta_fin = theta_ini + (360 - 2 * delta_theta)
        wedge = (theta_ini, theta_fin)
    if wedge is not None:
        if len(wedge) == 2:
            if wedge[0] > wedge[1]:
                msg = "2nd value of wedge smaller than first one => +360"
                print(msg)
                wedge = (wedge[0], wedge[1] + 360)
        else:
            raise TypeError("Wedge should have exactly 2 values")

    # annulus to estimate mu & sigma should encompass the companion location
    indices = get_annular_wedge(pca_res, inner_radius=radius_int,
                                width=min(annulus_width, 2 * fwhm),
                                wedge=wedge)
    yy, xx = indices
    if f_guess is not None and psfn is not None:
        indices_inv = get_annular_wedge(pca_res_inv,
                                        inner_radius=radius_int,
                                        width=min(annulus_width, 2 * fwhm))
        yyi, xxi = indices_inv
        all_res = np.concatenate((pca_res[yy, xx], pca_res_inv[yyi, xxi]))
        npx = len(yy) + len(yyi)
    else:
        all_res = pca_res[yy, xx]
        npx = len(yy)
    mu = np.nanmean(all_res)
    all_res -= mu
    area = np.pi * (fwhm / 2) ** 2
    ddof = min(int(npx * (1.0 - (1.0 / area))), npx - 1)
    sigma = np.nanstd(all_res, ddof=ddof)

    return mu, sigma


def hessian(array):
    """
    Calculate the Hessian matrix with finite differences for any input array.

    Parameters
    ----------
       array : numpy ndarray
           Input array for which the Hessian matrix should be calculated.

    Returns
    -------
       hessian: numpy ndarray of shape (array.ndim, array.ndim) + array.shape
           The Hessian matrix associated to each element of the input array,
           e.g. for a 2D input, hessian[i, j, k, l] corresponds to the second
           derivative x_ij (ij can be y or x) at coordinates (k,l) of input
           array.
    """
    grad = np.gradient(array)
    hessian = np.empty((array.ndim, array.ndim) + array.shape,
                       dtype=array.dtype)
    for k, grad_k in enumerate(grad):
        # iterate over dimensions
        # apply gradient again to every component of the first derivative.
        tmp_grad = np.gradient(grad_k)
        for m, grad_km in enumerate(tmp_grad):
            hessian[k, m, :, :] = grad_km
    return hessian
