#! /usr/bin/env python
"""Module with function of merit definitions for the NEGFD optimization."""

__author__ = "Valentin Christiaens, O. Wertz, Carlos Alberto Gomez Gonzalez"
__all__ = ["chisquare_fd"]

import numpy as np
from .negfd_interp import interpolate_model
from .utils_negfd import cube_disk_free
from ..psfsub import pca


def chisquare_fd(
    modelParameters,
    cube,
    angs,
    disk_model,
    mask_fm,
    initialState,
    force_params=None,
    grid_param_list=None,
    fmerit="sum",
    mu_sigma=None,
    psfn=None,
    algo=pca,
    algo_options={},
    interp_order=-1,
    imlib="skimage",
    interpolation="biquintic",
    transmission=None,
    weights=None,
    debug=False,
    rot_options={},
):
    r"""

    Calculate the figure of merit to minimze residuals after disk subtraction.

    The reduced :math:`\chi^2` is defined as::
    .. math:: \chi^2_r = \frac{1}{N-Npar}\sum_{j=1}^{N} \frac{(I_j-\mu)^2}{\sigma^2}
    (mu_sigma is a tuple) or:
    .. math:: \chi^2_r = \frac{1}{N-Npar}\sum_{j=1}^{N} |I_j| (mu_sigma=None),
    where N is the number of pixels within the binary mask mask_fm, Npar the
    number of parameters to be fitted (4 for a 3D input cube, 3+n_ch for a 4D
    input cube), and :math:`I_j` the j-th pixel intensity.

    Parameters
    ----------
    modelParameters: tuple
        The free model parameters. E.g. (x, y, theta, scal, flux) for a 3D input
        cube (if force_params=None) or (x, y, theta, scal, f1, ..., fN) for a 4D
        cube with N spectral channels (if force_params=None).
    cube: 3d or 4d numpy ndarray
        Input ADI or ADI+IFS cube.
    angs : 1d numpy ndarray
        Vector of derotation angles to align North up in your cube images.
    psfs_norm: numpy.array
        The scaled psf expressed as a numpy.array.
    fwhm : float
        The FHWM in pixels.
    annulus_width: int, optional
        The width in pixels of the annulus on which the PCA is done.
    aperture_radius: int, optional
        The radius of the circular aperture in terms of the FWHM.
    initialState: numpy.array
        All initial parameters (including fixed ones) applied to the disk model
        image.
    force_params: None or list/tuple of bool, optional
        If not None, list/tuple of bool corresponding to parameters to fix.
    grid_params_list: list of lists/1d nd arrays, or None
        If input disk_model is a grid of either images (for 3D input cube) or
        spectral cubes (for a 4D input cube), this should be provided. It should
        be a list of either lists or 1d nd arrays corresponding to the parameter
        values sampled by the input disk model grid, with their lengths matching
        the respective first dimensions of disk_model.
    ncomp: int or None
        The number of principal components for PCA-based algorithms.
    mu_sigma: tuple of 2 floats or None, opt
        If set to None: not used, and falls back to original version of the
        algorithm, using fmerit.
        If set to anything but None: will compute the mean and standard
        deviation of pixel intensities in an annulus centered on the location
        of the companion, excluding the area directly adjacent to the companion.
    fmerit : {'sum', 'stddev'}, string optional
        Chooses the figure of merit to be used. stddev works better for close in
        companions sitting on top of speckle noise.
    collapse : {'median', 'mean', 'sum', 'trimmean', None}, str or None, opt
        Sets the way of collapsing the frames for producing a final image. If
        None then the cube of residuals is used when measuring the function of
        merit (instead of a single final frame).
    algo: python routine, opt {pca_annulus, pca_annular, pca, custom}
        Routine to be used to model and subtract the stellar PSF. From an input
        cube, derotation angles, and optional arguments, it should return a
        post-processed frame.
    algo_options: dict, opt
        Dictionary with additional parameters related to the algorithm
        (e.g. tol, min_frames_lib, max_frames_lib). If 'algo' is not a vip
        routine, this dict should contain all necessary arguments apart from
        the cube and derotation angles. Note: arguments such as ncomp, svd_mode,
        scaling, imlib, interpolation or collapse can also be included in this
        dict (the latter are also kept as function arguments for consistency
        with older versions of vip).
    interp_order: int or tuple of int, optional, {-1,0,1}
        [only used if grid_params_list is not None] Interpolation mode for model
        interpolation. If a tuple of integers, the length should match the
        number of grid dimensions and will trigger a different interpolation
        mode for the different parameters.
            - -1: Order 1 spline interpolation in logspace for the parameter
            - 0: nearest neighbour model
            - 1: Order 1 spline interpolation

    imlib : str, optional
        See the documentation of the ``vip_hci.preproc.frame_shift`` function.
    interpolation : str, optional
        See the documentation of the ``vip_hci.preproc.frame_shift`` function.
    transmission: numpy array, optional
        Array with 2 columns. First column is the radial separation in pixels.
        Second column is the off-axis transmission (between 0 and 1) at the
        radial separation given in column 1.
    weights : 1d array, optional
        If provided, the negative fake companion fluxes will be scaled according
        to these weights before injection in the cube. Can reflect changes in
        the observing conditions throughout the sequence.
    debug: bool, opt
        Whether to debug and plot the post-processed frame after injection of
        the negative fake companion.

    Returns
    -------
    out: float
        The reduced chi squared.

    """
    grid_ndim = disk_model.ndim-cube.ndim+1

    if cube.ndim == 3:
        multispectral = False
        if force_params is not None:
            grid_params = []
            df_params = []
            c_free = 0
            c_forced = 0
            for i in range(len(force_params)):
                if force_params[i]:
                    if i < grid_ndim:
                        grid_params.append(initialState[c_forced])
                    else:
                        df_params.append(initialState[c_forced])
                    c_forced += 1
                else:
                    if i < grid_ndim:
                        grid_params.append(modelParameters[c_free])
                    else:
                        df_params.append(modelParameters[c_free])
                    c_free += 1
            x, y, theta, scal = tuple(df_params[:4])
            flux_tmp = df_params[-1]
        else:
            try:
                if grid_ndim > 0:
                    grid_params = modelParameters[:grid_ndim]
                x, y, theta, scal = modelParameters[grid_ndim:grid_ndim+4]
                flux_tmp = modelParameters[grid_ndim+4:]
            except TypeError:
                msg = "modelParameters must be a tuple, {} was given"
                print(msg.format(type(modelParameters)))
    else:
        multispectral = True
        if force_params is not None:
            flux_fix = force_params[grid_ndim+4]
            for j in range(len(force_params) - (5+grid_ndim)):
                if force_params[j+5+grid_ndim] != flux_fix:
                    msg = "All fluxes need to be either free or fixed"
                    raise ValueError(msg)
            grid_params = []
            df_params = []
            c_free = 0
            c_forced = 0
            for i in range(len(force_params[:4+grid_ndim])):
                if force_params[i]:
                    if i < grid_ndim:
                        grid_params.append(initialState[c_forced])
                    else:
                        df_params.append(initialState[c_forced])
                    c_forced += 1
                else:
                    if i < grid_ndim:
                        grid_params.append(modelParameters[c_free])
                    else:
                        df_params.append(modelParameters[c_free])
                    c_free += 1
            if flux_fix:
                flux_tmp = initialState[c_forced:]
            else:
                flux_tmp = modelParameters[c_free:]
            x, y, theta, scal = tuple(df_params)
        else:
            try:
                if grid_ndim > 0:
                    grid_params = modelParameters[:grid_ndim]
                x = modelParameters[grid_ndim+0]
                y = modelParameters[grid_ndim+1]
                theta = modelParameters[grid_ndim+2]
                flux_tmp = np.array(modelParameters[grid_ndim+3:])
            except TypeError:
                msg = "modelParameters must be a tuple, {} was given"
                print(msg.format(type(modelParameters)))

    # apply temporal weights, if any
    if weights is None:
        flux = flux_tmp
    elif np.isscalar(flux_tmp):
        flux = flux_tmp * weights
    else:
        flux = np.outer(flux_tmp, weights)

    df_params = x, y, theta, scal, flux

    # interpolate in the model grid, if any
    if grid_ndim > 0:
        grid_params = tuple(grid_params)
        # Return infinity if requested grid params outside original bounds.
        for p in range(len(grid_param_list)):
            if grid_params[p] < grid_param_list[p][0]:
                return np.inf
            elif grid_params[p] > grid_param_list[p][-1]:
                return np.inf
        # Otherwise Interpolate disk_img from the input grid.
        disk_img = interpolate_model(grid_params, grid_param_list, disk_model,
                                     multispectral=multispectral,
                                     interp_order=interp_order)
    else:
        disk_img = disk_model.copy()

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

    # Create the cube with the negative fake companion injected
    cube_negfd = cube_disk_free(
        df_params,
        cube,
        angs,
        disk_img,
        psfn=None,
        imlib=imlib_rot,
        interpolation=interpolation,
        imlib_sh=imlib_sh,
        interpolation_sh=interpolation,
        transmission=transmission,
        weights=weights,
        **rot_options
    )

    # post-process the empty cube
    res = algo(cube=cube_negfd, angle_list=angs, **algo_options)
    values = res[np.where(mask_fm)]

    # Function of merit
    # in case algo is run on part of the field (e.g. annulus), discard:
    values = values[values != 0]
    ddf = values.size - len(modelParameters)
    if ddf < 1:
        msg = "Not enough pixels at the intersection of input binary mask and "
        msg += "area where the algorithm is run. Check mask_fm and algo_params."
        raise ValueError(msg)
    elif values.size < 10:
        msg = "WARNING: less than 10 pixels in the optimization area ("
        msg += "intersection of input binary mask and where the algorithm is "
        msg += "run). You may want to double-check mask_fm and algo_params."
        print(msg)
    if mu_sigma is None:
        # old version - delete?
        if fmerit == "sum":
            chi = np.sum(np.abs(values)) / (values.size - len(modelParameters))
        elif fmerit == "stddev":
            chi = np.std(values) * values.size / ddf
        else:
            raise RuntimeError("fmerit choice not recognized.")
    else:
        # true expression of a gaussian log probability
        mu = mu_sigma[0]
        sigma = mu_sigma[1]
        # check format
        if isinstance(mu, np.ndarray):
            if mu.shape == cube.shape[-2:]:
                mu = mu[np.where(mask_fm)]
                mu = mu[values != 0]
            else:
                msg = "If input mu is an array, it should have same shape as "
                msg += "cube frames"
                raise TypeError(msg)
        if isinstance(sigma, np.ndarray):
            if sigma.shape == cube.shape[-2:]:
                sigma = sigma[np.where(mask_fm)]
                sigma = sigma[values != 0]
            else:
                msg = "If input sigma is an array, it should have same shape as"
                msg += " cube frames"
                raise TypeError(msg)

        chi = np.sum(np.power((mu - values)/sigma, 2)) / ddf

    return chi
