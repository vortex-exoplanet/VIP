#! /usr/bin/env python
"""
Module with the function of merit definitions for the NEGFC optimization.
"""

__author__ = "Valentin Christiaens, O. Wertz, Carlos Alberto Gomez Gonzalez"
__all__ = ["chisquare_fd"]

import numpy as np
from hciplot import plot_frames
from skimage.draw import disk
from .utils_negfd import cube_disk_free
from ..var import frame_center, get_annulus_segments
from ..psfsub import pca_annulus, pca_annular, nmf_annular, pca
from ..preproc import cube_crop_frames


def chisquare_fd(modelParameters, cube, angs, disk_img, mask_fm, initialState,
                 force_pos=False, fmerit="sum", mu_sigma=None, psfn=None,
                 algo=pca, algo_options={}, imlib='skimage',
                 interpolation='biquintic', transmission=None, weights=None,
                 debug=False, **rot_options):
    r"""
    Calculate the reduced :math:`\chi^2`:
    .. math:: \chi^2_r = \frac{1}{N-Npar}\sum_{j=1}^{N} \frac{(I_j-\mu)^2}{\sigma^2}
    (mu_sigma is a tuple) or:
    .. math:: \chi^2_r = \frac{1}{N-Npar}\sum_{j=1}^{N} |I_j| (mu_sigma=None),
    where N is the number of pixels within the binary mask mask_fm, Npar the 
    number of parameters to be fitted (4 for a 3D input cube, 3+n_ch for a 4D 
    input cube), and :math:`I_j` the j-th pixel intensity.

    Parameters
    ----------
    modelParameters: tuple
        The model parameters: (r, theta, flux) for a 3D input cube, or
        (r, theta, f1, ..., fN) for a 4D cube with N spectral channels.
    cube: 3d or 4d numpy ndarray
        Input ADI or ADI+IFS cube.
    angs: numpy.array
        The parallactic angle fits image expressed as a numpy.array.
    psfs_norm: numpy.array
        The scaled psf expressed as a numpy.array.
    fwhm : float
        The FHWM in pixels.
    annulus_width: int, optional
        The width in pixels of the annulus on which the PCA is done.
    aperture_radius: int, optional
        The radius of the circular aperture in terms of the FWHM.
    initialState: numpy.array
        xy shift and rotation of the disk model image.
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
    fmerit : {'sum', 'stddev'}, string optional
        Chooses the figure of merit to be used. stddev works better for close in
        companions sitting on top of speckle noise.
    collapse : {'median', 'mean', 'sum', 'trimmean', None}, str or None, optional
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
    debug: bool, opt
        Whether to debug and plot the post-processed frame after injection of
        the negative fake companion.

    Returns
    -------
    out: float
        The reduced chi squared.

    """
    if cube.ndim == 3:
        if force_pos:
            x, y, theta, _ = initialState
            flux_tmp = modelParameters[0]
        else:
            try:
                x, y, theta, flux_tmp = modelParameters
            except TypeError:
                msg = "modelParameters must be a tuple, {} was given"
                print(msg.format(type(modelParameters)))
        df_params = x, y, theta, flux_tmp
    else:
        if force_pos:
            x, y, theta, _ = initialState
            flux_tmp = np.array(modelParameters)
        else:
            try:
                x = modelParameters[0]
                y = modelParameters[1]
                theta = modelParameters[2]
                flux_tmp = np.array(modelParameters[3:])
            except TypeError:
                msg = "modelParameters must be a tuple, {} was given"
                print(msg.format(type(modelParameters)))
        df_params = x, y, theta, flux_tmp

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

    #norm_weights = None
    if weights is None:
        flux = -flux_tmp
        #norm_weights=weights
    elif np.isscalar(flux_tmp):
        flux = -flux_tmp * weights
        #norm_weights=weights
        #norm_weights = weights/np.sum(weights)
    else:
        flux = -np.outer(flux_tmp, weights)
        #norm_weights=weights
        #norm_weights = weights/np.sum(weights)

    # Create the cube with the negative fake companion injected
    cube_negfd = cube_disk_free(modelParameters, cube, angs, disk_img, psfn=None,
                                imlib=imlib, interpolation=interpolation,
                                imlib_sh=imlib_sh, 
                                interpolation_sh=interpolation_sh,
                                transmission=transmission, weights=weights, 
                                **rot_options)

    # post-process the empty cube
    res = algo(cube=cube_negfd, angle_list=angs, **algo_options)
    values = res[np.where(mask_fm)]

    # # Perform PCA and extract the zone of interest
    # res = get_values_optimize_fd(cube_negfd,
    #     angs,
    #     ncomp,
    #     annulus_width,
    #     aperture_radius,
    #     fwhm,
    #     initialState[0],
    #     initialState[1],
    #     cube_ref=cube_ref,
    #     svd_mode=svd_mode,
    #     scaling=scaling,
    #     algo=algo,
    #     delta_rot=delta_rot,
    #     collapse=collapse,
    #     algo_options=algo_options,
    #     weights=norm_weights,
    #     imlib=imlib_rot,
    #     interpolation=interpolation,
    #     debug=debug,
    # )

    # if debug and collapse is not None:
    #     values, frpca = res
    #     plot_frames(frpca)
    # else:
    #     values = res

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
            chi = np.std(values) * values.size / ddf  # TODO: test std**2
        else:
            raise RuntimeError("fmerit choice not recognized.")
    else:
        # true expression of a gaussian log probability
        mu = mu_sigma[0]
        sigma = mu_sigma[1]
        chi = np.sum(np.power(mu - values, 2) / sigma**2) / ddf

    return chi


# def get_values_optimize_fd(
#     cube,
#     angs,
#     ncomp,
#     annulus_width,
#     aperture_radius,
#     fwhm,
#     r_guess,
#     theta_guess,
#     cube_ref=None,
#     svd_mode="lapack",
#     scaling=None,
#     algo=pca_annulus,
#     delta_rot=1,
#     imlib="vip-fft",
#     interpolation="lanczos4",
#     collapse="median",
#     algo_options={},
#     weights=None,
#     debug=False,
# ):
#     """Extracts a processed frame from the cube and returns the intensity values 
#     of the pixels included in the binary mask.

#     Parameters
#     ----------
#     cube: 3d or 4d numpy ndarray
#         Input ADI or ADI+IFS cube.
#     angs: numpy.array
#         The parallactic angle fits image expressed as a numpy.array.
#     ncomp: int or None
#         The number of principal components for PCA-based algorithms.
#     annulus_width: float
#         The width in pixels of the annulus on which the PCA is performed.
#     aperture_radius: float
#         The radius in fwhm of the circular aperture.
#     fwhm: float
#         Value of the FWHM of the PSF.
#     r_guess: float
#         The radial position of the center of the circular aperture. This
#         parameter is NOT the radial position of the candidate associated to the
#         Markov chain, but should be the fixed initial guess.
#     theta_guess: float
#         The angular position of the center of the circular aperture. This
#         parameter is NOT the angular position of the candidate associated to the
#         Markov chain, but should be the fixed initial guess.
#     cube_ref : numpy ndarray, 3d, optional
#         Reference library cube. For Reference Star Differential Imaging.
#     svd_mode : {'lapack', 'randsvd', 'eigen', 'arpack'}, str optional
#         Switch for different ways of computing the SVD and selected PCs.
#     scaling : {None, "temp-mean", spat-mean", "temp-standard",
#         "spat-standard"}, None or str optional
#         Pixel-wise scaling mode using ``sklearn.preprocessing.scale``
#         function. If set to None, the input matrix is left untouched. Otherwise:

#         * ``temp-mean``: temporal px-wise mean is subtracted.

#         * ``spat-mean``: spatial mean is subtracted.

#         * ``temp-standard``: temporal mean centering plus scaling pixel values
#           to unit variance (temporally).

#         * ``spat-standard``: spatial mean centering plus scaling pixel values
#           to unit variance (spatially).

#         DISCLAIMER: Using ``temp-mean`` or ``temp-standard`` scaling can improve
#         the speckle subtraction for ASDI or (A)RDI reductions. Nonetheless, this
#         involves a sort of c-ADI preprocessing, which (i) can be dangerous for
#         datasets with low amount of rotation (strong self-subtraction), and (ii)
#         should probably be referred to as ARDI (i.e. not RDI stricto sensu).
#     algo: python routine, opt {pca_annulus, pca_annular, pca, custom}
#         Routine to be used to model and subtract the stellar PSF. From an input
#         cube, derotation angles, and optional arguments, it should return a
#         post-processed frame.
#     delta_rot: float, optional
#         If algo is set to pca_annular, delta_rot is the angular threshold used
#         to select frames in the PCA library (see description of pca_annular).
#     imlib : str, optional
#         See the documentation of the ``vip_hci.preproc.frame_rotate`` function.
#     interpolation : str, optional
#         See the documentation of the ``vip_hci.preproc.frame_rotate`` function.
#     collapse : {'median', 'mean', 'sum', 'trimmean', None}, str or None, optional
#         Sets the way of collapsing the frames for producing a final image. If
#         None then the cube of residuals is returned.
#     algo_options: dict, opt
#         Dictionary with additional parameters related to the algorithm
#         (e.g. tol, min_frames_lib, max_frames_lib). If 'algo' is not a vip
#         routine, this dict should contain all necessary arguments apart from
#         the cube and derotation angles. Note: arguments such as ncomp, svd_mode,
#         scaling, imlib, interpolation or collapse can also be included in this
#         dict (the latter are also kept as function arguments for consistency
#         with older versions of vip).
#     weights : 1d array, optional
#         If provided, the negative fake companion fluxes will be scaled according
#         to these weights before injection in the cube. Can reflect changes in
#         the observing conditions throughout the sequence.
#     debug: boolean
#         If True, the cube is returned along with the values.

#     Returns
#     -------
#     values: numpy.array
#         The pixel values in the circular aperture after the PCA process.

#     If debug is True and collapse non-None, the PCA frame is also returned.

#     """

#     res = algo(cube=cube, angle_list=angs, **algo_options)

#     return res[np.where(mask_fm)]
