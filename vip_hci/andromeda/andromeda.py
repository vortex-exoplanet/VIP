"""
Implementation of the ANDROMEDA algorithm from [MUG09]_ / [CANT15]_

.. [MUG09]
   | Mugnier et al, 2009
   | **Optimal method for exoplanet detection by angular differential imaging**
   | *J. Opt. Soc. Am. A, 26(6), 1326–1334*
   | `doi:10.1364/JOSAA.26.001326 <http://doi.org/10.1364/JOSAA.26.001326>`_

.. [CANT15]
   | Cantalloube et al, 2015
   | **Direct exoplanet detection and characterization using the ANDROMEDA
     method: Performance on VLT/NaCo data**
   | *A&A, 582*
   | `doi:10.1051/0004-6361/201425571 <http://doi.org/10.1051/0004-6361/20142557
     1>`_, `arXiv:1508.06406 <http://arxiv.org/abs/1508.06406>`_

"""

from __future__ import division, print_function
from __future__ import absolute_import

__author__ = "Ralf Farkas"
__all__ = ["andromeda"]


import numpy as np

from ..var.filters import frame_filter_highpass, cube_filter_highpass
from ..conf.utils_conf import pool_map, fixed

from .utils import robust_std, create_distance_matrix, idl_round, idl_where
from .shift import calc_psf_shift_subpix
from .fit import fitaffine


global CUBE


def andromeda(cube, oversampling_fact, angles, psf, filtering_fraction=.25,
              min_sep=.5, annuli_width=1., roa=2., opt_method='lsq',
              nsmooth_snr=18, iwa=1., precision=50, homogeneous_variance=True,
              multiply_gamma=True, nproc=1,
              verbose=False):
    """ Exoplanet detection in ADI sequences by maximum-likelihood approach.

    Parameters
    ----------
    cube : 3d array_like
        Input cube.
        IDL parameter: ``IMAGES_1_INPUT``
    oversampling_fact : float
        Oversampling factor for the wavelength corresponding to the filter used
        for obtaining ``cube`` (defined as the ratio between the wavelength of
        the filter and the Shannon wavelength).
        IDL parameter: ``OVERSAMPLING_1_INPUT``
    angles : array_like
        List of parallactic angles associated with each frame in ``cube``.
        IDL parameter: ``ANGLES_INPUT``
    psf : 2d array_like
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
        IDL parameter: ``RATIO_OPT_AREA_INPUT``
    opt_method : {'no', 'total', 'lsq', 'robust'}, optional
        Method used to balance for the flux difference that exists between the
        two subtracted annuli in an optimal way during ADI.
        IDL parameter: ``OPT_METHOD_ANG_INPUT``
    nsmooth_snr : int, optional
        Number of pixels over which the radial robust standard deviation profile
        of the SNR map is smoothed to provide a global trend for the SNR map
        normalization. For ``nsmooth_snr=0`` the SNR map normalization is
        disabled, and the positivity constraint is applied when calculating the
        flux.
        IDL parameter: ``NSMOOTH_SNR_INPUT``
    iwa : float, optional
        Inner working angle / inner radius of the first annulus taken into
        account, expressed in $\lambda/D$.
        IDL parameter: ``IWA_INPUT``
    precision : int, optional
        Number of shifts applied to the PSF. Passed to
        ``calc_psf_shift_subpix`` , which then creates a 4D cube with shape
        (precision+1, precision+1, N, N).
        IDL parameter: ``PRECISION_INPUT``
    homogeneous_variance : bool, optional
        If set, variance is treated as homogeneous and is calculated as a mean
        of variance in each position through time.
        IDL parameter: ``HOMOGENEOUS_VARIANCE_INPUT``
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
    flux : 2d ndarray
        Calculated flux map.
        (IDL return value)
    snr : 2d ndarray
        Signal to noise ratio map (defined as the estimated flux divided by the
        estimated standard deviation of the flux).
        IDL parameter: ``SNR_OUTPUT``
    likelihood : 2d ndarray
        likelihood
        IDL parameter: ``LIKELIHOOD_OUTPUT``
    stdflux : 2d ndarray
        Map of the estimated standard deviation of the flux.
        IDL parameter: ``STDEVFLUX_OUTPUT``
    ext_radius : float
        Edge of the SNR map. Slightly decreased due to the normalization
        procedure. Useful to a posteriori reject potential companions that are
        too close to the edge to be analyzed.
        IDL parameter: ``EXT_RADIUS_OUTPUT``

    References
    ----------
    .. [MUG09]
       | Mugnier et al, 2009
       | **Optimal method for exoplanet detection by angular differential
         imaging**
       | *J. Opt. Soc. Am. A, 26(6), 1326–1334*
       | `doi:10.1364/JOSAA.26.001326 <http://doi.org/10.1364/JOSAA.26.001326>`_

    .. [CANT15]
       | Cantalloube et al, 2015
       | **Direct exoplanet detection and characterization using the ANDROMEDA
         method: Performance on VLT/NaCo data**
       | *A&A, 582*
       | `doi:10.1051/0004-6361/201425571
         <http://doi.org/10.1051/0004-6361/201425571>`_,
         `arXiv:1508.06406 <http://arxiv.org/abs/1508.06406>`_

    Notes
    -----
    Based on ANDROMEDA v2.2.

    The following IDL parameters were not implemented:
        - ROTOFF_INPUT
        - recentering (should be done in VIP before):
            - COORD_CENTRE_1_INPUT
            - COORD_CENTRE_2_INPUT
        - debug/expert testing testing
            - INDEX_NEG_INPUT
            - INDEX_POS_INPUT
            - ANNULI_LIMITS_INPUT
            - MASK_INPUT
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
        - SDI (IMAGES_2_INPUT, ...)
        - variances (VARIANCE_1_INPUT, ...)

    The ``HIGHPASS_CUBE`` function in IDL ANDROMEDA has a parameter ``MEDIAN``,
    which is true by default and which does two tings:

    1. The ``nan`` values in the cube are replaced by their surrounding 5x5px
       median (hardcoded!)
    2. The 5px border of each frame is replaced by the median of the 6th pixel
       row/column; e.g. for the left border:

        .. code:: idl

            image[0:5,*] = median(image[6,*])

        (This may be due to how IDL ``MEDIAN`` handles 2D images: The median is
        only applied to the *inner* region, the outer border of the image is
        untouched.)

    That functionality was removed alltogether, because:

    - We assume that the cube is already cleaned from ``nan`` values.
    - The hardcoded value of 5px is not general for all sorts of cubes processed
      in VIP
    - In the IDL version, the "border smoothening" seems to be off by one pixel,
      which results in discontinuities in the cube. Replicating the exact IDL
      ANDROMEDA behaviour would mean to repliate that bug too.

    """
    global CUBE  # assigned after high-pass filter

    angles = -angles  # VIP convention

    #===== verify input

    frames, npix, _ = cube.shape
    npixpsf, _ = psf.shape

    if npix%2 == 1:
        raise ValueError("The side of the images must be an even number, with "
                         "the star centered on the intersection between the "
                         "four central pixels.")

    if filtering_fraction < 0 or filtering_fraction > 1:
        raise ValueError("``filtering_fraction`` must be between 0 and 1")

    if roa < 1:
        raise ValueError("The optimization to subtraction area ``roa`` "
                         "must be >= 1")

    #==== setup

    positivity = False # TODO
    if nsmooth_snr == 0:
        # positivity = 1 # TODO should be the case based on the docs. but
        positivity = False
        # also note the comment in andromeda.pro:691:
        #   ;If post-normalisation, set to 0, else 1
    elif nsmooth_snr < 2:
        raise ValueError("`nsmooth_snr` must be >= 2")

    #===== initialize output

    flux = np.zeros_like(cube[0])
    snr = np.zeros_like(cube[0])
    likelihood = np.zeros_like(cube[0])
    stdflux = np.zeros_like(cube[0])

    #===== pre-processing

    # normalization...
    psf = psf / np.sum(psf)  # creates new array in memory

    # ...and spatial filterin on the PSF:
    if filtering_fraction != 1:
        psf = frame_filter_highpass(psf, "hann", hann_cutoff=filtering_fraction)

    # library of all different PSF positions
    psf_cube = calc_psf_shift_subpix(psf, precision=precision)

    # spatial filtering of the preprocessed image-cubes:
    if filtering_fraction != 1:
        if verbose:
            print("Pre-processing filtering of the images and the PSF: "
                  "done! F={}".format(filtering_fraction))
        cube = cube_filter_highpass(cube, mode="hann",
                                    hann_cutoff=filtering_fraction,
                                    verbose=False)

    CUBE = cube

    # definition of the width of each annuli 
    dmin = iwa  # in lambda/D
    dmax = (npix/2 - npixpsf/2) / (2*oversampling_fact)  # in lambda/D
    distarray_lambdaonD = dmin + np.arange(int((dmax-dmin/annuli_width + 1)
                                                * annuli_width))
    annuli_limits = oversampling_fact * 2 * distarray_lambdaonD  # in pixels
    annuli_number = len(annuli_limits) - 1

    #===== main loop
    res_all = pool_map(nproc, _process_annulus,
                       # start with outer annuli, they take longer:
                       fixed(range(annuli_number)[::-1]),
                       annuli_limits, roa, min_sep, oversampling_fact,
                       angles, opt_method, multiply_gamma, psf_cube,
                       homogeneous_variance, verbose, msg="annulus",
                       leave=False, verbose=False)

    for res in res_all:
        flux += res[0]
        snr += res[1]
        likelihood += res[2]
        stdflux += res[3]

    # post-processing of the output
    if nsmooth_snr != 0:
        if verbose:
            print("Normalizing SNR...")
        
        # normalize
        dmin = np.ceil(annuli_limits[0]).astype(int)
        dmax = np.ceil(annuli_limits[-2]).astype(int)  # TODO: skip last annuli?
        snr_norm, snr_std = normalize_snr(snr, nsmooth_snr=nsmooth_snr,
                                          dmin=dmin, dmax=dmax)

        # normalization of the standard deviation of the flux
        stdflux_norm = np.zeros((npix, npix))
        zone = snr_std != 0
        stdflux_norm[zone] = stdflux[zone] * snr_std[zone]

        ext_radius = (np.floor(annuli_limits[annuli_number-2]) /
                      (2*oversampling_fact)) # TODO same
        return flux, snr_norm, likelihood, stdflux_norm, ext_radius
    else:
        ext_radius = (np.floor(annuli_limits[annuli_number-1]) /
                      (2*oversampling_fact))
        return flux, snr, likelihood, stdflux, ext_radius


def _process_annulus(i, annuli_limits, roa, min_sep, oversampling_fact, angles,
          opt_method, multiply_gamma, psf_cube,
          homogeneous_variance, verbose=False):
    global CUBE

    rhomin = annuli_limits[i]
    rhomax = annuli_limits[i+1]  # -> 
    rhomax_opt = np.sqrt(roa*rhomax**2 - (roa-1)*rhomin**2)

    # compute indices from min_sep
    if verbose:
        print("  Pairing frames...")
    min_sep_pix = min_sep * oversampling_fact*2
    angmin = 2*np.arcsin(min_sep_pix/(2*rhomin))*180/np.pi
    index_neg, index_pos, indices_not_used = create_indices(angles, angmin)

    if len(indices_not_used) != 0:
        if verbose:
            print("  WARNING: {} frame(s) cannot be used because it wasn't "
              "possible to find any other frame to couple with them. "
              "Their indices are: {}".format(len(indices_not_used),
                                             indices_not_used))
        max_sep_pix = 2*rhomin*np.sin(np.deg2rad((max(angles) - 
                                                  min(angles))/4))
        max_sep_ld = max_sep_pix/(2*oversampling_fact)

        if verbose:
            print("  For all frames to be used in this annulus, the minimum"
              " separation must be set at most to {} *lambda/D "
              "(corresponding to {} pixels).".format(max_sep_ld,
                                                     max_sep_pix))

    #===== angular differences
    if verbose:
        print("  Performing angular difference...")

    res = diff_images(cube_pos=CUBE[index_pos], cube_neg=CUBE[index_neg],
                       rint=rhomin, rext=rhomax_opt,
                       opt_method=opt_method)
    cube_diff, gamma, gamma_prime = res

    if not multiply_gamma:
        # reset gamma & gamma_prime to 1 (they were returned by diff_images)
        gamma = np.ones_like(gamma)
        gamma_prime = np.ones_like(gamma_prime)
    # TODO: gamma

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
    res = andromeda_core(cube=cube_diff, index_neg=index_neg,
                         index_pos=index_pos, angles=angles,
                         psf_cube=psf_cube,
                         homogeneous_variance=homogeneous_variance,
                         rhomin=rhomin, rhomax=rhomax, gamma=gamma,
                         gamma_prime=gamma_prime, verbose=verbose)
    return res


def andromeda_core(cube, index_neg, index_pos, angles, psf_cube, rhomin, rhomax,
                   gamma=None, gamma_prime=None, homogeneous_variance=True,
                   positivity=False, verbose=False):
    """
    Parameters
    ----------
    cube
        IDL parameter: ``DIFF_IMAGES_INPUT``
    index_neg
    index_pos
    angles
        IDL parameter: ``ANGLES_INPUT``
    psf_cube
        IDL parameter: ``PSFCUBE_INPUT``
    rhomin
        IDL parameter: ``RHOMIN_INPUT``
    rhomax : float
        is ceiled for the pixel-for-loop.
        IDL parameter: ``RHOMAX_INPUT``
    gamma, gamma_prime
        IDL parameter: GAMMA_INPUT
    homogeneous_variance: bool, optional
        IDL parameter: ``HOMOGENEOUS_VARIANCE_INPUT``
    positivity : bool, optional
        postivity constraint. If set, assures that the numerator is >=0 when
        calculating ``flux = numerator / denominator``
        IDL parameter: ``POSITIVITY_INPUT``
    verbose : bool, optional
        print more.


    Returns
    -------
    flux
        IDL return value
    snr
        IDL parameter: ``SNR_OUTPUT``
    likelihood
        IDL parameter: ``LIKELIHOOD_OUTPUT``
    stdflux
        IDL parameter: ``STDEVFLUX_OUTPUT``


    Notes
    -----
    - kmax renamed to npairs
    - not implemented: GOOD_PIXELS_INPUT, MASK_INPUT, WITHOUT_GAMMA_INPUT,
      PATTERN_OUTPUT, WEIGHTS_INPUT


    """
    npairs, npix, _ = cube.shape
    npixpsf = psf_cube.shape[2]  # shape: (p+1, p+1, x, y)
    precision = psf_cube.shape[0]-1

    #===== verify + sanitize input
    if npix%2 == 1:
        raise ValueError("size of the cube is odd!")
    if npixpsf%2 == 1:
        raise ValueError("PSF has odd pixel size!")

    if gamma is None:
        if verbose:
            print("\tANDROMEDA_CORE: The scaling factor is not taken into "
                  "account to build the model!")

    # calculate variance
    variance_diff_2d = (cube**2).sum(0)/npairs - (cube.sum(0)/npairs)**2

    # calculate weights from variance
    if homogeneous_variance:
        varmean = np.mean(variance_diff_2d) # idlwrap.mean
        weights_diff_2d = np.zeros((npix, npix)) + 1/varmean
        if verbose:
            print("\tANDROMEDA_CORE: Variance is considered homogeneous, mean"
                  " {:.3f}".format(varmean))
    else:
        weights_diff_2d = ((variance_diff_2d > 0) /
                           (variance_diff_2d + (variance_diff_2d == 0)))
        if verbose:
            print("\tANDROMEDA_CORE: Variance is taken equal to the empirical"
                  " variance in each pixel (inhomogeneous, but constant in time")

    weighted_diff_images = cube * weights_diff_2d

    # create annuli
    d = create_distance_matrix(npix)
    select_pixels = ((d > rhomin) & (d < rhomax))

    if verbose:
        print("\tANDROMEDA_CORE: working with {} differential images, radius "
              "{} to {}".format(npairs, rhomin, rhomax))

    # definition of the expected pattern (if a planet is present)
    numerator = np.zeros((npix, npix))
    denominator = np.ones((npix, npix))

    parang = np.array([angles[index_neg], angles[index_pos]])*np.pi/180
        # shape (2,npairs) -> array([[1, 2, 3],
        #                             [4, 5, 6]])   (for npairs=3)
        # IDL: dimension = SIZE =  _, npairs,2, _, _

    for j in range(npix//2 - np.ceil(rhomax).astype(int),
                   npix//2 + np.ceil(rhomax).astype(int)):
        for i in range(npix//2 - np.ceil(rhomax).astype(int),
                       npix//2 + np.ceil(rhomax).astype(int)): # same ranges!
            # IDL: scans in different direction!
            if select_pixels[j,i]:
                x0 = i - (npix/2 - 0.5) # distance to center of rotation, in x
                y0 = j - (npix/2 - 0.5) # distance to center of rotation, in y

                decalx = x0 * np.cos(parang) - y0 * np.sin(parang)  # (2,npairs)
                decaly = y0 * np.cos(parang) + x0 * np.sin(parang)  # (2,npairs)

                subp_x = idl_round((decalx - np.floor(decalx).astype(int)) *
                                   precision).astype(int)  # (2,npairs)
                subp_y = idl_round((decaly - np.floor(decaly).astype(int)) *
                                   precision).astype(int)  # (2,npairs)

                # compute, for each k and for both positive and negative indices
                # the coordinates of the squares in which the psf will be placed
                # lef, bot, ... have shape (2,npairs)
                lef = npix//2 + np.floor(decalx).astype(int) - npixpsf//2 
                bot = npix//2 + np.floor(decaly).astype(int) - npixpsf//2
                rig = npix//2 + np.floor(decalx).astype(int) + npixpsf//2 - 1
                top = npix//2 + np.floor(decaly).astype(int) + npixpsf//2 - 1

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

                    patt_pos = np.zeros((px_ymax[k]-px_ymin[k]+1,
                                         px_xmax[k]-px_xmin[k]+1))
                    patt_neg = np.zeros((px_ymax[k]-px_ymin[k]+1,
                                         px_xmax[k]-px_xmin[k]+1))

                    # put the positive psf in the right place
                    patt_pos[bot[1,k]-px_ymin[k] : bot[1,k]-px_ymin[k]+npixpsf,
                             lef[1,k]-px_xmin[k] : lef[1,k]-px_xmin[k]+npixpsf
                             ] = psf_cube[subp_y[1,k], subp_x[1,k]]
                             #TODO: should add a +1 somewhere??

                    # same for the negative psf, with a multiplication by gamma!
                    patt_neg[bot[0,k]-px_ymin[k] : bot[0,k]-px_ymin[k]+npixpsf,
                             lef[0,k]-px_xmin[k] : lef[0,k]-px_xmin[k]+npixpsf
                             ] = psf_cube[subp_y[0,k], subp_x[0,k]]
                             #TODO: should add a +1 somewhere??

                    # subtraction between the two
                    if gamma is None:
                        pattern_cut = patt_pos - patt_neg
                    else:
                        pattern_cut = patt_pos - patt_neg * gamma[k]

                    # compare current (2D) map of small rectangle of weights:
                    weight_cut = weights_diff_2d[px_ymin[k]:px_ymax[k]+1,
                                                 px_xmin[k]:px_xmax[k]+1]
                                                 # TODO (y,x) is ugly!

                    num_part += np.sum(pattern_cut * weighted_diff_images[k,
                                                 px_ymin[k]:px_ymax[k]+1,
                                                 px_xmin[k]:px_xmax[k]+1])
                    # TODO this only work when indexing like wdi[k,y,x]
                    #   (shape 30,31), and NOT with [k,x,y]!!! -> fix

                    den_part += np.sum(pattern_cut**2  * weight_cut)


                numerator[j,i] = num_part
                denominator[j,i] = den_part

    if positivity:  # TODO
        numerator = np.minimum(numerator, 0)

    # computation of estimated flux for current assumed planet position:
    flux = numerator / denominator

    # computation of snr map:
    snr = numerator / np.sqrt(denominator)

    # computation of likelihood map:
    likelihood = 0.5 * snr**2

    # computation of the standard deviation on the estimated flux
    stdflux = flux / (snr + (snr==0))  # TODO: 0 values are replaced by 1, but
                                       # small values like 0.1 are kept. Is this
                                       # the right approach?

    return flux, snr, likelihood, stdflux


def create_indices(angles, angmin):
    """
    Given a monotonic array of ``angles``, this function computes and returns
    the couples of indices of the array for which the separation is the closest
    to the value ``angmin``, by using the highest possible number of angles, all
    if possible.


    Parameters
    ----------
    angles : 1d array_like
        ndarray containing the angles associated to each image. The array should
        be monotonic
    angmin : float
        The minimum acceptable difference between two angles of a couple.

    Returns
    -------
    indices_neg, indices_neg : ndarrays
        The couples of indices, so that ``index_pos[0]`` should be paired with
        ``index_neg[0]`` and so on.
    indices_not_used : list
        The list of the frames which were not used. This list should preferably
        be empty.

    Notes
    -----
    - ``WASTE`` flag removed, instead this function returns `indices_not_used`
       -> warning message moved from this function to ``andromeda()``

    """

    # make array monotonic -> increasing
    if angles[-1] < angles[0]:
        angles = -angles
    

    good_angles = idl_where(angles-angles[0] >= angmin)
    
    if len(good_angles) == 0:
        raise RuntimeError("Impossible to find any couple of angles! Try to "
                           "reduce the IWA first, else you need to reduce the "
                           "minimum separation.")

    indices_neg = [0]
    indices_pos = [good_angles[0]]
    indices_not_used = []

    for i in range(1, len(angles)):
        good_angles = idl_where((angles-angles[i] >= angmin))

        if len(good_angles) > 0:
            indices_neg.append(i)
            indices_pos.append(good_angles[0])
        else: # search in other direction
            if i not in indices_pos:
                good_angles_back = idl_where((angles[i]-angles >= angmin))

                if len(good_angles_back) > 0:
                    indices_neg.append(i)
                    indices_pos.append(good_angles_back[-1])
                else:
                    # no new couple found
                    indices_not_used.append(i)

    return np.array(indices_neg), np.array(indices_pos), indices_not_used


def diff_images(cube_pos, cube_neg, rint, rext, opt_method="lsq",
                 variance_pos=None, variance_neg=None,
                 verbose=False):
    """

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

        ``no`` / ``1``
           corresponds to ``diff_images = i1 - gamma*i2`` and
           ``gamma = gamma_prime = 0``
        ``total`` / ``2``
           total ratio optimization. ``diff_images = i1 - gamma*i2`` and
           ``gamma = sum(i1*i2 / sum(i2**2))``, ``gamma_prime = 0``
        ``lsq`` / ``3``
           least-squares optimization. ``diff_images = i1 - gamma*i2``,
           ``gamma = sum(i1*i2)/sum(i2**2)``, ``gamma_prime = 0``
        ``l1`` / ``4``
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

    Notes
    -----
    - ``GN_NO`` and ``GAIN`` keywords were never used in the IDL version, so
      they were not implemented.
    - VARIANCE_POS_INPUT, VARIANCE_NEG_INPUT, VARIANCE_TOT_OUTPUT,
      WEIGHTS_OUTPUT were removed


    """

    nimg, npix, _ = cube_pos.shape

    # initialize
    cube_diff = np.zeros_like(cube_pos)
    gamma = np.zeros(nimg)  # linear factor
    gamma_prime = np.zeros(nimg)  # affine factor. Only !=0 for 'l1' affine fit

    distarray = create_distance_matrix(npix)
    annulus = (distarray > rint) & (distarray <= rext)  # 2d True/False map

    # compute normalization factors
    if opt_method in ["no", 1]:
        # no renormalization 
        print("    DIFF_IMAGES: no optimisation is being performed. Note that "
              "keywords rint and rext will be ignored.")
        gamma += 1
    else:

        if verbose:
            print("    DIFF_IMAGES: optimization annulus limits: {:.1f} -> "
                  "{:.1f}".format(rint, rext))

        for i in range(nimg):
            if opt_method in ["total", 2]:
                gamma[i] = (np.sum(cube_pos[i][annulus]) /
                            np.sum(cube_neg[i][annulus]))
            elif opt_method in ["lsq", 3]:
                gamma[i] = (np.sum(cube_pos[i][annulus]*cube_neg[i][annulus]) /
                            np.sum(cube_neg[i][annulus]**2))
                if verbose:
                    print("    DIFF_IMAGES: Factor gamma_ls for difference #{}:"
                          " {}".format(i+1,gamma[i]))
            elif opt_method in ["l1", 4]: # L1-affine optimization
                ann_pos = cube_pos[i][annulus]
                ann_neg = cube_neg[i][annulus]
                gamma[i], gamma_prime[i] = fitaffine(y=ann_pos, x=ann_neg)
                if verbose:
                    print("    DIFF_IMAGES: Factor gamma and gamma_prime for "
                          "difference #{}/{}: {}, {}".format(i+1, nimg,
                                                             gamma[i],
                                                             gamma_prime[i]))
            else:
                raise ValueError("opt_method '{}' unknown".format(opt_method))

    if verbose:
        print("    DIFF_IMAGES: median gamma={:.3f}, median gamma_prime={:.3f}"
              "".format(np.median(gamma), np.median(gamma_prime)))

    # compute image differences
    for i in range(nimg):
        cube_diff[i] = cube_pos[i] - cube_neg[i]*gamma[i] - gamma_prime[i]

    return cube_diff, gamma, gamma_prime


def normalize_snr(snr, dmin, dmax, nsmooth_snr=0):
    """
    Normalize each pixel of the SNR map by the robust standard deviation of the
    annulus to which the pixel belongs.  The aim is to get rid of the decreasing
    trend from the center of the image to its edge in order to obtain a SNR map
    of mean 0 and of variance 1 as expected by the algorithm if the noise model
    (white) was right. Thanks to this operation, a constant threshold can be 
    applied on the SNR map to perform automatic detection.

    Parameters
    ----------
    snr : 2d-array
        Square image/SNR-map to be normalized by its own radial
        robust standard deviation.
    dmin : int [pixels]
        Radius of the smallest annulus processed by ANDROMEDA. Depends on the
        IWA.
    dmax : int [pixels]
        Radius of the widest annulus processed by ANDROMEDA. Depends on the size
        of the PSF and the annuli_width.
    nsmooth_snr : int, optional
        Number of pixels on which the robust sttdev
        radial profile is smoothed (e.g. if nsmooth_snr=8; the regarded
        annulus is smoothed w.r.t its four inner annulus and 
        its four outer adjacent annulus. 
        Must be >= 2!

    Returns
    -------
    snr_norm
        Normalized SNR map of mean 0 and variance 1.
    snr_std
        In order to calculate once for all the 
        2D map of the SNR radial robust standard deviation,
        this variable records it.

    Notes
    -----
    - in IDL ANDROMEDA, ``/FIT`` is disabled by default, so it was not (yet)
      implemented.

    """


    if nsmooth_snr < 2:
        raise ValueError("``nsmooth_snr`` must be >= 2")

    nsnr = snr.shape[0]


    it_nosmoo = [] # DBLARR(Nsnr/2.)
    it_robust = [] # DBLARR(Nsnr/2.)
    snr_norm = np.zeros_like(snr)  # output
    imaz_robust = np.zeros_like(snr)

    # build annulus
    tempo = create_distance_matrix(nsnr)

    # main calculations
    for i in range(dmin, dmax-nsmooth_snr+1):
        id1 = (tempo >= i) & (tempo <= i+nsmooth_snr)
        id2 = (tempo >= i-0.5) & (tempo <= i+0.5)
        it_nosmoo.append(robust_std(snr[id2]))
        it_robust.append(robust_std(snr[id1]))
        imaz_robust[id2] = it_robust[-1]
    
    # far zone:
    for i in range(dmax-nsmooth_snr+1, dmax-4+1):
        id1 = (tempo >= i) & (tempo <= dmax)
        id2 = (tempo >= i-0.5) & (tempo <= i+0.5)
        it_nosmoo.append(robust_std(snr[id2]))
        it_robust.append(robust_std(snr[id1]))
        imaz_robust[id2] = it_robust[-1]

    # normalizing the SNR by its radial std:
    zone = imaz_robust != 0
    snr_norm[zone] = snr[zone] / imaz_robust[zone]
    
    return snr_norm, imaz_robust

