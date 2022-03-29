#! /usr/bin/env python

"""
Module containing the Local Low-rank plus Sparse plus Gaussian-noise 
decomposition algorithm (Gomez Gonzalez et al. 2016) for ADI data.
"""

__author__ = 'Carlos Alberto Gomez Gonzalez'
__all__ = ['llsg',
           'thresholding']


import numpy as np
from scipy.linalg import qr
from multiprocessing import cpu_count
from astropy.stats import median_absolute_deviation
from ..config import time_ini, timing
from ..preproc import cube_derotate, cube_collapse
from ..var import get_annulus_segments, cube_filter_highpass
from .svd import svd_wrapper, get_eigenvectors
from ..config.utils_conf import pool_map, iterable


def llsg(cube, angle_list, fwhm, rank=10, thresh=1, max_iter=10,
         low_rank_ref=False, low_rank_mode='svd', auto_rank_mode='noise',
         residuals_tol=1e-1, cevr=0.9, thresh_mode='soft', nproc=1, asize=None, 
         n_segments=4, azimuth_overlap=None, radius_int=None, random_seed=None, 
         high_pass=None, collapse='median', full_output=False, verbose=True,
         debug=False, **rot_options):
    """ Local Low-rank plus Sparse plus Gaussian-noise decomposition (LLSG) as
    described in Gomez Gonzalez et al. 2016. This first version of our algorithm
    aims at decomposing ADI cubes into three terms L+S+G (low-rank, sparse and
    Gaussian noise). Separating the noise from the S component (where the moving
    planet should stay) allow us to increase the SNR of potential planets.

    The three tunable parameters are the *rank* or expected rank of the L
    component, the ``thresh`` or threshold for encouraging sparsity in the S
    component and ``max_iter`` which sets the number of iterations. The rest of
    parameters can be tuned at the users own risk (do it if you know what you're
    doing).

    Parameters
    ----------
    cube : numpy ndarray, 3d
        Input ADI cube.
    angle_list : numpy ndarray, 1d
        Corresponding parallactic angle for each frame.
    fwhm : float
        Known size of the FHWM in pixels to be used.
    rank : int, optional
        Expected rank of the L component.
    thresh : float, optional
        Factor that scales the thresholding step in the algorithm.
    max_iter : int, optional
        Sets the number of iterations.
    low_rank_ref :
        If True the first estimation of the L component is obtained from the
        remaining segments in the same annulus.
    low_rank_mode : {'svd', 'brp'}, optional
        Sets the method of solving the L update.
    auto_rank_mode : {'noise', 'cevr'}, str optional
        If ``rank`` is None, then ``auto_rank_mode`` sets the way that the
        ``rank`` is determined: the noise minimization or the cumulative
        explained variance ratio (when 'svd' is used).
    residuals_tol : float, optional
        The value of the noise decay to be used when ``rank`` is None and
        ``auto_rank_mode`` is set to ``noise``.
    cevr : float, optional
        Float value in the range [0,1] for selecting the cumulative explained
        variance ratio to choose the rank automatically (if ``rank`` is None).
    thresh_mode : {'soft', 'hard'}, optional
        Sets the type of thresholding.
    nproc : None or int, optional
        Number of processes for parallel computing. If None the number of
        processes will be set to cpu_count()/2. By default the algorithm works
        in single-process mode.
    asize : int or None, optional
        If ``asize`` is None then each annulus will have a width of ``2*asize``.
        If an integer then it is the width in pixels of each annulus.
    n_segments : int or list of ints, optional
        The number of segments for each annulus. When a single integer is given
        it is used for all annuli.
    azimuth_overlap : int or None, optional
        Sets the amount of azimuthal averaging.
    radius_int : int, optional
        The radius of the innermost annulus. By default is 0, if >0 then the
        central circular area is discarded.
    random_seed : int or None, optional
        Controls the seed for the Pseudo Random Number generator.
    high_pass : odd int or None, optional
        If set to an odd integer <=7, a high-pass filter is applied to the
        frames. The ``vip_hci.var.frame_filter_highpass`` is applied twice,
        first with the mode ``median-subt`` and a large window, and then with
        ``laplacian-conv`` and a kernel size equal to ``high_pass``. 5 is an
        optimal value when ``fwhm`` is ~4.
    collapse : {'median', 'mean', 'sum', 'trimmean'}, str optional
        Sets the way of collapsing the frames for producing a final image.
    full_output: bool, optional
        Whether to return the final median combined image only or with other
        intermediate arrays.
    verbose : bool, optional
        If True prints to stdout intermediate info.
    debug : bool, optional
        Whether to output some intermediate information.
    rot_options: dictionary, optional
        Dictionary with optional keyword values for "imlib", "interpolation",
        "border_mode", "mask_val", "edge_blend", "interp_zeros", "ker" (see 
        documentation of ``vip_hci.preproc.frame_rotate``)

    Returns
    -------
    frame_s : numpy ndarray, 2d
        Final frame (from the S component) after rotation and median-combination.

    If ``full_output`` is True, the following intermediate arrays are returned:
    list_l_array_der, list_s_array_der, list_g_array_der, frame_l, frame_s,
    frame_g

    """
    if cube.ndim != 3:
        raise TypeError("Input array is not a cube (3d array)")
    if not cube.shape[0] == angle_list.shape[0]:
        msg = "Angle list vector has wrong length. It must equal the number"
        msg += " frames in the cube"
        raise TypeError(msg)

    if low_rank_mode == 'brp':
        if rank is None:
            msg = "Auto rank only works with SVD low_rank_mode."
            msg += " Set a value for the rank parameter"
            raise ValueError(msg)
        if low_rank_ref:
            msg = "Low_rank_ref only works with SVD low_rank_mode"
            raise ValueError(msg)

    global cube_init
    if high_pass is not None:
        cube_init = cube_filter_highpass(cube, 'median-subt', median_size=19,
                                         verbose=False)
        cube_init = cube_filter_highpass(cube_init, 'laplacian-conv',
                                         kernel_size=high_pass, verbose=False)
    else:
        cube_init = cube

    if verbose:
        start_time = time_ini()
    n, y, x = cube.shape

    if azimuth_overlap == 0:
        azimuth_overlap = None

    if radius_int is None:
        radius_int = 0

    if nproc is None:
        nproc = cpu_count() // 2        # Hyper-threading doubles the # of cores

    # Same number of pixels per annulus
    if asize is None:
        annulus_width = int(np.ceil(2 * fwhm))  # as in the paper
    elif isinstance(asize, int):
        annulus_width = asize
    n_annuli = int((y / 2 - radius_int) / annulus_width)
    # TODO: asize in pxs to be consistent with other functions

    if n_segments is None:
        n_segments = [4 for _ in range(n_annuli)]   # as in the paper
    elif isinstance(n_segments, int):
        n_segments = [n_segments]*n_annuli
    elif n_segments == 'auto':
        n_segments = []
        n_segments.append(2)    # for first annulus
        n_segments.append(3)    # for second annulus
        ld = 2 * np.tan(360/4/2) * annulus_width
        for i in range(2, n_annuli):    # rest of annuli
            radius = i * annulus_width
            ang = np.rad2deg(2 * np.arctan(ld / (2 * radius)))
            n_segments.append(int(np.ceil(360/ang)))

    if verbose:
        print('Annuli = {}'.format(n_annuli))

    # Azimuthal averaging of residuals
    if azimuth_overlap is None:
        azimuth_overlap = 360   # no overlapping, single config of segments
    n_rots = int(360 / azimuth_overlap)

    matrix_s = np.zeros((n_rots, n, y, x))
    if full_output:
        matrix_l = np.zeros((n_rots, n, y, x))
        matrix_g = np.zeros((n_rots, n, y, x))

    # Looping the he annuli
    if verbose:
        print('Processing annulus: ')
    for ann in range(n_annuli):
        inner_radius = radius_int + ann * annulus_width
        n_segments_ann = n_segments[ann]
        if verbose:
            print('{} : in_rad={}, n_segm={}'.format(ann+1, inner_radius,
                                                     n_segments_ann))

        # TODO: pool_map as in xloci function: build first a list
        for i in range(n_rots):
            theta_init = i * azimuth_overlap
            indices = get_annulus_segments(cube[0], inner_radius,
                                           annulus_width, n_segments_ann,
                                           theta_init)

            patches = pool_map(nproc, _decompose_patch, indices,
                               iterable(range(n_segments_ann)), n_segments_ann,
                               rank, low_rank_ref, low_rank_mode, thresh,
                               thresh_mode, max_iter, auto_rank_mode, cevr,
                               residuals_tol, random_seed, debug, full_output)

            for j in range(n_segments_ann):
                yy = indices[j][0]
                xx = indices[j][1]

                if full_output:
                    matrix_l[i, :, yy, xx] = patches[j][0]
                    matrix_s[i, :, yy, xx] = patches[j][1]
                    matrix_g[i, :, yy, xx] = patches[j][2]
                else:
                    matrix_s[i, :, yy, xx] = patches[j]

    if full_output:
        list_s_array_der = [cube_derotate(matrix_s[k], angle_list, nproc=nproc,
                                          **rot_options)
                            for k in range(n_rots)]
        list_frame_s = [cube_collapse(list_s_array_der[k], mode=collapse)
                        for k in range(n_rots)]
        frame_s = cube_collapse(np.array(list_frame_s), mode=collapse)

        list_l_array_der = [cube_derotate(matrix_l[k], angle_list, nproc=nproc,
                                          **rot_options)
                            for k in range(n_rots)]
        list_frame_l = [cube_collapse(list_l_array_der[k], mode=collapse)
                        for k in range(n_rots)]
        frame_l = cube_collapse(np.array(list_frame_l), mode=collapse)

        list_g_array_der = [cube_derotate(matrix_g[k], angle_list, nproc=nproc,
                                          **rot_options)
                            for k in range(n_rots)]
        list_frame_g = [cube_collapse(list_g_array_der[k], mode=collapse)
                        for k in range(n_rots)]
        frame_g = cube_collapse(np.array(list_frame_g), mode=collapse)

    else:
        list_s_array_der = [cube_derotate(matrix_s[k], angle_list, nproc=nproc,
                                          **rot_options)
                            for k in range(n_rots)]
        list_frame_s = [cube_collapse(list_s_array_der[k], mode=collapse)
                        for k in range(n_rots)]

        frame_s = cube_collapse(np.array(list_frame_s), mode=collapse)

    if verbose:
        print('')
        timing(start_time)

    if full_output:
        return(list_l_array_der, list_s_array_der, list_g_array_der,
               frame_l, frame_s, frame_g)
    else:
        return frame_s


def _decompose_patch(indices, i_patch, n_segments_ann, rank, low_rank_ref,
                     low_rank_mode, thresh, thresh_mode, max_iter,
                     auto_rank_mode, cevr, residuals_tol, random_seed,
                     debug=False, full_output=False):
    """ Patch decomposition.
    """
    j = i_patch
    yy = indices[j][0]
    xx = indices[j][1]
    data_segm = cube_init[:, yy, xx]

    if low_rank_ref:
        ref_segments = list(range(n_segments_ann))
        ref_segments.pop(j)
        for m, n in enumerate(ref_segments):
            if m == 0:
                yy_ref = indices[n][0]
                xx_ref = indices[n][1]
            else:
                yy_ref = np.hstack((yy_ref, indices[n][0]))
                xx_ref = np.hstack((xx_ref, indices[n][1]))
        data_ref = cube_init[:, yy_ref, xx_ref]
    else:
        data_ref = data_segm

    patch = _patch_rlrps(data_segm, data_ref, rank, low_rank_ref,
                         low_rank_mode, thresh, thresh_mode,
                         max_iter, auto_rank_mode, cevr,
                         residuals_tol, random_seed, debug=debug,
                         full_output=full_output)
    return patch


def _patch_rlrps(array, array_ref, rank, low_rank_ref, low_rank_mode,
                 thresh, thresh_mode, max_iter, auto_rank_mode='noise',
                 cevr=0.9, residuals_tol=1e-2, random_seed=None, debug=False,
                 full_output=False):
    """ Patch decomposition based on GoDec/SSGoDec (Zhou & Tao 2011)
    """
    ############################################################################
    # Initializing L and S
    ############################################################################
    L = array
    if low_rank_ref:
        L_ref = array_ref.T
    else:
        L_ref = None
    S = np.zeros_like(L)
    random_state = np.random.RandomState(random_seed)
    itr = 0
    power = 0
    svdlib = 'lapack'

    while itr <= max_iter:
        ########################################################################
        # Updating L
        ########################################################################
        if low_rank_mode == 'brp':
            Y2 = random_state.randn(L.shape[1], rank)
            for _ in range(power + 1):
                Y1 = np.dot(L, Y2)
                Y2 = np.dot(L.T, Y1)
            Q, _ = qr(Y2, mode='economic')
            Lnew = np.dot(np.dot(L, Q), Q.T)

        elif low_rank_mode == 'svd':
            if itr == 0:
                PC = get_eigenvectors(rank, L, svdlib, mode=auto_rank_mode,
                                      cevr=cevr, noise_error=residuals_tol,
                                      data_ref=L_ref, debug=debug,
                                      collapse=True, scaling='temp-standard')
                rank = PC.shape[0]  # so we can use the optimized rank
                if low_rank_ref:
                    Lnew = np.dot(np.dot(PC, L).T, PC).T
                else:
                    Lnew = np.dot(np.dot(L, PC.T), PC)
            else:
                rank_i = min(rank, min(L.shape[0], L.shape[1]))
                PC = svd_wrapper(L, svdlib, rank_i, False,
                                 random_state=random_state)
                Lnew = np.dot(np.dot(L, PC.T), PC)

        else:
            raise RuntimeError('Low Rank estimation mode not recognized.')

        ########################################################################
        # Updating S
        ########################################################################
        T = L - Lnew + S
        threshold = np.sqrt(median_absolute_deviation(T.ravel())) * thresh

        # threshold = np.sqrt(median_absolute_deviation(T, axis=0)) * thresh
        # threshmat = np.zeros_like(T)
        # for i in range(threshmat.shape[0]):
        #     threshmat[i] = threshold
        # threshold = threshmat

        if debug:
            print('threshold = {:.3f}'.format(threshold))
        S = thresholding(T, threshold, thresh_mode)

        T -= S
        L = Lnew + T
        itr += 1

    G = array - L - S

    L = L.T
    S = S.T
    G = G.T

    if full_output:
        return L, S, G
    else:
        return S
    
    
def thresholding(array, threshold, mode):
    """ Array thresholding strategies.
    """
    x = array.copy()
    if mode == 'soft':
        j = np.abs(x) <= threshold
        x[j] = 0
        k = np.abs(x) > threshold
        if isinstance(threshold, float):
            x[k] = x[k] - np.sign(x[k]) * threshold
        else:
            x[k] = x[k] - np.sign(x[k]) * threshold[k]
    elif mode == 'hard':
        j = np.abs(x) < threshold
        x[j] = 0
    elif mode == 'nng':
        j = np.abs(x) <= threshold
        x[j] = 0
        j = np.abs(x) > threshold
        x[j] = x[j] - threshold**2/x[j]
    elif mode == 'greater':
        j = x < threshold
        x[j] = 0
    elif mode == 'less':
        j = x > threshold
        x[j] = 0
    else:
        raise RuntimeError('Thresholding mode not recognized')
    return x


