#! /usr/bin/env python

"""
Module with sigma clipping functions.
"""


__author__ = 'Carlos Alberto Gomez Gonzalez', 'V. Christiaens'
__all__ = ['clip_array',
           'sigma_filter']

import numpy as np

import warnings
try:
    from numba import njit
    no_numba = False
except ImportError:
    msg = "Numba python bindings are missing."
    warnings.warn(msg, ImportWarning)
    no_numba = True


def sigma_filter(frame_tmp, bpix_map, neighbor_box=3, min_neighbors=3,
                 half_res_y=False, verbose=False):
    """Sigma filtering of pixels in a 2d array.

    Parameters
    ----------
    frame_tmp : numpy ndarray
        Input 2d array, image.
    bpix_map: numpy ndarray
        Input array of the same size as frame_tmp, indicating the locations of
        bad/nan pixels by 1 (the rest of the array is set to 0)
    neighbor_box : int, optional
        The side of the window around each pixel where the sigma and
        median are calculated. If half_res_y, this is the horizontal side (the
        vertical side will be twice smaller).
    min_neighbors : int, optional
        Minimum number of good neighboring pixels to be able to correct the
        bad/nan pixels
    half_res_y: bool, optional
        Whether the input data have every couple of 2 rows identical, i.e. there
        is twice less angular resolution vertically than horizontally (e.g.
        SINFONI data).
    verbose: bool, optional
        Whether to print more information while running.

    Returns
    -------
    frame_corr : numpy ndarray
        Output array with corrected bad/nan pixels

    """
    if not no_numba:
        @njit
        def _sigma_filter_numba(frame_tmp, bpix_map, neighbor_box=3,
                                min_neighbors=3, half_res_y=False,
                                verbose=False):
            if not frame_tmp.ndim == 2:
                raise TypeError('Input array is not a frame or 2d array')

            sz_y = frame_tmp.shape[0]  # get image y-dim
            sz_x = frame_tmp.shape[1]  # get image x-dim
            bp = bpix_map.copy()       # temporary bpix map; important to make a copy!
            im = frame_tmp             # corrected image
            nb = int(np.sum(bpix_map))  # number of bad pixels remaining
            nit = 0                                 # number of iterations
            while nb > 0:
                nit += 1
                wb = np.where(bp)                   # find bad pixels
                gp = 1 - bp                         # temporary good pixel map
                for n in range(nb):
                    # 0/ Determine the box around each pixel
                    half_box_x = int(np.floor(neighbor_box/2.))
                    if half_res_y:
                        half_box_y = max(1, int(half_box_x/2))
                    else:
                        half_box_y = half_box_x
                    # half size of the box at
                    hbox_b = min(half_box_y, wb[0][n])
                    # the bottom of the pixel
                    # half size of the box at
                    hbox_t = min(half_box_y, sz_y-1-wb[0][n])
                    # the top of the pixel
                    # half size of the box to
                    hbox_l = min(half_box_x, wb[1][n])
                    # the left of the pixel
                    # half size of the box to
                    hbox_r = min(half_box_x, sz_x-1-wb[1][n])
                    # the right of the pixel
                    # in case we are close to an edge, we want to extend the box
                    # in the direction opposite to the edge:
                    if hbox_b < hbox_t:
                        hbox_t += half_box_y-hbox_b
                    elif hbox_t < hbox_b:
                        hbox_b += half_box_y-hbox_t
                    if hbox_l < hbox_r:
                        hbox_r += half_box_x-hbox_l
                    elif hbox_r < hbox_l:
                        hbox_l += half_box_x-hbox_r
                    sgp = gp[(wb[0][n]-hbox_b):(wb[0][n]+hbox_t+1),
                             (wb[1][n]-hbox_l):(wb[1][n]+hbox_r+1)]
                    if int(np.sum(sgp)) >= min_neighbors:
                        sim = im[(wb[0][n]-hbox_b):(wb[0][n]+hbox_t+1),
                                 (wb[1][n]-hbox_l):(wb[1][n]+hbox_r+1)]
                        px_x = int(wb[0][n])
                        px_y = int(wb[1][n])
                        gsgp = np.where(sgp)
                        gsim = []
                        for i in range(len(gsgp[0])):
                            gsim.append(sim[gsgp[0][i], gsgp[1][i]])
                        im[px_x, px_y] = np.median(np.array(gsim))
                        bp[px_x, px_y] = 0
                nb = int(np.sum(bp))
            if verbose:
                print('Required number of iterations in the sigma filter: ', nit)

            return im

    # TODO: If possible, replace this function using
    # scipy.ndimage.filters.generic_filter and astropy.stats.sigma_clip
    def _sigma_filter(frame_tmp, bpix_map, neighbor_box=3, min_neighbors=3,
                      half_res_y=False, verbose=False):
        """Same description as wrapper function."""

        if frame_tmp.ndim != 2:
            raise TypeError('Input array is not a frame or 2d array')

        sz_y = frame_tmp.shape[0]  # get image y-dim
        sz_x = frame_tmp.shape[1]  # get image x-dim
        bp = bpix_map.copy()       # temporary bpix map; important to make a copy!
        im = frame_tmp             # corrected image
        nb = int(np.sum(bpix_map))  # number of bad pixels remaining
        # In each iteration, correct only the bpix with sufficient good
        # 'neighbors'
        nit = 0                                 # number of iterations
        while nb > 0:
            nit += 1
            wb = np.where(bp)                   # find bad pixels
            gp = 1 - bp                         # temporary good pixel map
            for n in range(nb):
                # 0/ Determine the box around each pixel
                half_box_x = int(np.floor(neighbor_box/2.))
                if half_res_y:
                    half_box_y = max(1, int(half_box_x/2))
                else:
                    half_box_y = half_box_x
                # half size of the box at the
                hbox_b = min(half_box_y, wb[0][n])
                # bottom of the pixel
                # half size of the box at the
                hbox_t = min(half_box_y, sz_y-1-wb[0][n])
                # top of the pixel
                # half size of the box to the
                hbox_l = min(half_box_x, wb[1][n])
                # left of the pixel
                # half size of the box to the
                hbox_r = min(half_box_x, sz_x-1-wb[1][n])
                # right of the pixel
                # but in case we are at an edge with min size box, we want to
                # extend the box by one row/column of pixels in the direction
                # opposite to the edge:
                if half_box_y == 1:
                    if wb[0][n] == sz_y-1:
                        hbox_b = hbox_b+1
                    elif wb[0][n] == 0:
                        hbox_t = hbox_t+1
                    if wb[1][n] == sz_x-1:
                        hbox_l = hbox_l+1
                    elif wb[1][n] == 0:
                        hbox_r = hbox_r+1

                sgp = gp[int(wb[0][n]-hbox_b): int(wb[0][n]+hbox_t+1),
                         int(wb[1][n]-hbox_l): int(wb[1][n]+hbox_r+1)]
                if int(np.sum(sgp)) >= min_neighbors:
                    sim = im[int(wb[0][n]-hbox_b): int(wb[0][n]+hbox_t+1),
                             int(wb[1][n]-hbox_l): int(wb[1][n]+hbox_r+1)]
                    im[wb[0][n], wb[1][n]] = np.median(sim[np.where(sgp)])
                    bp[wb[0][n], wb[1][n]] = 0
            nb = int(np.sum(bp))
        if verbose:
            print('Required number of iterations in the sigma filter: ', nit)
        return im

    if no_numba:
        return _sigma_filter(frame_tmp, bpix_map, neighbor_box=3,
                             min_neighbors=3, verbose=False)
    else:
        return _sigma_filter_numba(frame_tmp, bpix_map, neighbor_box=3,
                                   min_neighbors=3, verbose=False)


def clip_array(array, lower_sigma, upper_sigma, out_good=False, neighbor=False,
               num_neighbor=3, mad=False, half_res_y=False):
    """Sigma clipping for detecting outlying values in 2d array. If the
    parameter 'neighbor' is True the clipping can be performed in a local patch
    around each pixel, whose size depends on 'neighbor' parameter.

    Parameters
    ----------
    array : numpy ndarray
        Input 2d array, image.
    lower_sigma : float
        Value for sigma, lower boundary.
    upper_sigma : float
        Value for sigma, upper boundary.
    out_good : bool, optional
        For choosing different outputs.
    neighbor : bool, optional
        For clipping over the median of the contiguous pixels.
    num_neighbor : int, optional
        The side of the window around each pixel where the sigma and
        median are calculated. If half_res_y, this is the horizontal side (the
        vertical side will be twice smaller).
    mad : {False, True}, bool optional
        If True, the median absolute deviation will be used instead of the
        standard deviation.
    min_std : float, optional
        Min standard deviation considered for the lower and upper sigma
        conditions. Can be useful for frames with padded edges (e.g. SPHERE/IFS)
    half_res_y: bool, optional
        Whether the input data have every couple of 2 rows identical, i.e. there
        is twice less angular resolution vertically than horizontally (e.g.
        SINFONI data).

    Returns
    -------
    good : array_like
        If out_good argument is true, returns the indices of not-outlying px.
    bad : array_like
        If out_good argument is false, returns a vector with the outlier px.

    """

    def _clip_array(array, lower_sigma, upper_sigma, out_good=False,
                    neighbor=False, num_neighbor=3, mad=False,
                    half_res_y=False):
        """Sigma clipping for detecting outlying values in 2d array. If the
        parameter 'neighbor' is True the clipping can be performed in a local patch
        around each pixel, whose size depends on 'neighbor' parameter.

        Parameters
        ----------
        array : numpy ndarray
            Input 2d array, image.
        lower_sigma : float
            Value for sigma, lower boundary.
        upper_sigma : float
            Value for sigma, upper boundary.
        out_good : bool, optional
            For choosing different outputs.
        neighbor : bool optional
            For clipping over the median of the contiguous pixels.
        num_neighbor : int, optional
            The side of the window around each pixel where the sigma and
            median are calculated. If half_res_y, this is the horizontal side (the
            vertical side will be twice smaller).
        mad : bool, optional
            If True, the median absolute deviation will be used instead of the
            standard deviation.
        half_res_y: bool, optional
            Whether the input data have every couple of 2 rows identical, i.e. there
            is twice less angular resolution vertically than horizontally (e.g.
            SINFONI data).

        Returns
        -------
        good : numpy ndarray
            If out_good argument is true, returns the indices of not-outlying px.
        bad : numpy ndarray
            If out_good argument is false, returns a vector with the outlier px.

        """
        if array.ndim != 2:
            raise TypeError("Input array is not two dimensional (frame)\n")

        ny, nx = array.shape
        bpm = array.copy()
        gpm = array.copy()

        if neighbor and num_neighbor:
            for y in range(ny):
                for x in range(nx):
                    # 0/ Determine the box around each pixel
                    half_box_x = int(np.floor(num_neighbor/2.))
                    if half_res_y:
                        half_box_y = max(1, int(half_box_x/2))
                    else:
                        half_box_y = half_box_x
                    # half size of the box at the
                    hbox_b = min(half_box_y, y)
                    # bottom of the pixel
                    # half size of the box at the
                    hbox_t = min(half_box_y, ny-1-y)
                    # top of the pixel
                    # half size of the box to the
                    hbox_l = min(half_box_x, x)
                    # left of the pixel
                    # half size of the box to the
                    hbox_r = min(half_box_x, nx-1-x)
                    # right of the pixel
                    # in case we are close to an edge, we want to extend the box
                    # in the direction opposite to the edge:
                    if hbox_b < hbox_t:
                        hbox_t += half_box_y-hbox_b
                    elif hbox_t < hbox_b:
                        hbox_b += half_box_y-hbox_t
                    if hbox_l < hbox_r:
                        hbox_r += half_box_x-hbox_l
                    elif hbox_r < hbox_l:
                        hbox_l += half_box_x-hbox_r

                    sub_arr = array[y-hbox_b:y+hbox_t+1,
                                    x-hbox_l:x+hbox_r+1]
                    neighbours = sub_arr.flatten()
                    neigh_list = []
                    remove_itself = True
                    for i in range(neighbours.shape[0]):
                        if neighbours[i] == array[y, x] and remove_itself:
                            remove_itself = False
                        else:
                            neigh_list.append(neighbours[i])

                    neigh_arr = np.array(neigh_list)
                    median = np.median(neigh_arr)
                    if mad:
                        abs_diff = []
                        for i in range(len(neigh_list)):#num_neighbor*num_neighbor-1):
                            abs_diff.append(np.absolute(median-neigh_arr[i]))
                        sigma = np.median(np.array(abs_diff))
                    else:
                        sigma = np.std(neigh_arr)
                    bad1 = array[y, x] < (median - lower_sigma * sigma)
                    bad2 = array[y, x] > (median + upper_sigma * sigma)
                    bpm[y, x] = bad1 | bad2
                    gpm[y, x] = 1.-bpm[y, x]
        else:
            median = np.median(array)
            sigma = np.std(array)
            for y in range(ny):
                for x in range(nx):
                    bad1 = array[y, x] < (median - lower_sigma * sigma)
                    bad2 = array[y, x] > (median + upper_sigma * sigma)
                    bpm[y, x] = bad1 | bad2
                    gpm[y, x] = 1.-bpm[y, x]

        if out_good:
            good = np.where(gpm)
            return good
        else:
            bad = np.where(bpm)
            return bad

    if no_numba:
        return _clip_array(array, lower_sigma, upper_sigma, out_good=out_good,
                           neighbor=neighbor, num_neighbor=num_neighbor,
                           mad=mad, half_res_y=half_res_y)
    else:
        _clip_array_numba = njit(_clip_array)
        return _clip_array_numba(array, lower_sigma, upper_sigma,
                                 out_good=out_good, neighbor=neighbor,
                                 num_neighbor=num_neighbor, mad=mad,
                                 half_res_y=half_res_y)