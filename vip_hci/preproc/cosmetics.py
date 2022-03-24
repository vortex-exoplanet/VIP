#! /usr/bin/env python

"""
Module with cosmetics procedures. Contains the function for bad pixel fixing.
Also functions for cropping cubes.
"""



__author__ = 'Carlos Alberto Gomez Gonzalez, V. Christiaens'
__all__ = ['cube_crop_frames',
           'cube_drop_frames',
           'frame_crop',
           'frame_pad',
           'cube_correct_nan',
           'approx_stellar_position']


from multiprocessing import cpu_count
import numpy as np
from astropy.stats import sigma_clipped_stats
from ..config.utils_conf import pool_map, iterable
from ..stats import sigma_filter
from ..var import frame_center, get_square


def cube_crop_frames(array, size, xy=None, force=False, verbose=True,
                     full_output=False):
    """Crops frames in a cube (3d or 4d array).

    Parameters
    ----------
    array : numpy ndarray
        Input 3d or 4d array.
    size : int
        Size of the desired central sub-array in each frame, in pixels.
    xy : tuple of ints
        X, Y coordinates of new frame center. If you are getting the
        coordinates from ds9 subtract 1, python has 0-based indexing.
    force : bool, optional
        ``size`` and the original size of the frames must be both even or odd.
        With ``force`` set to True this condition can be avoided.
    verbose : bool optional
        If True message of completion is showed.
    full_output: bool optional
        If true, returns cenx and ceny in addition to array_view.

    Returns
    -------
    array_out : numpy ndarray
        Cube with cropped frames.

    """
    if array.ndim == 3:
        temp_fr = array[0]
    elif array.ndim == 4:
        temp_fr = array[0, 0]
    else:
        raise TypeError('`Array` is not a cube (3d or 4d numpy.ndarray)')

    if xy is not None:
        cenx, ceny = xy
    else:
        ceny, cenx = frame_center(temp_fr)
    _, y0, x0 = get_square(temp_fr, size, y=ceny, x=cenx, position=True,
                           force=force, verbose=verbose)

    if not force:
        if temp_fr.shape[0] % 2 == 0:
            if size % 2 != 0:
                size += 1
        else:
            if size % 2 == 0:
                size += 1
    y1 = int(y0 + size)
    x1 = int(x0 + size)

    if array.ndim == 3:
        array_out = array[:, y0:y1, x0:x1]
    elif array.ndim == 4:
        array_out = array[:, :, y0:y1, x0:x1]

    if verbose:
        print("New shape: {}".format(array_out.shape))

    if full_output:
        return array_out, cenx, ceny
    else:
        return array_out


def frame_crop(array, size, cenxy=None, force=False, verbose=True):
    """ Crops a square frame (2d array). Uses the ``get_square`` function.

    Parameters
    ----------
    array : numpy ndarray
        Input frame.
    size : int, odd
        Size of the subframe.
    cenxy : tuple, optional
        Coordinates of the center of the subframe.
    force : bool, optional
        Size and the size of the 2d array must be both even or odd. With
        ``force`` set to False, the requested size is flexible (i.e. +1 can be
        applied to requested crop size for its parity to match the input size).
        If ``force`` set to True, the requested crop size is enforced, even if
        parities do not match (warnings are raised!).
    verbose : bool optional
        If True, a message of completion is shown.

    Returns
    -------
    array_view : numpy ndarray
        Sub array.

    """
    if array.ndim != 2:
        raise TypeError('`Array` is not a frame or 2d array')

    if not cenxy:
        ceny, cenx = frame_center(array)
    else:
        cenx, ceny = cenxy
    array_view = get_square(array, size, ceny, cenx, force=force,
                            verbose=verbose)

    if verbose:
        print("New shape: {}".format(array_view.shape))
    return array_view


def frame_pad(array, fac, fillwith=0, loc=0, scale=1, full_output=False):
    """ Pads a frame (2d array) equally on each sides, where the final frame
    size is set by a multiplicative factor applied to the original size. The 
    padding is set by fillwith, which can be either a fixed value or white 
    noise, characterized by (loc, scale).
    Uses the ``get_square`` function.

    Parameters
    ----------
    array : numpy ndarray
        Input frame.
    fac : float > 1.
        Ratio of the size between padded and input frame.
    fillwith : float or str, optional
        If a float or np.nan: value used for padding.
        If str, must be 'noise', which will inject white noise, using loc and 
        scale parameters.
    loc : float, optional
        If padding noise, mean of the white noise.
    scale : float, optional
        If padding noise, standard deviation of the white noise.
    full_output : bool, optional
        Whether to also return the indices of input frame within the padded 
        frame (in addition to padded frame).

    Returns
    -------
    array_out : numpy ndarray
        Padded array.
    ori_indices: tuple
        [returned if full_output=True] Indices of the bottom left and top 
        right vertices of the original image within the padded array:
        (y0, yN, x0, xN).  

    """
    
    if not array.ndim==2:
        raise TypeError("The input array must be 2d")
    
    y,x = array.shape
    cy_ori, cx_ori = frame_center(array)
    new_y = int(y*fac)
    new_x = int(x*fac)
    if new_y%2 != y%2:
        new_y-=1
    if new_x%2 != x%2:
        new_x-=1
    if fillwith == 'noise':
        array_out = np.random.normal(loc=loc, scale=scale, size=(new_y,new_x))
    else:
        array_out = np.zeros([new_y,new_x],dtype=array.dtype)
        array_out[:] = fillwith
    cy, cx = frame_center(array_out)
    y0 = int(cy-cy_ori)
    y1 = int(cy+cy_ori)
    if new_y%2:
        y1+=1
    x0 = int(cx-cx_ori)
    x1 = int(cx+cx_ori)
    if new_x%2:
        x1+=1
    array_out[y0:y1,x0:x1] = array.copy()
    ori_indices =  (y0, y1, x0, x1)
    
    if full_output:
        return array_out, ori_indices
    else:
        return array_out


def cube_drop_frames(array, n, m, parallactic=None, verbose=True):
    """
    Slice the cube so that all frames between ``n``and ``m`` are kept.

    Operates on axis 0 for 3D cubes, and on axis 1 for 4D cubes. This returns a
    modified *copy* of ``array``. The indices ``n`` and ``m`` are included and
    1-based.


    Parameters
    ----------
    array : numpy ndarray
        Input cube, 3d or 4d.
    n : int
        1-based index of the first frame to be kept. Frames before this one are
        dropped.
    m : int
        1-based index of the last frame to be kept. Frames after this one are
        dropped.
    parallactic : 1d ndarray, optional
        parallactic angles vector. If provided, a modified copy of
        ``parallactic`` is returned additionally.

    Returns
    -------
    array_view : numpy ndarray
        Cube with new size.
    parallactic : 1d numpy ndarray
        [parallactic != None] New parallactic angles.

    """
    if m > array.shape[0]:
        raise TypeError('End index must be smaller than the # of frames')

    if array.ndim == 4:
        array_view = array[:, n-1:m, :, :].copy()
    elif array.ndim == 3:
        array_view = array[n-1:m, :, :].copy()
    else:
        raise ValueError("only 3D and 4D cubes are supported!")

    if parallactic is not None:
        if not parallactic.ndim == 1:
            raise ValueError('Parallactic angles vector has wrong shape')
        parallactic = parallactic[n-1:m]

    if verbose:
        print("Cube successfully sliced")
        print("New cube shape: {}".format(array_view.shape))
        if parallactic is not None:
            msg = "New parallactic angles vector shape: {}"
            print(msg.format(parallactic.shape))

    if parallactic is not None:
        return array_view, parallactic
    else:
        return array_view


def frame_remove_stripes(array):
    """ Removes unwanted stripe artifact in frames with non-perfect bias or sky
    subtraction. Encountered this case on an LBT data cube.
    """
    lines = array[:50]
    lines = np.vstack((lines, array[-50:]))
    mean = lines.mean(axis=0)
    for i in range(array.shape[1]):
        array[:,i] = array[:,i] - mean[i]
    return array


def cube_correct_nan(cube, neighbor_box=3, min_neighbors=3, verbose=False,
                     half_res_y=False, nproc=1):
    """Sigma filtering of nan pixels in a whole frame or cube. Tested on
    SINFONI data.

    Parameters
    ----------
    cube : cube_like
        Input 3d or 2d array.
    neighbor_box : int, optional
        The side of the square window around each pixel where the sigma and
        median are calculated for the nan pixel correction.
    min_neighbors : int, optional
        Minimum number of good neighboring pixels to be able to correct the
        bad/nan pixels.
    verbose: bool, optional
        Whether to print more information or not during processing
    half_res_y: bool, optional
        Whether the input data have every couple of 2 rows identical, i.e. there
        is twice less angular resolution vertically than horizontally (e.g.
        SINFONI data). The algorithm goes twice faster if this option is
        rightfully set to True.
    nproc: None or int, optional
        Number of CPUs for multiprocessing (only used for 3D input). If None, 
        will automatically set it to half the available number of CPUs.

    Returns
    -------
    obj_tmp : numpy ndarray
        Output cube with corrected nan pixels in each frame
    """

    obj_tmp = cube.copy()

    ndims = obj_tmp.ndim
    if ndims != 2 and ndims != 3:
        raise TypeError("Input object is not two or three dimensional")

    if neighbor_box < 3 or neighbor_box % 2 == 0:
        raise ValueError('neighbor_box should be an odd value greater than 3')
    max_neigh = sum(range(3, neighbor_box + 2, 2))
    if min_neighbors > max_neigh:
        min_neighbors = max_neigh
        msg = "Warning! min_neighbors was reduced to {} to avoid bugs."
        print(msg.format(max_neigh))

    if ndims == 2:
        obj_tmp, nnanpix = nan_corr_2d(obj_tmp)
        if verbose:
            print("{} NaN pixels were corrected".format(nnanpix))

    elif ndims == 3:
        if nproc is None:
            nproc = cpu_count()//2
        n_z = obj_tmp.shape[0]
        if nproc == 1:
            for zz in range(n_z):
                obj_tmp[zz], nnanpix = nan_corr_2d(obj_tmp[zz], neighbor_box,
                                                   min_neighbors, half_res_y, 
                                                   verbose, True)
                if verbose:
                    msg = "In channel {}, {} NaN pixels were corrected"
                    print(msg.format(zz, nnanpix))
        else:
            if verbose:
                msg = "Correcting NaNs in multiprocessing..."
                print(msg)
            res = pool_map(nproc, nan_corr_2d, iterable(obj_tmp), neighbor_box,
                           min_neighbors, half_res_y, verbose, False) 
            obj_tmp = np.array(res, dtype=object)

    if verbose:
        print('All nan pixels are corrected.')

    return obj_tmp


def nan_corr_2d(obj_tmp, neighbor_box, min_neighbors, half_res_y, verbose,
                full_output=True):
    
    n_x = obj_tmp.shape[1]
    n_y = obj_tmp.shape[0]

    if half_res_y:
        if n_y % 2 != 0:
            raise ValueError("The input frames do not have an even number "
                             "of rows. Hence, you should probably not be "
                             "using the option half_res_y = True.")
        n_y = int(n_y / 2)
        frame = obj_tmp
        obj_tmp = np.zeros([n_y, n_x])
        for yy in range(n_y):
            obj_tmp[yy] = frame[2 * yy]

    # tuple with the 2D indices of each nan value of the frame
    nan_indices = np.where(np.isnan(obj_tmp))
    nan_map = np.zeros_like(obj_tmp)
    nan_map[nan_indices] = 1
    nnanpix = int(np.sum(nan_map))
    # Correct nan with iterative sigma filter
    obj_tmp = sigma_filter(obj_tmp, nan_map, neighbor_box=neighbor_box,
                           min_neighbors=min_neighbors, verbose=verbose,
                           half_res_y=half_res_y)
    if half_res_y:
        frame = obj_tmp
        n_y = 2 * n_y
        obj_tmp = np.zeros([n_y, n_x])
        for yy in range(n_y):
            obj_tmp[yy] = frame[int(yy / 2)]

    if full_output:
        return obj_tmp, nnanpix
    else:
        return obj_tmp
    

def approx_stellar_position(cube, fwhm, return_test=False, verbose=False):
    """Finds the approximate coordinates of the star, assuming it is the 
    brightest signal in the images. The algorithm can handle images dominated 
    by noise, since outliers are corrected based on the position of ths star in 
    other channels.

    Parameters
    ----------
    obj_tmp : numpy ndarray
        Input 3d cube
    fwhm : float or array 1D
        Input full width half maximum value of the PSF for each channel.
        This will be used as the standard deviation for Gaussian kernel
        of the Gaussian filtering.
        If float, it is assumed the same for all channels.
    return_test: bool, optional
        Whether the test result vector (a bool vector with whether the star
        centroid could be find in the corresponding channel) should be returned
        as well, along with the approx stellar coordinates.
    verbose: bool, optional
        Chooses whether to print some additional information.

    Returns:
    --------
    Array of y and x approx coordinates of the star in each channel of the cube
    if return_test: it also returns the test result vector
    """
    from ..metrics import peak_coordinates

    obj_tmp = cube.copy()
    n_z = obj_tmp.shape[0]

    if isinstance(fwhm, float) or isinstance(fwhm, int):
        fwhm_scal = fwhm
        fwhm = np.zeros((n_z))
        fwhm[:] = fwhm_scal

    # 1/ Write a 2-columns array with indices of all max pixel values in the cube
    star_tmp_idx = np.zeros([n_z, 2])
    star_approx_idx = np.zeros([n_z, 2])
    test_result = np.ones(n_z)
    for zz in range(n_z):
        star_tmp_idx[zz] = peak_coordinates(obj_tmp[zz], fwhm[zz])

    # 2/ Detect the outliers in each column
    _, med_y, stddev_y = sigma_clipped_stats(star_tmp_idx[:, 0], sigma=2.5)
    _, med_x, stddev_x = sigma_clipped_stats(star_tmp_idx[:, 1], sigma=2.5)
    lim_inf_y = med_y - 3 * stddev_y
    lim_sup_y = med_y + 3 * stddev_y
    lim_inf_x = med_x - 3 * stddev_x
    lim_sup_x = med_x + 3 * stddev_x

    if verbose:
        print("median y of star - 3sigma = ", lim_inf_y)
        print("median y of star + 3sigma = ", lim_sup_y)
        print("median x of star - 3sigma = ", lim_inf_x)
        print("median x of star + 3sigma = ", lim_sup_x)

    for zz in range(n_z):
        if (
            (star_tmp_idx[zz, 0] < lim_inf_y) or
            (star_tmp_idx[zz, 0] > lim_sup_y) or
            (star_tmp_idx[zz, 1] < lim_inf_x) or
            (star_tmp_idx[zz, 1] > lim_sup_x)
        ):
            test_result[zz] = 0

    # 3/ Replace by the median of neighbouring good coordinates if need be
    for zz in range(n_z):
        if test_result[zz] == 0:
            ii = 1
            inf_neigh = max(0, zz - ii)
            sup_neigh = min(n_z - 1, zz + ii)
            while test_result[inf_neigh] == 0 and test_result[sup_neigh] == 0:
                ii = ii + 1
                inf_neigh = max(0, zz - ii)
                sup_neigh = min(n_z - 1, zz + ii)
            if test_result[inf_neigh] == 1 and test_result[sup_neigh] == 1:
                star_approx_idx[zz] = np.floor((star_tmp_idx[sup_neigh] + \
                                                star_tmp_idx[inf_neigh]) / 2)
            elif test_result[inf_neigh] == 1:
                star_approx_idx[zz] = star_tmp_idx[inf_neigh]
            else:
                star_approx_idx[zz] = star_tmp_idx[sup_neigh]
        else:
            star_approx_idx[zz] = star_tmp_idx[zz]

    if return_test:
        return star_approx_idx, test_result.astype(bool)
    else:
        return star_approx_idx
