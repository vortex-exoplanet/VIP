#! /usr/bin/env python

"""
Module with various functions to create shapes, annuli and segments.
"""

__author__ = 'Carlos Alberto Gomez Gonzalez'
__all__ = ['dist',
           'frame_center',
           'get_square',
           'get_circle',
           'get_ellipse',
           'get_annulus_segments',
           'get_ell_annulus',
           'mask_circle',
           'create_ringed_spider_mask',
           'matrix_scaling',
           'prepare_matrix',
           'reshape_matrix']

import numpy as np
from skimage.draw import polygon
from skimage.draw import circle
from sklearn.preprocessing import scale

from ..conf.utils_conf import frame_or_shape


def mask_circle(array, radius, fillwith=0, mode='in'):
    """
    Mask the pixels inside/outside of a centered circle with ``fillwith``.

    Returns a modified copy of ``array``.

    Parameters
    ----------
    array : 2d/3d/4d numpy ndarray
        Input frame or cube.
    radius : int
        Radius of the circular aperture.
    fillwith : int, float or np.nan, optional
        Value to put instead of the masked out pixels.
    mode : {'in', 'out'}, optional
        When set to 'in' then the pixels inside the radius are set to
        ``fillwith``. When set to 'out' the pixels outside the circular mask are
        set to ``fillwith``.

    Returns
    -------
    array_masked : numpy ndarray
        Masked frame or cube.

    """
    if not isinstance(fillwith, (int, float)):
        raise ValueError('`fillwith` must be integer, float or np.nan')

    cy, cx = frame_center(array)

    ind = circle(cy, cx, radius)

    if mode == 'in':
        array_masked = array.copy()
        if array.ndim == 2:
            array_masked[ind] = fillwith
        elif array.ndim == 3:
            array_masked[:, ind[1], ind[0]] = fillwith
        elif array.ndim == 4:
            array_masked[:, :, ind[1], ind[0]] = fillwith

    elif mode == 'out':
        array_masked = np.full_like(array, fillwith)
        if array.ndim == 2:
            array_masked[ind] = array[ind]
        elif array.ndim == 3:
            array_masked[:, ind[1], ind[0]] = array[:, ind[1], ind[0]]
        elif array.ndim == 4:
            array_masked[:, :, ind[1], ind[0]] = array[:, :, ind[1], ind[0]]

    return array_masked


def create_ringed_spider_mask(im_shape, ann_out, ann_in=0, sp_width=10,
                              sp_angle=0):
    """
    Mask out information is outside the annulus and inside the spiders (zeros).

    Parameters
    ----------
    im_shape : tuple of int
        Tuple of length two with 2d array shape (Y,X).
    ann_out : int
        Outer radius of the annulus.
    ann_in : int
        Inner radius of the annulus.
    sp_width : int
        Width of the spider arms (3 branches).
    sp_angle : int
        angle of the first spider arm (on the positive horizontal axis) in
        counter-clockwise sense.

    Returns
    -------
    mask : numpy ndarray
        2d array of zeros and ones.

    """
    mask = np.zeros(im_shape)

    s = im_shape[0]
    r = s/2
    theta = np.arctan2(sp_width/2, r)

    t0 = np.array([theta, np.pi-theta, np.pi+theta, np.pi*2 - theta])
    t1 = t0 + sp_angle/180 * np.pi
    t2 = t1 + np.pi/3
    t3 = t2 + np.pi/3

    x1 = r * np.cos(t1) + s/2
    y1 = r * np.sin(t1) + s/2
    x2 = r * np.cos(t2) + s/2
    y2 = r * np.sin(t2) + s/2
    x3 = r * np.cos(t3) + s/2
    y3 = r * np.sin(t3) + s/2

    rr1, cc1 = polygon(y1, x1)
    rr2, cc2 = polygon(y2, x2)
    rr3, cc3 = polygon(y3, x3)

    cy, cx = frame_center(mask)
    rr0, cc0 = circle(cy, cx, min(ann_out, cy))
    rr4, cc4 = circle(cy, cx, ann_in)

    mask[rr0, cc0] = 1
    mask[rr1, cc1] = 0
    mask[rr2, cc2] = 0
    mask[rr3, cc3] = 0
    mask[rr4, cc4] = 0
    return mask


def dist(yc, xc, y1, x1):
    """
    Return the Euclidean distance between two points.
    """
    if not isinstance(xc, (float, int)):
        raise TypeError("`xc` must be a float or int")
    if not isinstance(yc, (float, int)):
        raise TypeError("`yc` must be a float or int")
    if not isinstance(x1, (float, int)):
        raise TypeError("`x1` must be a float or int")
    if not isinstance(y1, (float, int)):
        raise TypeError("`y1` must be a float or int")
    return np.sqrt((yc-y1)**2 + (xc-x1)**2)


def dist_matrix(n, cx=None, cy=None):
    """
    Create matrix with euclidian distances from a reference point (cx, cy).

    Parameters
    ----------
    n : int
        output image shape is (n, n)
    cx,cy : float
        reference point. Defaults to the center.

    Returns
    -------
    im : ndarray with shape (n, n)

    Notes
    -----
    This is a replacement for ANDROMEDA's DISTC.

    """
    if cx is None:
        cx = (n - 1) / 2
    if cy is None:
        cy = (n - 1) / 2

    yy, xx = np.ogrid[:n, :n]
    return np.sqrt((yy-cy)**2 + (xx-cx)**2)


def frame_center(array, verbose=False):
    """
    Return the coordinates y,x of the frame(s) center.

    Parameters
    ----------
    array : 2d/3d/4d numpy ndarray
        Frame or cube.
    verbose : bool optional
        If True the center coordinates are printed out.

    Returns
    -------
    cy, cx : float
        Coordinates of the center.

    """
    if array.ndim == 2:
        shape = array.shape
    elif array.ndim == 3:
        shape = array[0].shape
    elif array.ndim == 4:
        shape = array[0, 0].shape
    else:
        raise ValueError('`array` is not a 2d, 3d or 4d array')

    cy = shape[0] / 2 - 0.5
    cx = shape[1] / 2 - 0.5

    if verbose:
        print('Center px coordinates at x,y = ({}, {})'.format(cx, cy))
    return cy, cx


def get_square(array, size, y, x, position=False, force=False, verbose=True):
    """
    Return an square subframe from a 2d array or image.

    Parameters
    ----------
    array : 2d numpy ndarray
        Input frame.
    size : int
        Size of the subframe.
    y : int
        Y coordinate of the center of the subframe (obtained with the function
        ``frame_center``).
    x : int
        X coordinate of the center of the subframe (obtained with the function
        ``frame_center``).
    position : bool, optional
        If set to True return also the coordinates of the bottom-left vertex.
    force : bool, optional
        Size and the size of the 2d array must be both even or odd. With
        ``force`` set to True this condition can be avoided.
    verbose : bool optional
        If True, warning messages might be shown.

    Returns
    -------
    array_out : numpy ndarray
        Sub array.
    y0, x0 : int
        [position=True] Coordinates of the bottom-left vertex.

    """
    size_init = array.shape[0]  # assuming square frames

    if array.ndim != 2:
        raise TypeError('Input array is not a 2d array.')
    if not isinstance(size, int):
        raise TypeError('`Size` must be integer')
    if size >= size_init:  # assuming square frames
        msg = "`Size` is equal to or bigger than the initial frame size"
        raise ValueError(msg)

    if not force:
        # Even input size
        if size_init % 2 == 0:
            # Odd size
            if size % 2 != 0:
                size += 1
                if verbose:
                    print("`Size` is odd (while input frame size is even). "
                          "Setting `size` to {} pixels".format(size))
        # Odd input size
        else:
            # Even size
            if size % 2 == 0:
                size += 1
                if verbose:
                    print("`Size` is even (while input frame size is odd). "
                          "Setting `size` to {} pixels".format(size))
    else:
        # Even input size
        if size_init % 2 == 0:
            # Odd size
            if size % 2 != 0 and verbose:
                print("WARNING: `size` is odd while input frame size is even. "
                      "Make sure the center coordinates are set properly")
        # Odd input size
        else:
            # Even size
            if size % 2 == 0 and verbose:
                print("WARNING: `size` is even while input frame size is odd. "
                      "Make sure the center coordinates are set properly")

    # wing is added to the sides of the subframe center
    wing = (size - 1) / 2

    y0 = int(y - wing)
    y1 = int(y + wing + 1)  # +1 cause endpoint is excluded when slicing
    x0 = int(x - wing)
    x1 = int(x + wing + 1)

    if y0 < 0 or x0 < 0 or y1 > size_init or x1 > size_init:
        # assuming square frames
        raise RuntimeError('square cannot be obtained with size={}, y={}, x={}'
                           ''.format(size, y, x))

    array_out = array[y0: y1, x0: x1].copy()

    if position:
        return array_out, y0, x0
    else:
        return array_out


def get_circle(array, radius, cy=None, cx=None, mode="mask"):
    """
    Return a centered circular region from a 2d ndarray.

    Parameters
    ----------
    array : numpy ndarray
        Input 2d array or image.
    radius : int
        The radius of the circular region.
    output_values : bool, optional
        Sets the type of output.
    cy, cx : int, optional
        Coordinates of the circle center. If one of them is ``None``, the center
        of ``array`` is used.
    mode : {'mask', 'val'}, optional
        Controls what is returned: array with circular mask applied, or values
        of the pixels in the circular region.

    Returns
    -------
    masked : numpy ndarray
        [mode="mask"] Input array with the circular mask applied.
    values : numpy ndarray
        [mode="val"] 1d array with the values of the pixels in the circular
        region.

    Notes
    -----
    An alternative implementation would use ``skimage.draw.circle``. ``circle``
    performs better on large ``array``s (e.g. 1000px, 10.000px), while the
    current implementation is faster for small ``array``s (e.g. 100px). See
    `test_shapes.py` for benchmark details.

    """
    if array.ndim != 2:
        raise TypeError('Input array is not a frame or 2d array.')
    sy, sx = array.shape
    if cy is None or cx is None:
        cy, cx = frame_center(array, verbose=False)

    # ogrid is a multidim mesh creator (faster than mgrid):
    yy, xx = np.ogrid[:sy, :sx]
    circle = (yy - cy) ** 2 + (xx - cx) ** 2  # eq of circle. sq dist to center
    circle_mask = circle < radius ** 2  # boolean mask
    if mode == "mask":
        return array * circle_mask
    elif mode == "val":
        return array[circle_mask]
    else:
        raise ValueError("mode '{}' unknown!".format(mode))


def get_ellipse(data, a, b, pa, cy=None, cx=None, mode="ind"):
    """
    Return a centered elliptical region from a 2d ndarray.

    Parameters
    ----------
    data : numpy ndarray or tuple
        Input 2d array (image) or tuple with a shape.
    a : float
        Semi-major axis.
    b : float
        Semi-minor axis.
    pa : deg, float
        The PA of the semi-major axis in degrees.
    cy, cx : int or None, optional
        Coordinates of the circle center. If ``None``, the center is determined
        by the ``frame_center`` function.
    mode : {'ind', 'val', 'mask', 'bool'}, optional
        Controls what is returned: indices of selected pixels, values of
        selected pixels, or a boolean mask.

    Returns
    -------
    indices : tuple(y, x)
        [mode='ind'] Coordinates of the inner elliptical region.
    values : 1d ndarray
        [mode='val'] Values of the pixels in the inner elliptical region.
    masked : 2d ndarray
        [mode='mask'] Input image where the outer region is masked with ``0``.
    bool_mask : 2d boolean ndarray
        [mode='bool'] A boolean mask where ``True`` is the inner region.

    """

    def distance(yc, xc, y1, x1):
        return np.sqrt((yc - y1) ** 2 + (xc - x1) ** 2)
    # --------------------------------------------------------------------------

    array = frame_or_shape(data)

    if cy is None or cx is None:
        cy, cx = frame_center(array, verbose=False)

    # Definition of other parameters of the ellipse
    f = np.sqrt(a ** 2 - b ** 2)  # dist between center and foci of the ellipse
    pa_rad = np.deg2rad(pa)
    pos_f1 = (cy + f * np.cos(pa_rad), cx + f * np.sin(pa_rad))  # first focus
    pos_f2 = (cy - f * np.cos(pa_rad), cx - f * np.sin(pa_rad))  # second focus

    # ogrid is a multidim mesh creator (faster than mgrid):
    yy, xx = np.ogrid[:array.shape[0], :array.shape[1]]
    ellipse = (distance(yy, xx, pos_f1[0], pos_f1[1]) +
               distance(yy, xx, pos_f2[0], pos_f2[1]))
    ellipse_mask = ellipse < 2 * a  # boolean mask

    if mode == "ind":
        return np.where(ellipse_mask)
    elif mode == "val":
        return array[ellipse_mask]
    elif mode == "mask":
        return array * ellipse_mask
    elif mode == "bool":
        return ellipse_mask
    else:
        raise ValueError("mode '{}' unknown!".format(mode))


def get_annulus_segments(data, inner_radius, width, nsegm=1, theta_init=0,
                         optim_scale_fact=1, mode="ind"):
    """
    Return indices or values in segments of a centerered annulus.

    The annulus is defined by ``inner_radius <= annulus < inner_radius+width``.

    Parameters
    ----------
    data : 2d numpy ndarray or tuple
        Input 2d array (image) ot tuple with its shape.
    inner_radius : float
        The inner radius of the donut region.
    width : float
        The size of the annulus.
    nsegm : int
        Number of segments of annulus to be extracted.
    theta_init : int
        Initial azimuth [degrees] of the first segment, counting from the
        postivie y-axis clockwise.
    optim_scale_fact : float
        To enlarge the width of the segments, which can then be used as
        optimization segments (e.g. in LOCI).
    mode : {'ind', 'val', 'mask'}, optional
        Controls what is returned: indices of selected pixels, values of
        selected pixels, or a boolean mask.

    Returns
    -------
    indices : list of ndarrays
        [mode='ind'] Coordinates of pixels for each annulus segment.
    values : list of ndarrays
        [mode='val'] Pixel values.
    masked : list of ndarrays
        [mode='mask'] Copy of ``data`` with masked out regions.

    Notes
    -----
    Moving from ``get_annulus`` to ``get_annulus_segments``:

    .. code::python
        # get_annulus handles one single segment only, so note the ``[0]`` after
        the call to get_annulus_segments if you want to work with one single
        segment only.

        get_annulus(arr, 2, 3, output_indices=True)
        # is the same as
        get_annulus_segments(arr, 2, 3)[0]

        get_annulus(arr, inr, w, output_values=True)
        # is the same as
        get_annulus_segments(arr, inr, w, mode="val")[0]

        get_annulus(arr, inr, w)
        # is the same as
        get_annulus_segments(arr, inr, w, mode="mask")[0]

        # the only difference is the handling of the border values:
        # get_annulus_segments is `in <= ann < out`, while get_annulus is
        # `in <= ann <= out`. But that should make no difference in practice.

    """
    array = frame_or_shape(data)

    if not isinstance(nsegm, int):
        raise TypeError('`nsegm` must be an integer')

    cy, cx = frame_center(array)
    azimuth_coverage = np.deg2rad(int(np.ceil(360 / nsegm)))
    twopi = 2 * np.pi

    xx, yy = np.mgrid[:array.shape[0], :array.shape[1]]
    rad = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    phi = np.arctan2(yy - cy, xx - cx)
    phirot = phi % twopi
    outer_radius = inner_radius + (width*optim_scale_fact)
    masks = []

    for i in range(nsegm):
        phi_start = np.deg2rad(theta_init) + (i * azimuth_coverage)
        phi_end = phi_start + azimuth_coverage

        if phi_start < twopi and phi_end > twopi:
            masks.append((rad >= inner_radius) & (rad < outer_radius) &
                         (phirot >= phi_start) & (phirot <= twopi) |
                         (rad >= inner_radius) & (rad < outer_radius) &
                         (phirot >= 0) & (phirot < phi_end - twopi))
        elif phi_start >= twopi and phi_end > twopi:
            masks.append((rad >= inner_radius) & (rad < outer_radius) &
                         (phirot >= phi_start - twopi) &
                         (phirot < phi_end - twopi))
        else:
            masks.append((rad >= inner_radius) & (rad < outer_radius) &
                         (phirot >= phi_start) & (phirot < phi_end))

    if mode == "ind":
        return [np.where(mask) for mask in masks]
    elif mode == "val":
        return [array[mask] for mask in masks]
    elif mode == "mask":
        return [array*mask for mask in masks]
    else:
        raise ValueError("mode '{}' unknown!".format(mode))


def get_ell_annulus(data, a, b, PA, width, cy=None, cx=None, mode="ind"):
    """
    Return a centered elliptical annulus from a 2d ndarray

    All the rest pixels are set to zeros.

    Parameters
    ----------
    data : numpy ndarray or tuple
        Input 2d array (image) or tuple with a shape.
    a : float
        Semi-major axis.
    b : float
        Semi-minor axis.
    PA : deg, float
        The PA of the semi-major axis in degrees.
    width : float
        The size of the annulus along the semi-major axis; it is proportionnally
        thinner along the semi-minor axis.
    output_values : {False, True}, optional
        If True returns the values of the pixels in the annulus.
    cy, cx : int or None, optional
        Coordinates of the circle center. If ``None``, the center is determined
        by the ``frame_center`` function.
    mode : {'ind', 'val', 'mask'}, optional
        Controls what is returned: indices of selected pixels, values of
        selected pixels, or a boolean mask.

    Returns
    -------
    indices : tuple(y, x)
        [mode='ind'] Coordinates of the inner elliptical region.
    values : 1d ndarray
        [mode='val'] Values of the pixels in the inner elliptical region.
    masked : 2d ndarray
        [mode='mask'] Input image where the outer region is masked with ``0``.

    """
    array = frame_or_shape(data)

    hwa = width / 2  # half width for a
    hwb = (width * b / a) / 2  # half width for b

    big_ellipse = get_ellipse(array, a + hwa, b + hwb, PA, cy=cy, cx=cx,
                              mode="bool")
    small_ellipse = get_ellipse(array, a - hwa, b - hwb, PA, cy=cy, cx=cx,
                                mode="bool")

    ell_ann_mask = big_ellipse ^ small_ellipse

    if mode == "ind":
        return np.where(ell_ann_mask)
    elif mode == "val":
        return array[ell_ann_mask]
    elif mode == "mask":
        return array * ell_ann_mask
    elif mode == "bool":
        return ell_ann_mask
    else:
        raise ValueError("mode '{}' unknown!".format(mode))


def matrix_scaling(matrix, scaling):
    """
    Scale a matrix using ``sklearn.preprocessing.scale`` function.

    Parameters
    ----------
    matrix : 2d numpy ndarray
        Input 2d array.
    scaling : None or string
        Scaling method.

        ``None``
            no scaling is performed on the input data before SVD
        ``"temp-mean"``
            temporal px-wise mean subtraction
        ``"spat-mean"``
            the spatial mean is subtracted
        ``temp-standard"``
            temporal mean centering plus scaling to unit variance
        ``"spat-standard"``
            spatial mean centering plus scaling to unit variance

    Returns
    -------
    matrix : 2d numpy ndarray
        2d array with scaled values.

    """
    if scaling is None:
        pass
    elif scaling == 'temp-mean':
        matrix = scale(matrix, with_mean=True, with_std=False)
    elif scaling == 'spat-mean':
        matrix = scale(matrix, with_mean=True, with_std=False, axis=1)
    elif scaling == 'temp-standard':
        matrix = scale(matrix, with_mean=True, with_std=True)
    elif scaling == 'spat-standard':
        matrix = scale(matrix, with_mean=True, with_std=True, axis=1)
    else:
        raise ValueError('Scaling mode not recognized')

    return matrix


def prepare_matrix(array, scaling=None, mask_center_px=None, mode='fullfr',
                   inner_radius=None, outer_radius=None, verbose=True):
    """
    Build the matrix for the SVD/PCA and other matrix decompositions.

    Center the data and mask the frame's central area if needed.

    Parameters
    ----------
    array : 3d numpy ndarray
        Input cube.
    scaling : {None, "temp-mean", spat-mean", "temp-standard", "spat-standard"},
        None or str optional
        Pixel-wise scaling mode using ``sklearn.preprocessing.scale`` function.
        If set to None, the input matrix is left untouched. Otherwise:

        ``temp-mean``: temporal px-wise mean is subtracted.

        ``spat-mean``: spatial mean is subtracted.

        ``temp-standard``: temporal mean centering plus scaling pixel values
        to unit variance.

        ``spat-standard``: spatial mean centering plus scaling pixel values
        to unit variance.

    mask_center_px : None or int, optional
        [mode='fullfr'] Whether to mask the center of the frames or not.
    mode : {'fullfr', 'annular'}, optional
        Whether to use the whole frames or a single annulus.
    inner_radius : int or float, optional
        [mode='annular'] Distance in pixels from the center of the frame to the
        inner radius of the annulus.
    outer_radius : int or float, optional
        [mode='annular'] Distance in pixels from the center of the frame to the
        outer radius of the annulus.
    verbose : bool, optional
        If True prints intermediate info.

    Returns
    -------
    matrix : 2d numpy ndarray
        Out matrix whose rows are vectorized frames from the input cube.
    ind : tuple
        [mode='annular'] Indices of the annulus as ``(yy, xx)``.

    """
    if mode == 'annular':
        if inner_radius is None or outer_radius is None:
            raise ValueError('`inner_radius` and `outer_radius` must be defined'
                             ' in annular mode')
        fr_size = array.shape[1]
        annulus_width = int(np.round(outer_radius - inner_radius))
        ind = get_annulus_segments((fr_size, fr_size), inner_radius,
                                   annulus_width, nsegm=1)[0]
        yy, xx = ind
        matrix = array[:, yy, xx]

        matrix = matrix_scaling(matrix, scaling)

        if verbose:
            msg = 'Done vectorizing the cube annulus. Matrix shape: ({}, {})'
            print(msg.format(matrix.shape[0], matrix.shape[1]))
        return matrix, ind

    elif mode == 'fullfr':
        if mask_center_px:
            array = mask_circle(array, mask_center_px)

        nfr = array.shape[0]
        matrix = np.reshape(array, (nfr, -1))  # == for i: array[i].flatten()

        matrix = matrix_scaling(matrix, scaling)

        if verbose:
            msg = 'Done vectorizing the frames. Matrix shape: ({}, {})'
            print(msg.format(matrix.shape[0], matrix.shape[1]))
        return matrix


def reshape_matrix(array, y, x):
    """
    Convert a matrix whose rows are vect. frames to a cube with reshaped frames.

    Parameters
    ----------
    array : 2d ndarray
        Input data of shape ``(nframes, npixels)``. Every row (``array[n]``)
        corresponds to one vectorized ("flattened") 2d frame.
    y, x : int
        desired height and width of the frames. ``y*x = npixels``

    Returns
    -------
    cube : 3d ndarray
        Cube of shape ``(nframes, y, x)``.

    Examples
    --------
    .. code:: python

        In [1]: vect_frames = np.array([[1, 1, 1, 2, 2, 2], [1, 2, 3, 4, 5, 6]])
        In [2]: cube = vip.var.reshape_matrix(vect_frames, 2, 3)
        In [3]: cube
        Out[3]:
        array([[[1, 1, 1],
                [2, 2, 2]],

               [[1, 2, 3],
                [4, 5, 6]]])
        In [4]: cube.shape
        Out[4]: (2, 2, 3)

    """
    return array.reshape(array.shape[0], y, x)


