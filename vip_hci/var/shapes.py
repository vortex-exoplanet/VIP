#! /usr/bin/env python

"""
Module with various functions.
"""

from __future__ import division, print_function

__author__ = 'Carlos Alberto Gomez Gonzalez'
__all__ = ['dist',
           'frame_center',
           'get_square',
           'get_circle',
           'get_ellipse',
           'get_annulus_segments',
           'get_annulus',
           'get_annulus_cube',
           'get_ell_annulus',
           'mask_circle',
           'create_ringed_spider_mask',
           'matrix_scaling',
           'prepare_matrix',
           'reshape_matrix',
           'get_indices_annulus']

import numpy as np
from skimage.draw import polygon
from skimage.draw import circle
from sklearn.preprocessing import scale


def mask_circle(array, radius, fillwith=0, mode='in'):
    """ Masks the pixels inside/outside (depending on ``mode``) of a centered
    circle from a frame or (3d or 4d) cube, replacing the values with the value
    of ``fillwith``.

    Parameters
    ----------
    array : array_like
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
    array_masked : array_like
        Masked frame or cube.
    """
    if not isinstance(fillwith, (int, float)):
        raise ValueError('`Fillwith` must be integer, float or np.nan')

    if array.ndim == 2:
        cy, cx = frame_center(array)
    elif array.ndim == 3:
        cy, cx = frame_center(array[0])
    elif array.ndim == 4:
        cy, cx = frame_center(array[0][0])
    else:
        raise ValueError('`Array` must be a 2d, 3d or 4d np.ndarray')

    ind = circle(cy, cx, radius)

    if mode == 'in':
        array_masked = array.copy()
    elif mode == 'out':
        array_masked = np.ones_like(array) * np.nan

    if mode == 'in':
        if array.ndim == 2:
            array_masked[ind] = fillwith
        elif array.ndim == 3:
            array_masked[:, ind[1], ind[0]] = fillwith
        elif array.ndim == 4:
            array_masked[:, :, ind[1], ind[0]] = fillwith

    elif mode == 'out':
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
    mask : array_like
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
    """ Returns the Euclidean distance between two points.
    """
    return np.sqrt((yc-y1)**2 + (xc-x1)**2)


def frame_center(array, verbose=False):
    """ Returns the coordinates y,x of the frame(s) center.

    Parameters
    ----------
    array : array_like
        2d or 3d array.
    verbose : bool optional
        If True the center coordinates are printed out.
    """
    if array.ndim == 2:
        cy = array.shape[0]/2 - 0.5
        cx = array.shape[1]/2 - 0.5
    elif array.ndim == 3:
        cy = array[0].shape[0] / 2 - 0.5
        cx = array[0].shape[1] / 2 - 0.5
    elif array.ndim == 4:
        cy = array[0, 0].shape[0] / 2 - 0.5
        cx = array[0, 0].shape[1] / 2 - 0.5
    else:
        raise ValueError('Input array is not a 2d, 3d or 4d array')

    if verbose:
        print('Center px coordinates at x,y = ({}, {})'.format(cx, cy))
    return cy, cx

    
def get_square(array, size, y, x, position=False, force=False):
    """ Returns an square subframe from a 2d array or image.
    
    Parameters
    ----------
    array : array_like
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
        
    Returns
    -------
    array_view : array_like
        Sub array.
        
    """
    if array.ndim != 2:
        raise TypeError('Input array is not a frame or 2d array.')

    ary, arx = array.shape

    if not force:
        if ary % 2 == 0:    # assuming square frames
            if size % 2 != 0:
                size += 1
                print('Subframe size is odd (while frame size is even)')
                print('Setting subframe size to {} pixels'.format(size))
        else:
            if size % 2 == 0:
                size += 1
                print('Subframe size is even (while frame size is odd)')
                print('Setting subframe size to {} pixels'.format(size))

    # wing is added to the sides of the subframe center
    if size % 2 != 0:
        wing = size//2
    else:
        wing = size/2 - 0.5

    y0 = int(y-wing)
    y1 = int(y+wing+1)  # +1 cause endpoint is excluded when slicing
    x0 = int(x-wing)
    x1 = int(x+wing+1)
    if (y0 or x0) < 0 or (y1 or x1) > ary:   # assuming square frames
        msg = 'square cannot be obtained given the size and y,x combination'
        raise RuntimeError(msg)

    array_view = array[y0: y1, x0: x1].copy()
    
    if position:
        return array_view, y0, x0
    else:
        return array_view


def get_circle(array, radius, output_values=False, cy=None, cx=None):
    """Returns a centered circular region from a 2d ndarray. All the rest
    pixels are set to zeros.

    Parameters
    ----------
    array : array_like
        Input 2d array or image.
    radius : int
        The radius of the circular region.
    output_values : bool, optional
        Sets the type of output.
    cy, cx : int, optional
        Coordinates of the circle center.

    Returns
    -------
    values : array_like
        1d array with the values of the pixels in the circular region.
    array_masked : array_like
        Input array with the circular mask applied.
    """
    if array.ndim != 2:
        raise TypeError('Input array is not a frame or 2d array.')
    sy, sx = array.shape
    if cy is None or cx is None:
        cy, cx = frame_center(array, verbose=False)

    # ogrid is a multidim mesh creator (faster than mgrid)
    yy, xx = np.ogrid[:sy,:sx]
    circle = (yy - cy) ** 2 + (xx - cx) ** 2  # eq of circle. sq dist to center
    circle_mask = circle < radius ** 2  # mask of 1's and 0's
    if output_values:
        values = array[circle_mask]
        return values
    else:
        array_masked = array * circle_mask
        return array_masked


def get_ellipse(array, a, b, PA, output_values=False, cy=None, cx=None,
                output_indices=False):
    """ Returns a centered elliptical region from a 2d ndarray. All the rest 
    pixels are set to zeros.
    
    Parameters
    ----------
    array : array_like
        Input 2d array or image. 
    a : float or int
        Semi-major axis.
    b : float or int
        Semi-minor axis.
    PA : deg, float
        The PA of the semi-major axis.
    output_values : bool, optional
        If True returns the values of the pixels in the annulus.
    cy, cx : int or None, optional
        Coordinates of the circle center. If ``None``, the center is determined
        by the ``frame_center`` function.
    output_indices : bool, optional
        If True returns the indices inside the annulus.
    
    Returns
    -------
    Depending on output_values, output_indices:
    values : array_like
        1d array with the values of the pixels in the circular region.
    array_masked : array_like
        Input array with the circular mask applied.
    y, x : array_like
        Coordinates of pixels in circle.
    """

    if array.ndim != 2:
        raise TypeError('Input array is not a frame or 2d array.')
    sy, sx = array.shape
    if cy is None or cx is None:
        cy, cx = frame_center(array, verbose=False)

    # Definition of other parameters of the ellipse
    f = np.sqrt(a ** 2 - b ** 2)  # distance between center and foci of the ellipse
    PA_rad = np.deg2rad(PA)
    pos_f1 = (cy + f * np.cos(PA_rad), cx + f * np.sin(PA_rad))  # coords of first focus
    pos_f2 = (cy - f * np.cos(PA_rad), cx - f * np.sin(PA_rad))  # coords of second focus

    yy, xx = np.ogrid[:sy, :sx]
    # ogrid is a multidim mesh creator (faster than mgrid)
    ellipse = dist(yy, xx, pos_f1[0], pos_f1[1]) + dist(yy, xx, pos_f2[0],
                                                        pos_f2[1])
    ellipse_mask = ellipse < 2 * a  # mask of 1's and 0's

    if output_values and not output_indices:
        values = array[ellipse_mask]
        return values
    elif output_indices and not output_values:
        indices = np.array(np.where(ellipse_mask))
        y = indices[0]
        x = indices[1]
        return y, x
    elif output_indices and output_values:
        msg = 'output_values and output_indices cannot be both True.'
        raise ValueError(msg)
    else:
        array_masked = array * ellipse_mask
        return array_masked


def get_annulus_segments(array, inner_radius, width, nsegm=1, theta_init=0,
                         optim_scale_fact=1, output_values=False):
    """ Returns indices or values in segments of a centerered annulus from a
    2d ndarray.

    Parameters
    ----------
    array : array_like
        Input 2d array or image.
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
    output_values : bool, optional
        If True returns the values of the pixels in the each quadrant instead
        of the indices.

    Returns
    -------
    indices : list with lenght nsegm
        Coordinates of pixels for each annulus segment.
    If output_values is True the pixel values are returned instead.

    """
    if array.ndim != 2:
        raise TypeError('Input array is not a frame or 2d array')
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
    indices = []

    for i in range(nsegm):
        phi_start = np.deg2rad(theta_init) + (i * azimuth_coverage)
        phi_end = phi_start + azimuth_coverage

        if phi_start < twopi and phi_end > twopi:
            indices.append(
                np.where((rad >= inner_radius) & (rad < outer_radius) &
                         (phirot >= phi_start) & (phirot <= twopi) |
                         (rad >= inner_radius) & (rad < outer_radius) &
                         (phirot >= 0) & (phirot < phi_end - twopi)))
        elif phi_start >= twopi and phi_end > twopi:
            indices.append(
                np.where((rad >= inner_radius) & (rad < outer_radius) &
                         (phirot >= phi_start - twopi) &
                         (phirot < phi_end - twopi)))
        else:
            indices.append(
                    np.where((rad >= inner_radius) & (rad < outer_radius) &
                             (phirot >= phi_start) & (phirot < phi_end)))

    if output_values:
        values = [array[segment] for segment in indices]
        return np.array(values)
    else:
        return indices


# TODO: remove this in VIP v1.0.0. Replaced with get_annulus_segments
def get_annulus(array, inner_radius, width, output_values=False, 
                output_indices=False):                                          
    """Returns a centerered annulus from a 2d ndarray. All the rest pixels are 
    set to zeros. 
    
    Parameters
    ----------
    array : array_like
        Input 2d array or image. 
    inner_radius : float
        The inner radius of the donut region.
    width : int
        The size of the annulus.
    output_values : bool, optional
        If True returns the values of the pixels in the annulus.
    output_indices : bool, optional
        If True returns the indices inside the annulus.
    
    Returns
    -------
    Depending on output_values, output_indices:
    values : array_like
        1d array with the values of the pixels in the annulus.
    array_masked : array_like
        Input array with the annular mask applied.
    y, x : array_like
        Coordinates of pixels in annulus.
    """
    if array.ndim != 2:
        raise TypeError('Input array is not a frame or 2d array')
    array = array.copy()
    cy, cx = frame_center(array)
    yy, xx = np.mgrid[:array.shape[0], :array.shape[1]]
    circle = np.sqrt((xx - cx)**2 + (yy - cy)**2)                                                                               
    donut_mask = (circle <= (inner_radius + width)) & (circle >= inner_radius)
    if output_values and not output_indices:
        values = array[donut_mask]
        return values
    elif output_indices and not output_values:      
        indices = np.array(np.where(donut_mask))
        y = indices[0]
        x = indices[1]
        return y, x
    elif output_indices and output_values:
        raise ValueError('output_values and output_indices cannot be both True')
    else:
        array_masked = array*donut_mask
        return array_masked


# TODO: VIP v1.0.0: make use of get_annulus_segments instead
def get_annulus_cube(array, inner_radius, width, output_values=False):     
    """ Returns a centerered annulus from a 3d ndarray. All the rest pixels are 
    set to zeros. 
    
    Parameters
    ----------
    array : array_like
        Input 2d array or image. 
    inner_radius : int
        The inner radius of the donut region.
    width : int
        The size of the annulus.
    output_values : {False, True}, optional
        If True returns the values of the pixels in the annulus.
    
    Returns
    -------
    Depending on output_values:
    values : array_like
        1d array with the values of the pixels in the circular region.
    array_masked : array_like
        Input array with the annular mask applied.

    """
    if array.ndim != 3:
        raise TypeError('Input array is not a cube or 3d array.')
    arr_annulus = np.empty_like(array)
    if output_values:
        values = []
        for i in range(array.shape[0]):
            values.append(get_annulus(array[i], inner_radius, width,
                                      output_values=True))
        return np.array(values)
    else:
        for i in range(array.shape[0]):
            arr_annulus[i] = get_annulus(array[i], inner_radius, width)
        return arr_annulus
    

def get_ell_annulus(array, a, b, PA, width, output_values=False,
                    output_indices=False, cy=None, cx=None):
    """Returns a centered elliptical annulus from a 2d ndarray. All the rest 
    pixels are set to zeros. 

    Parameters
    ----------
    array : array_like
        Input 2d array or image. 
    a : flt
        Semi-major axis.
    b : flt
        Semi-minor axis.
    PA : deg
        The PA of the semi-major axis.
    width : flt
        The size of the annulus along the semi-major axis; it is proportionnally 
        thinner along the semi-minor axis).
    output_values : {False, True}, optional
        If True returns the values of the pixels in the annulus.
    output_indices : {False, True}, optional
        If True returns the indices inside the annulus.
    cy,cx: float, optional
        Location of the center of the annulus to be defined. If not provided, 
    it assumes the annuli are centered on the frame.

    Returns
    -------
    Depending on output_values, output_indices:
    values : array_like
        1d array with the values of the pixels in the annulus.
    array_masked : array_like
        Input array with the annular mask applied.
    y, x : array_like
        Coordinates of pixels in annulus.

    """
    if array.ndim != 2:
        raise TypeError('Input array is not a frame or 2d array.')
    if cy is None or cx is None:
        cy, cx = frame_center(array)
    sy, sx = array.shape

    width_a = width
    width_b = width * b / a

    # Definition of big ellipse
    f_big = np.sqrt((a + width_a / 2) ** 2 - (b + width_b / 2) ** 2)
            # distance between center and foci of the ellipse
    PA_rad = np.deg2rad(PA)
    pos_f1_big = (cy + f_big * np.cos(PA_rad),
                  cx + f_big * np.sin(PA_rad))  # coords of first focus
    pos_f2_big = (cy - f_big * np.cos(PA_rad),
                  cx - f_big * np.sin(PA_rad))  # coords of second focus

    # Definition of small ellipse
    f_sma = np.sqrt((a - width_a / 2) ** 2 - (b - width_b / 2) ** 2)
            # distance between center and foci of the ellipse
    pos_f1_sma = (cy + f_sma * np.cos(PA_rad),
                  cx + f_sma * np.sin(PA_rad))  # coords of first focus
    pos_f2_sma = (cy - f_sma * np.cos(PA_rad),
                  cx - f_sma * np.sin(PA_rad))  # coords of second focus

    yy, xx = np.ogrid[:sy, :sx]
    big_ellipse = dist(yy, xx, pos_f1_big[0], pos_f1_big[1]) + \
                  dist(yy, xx, pos_f2_big[0], pos_f2_big[ 1])
    small_ellipse = dist(yy, xx, pos_f1_sma[0], pos_f1_sma[1]) + \
                    dist(yy, xx, pos_f2_sma[0], pos_f2_sma[1])
    ell_ann_mask = ((big_ellipse < 2 * (a + width / 2)) &
                    (small_ellipse >= 2 * (a - width / 2)))# mask of 1's and 0's

    if output_values and not output_indices:
        values = array[ell_ann_mask]
        return values
    elif output_indices and not output_values:
        indices = np.array(np.where(ell_ann_mask))
        y = indices[0]
        x = indices[1]
        return y, x
    elif output_indices and output_values:
        msg = 'output_values and output_indices cannot be both True.'
        raise ValueError(msg)
    else:
        array_masked = array * ell_ann_mask
        return array_masked


def matrix_scaling(matrix, scaling):
    """ Scales a matrix using sklearn.preprocessing.scale function.

    Parameters
    ----------
    matrix : array_like
        Input 2d array.
    scaling : {None, 'temp-mean', 'spat-mean', 'temp-standard', 'spat-standard'}, str optional
        With None, no scaling is performed on the input data before SVD. With
        "temp-mean" then temporal px-wise mean subtraction is done, with
        "spat-mean" then the spatial mean is subtracted, with "temp-standard"
        temporal mean centering plus scaling to unit variance is done and with
        "spat-standard" spatial mean centering plus scaling to unit variance is
        performed.

    Returns
    -------
    matrix : array_like
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
                   annulus_radius=None, annulus_width=None, verbose=True):
    """ Builds the matrix for the SVD/PCA and other matrix decompositions,
    centers the data and masks the frames central area if needed.

    Parameters
    ----------
    array : array_like
        Input cube, 3d array.
    scaling : {None, 'temp-mean', 'spat-mean', 'temp-standard', 'spat-standard'}
        With None, no scaling is performed on the input data before SVD. With
        "temp-mean" then temporal px-wise mean subtraction is done, with
        "spat-mean" then the spatial mean is subtracted, with "temp-standard"
        temporal mean centering plus scaling to unit variance is done and with
        "spat-standard" spatial mean centering plus scaling to unit variance is
        performed.
    mask_center_px : None or Int, optional
        Whether to mask the center of the frames or not.
    mode : {'fullfr', 'annular'}, str optional
        Whether to use the whole frames or a single annulus.
    annulus_radius : float, optional
        Distance in pixels from the center of the frame to the center of the
        annulus.
    annulus_width : float, optional
        Width of the annulus in pixels.
    verbose : {True, False}, bool optional
        If True prints intermediate info and timing.

    Returns
    -------
    If mode is `annular` then the indices of the annulus (yy, xx) are returned
    along with the matrix.

    matrix : array_like
        Out matrix whose rows are vectorized frames from the input cube.

    """
    if mode == 'annular':
        if annulus_radius is None or annulus_width is None:
            raise ValueError('Annulus_radius and/or annulus_width can be None '
                             'in annular mode')

        ind = get_annulus(array[0], annulus_radius - annulus_width / 2,
                          annulus_width, output_indices=True)
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
    """ Converts a matrix whose rows are vectorized frames to a cube with
    reshaped frames.
    """
    return array.reshape(array.shape[0], y, x)


def get_indices_annulus(shape, inrad, outrad, mask=None, maskrad=None,
                        verbose=False):
    """ mask is a list of tuples X,Y
    # TODO: documentation
    """
    framemp = np.zeros(shape)
    if mask is not None:
        if not isinstance(mask, list):
            raise TypeError('Mask should be a list of tuples')
        if maskrad is None:
            raise ValueError('Fwhm not given')
        for xy in mask:
            # patch_size/2 diameter aperture
            cir = circle(xy[1], xy[0], maskrad, shape)
            framemp[cir] = 1

    annulus_width = outrad - inrad
    cy, cx = frame_center(framemp)
    yy, xx = np.mgrid[:framemp.shape[0], :framemp.shape[1]]
    circ = np.sqrt((xx - cx)**2 + (yy - cy)**2)
    donut_mask = (circ <= (inrad + annulus_width)) & (circ >= inrad)
    y, x = np.where(donut_mask)
    if mask is not None:
        npix = y.shape[0]
        ymask, xmask = np.where(framemp)    # masked pixels where == 1
        inds = []
        for i, tup in enumerate(zip(y, x)):
            if tup in zip(ymask, xmask):
                inds.append(i)
        y = np.delete(y, inds)
        x = np.delete(x, inds)

    if verbose:
        print(y.shape[0], 'pixels in annulus')
    return y, x


