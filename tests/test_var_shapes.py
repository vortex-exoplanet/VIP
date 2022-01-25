"""
Tests for var/shapes.py

"""

__author__ = "Ralf Farkas"

from .helpers import aarc, np
from vip_hci.var import (frame_center, mask_circle, get_square, get_circle, 
                         get_annulus_segments, dist, matrix_scaling, 
                         reshape_matrix, get_ell_annulus, get_ellipse)


PRETTY_ODD = np.array([
    [1, 1, 1, 1, 1],
    [1, 2, 2, 2, 1],
    [1, 2, 3, 2, 1],
    [1, 2, 2, 2, 1],
    [1, 1, 1, 1, 1]
])

PRETTY_EVEN = np.array([
    [1, 1, 1, 1, 1, 1],
    [1, 2, 2, 2, 2, 1],
    [1, 2, 3, 3, 2, 1],
    [1, 2, 3, 3, 2, 1],
    [1, 2, 2, 2, 2, 1],
    [1, 1, 1, 1, 1, 1]
])


def test_frame_center():
    frames = 39
    nlambda = 2

    res44 = (2.0, 2.0) # replaced (1.5, 1.5) considering new convention
    res55 = (2.0, 2.0)

    # 2D
    assert frame_center(np.zeros((4, 4))) == res44
    assert frame_center(np.zeros((5, 5))) == res55

    # 3D
    assert frame_center(np.zeros((frames, 4, 4))) == res44
    assert frame_center(np.zeros((frames, 5, 5))) == res55

    # 4D
    assert frame_center(np.zeros((nlambda, frames, 4, 4))) == res44
    assert frame_center(np.zeros((nlambda, frames, 5, 5))) == res55


def test_mask_circle():

    size = 5
    radius = 2

    ones = np.ones((size, size))

    # "in" and "out" should be complementary
    res_in = mask_circle(ones, radius=radius, mode="in")
    res_out = mask_circle(ones, radius=radius, mode="out")
    aarc(res_in+res_out, ones)

    # radius=2 -> central region should be 3x3 pixels = 9 pixels
    aarc(res_out.sum(), 9)


def test_get_square():
    aarc(get_square(PRETTY_ODD, size=3, x=2, y=2),
         np.array([[2, 2, 2],
                   [2, 3, 2],
                   [2, 2, 2]]))

    aarc(get_square(PRETTY_ODD, size=2, x=2, y=2),
         np.array([[2, 2, 2],
                   [2, 3, 2],
                   [2, 2, 2]]))
    # -> prints warning

    aarc(get_square(PRETTY_ODD, size=2, x=2, y=2, force=True),
         np.array([[2, 2],
                   [2, 3]]))

    aarc(get_square(PRETTY_EVEN, size=2, x=3, y=3),
         np.array([[3, 3],
                  [3, 3]]))

    aarc(get_square(PRETTY_EVEN, size=3, x=3, y=3),
         np.array([[2, 2, 2, 2],
                   [2, 3, 3, 2],
                   [2, 3, 3, 2],
                   [2, 2, 2, 2]]))
    # -> prints warning

    aarc(get_square(PRETTY_EVEN, size=2, x=4, y=2, force=True),
         np.array([[2, 2],
                   [3, 2]]))


def test_get_circle():

    ar = np.ones((10, 10), dtype=int)
    aarc(get_circle(ar, radius=4),
         np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 1, 1, 1, 1, 1, 0, 0],
                   [0, 0, 1, 1, 1, 1, 1, 1, 1, 0],
                   [0, 0, 1, 1, 1, 1, 1, 1, 1, 0],
                   [0, 0, 1, 1, 1, 1, 1, 1, 1, 0],
                   [0, 0, 1, 1, 1, 1, 1, 1, 1, 0],
                   [0, 0, 1, 1, 1, 1, 1, 1, 1, 0],
                   [0, 0, 0, 1, 1, 1, 1, 1, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]))
         # OLD CONVENTION:    
         # np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         #           [0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
         #           [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
         #           [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
         #           [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
         #           [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
         #           [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
         #           [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
         #           [0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
         #           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]))

    aarc(get_circle(PRETTY_ODD, radius=4, mode="val"),
         np.array([1, 1, 1, 1, 1, 1, 2, 2, 2, 1, 1, 2, 3, 2, 1, 1, 2, 2,
                   2, 1, 1, 1, 1, 1, 1]))

    aarc(get_circle(PRETTY_EVEN, radius=4, mode="val"),
         np.array([1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 2, 3, 3, 2, 1,
                   1, 2, 3, 3, 2, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1]))


def test_get_annulus_segments():
    arr = np.ones((10, 10))

    # single segment, like the old get_annulus. Note the ``[0]``.

    res = get_annulus_segments(arr, 2, 3)[0]

    truth = (np.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4,
                       4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7,
                       7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9]),
             np.array([3, 4, 5, 6, 7, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1,
                       2, 3, 7, 8, 9, 1, 2, 3, 7, 8, 9, 1, 2, 3, 7, 8, 9, 1, 2, 3, 4, 5,
                       6, 7, 8, 9, 2, 3, 4, 5, 6, 7, 8, 3, 4, 5, 6, 7]))

    # OLD CONVENTION:
    # truth = (np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3,
    #                    3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6,
    #                    6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9,
    #                    9, 9]),
    #          np.array([3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1,
    #                    2, 3, 6, 7, 8, 9, 0, 1, 2, 7, 8, 9, 0, 1, 2, 7, 8, 9, 0, 1, 2, 3,
    #                    6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 3, 4,
    #                    5, 6]))

    aarc(res, truth)

    res = get_annulus_segments(arr, 2, 3, mode="val")[0]
    truth = np.array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                      1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                      1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                      1., 1., 1., 1., 1., 1., 1., 1., 1.])

    aarc(res, truth)

    # multiple segments:

    res = get_annulus_segments(PRETTY_EVEN, 2, 3, nsegm=2)
    truth = [(np.array([3, 4, 4, 4, 5, 5, 5, 5, 5, 5]),
              np.array([5, 0, 1, 5, 0, 1, 2, 3, 4, 5])),
             (np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 3, 3]),
              np.array([0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 5, 0, 1]))]

    #aarc(res, truth)
    # TODO: cannot compare using `allclose`, as elements have variable length!
    assert repr(res) == repr(truth)

    res = get_annulus_segments(PRETTY_EVEN, 2, 3, nsegm=3)
    truth = [(np.array([3, 4, 5, 5, 5, 5]), np.array([5, 5, 2, 3, 4, 5])),
             (np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]),
              np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1])),
             (np.array([0, 0, 0, 0, 1, 1, 1, 1, 2]), np.array([2, 3, 4, 5, 2, 3, 4, 5, 5]))]

    assert repr(res) == repr(truth)
    # TODO: cannot compare using `allclose`, as elements have variable length!

    # tuple as input:

    res = get_annulus_segments((6, 6), 2, 3, nsegm=3)
    assert repr(res) == repr(truth)

    # masked arr:

    res = get_annulus_segments(arr, 2, 3, mode="mask")[0]
    truth = np.array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                      [0., 0., 0., 1., 1., 1., 1., 1., 0., 0.],
                      [0., 0., 1., 1., 1., 1., 1., 1., 1., 0.],
                      [0., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                      [0., 1., 1., 1., 0., 0., 0., 1., 1., 1.],
                      [0., 1., 1., 1., 0., 0., 0., 1., 1., 1.],
                      [0., 1., 1., 1., 0., 0., 0., 1., 1., 1.],
                      [0., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                      [0., 0., 1., 1., 1., 1., 1., 1., 1., 0.],
                      [0., 0., 0., 1., 1., 1., 1., 1., 0., 0.]])
    aarc(res, truth)

    # tuple as input:

    res = get_annulus_segments((10, 10), 2, 3, mode="mask")[0]
    # masking a zeros array -> only zeros left!
    assert res.sum() == 0


def test_dist():
    assert dist(0, 0, 1, 1) == np.sqrt(2)
    assert dist(1, 2, 3, 4) == 2 * np.sqrt(2)


def test_get_ellipse():
    f = np.ones((6, 10))

    # masked array:
    fem = get_ellipse(f, 4, 2, 90, mode="mask")
    assert fem.sum() == 24  # outer region masked, 28 pixels kept

    # values:
    fev = get_ellipse(f, 4, 2, 90, mode="val")
    assert fev.sum() == 24

    # indices:
    fei = get_ellipse(f, 4, 2, 90, mode="ind")
    assert fei[0].shape == fei[1].shape == (24,)


def test_get_ell_annulus():

    f = np.ones((15, 30))

    fa = get_ell_annulus(f, 8, 3, 90, 6, mode="mask")
    assert fa.sum() == 114


def test_reshape_matrix():
    vectorized_frames = np.array([[1, 1, 1, 2, 2, 2], [1, 2, 3, 4, 5, 6]])
    cube = reshape_matrix(vectorized_frames, 2, 3)

    assert cube.shape == (2, 2, 3)  # 2 frames of 2x3

    cube_truth = np.array([[[1, 1, 1],
                            [2, 2, 2]],

                           [[1, 2, 3],
                            [4, 5, 6]]])

    aarc(cube, cube_truth)


def test_matrix_scaling():
    """
    The "truth" values were verified by hand.
    """
    m = np.array([[6, 12, 18], [0, 0, 12]], dtype=float)

    res = matrix_scaling(m, None)
    truth = m
    aarc(res, truth)

    res = matrix_scaling(m, "temp-mean")
    truth = np.array([[ 3,  6,  3],
                      [-3, -6, -3]])
    aarc(res, truth)

    res = matrix_scaling(m, "spat-mean")
    truth = np.array([[-6,  0,  6],
                      [-4, -4,  8]])
    aarc(res, truth)

    res = matrix_scaling(m, "temp-standard")
    truth = np.array([[ 1,  1,  1],
                      [-1, -1, -1]])
    aarc(res, truth)

    res = matrix_scaling(m, "spat-standard")
    truth = np.array([[-np.sqrt(3/2), 0, np.sqrt(3/2)],
                      [-np.sqrt(1/2), -np.sqrt(1/2), np.sqrt(2)]])
    aarc(res, truth)
