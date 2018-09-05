"""
Tests for var/shapes.py

"""

from __future__ import division, print_function

__author__ = "Ralf Farkas"

import numpy as np
import vip_hci as vip

from helpers import aarc


pretty_odd = np.array([
    [1, 1, 1, 1, 1],
    [1, 2, 2, 2, 1],
    [1, 2, 3, 2, 1],
    [1, 2, 2, 2, 1],
    [1, 1, 1, 1, 1]
])

pretty_even = np.array([
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

    res44 = (1.5, 1.5)
    res55 = (2.0, 2.0)

    # 2D
    assert vip.var.frame_center(np.zeros((4, 4))) == res44
    assert vip.var.frame_center(np.zeros((5, 5))) == res55

    # 3D
    assert vip.var.frame_center(np.zeros((frames, 4, 4))) == res44
    assert vip.var.frame_center(np.zeros((frames, 5, 5))) == res55

    # 4D
    assert vip.var.frame_center(np.zeros((nlambda, frames, 4, 4))) == res44
    assert vip.var.frame_center(np.zeros((nlambda, frames, 5, 5))) == res55


def test_mask_circle():

    size = 5
    radius = 2

    ones = np.ones((size, size))

    # "in" and "out" should be complementary
    res_in = vip.var.mask_circle(ones, radius=radius, mode="in")
    res_out = vip.var.mask_circle(ones, radius=radius, mode="out")
    aarc(res_in+res_out, ones)

    # radius=2 -> central region should be 3x3 pixels = 9 pixels
    aarc(res_out.sum(), 9)


def test_get_square():
    aarc(vip.var.get_square(pretty_odd, size=3, x=2, y=2),
         np.array([[2, 2, 2],
                   [2, 3, 2],
                   [2, 2, 2]]))

    aarc(vip.var.get_square(pretty_odd, size=2, x=2, y=2),
         np.array([[2, 2, 2],
                   [2, 3, 2],
                   [2, 2, 2]]))
    # -> prints warning

    aarc(vip.var.get_square(pretty_odd, size=2, x=2, y=2, force=True),
         np.array([[2, 2],
                   [2, 3]]))

    aarc(vip.var.get_square(pretty_even, size=2, x=3, y=3),
         np.array([[3, 3],
                  [3, 3]]))

    aarc(vip.var.get_square(pretty_even, size=3, x=3, y=3),
         np.array([[2, 2, 2, 2],
                   [2, 3, 3, 2],
                   [2, 3, 3, 2],
                   [2, 2, 2, 2]]))
    # -> prints warning

    aarc(vip.var.get_square(pretty_even, size=2, x=4, y=2, force=True),
         np.array([[2, 2],
                   [3, 2]]))


def test_get_circle():

    ar = np.ones((10, 10), dtype=int)
    aarc(vip.var.get_circle(ar, radius=4, output_values=False),
         np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
                   [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
                   [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                   [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                   [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                   [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                   [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
                   [0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]))

    aarc(vip.var.get_circle(pretty_odd, radius=4, output_values=True),
         np.array([1, 1, 1, 1, 1, 1, 2, 2, 2, 1, 1, 2, 3, 2, 1, 1, 2, 2,
                   2, 1, 1, 1, 1, 1, 1]))

    aarc(vip.var.get_circle(pretty_even, radius=4, output_values=True),
         np.array([1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 2, 3, 3, 2, 1,
                   1, 2, 3, 3, 2, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1]))


def test_get_annulus_segments():
    arr = np.ones((10, 10))

    # single segment, like the old get_annulus. Note the ``[0]``.

    res = vip.var.get_annulus_segments(arr, 2, 3)[0]

    truth = (np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3,
                       3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6,
                       6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9,
                       9, 9]),
             np.array([3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1,
                       2, 3, 6, 7, 8, 9, 0, 1, 2, 7, 8, 9, 0, 1, 2, 7, 8, 9, 0, 1, 2, 3,
                       6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 3, 4,
                       5, 6]))

    aarc(res, truth)

    res = vip.var.get_annulus_segments(arr, 2, 3, mode="val")[0]
    truth = np.array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                      1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                      1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                      1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])

    aarc(res, truth)

    # multiple segments:

    res = vip.var.get_annulus_segments(pretty_even, 2, 3, nsegm=2)
    truth = [(np.array([0, 0, 0, 1, 1, 2, 3, 4, 4, 5, 5, 5]),
              np.array([3, 4, 5, 4, 5, 5, 5, 4, 5, 3, 4, 5])),
             (np.array([0, 0, 0, 1, 1, 2, 3, 4, 4, 5, 5, 5]),
              np.array([0, 1, 2, 0, 1, 0, 0, 0, 1, 0, 1, 2]))]

    aarc(res, truth)

    res = vip.var.get_annulus_segments(pretty_even, 2, 3, nsegm=3)
    truth = [(np.array([2, 3, 4, 4, 5, 5, 5]),
              np.array([5, 5, 4, 5, 3, 4, 5])),
             (np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1]),
              np.array([0, 1, 2, 3, 4, 5, 0, 1, 4, 5])),
             (np.array([2, 3, 4, 4, 5, 5, 5]),
              np.array([0, 0, 0, 1, 0, 1, 2]))]

    assert repr(res) == repr(truth)
    # TODO: cannot compare using `allclose`, as elements have variable length!

    # tuple as input:

    res = vip.var.get_annulus_segments((6, 6), 2, 3, nsegm=3)
    assert repr(res) == repr(truth)

    # masked arr:

    res = vip.var.get_annulus_segments(arr, 2, 3, mode="mask")[0]
    truth = np.array([[0., 0., 0., 1., 1., 1., 1., 0., 0., 0.],
                      [0., 1., 1., 1., 1., 1., 1., 1., 1., 0.],
                      [0., 1., 1., 1., 1., 1., 1., 1., 1., 0.],
                      [1., 1., 1., 1., 0., 0., 1., 1., 1., 1.],
                      [1., 1., 1., 0., 0., 0., 0., 1., 1., 1.],
                      [1., 1., 1., 0., 0., 0., 0., 1., 1., 1.],
                      [1., 1., 1., 1., 0., 0., 1., 1., 1., 1.],
                      [0., 1., 1., 1., 1., 1., 1., 1., 1., 0.],
                      [0., 1., 1., 1., 1., 1., 1., 1., 1., 0.],
                      [0., 0., 0., 1., 1., 1., 1., 0., 0., 0.]])
    aarc(res, truth)

    # tuple as input:

    res = vip.var.get_annulus_segments((10, 10), 2, 3, mode="mask")[0]
    # masking a zeros array -> only zeros left!
    assert res.sum() == 0


def test_dist():
    assert vip.var.dist(0, 0, 1, 1) == np.sqrt(2)
    assert vip.var.dist(1, 2, 3, 4) == 2 * np.sqrt(2)


def test_get_ellipse():
    f = np.ones((6, 10))

    # masked array:
    fem = vip.var.get_ellipse(f, 4, 2, 90, mode="mask")
    assert fem.sum() == 28  # outer region masked, 28 pixels kept

    # values:
    fev = vip.var.get_ellipse(f, 4, 2, 90, mode="val")
    assert fev.sum() == 28

    # indices:
    fei = vip.var.get_ellipse(f, 4, 2, 90, mode="ind")
    assert fei[0].shape == fei[1].shape == (28,)
