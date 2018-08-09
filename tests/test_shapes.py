"""
Tests for var/shapes.py

"""

from __future__ import division, print_function

__author__ = "Ralf Farkas"

import numpy as np
import vip_hci as vip

from helpers import aarc


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
