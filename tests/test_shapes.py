from __future__ import division, print_function

import numpy as np
import pytest

import vip_hci as vip

array = np.array


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
