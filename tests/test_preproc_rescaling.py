"""
Tests for preproc/rescaling.py

"""

from __future__ import division, print_function

__author__ = "Ralf Farkas"

import numpy as np
from vip_hci.preproc.rescaling import _find_indices_sdi

from helpers import aarc
from pytest import mark
parametrize = mark.parametrize


@parametrize("dist, index_ref, truth",
             [
                (6, 0, [2, 3, 4, 5, 6, 7, 8, 9]),
                (6, 5, [0, 1, 2]),
                (10, 0, [1, 2, 3, 4, 5, 6, 7, 8, 9]),
                (10, 5, [0, 1, 2, 3, 9]),
                (15, 5, [0, 1, 2, 3, 8, 9]),
                (20, 0, [1, 2, 3, 4, 5, 6, 7, 8, 9]),
                (20, 5, [0, 1, 2, 3, 4, 7, 8, 9])
             ])
def test_find_indices_sdi(dist, index_ref, truth):
    wl = np.arange(10) + 1
    fwhm = 4

    indices = _find_indices_sdi(wl, dist, index_ref, fwhm, debug=True)

    aarc(indices, truth)
