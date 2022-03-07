"""
Tests for the conf submodule.
"""

import numpy as np
from vip_hci.config.mem import check_enough_memory
from vip_hci.config.utils_conf import check_array


def test_check_enough_memory():
    assert check_enough_memory(0, factor=1, raise_error=False) is True

    assert check_enough_memory(1e13, factor=1, raise_error=False) is False


def test_check_array():
    a1 = np.zeros((1))
    a2 = np.zeros((1, 1))
    a3 = np.zeros((1, 1, 1))
    a4 = np.zeros((1, 1, 1, 1))

    try:
        check_array([2, 3], dim=1)
    except TypeError:
        raise

    try:
        check_array((2, 3), dim=1)
    except TypeError:
        raise

    try:
        check_array(a1, dim=1)
    except TypeError:
        raise

    try:
        check_array(a2, dim=2)
    except TypeError:
        raise

    try:
        check_array(a3, dim=3)
    except TypeError:
        raise

    try:
        check_array(a4, dim=4)
    except TypeError:
        raise

    try:
        check_array(a4, dim=(3, 4))
    except TypeError:
        raise

    try:
        check_array(a4, dim=(2, 3, 4))
    except TypeError:
        raise

    try:
        check_array(a1, dim=2)
    except TypeError:
        pass

    try:
        check_array(a2, dim=3)
    except TypeError:
        pass

    try:
        check_array(a3, dim=4)
    except TypeError:
        pass

    try:
        check_array([2, 3], dim=2)
    except TypeError:
        pass

