"""
Tests for the conf submodule.
"""

__author__ = "Ralf Farkas"

from vip_hci.conf.mem import check_enough_memory


def test_check_enough_memory():
    assert check_enough_memory(0, factor=1) is True

    # there shouldn't be 10Tb of free memory on a normal machine...
    assert check_enough_memory(10*1e12) is False
