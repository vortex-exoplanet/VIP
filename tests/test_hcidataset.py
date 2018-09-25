"""
Tests for HCIDataset.
"""

from __future__ import division, print_function, absolute_import

__author__ = "Ralf Farkas"

import tempfile
import os
from helpers import aarc, np

from vip_hci.hci_dataset import HCIDataset


def test_saveable_dataset():
    """
    Test the HCIDataset.save() and .load() methods
    """
    # build HCIDataset
    cube = np.zeros((5, 10, 10))
    angles = np.linspace(1, 2, 5)
    fwhm = 4  # test non-numpy type saving/loading

    ds = HCIDataset(cube=cube, angles=angles, fwhm=fwhm)

    # save
    fd, fn = tempfile.mkstemp(prefix="vip_")
    ds.save(fn)

    # restore
    ds2 = HCIDataset.load(fn)

    # compare
    aarc(ds2.cube, cube)
    aarc(ds2.angles, angles)
    assert ds2.fwhm == fwhm

    # cleanup
    os.remove(fn)
