"""
Tests for HCIDataset.
"""

__author__ = "Ralf Farkas"

import os
import sys

sys.path.append(".../tests")
from tests.helpers import aarc, np

from vip_hci.objects.dataset import Dataset


def test_saveable_dataset(tmp_path):
    """
    Test the HCIDataset.save() and .load() methods
    """
    # build HCIDataset
    cube = np.zeros((5, 10, 10))
    angles = np.linspace(1, 2, 5)
    fwhm = 4  # test non-numpy type saving/loading

    ds = Dataset(cube=cube, angles=angles, fwhm=fwhm)

    # save
    fn = tmp_path / 'test'
    ds.save(fn)

    # restore
    ds2 = Dataset.load(fn)

    # compare
    aarc(ds2.cube, cube)
    aarc(ds2.angles, angles)
    assert ds2.fwhm == fwhm
