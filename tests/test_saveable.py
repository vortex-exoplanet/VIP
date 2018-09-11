from __future__ import print_function, division

import tempfile
import os

import numpy as np
import vip_hci as vip


def test_saveable_dataset():
    cube = np.zeros((5, 10, 10))
    angles = np.linspace(1, 2, 5)
    fwhm = 4  # test non-numpy type saving/loading

    ds = vip.HCIDataset(cube=cube, angles=angles, fwhm=fwhm)

    fd, fn = tempfile.mkstemp(prefix="vip_")

    ds.save(fn)

    ds2 = vip.HCIDataset.load(fn)

    assert np.allclose(ds2.cube, cube)
    assert np.allclose(ds2.angles, angles)
    assert ds2.fwhm == fwhm

    os.remove(fn)
