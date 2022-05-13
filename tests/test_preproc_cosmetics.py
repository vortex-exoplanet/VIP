"""
Tests for the bad pixel detection and correction routines.

"""

import numpy as np
from vip_hci.preproc import cube_correct_nan


# NaN correction
def test_nan_corr():
    sz = (24, 24)
    idx0 = 10
    idx1 = 20
    m0 = 0
    m1 = 2
    s0 = 1
    s1 = 1

    im1 = np.random.normal(loc=m0, scale=s0, size=sz)
    im2 = im1 = np.random.normal(loc=m1, scale=s1, size=sz)
    im1[idx0, idx0] = np.nan
    im1[idx0+1, idx0] = np.nan
    im2[idx1, idx1] = np.nan
    im2[idx1+1, idx1] = np.nan
    cube = np.array([im1, im2])
    cube_c = cube_correct_nan(cube, verbose=True)

    # check NaNs were appropriately corrected
    assert np.abs(cube_c[0, idx0, idx0]-m0) < 4*s0
    assert np.abs(cube_c[1, idx1, idx1]-m1) < 4*s1

    # Test half res y
    cube_c = cube_correct_nan(cube, nproc=2, half_res_y=True, verbose=True)
    assert np.abs(cube_c[0, idx0, idx0]-m0) < 4*s0
    assert np.abs(cube_c[1, idx1, idx1]-m1) < 4*s1
