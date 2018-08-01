from __future__ import division, print_function

import numpy as np

import vip_hci as vip
from vip_hci.metrics.fakecomp import cube_inject_companions


# ===== utility functions

def aarc(actual, desired, rtol=1e-5, atol=1e-6):
    """
    Assert array-compare. Like ``np.allclose``, but with different defaults.

    Notes
    -----
    Default values for
    - ``np.allclose``: ``atol=1e-8, rtol=1e-5``
    - ``np.testing.assert_allclose``: ``atol=0, rtol=1e-7``

    IDL's `FLOAT` is 32bit (precision of ~1e-6 to ~1e-7), while python uses
    64-bit floats internally. To pass the comparison, `atol` has to be chosen
    accordingly. The contribution of `rtol` is dominant for large numbers, where
    an absolute comparison to `1e-6` would not make sense.

    """
    __tracebackhide__ = True  # Hide traceback for pytest
    np.testing.assert_allclose(actual, desired, rtol=rtol, atol=atol)


def test_cube_inject_companions():
    cube = np.zeros((3, 5, 5))
    psf = np.ones((1, 1))
    angles = np.array([0, 90, 180])

    cube4 = np.zeros((2, 3, 5, 5))
    psf4 = np.ones((2, 1, 1))

    # single injection:
    c, yx = cube_inject_companions(cube, psf_template=psf, angle_list=angles,
                                   flevel=3, rad_dists=2, n_branches=1,
                                   full_output=True,
                                   plsc=1, verbose=True)

    c_ref = np.array([[[0., 0., 0., 0., 0.],
                       [0., 0., 0., 0., 0.],
                       [0., 0., 0., 0., 3.],
                       [0., 0., 0., 0., 0.],
                       [0., 0., 0., 0., 0.]],

                      [[0., 0., 3., 0., 0.],
                       [0., 0., 0., 0., 0.],
                       [0., 0., 0., 0., 0.],
                       [0., 0., 0., 0., 0.],
                       [0., 0., 0., 0., 0.]],

                      [[0., 0., 0., 0., 0.],
                       [0., 0., 0., 0., 0.],
                       [3., 0., 0., 0., 0.],
                       [0., 0., 0., 0., 0.],
                       [0., 0., 0., 0., 0.]]])

    aarc(c, c_ref)
    aarc(yx, [(2, 4)])

    # multiple injections:
    c, yx = cube_inject_companions(cube, psf_template=psf, angle_list=angles,
                                   flevel=3, rad_dists=[1, 2], n_branches=2,
                                   full_output=True,
                                   plsc=1, verbose=True)

    aarc(yx, [(2, 3), (2, 4), (2, 1), (2, 0)])

    # 4D case:
    c, yx = cube_inject_companions(cube4, psf_template=psf4, angle_list=angles,
                                   flevel=3, rad_dists=[1, 2], n_branches=2,
                                   full_output=True,
                                   plsc=1, verbose=True)

    aarc(yx, [(2, 3), (2, 4), (2, 1), (2, 0)])
