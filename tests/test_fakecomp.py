"""

Tests for `metrics/fakecomp.py`.

"""

from __future__ import division, print_function

import numpy as np
import pytest

import vip_hci as vip

allclose = np.testing.assert_allclose
array = np.array


@pytest.mark.filterwarnings("ignore:invalid value encountered in true_divide")
def test_normalize_psf_shapes():
    """
    Test if normalize_psf produces the expected shapes.
    """
    # `Force_odd` is True therefore `size` was set to 19
    res_even = vip.metrics.normalize_psf(np.ones((20, 20)), size=18)
    res_odd = vip.metrics.normalize_psf(np.ones((21, 21)), size=18)
    assert res_even.shape == res_odd.shape == (19, 19)

    res_even = vip.metrics.normalize_psf(np.ones((20, 20)), size=18,
                                         force_odd=False)
    res_odd = vip.metrics.normalize_psf(np.ones((21, 21)), size=18,
                                        force_odd=False)
    assert res_even.shape == res_odd.shape == (18, 18)

    # set to odd size
    res_even = vip.metrics.normalize_psf(np.ones((20, 20)), size=19)
    res_odd = vip.metrics.normalize_psf(np.ones((21, 21)), size=19)
    assert res_even.shape == res_odd.shape == (19, 19)

    res_even = vip.metrics.normalize_psf(np.ones((20, 20)), size=19,
                                         force_odd=False)
    res_odd = vip.metrics.normalize_psf(np.ones((21, 21)), size=19,
                                        force_odd=False)
    assert res_even.shape == res_odd.shape == (19, 19)
