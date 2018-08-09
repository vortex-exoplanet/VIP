"""
Tests for var/fit_2d.py.

"""

from __future__ import division, print_function

import numpy as np
import pytest

from vip_hci.var.fit_2d import (create_synth_psf,
                                fit_2dgaussian, fit_2dmoffat, fit_2dairydisk)


def aarc(actual, desired, rtol=1e-5, atol=1e-6):
    __tracebackhide__ = True  # Hide traceback for pytest
    np.testing.assert_allclose(actual, desired, rtol=rtol, atol=atol)


def put(big_frame, small_frame, y, x):
    big_frame = big_frame.copy()
    h, w = small_frame.shape

    big_frame[y:y+h, x:x+w] = small_frame

    return big_frame


def put_center(big_frame, small_frame, y, x):
    offs_y = small_frame.shape[0]//2
    offs_x = small_frame.shape[1]//2

    return put(big_frame, small_frame, y-offs_y, x-offs_x)


@pytest.mark.parametrize("psfsize", [6, 7], ids=lambda x: "psf{}".format(x))
@pytest.mark.parametrize("framesize", [10, 11], ids=lambda x: "fr{}".format(x))
@pytest.mark.parametrize("y,x", [(4, 4), (4, 6), (6, 5)])
@pytest.mark.parametrize("psf_model, fit_fkt",
                         [
                            pytest.param("gauss", fit_2dgaussian, id="gauss"),
                            pytest.param("moff", fit_2dmoffat, id="moff"),
                            pytest.param("airy", fit_2dairydisk, id="airy")
                         ])
def test_fit2d(psf_model, fit_fkt, y, x, framesize, psfsize):
    frame = np.zeros((framesize, framesize))
    psf = create_synth_psf(psf_model, shape=(psfsize, psfsize))

    inj_frame = put_center(frame, psf, y, x)

    y_out, x_out = fit_fkt(inj_frame)

    # correct "half-pixel centering", to make output of fit_2d* comparable
    # with `put`.
    if (
        (framesize % 2 == 0 and psfsize % 2 == 0) or
        (framesize % 2 == 1 and psfsize % 2 == 0)
       ):
        y_exp = y - 0.5
        x_exp = x - 0.5
    else:
        y_exp = y
        x_exp = x

    yx_real = np.unravel_index(inj_frame.argmax(), inj_frame.shape)
    print("demanded injection:   {}".format((y, x)))
    print("brightes pixel:       {}".format(yx_real))
    print("fit should return:    {}".format((y_exp, x_exp)))
    print("fitted injection:     {}".format((y_out, x_out)))

    aarc((y_out, x_out), (y_exp, x_exp), atol=0.05)
