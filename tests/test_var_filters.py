"""
Tests for var/filters.py

The tests for the filtering functions (test_highpass, test_lowpass and
test_iuwt) do NOT verify the actual filtering, as the results produced by the
various filtering algorithms are very diverse. Instead, it is ONLY tested IF the
algorithms actually run.

"""


__author__ = "Ralf Farkas"

from .helpers import aarc, np, param, parametrize
from vip_hci.var.filters import (fft, ifft,
                                 cube_filter_iuwt,
                                 cube_filter_highpass,
                                 cube_filter_lowpass,
                                 frame_filter_highpass,
                                 frame_filter_lowpass)


CUBE = np.ones((5, 10, 10), dtype=float)
FRAME = np.arange(100, dtype=float).reshape((10, 10))


@parametrize("filter_mode",
             [
                 "laplacian",
                 "laplacian-conv",
                 "median-subt",
                 "gauss-subt",
                 "fourier-butter",
                 "hann"
             ]
             )
@parametrize("data, fkt",
             [
                 param(CUBE, cube_filter_highpass, id="cube"),
                 param(FRAME, frame_filter_highpass, id="frame")
             ],
             ids=lambda x: (x.__name__ if callable(x) else None)
             )
def test_highpass(data, fkt, filter_mode):
    res = fkt(data, mode=filter_mode)
    assert res.shape == data.shape


@parametrize("filter_mode", ["median", "gauss"])
@parametrize("data, fkt",
             [
                 param(CUBE, cube_filter_lowpass, id="cube"),
                 param(FRAME, frame_filter_lowpass, id="frame")
             ],
             ids=lambda x: (x.__name__ if callable(x) else None)
             )
def test_lowpass(data, fkt, filter_mode):
    res = fkt(data, mode=filter_mode)
    assert res.shape == data.shape


def test_iuwt():
    res = cube_filter_iuwt(CUBE)
    assert res.shape == CUBE.shape


def test_fft_ifft():
    global FRAME
    FRAME = np.arange(100).reshape((10, 10))

    res = ifft(fft(FRAME))
    aarc(res, FRAME)
