"""Helper functions for tests"""

__author__ = "Ralf Farkas, Thomas Bédrine"

__all__ = ["check_detection", "download_resource"]

from requests.exceptions import ReadTimeout
from pytest import mark, param, raises, fixture
from ratelimit import limits, sleep_and_retry
from astropy.utils.data import download_file
import vip_hci as vip
import numpy as np

filterwarnings = mark.filterwarnings
parametrize = mark.parametrize


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


def check_detection(frame, yx_exp, fwhm, snr_thresh, deltapix=3):
    """
    Verify if injected companion is recovered.

    Parameters
    ----------
    frame : 2d ndarray
    yx_exp : tuple(y, x)
        Expected position of the fake companion (= injected position).
    fwhm : int or float
        FWHM.
    snr_thresh : int or float, optional
        S/N threshold.
    deltapix : int or float, optional
        Error margin in pixels, between the expected position and the recovered.

    """

    def verify_expcoord(vectory, vectorx, exp_yx):
        return any(
            np.allclose(coor[0], expec[0], atol=deltapix)
            and np.allclose(coor[1], expec[1], atol=deltapix)
            for coor in zip(vectory, vectorx)
            for expec in exp_yx
        )

    table = vip.metrics.detection(
        frame,
        fwhm=fwhm,
        mode="lpeaks",
        bkg_sigma=5,
        matched_filter=False,
        mask=True,
        snr_thresh=snr_thresh,
        plot=False,
        debug=True,
        full_output=True,
        verbose=True,
    )
    msg = "Injected companion not recovered"
    assert verify_expcoord(table.y, table.x, yx_exp), msg


@sleep_and_retry
@limits(calls=1, period=1)
def download_resource(url):
    attempts = 5
    while attempts > 0:
        try:
            return download_file(url, cache=True)
        except ReadTimeout:
            attempts -= 1

    raise TimeoutError("Resource could not be accessed due to too many timeouts.")
