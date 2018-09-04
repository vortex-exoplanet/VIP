"""
Tests for the post-processing pipeline, using the functional API.

"""

from __future__ import division, print_function

import copy
import numpy as np

import vip_hci as vip

import pytest


def print_debug(s, *args, **kwargs):
    print(("\033[34m" + s + "\033[0m").format(*args, **kwargs))


@pytest.fixture(scope="module")
def injected_cube_position(example_dataset):
    """
    Inject a fake companion into an example cube.

    Parameters
    ----------
    example_dataset : fixture
        Taken automatically from ``conftest.py``.

    Returns
    -------
    dsi : HCIDataset
    injected_position_yx : tuple(y, x)

    """
    print_debug("injecting fake planet...")
    dsi = copy.copy(example_dataset)
    # we chose a shallow copy, as we will not use any in-place operations
    # (like +=). Using `deepcopy` would be safer, but consume more memory.

    dsi.cube, yx = vip.metrics.cube_inject_companions(dsi.cube,
                                                      dsi.psfn,
                                                      dsi.angles,
                                                      flevel=20000,
                                                      plsc=dsi.px_scale,
                                                      rad_dists=30,
                                                      full_output=True,
                                                      verbose=True)
    injected_position_yx = yx[0]  # -> tuple

    return dsi, injected_position_yx


# ====== algos


def algo_medsub(ds):
    return vip.medsub.median_sub(ds.cube, ds.angles, fwhm=ds.fwhm,
                                 mode="fullfr")


def algo_medsub_annular(ds):
    return vip.medsub.median_sub(ds.cube, ds.angles, fwhm=ds.fwhm,
                                 mode="annular")


def algo_xloci(ds):
    """
    Notes
    -----
    Running time:
    - Mac R:     0:01:07.431246
    - IPAG-calc: 0:00:02.574641
    - Travis CI: 0:00:05.652058
    """

    return vip.leastsq.xloci(ds.cube, ds.angles, fwhm=ds.fwhm,
                             radius_int=20)  # <- speed up
                             

def algo_pca(ds):
    return vip.pca.pca(ds.cube, ds.angles)


def algo_pca_annular(ds):
    return vip.pca.pca_annular(ds.cube, ds.angles, fwhm=ds.fwhm)


def algo_andromeda(ds):
    res = vip.andromeda.andromeda(ds.cube, oversampling_fact=1,
                                  angles=-ds.angles, psf=ds.psf)
    # TODO: different angles convention!

    contrast, snr, snr_n, stdcontrast, stdcontrast_n, likelihood, r = res
    return snr_n


def algo_andromeda_fast(ds):
    res = vip.andromeda.andromeda(ds.cube, oversampling_fact=0.5,
                                  fast=10,
                                  angles=-ds.angles, psf=ds.psf)
    # TODO: different angles convention!

    contrast, snr, snr_n, stdcontrast, stdcontrast_n, likelihood, r = res
    return snr_n


# ====== SNR map / detection


def snrmap_fast(frame, ds):
    return vip.metrics.snrmap_fast(frame, fwhm=ds.fwhm)


def snrmap(frame, ds):
    return vip.metrics.snrmap(frame, fwhm=ds.fwhm, mode="sss")


def detect_max(frame, yx_exp, tolerance_percent=4):
    """
    Verify if detected companion matches injected position using `max`.

    Parameters
    ----------
    frame : 2d ndarray
    yx_exp : tuple(y, x)
        Expected position of the fake companion (= injected position).
    tolerance_percent : float, optional
        The recovered position of the fake companion is allowed to be off by
        ``tolerance_percent`` of the frame size. This is mainly needed because
        of the "poor" performance of snrmap_fast (see notes).

    Notes
    -----
    distances from injection:
    - medsub fullframe + snrmap:       1.4142135623730951
    - medsub fullframe + snrmap fast:  2.23606797749979
    - pca fullframe + snrmap fast:     2.23606797749979
    - medsub annular + snrmap fast:    3.16
    - pca annular + snrmap fast:       2.23606797749979

    """
    tolerance = frame.shape[0] / 100 * tolerance_percent

    yx = np.unravel_index(frame.argmax(), frame.shape)

    dist = np.sqrt((yx[0] - yx_exp[0])**2 + (yx[1] - yx_exp[1])**2)

    print_debug("dist: {}", dist)

    assert dist < tolerance, "Detected maximum does not match injection"


@pytest.mark.parametrize("algo, make_detmap",
                         [
                            (algo_medsub, snrmap_fast),
                            (algo_medsub, snrmap),
                            (algo_medsub_annular, snrmap_fast),
                            (algo_xloci, snrmap_fast),
                            (algo_pca, snrmap_fast),
                            (algo_pca_annular, snrmap_fast),
                            (algo_andromeda, None),
                         ],
                         ids=lambda x: (x.__name__.replace("algo_", "")
                                        if callable(x) else x))
def test_algos(injected_cube_position, algo, make_detmap):
    ds, position = injected_cube_position
    frame = algo(ds)

    if make_detmap is not None:
        detmap = make_detmap(frame, ds)
    else:
        detmap = frame

    detect_max(detmap, position)
