"""
Tests for metrics/snr.py

"""

import copy
from .helpers import fixture, np
from vip_hci.psfsub import pca
from vip_hci.metrics import detection


@fixture(scope="module")
def get_frame_snrmap(example_dataset_adi):
    """
    Inject a fake companion into an example cube.

    Parameters
    ----------
    example_dataset_adi : fixture
        Taken automatically from ``conftest.py``.

    Returns
    -------
    frame : VIP Frame
    planet_position : tuple(y, x)

    """
    dsi = copy.copy(example_dataset_adi)
    # we chose a shallow copy, as we will not use any in-place operations
    # (like +=). Using `deepcopy` would be safer, but consume more memory.

    print("producing a final frame...")
    res_frame = pca(dsi.cube, dsi.angles, ncomp=10)
    return res_frame, (63, 63), dsi.fwhm


def test_detection_log(get_frame_snrmap):
    res_frame, coord, fwhm = get_frame_snrmap
    table = detection(res_frame, fwhm, psf=None, mode='log', plot=False)
    check = False
    for i in range(len(table.y)):
        if np.allclose(table.y[i], coord[0], atol=2) and \
           np.allclose(table.x[i], coord[1], atol=2):
            check = True
    assert check


def test_detection_dog(get_frame_snrmap):
    res_frame, coord, fwhm = get_frame_snrmap
    table = detection(res_frame, fwhm, psf=None, mode='dog', plot=False)
    check = False
    for i in range(len(table.y)):
        if np.allclose(table.y[i], coord[0], atol=2) and \
           np.allclose(table.x[i], coord[1], atol=2):
            check = True
    assert check


def test_detection_lpeaks(get_frame_snrmap):
    res_frame, coord, fwhm = get_frame_snrmap
    table = detection(res_frame, fwhm, psf=None, mode='lpeaks', plot=False)
    check = False
    for i in range(len(table.y)):
        if np.allclose(table.y[i], coord[0], atol=2) and \
           np.allclose(table.x[i], coord[1], atol=2):
            check = True
    assert check


def test_detection_snrmap(get_frame_snrmap):
    res_frame, coord, fwhm = get_frame_snrmap
    table = detection(res_frame, fwhm, psf=None, mode='snrmapf',
                      plot=False, snr_thresh=5, nproc=2)
    check = False
    for i in range(len(table.y)):
        if np.allclose(table.y[i], coord[0], atol=2) and \
           np.allclose(table.x[i], coord[1], atol=2):
            check = True
    assert check
