"""
Tests for metrics/snr.py

"""

import copy
from .helpers import fixture, np
from vip_hci.psfsub import pca
from vip_hci.hci_dataset import Frame
from vip_hci.metrics import snrmap, frame_report


@fixture(scope="module")
def get_frame(example_dataset_adi):
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
    frame = Frame(res_frame, fwhm=dsi.fwhm)
    return frame, (63, 63)


atol = 2
plot = False


def test_snrmap_sss(get_frame):
    frame, positions = get_frame
    y0, x0 = positions
    snmap = snrmap(frame.data, fwhm=frame.fwhm, plot=plot, nproc=2)
    y1, x1 = np.where(snmap == snmap.max())
    assert np.allclose(x1, x0, atol=atol) and np.allclose(y1, y0, atol=atol)


def test_snrmap_masked(get_frame):
    frame, positions = get_frame
    y0, x0 = positions
    snmap = snrmap(frame.data, fwhm=frame.fwhm, plot=plot, nproc=2,
                   known_sources=(int(x0), int(y0)))
    y1, x1 = np.where(snmap == snmap.max())
    assert np.allclose(x1, x0, atol=atol) and np.allclose(y1, y0, atol=atol)


def test_snrmap_fast(get_frame):
    frame, positions = get_frame
    y0, x0 = positions
    snmap = snrmap(frame.data, fwhm=frame.fwhm, plot=plot, approximated=True,
                   nproc=2)
    y1, x1 = np.where(snmap == snmap.max())
    assert np.allclose(x1, x0, atol=atol) and np.allclose(y1, y0, atol=atol)

        
def test_frame_report(get_frame):
    frame, positions = get_frame
    y0, x0 = positions
    # getting the snr of the max pixel
    source_xy, _, _, meansnr = frame_report(frame.data, frame.fwhm,
                                            source_xy=None, verbose=True)
    x1 = source_xy[0]
    y1 = source_xy[1]
    assert np.allclose(x1, x0, atol=atol) and np.allclose(y1, y0, atol=atol)

    assert meansnr > 5
