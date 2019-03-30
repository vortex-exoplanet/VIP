"""
Tests for the post-processing pipeline, using the functional API.

"""

__author__ = "Carlos Alberto Gomez Gonzalez"

import copy
import vip_hci as vip
from .helpers import np, parametrize, fixture


def print_debug(s, *args, **kwargs):
    print(("\033[34m" + s + "\033[0m").format(*args, **kwargs))


@fixture(scope="module")
def injected_cube_position(example_dataset_ifs):
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
    dsi = copy.copy(example_dataset_ifs)
    # we chose a shallow copy, as we will not use any in-place operations
    # (like +=). Using `deepcopy` would be safer, but consume more memory.

    dsi.cube, yx = vip.metrics.cube_inject_companions(dsi.cube,
                                                      dsi.psfn,
                                                      dsi.angles,
                                                      flevel=100,
                                                      plsc=dsi.px_scale,
                                                      rad_dists=30,
                                                      full_output=True,
                                                      verbose=True)
    injected_position_yx = yx[0]  # -> tuple

    return dsi, injected_position_yx


# ====== algos


def algo_medsub(ds):
    return vip.medsub.median_sub(ds.cube, ds.angles, fwhm=ds.fwhm,
                                 scale_list=ds.wavelengths)


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
                             scale_list=ds.wavelengths,
                             radius_int=20)  # <- speed up


def algo_pca(ds):
    return vip.pca.pca(ds.cube, ds.angles, scale_list=ds.wavelengths)


# ====== SNR map
def snrmap_fast(frame, ds):
    return vip.metrics.snrmap_fast(frame, fwhm=np.mean(ds.fwhm))


def snrmap(frame, ds):
    return vip.metrics.snrmap(frame, fwhm=np.mean(ds.fwhm), mode="sss")


# ====== Detection with ``vip_hci.metrics.detection``, by default with a
# location error or 3px
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
        for coor in zip(vectory, vectorx):
            print(coor, exp_yx)
            if np.allclose(coor[0], exp_yx[0], atol=deltapix) and \
                    np.allclose(coor[1], exp_yx[1], atol=deltapix):
                return True
        return False

    table = vip.metrics.detection(frame, fwhm=fwhm, mode='lpeaks', bkg_sigma=5,
                                  matched_filter=False, mask=True,
                                  snr_thresh=snr_thresh, plot=False,
                                  debug=False, full_output=True, verbose=True)
    msg = "Injected companion not recovered"
    assert verify_expcoord(table.y, table.x, yx_exp), msg


@parametrize("algo, make_detmap",
    [
        (algo_medsub, snrmap_fast),
        (algo_xloci, snrmap_fast),
        (algo_pca, snrmap_fast),

    ],
    ids=lambda x: (x.__name__.replace("algo_", "") if callable(x) else x))
def test_algos(injected_cube_position, algo, make_detmap):
    ds, position = injected_cube_position
    frame = algo(ds)

    if make_detmap is not None:
        detmap = make_detmap(frame, ds)
    else:
        detmap = frame

    check_detection(detmap, position, np.mean(ds.fwhm), snr_thresh=2)
