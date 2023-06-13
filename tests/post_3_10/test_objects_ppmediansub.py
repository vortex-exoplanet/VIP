"""Test for the PostProc object dedicated to median subtraction."""

__author__ = "Thomas Bédrine"

import copy

import numpy as np
from dataclass_builder import update

import sys

sys.path.append(".../tests")
from tests.helpers import fixture, check_detection
from vip_hci.objects import MedianBuilder
from tests.snapshots.snapshot_psfsub import PSFADI_PATH


# Note : this function comes from the former test for adi psfsub, I did not write it,
# and I didn't found the author (feel free to add the author if you know them)
@fixture(scope="module")
def injected_cube_position(example_dataset_adi):
    """
    Inject a fake companion into an example cube.

    Parameters
    ----------
    example_dataset_adi : fixture
        Taken automatically from ``conftest.py``.

    Returns
    -------
    dsi : VIP Dataset

    """
    print("injecting fake planet...")
    dsi = copy.copy(example_dataset_adi)
    # we chose a shallow copy, as we will not use any in-place operations
    # (like +=). Using `deepcopy` would be safer, but consume more memory.

    dsi.inject_companions(300, rad_dists=30)

    return dsi


def test_median_object(injected_cube_position):
    """
    Compare frames obtained through procedural and object versions of median sub.

    Generate a frame with ``vip_hci.objects.ppmediansub`` and ensure they match with
    their procedural counterpart. This is done by getting the snapshot of the
    ``vip_hci.psfsub.median_sub`` function, generated preemptively with
    ``tests..snapshots.snapshot_psfsub``.

    """
    betapic = injected_cube_position

    imlib_rot = "vip-fft"
    interpolation = None

    # Testing the full frame version of median subtraction

    position = np.load(f"{PSFADI_PATH}medsub_adi_detect.npy")
    exp_frame = np.load(f"{PSFADI_PATH}medsub_adi.npy")

    medsub_obj = MedianBuilder(
        dataset=betapic,
        mode="fullfr",
        imlib=imlib_rot,
        interpolation=interpolation,
        verbose=False,
    ).build()

    medsub_obj.run()
    medsub_obj.make_snrmap()

    check_detection(medsub_obj.snr_map, position, betapic.fwhm, snr_thresh=2)
    assert np.allclose(np.abs(medsub_obj.frame_final), np.abs(exp_frame), atol=1e-2)

    # Testing the annular version of median subtraction

    position = np.load(f"{PSFADI_PATH}medsub_ann_adi_detect.npy")
    exp_frame = np.load(f"{PSFADI_PATH}medsub_ann_adi.npy")

    update(medsub_obj, MedianBuilder(mode="annular"))

    medsub_obj.run()
    medsub_obj.make_snrmap()

    check_detection(medsub_obj.snr_map, position, betapic.fwhm, snr_thresh=2)
    assert np.allclose(np.abs(medsub_obj.frame_final), np.abs(exp_frame), atol=1e-2)
