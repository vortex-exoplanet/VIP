"""Test for the PostProc object dedicated to frame differencing."""

__author__ = "Thomas Bédrine"

import copy

import numpy as np
from dataclass_builder import update

import sys

sys.path.append(".../tests")
from tests.helpers import fixture, check_detection
from vip_hci.objects import FrameDiffBuilder
from tests.snapshots.snapshot_psfsub import PSFADI_PATH


# Note : this function comes from the former test for adi psfsub, I did not write it,
# and I didn't found the author (feel free to add the author if you know them)
@fixture(scope="module")
def injected_cube_position_adi(example_dataset_adi):
    """
    Inject a fake companion into an example cube.

    Parameters
    ----------
    example_dataset_adi : fixture
        Taken automatically from ``conftest.py``.

    Returns
    -------
    dsi : VIP Dataset
    injected_position_yx : tuple(y, x)

    """
    print("injecting fake planet...")
    dsi = copy.copy(example_dataset_adi)
    # we chose a shallow copy, as we will not use any in-place operations
    # (like +=). Using `deepcopy` would be safer, but consume more memory.

    dsi.inject_companions(300, rad_dists=30)

    return dsi


def test_frame_diff_object(injected_cube_position_adi):
    """
    Compare frames obtained through procedural and object versions of frame diff.

    Generate a frame with ``vip_hci.objects.ppframediff`` and ensure they match with
    their procedural counterpart. This is done by getting the snapshot of the
    ``vip_hci.psfsub.frame_diff`` function, generated preemptively with
    ``tests..snapshots.save_snapshots_psfsub``.

    """
    betapic = injected_cube_position_adi

    fwhm = betapic.fwhm
    imlib_rot = "vip-fft"
    interpolation = None

    # Testing frame differencing

    position = np.load(f"{PSFADI_PATH}framediff_adi_detect.npy")
    exp_frame = np.load(f"{PSFADI_PATH}framediff_adi.npy")

    framediff_obj = FrameDiffBuilder(
        dataset=betapic,
        metric="l1",
        dist_threshold=90,
        delta_rot=0.5,
        radius_int=4,
        asize=fwhm,
        nproc=None,
        imlib=imlib_rot,
        interpolation=interpolation,
        verbose=False,
    ).build()

    framediff_obj.run()
    framediff_obj.make_snrmap()

    check_detection(framediff_obj.snr_map, position, betapic.fwhm, snr_thresh=2)
    assert np.allclose(
        np.abs(exp_frame), np.abs(framediff_obj.frame_final), atol=1.0e-2
    )

    # Testing frame differencing (with a median a 4 similar frames)

    position = np.load(f"{PSFADI_PATH}framediff4_adi_detect.npy")
    exp_frame = np.load(f"{PSFADI_PATH}framediff4_adi.npy")

    update(framediff_obj, FrameDiffBuilder(n_similar=4))

    framediff_obj.run()
    framediff_obj.make_snrmap()

    check_detection(framediff_obj.snr_map, position, betapic.fwhm, snr_thresh=2)
    assert np.allclose(
        np.abs(exp_frame), np.abs(framediff_obj.frame_final), atol=1.0e-2
    )
