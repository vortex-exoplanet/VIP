"""Test for the PostProc object dedicated to LLSG."""

__author__ = "Thomas Bédrine"

import copy
import numpy as np

from .helpers import fixture, check_detection
from vip_hci.objects import LLSGBuilder
from .snapshots.snapshot_psfsub import PSFADI_PATH


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


def test_llsg_object(injected_cube_position):
    """
    Compare frames obtained through procedural and object versions of LLSG.

    Generate a frame with ``vip_hci.objects.ppllsg`` and ensure they match with
    their procedural counterpart. This is done by getting the snapshot of the
    ``vip_hci.psfsub.llsg`` function, generated preemptively with
    ``tests.snapshots.save_snapshots_psfsub``.

    """
    betapic = injected_cube_position

    imlib_rot = "vip-fft"
    interpolation = None

    # Testing LLSG

    position = np.load(f"{PSFADI_PATH}llsg_adi_detect.npy")
    exp_frame = np.load(f"{PSFADI_PATH}llsg_adi.npy")

    llsg_obj = LLSGBuilder(
        dataset=betapic,
        rank=5,
        thresh=1,
        max_iter=20,
        random_seed=10,
        verbose=False,
    ).build()

    llsg_obj.run(
        imlib=imlib_rot,
        interpolation=interpolation,
    )
    llsg_obj.make_snrmap()

    check_detection(llsg_obj.snr_map, position, betapic.fwhm, snr_thresh=2)
    assert np.allclose(np.abs(exp_frame), np.abs(llsg_obj.frame_final), atol=1.0e-2)
