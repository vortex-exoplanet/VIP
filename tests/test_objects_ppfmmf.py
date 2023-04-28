"""Test for the PostProc object dedicated to FMMF."""

__author__ = "Thomas Bédrine"

import copy

import numpy as np
from dataclass_builder import update

from .helpers import fixture, check_detection
from vip_hci.objects import FMMFBuilder
from .snapshots.snapshot_invprob import INVADI_PATH


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

    dsi.inject_companions(300, rad_dists=30, verbose=False)

    return dsi


def test_fmmf_object(injected_cube_position):
    """
    Compare frames obtained through procedural and object versions of FMMF.

    Generate a frame and a S/N map with ``vip_hci.objects.ppfmmf`` and ensure they
    match with their procedural counterpart. This is done by getting the snapshot of the
    ``vip_hci.invprob.fmmf`` function, generated preemptively with
    ``tests.snapshots.snapshot_invprob``. Tests both 'KLIP' and 'LOCI' models.

    """
    betapic = injected_cube_position

    imlib_rot = "opencv"

    # Testing FMMF (with the KLIP model)

    position = np.load(f"{INVADI_PATH}fmmf_kl_adi_detect.npy")
    exp_frame = np.load(f"{INVADI_PATH}fmmf_kl_adi.npy")
    exp_snr = np.load(f"{INVADI_PATH}fmmf_kl_snr_adi.npy")

    fmmf_obj = FMMFBuilder(
        dataset=betapic,
        model="KLIP",
        var="FR",
        nproc=None,
        min_r=26,
        max_r=34,
        param={"ncomp": 10, "tolerance": 0.005, "delta_rot": 0.5},
        crop=5,
        imlib=imlib_rot,
        verbose=True,
    ).build()

    fmmf_obj.run()

    check_detection(fmmf_obj.snr_map, position, betapic.fwhm, snr_thresh=2)
    assert np.allclose(np.abs(exp_frame), np.abs(fmmf_obj.frame_final), atol=1.0e-2)
    assert np.allclose(np.abs(exp_snr), np.abs(fmmf_obj.snr_map), atol=1.0e-2)

    # Testing FMMF (with the LOCI model)

    position = np.load(f"{INVADI_PATH}fmmf_lo_adi_detect.npy")
    exp_frame = np.load(f"{INVADI_PATH}fmmf_lo_adi.npy")
    exp_snr = np.load(f"{INVADI_PATH}fmmf_lo_snr_adi.npy")

    update(fmmf_obj, FMMFBuilder(model="LOCI"))

    fmmf_obj.run()

    check_detection(fmmf_obj.snr_map, position, betapic.fwhm, snr_thresh=2)
    assert np.allclose(np.abs(exp_frame), np.abs(fmmf_obj.frame_final), atol=1.0e-2)
    assert np.allclose(np.abs(exp_snr), np.abs(fmmf_obj.snr_map), atol=1.0e-2)
