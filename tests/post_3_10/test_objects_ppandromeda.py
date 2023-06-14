"""Test for the PostProc object dedicated to ANDROMEDA."""

__author__ = "Thomas Bédrine"

import copy

import numpy as np
from dataclass_builder import update

import sys

sys.path.append(".../tests")
from tests.helpers import fixture, check_detection
from vip_hci.config import VLT_NACO
from vip_hci.objects import AndroBuilder
from tests.snapshots.snapshot_invprob import INVADI_PATH


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


def test_andromeda_object(injected_cube_position):
    """
    Compare frames obtained through procedural and object versions of ANDROMEDA.

    Generate a frame and a S/N map with ``vip_hci.objects.ppandromeda`` and ensure they
    match with their procedural counterpart. This is done by getting the snapshot of the
    ``vip_hci.invprob.andromeda`` function, generated preemptively with
    ``tests..snapshots.snapshot_invprob``.

    """
    betapic = injected_cube_position

    lbda = VLT_NACO["lambdal"]
    diam = VLT_NACO["diam"]
    resel = (lbda / diam) * 206265
    nyquist_samp = resel / 2.0
    oversamp_fac = nyquist_samp / betapic.px_scale

    # Testing Andromeda with lsq optimization method

    position = np.load(f"{INVADI_PATH}andro_adi_detect.npy")
    exp_frame = np.load(f"{INVADI_PATH}andro_adi.npy")
    exp_snr = np.load(f"{INVADI_PATH}andro_snr_adi.npy")

    andro_obj = AndroBuilder(
        dataset=betapic,
        oversampling_fact=oversamp_fac,
        filtering_fraction=0.25,
        min_sep=0.5,
        annuli_width=1.0,
        roa=2,
        opt_method="lsq",
        nsmooth_snr=18,
        iwa=2,
        owa=None,
        precision=50,
        fast=False,
        homogeneous_variance=True,
        ditimg=1.0,
        ditpsf=None,
        tnd=1.0,
        total=False,
        multiply_gamma=True,
        verbose=False,
    ).build()

    andro_obj.run()

    check_detection(andro_obj.snr_map, position, betapic.fwhm, snr_thresh=2)
    assert np.allclose(np.abs(exp_frame), np.abs(andro_obj.frame_final), atol=1.0e-2)
    assert np.allclose(np.abs(exp_snr), np.abs(andro_obj.snr_map), atol=1.0e-2)

    # Testing Andromeda with l1 optimization method

    position = np.load(f"{INVADI_PATH}androl1_adi_detect.npy")
    exp_frame = np.load(f"{INVADI_PATH}androl1_adi.npy")
    exp_snr = np.load(f"{INVADI_PATH}androl1_snr_adi.npy")

    update(andro_obj, AndroBuilder(opt_method="l1", owa=None))

    andro_obj.run()

    check_detection(andro_obj.snr_map, position, betapic.fwhm, snr_thresh=2)
    assert np.allclose(np.abs(exp_frame), np.abs(andro_obj.frame_final), atol=1.0e-2)
    assert np.allclose(np.abs(exp_snr), np.abs(andro_obj.snr_map), atol=1.0e-2)
