"""Test for the PostProc object dedicated to LOCI."""

__author__ = "Thomas Bédrine"

import copy

import numpy as np

from .helpers import fixture, check_detection
from vip_hci.objects import LOCIBuilder
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


def test_loci_object(injected_cube_position):
    """
    Compare frames obtained through procedural and object versions of LOCI.

    Generate a frame with ``vip_hci.objects.pploci`` and ensure they match with
    their procedural counterpart. This is done by getting the snapshot of the
    ``vip_hci.psfsub.xloci`` function, generated preemptively with
    ``tests.snapshots.save_snapshots_psfsub``.

    """
    betapic = injected_cube_position

    fwhm = betapic.fwhm
    imlib_rot = "vip-fft"
    interpolation = None

    # Testing LOCI

    position = np.load(f"{PSFADI_PATH}loci_adi_detect.npy")
    exp_frame = np.load(f"{PSFADI_PATH}loci_adi.npy")

    loci_obj = LOCIBuilder(
        dataset=betapic,
        asize=fwhm,
        n_segments="auto",
        nproc=None,
        metric="correlation",
        radius_int=20,
        dist_threshold=90,
        delta_rot=0.5,
        optim_scale_fact=3,
        solver="lstsq",
        tol=0.01,
        imlib=imlib_rot,
        interpolation=interpolation,
        verbose=False,
    ).build()

    loci_obj.run()
    loci_obj.make_snrmap()

    check_detection(loci_obj.snr_map, position, fwhm, snr_thresh=2)
    assert np.allclose(np.abs(exp_frame), np.abs(loci_obj.frame_final), atol=1.0e-2)
