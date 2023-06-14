"""Test for the PostProc object dedicated to non-negative matrix factorization."""

__author__ = "Thomas Bédrine"

import copy

import numpy as np
from dataclass_builder import update

import sys

sys.path.append(".../tests")
from tests.helpers import fixture, check_detection
from vip_hci.objects import NMFBuilder
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


def test_nmf_object(injected_cube_position):
    """
    Compare detections obtained through procedural and object versions of NMF.

    Generate a frame with ``vip_hci.objects.ppnmf`` and ensure its detection match
    with its procedural counterpart. This is done by getting the snapshot of the
    ``vip_hci.psfsub.nmf`` function, generated preemptively with
    ``tests..snapshots.save_snapshots_psfsub``. Also done with
    ``vip_hci.psfsub.nmf_annular`` for the annular version of NMF.

    """

    betapic = injected_cube_position

    fwhm = betapic.fwhm
    imlib_rot = "vip-fft"
    interpolation = None

    # Testing the full frame version of NMF

    position = np.load(f"{PSFADI_PATH}nmf_adi_detect.npy")

    nmf_obj = NMFBuilder(
        dataset=betapic,
        ncomp=14,
        max_iter=10000,
        init_svd="nndsvdar",
        mask_center_px=None,
        imlib=imlib_rot,
        interpolation=interpolation,
        verbose=False,
    ).build()

    nmf_obj.run()
    nmf_obj.make_snrmap()

    check_detection(nmf_obj.snr_map, position, betapic.fwhm, snr_thresh=2)

    # Testing the annular version of NMF

    update(
        nmf_obj,
        NMFBuilder(
            ncomp=9,
            radius_int=0,
            asize=fwhm,
            verbose=False,
        ),
    )

    nmf_obj.run()
    nmf_obj.make_snrmap()

    check_detection(nmf_obj.snr_map, position, betapic.fwhm, snr_thresh=2)
