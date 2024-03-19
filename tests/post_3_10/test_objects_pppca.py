"""Test for the PostProc object dedicated to principal component analysis."""

__author__ = "Carlos Gomez Gonzalez, Thomas Bédrine"

import copy

import numpy as np
from dataclass_builder import update
import sys
sys.path.append(".../tests")
from tests.helpers import fixture, check_detection
from vip_hci.objects import PCABuilder
from tests.snapshots.snapshot_psfsub import PSFADI_PATH

NO_FRAME_CASE = ["pca_drot", "pca_ann_auto"]
PREV_CASE = ["pca_grid_list"]


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


def test_pca_object(injected_cube_position):
    """
    Compare frames obtained through procedural and object versions of PCA.

    Generate a frame with ``vip_hci.objects.pppca`` and ensure they match with
    their procedural counterpart. This is done by getting the snapshots of the
    ``vip_hci.psfsub.pca`` and ``vip_hci.psfsub.pca_annular`` functions,
    generated preemptively with ``tests.snapshots.snapshot_psfsub``. There are a
    lot more cases to test than in regular post-processing methods because PCA
    is the "go-to" method of the package.

    """
    betapic = injected_cube_position

    # Testing the basic version of PCA

    position = np.load(f"{PSFADI_PATH}pca_adi_detect.npy").copy()
    exp_frame = np.load(f"{PSFADI_PATH}pca_adi.npy").copy()

    pca_obj = PCABuilder(dataset=betapic, svd_mode="arpack").build()

    pca_obj.run()
    pca_obj.make_snrmap()

    check_detection(pca_obj.snr_map, position, betapic.fwhm, snr_thresh=2)
    assert np.allclose(np.abs(pca_obj.frame_final), np.abs(exp_frame),
                       atol=1e-2)

    pca_test_set = [
        {
            "case_name": "pca_left_eigv",
            "update_params": {
                "svd_mode": "lapack",
                "left_eigv": True,
            },
            "runmode": "classic",
        },
        {
            "case_name": "pca_linalg",
            "update_params": {"left_eigv": False, "svd_mode": "eigen"},
            "runmode": "classic",
        },
        {
            "case_name": "pca_drot",
            "update_params": {
                "ncomp": 4,
                "svd_mode": "randsvd",
                "delta_rot": 0.5,
                "source_xy": betapic.injections_yx[0][::-1],
            },
            "runmode": "classic",
        },
        {
            "case_name": "pca_cevr",
            "update_params": {
                "ncomp": 0.95,
                "svd_mode": "lapack",
                "delta_rot": None,
                "source_xy": None,
            },
            "runmode": "classic",
        },
        {
            "case_name": "pca_grid",
            "update_params": {
                "ncomp": (1, 2),
                "source_xy": betapic.injections_yx[0][::-1],
                "verbose": False,
            },
            "runmode": "classic",
        },
        {
            "case_name": "pca_grid_list",
            "update_params": {
                "ncomp": [1, 2],
                "source_xy": None,
                "verbose": False,
            },
            "runmode": "classic",
        },
        {
            "case_name": "pca_ann",
            "update_params": {
                "ncomp": 1,
                "source_xy": None,
                "delta_rot": (0.1, 1),
                "n_segments": "auto",
            },
            "runmode": "annular",
        },
        {
            "case_name": "pca_ann_left_eigv",
            "update_params": {"left_eigv": True},
            "runmode": "annular",
        },
        {
            "case_name": "pca_ann_auto",
            "update_params": {"left_eigv": False, "ncomp": "auto"},
            "runmode": "annular",
        },
    ]
    # Testing all alternatives of PCA listed above

    for pp, pca_test in enumerate(pca_test_set):
        case_name = pca_test["case_name"]
        update_params = pca_test["update_params"]
        runmode = pca_test["runmode"]

        if case_name in PREV_CASE:
            update(pca_obj, PCABuilder(**update_params))
            pca_obj.run(runmode=runmode)
            # compare to frame_final from previous iteration
            assert np.allclose(
                np.abs(pca_obj.frame_final), np.abs(pca_obj.frames_final[-1]),
                atol=1e-2
            )
        else:
            pos = np.load(f"{PSFADI_PATH}{case_name}_adi_detect.npy").copy()
            if case_name not in NO_FRAME_CASE:
                exp_frame = np.load(f"{PSFADI_PATH}{case_name}_adi.npy").copy()

            update(pca_obj, PCABuilder(**update_params))

            pca_obj.run(runmode=runmode)
            pca_obj.make_snrmap()

            check_detection(pca_obj.snr_map, pos, betapic.fwhm, snr_thresh=2)
            if case_name not in NO_FRAME_CASE:
                assert np.allclose(
                    np.abs(pca_obj.frame_final), np.abs(exp_frame), atol=1e-2
                )
