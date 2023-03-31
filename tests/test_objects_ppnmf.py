"""Test for the PostProc object dedicated to non-negative matrix factorization."""

__author__ = "Thomas Bédrine"

import copy

import numpy as np
from dataclass_builder import update

from .helpers import fixture
from vip_hci.objects import NMFBuilder
from vip_hci.psfsub import nmf
from vip_hci.psfsub import nmf_annular
from vip_hci.metrics import detection


@fixture(scope="module")
def setup_dataset(example_dataset_adi):
    betapic = copy.copy(example_dataset_adi)

    return betapic


def test_nmf_object(setup_dataset):
    """
    Compare frames obtained through procedural and object versions of NMF.

    Generate a frame with both ``vip_hci.psfsub.nmf`` and ``vip_hci.objects.ppnmf`` and
    ensure they match. Annular case is tested too with ``vip_hci.psfsub.nmf_annular``.

    NMF case is quite different from the usual ones, because NMF will generate slightly
    different frames from an iteration to another. Instead of comparing the ndarrays of
    the frames, we detect the companion candidate in both frames and ensure their
    coordinates are close from each other.

    """

    def verify_expcoord(frame_1, frame_2):
        deltapix = 3
        y_1, x_1 = detection(
            array=frame_1,
            fwhm=fwhm,
            mode="lpeaks",
            bkg_sigma=5,
            matched_filter=False,
            mask=True,
            snr_thresh=2,
            plot=False,
            debug=True,
            full_output=False,
            verbose=False,
        )

        y_2, x_2 = detection(
            array=frame_2,
            fwhm=fwhm,
            mode="lpeaks",
            bkg_sigma=5,
            matched_filter=False,
            mask=True,
            snr_thresh=2,
            plot=False,
            debug=True,
            full_output=False,
            verbose=False,
        )

        if np.allclose(y_1[0], y_2[0], atol=deltapix) and np.allclose(
            x_1[0], x_2[0], atol=deltapix
        ):
            return True
        return False

    betapic = setup_dataset
    cube = betapic.cube
    angles = betapic.angles
    fwhm = betapic.fwhm

    imlib_rot = "vip-fft"
    interpolation = None

    # Test for full-frame NMF

    fr_nmf = nmf(
        cube=cube,
        angle_list=angles,
        ncomp=14,
        max_iter=10000,
        init_svd="nndsvdar",
        mask_center_px=None,
        imlib=imlib_rot,
        interpolation=interpolation,
    )

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

    assert verify_expcoord(fr_nmf, nmf_obj.frame_final)

    # Test for annular NMF

    fr_nmf_ann = nmf_annular(
        cube=cube,
        angle_list=angles,
        ncomp=9,
        max_iter=10000,
        init_svd="nndsvdar",
        radius_int=0,
        nproc=None,
        fwhm=fwhm,
        asize=fwhm,
        imlib=imlib_rot,
        interpolation=interpolation,
    )

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

    assert verify_expcoord(fr_nmf_ann, nmf_obj.frame_final)
