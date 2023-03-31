"""Test for the PostProc object dedicated to non-negative matrix factorization."""
import copy

import numpy as np
from dataclass_builder import update

from .helpers import fixture
from vip_hci.objects import NMFBuilder
from vip_hci.psfsub import nmf
from vip_hci.psfsub import nmf_annular


@fixture(scope="module")
def test_nmf_object(example_dataset_adi):
    betapic = copy.copy(example_dataset_adi)
    cube = betapic.cube
    angles = betapic.angles
    fwhm = betapic.fwhm

    imlib_rot = "vip-fft"
    interpolation = None

    fr_nmf = nmf(
        cube=cube,
        angle_list=angles,
        fwhm=fwhm,
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

    assert np.allclose(np.abs(fr_nmf), np.abs(nmf_obj.frame_final), atol=1.0e-2)

    fr_nmf_ann = nmf_annular(
        cube=cube,
        angle_list=angles,
        fwhm=fwhm,
        ncomp=14,
        max_iter=10000,
        init_svd="nndsvdar",
        mask_center_px=None,
        imlib=imlib_rot,
        interpolation=interpolation,
    )

    update(
        nmf_obj,
        NMFBuilder(
            ncomp=9,
            radius_int=0,
            nproc=None,
            fwhm=fwhm,
            asize=fwhm,
            verbose=False,
        ),
    )

    nmf_obj.run()

    assert np.allclose(np.abs(fr_nmf_ann), np.abs(nmf_obj.frame_final), atol=1.0e-2)
