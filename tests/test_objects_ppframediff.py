"""Test for the PostProc object dedicated to frame differencing."""
import copy

import numpy as np

from .helpers import fixture
from vip_hci.objects import FrameDiffBuilder
from vip_hci.psfsub import frame_diff


@fixture(scope="module")
def test_frame_diff_object(example_dataset_adi):
    betapic = copy.copy(example_dataset_adi)
    cube = betapic.cube
    angles = betapic.angles
    fwhm = betapic.fwhm

    imlib_rot = "vip-fft"
    interpolation = None

    fr_fdiff = frame_diff(
        cube=cube,
        angle_list=angles,
        fwhm=fwhm,
        metric="l1",
        dist_threshold=90,
        delta_rot=0.5,
        radius_int=4,
        asize=fwhm,
        nproc=None,
        imlib=imlib_rot,
        interpolation=interpolation,
    )

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

    assert np.allclose(np.abs(fr_fdiff), np.abs(framediff_obj.frame_final), atol=1.0e-2)
