"""Test for the PostProc object dedicated to median subtraction."""
import numpy as np

from tests.conftest import fixture
from vip_hci.objects import MedianBuilder
from vip_hci.psfsub import median_sub


@fixture(scope="module")
def test_median_object(example_dataset_adi):
    betapic = example_dataset_adi
    cube = betapic.cube
    angles = betapic.angles
    fwhm = betapic.fwhm

    imlib_rot = "vip-fft"
    interpolation = None

    fr_adi = median_sub(
        cube=cube,
        angle_list=angles,
        fwhm=fwhm,
        mode="fullfr",
        imlib=imlib_rot,
        interpolation=interpolation,
    )

    medsub_obj = MedianBuilder(
        dataset=betapic,
        mode="fullfr",
        imlib=imlib_rot,
        interpolation=interpolation,
        verbose=False,
    ).build()

    assert np.allclose(np.abs(fr_adi), np.abs(medsub_obj.frame_final), atol=1.0e-2)
