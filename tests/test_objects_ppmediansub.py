"""Test for the PostProc object dedicated to median subtraction."""

__author__ = "Thomas Bédrine"

import copy

import numpy as np

from .helpers import fixture
from vip_hci.objects import MedianBuilder
from vip_hci.psfsub import median_sub


@fixture(scope="module")
def setup_dataset(example_dataset_adi):
    betapic = copy.copy(example_dataset_adi)

    return betapic


def test_median_object(setup_dataset):
    """
    Compare frames obtained through procedural and object versions of median sub.

    Generate a frame with both ``vip_hci.psfsub.median_sub`` and
    ``vip_hci.objects.ppmediansub`` and ensure they match.

    """
    betapic = setup_dataset
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

    medsub_obj.run()

    assert np.allclose(np.abs(fr_adi), np.abs(medsub_obj.frame_final), atol=1.0e-2)
