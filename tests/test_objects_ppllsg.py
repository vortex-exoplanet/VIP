"""Test for the PostProc object dedicated to LLSG."""

__author__ = "Thomas Bédrine"

import copy

import numpy as np

from .helpers import fixture
from vip_hci.objects import LLSGBuilder
from vip_hci.psfsub import llsg


@fixture(scope="module")
def setup_dataset(example_dataset_adi):
    betapic = copy.copy(example_dataset_adi)

    return betapic


def test_llsg_object(setup_dataset):
    """
    Compare frames obtained through procedural and object versions of LLSG.

    Generate a frame with both ``vip_hci.psfsub.llsg`` and ``vip_hci.objects.ppllsg``
    and ensure they match.

    """
    betapic = setup_dataset
    cube = betapic.cube
    angles = betapic.angles
    fwhm = betapic.fwhm

    imlib_rot = "vip-fft"
    interpolation = None

    fr_llsg = llsg(
        cube=cube,
        angle_list=angles,
        fwhm=fwhm,
        rank=5,
        thresh=1,
        max_iter=20,
        random_seed=10,
        imlib=imlib_rot,
        interpolation=interpolation,
    )

    llsg_obj = LLSGBuilder(
        dataset=betapic,
        rank=5,
        thresh=1,
        max_iter=20,
        random_seed=10,
        verbose=False,
    ).build()

    llsg_obj.run(
        imlib=imlib_rot,
        interpolation=interpolation,
    )

    assert np.allclose(np.abs(fr_llsg), np.abs(llsg_obj.frame_final), atol=1.0e-2)
