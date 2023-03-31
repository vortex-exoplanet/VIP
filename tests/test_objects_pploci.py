"""Test for the PostProc object dedicated to LOCI."""

__author__ = "Thomas Bédrine"

import copy

import numpy as np

from .helpers import fixture
from vip_hci.objects import LOCIBuilder
from vip_hci.psfsub import xloci


@fixture(scope="module")
def setup_dataset(example_dataset_adi):
    betapic = copy.copy(example_dataset_adi)

    return betapic


def test_loci_object(setup_dataset):
    """
    Compare frames obtained through procedural and object versions of LOCI.

    Generate a frame with both ``vip_hci.psfsub.loci`` and ``vip_hci.objects.pploci``
    and ensure they match.

    """
    betapic = setup_dataset
    cube = betapic.cube
    angles = betapic.angles
    fwhm = betapic.fwhm

    imlib_rot = "vip-fft"
    interpolation = None

    fr_loci = xloci(
        cube=cube,
        angle_list=angles,
        fwhm=fwhm,
        asize=fwhm,
        n_segments="auto",
        nproc=None,
        metric="correlation",
        dist_threshold=90,
        delta_rot=0.5,
        optim_scale_fact=3,
        solver="lstsq",
        tol=0.01,
        radius_int=fwhm,
        imlib=imlib_rot,
        interpolation=interpolation,
    )

    loci_obj = LOCIBuilder(
        dataset=betapic,
        asize=fwhm,
        n_segments="auto",
        nproc=None,
        metric="correlation",
        dist_threshold=90,
        delta_rot=0.5,
        optim_scale_fact=3,
        solver="lstsq",
        tol=0.01,
        radius_int=fwhm,
        imlib=imlib_rot,
        interpolation=interpolation,
        verbose=False,
    ).build()

    loci_obj.run()

    assert np.allclose(np.abs(fr_loci), np.abs(loci_obj.frame_final), atol=1.0e-2)
