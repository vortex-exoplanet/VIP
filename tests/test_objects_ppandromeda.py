"""Test for the PostProc object dedicated to ANDROMEDA."""

__author__ = "Thomas Bédrine"

import copy

import numpy as np

from .helpers import fixture
from vip_hci.config import VLT_NACO
from vip_hci.invprob import andromeda
from vip_hci.objects import AndroBuilder


@fixture(scope="module")
def setup_dataset(example_dataset_adi):
    betapic = copy.copy(example_dataset_adi)

    return betapic


def test_andromeda_object(setup_dataset):
    """
    Compare frames obtained through procedural and object versions of ANDROMEDA.

    Generate a frame and a S/N map with both ``vip_hci.invprob.andromeda`` and
    ``vip_hci.objects.ppandromeda`` and ensure they match.

    """
    betapic = setup_dataset

    cube = betapic.cube
    angles = betapic.angles
    psf = betapic.psf

    lbda = VLT_NACO["lambdal"]
    diam = VLT_NACO["diam"]
    resel = (lbda / diam) * 206265
    nyquist_samp = resel / 2.0
    oversamp_fac = nyquist_samp / betapic.px_scale

    fr_andro, _, snr_andro, _, _, _, _ = andromeda(
        cube=cube,
        angles=angles,
        psf=psf,
        oversampling_fact=oversamp_fac,
        filtering_fraction=0.25,
        min_sep=0.5,
        annuli_width=1.0,
        roa=2,
        opt_method="lsq",
        nsmooth_snr=18,
        iwa=2,
        owa=None,
        precision=50,
        fast=False,
        homogeneous_variance=True,
        ditimg=1.0,
        ditpsf=None,
        tnd=1.0,
        total=False,
        multiply_gamma=True,
        verbose=False,
    )

    andro_obj = AndroBuilder(
        dataset=betapic,
        oversampling_fact=oversamp_fac,
        filtering_fraction=0.25,
        min_sep=0.5,
        annuli_width=1.0,
        roa=2,
        opt_method="lsq",
        nsmooth_snr=18,
        iwa=2,
        owa=None,
        precision=50,
        fast=False,
        homogeneous_variance=True,
        ditimg=1.0,
        ditpsf=None,
        tnd=1.0,
        total=False,
        multiply_gamma=True,
        verbose=False,
    ).build()

    andro_obj.run()

    assert np.allclose(np.abs(fr_andro), np.abs(andro_obj.frame_final), atol=1.0e-2)
    assert np.allclose(np.abs(snr_andro), np.abs(andro_obj.snr_map), atol=1.0e-2)
