"""Test for the PostProc object dedicated to ANDROMEDA."""
import numpy as np

from tests.conftest import fixture
from vip_hci.config import VLT_NACO
from vip_hci.invprob import andromeda
from vip_hci.objects import AndroBuilder


@fixture(scope="module")
def test_andromeda_object(example_dataset_adi):
    betapic = example_dataset_adi
    cube = betapic.cube
    angles = betapic.angles
    psf = betapic.psf
    fwhm = betapic.fwhm

    lbda = VLT_NACO["lambdal"]
    diam = VLT_NACO["diam"]
    resel = (lbda / diam) * 206265
    nyquist_samp = resel / 2.0
    oversamp_fac = nyquist_samp / betapic.px_scale

    fr_andro, _, snr_andro, _, _, _, _ = andromeda(
        cube=cube,
        angles=angles,
        fwhm=fwhm,
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
        nproc=1,
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
        nproc=1,
        verbose=False,
    ).build()

    assert np.allclose(np.abs(fr_andro), np.abs(andro_obj.frame_final), atol=1.0e-2)
    assert np.allclose(np.abs(snr_andro), np.abs(andro_obj.snr_map), atol=1.0e-2)
