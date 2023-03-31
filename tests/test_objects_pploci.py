"""Test for the PostProc object dedicated to LOCI."""
import numpy as np

from tests.conftest import fixture
from vip_hci.objects import LOCIBuilder
from vip_hci.psfsub import xloci


@fixture(scope="module")
def test_loci_object(example_dataset_adi):
    betapic = example_dataset_adi
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

    assert np.allclose(np.abs(fr_loci), np.abs(loci_obj.frame_final), atol=1.0e-2)
