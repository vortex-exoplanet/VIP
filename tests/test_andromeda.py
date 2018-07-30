"""
Tests for the andromeda submodule.

"""

from __future__ import division, print_function

import os
import numpy as np
from astropy.utils.data import download_file

import vip_hci as vip
from vip_hci.andromeda.andromeda import andromeda


# ===== utility functions

def aarc(actual, desired, rtol=1e-5, atol=1e-6):
    """
    Assert array-compare. Like ``np.allclose``, but with different defaults.

    Notes
    -----
    Default values for
    - ``np.allclose``: ``atol=1e-8, rtol=1e-5``
    - ``np.testing.assert_allclose``: ``atol=0, rtol=1e-7``

    IDL's `FLOAT` is 32bit (precision of ~1e-6 to ~1e-7), while python uses
    64-bit floats internally. To pass the comparison, `atol` has to be chosen
    accordingly. The contribution of `rtol` is dominant for large numbers, where
    an absolute comparison to `1e-6` would not make sense.

    """
    __tracebackhide__ = True  # Hide traceback for py.test
    np.testing.assert_allclose(actual, desired, rtol=rtol, atol=atol)


def load_dataset():
    url_prefix = ("https://github.com/carlgogo/vip-tutorial/raw/"
                  "ced844dc807d3a21620fe017db116d62fbeaaa4a")

    f1 = download_file("{}/naco_betapic.fits".format(url_prefix), cache=True)
    f2 = download_file("{}/naco_psf.fits".format(url_prefix), cache=True)

    # vip.fits.info_fits(f1)
    # vip.fits.info_fits(f2)

    # load fits
    cube = vip.fits.open_fits(f1, 0)
    angles = vip.fits.open_fits(f1, 1).flatten()  # shape (61,1) -> (61,)
    psf = vip.fits.open_fits(f2)

    # create dataset object
    data = vip.HCIDataset(cube, angles=angles, psf=psf)

    # crop
    data.crop_frames(size=100, force=True)
    data.normalize_psf(size=38, force_odd=False)

    # set PSF
    data.psf = data.psfn

    # ===== export
    # data.save("{}_crop".format(f1))
    # vip.fits.write_fits("{}_crop".format(f2), data.psf)
    # idl.f1 = f1+"_crop"
    # idl.f2 = f2+"_crop"

    return data


def generate_test_data():
    """
    Run IDL ANDROMEDA and create ``andromeda_idl.npz``.
    """
    global DATACUBE

    from idlpy import IDL as idl

    idl.cube = DATACUBE.cube
    idl.angles = DATACUBE.angles
    idl.psf = DATACUBE.psf

    idl.run("andromeda_contrast_1 = ANDROMEDA("
            "IMAGES_1_INPUT=cube,"
            "ANGLES_INPUT=angles,"
            "PSF_PLANET_INPUT=psf,"
            "OVERSAMPLING_1_INPUT=1,"
            "FILTERING_FRACTION_INPUT=1,"  # turn off high pass
            "MINIMUM_SEPARATION_INPUT=0.3,"
            "OPT_METHOD_ANG_INPUT=1,"  # no optimization
            "NSMOOTH_SNR_INPUT=0,"  # turn off smoothing
            "/HOMOGENEOUS_VARIANCE_INPUT,"
            ""  # output:
            "SNR_OUTPUT=andromeda_snr_1,"
            "SNR_NORM_OUTPUT=andromeda_snr_norm_1,"
            "LIKELIHOOD_OUTPUT=andromeda_likelihood_1,"
            "STDDEVCONTRAST_OUTPUT=andromeda_stddevcontrast_1,"
            "STDDEVCONTRAST_NORM_OUTPUT=andromeda_stddevcontrast_norm_1,"
            "EXT_RADIUS_OUTPUT=andromeda_ext_radius_1)")

    OUT = {}
    OUT["andromeda_contrast_1"] = idl.andromeda_contrast_1
    OUT["andromeda_snr_1"] = idl.andromeda_snr_1
    OUT["andromeda_snr_norm_1"] = idl.andromeda_snr_norm_1
    OUT["andromeda_likelihood_1"] = idl.andromeda_likelihood_1
    OUT["andromeda_stdcontrast_1"] = idl.andromeda_stddevcontrast_1
    OUT["andromeda_stdcontrast_norm_1"] = idl.andromeda_stddevcontrast_norm_1

    np.savez_compressed(os.path.join(CURRDIR, "andromeda_idl.npz"), **OUT)


# ===== tests

def test_andromeda():
    # TODO: test `fast`
    # TODO: test `nsnmooth_snr`
    # TODO: test `homogeneous_variance`
    # TODO: test `filtering_fraction`
    # TODO: test `oversampling_fact`
    # TODO: test `min_sep`
    # TODO: test `opt_method`

    global IDL_DATA, DATACUBE

    out = andromeda(DATACUBE.cube,
                    angles=DATACUBE.angles,
                    psf=DATACUBE.psf,
                    oversampling_fact=1,
                    filtering_fraction=1,  # turn off high pass
                    min_sep=0.3,
                    opt_method="no",
                    nsmooth_snr=0,  # turn off smoothing
                    homogeneous_variance=True,
                    )
    contrast, snr, snr_norm, stdcontrast, stdcontrast_norm, likelihood, _ = out

    aarc(contrast, IDL_DATA["andromeda_contrast_1"], atol=1e-2)
    aarc(snr, IDL_DATA["andromeda_snr_1"], atol=1e-4)
    aarc(snr_norm, IDL_DATA["andromeda_snr_norm_1"], atol=1e-4)
    aarc(stdcontrast, IDL_DATA["andromeda_stdcontrast_1"])
    aarc(stdcontrast_norm, IDL_DATA["andromeda_stdcontrast_norm_1"])
    aarc(likelihood, IDL_DATA["andromeda_likelihood_1"], atol=1e-4)


CURRDIR = os.path.dirname(__file__)

try:
    IDL_DATA = np.load(os.path.join(CURRDIR, "andromeda_idl.npz"))
except FileNotFoundError:
    print("Could not load andromeda_idl.npz. Try running generate_test_data() "
          "to create it.")

DATACUBE = load_dataset()
