"""Generate snapshots for every invprob function."""

__author__ = "Thomas Bedrine"

__all__ = "INVADI_PATH"

import copy

import numpy as np
from vip_hci.invprob import fmmf, andromeda
from vip_hci.metrics import detection
from vip_hci.fits import open_fits
from vip_hci.objects import Dataset
from vip_hci.config import VLT_NACO

from requests.exceptions import ReadTimeout
from ratelimit import limits, sleep_and_retry
from astropy.utils.data import download_file

INVADI_PATH = "./tests/snapshots/invprob_adi/"
DATASET_ELEMENTS = ["cube", "angles", "psf"]


@sleep_and_retry
@limits(calls=1, period=1)
def download_resource(url):
    attempts = 5
    while attempts > 0:
        try:
            return download_file(url, cache=True)
        except ReadTimeout:
            attempts -= 1

    raise TimeoutError("Resource could not be accessed due to too many timeouts.")


def make_dataset_adi():
    """
    Download example FITS cube from github + prepare Dataset object.

    Returns
    -------
    dataset : HCIDataset

    Notes
    -----
    We use the helper function ``download_resource`` which handles the request
    and puts it to sleep for a defined duration if too many requests are done.
    They inherently call the Astropy's ``download_file`` function which uses caching,
    so the file is downloaded at most once per test run.

    """
    print("downloading data...")

    url_prefix = "https://github.com/vortex-exoplanet/VIP_extras/raw/master/datasets"

    f1 = download_resource(f"{url_prefix}/naco_betapic_cube_cen.fits")
    f2 = download_resource(f"{url_prefix}/naco_betapic_psf.fits")
    f3 = download_resource(f"{url_prefix}/naco_betapic_pa.fits")

    # load fits
    cube = open_fits(f1)
    angles = open_fits(f3).flatten()  # shape (61,1) -> (61,)
    psf = open_fits(f2)

    # create dataset object
    dataset = Dataset(cube=cube, angles=angles, psf=psf, px_scale=VLT_NACO["plsc"])

    dataset.normalize_psf(size=20, force_odd=False)

    # overwrite PSF for easy access
    dataset.psf = dataset.psfn

    return dataset


# Note : this function comes from the former test for adi psfsub, I did not write it,
# and I didn't found the author (feel free to add the author if you know them)
def injected_cube_position():
    """
    Inject a fake companion into an example cube.

    Returns
    -------
    dsi : VIP Dataset
    injected_position_yx : tuple(y, x)

    """
    print("injecting fake planet...")
    dsi = copy.copy(make_dataset_adi())
    # we chose a shallow copy, as we will not use any in-place operations
    # (like +=). Using `deepcopy` would be safer, but consume more memory.

    dsi.inject_companions(300, rad_dists=30, verbose=False)

    return dsi, dsi.injections_yx[0]


# TODO: manually saving snapshots can be long, the use of pytest-snapshots is highly
# recommended, but only if you can overload the assert_match function, which currently
# only support exact equal assertions, thus being incompatible with ``np.allclose``
# More info on : https://pypi.org/project/pytest-snapshot/
def save_snapshots_invprob_adi():
    """
    Save expected results (snapshots) for all PSF subtraction functions.

    The obtained frame and the list of companion candidates are saved in separate numpy
    files to be loaded in tests that require to compare if said function still generate
    an appropriate result.

    """
    betapic, position = injected_cube_position()

    cube = betapic.cube
    angles = betapic.angles
    fwhm = betapic.fwhm
    psf = betapic.psf
    snr_thresh = 2

    lbda = VLT_NACO["lambdal"]
    diam = VLT_NACO["diam"]
    resel = (lbda / diam) * 206265
    nyquist_samp = resel / 2.0
    oversamp_fac = nyquist_samp / betapic.px_scale

    imlib_rot = "vip-fft"
    interpolation = None

    # Andromeda (LSQ optimization method)

    andro_adi, _, andro_snr_adi, _, _, _, _ = andromeda(
        cube=cube,
        angle_list=angles,
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

    # Andromeda (l1 optimization method)

    androl1_adi, _, androl1_snr_adi, _, _, _, _ = andromeda(
        cube=cube,
        angle_list=angles,
        psf=psf,
        oversampling_fact=oversamp_fac,
        filtering_fraction=0.25,
        min_sep=0.5,
        annuli_width=1.0,
        roa=2,
        opt_method="l1",
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

    # FMMF (KLIP model)

    fmmf_kl_adi, fmmf_kl_snr_adi = fmmf(
        cube=cube,
        angle_list=angles,
        fwhm=fwhm,
        psf=psf,
        model="KLIP",
        var="FR",
        nproc=None,
        min_r=26,
        max_r=34,
        param={"ncomp": 10, "tolerance": 0.005, "delta_rot": 0.5},
        crop=5,
        imlib="opencv",
    )

    # FMMF (LOCI model)

    fmmf_lo_adi, fmmf_lo_snr_adi = fmmf(
        cube=cube,
        angle_list=angles,
        fwhm=fwhm,
        psf=psf,
        model="LOCI",
        var="FR",
        nproc=None,
        min_r=26,
        max_r=34,
        param={"ncomp": 10, "tolerance": 0.005, "delta_rot": 0.5},
        crop=5,
        imlib="opencv",
    )

    for name, value in locals().items():
        if isinstance(value, np.ndarray) and name not in DATASET_ELEMENTS:
            np.save(f"{INVADI_PATH}{name}.npy", value)
            if "snr" in name:
                detect = detection(
                    value,
                    fwhm=fwhm,
                    mode="lpeaks",
                    bkg_sigma=5,
                    matched_filter=False,
                    mask=True,
                    snr_thresh=snr_thresh,
                    plot=False,
                    debug=True,
                    full_output=False,
                    verbose=False,
                )
                det_array = np.stack(detect, axis=-1)
                name = name.replace("_snr", "")
                np.save(f"{INVADI_PATH}{name}_detect.npy", det_array)
    return


save_snapshots_invprob_adi()
