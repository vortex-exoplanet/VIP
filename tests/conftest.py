"""
Configuration file for pytest, containing global ("session-level") fixtures.

"""

import pytest

from astropy.utils.data import download_file

import vip_hci as vip


@pytest.fixture(scope="session")
def example_dataset():
    """
    Download example FITS cube from github + prepare HCIDataset object.

    Returns
    -------
    dataset : HCIDataset

    Notes
    -----
    Astropy's ``download_file`` uses caching, so the file is downloaded at most
    once per test run.

    """
    print("downloading data...")

    url_prefix = ("https://github.com/carlgogo/vip-tutorial/raw/"
                  "ced844dc807d3a21620fe017db116d62fbeaaa4a")

    f1 = download_file("{}/naco_betapic.fits".format(url_prefix), cache=True)
    f2 = download_file("{}/naco_psf.fits".format(url_prefix), cache=True)

    # load fits
    cube = vip.fits.open_fits(f1, 0)
    angles = vip.fits.open_fits(f1, 1).flatten()  # shape (61,1) -> (61,)
    psf = vip.fits.open_fits(f2)

    # create dataset object
    dataset = vip.HCIDataset(cube, angles=angles, psf=psf, px_scale=0.01225)

    # crop
    dataset.crop_frames(size=100, force=True)
    dataset.normalize_psf(size=38, force_odd=False)

    # overwrite PSF for easy access
    dataset.psf = dataset.psfn

    return dataset
