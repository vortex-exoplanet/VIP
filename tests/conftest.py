"""
Configuration file for pytest, containing global ("session-level") fixtures.

"""

import pytest
from astropy.utils.data import download_file
import vip_hci as vip
import numpy as np


@pytest.fixture(scope="session")
def example_dataset_adi():
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

    url_prefix = "https://github.com/vortex-exoplanet/VIP_extras/raw/master/datasets"

    f1 = download_file("{}/naco_betapic_cube_cen.fits".format(url_prefix),
                       cache=True)
    f2 = download_file("{}/naco_betapic_psf.fits".format(url_prefix),
                       cache=True)
    f3 = download_file("{}/naco_betapic_pa.fits".format(url_prefix),
                       cache=True)

    # load fits
    cube = vip.fits.open_fits(f1)
    angles = vip.fits.open_fits(f3).flatten()  # shape (61,1) -> (61,)
    psf = vip.fits.open_fits(f2)

    # create dataset object
    dataset = vip.Dataset(cube, angles=angles, psf=psf,
                          px_scale=vip.config.VLT_NACO['plsc'])

    dataset.normalize_psf(size=20, force_odd=False)

    # overwrite PSF for easy access
    dataset.psf = dataset.psfn

    return dataset


@pytest.fixture(scope="session")
def example_dataset_ifs():
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

    url_prefix = "https://github.com/vortex-exoplanet/VIP_extras/raw/master/datasets"

    f1 = download_file("{}/sphere_v471tau_cube.fits".format(url_prefix),
                       cache=True)
    f2 = download_file("{}/sphere_v471tau_psf.fits".format(url_prefix),
                       cache=True)
    f3 = download_file("{}/sphere_v471tau_pa.fits".format(url_prefix),
                       cache=True)
    f4 = download_file("{}/sphere_v471tau_wl.fits".format(url_prefix),
                       cache=True)

    # load fits
    cube = vip.fits.open_fits(f1)
    angles = vip.fits.open_fits(f3).flatten()
    psf = vip.fits.open_fits(f2)
    wl = vip.fits.open_fits(f4)

    # create dataset object
    dataset = vip.Dataset(cube, angles=angles, psf=psf,
                          px_scale=vip.config.VLT_SPHERE_IFS['plsc'],
                          wavelengths=wl)

    # crop
    dataset.crop_frames(size=100, force=True)
    dataset.normalize_psf(size=None, force_odd=False)

    # overwrite PSF for easy access
    dataset.psf = dataset.psfn

    return dataset


@pytest.fixture(scope="session")
def example_dataset_rdi():
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

    url_prefix = "https://github.com/vortex-exoplanet/VIP_extras/raw/master/datasets"

    f1 = download_file("{}/naco_betapic_cube_cen.fits".format(url_prefix),
                       cache=True)
    f2 = download_file("{}/naco_betapic_psf.fits".format(url_prefix),
                       cache=True)
    f3 = download_file("{}/naco_betapic_pa.fits".format(url_prefix),
                       cache=True)

    # load fits
    cube = vip.fits.open_fits(f1)
    angles = vip.fits.open_fits(f3).flatten()  # shape (61,1) -> (61,)
    psf = vip.fits.open_fits(f2)
    # creating a variable flux screen
    scr = vip.var.create_synth_psf('moff', (101, 101), fwhm=50)
    scrcu = np.array([scr * i for i in np.linspace(-1e2, 1e2, num=31)], 
                     dtype=np.float32)
    
    # OLD: scaling ?!
    # upscaling (1.2) and taking half of the frames, reversing order
    #cube_upsc = cube_px_resampling(cube[::-1], 1.2, verbose=False)[::2]
    # cropping and adding the flux screen
    #cube_ref = cube_crop_frames(cube_upsc, 101, verbose=False) + scrcu

    # NEW: take centro-symmetric images (i.e. flip along both x and y)
    cube_rot = cube[::-1]
    cube_rot = cube_rot[::2]
    cube_rot = np.flip(cube_rot, axis=1)
    cube_rot = np.flip(cube_rot, axis=2)
    
    # cropping and adding the flux screen
    cube_ref = cube_rot + scrcu    

    # create dataset object
    dataset = vip.Dataset(cube, angles=angles, psf=psf, cuberef=cube_ref,
                          px_scale=vip.config.VLT_NACO['plsc'])

    dataset.normalize_psf(size=20, force_odd=False)

    # overwrite PSF for easy access
    dataset.psf = dataset.psfn

    return dataset
