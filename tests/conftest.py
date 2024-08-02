"""
Configuration file for pytest, containing global ("session-level") fixtures.

"""
import copy
import time

import numpy as np
import pytest

import vip_hci as vip
from .helpers import download_resource


@pytest.fixture(scope="session")
def example_dataset_adi():
    """
    Download example FITS cube from github + prepare HCIDataset object.

    Returns
    -------
    dataset : HCIDataset

    Notes
    -----
    We use the helper function ``download_resource`` which handles the request
    and puts it to sleep for a defined duration if too many requests are done.
    They inherently call the Astropy's ``download_file`` function which uses
    caching, so the file is downloaded at most once per test run.

    """
    print("downloading data...")

    url_d = "https://github.com/vortex-exoplanet/VIP_extras/raw/master/datasets"

    f1 = download_resource("{}/naco_betapic_cube_cen.fits".format(url_d))
    f2 = download_resource("{}/naco_betapic_psf.fits".format(url_d))
    f3 = download_resource("{}/naco_betapic_derot_angles.fits".format(url_d))

    # load fits
    cube = vip.fits.open_fits(f1)
    angles = vip.fits.open_fits(f3).flatten()  # shape (61,1) -> (61,)
    psf = vip.fits.open_fits(f2)

    # create dataset object
    dataset = vip.objects.Dataset(
        cube, angles=angles, psf=psf, px_scale=vip.config.VLT_NACO["plsc"]
    )

    dataset.normalize_psf(size=20, force_odd=False)

    # overwrite PSF for easy access
    dataset.psf = dataset.psfn

    return dataset


def injected_cube_position_adi(example_dataset_adi):
    """
    Inject a fake companion into an example cube.

    Parameters
    ----------
    example_dataset_adi : fixture
        Taken automatically from ``conftest.py``.

    Returns
    -------
    dsi : VIP Dataset
    injected_position_yx : tuple(y, x)

    """
    print("injecting fake planet...")
    dsi = copy.copy(example_dataset_adi)
    # we chose a shallow copy, as we will not use any in-place operations
    # (like +=). Using `deepcopy` would be safer, but consume more memory.

    dsi.inject_companions(300, rad_dists=30)

    return dsi, dsi.injections_yx[0]


@pytest.fixture(scope="session")
def example_dataset_ifs():
    """
    Download example FITS cube from github + prepare HCIDataset object.

    Returns
    -------
    dataset : HCIDataset

    Notes
    -----
    We use the helper function ``download_resource`` which handles the request
    and puts it to sleep for a defined duration if too many requests are done.
    They inherently call the Astropy's ``download_file`` function which uses
    caching, so the file is downloaded at most once per test run.

    """
    print("downloading data...")

    url_d = "https://github.com/vortex-exoplanet/VIP_extras/raw/master/datasets"

    f1 = download_resource("{}/sphere_v471tau_cube.fits".format(url_d))
    f2 = download_resource("{}/sphere_v471tau_psf.fits".format(url_d))
    f3 = download_resource("{}/sphere_v471tau_pa.fits".format(url_d))
    f4 = download_resource("{}/sphere_v471tau_wl.fits".format(url_d))

    # load fits
    cube = vip.fits.open_fits(f1)
    angles = vip.fits.open_fits(f3).flatten()
    psf = vip.fits.open_fits(f2)
    wl = vip.fits.open_fits(f4)

    # create dataset object
    dataset = vip.objects.Dataset(
        cube,
        angles=angles,
        psf=psf,
        px_scale=vip.config.VLT_SPHERE_IFS["plsc"],
        wavelengths=wl,
    )

    # crop
    dataset.crop_frames(size=100, force=True)
    dataset.normalize_psf(size=None, force_odd=False)

    # overwrite PSF for easy access
    dataset.psf = dataset.psfn

    return dataset


@pytest.fixture(scope="session")
def example_dataset_ifs_crop():
    """
    Download example FITS cube from github + prepare HCIDataset object, after\
    cropping to only 3 sp. channels (faster NEGFC test).

    Returns
    -------
    dataset : HCIDataset

    Notes
    -----
    We use the helper function ``download_resource`` which handles the request
    and puts it to sleep for a defined duration if too many requests are done.
    They inherently call the Astropy's ``download_file`` function which uses
    caching, so the file is downloaded at most once per test run.

    """
    print("downloading data...")

    url_d = "https://github.com/vortex-exoplanet/VIP_extras/raw/master/datasets"

    f1 = download_resource("{}/sphere_v471tau_cube.fits".format(url_d))
    f2 = download_resource("{}/sphere_v471tau_psf.fits".format(url_d))
    f3 = download_resource("{}/sphere_v471tau_pa.fits".format(url_d))
    f4 = download_resource("{}/sphere_v471tau_wl.fits".format(url_d))

    # load fits
    cube = vip.fits.open_fits(f1)
    cube = cube[-3:]
    angles = vip.fits.open_fits(f3).flatten()
    psf = vip.fits.open_fits(f2)
    psf = psf[-3:]
    wl = vip.fits.open_fits(f4)
    wl = wl[-3:]

    # create dataset object
    dataset = vip.objects.Dataset(
        cube,
        angles=angles,
        psf=psf,
        px_scale=vip.config.VLT_SPHERE_IFS["plsc"],
        wavelengths=wl,
    )

    # crop
    dataset.crop_frames(size=100, force=True)
    dataset.normalize_psf(size=None, force_odd=False)

    # overwrite PSF for easy access
    dataset.psf = dataset.psfn

    return dataset


@pytest.fixture(scope="session")
def example_dataset_rdi():
    """Download example FITS cube from github + prepare HCIDataset object.

    As reference cube, we consider the last 21 frames of the beta Pic datacube
    in which we have removed the companion. In the RDI tests, we then only use
    the 40 first frames of the Beta Pic cube.

    Returns
    -------
    dataset : HCIDataset

    Notes
    -----
    We use the helper function ``download_resource`` which handles the request
    and puts it to sleep for a defined duration if too many requests are done.
    They inherently call the Astropy's ``download_file`` function which uses
    caching, so the file is downloaded at most once per test run.

    """
    print("downloading data...")

    url_d = "https://github.com/vortex-exoplanet/VIP_extras/raw/master/datasets"

    f1 = download_resource("{}/naco_betapic_cube_cen.fits".format(url_d))
    f2 = download_resource("{}/naco_betapic_psf.fits".format(url_d))
    f3 = download_resource("{}/naco_betapic_derot_angles.fits".format(url_d))

    # load fits
    cube = vip.fits.open_fits(f1)
    angles = vip.fits.open_fits(f3).flatten()  # shape (61,1) -> (61,)
    psf = vip.fits.open_fits(f2)

    idx_sep = 41  # index where we split SCI from REF from the original cube

    # NEW: just consider the 40 first and last 21 frames for SCI and REF
    # where we remove beta pic b from the last 21 frames
    r_b = 0.452 / vip.config.VLT_NACO['plsc']  # Absil+2013
    theta_b = 211.0 + 90  # + 104.84 - 0.45 (incl. in derot_angs now) Absil+2013
    flux_b = 550.2
    pl_par = [(r_b, theta_b, flux_b)]
    cube_ref = vip.fm.cube_planet_free(pl_par, cube[idx_sep:],
                                       angles[idx_sep:], psf)

    # creating a variable flux screen
    scr = vip.var.create_synth_psf("moff", (101, 101), fwhm=50)
    scrcu = np.array(
        [scr * i for i in np.linspace(-1e2, 1e2, num=len(cube_ref))],
        dtype=np.float32
    )

    cube_ref += scrcu

    cube_sci = cube[:idx_sep]
    angles_sci = angles[:idx_sep]

    # OLD 1: scaling ?!
    # upscaling (1.2) and taking half of the frames, reversing order
    # cube_upsc = cube_px_resampling(cube[::-1], 1.2, verbose=False)[::2]
    # cropping and adding the flux screen
    # cube_ref = cube_crop_frames(cube_upsc, 101, verbose=False) + scrcu

    # OLD 2: take centro-symmetric images (i.e. flip along both x and y)
    # cube_rot = cube[::-1]
    # cube_rot = cube_rot[::2]
    # cube_rot = np.flip(cube_rot, axis=1)
    # cube_rot = np.flip(cube_rot, axis=2)

    # cropping and adding the flux screen
    # cube_ref = cube_rot + scrcu

    # create dataset object
    dataset = vip.objects.Dataset(
        cube_sci,
        angles=angles_sci,
        psf=psf,
        cuberef=cube_ref,
        px_scale=vip.config.VLT_NACO["plsc"],
    )

    dataset.normalize_psf(size=20, force_odd=False)

    # overwrite PSF for easy access
    dataset.psf = dataset.psfn

    return dataset


@pytest.fixture(autouse=True)
def time_test():
    """Time a test and print out how long it took."""
    before = time.time()
    yield
    after = time.time()
    print(f"Test took {after - before:.02f} seconds!")


@pytest.fixture(autouse=True, scope="session")
def time_all_tests():
    """Time all tests and print out how long they took."""
    before = time.time()
    yield
    after = time.time()
    print(f"Total test time: {after - before:.02f} seconds!")
