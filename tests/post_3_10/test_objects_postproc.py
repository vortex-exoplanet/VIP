"""Test for the PostProc object."""

__author__ = "Thomas Bédrine"

import numpy as np
import pytest

from requests.exceptions import ReadTimeout
from ratelimit import limits, sleep_and_retry
from astropy.utils.data import download_file

from vip_hci.objects import PostProc, PPResult, Dataset
from vip_hci.fits import open_fits
from vip_hci.config import VLT_NACO


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


class TestParams:
    fwhm = None
    ncomp = None
    delta_rot = None


def test_postproc_object():
    """Create a PostProc object and test various actions on it."""
    t_postproc = PostProc(verbose=False)

    # Test if trying to update the dataset without any specified triggers AttributeError

    with pytest.raises(AttributeError) as excinfo:
        t_postproc._update_dataset()

    assert "No dataset was specified" in str(excinfo.value)

    # Test if trying to get parameters from non-existing results triggers AttributeError

    with pytest.raises(AttributeError) as excinfo:
        t_postproc.get_params_from_results(session_id=-1)

    assert "No results were saved yet" in str(excinfo.value)

    # Test if trying to get parameters for non-existing session triggers ValueError

    results = PPResult()
    t_postproc.results = results

    with pytest.raises(ValueError) as excinfo:
        t_postproc.get_params_from_results(session_id=-1)

    assert "ID is higher" in str(excinfo.value)

    # Test if trying to get parameters for incompatible algorithm triggers ValueError

    v_frame = np.zeros((10, 10))
    v_params = {"param1": None, "param2": None, "param3": 1.0}
    v_algo = "pca"

    results.register_session(frame=v_frame, algo_name=v_algo, params=v_params)

    t_postproc._algo_name = "median_sub"

    with pytest.raises(ValueError) as excinfo:
        t_postproc.get_params_from_results(session_id=-1)

    assert "does not match" in str(excinfo.value)

    # Test if non-overloaded run method triggers NotImplementedError

    with pytest.raises(NotImplementedError) as excinfo:
        t_postproc.run()

    assert excinfo.type == NotImplementedError

    # Test if the create_parameters_dict filters parameters correctly

    t_postproc.ncomp = 10
    t_postproc.fwhm = 4
    t_postproc.delta_rot = 0.5
    expected_class_params = {"ncomp": 10, "fwhm": 4, "delta_rot": 0.5}

    params_dict = t_postproc._create_parameters_dict(TestParams)

    assert expected_class_params == params_dict

    # Test if the update_dataset correctly adds a new dataset to self

    betapic = make_dataset_adi()

    t_postproc._update_dataset(dataset=betapic)

    assert t_postproc.dataset == betapic

    # Test if the explicit_dataset extracts correctly the params from dataset

    t_postproc._explicit_dataset()

    assert np.allclose(np.abs(t_postproc.cube), np.abs(betapic.cube), atol=1e-2)
    assert t_postproc.fwhm == betapic.fwhm
    assert np.allclose(np.abs(t_postproc.angle_list), np.abs(betapic.angles), atol=1e-2)
    assert np.allclose(np.abs(t_postproc.psf), np.abs(betapic.psfn), atol=1e-2)
