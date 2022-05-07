"""
Tests for metrics/completeness.py

"""
import copy
from .helpers import fixture, np
from vip_hci.psfsub import pca
from vip_hci.metrics import completeness_curve, completeness_map
from vip_hci.preproc import frame_crop


@fixture(scope="module")
def get_cube(example_dataset_adi):
    """
    Get the ADI sequence from conftest.py.

    Parameters
    ----------
    example_dataset_adi : fixture
        Taken automatically from ``conftest.py``.

    Returns
    -------
    dsi : VIP Dataset

    """
    dsi = copy.copy(example_dataset_adi)
    starphot = 764939.6  # Absil et al. (2013)

    return dsi, starphot


def test_completeness_curve(get_cube):

    ds, starphot = get_cube

    expected_res = np.array([0.00052709])
    psf = frame_crop(ds.psf[1:, 1:], 11)
    # Note: setting ini_contrast to None calls the contrast_curve function
    an_dist, comp_curve = completeness_curve(ds.cube, ds.angles, psf,
                                             ds.fwhm, pca, an_dist=[20],
                                             ini_contrast=None,  # expected_res,
                                             starphot=starphot, plot=True, 
                                             algo_dict={'imlib':'opencv'})

    if np.allclose(comp_curve/expected_res, [1], atol=0.5):
        check = True
    else:
        print(comp_curve)
        check = False

    msg = "Issue with completeness curve estimation"
    assert check, msg


def test_completeness_map(get_cube):

    ds, starphot = get_cube

    expected_res = np.array([0.00042915])
    psf = frame_crop(ds.psf, 11, force=True)
    an_dist, comp_map = completeness_map(ds.cube, ds.angles, psf, ds.fwhm, pca, 
                                         an_dist=[20], 
                                         ini_contrast=expected_res,
                                         starphot=starphot,
                                         algo_dict={'imlib':'opencv'})

    if np.allclose(comp_map[:, -2]/expected_res, [1], atol=0.5):
        check = True
    else:
        print(comp_map[:, -2])
        check = False

    msg = "Issue with completeness map estimation"
    assert check, msg
