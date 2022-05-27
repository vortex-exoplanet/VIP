"""
Tests for fm/negfc*.py (4D IFS+ADI cube)

"""

import copy
from .helpers import aarc, np, parametrize, fixture
from vip_hci.fm import firstguess
from vip_hci.psfsub import pca_annulus


# ====== utility function for injection
@fixture(scope="module")
def injected_cube_position(example_dataset_ifs_crop):
    """
    Inject a fake companion into an example cube.

    Parameters
    ----------
    example_dataset_ifs_crop : fixture
        Taken automatically from ``conftest.py``.

    Returns
    -------
    dsi : VIP Dataset
    injected_position_yx : tuple(y, x)

    """
    dsi = copy.copy(example_dataset_ifs_crop)
    # we chose a shallow copy, as we will not use any in-place operations
    # (like +=). Using `deepcopy` would be safer, but consume more memory.

    gt = (30, 0, 50)  # flux of 50 in all channels
    dsi.inject_companions(gt[2], rad_dists=gt[0], theta=gt[1])

    return dsi, dsi.injections_yx[0], gt


# ====== Actual negfc tests for different parameters
# Note: MCMC too slow to test & doesn't increase much coverage compared to 3D
@parametrize("pca_algo, negfc_algo, ncomp, mu_sigma, fm",
             [
                 (pca_annulus, firstguess, 4, True, None),
                 (pca_annulus, firstguess, 4, False, 'sum')
                 ])
def test_algos(injected_cube_position, pca_algo, negfc_algo, ncomp, mu_sigma,
               fm):
    ds, yx, gt = injected_cube_position

    # run firstguess with simplex only if followed by mcmc or nested sampling
    fwhm_m = np.mean(ds.fwhm)
    res0 = firstguess(ds.cube, ds.angles, ds.psf, ncomp=ncomp,
                      planets_xy_coord=np.array([[yx[1], yx[0]]]), fwhm=fwhm_m,
                      simplex=negfc_algo == firstguess, algo=pca_algo, fmerit=fm,
                      mu_sigma=mu_sigma, imlib='opencv', aperture_radius=2,
                      annulus_width=4*fwhm_m)
    res = (res0[0][0], res0[1][0], np.mean(res0[2][0]))

    # compare results
    aarc(res, gt, rtol=1e-1, atol=2)  # diff within 2 +- 10% gt (for all 3)
