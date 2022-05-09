"""
Tests for metrics/contrcurve.py

"""
import copy
from .helpers import fixture, np
from vip_hci.config import VLT_NACO
from vip_hci.psfsub import pca
from vip_hci.metrics import contrast_curve
from vip_hci.preproc import frame_crop
from vip_hci.fm.utils_negfc import find_nearest, cube_planet_free


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


def test_contrast_curve(get_cube):

    ds, starphot = get_cube

    # first empty the cube from planet b
    r_b =  0.452/ds.px_scale # Absil+2013
    theta_b = 211.+90+104.84-0.45 # # Absil+2013
    flux_b = 550.2
    pl_par = [(r_b, theta_b, flux_b)]
    cube = cube_planet_free(pl_par, ds.cube, ds.angles, ds.psf)

    psf = frame_crop(ds.psf[1:, 1:], 11)
    plsc = VLT_NACO['plsc']
    trans = np.zeros([2,10])
    trans[0] = np.linspace(0, cube.shape[-1], 10)
    trans[1,:] = 1
    cc1 = contrast_curve(cube, ds.angles, psf, ds.fwhm, pxscale=plsc,
                         starphot=starphot, algo=pca, nbranch=3, ncomp=9,
                         transmission=trans, plot=True, debug=True)
    cc2 = contrast_curve(cube, ds.angles, psf, ds.fwhm, pxscale=plsc,
                         starphot=starphot, algo=pca, nbranch=3, ncomp=9,
                         transmission=trans, interp_order=None, plot=True, 
                         save_plot='test.pdf', debug=True)
    
    rad = np.array(cc1['distance'])
    gauss_cc1 = np.array(cc1['sensitivity_gaussian'])
    student_cc1 = np.array(cc1['sensitivity_student'])
    gauss_cc2 = np.array(cc2['sensitivity_gaussian'])
    student_cc2 = np.array(cc2['sensitivity_student'])
    sigma_corr = np.array(cc1['sigma corr'])

    # check that at 0.2'' 5-sigma cc < 4e-3 - Gaussian statistics
    idx_r = find_nearest(rad*plsc, 0.2)
    cc_gau = gauss_cc1[idx_r]
    corr_r = sigma_corr[idx_r]
    if cc_gau < 4e-3:
        check = True
    else:
        check = False
    msg = "Contrast too shallow compared to expectations: {} > {}"
    assert check, msg.format(cc_gau, 4e-3)

    # check that at 0.2'' 5-sigma cc: Student statistics > Gaussian statistics
    cc_stu = student_cc1[idx_r]
    if cc_stu < 4e-3*corr_r and cc_stu > cc_gau:
        check = True
    elif cc_stu < 4e-3*corr_r:
        check = False
        msg = "Student-statistics cc smaller than Gaussian statistics cc"
    else:
        check = False
    msg = "Contrast too shallow compared to expectations: {} > {}"
    assert check, msg.format(cc_stu, 4e-3*corr_r)

    # check that at 0.4'' 5-sigma cc < 4e-4
    idx_r = find_nearest(rad*plsc, 0.4)
    cc_gau = gauss_cc2[idx_r]
    corr_r = sigma_corr[idx_r]

    if cc_gau < 4e-4:
        check = True
    else:
        check = False
    msg = "Contrast too shallow compared to expectations: {} > {}"
    assert check, msg.format(cc_gau, 4e-4)

    # check that at 0.4'' 5-sigma cc: Student statistics > Gaussian statistics
    cc_stu = student_cc2[idx_r]
    if cc_stu < 4e-4*corr_r and cc_stu > cc_gau:
        check = True
    elif cc_stu < 4e-4*corr_r:
        check = False
        msg = "Student-statistics cc smaller than Gaussian statistics cc"
    else:
        check = False
        msg = "Contrast too shallow compared to expectations: {} > {}"
    assert check, msg.format(cc_stu, 4e-4*corr_r)
