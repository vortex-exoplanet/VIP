"""
Tests for metrics/contrcurve.py

"""
import copy
from .helpers import fixture, np
from vip_hci.config import VLT_NACO
from vip_hci.psfsub.pca_local import pca_annular
from vip_hci.metrics import contrast_curve
from vip_hci.preproc import frame_crop
from vip_hci.fm.utils_negfc import find_nearest

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

    starphot = 764939.6 #Absil et al. (2013)

    return dsi, starphot


def test_contrast_curve(get_cube):
    
    ds, starphot = get_cube
    
    expected_res = np.array([1200/starphot])
    cen = ds.psf.shape[-1]//2
    psf = frame_crop(ds.psf,10,cenxy=[cen,cen])
    plsc = VLT_NACO['plsc']
    cc = contrast_curve(ds.cube, ds.angles, psf, ds.fwhm, pxscale=plsc, 
                        starphot=starphot, algo=pca_annular, nbranch=3,
                        plot=True, debug=True)
    
    rad = np.array(cc['distance'])
    gauss_cc = np.array(cc['sensitivity_gaussian'])
    student_cc = np.array(cc['sensitivity_student'])
    sigma_corr = np.array(cc['sigma corr'])

    
    if np.allclose(cc/expected_res, [1], atol=0.5):
        check=True
    else:
        print(cc)
        check=False
        
    # check that at 0.2'' 5-sigma cc < 4e-4 (Absil+2013) - Gaussian statistics
    idx_r = find_nearest(rad/plsc, 0.2)
    cc_gau = gauss_cc[idx_r]
    if cc_gau < 4e-4:
        check = True
    else:
        check = False
    msg = "Contrast too shallow compared to expectations"
    assert check, msg        
     
    # check that at 0.2'' 5-sigma cc: Student statistics > Gaussian statistics
    cc_stu = student_cc[idx_r]
    if cc_stu < 4e-4*sigma_corr and cc_stu>cc_gau:
        check = True
    elif cc_stu < 4e-4*sigma_corr:
        check = False
        msg = "Student-statistics cc smaller than Gaussian statistics cc"
    else:
        check = False
        msg = "Contrast too shallow compared to expectations"
    assert check, msg
    
    # check that at 0.4'' 5-sigma cc < 1e-4 (Absil+2013)
    idx_r = find_nearest(rad/plsc, 0.4)
    cc_gau = gauss_cc[idx_r]
    if cc_gau < 1e-4:
        check = True
    else:
        check = False
    msg = "Contrast too shallow compared to expectations"
    assert check, msg  
      
    # check that at 0.4'' 5-sigma cc: Student statistics > Gaussian statistics
    cc_stu = student_cc[idx_r]
    if cc_stu < 1e-4*sigma_corr and cc_stu>cc_gau:
        check = True
    elif cc_stu < 1e-4*sigma_corr:
        check = False
        msg = "Student-statistics cc smaller than Gaussian statistics cc"
    else:
        check = False
        msg = "Contrast too shallow compared to expectations"
    assert check, msg