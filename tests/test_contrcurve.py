"""
Tests for metrics/contrcurve.py

"""
import copy
from .helpers import fixture, np
from vip_hci.psfsub.pca_local import pca_annular
from vip_hci.metrics import completeness_curve,completeness_map
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

    return dsi

def test_completeness_curve(get_cube):
    
    ds = get_cube
    
    excpected_res=np.array([[500]])
    cen=ds.psf.shape[-1]//2
    psf=frame_crop(ds.psf,10,cenxy=[cen,cen])
    an_dist,comp_curve = completeness_curve(ds.cube,ds.angles,psf,    
                                    ds.fwhm,pca_annular,an_dist=[20],
                                    ini_contrast=excpected_res,plot=False)
    
    if np.allclose(comp_curve/excpected_res, [1], atol=0.5):
        check=True
    else:
        print(comp_curve)
        check=False
        
    msg = "Issue with completeness curve estimation"
    assert check, msg

def test_completeness_map(get_cube):
    
    ds = get_cube
    
    excpected_res=np.array([[1200]])
    cen=ds.psf.shape[-1]//2
    psf=frame_crop(ds.psf,10,cenxy=[cen,cen])
    an_dist,comp_map = completeness_map(ds.cube,ds.angles,psf,    
                                    ds.fwhm,pca_annular,an_dist=[20],
                                    ini_contrast=excpected_res)
    
    if np.allclose(comp_map[:,-2]/excpected_res, [1], atol=0.5):
        check=True
    else:
        print(comp_map[:,-2])
        check=False
        
    msg = "Issue with completeness map estimation"
    assert check, msg



    
    
    
