"""
Tests for the bad pixel detection and correction routines.

"""

import numpy as np
from vip_hci.preproc import (cube_fix_badpix_isolated, cube_fix_badpix_clump, 
                             cube_fix_badpix_annuli, cube_fix_badpix_ifs)
from vip_hci.var import create_synth_psf


# isolated bad pixel correction
def test_badpix_iso():
    sz = (25,25)
    idx0 = 10
    idx1 = 20
    m0 = 0
    m1 = 2
    s0 = 1
    s1 = 1
    
    im1 = np.random.normal(loc=m0, scal=s0, size=sz)
    im2 = im1 = np.random.normal(loc=m1, scal=s1, size=sz)
    im1[idx0,idx0] = 10
    im2[idx1,idx1] = -5
    cube = np.array([im1, im2])
    cube_c, bpix_map = cube_fix_badpix_isolated(cube, sigma_clip=5,
                                                frame_by_frame=True,
                                                full_output=True)
                    
    # check bad pixels were correctly identified
    assert bpix_map[0, idx0, idx0] == 1
    assert bpix_map[1, idx1, idx1] == 1
    
    # check they were appropriately corrected                                     
    assert np.abs(cube_c[0, idx0, idx0]-m0)<2*s0
    assert np.abs(cube_c[1, idx1, idx1]-m1)<2*s1
    
    # Test protect mask
    cube_c, bpix_map = cube_fix_badpix_isolated(cube, sigma_clip=7,
                                                frame_by_frame=True,
                                                protect_mask=5, mad=True,
                                                full_output=True)
    # check bad pixels were correctly identified (first one within mask)
    assert bpix_map[0, idx0, idx0] == 0
    assert bpix_map[1, idx1, idx1] == 1
    
    
    
# clumpy bad pixel correction
def test_badpix_clump():
    sz = (24,24)
    idx0 = 10
    idx1 = 20
    m0 = 0
    m1 = 2
    s0 = 1
    s1 = 1
    
    im1 = np.random.normal(loc=m0, scal=s0, size=sz)
    im2 = im1 = np.random.normal(loc=m1, scal=s1, size=sz)
    im1[idx0,idx0] = 10
    im1[idx0+1,idx0] = 12
    im1[idx0+1,idx0+1] = 14
    im2[idx1,idx1] = -20
    im2[idx1+1,idx1] = 20
    im2[idx1,idx1+1] = -50
    im2[idx1+1,idx1+1] = -50
    
    cube = np.array([im1, im2])
    cube_c, bpix_map = cube_fix_badpix_clump(cube, sig=5, fwhm=6, 
                                             full_output=True)
                    
    # check bad pixels were correctly identified
    assert bpix_map[0, idx0, idx0] == 1
    assert bpix_map[0, idx0+1, idx0] == 1
    assert bpix_map[0, idx0+1, idx0+1] == 1
    assert bpix_map[1, idx1, idx1] == 1
    assert bpix_map[1, idx1+1, idx1] == 1
    assert bpix_map[1, idx1, idx1+1] == 1
    assert bpix_map[1, idx1+1, idx1+1] == 1
    
    # check they were appropriately corrected                                     
    assert np.abs(cube_c[0, idx0, idx0]-m0)<2*s0
    assert np.abs(cube_c[0, idx0+1, idx0]-m0)<2*s0
    assert np.abs(cube_c[0, idx0+1, idx0+1]-m0)<2*s0
    assert np.abs(cube_c[1, idx1, idx1]-m1)<2*s1
    assert np.abs(cube_c[1, idx1+1, idx1]-m1)<2*s1
    assert np.abs(cube_c[1, idx1, idx1+1]-m1)<2*s1
    assert np.abs(cube_c[1, idx1+1, idx1+1]-m1)<2*s1
    
    # Test protect mask + mad
    cube_c, bpix_map = cube_fix_badpix_clump(cube, sig=5, fwhm=6, 
                                             protect_mask=5, mad=True,
                                             full_output=True)
    # check bad pixels were correctly identified (first ones within mask)
    assert bpix_map[0, idx0, idx0] == 0
    assert bpix_map[0, idx0+1, idx0] == 0
    assert bpix_map[0, idx0+1, idx0+1] == 0
    assert bpix_map[1, idx1, idx1] == 1
    assert bpix_map[1, idx1+1, idx1] == 1
    assert bpix_map[1, idx1, idx1+1] == 1
    assert bpix_map[1, idx1+1, idx1+1] == 1
    
    # Test 2d input
    cube_c, bpix_map = cube_fix_badpix_isolated(cube[0], sig=5, fwhm=6, 
                                                full_output=True)
    assert bpix_map[idx0, idx0] == 0
    assert bpix_map[idx0+1, idx0] == 0
    assert bpix_map[idx0+1, idx0+1] == 0
    
    # Test half res y option
    cube_c, bpix_map = cube_fix_badpix_isolated(cube[1], sig=5, fwhm=6, 
                                                half_res_y = True,
                                                protect_mask = 5, mad = True,
                                                full_output = True)
    assert bpix_map[idx0, idx0] == 1
    assert bpix_map[idx0+1, idx0] == 1
    assert bpix_map[idx0, idx0+1] == 1
    assert bpix_map[idx0+1, idx0+1] == 1
    

def test_badpix_ann():
    sz = (25,25)
    idx0 = 10
    idx1 = 20
    m0 = 0
    s0 = 1
    
    im1 = 2e4*create_synth_psf(shape=sz, model='airy', fwhm=4)
    im1 += np.random.normal(loc=m0, scal=s0, size=sz)
    im2 = 3e4*create_synth_psf(shape=sz, model='airy', fwhm=4.5)
    im2 += np.random.normal(loc=m0, scal=s0, size=sz)
    im1[idx0,idx0] = -10
    im1[idx0+1,idx1] = -10
    im2[idx1,idx1] = -20
    im2[idx1+1,idx1] = -20
    
    cube = np.array([im1, im2])
    cube_c, bpm, _ = cube_fix_badpix_annuli(cube, fwhm=[4,4.5], sig=5., 
                                            full_output=True)
    
    assert bpm[0, idx0, idx0] == 1
    assert bpm[0, idx0+1, idx0] == 1
    assert bpm[1, idx1, idx1] == 1
    assert bpm[1, idx1+1, idx1] == 1
    
    # protect mask+half_res_y
    cube_c, bpm, _ = cube_fix_badpix_annuli(cube, fwhm=[4,4.5], sig=5.,
                                            protect_mask=5, half_res_y=True, 
                                            full_output=True)
    
    assert bpm[0, idx0, idx0] == 0
    assert bpm[0, idx0+1, idx0] == 0
    assert bpm[1, idx1, idx1] == 1
    assert bpm[1, idx1+1, idx1] == 1


def test_badpix_ifs():
    n_ch = 5
    sz=25
    idx0 = 10
    idx1 = 20
    m0 = 0
    s0 = 1
    fwhms = np.linspace(2, 4, n_ch, endpoint=True)
    fluxes = np.array([2, 4, 5, 3., 1])*1e4
    cube = np.zeros([n_ch, sz, sz])

    
    for i in range(n_ch):
        cube[i] = fluxes[i]*create_synth_psf(shape=(sz, sz), model='moffat', 
                                             fwhm=fwhms[i])
        cube[i] += np.random.normal(loc=m0, scal=s0, size=(sz,sz))
        cube[i, idx0, idx0] = -10
        cube[i, idx1, idx1] = -50
        cube[i, idx1+1, idx1] = -20
        cube[i, idx1+1, idx1+1] = -40


    # identify bad pixels
    cube_c, bpm, _ = cube_fix_badpix_ifs(cube, lbdas=fwhms/2., clumps=False, 
                                         sigma_clip=5., full_output=True,
                                         max_nit=5)
    assert np.allclose(bpm[:, idx0, idx0], np.ones(n_ch))
    assert np.allclose(bpm[:, idx1, idx1], np.ones(n_ch))
            
    # protect mask + clumps
    cube_c, bpm, _ = cube_fix_badpix_ifs(cube, lbdas=fwhms/2., clumps=True, 
                                         sigma_clip=5., protect_mask=5,
                                         full_output=True, max_nit=5)
    
    assert np.allclose(bpm[:, idx0, idx0], np.ones(n_ch))
    assert np.allclose(bpm[:, idx1, idx1], np.ones(n_ch))
    assert np.allclose(bpm[:, idx1+1, idx1], np.ones(n_ch))
    assert np.allclose(bpm[:, idx1+1, idx1+1], np.ones(n_ch))