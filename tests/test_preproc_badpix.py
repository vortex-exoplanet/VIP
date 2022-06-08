"""
Tests for the bad pixel detection and correction routines.

"""

import numpy as np
from vip_hci.preproc import (cube_fix_badpix_isolated, cube_fix_badpix_clump,
                             cube_fix_badpix_annuli, cube_fix_badpix_ifs,
                             cube_fix_badpix_interp)
from vip_hci.var import (create_synth_psf, dist, get_annulus_segments,
                         frame_center)


# isolated bad pixel correction
def test_badpix_iso():
    sz = (25, 25)
    idx0 = 10
    idx1 = 20
    m0 = 0
    m1 = 2
    s0 = 1
    s1 = 1

    im1 = np.random.normal(loc=m0, scale=s0, size=sz)
    im2 = im1 = np.random.normal(loc=m1, scale=s1, size=sz)
    im1[idx0, idx0] = 350
    im2[idx1, idx1] = -420
    cube = np.array([im1, im2])
    cube_c, bpix_map = cube_fix_badpix_isolated(cube, sigma_clip=5,
                                                frame_by_frame=True,
                                                full_output=True)

    # check bad pixels were correctly identified
    assert bpix_map[0, idx0, idx0] == 1
    assert bpix_map[1, idx1, idx1] == 1

    # check they were appropriately corrected
    assert np.abs(cube_c[0, idx0, idx0]-m0) < 3*s0
    assert np.abs(cube_c[1, idx1, idx1]-m1) < 3*s1

    # Test protect mask
    cube_c, bpix_map = cube_fix_badpix_isolated(cube, sigma_clip=5,
                                                frame_by_frame=True,
                                                protect_mask=5, mad=True,
                                                full_output=True)
    # check bad pixels were correctly identified (first one within mask)
    assert bpix_map[0, idx0, idx0] == 0
    assert bpix_map[1, idx1, idx1] == 1
    
    
# isolated bad pixel correction
def test_badpix_iso2():
    sz = (25, 25)
    idx0 = 10
    m0 = 0
    m1 = 2
    s0 = 1
    s1 = 1

    im1 = np.random.normal(loc=m0, scale=s0, size=sz)
    im2 = im1 = np.random.normal(loc=m1, scale=s1, size=sz)
    im1[idx0, idx0] = -435
    im2[idx0, idx0] = -420
    cube = np.array([im1, im2])
    cube_c, bpix_map = cube_fix_badpix_isolated(cube, sigma_clip=5,
                                                frame_by_frame=False,
                                                full_output=True)

    # check bad pixels were correctly identified
    assert bpix_map[idx0, idx0] == 1

# clumpy bad pixel correction
def test_badpix_clump():
    sz = (24, 24)
    idx0 = 10
    idx1 = 20
    m0 = 0
    m1 = 2
    s0 = 1
    s1 = 1

    im1 = np.random.normal(loc=m0, scale=s0, size=sz)
    im2 = im1 = np.random.normal(loc=m1, scale=s1, size=sz)
    im1[idx0, idx0] = 30
    im1[idx0+1, idx0] = 42
    im1[idx0+1, idx0+1] = 54
    im2[idx1, idx1] = -50
    im2[idx1+1, idx1] = -50
    im2[idx1, idx1+1] = -60
    im2[idx1+1, idx1+1] = -60

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
    assert np.abs(cube_c[0, idx0, idx0]-m0) < 4*s0
    assert np.abs(cube_c[0, idx0+1, idx0]-m0) < 4*s0
    assert np.abs(cube_c[0, idx0+1, idx0+1]-m0) < 4*s0
    assert np.abs(cube_c[1, idx1, idx1]-m1) < 4*s1
    assert np.abs(cube_c[1, idx1+1, idx1]-m1) < 4*s1
    assert np.abs(cube_c[1, idx1, idx1+1]-m1) < 4*s1
    assert np.abs(cube_c[1, idx1+1, idx1+1]-m1) < 4*s1

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
    cube_c, bpix_map = cube_fix_badpix_clump(cube[0], sig=5, fwhm=6,
                                             full_output=True)
    assert bpix_map[idx0, idx0] == 1
    assert bpix_map[idx0+1, idx0] == 1
    assert bpix_map[idx0+1, idx0+1] == 1

    # Test half res y option
    cube_c, bpix_map = cube_fix_badpix_clump(cube[1], sig=5, fwhm=6,
                                             half_res_y=True,
                                             protect_mask=5, mad=True,
                                             full_output=True)
    assert bpix_map[idx1, idx1] == 1
    assert bpix_map[idx1+1, idx1] == 1
    assert bpix_map[idx1, idx1+1] == 1
    assert bpix_map[idx1+1, idx1+1] == 1


def test_badpix_ann():
    sz = (24, 24)
    idx0 = 8
    idx1 = 20
    m0 = 0
    s0 = 1

    im1 = 2e4*create_synth_psf(shape=sz, model='airy', fwhm=4)
    im1 += np.random.normal(loc=m0, scale=s0, size=sz)
    im2 = 3e4*create_synth_psf(shape=sz, model='airy', fwhm=4.5)
    im2 += np.random.normal(loc=m0, scale=s0, size=sz)
    im1[idx0, idx0] = -3000
    im1[idx0+1, idx0] = -3000
    im2[idx1, idx1] = -5000
    im2[idx1+1, idx1] = -5000

    cube = np.array([im1, im2])
    cube_c, bpm, _ = cube_fix_badpix_annuli(cube, fwhm=[4, 4.5], sig=5.,
                                            full_output=True)

    assert bpm[0, idx0, idx0] == 1
    assert bpm[0, idx0+1, idx0] == 1
    assert bpm[1, idx1, idx1] == 1
    assert bpm[1, idx1+1, idx1] == 1

    # protect mask+half_res_y
    cube_c, bpm, _ = cube_fix_badpix_annuli(cube, fwhm=[4, 4.5], sig=5.,
                                            protect_mask=7, half_res_y=True,
                                            full_output=True)

    assert bpm[0, idx0, idx0] == 0
    assert bpm[0, idx0+1, idx0] == 0
    assert bpm[1, idx1, idx1] == 1
    assert bpm[1, idx1+1, idx1] == 1

    # test kernel correction
    cy, cx = frame_center(cube)
    cube_c_gau = cube_fix_badpix_interp(cube, bpm, mode='gauss', fwhm=1)
    cube_c_fft = cube_fix_badpix_interp(cube, bpm, mode='fft', nit=50)

    r0 = dist(cy, cx, idx0, idx0)
    ann = get_annulus_segments(cube_c[0], r0-1, 3, mode='val')
    med_val_ann = np.median(ann)
    assert (cube_c_gau[0, idx0, idx0]-med_val_ann) < 4*s0
    assert (cube_c_fft[0, idx0, idx0]-med_val_ann) < 4*s0

    r1 = dist(cy, cx, idx1, idx1)
    ann = get_annulus_segments(cube_c[1], r1-1, 3, mode='val')
    med_val_ann = np.median(ann)
    assert (cube_c_gau[1, idx1, idx1]-med_val_ann) < 4*s0


def test_badpix_ifs1():
    n_ch = 15
    sz = 55
    idx0 = 25
    idx1 = 40
    m0 = 0
    s0 = 1
    fwhms = np.linspace(4, 8, n_ch, endpoint=True)
    fluxes = np.linspace(1, 10, n_ch)*1e4
    cube = np.zeros([n_ch, sz, sz])

    for i in range(n_ch):
        cube[i] = fluxes[i]*create_synth_psf(shape=(sz, sz), model='moff',
                                             fwhm=fwhms[i])
        cube[i] += np.random.normal(loc=m0, scale=s0, size=(sz,sz))
        cube[i, idx0, idx0] = -2200
        cube[i, idx1, idx1] = -2400
        cube[i, idx1+1, idx1] = -2959
        cube[i, idx1+1, idx1+1] = -2960

    # identify bad pixels
    cube_c, bpm, _ = cube_fix_badpix_ifs(cube, lbdas=fwhms/2., clumps=False,
                                         sigma_clip=5., num_neig=9, 
                                         full_output=True)
    assert np.allclose(bpm[:, idx0, idx0], np.ones(n_ch))
    
    
def test_badpix_ifs2():
    n_ch = 15
    sz = 55
    idx0 = 25
    idx1 = 40
    m0 = 0
    s0 = 1
    fwhms = np.linspace(4, 8, n_ch, endpoint=True)
    fluxes = np.linspace(1, 10, n_ch)*1e4
    cube2 = np.zeros([n_ch, sz, sz])

    for i in range(n_ch):
        cube2[i] = fluxes[i]*create_synth_psf(shape=(sz, sz), model='moff',
                                             fwhm=fwhms[i])
        cube2[i] += np.random.normal(loc=m0, scale=s0, size=(sz, sz))
        cube2[i, idx0, idx0] = -600
        cube2[i, idx1, idx1] = -700
        cube2[i, idx1+1, idx1+1] = -660

    # protect mask + clumps
    cube_c2, bpm2, _ = cube_fix_badpix_ifs(cube2, lbdas=fwhms/2., clumps=True,
                                           sigma_clip=4., protect_mask=5,
                                           full_output=True, num_neig=11,
                                           max_nit=5)

    assert np.allclose(bpm2[:, idx0, idx0], np.zeros(n_ch))
    assert np.allclose(bpm2[:, idx1, idx1], np.ones(n_ch))
    assert np.allclose(bpm2[:, idx1+1, idx1+1], np.ones(n_ch))
