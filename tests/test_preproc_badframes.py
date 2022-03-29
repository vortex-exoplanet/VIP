"""
Tests for the badframes detection routines.

"""

import numpy as np
import vip_hci as vip


def test_badfr_corr():
    im1 = vip.var.create_synth_psf(shape=(19, 19), fwhm=2)
    im2 = vip.var.create_synth_psf(shape=(19, 19), fwhm=(2, 4))
    cube = np.array([im1, im2, im1, im2])
    gind, bind = vip.preproc.cube_detect_badfr_correlation(cube, 0, 9,
                                                           plot=False)
    assert np.array_equal(gind, np.array([0, 2]))
    assert np.array_equal(bind, np.array([1, 3]))


def test_badfr_ellip():
    im1 = vip.var.create_synth_psf(shape=(19, 19), fwhm=2)
    im2 = vip.var.create_synth_psf(shape=(19, 19), fwhm=(2, 4))
    cube = np.array([im1, im2, im1, im2])
    gind, bind = vip.preproc.cube_detect_badfr_ellipticity(cube, 2,
                                                           crop_size=17,
                                                           plot=False)
    assert np.array_equal(gind, np.array([0, 2]))
    assert np.array_equal(bind, np.array([1, 3]))


def test_badfr_pxstat():
    im1_1 = vip.var.create_synth_psf(shape=(19, 19), fwhm=3)
    im1_2 = vip.var.create_synth_psf(shape=(19, 19), fwhm=4)
    im1_3 = vip.var.create_synth_psf(model='moff', shape=(19, 19), fwhm=6) - 0.3
    im1_3 = vip.var.mask_circle(im1_3, 4)
    im1 = im1_2 - im1_1
    im2 = im1 + im1_3 + 1
    im3 = im1 + im1_3 + 0.2
    cube = np.array([im1, im1, im1, im1, im1, im1, im3, im2, im2, im3, im1,
                     im1, im1, im1, im1, im1])
    gind, bind = vip.preproc.cube_detect_badfr_pxstats(cube, in_radius=3,
                                                       width=3, window=None,
                                                       plot=False)
    assert np.array_equal(gind, np.array([0,  1,  2,  3,  4,  5,  6,  9, 10,
                                          11, 12, 13, 14, 15]))
    assert np.array_equal(bind, np.array([7, 8]))

