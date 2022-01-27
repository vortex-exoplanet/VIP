"""
Tests for preproc/rescaling.py

"""

__author__ = "Ralf Farkas"

from .helpers import np, aarc, raises, parametrize
from vip_hci.preproc.rescaling import (cube_px_resampling,
                                       frame_px_resampling,
                                       cube_rescaling_wavelengths,
                                       check_scal_vector,
                                       _find_indices_sdi)
from vip_hci.var import mask_circle

CUBE = np.ones((10, 100, 100))
FRAME = np.ones((100, 100))


@parametrize("imlib", ["vip-fft", "ndimage", "opencv"])
def test_cube_px_resampling(imlib):

    # === enlargen ===

    res = cube_px_resampling(CUBE, scale=2, imlib=imlib)
    assert res.shape == (10, 200, 200)

    # === shrink ===

    res = cube_px_resampling(CUBE, scale=0.5, imlib=imlib)
    assert res.shape == (10, 50, 50)


@parametrize("imlib,keep_center", 
             [("vip-fft", False),
              ("ndimage", False), 
              ("opencv", False),
              ("vip-fft", True),
              ("ndimage", True), 
              ("opencv", True)])
def test_frame_px_resampling(imlib, keep_center):
    """

    Notes
    -----
    Testing of the different ``interpolation`` is not needed, we rely on the
    external modules here. The calling signature is the same, so there is not
    much VIP can do.
    """

    # === enlargen ===
    frame_star = FRAME.copy()
    frame_star[48:53,48:53] = 2
    frame_star[49:52,49:52] = 4
    frame_star[50,50] = 8
    res = frame_px_resampling(frame_star, scale=2, imlib=imlib, 
                              keep_center=keep_center, verbose=True)
    assert res.shape == (200, 200)

    if keep_center:
        # further check whether max pixel (star proxy) is still centered
        max_idx = np.argmax(res)
        max_coords = np.unravel_index(max_idx,res.shape)
        assert max_coords == (100,100)

    # === enlargen with mask ===
    res = frame_px_resampling(mask_circle(FRAME, 4, np.nan), scale=2, 
                              imlib=imlib, keep_center=keep_center,
                              verbose=True)
    assert res.shape == (200, 200)

    # === shrink ===
    res = frame_px_resampling(FRAME, scale=0.5, imlib=imlib, 
                              keep_center=keep_center, verbose=True)
    assert res.shape == (50, 50)

    

@parametrize("imlib,interpolation",
             [
                 ("vip-fft", None),
                 ("opencv", "lanczos4"),
                 ("ndimage", "bicubic"),
             ])
def test_cube_rescaling_wavelengths(imlib, interpolation):
    scal_list = np.arange(10) + 1  # no zero

    # === basic function ===

    res1 = cube_rescaling_wavelengths(CUBE, scal_list, imlib=imlib,
                                      interpolation=interpolation)
    cube1, med1, y1, x1, cy1, cx1 = res1

    assert cube1.shape == (10, 1000, 1000)  # frame size x10 x10

    for i in range(cube1.shape[0]):
        aarc(cube1[i].mean() * scal_list[i]**2, 1)

    # === undo ===

    res2 = cube_rescaling_wavelengths(cube1, scal_list, imlib=imlib,
                                      interpolation=interpolation,
                                      inverse=True, x_in=100, y_in=100)
    cube2, med2, y2, x2, cy2, cx2 = res2

    aarc(cube2, CUBE, 1e-2, 1e-2)


def test_check_scal_vector():
    scal_vec = np.array([2, 8, 4])

    # === basic function ===

    res = check_scal_vector(scal_vec)
    truth = np.array([4, 1, 2])

    aarc(res, truth)

    # === do nothing if min factor is already 1 ===

    res2 = check_scal_vector(res)
    aarc(res2, res)

    # === wrong input value ===
    with raises(TypeError):
        check_scal_vector(42)


@parametrize("dist, index_ref, truth",
             [
                (6, 0, [2, 3, 4, 5, 6, 7, 8, 9]),
                (6, 5, [0, 1, 2]),
                (10, 0, [1, 2, 3, 4, 5, 6, 7, 8, 9]),
                (10, 5, [0, 1, 2, 3, 9]),
                (15, 5, [0, 1, 2, 3, 8, 9]),
                (20, 0, [1, 2, 3, 4, 5, 6, 7, 8, 9]),
                (20, 5, [0, 1, 2, 3, 4, 7, 8, 9])
             ])
def test_find_indices_sdi(dist, index_ref, truth):
    wl = np.arange(10) + 1
    fwhm = 4

    indices = _find_indices_sdi(wl, dist, index_ref, fwhm, debug=True)

    aarc(indices, truth)
