"""
Tests for preproc/rescaling.py

"""

__author__ = "Valentin Christiaens"

from .helpers import np, aarc, parametrize
from vip_hci.preproc.derotation import (cube_derotate, _define_annuli, 
                                        _find_indices_adi)
from vip_hci.preproc import cube_crop_frames
from vip_hci.var import mask_circle


CUBE_even = np.ones((4, 80, 80))
CUBE_odd = np.ones((4, 81, 81))


@parametrize("imlib,interpolation,border_mode,edge_blend",
             [
                 ("vip-fft", None, 'constant', None),
                 ("vip-fft", None, 'reflect', 'noise'),
                 ("opencv", "lanczos4", 'edge', 'interp+noise'),
                 ("skimage", "bicubic", 'symmetric', 'interp+noise'),
                 ("skimage", "biquintic", 'wrap', 'noise'),
             ])
def test_cube_derotate(imlib, interpolation, border_mode, edge_blend):
    """
    Note: this calls most other routines defined in rotation.py
    """
    # even
    res = CUBE_even.copy()
    angles = np.array([120,90,60,45])
    for i in range(24): # yields 360deg multiples for all derotation angles
        res = cube_derotate(res, angles, imlib=imlib, border_mode=border_mode, 
                            edge_blend=edge_blend, nproc=1+(i%2))
    # just consider cropped image for evaluation
    # to avoid differences due to edge effects
    res = cube_crop_frames(res, 50)
    CUBE_test = cube_crop_frames(CUBE_even, 50)
    aarc(res, CUBE_test, rtol=1e-1, atol=1e-1)
    
    # odd
    res = CUBE_odd.copy()
    angles = np.array([120,90,60,45])
    for i in range(24): # yields 360deg multiples for all derotation angles
        res = cube_derotate(res, angles, imlib=imlib, border_mode=border_mode, 
                            edge_blend=edge_blend, nproc=1+(i%2))
    # just consider cropped image for evaluation
    # to avoid differences due to edge effects
    res = cube_crop_frames(res, 51)
    CUBE_test = cube_crop_frames(CUBE_odd, 51)
    aarc(res, CUBE_test, rtol=1e-1, atol=1e-1)
    
    
@parametrize("imlib,interpolation,border_mode,edge_blend,interp_zeros",
              [
                  ("vip-fft", None, 'reflect', 'interp', True),
                  ("opencv", "lanczos4", 'edge', 'interp', True),
                  ("skimage", "biquintic", 'reflect', 'interp', True),
              ])   
def test_cube_derotate_mask(imlib, interpolation, border_mode, edge_blend,
                            interp_zeros):
    """
    Note: this calls most other routines defined in rotation.py
    """
    # mask with nans
    CUBE_odd_mask = mask_circle(CUBE_odd, 4, np.nan)
    res = CUBE_odd_mask.copy()
    angles = np.array([180,120,90,60])
    for i in range(6): # yields 360deg multiples for all derotation angles
        res = cube_derotate(res, angles, border_mode=border_mode, imlib=imlib, 
                            interpolation=interpolation, nproc=1+(i%2),
                            mask_val=np.nan, edge_blend=edge_blend, 
                            interp_zeros=False)
    # just consider cropped image for evaluation
    # to avoid differences due to edge effects
    res = cube_crop_frames(res, 51)
    CUBE_test = cube_crop_frames(CUBE_odd_mask, 51)
    aarc(res, CUBE_test, rtol=1e-1, atol=1e-1)
    
    # mask with zeros
    CUBE_odd_mask = mask_circle(CUBE_odd, 4, 0)
    res = CUBE_odd_mask.copy()
    angles = np.array([180,120,90,60])
    for i in range(6): # yields 360deg multiples for all derotation angles
        res = cube_derotate(res, angles, border_mode=border_mode, imlib=imlib, 
                            interpolation=interpolation, nproc=1+(i%2),
                            mask_val=0, edge_blend=edge_blend, 
                            interp_zeros=interp_zeros)
    # just consider cropped image for evaluation
    # to avoid differences due to edge effects
    res = cube_crop_frames(res, 51)
    CUBE_test = cube_crop_frames(CUBE_odd_mask, 51)
    aarc(res, CUBE_test, rtol=1e-1, atol=1e-1)


def test_define_annuli():
    angles = np.array([120,90,60,30,0])
    ann = 0
    n_annuli = 10
    fwhm = 4
    radius_int = 2
    annulus_width = 4
    delta_rot = 1
    n_segments = 1
    verbose=True
    res = _define_annuli(angles, ann, n_annuli, fwhm, radius_int, annulus_width,
                         delta_rot, n_segments, verbose, strict=False)

    pa_threshold, inner_radius, ann_center = res

    aarc(pa_threshold, 53.13, rtol=1e-1, atol=1)
    aarc(inner_radius, radius_int + ann * annulus_width, rtol=1e-1, atol=1)
    aarc(ann_center, inner_radius + annulus_width/2, rtol=1e-1, atol=1)


@parametrize("frame, nframes, out_closest, truncate, max_frames, truth",
             [
                (0, None, 0, 0, 10, [3,4,5,6]),
                (1, None, 0, 0, 10, [3,4,5,6]),
                (2, None, 0, 0, 10, [4,5,6]),
                (3, None, 0, 0, 10, [0,1,5,6]),
                (3, None, 1, 0, 10, (2,4)),
                (3, 2, 0, 0, 10, [1, 5]),
                (3, None, 0, 1, 3, [1, 5,6])])
def test_find_indices_adi(frame, nframes, out_closest, truncate, max_frames, 
                          truth):
    angles = np.array([130,120,90,60,30,10,0])
    
    indices = _find_indices_adi(angles, frame=frame, thr=42, nframes=nframes, 
                                out_closest=out_closest, truncate=truncate, 
                                max_frames=max_frames)

    aarc(indices, truth)
