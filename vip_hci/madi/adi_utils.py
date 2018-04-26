#! /usr/bin/env python

"""
Module with ADI helper functions.
"""

from __future__ import division, print_function

__author__ = 'Carlos Alberto Gomez Gonzalez'
__all__ = []

import numpy as np


def _find_indices(angle_list, frame, thr, nframes=None, out_closest=False,
                  truncate=False, max_frames=200):
    """ Returns the indices to be left in frames library for annular ADI median
    subtraction, LOCI or annular PCA.

    # TODO: find a more pythonic way to to this!

    Parameters
    ----------
    angle_list : array_like, 1d
        Vector of parallactic angle (PA) for each frame.
    frame : int
        Index of the current frame for which we are applying the PA threshold.
    thr : float
        PA threshold.
    nframes : int or None, optional
        Number of indices to be left. For annular ADI median subtraction,
        where we keep the closest frames (after the PA threshold). If None then
        all the indices are returned (after the PA threshold).
    out_closest : bool, optional
        If True then the function returns the indices of the 2 closest frames.
    truncate : bool, optional
        Useful for annular PCA, when we want to discard too far away frames and
        avoid increasing the computational cost.
    max_frames : int, optional
        Max frames to leave if ``truncate`` is True.

    Returns
    -------
    indices : array_like, 1d
        Vector with the indices left.

    If ``out_closest`` is True then the function returns instead:
    index_prev, index_foll
    """
    n = angle_list.shape[0]
    index_prev = 0
    index_foll = frame
    for i in range(0, frame):
        if np.abs(angle_list[frame] - angle_list[i]) < thr:
            index_prev = i
            break
        else:
            index_prev += 1
    for k in range(frame, n):
        if np.abs(angle_list[k] - angle_list[frame]) > thr:
            index_foll = k
            break
        else:
            index_foll += 1

    if out_closest:
        return index_prev, index_foll-1
    else:
        if nframes is not None:
            # For annular ADI median subtraction, returning n_frames closest
            # indices (after PA thresholding)
            window = nframes // 2
            ind1 = index_prev-window
            ind1 = max(ind1, 0)
            ind2 = index_prev
            ind3 = index_foll
            ind4 = index_foll+window
            ind4 = min(ind4, n)
            indices = np.array(list(range(ind1, ind2)) +
                               list(range(ind3, ind4)))
        else:
            # For annular PCA, returning all indices (after PA thresholding)
            half1 = range(0, index_prev)
            half2 = range(index_foll, n)

            # This truncation is done on the annuli after 10*FWHM and the goal
            # is to keep min(num_frames/2, 200) in the library after discarding
            # those based on the PA threshold
            if truncate:
                thr = min(n//2, max_frames)
                if frame < thr:
                    half1 = range(max(0, index_prev - thr//2), index_prev)
                    half2 = range(index_foll,
                                  min(index_foll + thr - len(half1), n))
                else:
                    half2 = range(index_foll, min(n, thr//2 + index_foll))
                    half1 = range(max(0, index_prev - thr + len(half2)),
                                  index_prev)
            indices = np.array(list(half1) + list(half2))

        return indices


def _compute_pa_thresh(ann_center, fwhm, delta_rot=1):
    """ Computes the parallactic angle theshold[degrees]
    Replacing approximation: delta_rot * (fwhm/ann_center) / np.pi * 180
    """
    return np.rad2deg(2 * np.arctan(delta_rot * fwhm / (2 * ann_center)))


def _define_annuli(angle_list, ann, n_annuli, fwhm, radius_int, annulus_width,
                   delta_rot, n_segments, verbose):
    """ Function that defines the annuli geometry using the input parameters.
    Returns the parallactic angle threshold, the inner radius and the annulus
    center for each annulus.
    """
    if ann == n_annuli - 1:
        inner_radius = radius_int + (ann * annulus_width - 1)
    else:
        inner_radius = radius_int + ann * annulus_width
    ann_center = inner_radius + (annulus_width / 2)
    pa_threshold = _compute_pa_thresh(ann_center, fwhm, delta_rot)

    mid_range = np.abs(np.amax(angle_list) - np.amin(angle_list)) / 2
    if pa_threshold >= mid_range - mid_range * 0.1:
        new_pa_th = float(mid_range - mid_range * 0.1)
        if verbose:
            msg = 'PA threshold {:.2f} is too big, will be set to {:.2f}'
            print(msg.format(pa_threshold, new_pa_th))
        pa_threshold = new_pa_th

    if verbose:
        msg2 = 'Annulus {}, PA thresh = {:.2f}, Inn radius = {:.2f}, '
        msg2 += 'Ann center = {:.2f}, N segments = {} '
        print(msg2.format(ann+1, pa_threshold, inner_radius, ann_center,
                          n_segments))
    return pa_threshold, inner_radius, ann_center