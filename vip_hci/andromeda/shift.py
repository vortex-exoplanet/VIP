"""
subpixel shifting.
"""
from __future__ import division, print_function

__author__ = 'Ralf Farkas'
__all__ = ['calc_psf_shift_subpix']


import numpy as np


def calc_psf_shift_subpix(psf, precision):
    """
    Computes a stack of subpixel-shifted versions of the PSF.

    

    Parameters
    ----------
    psf : 2d array_like
        The PSF that is to be shifted. Assumed square.
    precision : int
        number of pixel subdivisions for the planet's signal pattern
        computation, i.e., inverse of the shifting pitch


    Returns
    -------
    psf_cube : 4d ndarray
        4d array that contains all the shifted versions of the PSF. The first 
        two indices contain the fraction of the shift in the x and y directions,
        the last two refer to the spatial position on the grid. The shape
        of the array is (precision+1, precision+1, n, n), where n is the
        side of the PSF used. The image ``psf_cube[i, j, :, :]`` thus
        corresponds to ``psf`` shifted by ``i/precision`` in x and
        ``j/precision`` in y.

    Notes
    -----
    The IDL implementation has an inversed shape of
    (n, n, precision+1, precision+1), and indexing works like
    ``psf_cube[*, *, i, j]``, where ``i`` is the column and ``j`` the row.

    based on `LibAndromeda/oneralib/calc_psf_shift_subpix.pro`, v1.2 2010/05/27


    """

    n = psf.shape[0]
    psf_cube = np.zeros((precision+1, precision+1, n, n))

    for i_column in range(precision+1):
        decalx = i_column/precision
        for j_row in range(precision+1):
            decaly = j_row/precision
            psf_cube[j_row, i_column] = subpixel_shift(psf, decalx, decaly)

    return psf_cube



def subpixel_shift(image, xshift, yshift):
    """
    Subpixel shifting of ``image`` using fourier transformation.

    Parameters
    ----------
    image : 2d array
        The image to be shifted.
    xshift : float
        Amount of desired shift in X direction
    yshift : float
        Amount of desired shift in Y direction

    Notes
    -----
    based on `LibAndromeda/oneralib/subpixel_shift.pro`, v 1.3 2009/05/28

    """

    npix = image.shape[0]

    if image.shape[0] != image.shape[1]:
        raise ValueError("`image` must be square")

    if image.shape[0]%2 != 0:
        pass

    image_ft = np.fft.fft2(image) # no np.fft.fftshift applied!
    ramp = np.outer(np.ones(npix), np.arange(npix) - npix/2)
    tilt = (-2*np.pi / npix) * (xshift*ramp + yshift*ramp.T)
    shift_fact_fft = np.fft.fftshift(np.cos(tilt) + 1j*np.sin(tilt))

    shifted_image = np.fft.ifft2(image_ft * shift_fact_fft).real
    # TODO: real or abs?

    return shifted_image

