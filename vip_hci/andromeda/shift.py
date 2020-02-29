"""
subpixel shifting.
"""


__author__ = 'Ralf Farkas'
__all__ = []


import numpy as np


def calc_psf_shift_subpix(psf, precision):
    """
    Compute a stack of subpixel-shifted versions of the PSF.

    Parameters
    ----------
    psf : 2d numpy ndarray
        The PSF that is to be shifted. Assumed square.
    precision : int
        Number of pixel subdivisions for the planet's signal pattern
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

    based on ``LibAndromeda/oneralib/calc_psf_shift_subpix.pro``,
    v1.2 2010/05/27

    The IDL version has an optional keyword ``INTERP_INPUT``, which is not used
    in andromeda. It was not implemented.

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
    image : 2d numpy ndarray
        The image to be shifted.
    xshift : float
        Amount of desired shift in X direction.
    yshift : float
        Amount of desired shift in Y direction.

    Returns
    -------
    shifted_image : 2d ndarray
        Input ``image`` shifted by ``xshift`` and ``yshift``.

    Notes
    -----
    based on ``LibAndromeda/oneralib/subpixel_shift.pro``, v1.3 2009/05/28

    """
    npix = image.shape[0]

    if npix != image.shape[1]:
        raise ValueError("`image` must be square")

    ramp = np.outer(np.ones(npix), np.arange(npix) - npix/2)
    tilt = (-2*np.pi / npix) * (xshift*ramp + yshift*ramp.T)
    fact = np.fft.fftshift(np.cos(tilt) + 1j*np.sin(tilt))

    image_ft = np.fft.fft2(image)  # no np.fft.fftshift applied!
    shifted_image = np.fft.ifft2(image_ft * fact).real

    return shifted_image
