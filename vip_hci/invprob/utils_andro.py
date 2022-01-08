"""
Util functions for ANDROMEDA.
"""


__author__ = 'Ralf Farkas'
__all__ = []


import numpy as np


def robust_std(x):
    """
    Calculate and return the *robust* standard deviation of a point set.

    Corresponds to the *standard deviation* of the point set, without taking
    into account the outlier points.

    Parameters
    ----------
    x : numpy ndarray
        point set

    Returns
    -------
    std : np.float64
        robust standard deviation of the point set

    Notes
    -----
    based on ``LibAndromeda/robust_stddev.pro`` v1.1 2016/02/16

    """
    median_absolute_deviation = np.median(np.abs(x - np.median(x)))
    return median_absolute_deviation / 0.6745


def idl_round(x):
    """
    Round to the *nearest* integer, half-away-from-zero.

    Parameters
    ----------
    x : array-like
        Number or array to be rounded

    Returns
    -------
    r_rounded : array-like
        note that the returned values are floats

    Notes
    -----
    IDL ``ROUND`` rounds to the *nearest* integer (commercial rounding),
    unlike numpy's round/rint, which round to the nearest *even*
    value (half-to-even, financial rounding) as defined in IEEE-754
    standard.

    """
    return np.trunc(x + np.copysign(0.5, x))


def idl_where(array_expression):
    """
    Return a list of indices matching the ``array_expression``.

    Port of IDL's ``WHERE`` function.

    Parameters
    ----------
    array_expression : numpy ndarray / expression
        an expression like ``array > 0``

    Returns
    -------
    res : ndarray
        list of 'good' indices


    Notes
    -----
    - The IDL version returns ``[-1]`` when no match was found, this function
      returns ``[]``, which is more "pythonic".

    """
    res = np.array([i for i, e in enumerate(array_expression.flatten()) if e])
    return res


def fitaffine(x, y, debug=False):
    """
    Optimize affine equation ``y=bx+a`` in a robust way.

    This procedure calculates the best parameters ``a`` and ``b`` that optimize
    the affine equation ``y=bx+a`` in a robust way from a set of point
    ``(xi, yi)``.


    Parameters
    ----------
    x,y : 1d numpy ndarray
        The data to be fitted in a robust affine optimisation. Note the
        "reversed" order of the parameters, compared to the IDL implementation
        (see notes).
    debug : bool, optional
        Show debug output.

    Returns
    -------
    b,a : floats
        parameters which satisfy ``y = bx + a``.
        ``b`` corresponds to ``gamma`` in ANDROMEDA, ``a`` is ``gamma_prime``.

    Notes
    -----
    - ported and adapted from ``LibAndromeda/fitaffine.pro`` v1.1 2016/02/16
    - IDL version adapted from "Numerical Recipies (3rd, 2007)", p.818
    - the ``abdev`` return value was removed, as not used in ANDROMEDA.

    Pay attention to the order of the parameters!

    .. code:: IDL

        FITAFFINE, DATA_INPUT=[[y],[x]], GAMMA_OUTPUT=g1

    .. code:: python

        g1 = fitaffine(x, y)

    """
    ndata = x.shape[0]

    if debug:
        print("FITAFFINE: ***next dataset***")

    # first guess for a and b (LS):
    sx = np.sum(x)
    sy = np.sum(y)
    sxy = np.sum(x*y)
    sxx = np.sum(x**2)

    delta = ndata*sxx - sx**2
    a_ls = (sxx*sy - sx*sxy)/delta
    b_ls = (ndata*sxy - sx*sy)/delta

    if debug:
        print("FITAFFINE: first guess LS: {} + {} x".format(a_ls, b_ls))

    # chi-square to choose the first iteration step:
    chisq = np.sum((y - (a_ls + b_ls*x))**2)
    sigb = np.sqrt(chisq/delta)

    # guessed bracket at 3 sigma for b:
    a = a_ls
    b = b_ls
    b1 = b_ls
    f1, a = rofunc(x=x, y=y, b=b1)
    if debug:
        print("FITAFFINE: entering iteration loop")

    if sigb > 0 and f1 != 0:
        if f1 > 0:
            b2 = b1 + 3*sigb
        else:
            b2 = b1 - 3*sigb

        f2, a = rofunc(x, y, b=b2)

        # bracketing
        while f1*f2 > 0:
            b = b2 + 1.6*(b2-b1)
            b1 = b2
            f1 = f2
            b2 = b
            f2, a = rofunc(x, y, b=b2)

        # bisection:
        sigb = 0.01*sigb
        while np.abs(b2-b1) > sigb:
            b = b1 + 0.5*(b2-b1)
            f, a = rofunc(x, y, b=b)
            if f*f1 >= 0:
                f1 = f
                b1 = b
            else:
                f2 = f
                b2 = b

    if debug:
        print("FITAFFINE: *end of iterative loop*")
        print("FITAFFINE: equation of the robust fit: {} + {} x".format(a, b))

    return b, a


def rofunc(x, y, b):
    """
    Compute the affine parameter for robust affine fit.

    This function calculates the parameter ``a`` for a given value of ``b``
    that solves the equation ``0 = Sum_i (Xi * SIGN(Yi-a-b*Xi))`` where
    ``Xi`` is the points-set to be fitted by the point set ``Yi`` following
    ``Y=A+BX``. This function takes place in the framework of a robust affine
    fit.

    Parameters
    ----------
    x : numpy ndarray
        the fitted data set
        in IDL: DATA_INPUT[0]
    y : numpy ndarray
        the original data to be fitted
        in IDL: DATA_INPUT[1]
    b : float
        The known value ``b`` to calculate ``a``.

    Returns
    -------
    sum_result : float
    a : float
        The computed value of ``a``

    Notes
    -----
    - ported from LibAndromeda/rofunc.pro v1.1 2016/02/18
    - removed ``abdev`` variable, which is not useful here.
    - IDL version adapted from Numerical Recipies 3rd edition (2007) - p.818

    """
    epsilon = 1e-5  # convergence criteria

    arr = y - b*x
    a = np.median(arr)

    sum_result = 0
    for j in range(len(x)):
        d = y[j] - (b*x[j] + a)
        if y[j] != 0:
            d /= np.abs(y[j])
        if np.abs(d) > epsilon:
            if d >= 0:
                sum_result += x[j]
            else:
                sum_result -= x[j]

    return sum_result, a


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

