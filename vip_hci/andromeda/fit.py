"""
Robust linear fitting.
"""


__author__ = 'Ralf Farkas'
__all__ = []


import numpy as np


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
