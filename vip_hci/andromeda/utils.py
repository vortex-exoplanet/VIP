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
