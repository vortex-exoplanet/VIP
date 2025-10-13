#! /usr/bin/env python
"""
Module with utility function called from within the NEGFC algorithm.
"""

__author__ = 'Valentin Christiaens'
__all__ = ['find_nearest']

import numpy as np


def find_nearest(array, value, output='index', constraint=None, n=1):
    """Find the indices, and optionally the values, of an array's n closest\
    elements to a certain value. By default, only returns the index/indices.

    Possible constraints: 'ceil', 'floor', None ("ceil" will return the closest
    element with a value greater than 'value', "floor" the opposite).

    Parameters
    ----------
    array: 1d numpy array or list
        Array in which to check the closest element to value.
    value: float
        Value for which the algorithm searches for the n closest elements in
        the array.
    output: str, opt {'index','value','both' }
        Set what is returned
    constraint: str, opt {None, 'ceil', 'floor', 'ceil=', 'floor='}
        If not None, will check for the closest element larger (or equal) than
        value if set to 'ceil' ('ceil='), or closest element smaller (or equal)
        than value if set to 'floor' ('floor=').
    n: int, opt
        Number of elements to be returned, sorted by proximity to the values.
        Default: only the closest value is returned.

    Returns
    -------
    [output='index']: index/indices of the closest n value(s) in the array;
    [output='value']: the closest n value(s) in the array,
    [output='both']: closest value(s) and index/-ices, respectively.

    """
    array = np.asarray(array)
    if constraint is None:
        fm = np.abs(array - value)
        idx = np.argpartition(fm, n)[:n]
    elif 'floor' in constraint or 'ceil' in constraint:
        indices = np.arange(len(array), dtype=np.int32)
        if 'floor' in constraint:
            fm = -(array - value)
        else:
            fm = array - value
        if '=' in constraint:
            crop_indices = indices[fm >= 0]
            fm = fm[fm >= 0]
        else:
            crop_indices = indices[fm > 0]
            fm = fm[fm > 0]
        idx = np.argpartition(fm, n)[:n]
        idx = crop_indices[idx]
        if len(idx) == 0:
            msg = "No indices match the constraint ({} w.r.t {:.2f})"
            print(msg.format(constraint, value))
            raise ValueError("No indices match the constraint")
    else:
        raise ValueError("Constraint not recognised")

    if n == 1:
        idx = idx[0]

    if output == 'index':
        return idx
    elif output == 'value':
        return array[idx]
    else:
        return array[idx], idx
