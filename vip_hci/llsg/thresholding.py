#! /usr/bin/env python

"""
"""



__author__ = 'Carlos Alberto Gomez Gonzalez'
__all__ = ['thresholding']

import numpy as np


def thresholding(array, threshold, mode):
    """ Array thresholding strategies.
    """
    x = array.copy()
    if mode == 'soft':
        j = np.abs(x) <= threshold
        x[j] = 0
        k = np.abs(x) > threshold
        if isinstance(threshold, float):
            x[k] = x[k] - np.sign(x[k]) * threshold
        else:
            x[k] = x[k] - np.sign(x[k]) * threshold[k]
    elif mode == 'hard':
        j = np.abs(x) < threshold
        x[j] = 0
    elif mode == 'nng':
        j = np.abs(x) <= threshold
        x[j] = 0
        j = np.abs(x) > threshold
        x[j] = x[j] - threshold**2/x[j]
    elif mode == 'greater':
        j = x < threshold
        x[j] = 0
    elif mode == 'less':
        j = x > threshold
        x[j] = 0
    else:
        raise RuntimeError('Thresholding mode not recognized')
    return x