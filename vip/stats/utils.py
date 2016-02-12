#! /usr/bin/env python

"""
Various stat functions.
"""

from __future__ import division

__author__ = 'C. Gomez @ ULg'
__all__ = ['descriptive_stats']

import numpy as np


def descriptive_stats(array, verbose=True, label=''):
    """ Simple statistics from vector.
    """
    mean = np.mean(array)
    median = np.median(array)
    mini = np.min(array)
    maxi = np.max(array)
    first_qu = np.percentile(array, 25)
    third_qu = np.percentile(array, 75)
    
    if verbose:
        msg = label
        msg += 'min={:.1f} / 1st QU={:.1f} / ave={:.1f} / med={:.1f} / 3rd QU={:.1f} / max={:.1f}'
        print msg.format(mini, first_qu, mean, median, third_qu, maxi)
        
    return mini, first_qu, mean, median, third_qu, maxi