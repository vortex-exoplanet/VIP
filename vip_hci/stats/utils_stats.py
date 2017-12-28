#! /usr/bin/env python

"""
Various stat functions.
"""

from __future__ import division, print_function

__author__ = 'Carlos Alberto Gomez Gonzalez'
__all__ = ['descriptive_stats']

import numpy as np
from matplotlib.pyplot import boxplot


def descriptive_stats(array, verbose=True, label='', mean=False, plot=False):
    """ Simple statistics from vector.
    """
    if mean:  mean = np.mean(array)
    median = np.median(array)
    mini = np.min(array)
    maxi = np.max(array)
    first_qu = np.percentile(array, 25)
    third_qu = np.percentile(array, 75)
    
    if verbose:
        msg = label
        if mean:
            msg += 'min={:.1f} / 1st QU={:.1f} / ave={:.1f} / med={:.1f} / 3rd QU={:.1f} / max={:.1f}'
            print(msg.format(mini, first_qu, mean, median, third_qu, maxi))
        else:
            msg += 'min={:.1f} / 1st QU={:.1f} / med={:.1f} / 3rd QU={:.1f} / max={:.1f}'
            print(msg.format(mini, first_qu, median, third_qu, maxi))
    
    if plot:
        boxplot(array, vert=False, meanline=mean, showfliers=True, sym='.') #whis=range)
    
    if mean:
        return mini, first_qu, mean, median, third_qu, maxi
    else:
        return mini, first_qu, median, third_qu, maxi