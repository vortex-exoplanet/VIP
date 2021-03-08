#! /usr/bin/env python

"""
Probability of a point source to be a background star.
"""

__author__ = 'V. Christiaens'
__all__ = ['bkg_star_proba']

from scipy.special import factorial    
import numpy as np


def bkg_star_proba(n_dens, sep, n_bkg=1, verbose=True, full_output=False):
    """ Given an input density of background star brighter than a certain 
    magnitude (obtained e.g. from the Besancon model or TRILEGAL), and the 
    separation of n_bkg point source, estimate the probability of having n_bkg 
    or more background stars in a disk with radius equal to the largest 
    separation.
    The probability is estimated using a spatial Poisson point process.

    Parameters
    ----------
    n_dens : float
        Number density of background stars in the direction of the object of 
        interest, in arcsec^-2.
    sep : float or numpy 1d array
        Separation of the point sources with respect to central star, in arcsec.
    n_bkg : int, opt
        Number of point sources in the field, and for which the separation is
        provided.
    verbose: bool, opt
        Whether to print the probabilities for 0 to n_bkg point sources.
    full_output: bool, opt
        Whether to also return probabilities of 0 to n_bkg-1 point sources

    Returns
    -------
    proba : float
        Probability between 0% and 100%.
    [probas : np 1d array] if full_output is True
        Probabilities of getting 0 to n_bkg-1 point sources

    """
    
    if n_bkg < 1 or not isinstance(n_bkg,int):
        raise TypeError("n_bkg should be a strictly positive integer.")
    
    if not isinstance(sep, float):
        if isinstance(sep, np.ndarray):
            if sep.ndim!=1 or sep.shape[0]!=n_bkg:
                raise TypeError("if sep is a np array, its len should be n_bkg")
            else:
                sep = np.amax(sep)
        else:
            raise TypeError("sep can only be a float or a np 1d array")

    B = np.pi*sep**2 
    probas = np.zeros(n_bkg)
    for i in range(n_bkg):
        probas[i] = np.exp(-n_dens*B)*(n_dens*B)**i/float(factorial(i))
        if verbose:
            msg = "Proba of having {:.0f} bkg star in a disk of "
            msg += "{:.1}'' radius: {:.1f}%"
            print(msg.format(i,sep,probas[i]))

    proba = 1-np.sum(probas)
    if verbose:
        msg = "Proba of having {:.0f} bkg star or more in a disk of "
        msg += "{:.1}'' radius: {:.1f}%"
        print(msg.format(n_bkg,sep,proba))

    if full_output:
        return proba, probas
    else:
        return proba