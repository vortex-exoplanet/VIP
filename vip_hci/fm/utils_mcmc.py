#! /usr/bin/env python

"""
Module with utility functions to the MCMC (``emcee``) sampling for NEGFC 
parameter estimation.
"""


__author__ = 'V. Christiaens, O. Wertz, Carlos Alberto Gomez Gonzalez'
__all__ = ['gelman_rubin',
           'gelman_rubin_from_chain',
           'autocorr_func_1d',
           'auto_window',
           'autocorr',
           'autocorr_test']

import numpy as np
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


def gelman_rubin(x):
    r"""
    Determine the Gelman-Rubin :math:`\hat{R}` statistical test between Markov 
    chains.
    
    Parameters
    ----------
    x: numpy.array
        The numpy.array on which the Gelman-Rubin test is applied. This array
        should contain at least 2 set of data, i.e. x.shape >= (2,).
        
    Returns
    -------
    out: float
        The Gelman-Rubin :math:`\hat{R}`.

    Example
    -------
    >>> x1 = np.random.normal(0.0,1.0,(1,100))
    >>> x2 = np.random.normal(0.1,1.3,(1,100))
    >>> x = np.vstack((x1,x2))
    >>> gelman_rubin(x)
    1.0366629898991262
    >>> gelman_rubin(np.vstack((x1,x1)))
    0.99
        
    """
    if np.shape(x) < (2,):
        msg = 'Gelman-Rubin diagnostic requires multiple chains of the same '
        msg += 'length'
        raise ValueError(msg)

    try:
        m, n = np.shape(x)
    except ValueError:
        print("Bad shape for the chains")
        return

    # Calculate between-chain variance
    B_over_n = np.sum((np.mean(x, 1) - np.mean(x)) ** 2) / (m - 1)

    # Calculate within-chain variances
    W = np.sum([(x[i] - xbar) ** 2 for i, xbar in
                enumerate(np.mean(x, 1))]) / (m * (n - 1))
    # (over) estimate of variance
    s2 = W * (n - 1) / n + B_over_n

    # Pooled posterior variance estimate
    V = s2 + B_over_n / m

    # Calculate PSRF
    R = V / W

    return R


def gelman_rubin_from_chain(chain, burnin):
    r""" Pack the MCMC chain and determine the Gelman-Rubin :math:`\hat{R}` 
    statistical test. In other words, two sub-sets are extracted from the chain 
    (burnin parts are taken into account) and the Gelman-Rubin statistical test 
    is performed.
    
    Parameters
    ----------
    chain: numpy.array
        The MCMC chain with the shape walkers x steps x model_parameters
    burnin: float \in [0,1]
        The fraction of a walker which is discarded.
        
    Returns
    -------
    out: float
        The Gelman-Rubin :math:`\hat{R}`.
        
    """
    dim = chain.shape[2]
    k = chain.shape[1]
    thr0 = int(np.floor(burnin*k))
    thr1 = int(np.floor((1-burnin) * k * 0.25))
    rhat = np.zeros(dim)
    for j in range(dim):
        part1 = chain[:, thr0:thr0+thr1, j].reshape((-1))
        part2 = chain[:, thr0+3*thr1:thr0+4*thr1, j].reshape((-1))
        series = np.vstack((part1, part2))
        rhat[j] = gelman_rubin(series)
    return rhat


def next_pow_two(n):
    i = 1
    while i < n:
        i = i << 1
    return i


def autocorr_func_1d(x, norm=True):
    x = np.atleast_1d(x)
    if len(x.shape) != 1:
        raise ValueError("invalid dimensions for 1D autocorrelation function")
    n = next_pow_two(len(x))

    # Compute the FFT and then (from that) the auto-correlation function
    f = np.fft.fft(x - np.mean(x), n=2 * n)
    acf = np.fft.ifft(f * np.conjugate(f))[: len(x)].real
    acf /= 4 * n

    # Optionally normalize
    if norm:
        acf /= acf[0]

    return acf


def auto_window(taus, c):
    m = np.arange(len(taus)) < c * taus
    if np.any(m):
        return np.argmin(m)
    return len(taus) - 1


def autocorr(y, c=5.0):
    f = np.zeros(y.shape[1])
    for yy in y:
        f += autocorr_func_1d(yy)
    f /= len(y)
    taus = 2.0 * np.cumsum(f) - 1.0
    window = auto_window(taus, c)
    return taus[window]


def autocorr_test(chain):
    N = chain.shape[1]
    tau = autocorr(chain)
    return tau/N