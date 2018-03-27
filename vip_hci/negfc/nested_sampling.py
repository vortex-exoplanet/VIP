#! /usr/bin/env python

"""
Module with functions for posterior sampling of the NEGFC parameters using
nested sampling (``nestle``).
"""

from __future__ import division, print_function

__author__ = 'Carlos Alberto Gomez Gonzalez',
__all__ = ['nested_negfc_sampling',
           'nested_sampling_results']

import nestle
import corner
import numpy as np
from matplotlib import pyplot as plt
from ..conf import time_ini, timing
from .mcmc_sampling import lnlike, confidence, show_walk_plot


def nested_negfc_sampling(init, cube, angs, plsc, psf, fwhm, annulus_width=2,
                          aperture_radius=1, ncomp=10, scaling=None,
                          svd_mode='lapack', cube_ref=None, collapse='median',
                          w=(5, 5, 200), method='single', npoints=100,
                          dlogz=0.1, decline_factor=None, rstate=None,
                          verbose=True):
    """ Runs a nested sampling algorithm in order to determine the position and
    the flux of the planet using the 'Negative Fake Companion' technique. The
    result of this procedure is a a ``nestle`` object containing the samples
    from the posterior distributions of each of the 3 parameters. It provides
    pretty good results (value plus error bars) compared to a more CPU intensive
    Monte Carlo approach with the affine invariant sampler (``emcee``).

    Parameters
    ----------
    init: array_like or tuple of length 3
        The first guess for the position and flux of the planet, respectively.
        It serves for generating the bounds of the log prior function (uniform
        in a bounded interval).
    cube: array_like
        Frame sequence of cube.
    angs: array_like
        The relative path to the parallactic angle fits image or the angs itself.
    plsc: float
        The platescale, in arcsec per pixel.
    psf: array_like
        The PSF template. It must be centered and the flux in a 1*FWHM aperture
        must equal 1.
    fwhm : float
        The FHWM in pixels.
    annulus_width: float, optional
        The width in pixel of the annulus on which the PCA is performed.
    aperture_radius: float, optional
        The radius of the circular aperture.
    ncomp: int optional
        The number of principal components.
    scaling : {'temp-mean', 'temp-standard'} or None, optional
        With None, no scaling is performed on the input data before SVD. With
        "temp-mean" then temporal px-wise mean subtraction is done and with
        "temp-standard" temporal mean centering plus scaling to unit variance
        is done.
    svd_mode : {'lapack', 'randsvd', 'eigen', 'arpack'}, str optional
        Switch for different ways of computing the SVD and selected PCs.
    cube_ref: array_like, 3d, optional
        Reference library cube. For Reference Star Differential Imaging.
    collapse : {'median', 'mean', 'sum', 'trimmean', None}, str or None, optional
        Sets the way of collapsing the frames for producing a final image. If
        None then the cube of residuals is used when measuring the function of
        merit (instead of a single final frame).
    w : tuple of length 3
        The size of the bounds (around the initial state ``init``) for each
        parameter.
    method : {"single", "multi", "classic"}, str optional
        Flavor of nested sampling. Single ellipsoid works well for the NEGFC and
        is the default.
    npoints : int optional
        Number of active points. At least ndim+1 (4 will produce bad results).
        For problems with just a few parameters (<=5) like the NEGFC, good
        results are obtained with 100 points (default).
    dlogz : Estimated remaining evidence
        Iterations will stop when the estimated contribution of the remaining
        prior volume to the total evidence falls below this threshold.
        Explicitly, the stopping criterion is log(z + z_est) - log(z) < dlogz
        where z is the current evidence from all saved samples, and z_est is the
        estimated contribution from the remaining volume. This option and
        decline_factor are mutually exclusive. If neither is specified, the
        default is dlogz=0.5.
    decline_factor : float, optional
        If supplied, iteration will stop when the weight (likelihood times prior
        volume) of newly saved samples has been declining for
        decline_factor * nsamples consecutive samples. A value of 1.0 seems to
        work pretty well.
    rstate : random instance, optional
        RandomState instance. If not given, the global random state of the
        numpy.random module will be used.

    Returns
    -------
    res : nestle object
        ``Nestle`` object with the nested sampling results, including the
        posterior samples.

    Notes
    -----
    Nested Sampling is a computational approach for integrating posterior
    probability in order to compare models in Bayesian statistics. It is similar
    to Markov Chain Monte Carlo (MCMC) in that it generates samples that can be
    used to estimate the posterior probability distribution. Unlike MCMC, the
    nature of the sampling also allows one to calculate the integral of the
    distribution. It also happens to be a pretty good method for robustly
    finding global maxima.

    Nestle documentation:
    http://kbarbary.github.io/nestle/

    Convergence:
    http://kbarbary.github.io/nestle/stopping.html
    Nested sampling has no well-defined stopping point. As iterations continue,
    the active points sample a smaller and smaller region of prior space.
    This can continue indefinitely. Unlike typical MCMC methods, we don't gain
    any additional precision on the results by letting the algorithm run longer;
    the precision is determined at the outset by the number of active points.
    So, we want to stop iterations as soon as we think the active points are
    doing a pretty good job sampling the remaining prior volume - once we've
    converged to the highest-likelihood regions such that the likelihood is
    relatively flat within the remaining prior volume.

    Method:
    The trick in nested sampling is to, at each step in the algorithm,
    efficiently choose a new point in parameter space drawn with uniform
    probability from the parameter space with likelihood greater than the
    current likelihood constraint. The different methods all use the
    current set of active points as an indicator of where the target
    parameter space lies, but differ in how they select new points from  it.
    "classic" is close to the method described in Skilling (2004).
    "single", Mukherjee, Parkinson & Liddle (2006), Determines a single
    ellipsoid that bounds all active points,
    enlarges the ellipsoid by a user-settable factor, and selects a new point
    at random from within the ellipsoid.
    "multiple", Shaw, Bridges & Hobson (2007) and Feroz, Hobson & Bridges 2009
    (Multinest). In cases where the posterior is multi-modal,
    the single-ellipsoid method can be extremely inefficient: In such
    situations, there are clusters of active points on separate
    high-likelihood regions separated by regions of lower likelihood.
    Bounding all points in a single ellipsoid means that the ellipsoid
    includes the lower-likelihood regions we wish to avoid
    sampling from.
    The solution is to detect these clusters and bound them in separate
    ellipsoids. For this, we use a recursive process where we perform
    K-means clustering with K=2. If the resulting two ellipsoids have a
    significantly lower total volume than the parent ellipsoid (less than half),
    we accept the split and repeat the clustering and volume test on each of
    the two subset of points. This process continues recursively.
    Alternatively, if the total ellipse volume is significantly greater
    than expected (based on the expected density of points) this indicates
    that there may be more than two clusters and that K=2 was not an
    appropriate cluster division.
    We therefore still try to subdivide the clusters recursively. However,
    we still only accept the final split into N clusters if the total volume
    decrease is significant.

    """

    def prior_transform(x):
        """ x:[0,1]

        The prior transform is dinamically created with these bound:
        [radius-w1:radius+w1], [theta-w2:theta+w2], [flux-w3:flux+w3]

        Notes
        -----
        The prior transform function is used to specify the Bayesian prior for the
        problem, in a round-about way. It is a transformation from a space where
        variables are independently and uniformly distributed between 0 and 1 to
        the parameter space of interest. For independent parameters, this would be
        the product of the inverse cumulative distribution function (also known as
        the percent point function or quantile function) for each parameter.
        http://kbarbary.github.io/nestle/prior.html

        """
        a1 = 2 * w[0]
        a2 = init[0] - w[0]
        b1 = 2 * w[1]
        b2 = init[1] - w[1]
        c1 = 2 * w[2]
        c2 = init[2] - w[2]
        return np.array([a1 * x[0] + a2, b1 * x[1] + b2, c1 * x[2] + c2])

    def f(param):
        return lnlike(param=param, cube=cube, angs=angs, plsc=plsc,
                      psf_norm=psf, fwhm=fwhm, annulus_width=annulus_width,
                      aperture_radius=aperture_radius, initial_state=init,
                      cube_ref=cube_ref, svd_mode=svd_mode, scaling=scaling,
                      fmerit='sum', ncomp=ncomp, collapse=collapse)

    # -------------------------------------------------------------------------
    if verbose:  start = time_ini()

    if verbose:
        print('Prior bounds on parameters:')
        print('Radius [{},{}]'.format(init[0] - w[0], init[0] + w[0], ))
        print('Theta [{},{}]'.format(init[1] - w[1], init[1] + w[1]))
        print('Flux [{},{}]'.format(init[2] - w[2], init[2] + w[2]))
        print('\nUsing {} active points'.format(npoints))

    res = nestle.sample(f, prior_transform, ndim=3, method=method,
                        npoints=npoints, rstate=rstate, dlogz=dlogz,
                        decline_factor=decline_factor)

    # if verbose:  print; timing(start)
    if verbose:
        print('\nTotal running time:')
        timing(start)
    return res


def nested_sampling_results(ns_object, burnin=0.4, bins=None):
    """ Shows the results of the Nested Sampling, summary, parameters with errors,
    walk and corner plots.
    """
    res = ns_object
    nsamples = res.samples.shape[0]
    indburnin = np.percentile(np.array(range(nsamples)), burnin * 100)

    print(res.summary())

    print(
        '\nNatural log of prior volume and Weight corresponding to each sample')
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(res.logvol, '.', alpha=0.5, color='gray')
    plt.xlabel('samples')
    plt.ylabel('logvol')
    plt.vlines(indburnin, res.logvol.min(), res.logvol.max(),
               linestyles='dotted')
    plt.subplot(1, 2, 2)
    plt.plot(res.weights, '.', alpha=0.5, color='gray')
    plt.xlabel('samples')
    plt.ylabel('weights')
    plt.vlines(indburnin, res.weights.min(), res.weights.max(),
               linestyles='dotted')
    plt.show()

    print("\nWalk plots before the burnin")
    show_walk_plot(np.expand_dims(res.samples, axis=0))
    if burnin > 0:
        print("\nWalk plots after the burnin")
        show_walk_plot(np.expand_dims(res.samples[indburnin:], axis=0))

    mean, cov = nestle.mean_and_cov(res.samples[indburnin:],
                                    res.weights[indburnin:])
    print("\nWeighted mean +- sqrt(covariance)")
    print("Radius = {:.3f} +/- {:.3f}".format(mean[0], np.sqrt(cov[0, 0])))
    print("Theta = {:.3f} +/- {:.3f}".format(mean[1], np.sqrt(cov[1, 1])))
    print("Flux = {:.3f} +/- {:.3f}".format(mean[2], np.sqrt(cov[2, 2])))

    if bins is None:
        bins = int(np.sqrt(res.samples[indburnin:].shape[0]))
        print("\nHist bins =", bins)
    ranges = None

    fig = corner.corner(res.samples[indburnin:], bins=bins,
                        labels=["$r$", r"$\theta$", "$f$"],
                        weights=res.weights[indburnin:], range=ranges,
                        plot_contours=True)
    fig.set_size_inches(8, 8)

    print('\nConfidence intervals')
    _ = confidence(res.samples[indburnin:], cfd=68, bins=bins,
                   weights=res.weights[indburnin:],
                   gaussian_fit=True, verbose=True, save=False)