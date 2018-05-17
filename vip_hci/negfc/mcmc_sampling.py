#! /usr/bin/env python

"""
Module with the MCMC (``emcee``) sampling for NEGFC parameter estimation.
"""
from __future__ import division, print_function

__author__ = 'O. Wertz, Carlos Alberto Gomez Gonzalez'
__all__ = ['mcmc_negfc_sampling',
           'chain_zero_truncated',
           'show_corner_plot',
           'show_walk_plot',
           'confidence']

import numpy as np
import os
import emcee
import inspect
import datetime
import corner
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.mlab import normpdf
from scipy.stats import norm
from ..fits import open_adicube, open_fits
from ..metrics import cube_inject_companions
from ..conf import time_ini, timing
from ..conf.utils_conf import sep
from .simplex_fmerit import get_values_optimize
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)



def lnprior(param, bounds):
    """ Define the prior log-function.
    
    Parameters
    ----------
    param: tuple
        The model parameters.
    bounds: list
        The bounds for each model parameter.
        Ex: bounds = [(10,20),(0,360),(0,5000)]
    
    Returns
    -------
    out: float.
        0 if all the model parameters satisfy the prior conditions defined here.
        -np.inf if at least one model parameters is out of bounds.
    """
    
    try:
        r, theta, flux = param
    except TypeError:
        print('param must be a tuple, {} given'.format(type(param)))

    try:
        r_bounds, theta_bounds, flux_bounds = bounds
    except TypeError:
        print('bounds must be a list of tuple, {} given'.format(type(bounds)))
        
    if r_bounds[0] <= r <= r_bounds[1] and \
       theta_bounds[0] <= theta <= theta_bounds[1] and \
       flux_bounds[0] <= flux <= flux_bounds[1]:
        return 0.0
    else:
        return -np.inf


def lnlike(param, cube, angs, plsc, psf_norm, fwhm, annulus_width,
           ncomp, aperture_radius, initial_state, cube_ref=None,
           svd_mode='lapack', scaling='temp-mean', fmerit='sum', imlib='opencv',
           interpolation='lanczos4', collapse='median', debug=False):
    """ Define the likelihood log-function.
    
    Parameters
    ----------
    param: tuple
        The model parameters, typically (r, theta, flux).
    cube: numpy.array
        The cube of fits images expressed as a numpy.array.
    angs: numpy.array
        The parallactic angle fits image expressed as a numpy.array.
    plsc: float
        The platescale, in arcsec per pixel.
    psf_norm: numpy.array
        The scaled psf expressed as a numpy.array.
    annulus_width: float
        The width of the annulus of interest in terms of the FWHM.
    ncomp: int
        The number of principal components.
    fwhm : float
        The FHWM in pixels.
    aperture_radius: float
        The radius of the circular aperture in terms of the FWHM.
    initial_state: numpy.array
        The initial guess for the position and the flux of the planet.
    cube_ref: array_like, 3d, optional
        Reference library cube. For Reference Star Differential Imaging.
    svd_mode : {'lapack', 'randsvd', 'eigen', 'arpack'}, str optional
        Switch for different ways of computing the SVD and selected PCs.
    scaling : {'temp-mean', 'temp-standard'} or None, optional
        With None, no scaling is performed on the input data before SVD. With
        "temp-mean" then temporal px-wise mean subtraction is done and with
        "temp-standard" temporal mean centering plus scaling to unit variance
        is done.
    fmerit : {'sum', 'stddev'}, string optional
        Chooses the figure of merit to be used. stddev works better for close in
        companions sitting on top of speckle noise.
    imlib : str, optional
        See the documentation of the ``vip_hci.preproc.frame_shift`` function.
    interpolation : str, optional
        See the documentation of the ``vip_hci.preproc.frame_shift`` function.
    collapse : {'median', 'mean', 'sum', 'trimmean', None}, str or None, optional
        Sets the way of collapsing the frames for producing a final image. If
        None then the cube of residuals is used when measuring the function of
        merit (instead of a single final frame).
    debug: boolean
        If True, the cube is returned along with the likelihood log-function.
        
    Returns
    -------
    out: float
        The log of the likelihood.
        
    """
    # Create the cube with the negative fake companion injected
    cube_negfc = cube_inject_companions(cube, psf_norm, angs, flevel=-param[2],
                                        plsc=plsc, rad_dists=[param[0]],
                                        n_branches=1, theta=param[1],
                                        imlib=imlib,
                                        interpolation=interpolation,
                                        verbose=False)
                                  
    # Perform PCA and extract the zone of interest
    values = get_values_optimize(cube_negfc, angs, ncomp, annulus_width*fwhm,
                                 aperture_radius*fwhm, initial_state[0],
                                 initial_state[1], cube_ref=cube_ref,
                                 svd_mode=svd_mode, scaling=scaling,
                                 imlib=imlib, interpolation=interpolation,
                                 collapse=collapse)
    
    # Function of merit
    if fmerit == 'sum':
        lnlikelihood = -0.5 * np.sum(np.abs(values))
    elif fmerit == 'stddev':
        values = values[values != 0]
        lnlikelihood = -np.std(np.abs(values))
    else:
        raise RuntimeError('fmerit choice not recognized.')
    
    if debug:
        return lnlikelihood, cube_negfc
    else:
        return lnlikelihood


def lnprob(param,bounds, cube, angs, plsc, psf_norm, fwhm,
           annulus_width, ncomp, aperture_radius, initial_state, cube_ref=None,
           svd_mode='lapack', scaling='temp-mean', fmerit='sum', imlib='opencv',
           interpolation='lanczos4', collapse='median', display=False):
    """ Define the probability log-function as the sum between the prior and
    likelihood log-funtions.
    
    Parameters
    ----------
    param: tuple
        The model parameters.
    bounds: list
        The bounds for each model parameter.
        Ex: bounds = [(10,20),(0,360),(0,5000)]
    cube: numpy.array
        The cube of fits images expressed as a numpy.array.
    angs: numpy.array
        The parallactic angle fits image expressed as a numpy.array.
    plsc: float
        The platescale, in arcsec per pixel.
    psf_norm: numpy.array
        The scaled psf expressed as a numpy.array.
    fwhm : float
        The FHWM in pixels.
    annulus_width: float
        The width in pixel of the annulus on wich the PCA is performed.
    ncomp: int
        The number of principal components.
    aperture_radius: float
        The radius of the circular aperture.
    initial_state: numpy.array
        The initial guess for the position and the flux of the planet.
    cube_ref : array_like, 3d, optional
        Reference library cube. For Reference Star Differential Imaging.
    svd_mode : {'lapack', 'randsvd', 'eigen', 'arpack'}, str optional
        Switch for different ways of computing the SVD and selected PCs.
    scaling : {'temp-mean', 'temp-standard'} or None, optional
        With None, no scaling is performed on the input data before SVD. With
        "temp-mean" then temporal px-wise mean subtraction is done and with
        "temp-standard" temporal mean centering plus scaling to unit variance
        is done.
    fmerit : {'sum', 'stddev'}, string optional
        Chooses the figure of merit to be used. stddev works better for close in
        companions sitting on top of speckle noise.
    imlib : str, optional
        See the documentation of the ``vip_hci.preproc.frame_rotate`` function.
    interpolation : str, optional
        See the documentation of the ``vip_hci.preproc.frame_rotate`` function.
    collapse : {'median', 'mean', 'sum', 'trimmean', None}, str or None, optional
        Sets the way of collapsing the frames for producing a final image. If
        None then the cube of residuals is used when measuring the function of
        merit (instead of a single final frame).
    display: boolean
        If True, the cube is displayed with ds9.
        
    Returns
    -------
    out: float
        The probability log-function.
    
    """
    if initial_state is None:
        initial_state = param
    
    lp = lnprior(param, bounds)
    
    if np.isinf(lp):
        return -np.inf       
    
    return lp + lnlike(param, cube, angs, plsc, psf_norm, fwhm,
                       annulus_width, ncomp, aperture_radius, initial_state,
                       cube_ref, svd_mode, scaling, fmerit, imlib,
                       interpolation, collapse, display)


def gelman_rubin(x):
    """
    Determine the Gelman-Rubin \hat{R} statistical test between Markov chains.
    
    Parameters
    ----------
    x: numpy.array
        The numpy.array on which the Gelman-Rubin test is applied. This array
        should contain at least 2 set of data, i.e. x.shape >= (2,).
        
    Returns
    -------
    out: float
        The Gelman-Rubin \hat{R}.

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
    """ Pack the MCMC chain and determine the Gelman-Rubin \hat{R} statistical
    test. In other words, two sub-sets are extracted from the chain (burnin
    parts are taken into account) and the Gelman-Rubin statistical test is
    performed.
    
    Parameters
    ----------
    chain: numpy.array
        The MCMC chain with the shape walkers x steps x model_parameters
    burnin: float \in [0,1]
        The fraction of a walker which is discarded.
        
    Returns
    -------
    out: float
        The Gelman-Rubin \hat{R}.
        
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


def mcmc_negfc_sampling(cube, angs, psfn, ncomp, plsc, initial_state, fwhm=4,
                        annulus_width=3, aperture_radius=4, cube_ref=None,
                        svd_mode='lapack', scaling='temp-mean', fmerit='sum',
                        imlib='opencv', interpolation='lanczos4',
                        collapse='median', nwalkers=1000, bounds=None, a=2.0,
                        burnin=0.3, rhat_threshold=1.01, rhat_count_threshold=1,
                        niteration_min=0, niteration_limit=1e2,
                        niteration_supp=0, check_maxgap=1e4, nproc=1,
                        output_file=None, display=False, verbosity=0,
                        save=False):
    """ Runs an affine invariant mcmc sampling algorithm in order to determine
    the position and the flux of the planet using the 'Negative Fake Companion'
    technique. The result of this procedure is a chain with the samples from the
    posterior distributions of each of the 3 parameters.
    
    This technique can be summarized as follows:
    1) We inject a negative fake companion (one candidate) at a given position
    and characterized by a given flux, both close to the expected values.
    2) We run PCA on an full annulus which pass through the initial guess,
    regardless of the position of the candidate.
    3) We extract the intensity values of all the pixels contained in a
    circular aperture centered on the initial guess.
    4) We calculate the function of merit. The associated chi^2 is given by
    chi^2 = sum(|I_j|) where j \in {1,...,N} with N the total number of pixels
    contained in the circular aperture.
    The steps 1) to 4) are looped. At each iteration, the candidate model
    parameters are defined by the emcee Affine Invariant algorithm.
    
    Parameters
    ----------
    cube: numpy.array
        ADI fits cube.
    angs: numpy.array
        The parallactic angle vector.
    psfn: numpy.array
        PSF array. The PSF must be centered and the flux in a 1*FWHM aperture
        must equal 1 (use ``vip_hci.phot.psf_norm``).
    ncomp: int
        The number of principal components.
    plsc: float
        The platescale, in arcsec per pixel.
    annulus_width: float, optional
        The width in pixel of the annulus on which the PCA is performed.
    aperture_radius: float, optional
        The radius of the circular aperture.
    nwalkers: int optional
        The number of Goodman & Weare 'walkers'.
    initial_state: numpy.array
        The first guess for the position and flux of the planet, respectively.
        Each walker will start in a small ball around this preferred position.
    cube_ref : array_like, 3d, optional
        Reference library cube. For Reference Star Differential Imaging.
    svd_mode : {'lapack', 'randsvd', 'eigen', 'arpack'}, str optional
        Switch for different ways of computing the SVD and selected PCs.
        'randsvd' is not recommended for the negative fake companion technique.
    scaling : {'temp-mean', 'temp-standard'} or None, optional
        With None, no scaling is performed on the input data before SVD. With
        "temp-mean" then temporal px-wise mean subtraction is done and with
        "temp-standard" temporal mean centering plus scaling to unit variance
        is done.
    fmerit : {'sum', 'stddev'}, string optional
        Chooses the figure of merit to be used. stddev works better for close in
        companions sitting on top of speckle noise.
    imlib : str, optional
        See the documentation of the ``vip_hci.preproc.frame_rotate`` function.
    interpolation : str, optional
        See the documentation of the ``vip_hci.preproc.frame_rotate`` function.
    collapse : {'median', 'mean', 'sum', 'trimmean', None}, str or None, optional
        Sets the way of collapsing the frames for producing a final image. If
        None then the cube of residuals is used when measuring the function of
        merit (instead of a single final frame).
    bounds: numpy.array or list, default=None, optional
        The prior knowledge on the model parameters. If None, large bounds will
        be automatically estimated from the initial state.
    a: float, default=2.0
        The proposal scale parameter. See notes.
    burnin: float, default=0.3
        The fraction of a walker which is discarded.
    rhat_threshold: float, default=0.01
        The Gelman-Rubin threshold used for the test for nonconvergence.
    rhat_count_threshold: int, optional
        The Gelman-Rubin test must be satisfied 'rhat_count_threshold' times in
        a row before claiming that the chain has converged.
    niteration_min: int, optional
        Steps per walker lower bound. The simulation will run at least this
        number of steps per walker.
    niteration_limit: int, optional
        Steps per walker upper bound. If the simulation runs up to
        'niteration_limit' steps without having reached the convergence
        criterion, the run is stopped.
    niteration_supp: int, optional
        Number of iterations to run after having "reached the convergence".
    check_maxgap: int, optional
        Maximum number of steps per walker between two Gelman-Rubin test.
    nproc: int, optional
        The number of processes to use for parallelization.
    output_file: str, optional
        The name of the output file which contains the MCMC results in the case
        ``save`` is True.
    display: bool, optional
        If True, the walk plot is displayed at each evaluation of the Gelman-
        Rubin test.
    verbosity: 0, 1 or 2, optional
        Verbosity level. 0 for no output and 2 for full information.
    save: bool, optional
        If True, the MCMC results are pickled.
                    
    Returns
    -------
    out : numpy.array
        The MCMC chain.
        
    Notes
    -----
    The parameter 'a' must be > 1. For more theoretical information concerning
    this parameter, see Goodman & Weare, 2010, Comm. App. Math. Comp. Sci.,
    5, 65, Eq. [9] p70.
    
    The parameter 'rhat_threshold' can be a numpy.array with individual
    threshold value for each model parameter.
    """
    if verbosity == 1 or verbosity == 2:
        start_time = time_ini()
        print("        MCMC sampler for the NEGFC technique       ")
        print(sep)

    # If required, one create the output folder.
    if save:
        
        if output_file is None:
            output_file = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            
        try:
            os.makedirs('results/'+output_file)
        except OSError as exc:
            if exc.errno == 17 and os.path.isdir('results/'+output_file):
                # errno.EEXIST == 17 -> File exists
                pass
            else:
                raise


    if not isinstance(cube, np.ndarray) or cube.ndim != 3:
        raise ValueError('`cube` must be a 3D numpy array')

    if cube_ref is not None:
        if not isinstance(cube_ref, np.ndarray) or cube_ref.ndim != 3:
            raise ValueError('`cube_ref` must be a 3D numpy array')
    
    # #########################################################################
    # Initialization of the variables
    # #########################################################################
    dim = 3     # There are 3 model parameters: rad, theta, flux
    itermin = niteration_min
    limit = niteration_limit
    supp = niteration_supp
    maxgap = check_maxgap
    initial_state = np.array(initial_state)
    
    if itermin > limit:
        itermin = 0

    fraction = 0.3
    geom = 0
    lastcheck = 0
    konvergence = np.inf
    rhat_count = 0
    chain = np.empty([nwalkers, 1, dim])
    isamples = np.empty(0)
    pos = initial_state + np.random.normal(0, 1e-1, (nwalkers, 3))
    nIterations = limit + supp
    rhat = np.zeros(dim)
    stop = np.inf

    if bounds is None:
        bounds = [(initial_state[0] - annulus_width/2.,
                   initial_state[0] + annulus_width/2.),  # radius
                  (initial_state[1] - 10, initial_state[1] + 10),   # angle
                  (0, 2 * initial_state[2])]   # flux
    
    sampler = emcee.EnsembleSampler(nwalkers, dim, lnprob, a,
                                    args=([bounds, cube, angs, plsc, psfn,
                                           fwhm, annulus_width, ncomp,
                                           aperture_radius, initial_state,
                                           cube_ref, svd_mode, scaling, fmerit,
                                           imlib, interpolation, collapse]),
                                    threads=nproc)
    start = datetime.datetime.now()

    # #########################################################################
    # Affine Invariant MCMC run
    # #########################################################################
    if verbosity == 2:
        print('\nStart of the MCMC run ...')
        print('Step  |  Duration/step (sec)  |  Remaining Estimated Time (sec)')
                             
    for k, res in enumerate(sampler.sample(pos, iterations=nIterations,
                                           storechain=True)):
        elapsed = (datetime.datetime.now()-start).total_seconds()
        if verbosity == 2:
            if k == 0:
                q = 0.5
            else:
                q = 1
            print('{}\t\t{:.5f}\t\t\t{:.5f}'.format(k, elapsed * q,
                                                    elapsed * (limit-k-1) * q))
            
        start = datetime.datetime.now()

        # ---------------------------------------------------------------------
        # Store the state manually in order to handle with dynamical sized chain
        # ---------------------------------------------------------------------
        # Check if the size of the chain is long enough.
        s = chain.shape[1]
        if k+1 > s:     # if not, one doubles the chain length
            empty = np.zeros([nwalkers, 2*s, dim])
            chain = np.concatenate((chain, empty), axis=1)
        # Store the state of the chain
        chain[:, k] = res[0]

        # ---------------------------------------------------------------------
        # If k meets the criterion, one tests the non-convergence.
        # ---------------------------------------------------------------------
        criterion = np.amin([np.ceil(itermin*(1+fraction)**geom),
                             lastcheck+np.floor(maxgap)])
        if k == criterion:
            if verbosity == 2:
                print('\n   Gelman-Rubin statistic test in progress ...')
            
            geom += 1
            lastcheck = k
            if display:
                show_walk_plot(chain)
                
            if save:
                import pickle
                fname = 'results/{f}/{f}_temp_k{k}'.format(f=output_file, k=k)
                data = {'chain': sampler.chain,
                        'lnprob': sampler.lnprobability,
                         'AR': sampler.acceptance_fraction}
                with open(fname, 'wb') as fileSave:
                    pickle.dump(data, fileSave)
                
            # We only test the rhat if we have reached the min # of steps
            if (k+1) >= itermin and konvergence == np.inf:
                thr0 = int(np.floor(burnin*k))
                thr1 = int(np.floor((1-burnin)*k*0.25))

                # We calculate the rhat for each model parameter.
                for j in range(dim):
                    part1 = chain[:, thr0:thr0 + thr1, j].reshape(-1)
                    part2 = chain[:, thr0 + 3 * thr1:thr0 + 4 * thr1, j
                                 ].reshape(-1)
                    series = np.vstack((part1, part2))
                    rhat[j] = gelman_rubin(series)
                if verbosity == 1 or verbosity == 2:
                    print('   r_hat = {}'.format(rhat))
                    cond = rhat <= rhat_threshold
                    print('   r_hat <= threshold = {} \n'.format(cond))
                # We test the rhat.
                if (rhat <= rhat_threshold).all():
                    rhat_count += 1
                    if rhat_count < rhat_count_threshold:
                        if verbosity == 1 or verbosity == 2:
                            msg = "Gelman-Rubin test OK {}/{}"
                            print(msg.format(rhat_count, rhat_count_threshold))
                    elif rhat_count >= rhat_count_threshold:
                        if verbosity == 1 or verbosity == 2:
                            print('... ==> convergence reached')
                        konvergence = k
                        stop = konvergence + supp
                else:
                    rhat_count = 0

        # We have reached the maximum number of steps for our Markov chain.
        if k+1 >= stop:
            if verbosity == 1 or verbosity == 2:
                print('We break the loop because we have reached convergence')
            break
      
    if k == nIterations-1:
        if verbosity == 1 or verbosity == 2:
            print("We have reached the limit # of steps without convergence")
            
    # #########################################################################
    # Construction of the independent samples
    # #########################################################################
    temp = np.where(chain[0, :, 0] == 0.0)[0]
    if len(temp) != 0:
        idxzero = temp[0]
    else:
        idxzero = chain.shape[1]
    
    idx = int(np.amin([np.floor(2e5/nwalkers), np.floor(0.1*idxzero)]))
    if idx == 0:
        isamples = chain[:, 0:idxzero, :]
    else:
        isamples = chain[:, idxzero-idx:idxzero, :]

    if save:
        import pickle
        frame = inspect.currentframe()
        args, _, _, values = inspect.getargvalues(frame)
        input_parameters = {j: values[j] for j in args[1:]}
        
        output = {'isamples': isamples,
                  'chain': chain_zero_truncated(chain),
                  'input_parameters': input_parameters,
                  'AR': sampler.acceptance_fraction,
                  'lnprobability': sampler.lnprobability}
                  
        with open('results/'+output_file+'/MCMC_results', 'wb') as fileSave:
            pickle.dump(output, fileSave)
        
        msg = "\nThe file MCMC_results has been stored in the folder {}"
        print(msg.format('results/'+output_file+'/'))

    if verbosity == 1 or verbosity == 2:
        timing(start_time)
                                    
    return chain_zero_truncated(chain)

                                    
def chain_zero_truncated(chain):
    """
    Return the Markov chain with the dimension: walkers x steps* x parameters,
    where steps* is the last step before having 0 (not yet constructed chain).
    
    Parameters
    ----------
    chain: numpy.array
        The MCMC chain.
        
    Returns
    -------
    out: numpy.array
        The truncated MCMC chain, that is to say, the chain which only contains
        relevant information.
    """
    try:
        idxzero = np.where(chain[0, :, 0] == 0.0)[0][0]
    except:
        idxzero = chain.shape[1]
    return chain[:, 0:idxzero, :]
 
   
def show_walk_plot(chain, save=False, **kwargs):
    """
    Display or save a figure showing the path of each walker during the MCMC run
    
    Parameters
    ----------
    chain: numpy.array
        The Markov chain. The shape of chain must be nwalkers x length x dim.
        If a part of the chain is filled with zero values, the method will
        discard these steps.
    save: boolean, default: False
        If True, a pdf file is created.
    kwargs:
        Additional attributes are passed to the matplotlib plot method.
                                                        
    Returns
    -------
    Display the figure or create a pdf file named walk_plot.pdf in the working
    directory.
    
    """
    temp = np.where(chain[0, :, 0] == 0.0)[0]
    if len(temp) != 0:
        chain = chain[:, :temp[0], :]

    labels = kwargs.pop('labels', ["$r$", r"$\theta$", "$f$"])
    fig, axes = plt.subplots(3, 1, sharex=True,
                             figsize=kwargs.pop('figsize', (8, 6)))
    axes[2].set_xlabel(kwargs.pop('xlabel', 'step number'))
    axes[2].set_xlim(kwargs.pop('xlim', [0, chain.shape[1]]))
    color = kwargs.pop('color', 'k')
    alpha = kwargs.pop('alpha', 0.4)
    for j in range(3):
        axes[j].plot(chain[:, :, j].T, color=color, alpha=alpha, **kwargs)
        axes[j].yaxis.set_major_locator(MaxNLocator(5))
        axes[j].set_ylabel(labels[j])
    fig.tight_layout(h_pad=0)
    if save:
        plt.savefig('walk_plot.pdf')
        plt.close(fig)
    else:
        plt.show()


def show_corner_plot(chain, burnin=0.5, save=False, **kwargs):
    """
    Display or save a figure showing the corner plot (pdfs + correlation plots)
    
    Parameters
    ----------
    chain: numpy.array
        The Markov chain. The shape of chain must be nwalkers x length x dim.
        If a part of the chain is filled with zero values, the method will
        discard these steps.
    burnin: float, default: 0
        The fraction of a walker we want to discard.
    save: boolean, default: False
        If True, a pdf file is created.
     
     kwargs:
        Additional attributs are passed to the corner.corner() method.
                    
    Returns
    -------
    Display the figure or create a pdf file named walk_plot.pdf in the working
    directory.
        
    Raises
    ------
    ImportError
    
    """
    try:
        temp = np.where(chain[0, :, 0] == 0.0)[0]
        if len(temp) != 0:
            chain = chain[:, :temp[0], :]
        length = chain.shape[1]
        indburn = int(np.floor(burnin*(length-1)))
        chain = chain[:, indburn:length, :].reshape((-1, 3))
    except IndexError:
        pass

    if chain.shape[0] == 0:
        print("It seems the chain is empty. Have you already run the MCMC?")
    else:
        labels = kwargs.pop('labels', ["$r$", r"$\theta$", "$f$"])
        fig = corner.corner(chain, labels=labels, **kwargs)
    if save:
        plt.savefig('corner_plot.pdf')
        plt.close(fig)
    else:
        plt.show()


def confidence(isamples, cfd=68.27, bins=100, gaussian_fit=False, weights=None,
               verbose=True, save=False, **kwargs):
    """
    Determine the highly probable value for each model parameter, as well as
    the 1-sigma confidence interval.
    
    Parameters
    ----------
    isamples: numpy.array
        The independent samples for each model parameter.
    cfd: float, optional
        The confidence level given in percentage.
    bins: int, optional
        The number of bins used to sample the posterior distributions.
    gaussian_fit: boolean, optional
        If True, a gaussian fit is performed in order to determine (\mu,\sigma)
    weights : (n, ) array_like or None, optional
        An array of weights for each sample.
    verbose: boolean, optional
        Display information in the shell.
    save: boolean, optional
        If "True", a txt file with the results is saved in the output
        repository.
    kwargs: optional
        Additional attributes are passed to the matplotlib hist() method.
        
    Returns
    -------
    out: tuple
        A 2 elements tuple with the highly probable solution and the confidence
        interval.
        
    """

    plsc = kwargs.pop('plsc', 0.001)
    title = kwargs.pop('title', None)
        
    output_file = kwargs.pop('filename', 'confidence.txt')
        
    try:
        l = isamples.shape[1]
    except:
        l = 1
     
    confidenceInterval = {}
    val_max = {}
    pKey = ['r', 'theta', 'f']
    
    if cfd == 100:
        cfd = 99.9
        
    #########################################
    ##  Determine the confidence interval  ##
    #########################################
    if gaussian_fit:
        mu = np.zeros(3)
        sigma = np.zeros_like(mu)
    
    if gaussian_fit:
        fig, ax = plt.subplots(2, 3, figsize=(12,8))
    else:
        fig, ax = plt.subplots(1, 3, figsize=(12,4))
    
    for j in range(l):
        label_file = ['r', 'theta', 'flux']
        label = [r'$\Delta r$', r'$\Delta \theta$', r'$\Delta f$']
        
        if gaussian_fit:
            n, bin_vertices, _ = ax[0][j].hist(isamples[:,j], bins=bins,
                                               weights=weights, histtype='step',
                                               edgecolor='gray')
        else:
            n, bin_vertices, _ = ax[j].hist(isamples[:,j], bins=bins,
                                            weights=weights, histtype='step',
                                            edgecolor='gray')
        bins_width = np.mean(np.diff(bin_vertices))
        surface_total = np.sum(np.ones_like(n)*bins_width * n)
        n_arg_sort = np.argsort(n)[::-1]
        
        test = 0
        pourcentage = 0
        for k, jj in enumerate(n_arg_sort):
            test = test + bins_width*n[jj]
            pourcentage = test/surface_total*100
            if pourcentage > cfd:
                if verbose:
                    msg = 'percentage for {}: {}%'
                    print(msg.format(label_file[j], pourcentage))
                break
        n_arg_min = n_arg_sort[:k].min()
        n_arg_max = n_arg_sort[:k+1].max()
        
        if n_arg_min == 0:
            n_arg_min += 1
        if n_arg_max == bins:
            n_arg_max -= 1
        
        val_max[pKey[j]] = bin_vertices[n_arg_sort[0]]+bins_width/2.
        confidenceInterval[pKey[j]] = np.array([bin_vertices[n_arg_min-1],
                                               bin_vertices[n_arg_max+1]]
                                               - val_max[pKey[j]])
                        
        arg = (isamples[:, j] >= bin_vertices[n_arg_min - 1]) * \
              (isamples[:, j] <= bin_vertices[n_arg_max + 1])
        if gaussian_fit:
            ax[0][j].hist(isamples[arg,j], bins=bin_vertices,
                          facecolor='gray', edgecolor='darkgray',
                          histtype='stepfilled', alpha=0.5)
            ax[0][j].vlines(val_max[pKey[j]], 0, n[n_arg_sort[0]],
                            linestyles='dashed', color='red')
            ax[0][j].set_xlabel(label[j])
            if j == 0:
                ax[0][j].set_ylabel('Counts')

            mu[j], sigma[j] = norm.fit(isamples[:, j])
            n_fit, bins_fit = np.histogram(isamples[:, j], bins, normed=1,
                                           weights=weights)
            ax[1][j].hist(isamples[:, j], bins, normed=1, weights=weights,
                          facecolor='gray', edgecolor='darkgray',
                          histtype='step')
            y = normpdf(bins_fit, mu[j], sigma[j])
            ax[1][j].plot(bins_fit, y, 'r--', linewidth=2, alpha=0.7)

            ax[1][j].set_xlabel(label[j])
            if j == 0:
                ax[1][j].set_ylabel('Counts')

            if title is not None:
                msg = r"{}   $\mu$ = {:.4f}, $\sigma$ = {:.4f}"
                ax[1][j].set_title(msg.format(title, mu[j], sigma[j]),
                                   fontsize=10)

        else:
            ax[j].hist(isamples[arg,j],bins=bin_vertices, facecolor='gray',
                       edgecolor='darkgray', histtype='stepfilled',
                       alpha=0.5)
            ax[j].vlines(val_max[pKey[j]], 0, n[n_arg_sort[0]],
                         linestyles='dashed', color='red')
            ax[j].set_xlabel(label[j])
            if j == 0:
                ax[j].set_ylabel('Counts')

            if title is not None:
                ax[1].set_title(title, fontsize=10)

        plt.tight_layout(w_pad=0.1)

    if save:
        if gaussian_fit:
            plt.savefig('confi_hist_flux_r_theta_gaussfit.pdf')
        else:
            plt.savefig('confi_hist_flux_r_theta.pdf')

    if verbose:
        print('\n\nConfidence intervals:')
        print('r: {} [{},{}]'.format(val_max['r'],
                                     confidenceInterval['r'][0],
                                     confidenceInterval['r'][1]))
        print('theta: {} [{},{}]'.format(val_max['theta'],
                                         confidenceInterval['theta'][0],
                                         confidenceInterval['theta'][1]))
        print('flux: {} [{},{}]'.format(val_max['f'],
                                        confidenceInterval['f'][0],
                                        confidenceInterval['f'][1]))
        if gaussian_fit:
            print()
            print('Gaussian fit results:')
            print('r: {} +-{}'.format(mu[0], sigma[0]))
            print('theta: {} +-{}'.format(mu[1], sigma[1]))
            print('f: {} +-{}'.format(mu[2], sigma[2]))

    ##############################################
    ##  Write inference results in a text file  ##
    ##############################################
    if save:
        with open(output_file, "w") as f:
            f.write('###########################')
            f.write('####   INFERENCE TEST   ###')
            f.write('###########################')
            f.write(' ')
            f.write('Results of the MCMC fit')
            f.write('----------------------- ')
            f.write(' ')
            f.write('>> Position and flux of the planet (highly probable):')
            f.write('{} % confidence interval'.format(cfd))
            f.write(' ')

            for i in range(3):
                confidenceMax = confidenceInterval[pKey[i]][1]
                confidenceMin = -confidenceInterval[pKey[i]][0]
                if i == 2:
                    text = '{}: \t\t\t{:.3f} \t-{:.3f} \t+{:.3f}'
                else:
                    text = '{}: \t\t\t{:.3f} \t\t-{:.3f} \t\t+{:.3f}'
                    
                f.write(text.format(pKey[i], val_max[pKey[i]],
                                    confidenceMin, confidenceMax))
            
            f.write(' ')
            f.write('Platescale = {} mas'.format(plsc*1000))
            f.write('r (mas): \t\t{:.2f} \t\t-{:.2f} \t\t+{:.2f}'.format(
                        val_max[pKey[0]]*plsc*1000,
                        -confidenceInterval[pKey[0]][0]*plsc*1000,
                        confidenceInterval[pKey[0]][1]*plsc*1000))

    if gaussian_fit:
        return mu, sigma
    else:
        return val_max, confidenceInterval
