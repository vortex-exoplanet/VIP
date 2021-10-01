#! /usr/bin/env python

"""
Module with the MCMC (``emcee``) sampling for model spectra parameter 
estimation.
"""

__author__ = 'V. Christiaens, O. Wertz, Carlos Alberto Gomez Gonzalez'
__all__ = ['mcmc_spec_sampling',
           'spec_lnprob',
           'spec_chain_zero_truncated',
           'spec_show_corner_plot',
           'spec_show_walk_plot',
           'spec_confidence']

from astropy import constants as con
import numpy as np
import os
from os.path import isdir, isfile
import emcee
import inspect
import datetime
import corner
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
import pickle
from scipy.stats import norm
from ..conf import time_ini, timing
from ..conf.utils_conf import sep
from ..fits import open_fits, write_fits
from ..negfc.utils_mcmc import gelman_rubin, autocorr_test
from .chi import goodness_of_fit
from .model_resampling import make_model_from_params, make_resampled_models
from .utils_spec import mj_from_rj_and_logg
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


def spec_lnprior(params, labels, bounds, priors=None):
    """ Define the prior log-function.
    
    Parameters
    ----------
    params: tuple
        The model parameters.
    labels: Tuple of strings
        Tuple of labels in the same order as initial_state, that is:
        - first all parameters related to loaded models (e.g. 'Teff', 'logg')
        - then the planet photometric radius 'R', in Jupiter radius
        - (optionally) the flux of emission lines (labels should match those in
        the em_lines dictionary), in units of the model spectrum (times mu)
        - (optionally) the optical extinction 'Av', in mag
        - (optionally) the ratio of total to selective optical extinction 'Rv'
        - (optionally) 'Tbb1', 'Rbb1', 'Tbb2', 'Rbb2', etc. for each extra bb
        contribution.
    bounds: dictionary
        Each entry should be associated with a tuple corresponding to lower and 
        upper bounds respectively. Bounds should be provided for ALL model
        parameters, including 'R' (planet photometric radius). 'Av' (optical 
        extinction) is optional. If provided here, Av will also be fitted.
        All keywords that are neither 'R', 'Av' nor 'M' will 
        be considered model grid parameters.
        Example for BT-SETTL: bounds = {'Teff':(1000,2000), 'logg':(3.0,4.5),
        'R':(0.1,5), 'Av':(0.,2.5)}
        'M' can be used for a prior on the mass of the planet. In that case the
        corresponding prior log probability is computed from the values for 
        parameters 'logg' and 'R'.
    priors: dictionary, opt
        If not None, sets prior estimates for each parameter of the model. Each 
        entry should be set to either None (no prior) or a tuple of 2 elements 
        containing prior estimate and uncertainty on the estimate.
        Missing entries (i.e. provided in bounds dictionary but not here) will
        be associated no prior.
        e.g. priors = {'Teff':(1600,100), 'logg':(3.5,0.5),
                       'R':(1.6,0.1), 'Av':(1.8,0.2), 'M':(10,3)}
        Important: dictionary entry names should match exactly those of bounds.
    
    Returns
    -------
    out: float.
        If all parameters are within bounds:
            * 0 if no prior,
            * the sum of gaussian log proba for each prior otherwise.
        If at least one model parameters is out of bounds:
        returns -np.inf 
    """
    n_params = len(params)
    n_labs = len(labels)
    n_dico = len(bounds)    
    if n_dico!=n_params or n_dico != n_labs:
        raise TypeError('params, labels and bounds should have same length')

    cond = True
    for pp in range(n_params):
        if not bounds[labels[pp]][0] <= params[pp] <= bounds[labels[pp]][1]:
            cond=False

    if cond:
        lnprior = 0.
        if priors is not None:
            for key, prior in priors.items():
                if key == 'M' and 'logg' in labels:
                    idx_logg =labels.index("logg")
                    idx_rp = labels.index("R")
                    rp = params[idx_rp]
                    logg = params[idx_logg]
                    mp = mj_from_rj_and_logg(rp, logg)
                    lnprior += -0.5 * (mp - prior[0])**2 / prior[1]**2
                else:
                    idx_prior = labels.index(key)
                    lnprior += -0.5*(params[idx_prior]-prior[0])**2/prior[1]**2
        return lnprior
    else:
        return -np.inf


def spec_lnlike(params, labels, grid_param_list, lbda_obs, spec_obs, err_obs, 
                dist, model_grid=None, model_reader=None, em_lines={}, 
                em_grid={}, dlbda_obs=None, instru_corr=None, 
                instru_fwhm=None, instru_idx=None, filter_reader=None, 
                AV_bef_bb=False, units_obs='si', units_mod='si', interp_order=1):
    """ Define the likelihood log-function.
    
    Parameters
    ----------
    params : tuple
        Set of models parameters for which the model grid has to be 
        interpolated.
    labels: Tuple of strings
        Tuple of labels in the same order as initial_state, that is:
        - first all parameters related to loaded models (e.g. 'Teff', 'logg')
        - then the planet photometric radius 'R', in Jupiter radius
        - (optionally) the flux of emission lines (labels should match those in
        the em_lines dictionary), in units of the model spectrum (times mu)
        - (optionally) the optical extinction 'Av', in mag
        - (optionally) the ratio of total to selective optical extinction 'Rv'
        - (optionally) 'Tbb1', 'Rbb1', 'Tbb2', 'Rbb2', etc. for each extra bb
        contribution.
    grid_param_list : list of 1d numpy arrays/lists OR None
        - If list, should contain list/numpy 1d arrays with available grid of 
        model parameters. 
        - Set to None for a pure n-blackbody fit, n=1,2,...
        - Note1: model grids should not contain grids on radius and Av, but 
        these should still be passed in initial_state (Av optional).
        - Note2: for a combined grid model + black body, just provide
        the grid parameter list here, and provide values for 'Tbbn' and 'Rbbn'
        in initial_state, labels and bounds.
    lbda_obs : numpy 1d ndarray or list
        Wavelength of observed spectrum. If several instruments, should be 
        ordered per instrument, not necessarily as monotonically increasing 
        wavelength. Hereafter, n_ch = len(lbda_obs).
    spec_obs : numpy 1d ndarray or list
        Observed spectrum for each value of lbda_obs.
    err_obs : numpy 1d/2d ndarray or list
        Uncertainties on the observed spectrum. If 2d array, should be [2,n_ch]
        where the first (resp. second) column corresponds to lower (upper) 
        uncertainty, and n_ch is the length of lbda_obs and spec_obs.
    dist :  float
        Distance in parsec, used for flux scaling of the models.
    model_grid : numpy N-d array, optional
        If provided, should contain the grid of model spectra for each
        free parameter of the given grid. I.e. for a grid of n_T values of Teff 
        and n_g values of Logg, the numpy array should be n_T x n_g x n_ch x 2, 
        where n_ch is the number of wavelengths for the observed spectrum,
        and the last 2 dims are for wavelength and fluxes respectively.
        If provided, takes precedence over model_name/model_reader.
    model_reader : python routine, opt
        External routine that reads a model file and returns a 2D numpy array, 
        where the first column corresponds to wavelengths, and the second 
        contains model values. See example routine in model_interpolation() 
        description.
    em_lines: dictionary, opt
        Dictionary of emission lines to be added on top of the model spectrum.
        Each dict entry should be the name of the line, assigned to a tuple of
        4 values: 
            1) the wavelength (in mu); 
            2) a string indicating whether line intensity is expressed in flux 
            ('F'), luminosity ('L') or log(L/LSun) ("LogL");
            3) the FWHM of the gaussian (or None if to be set automatically); 
            4) whether the FWHM is expressed in 'nm', 'mu' or 'km/s'. 
        The third and fourth can also be set to None. In that case, the FWHM of 
        the gaussian will automatically be set to the equivalent width of the
        line, calculated from the flux to be injected and the continuum 
        level (measured in the grid model to which the line is injected). 
        Examples: em_lines = {'BrG':(2.1667,'F',None, None)};
                  em_lines = {'BrG':(2.1667,'LogL', 100, 'km/s')}
    em_grid: dictionary pointing to lists, opt
        Dictionary where each entry corresponds to an emission line and points
        to a list of values to inject for emission line fluxes. For computation 
        efficiency, interpolation will be performed between the points of this 
        grid during the MCMC sampling. Dict entries should match labels and 
        em_lines.
    dlbda_obs: numpy 1d ndarray or list, optional
        Spectral channel width for the observed spectrum. It should be provided 
        IF one wants to weigh each point based on the spectral 
        resolution of the respective instruments (as in Olofsson et al. 2016).
    instru_corr : numpy 2d ndarray or list, optional
        Spectral correlation throughout post-processed images in which the 
        spectrum is measured. It is specific to the combination of instrument, 
        algorithm and radial separation of the companion from the central star.
        Can be computed using distances.spectral_correlation(). In case of
        a spectrum obtained with different instruments, build it with
        distances.combine_corrs(). If not provided, it will consider the 
        uncertainties in each spectral channels are independent. See Greco & 
        Brandt (2017) for details.
    instru_fwhm : float or list, optional
        The instrumental spectral fwhm provided in nm. This is used to convolve
        the model spectrum. If several instruments are used, provide a list of 
        instru_fwhm values, one for each instrument whose spectral resolution
        is coarser than the model - including broad band
        filter FWHM if relevant.
    instru_idx: numpy 1d array, optional
        1d array containing an index representing each instrument used 
        to obtain the spectrum, label them from 0 to n_instru. Zero for points 
        that don't correspond to any instru_fwhm provided above, and i in 
        [1,n_instru] for points associated to instru_fwhm[i-1]. This parameter 
        must be provided if the spectrum consists of points obtained with 
        different instruments.
    filter_reader: python routine, optional
        External routine that reads a filter file and returns a 2D numpy array, 
        where the first column corresponds to wavelengths, and the second 
        contains transmission values. Important: if not provided, but strings 
        are detected in instru_fwhm, the default format assumed for the files:
        - first row containing header
        - starting from 2nd row: 1st column: WL in mu, 2nd column: transmission
        Note: files should all have the same format and wavelength units.
    AV_bef_bb: bool, optional
        If both extinction and an extra bb component are free parameters, 
        whether to apply extinction before adding the BB component (e.g. 
        extinction mostly from circumplanetary dust) or after the BB component
        (e.g. mostly insterstellar extinction).
    units_obs : str, opt {'si','cgs','jy'}
        Units of observed spectrum. 'si' for W/m^2/mu; 'cgs' for ergs/s/cm^2/mu 
        or 'jy' for janskys.
    units_mod: str, opt {'si','cgs','jy'}
        Units of the model. 'si' for W/m^2/mu; 'cgs' for ergs/s/cm^2/mu or 'jy'
        for janskys. If different to units_obs, the spectrum units will be 
        converted.
    interp_order: int, opt, {-1,0,1} 
        Interpolation mode for model interpolation.
        -1: log interpolation (i.e. linear interpolatlion on log(Flux))
        0: nearest neighbour model.
        1: Order 1 spline interpolation.
        
    Returns
    -------
    out: float
        The log of the likelihood.
        
    """

    if grid_param_list is not None:
        if model_grid is None and model_reader is None:
            msg = "model_name and model_reader must be provided"
            raise TypeError(msg)
                           
    lbda_mod, spec_mod = make_model_from_params(params, labels, grid_param_list, 
                                                dist, lbda_obs, model_grid, 
                                                model_reader, em_lines, em_grid, 
                                                dlbda_obs, instru_fwhm, 
                                                instru_idx, filter_reader, 
                                                AV_bef_bb, units_obs, units_mod, 
                                                interp_order)
    
    # evaluate the goodness of fit indicator
    chi = goodness_of_fit(lbda_obs, spec_obs, err_obs, lbda_mod, spec_mod, 
                          dlbda_obs=dlbda_obs, instru_corr=instru_corr, 
                          instru_fwhm=instru_fwhm, instru_idx=instru_idx, 
                          filter_reader=filter_reader, plot=False, outfile=None)
    
    # log likelihood
    lnlikelihood = -0.5 * chi
    
    return lnlikelihood


def spec_lnprob(params, labels, bounds, grid_param_list, lbda_obs, spec_obs, 
                err_obs, dist, model_grid=None, model_reader=None, em_lines={},
                em_grid={}, dlbda_obs=None, instru_corr=None, instru_fwhm=None, 
                instru_idx=None, filter_reader=None, AV_bef_bb=False, 
                units_obs='si', units_mod='si', interp_order=1, priors=None, 
                physical=True):
    """ Define the probability log-function as the sum between the prior and
    likelihood log-functions.
    
    Parameters
    ----------
    params: tuple
        The model parameters.
    labels: Tuple of strings
        Tuple of labels in the same order as initial_state, that is:
        - first all parameters related to loaded models (e.g. 'Teff', 'logg')
        - then the planet photometric radius 'R', in Jupiter radius
        - (optionally) the flux of emission lines (labels should match those in
        the em_lines dictionary), in units of the model spectrum (times mu)
        - (optionally) the optical extinction 'Av', in mag
        - (optionally) the ratio of total to selective optical extinction 'Rv'
        - (optionally) 'Tbb1', 'Rbb1', 'Tbb2', 'Rbb2', etc. for each extra bb
        contribution.
    bounds: dictionary
        Each entry should be associated with a tuple corresponding to lower and 
        upper bounds respectively. Bounds should be provided for ALL model
        parameters, including 'R' (planet photometric radius). 'Av' (optical 
        extinction) is optional. If provided here, Av will also be fitted.
        All keywords that are neither 'R', 'Av' nor 'M' will 
        be considered model grid parameters.
        Example for BT-SETTL: bounds = {'Teff':(1000,2000), 'logg':(3.0,4.5),
        'R':(0.1,5), 'Av':(0.,2.5)}
        'M' can be used for a prior on the mass of the planet. In that case the
        corresponding prior log probability is computed from the values for 
        parameters 'logg' and 'R'.
    grid_param_list : list of 1d numpy arrays/lists OR None
        - If list, should contain list/numpy 1d arrays with available grid of 
        model parameters. 
        - Set to None for a pure n-blackbody fit, n=1,2,...
        - Note1: model grids should not contain grids on radius and Av, but 
        these should still be passed in initial_state (Av optional).
        - Note2: for a combined grid model + black body, just provide
        the grid parameter list here, and provide values for 'Tbbn' and 'Rbbn'
        in initial_state, labels and bounds.
    lbda_obs : numpy 1d ndarray or list
        Wavelength of observed spectrum. If several instruments, should be 
        ordered per instrument, not necessarily as monotonically increasing 
        wavelength. Hereafter, n_ch = len(lbda_obs).
    spec_obs : numpy 1d ndarray or list
        Observed spectrum for each value of lbda_obs.
    err_obs : numpy 1d/2d ndarray or list
        Uncertainties on the observed spectrum. If 2d array, should be [2,n_ch]
        where the first (resp. second) column corresponds to lower (upper) 
        uncertainty, and n_ch is the length of lbda_obs and spec_obs.
    dist :  float
        Distance in parsec, used for flux scaling of the models.
    model_grid : numpy N-d array, optional
        If provided, should contain the grid of model spectra for each
        free parameter of the given grid. I.e. for a grid of n_T values of Teff 
        and n_g values of Logg, the numpy array should be n_T x n_g x n_ch x 2, 
        where n_ch is the number of wavelengths for the observed spectrum,
        and the last 2 dims are for wavelength and fluxes respectively.
        If provided, takes precedence over model_name/model_reader.
    model_reader : python routine
        External routine that reads a model file and returns a 2D numpy array, 
        where the first column corresponds to wavelengths, and the second 
        contains model values. See example routine in model_interpolation() 
        description.
    em_lines: dictionary, opt
        Dictionary of emission lines to be added on top of the model spectrum.
        Each dict entry should be the name of the line, assigned to a tuple of
        4 values: 
            1) the wavelength (in mu); 
            2) a string indicating whether line intensity is expressed in flux 
            ('F'), luminosity ('L') or log(L/LSun) ("LogL");
            3) the FWHM of the gaussian (or None if to be set automatically); 
            4) whether the FWHM is expressed in 'nm', 'mu' or 'km/s'. 
        The third and fourth can also be set to None. In that case, the FWHM of 
        the gaussian will automatically be set to the equivalent width of the
        line, calculated from the flux to be injected and the continuum 
        level (measured in the grid model to which the line is injected). 
        Examples: em_lines = {'BrG':(2.1667,'F',None, None)};
                  em_lines = {'BrG':(2.1667,'LogL', 100, 'km/s')}
    em_grid: dictionary pointing to lists, opt
        Dictionary where each entry corresponds to an emission line and points
        to a list of values to inject for emission line fluxes. For computation 
        efficiency, interpolation will be performed between the points of this 
        grid during the MCMC sampling. Dict entries should match labels and 
        em_lines.
    dlbda_obs: numpy 1d ndarray or list, optional
        Spectral channel width for the observed spectrum. It should be provided 
        IF one wants to weigh each point based on the spectral 
        resolution of the respective instruments (as in Olofsson et al. 2016).
    instru_corr : numpy 2d ndarray or list, optional
        Spectral correlation throughout post-processed images in which the 
        spectrum is measured. It is specific to the combination of instrument, 
        algorithm and radial separation of the companion from the central star.
        Can be computed using distances.spectral_correlation(). In case of
        a spectrum obtained with different instruments, build it with
        distances.combine_corrs(). If not provided, it will consider the 
        uncertainties in each spectral channels are independent. See Greco & 
        Brandt (2017) for details.
    instru_fwhm : float or list, optional
        The instrumental spectral fwhm provided in nm. This is used to convolve
        the model spectrum. If several instruments are used, provide a list of 
        instru_fwhm values, one for each instrument whose spectral resolution
        is coarser than the model - including broad band
        filter FWHM if relevant.
    instru_idx: numpy 1d array, optional
        1d array containing an index representing each instrument used 
        to obtain the spectrum, label them from 0 to n_instru. Zero for points 
        that don't correspond to any instru_fwhm provided above, and i in 
        [1,n_instru] for points associated to instru_fwhm[i-1]. This parameter 
        must be provided if the spectrum consists of points obtained with 
        different instruments.
    filter_reader: python routine, optional
        External routine that reads a filter file and returns a 2D numpy array, 
        where the first column corresponds to wavelengths, and the second 
        contains transmission values. Important: if not provided, but strings 
        are detected in instru_fwhm, the default format assumed for the files:
        - first row containing header
        - starting from 2nd row: 1st column: WL in mu, 2nd column: transmission
        Note: files should all have the same format and wavelength units.
    AV_bef_bb: bool, optional
        If both extinction and an extra bb component are free parameters, 
        whether to apply extinction before adding the BB component (e.g. 
        extinction mostly from circumplanetary dust) or after the BB component
        (e.g. mostly insterstellar extinction).
    units_obs : str, opt {'si','cgs','jy'}
        Units of observed spectrum. 'si' for W/m^2/mu; 'cgs' for ergs/s/cm^2/mu 
        or 'jy' for janskys.
    units_mod: str, opt {'si','cgs','jy'}
        Units of the model. 'si' for W/m^2/mu; 'cgs' for ergs/s/cm^2/mu or 'jy'
        for janskys. If different to units_obs, the spectrum units will be 
        converted.
    interp_order: int, opt, {-1,0,1} 
        Interpolation mode for model interpolation.
        -1: log interpolation (i.e. linear interpolatlion on log(Flux))
        0: nearest neighbour model.
        1: Order 1 spline interpolation.
    priors: dictionary, opt
        If not None, sets prior estimates for each parameter of the model. Each 
        entry should be set to either None (no prior) or a tuple of 2 elements 
        containing prior estimate and uncertainty on the estimate.
        Missing entries (i.e. provided in bounds dictionary but not here) will
        be associated no prior.
        e.g. priors = {'Teff':(1600,100), 'logg':(3.5,0.5),
                       'R':(1.6,0.1), 'Av':(1.8,0.2), 'M':(10,3)}
        Important: dictionary entry names should match exactly those of bounds.
    physical: bool, opt
        In case of extra black body component(s) to a photosphere, whether to 
        force lower temperature than the photosphere effective temperature.
        
    Returns
    -------
    out: float
        The probability log-function.
    
    """
    
    lp = spec_lnprior(params, labels, bounds, priors)
    
    if np.isinf(lp):
        return -np.inf
    
    if 'Tbb1' in labels:
        condT = ('T' in labels or 'Teff' in labels)
        n_bb = 0
        for label in labels:
            if 'Tbb' in label:
                n_bb+=1
        idx_Tbb1 = labels.index("Tbb1")
        for ii in range(n_bb):
            idx = ii*2
            if grid_param_list is not None and condT:
                if 'T' in labels:
                    idx_Teff = labels.index("T")
                else:
                    idx_Teff = labels.index("Teff")
                # COND 1: only allow Tbb < Teff
                if physical and params[idx_Tbb1+idx] >= params[idx_Teff]:
                    return -np.inf
                elif physical:
                    idx_R= labels.index("R")
                    R = params[idx_R]
                    Rbb = params[idx_Tbb1+idx+1]
                    if Rbb<=R:
                        pass
                    else:
                        Teq = params[idx_Teff] * np.sqrt(R/(2*Rbb))
                        # COND 2: only allow Tbb <= Teq at Rbb
                        if params[idx_Tbb1+idx] >= Teq:
                            return -np.inf
            if ii>0:
                # avoid unnecessary degenerecaies by only allowing:
                # Teff1 > Teff2, Teff2 > Teff3
                if params[idx_Tbb1+idx] >= params[idx_Tbb1+idx-2]:
                    return -np.inf
        
    return lp + spec_lnlike(params, labels, grid_param_list, lbda_obs, spec_obs, 
                            err_obs, dist, model_grid, model_reader, em_lines,
                            em_grid, dlbda_obs, instru_corr, instru_fwhm, 
                            instru_idx, filter_reader, AV_bef_bb, units_obs, 
                            units_mod, interp_order)


def mcmc_spec_sampling(lbda_obs, spec_obs, err_obs, dist, grid_param_list, 
                       initial_state, labels, bounds, resamp_before=True, 
                       model_grid=None, model_reader=None, em_lines={}, 
                       em_grid={}, dlbda_obs=None, instru_corr=None, 
                       instru_fwhm=None, instru_idx=None, filter_reader=None, 
                       AV_bef_bb=False, units_obs='si', units_mod='si', 
                       interp_order=1, priors=None, physical=True, 
                       interp_nonexist=True, ini_ball=1e-1, a=2.0, 
                       nwalkers=1000, niteration_min=10, niteration_limit=1000, 
                       niteration_supp=0, check_maxgap=20, conv_test='ac', 
                       ac_c=50, ac_count_thr=3, burnin=0.3, rhat_threshold=1.01, 
                       rhat_count_threshold=1, grid_name='resamp_grid.fits', 
                       output_dir='specfit/', output_file=None, nproc=1, 
                       display=False, verbosity=0, save=False):
    """ Runs an affine invariant MCMC sampling algorithm in order to determine
    the most likely parameters for given spectral model and observed spectrum. 
    Allowed features:
        - Spectral models can either be read from a grid (e.g. BT-SETTL) or 
        be purely parametric (e.g. a blackbody model). 
        - Extinction (A_V) and total-to-selective optical extinction ratio (R_V)
        can be sampled. Default: A_V=0. If non-zero, default R_V=3.1 (ISM).
        - A dictionary of emission lines can be provided and their flux can be 
        sampled too.
        - Gaussian priors can be provided for each parameter, including the 
        mass of the object. The latter will be used if 'logg' is a parameter.
        - Spectral correlation between measurements will be taken into account 
        if provided in 'instru_corr'.
        - Convolution of the model spectra with instrumental FWHM or 
        photometric filter can be performed using 'instru_fwhm' and/or 
        'filter_reader' (done before resampling to observed).
        - The weight of each observed point will be directly proportional to
        Delta lbda_i/lbda_i, where Delta lbda_i is either the FWHM of the 
        photometric filter (imager) or the width of the spectral channel (IFS).
        - MCMC convergence criterion can either be based on auto-correlation 
        time (default) or the Gelman-Rubin test.
    The result of this procedure is a chain with the samples from the posterior 
    distributions of each of the free parameters in the model.
    More details in Christiaens et al. (2021).
    
    Parameters
    ----------
    lbda_obs : numpy 1d ndarray or list
        Wavelength of observed spectrum. If several instruments, should be 
        ordered per instrument, not necessarily as monotonically increasing 
        wavelength. Hereafter, n_ch = len(lbda_obs).
    spec_obs : numpy 1d ndarray or list
        Observed spectrum for each value of lbda_obs.
    err_obs : numpy 1d/2d ndarray or list
        Uncertainties on the observed spectrum. If 2d array, should be [2,n_ch]
        where the first (resp. second) column corresponds to lower (upper) 
        uncertainty, and n_ch is the length of lbda_obs and spec_obs.
    dist :  float
        Distance in parsec, used for flux scaling of the models.
    grid_param_list : list of 1d numpy arrays/lists OR None
        - If list, should contain list/numpy 1d arrays with available grid of 
        model parameters. 
        - Set to None for a pure n-blackbody fit, n=1,2,...
        - Note1: model grids should not contain grids on radius and Av, but 
        these should still be passed in initial_state (Av optional).
        - Note2: for a combined grid model + black body, just provide
        the grid parameter list here, and provide values for 'Tbbn' and 'Rbbn'
        in initial_state, labels and bounds.
    initial_state: tuple of floats
        Initial guess on the best fit parameters of the spectral fit. Length of 
        the tuple should match the total number of free parameters. Walkers
        will all be initialised in a small ball of parameter space around that
        first guess.
        - first all parameters related to loaded models (e.g. 'Teff', 'logg')
        - then the planet photometric radius 'R', in Jupiter radius
        - (optionally) the intensity of emission lines (labels must match 
        those in the em_lines dict), in units of the model spectrum (x mu)
        - (optionally) the optical extinction 'Av', in mag
        - (optionally) the ratio of total to selective optical extinction 'Rv'
        - (optionally) 'Tbb1', 'Rbb1', 'Tbb2', 'Rbb2', etc. for each extra bb
        contribution
    labels: Tuple of strings
        Tuple of labels in the same order as initial_state, that is:
        - first all parameters related to loaded models (e.g. 'Teff', 'logg')
        - then the planet photometric radius 'R', in Jupiter radius
        - (optionally) the flux of emission lines (labels should match those in
        the em_lines dictionary), in units of the model spectrum (times mu)
        - (optionally) the optical extinction 'Av', in mag
        - (optionally) the ratio of total to selective optical extinction 'Rv'
        - (optionally) 'Tbb1', 'Rbb1', 'Tbb2', 'Rbb2', etc. for each extra bb
        contribution.
    bounds: dictionary
        Each entry should be associated with a tuple corresponding to lower and 
        upper bounds respectively. Bounds should be provided for ALL model
        parameters, including 'R' (planet photometric radius). 'Av' (optical 
        extinction) is optional. If provided here, Av will also be fitted.
        Example for BT-SETTL: bounds = {'Teff':(1000,2000), 'logg':(3.0,4.5),
        'R':(0.1,5), 'Av':(0.,2.5)}
        'M' can be used for a prior on the mass of the planet. In that case the
        corresponding prior log probability is computed from the values for 
        parameters 'logg' and 'R' (if both exist).
    resamp_before: bool, optional
        Whether to prepare the whole grid of resampled models before entering 
        the MCMC, i.e. to avoid doing it at every MCMC step. Recommended.
        Only reason not to: model grid is too large and individual models 
        require being opened and resampled at each step.
    model_grid : numpy N-d array, optional
        If provided, should contain the grid of model spectra for each
        free parameter of the given grid. I.e. for a grid of n_T values of Teff 
        and n_g values of Logg, the numpy array should be n_T x n_g x n_ch x 2, 
        where n_ch is the number of wavelengths for the observed spectrum,
        and the last 2 dims are for wavelength and fluxes respectively.
        If provided, takes precedence over filename/file_reader.
    model_reader : python routine, optional
        External routine that reads a model file and returns a 2D numpy array, 
        where the first column corresponds to wavelengths, and the second 
        contains model values. See example routine in model_interpolation() 
        description.
    em_lines: dictionary, opt
        Dictionary of emission lines to be added on top of the model spectrum.
        Each dict entry should be the name of the line, assigned to a tuple of
        4 values: 
            1) the wavelength (in mu); 
            2) a string indicating whether line intensity is expressed in flux 
            ('F'), luminosity ('L') or log(L/LSun) ("LogL");
            3) the FWHM of the gaussian (or None if to be set automatically); 
            4) whether the FWHM is expressed in 'nm', 'mu' or 'km/s'. 
        The third and fourth can also be set to None. In that case, the FWHM of 
        the gaussian will automatically be set to the equivalent width of the
        line, calculated from the flux to be injected and the continuum 
        level (measured in the grid model to which the line is injected). 
        Examples: em_lines = {'BrG':(2.1667,'F',None, None)};
                  em_lines = {'BrG':(2.1667,'LogL', 100, 'km/s')}
    em_grid: dictionary pointing to lists, opt
        Dictionary where each entry corresponds to an emission line and points
        to a list of values to inject for emission line fluxes. For computation 
        efficiency, interpolation will be performed between the points of this 
        grid during the MCMC sampling. Dict entries should match labels and 
        em_lines.
    dlbda_obs: numpy 1d ndarray or list, optional
        Spectral channel width for the observed spectrum. It should be provided 
        IF one wants to weigh each point based on the spectral 
        resolution of the respective instruments (as in Olofsson et al. 2016).
    instru_corr : numpy 2d ndarray or list, optional
        Spectral correlation throughout post-processed images in which the 
        spectrum is measured. It is specific to the combination of instrument, 
        algorithm and radial separation of the companion from the central star.
        Can be computed using distances.spectral_correlation(). In case of
        a spectrum obtained with different instruments, build it with
        distances.combine_corrs(). If not provided, it will consider the 
        uncertainties in each spectral channels are independent. See Greco & 
        Brandt (2017) for details.
    instru_fwhm : float OR list of either floats or strings, optional
        The instrumental spectral fwhm provided in nm. This is used to convolve
        the model spectrum. If several instruments are used, provide a list of 
        instru_fwhm values, one for each instrument whose spectral resolution
        is coarser than the model - including broad band filter FWHM if 
        relevant.
        If strings are provided, they should correspond to filenames (including 
        full paths) of text files containing the filter information for each 
        observed wavelength. Strict format: 
    instru_idx: numpy 1d array, optional
        1d array containing an index representing each instrument used 
        to obtain the spectrum, label them from 0 to n_instru. Zero for points 
        that don't correspond to any instru_fwhm provided above, and i in 
        [1,n_instru] for points associated to instru_fwhm[i-1]. This parameter 
        must be provided if the spectrum consists of points obtained with 
        different instruments.
    filter_reader: python routine, optional
        External routine that reads a filter file and returns a 2D numpy array, 
        where the first column corresponds to wavelengths, and the second 
        contains transmission values. Important: if not provided, but strings 
        are detected in instru_fwhm, the default file reader will be used. 
        It assumes the following format for the files:
        - first row containing header
        - starting from 2nd row: 1st column: wavelength, 2nd col.: transmission
        - Unit of wavelength can be provided in parentheses of first header key
        name: e.g. "WL(AA)" for angstrom, "wavelength(mu)" for micrometer or
        "lambda(nm)" for nanometer. Note: Only what is in parentheses matters.
        Important: filter files should all have the same format and WL units.
    AV_bef_bb: bool, optional
        If both extinction and an extra bb component are free parameters, 
        whether to apply extinction before adding the BB component (e.g. 
        extinction mostly from circumplanetary dust) or after the BB component
        (e.g. mostly insterstellar extinction).
    units_obs : str, opt {'si','cgs','jy'}
        Units of observed spectrum. 'si' for W/m^2/mu; 'cgs' for ergs/s/cm^2/mu 
        or 'jy' for janskys.
    units_mod: str, opt {'si','cgs','jy'}
        Units of the model. 'si' for W/m^2/mu; 'cgs' for ergs/s/cm^2/mu or 'jy'
        for janskys. If different to units_obs, the spectrum units will be 
        converted.
    interp_order: int, opt, {-1,0,1} 
        Interpolation mode for model interpolation.
        -1: log interpolation (i.e. linear interpolatlion on log(Flux))
        0: nearest neighbour model.
        1: Order 1 spline interpolation.
    priors: dictionary, opt
        If not None, sets prior estimates for each parameter of the model. Each 
        entry should be set to either None (no prior) or a tuple of 2 elements 
        containing prior estimate and uncertainty on the estimate.
        Missing entries (i.e. provided in bounds dictionary but not here) will
        be associated no prior.
        e.g. priors = {'Teff':(1600,100), 'logg':(3.5,0.5),
                       'R':(1.6,0.1), 'Av':(1.8,0.2), 'M':(10,3)}
        Important: dictionary entry names should match exactly those of bounds.
    physical: bool, opt
        In case of extra black body component(s) to a photosphere, whether to 
        force lower temperature than the photosphere effective temperature.
    interp_nonexist: bool, opt
        Whether to interpolate non-existing models in the grid. Only used if 
        resamp_before is set to True.
    ini_ball: float or string, default=1e-1
        Size of the initial ball in parameter space from which walkers start 
        their chain. If "uniform" is provided, a uniform ini_ball spanning
        the bounds interval will be used to initialise walkers.
    a: float, default=2.0
        The proposal scale parameter. See notes.
    nwalkers: int, default: 1000
        Number of walkers
    niteration_min: int, optional
        Steps per walker lower bound. The simulation will run at least this
        number of steps per walker.
    niteration_limit: int, optional
        Steps per walker upper bound. If the simulation runs up to
        'niteration_limit' steps without having reached the convergence
        criterion, the run is stopped.
    niteration_supp: int, optional
        Number of iterations to run after having "reached the convergence".
    burnin: float, default=0.3
        The fraction of a walker which is discarded.
    rhat_threshold: float, default=0.01
        The Gelman-Rubin threshold used for the test for nonconvergence.
    rhat_count_threshold: int, optional
        The Gelman-Rubin test must be satisfied 'rhat_count_threshold' times in
        a row before claiming that the chain has converged.
    check_maxgap: int, optional
        Maximum number of steps per walker between two convergence tests.
    conv_test: str, optional {'gb','autocorr'}
        Method to check for convergence: 
        - 'gb' for gelman-rubin test
        (http://digitalassets.lib.berkeley.edu/sdtr/ucb/text/305.pdf)
        - 'autocorr' for autocorrelation analysis 
        (https://emcee.readthedocs.io/en/stable/tutorials/autocorr/)
    nproc: int, optional
        The number of processes to use for parallelization.
    grid_name: str, optional
        Name of the fits file containing the model grid (numpy array) AFTER
        convolution+resampling as the observed spectrum given as input.
        If provided, will read it if it exists (and resamp_before is set
        to True), or make it and write it if it doesn't.
    output_dir: str, optional
        The name of the output directory which contains the output files in the 
        case  ``save`` is True.        
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
        The MCMC samples after truncation of zeros.
    lnprobability: emcee sample object
        The corresponding probabilities for each sample
        
    Notes
    -----
    The parameter `a` must be > 1. For more theoretical information concerning
    this parameter, see Goodman & Weare, 2010, Comm. App. Math. Comp. Sci.,
    5, 65, Eq. [9] p70.
    
    The parameter 'rhat_threshold' can be a numpy.array with individual
    threshold value for each model parameter.
    """
    nparams = len(initial_state)
    
    if grid_param_list is not None:
        if model_grid is None and model_reader is None:
            msg = "Either model_grid or model_reader have to be provided"
            raise TypeError(msg)
        n_gparams = len(grid_param_list)
        gp_dims = []
        for nn in range(n_gparams):
            gp_dims.append(len(grid_param_list[nn]))
        gp_dims = tuple(gp_dims)
    else:
        n_gparams = 0
        
    # format emission line dictionary and em grid
    if len(em_grid)>0:
        n_em = len(em_grid)
        em_dims = []
        for lab in labels:
            if lab in em_grid.keys():
                em_dims.append(len(em_grid[lab]))
        em_dims = tuple(em_dims)
        # update the grids depending on input units => make it surface flux
        idx_R = labels.index('R')
        for key, val in em_lines.items():
            if val[1] == 'L':
                idx_line = labels.index(key)
                # adapt grid
                R_si = initial_state[idx_R]*con.R_jup.value
                conv_fac = 4*np.pi*R_si**2
                tmp = np.array(em_grid[key])/conv_fac
                em_grid[key] = tmp.tolist()
                # adapt ini state 
                initial_state = list(initial_state)
                initial_state[idx_line] /= conv_fac
                initial_state = tuple(initial_state)
                #adapt bounds
                bounds_ori = list(bounds[key])
                bounds[key] = (bounds_ori[0]/conv_fac, bounds_ori[1]/conv_fac) 
            elif val[1] == 'LogL':
                idx_line = labels.index(key)
                R_si = initial_state[idx_R]*con.R_jup.value
                conv_fac = con.L_sun.value/(4*np.pi*R_si**2)
                tmp = np.power(10,np.array(em_grid[key]))*conv_fac
                em_grid[key] = tmp.tolist()  
                # adapt ini state 
                initial_state = list(initial_state)
                initial_state[idx_line] = conv_fac*10**initial_state[idx_line]
                initial_state = tuple(initial_state)
                #adapt bounds
                bounds_ori = list(bounds[key])
                bounds[key] = (conv_fac*10**bounds_ori[0], 
                               conv_fac*10**bounds_ori[1]) 
            if em_lines[key][2] is not None:
                if em_lines[key][-1] == 'km/s':
                    v = em_lines[key][2]
                    em_lines_tmp = list(em_lines[key])
                    em_lines_tmp[2] = (1000*v/con.c.value)*em_lines[key][0]
                    em_lines_tmp[3] = 'mu'
                    em_lines[key] = tuple(em_lines_tmp)
                elif em_lines[key][-1] == 'nm':
                    em_lines_tmp = list(em_lines[key])
                    em_lines_tmp[2] = em_lines[key][2]/1000
                    em_lines_tmp[3] = 'mu'
                    em_lines[key] = tuple(em_lines_tmp)
                elif em_lines[key][-1] != 'mu':
                    msg = "Invalid unit of FWHM for line injection"
                    raise ValueError(msg)
                
    if model_grid is not None and grid_param_list is not None:
        if model_grid.ndim-2 != n_gparams:
            msg = "Ndim-2 of model_grid should match len(grid_param_list)"
            raise TypeError(msg)
    
    if verbosity > 0:
        start_time = time_ini()
        print("       MCMC sampler for spectral fitting       ")
        print(sep)


    # If required, create the output folder.
    if save or resamp_before:
        if not output_dir:
            raise ValueError('Please provide an output directory path')
        if not isdir(output_dir):
            os.makedirs(output_dir)
        if output_dir[-1] != '/':
            output_dir = output_dir+'/'

        #output_file_tmp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
 
    # Check model grid parameters extend beyond bounds to avoid extrapolation
    if grid_param_list is not None:
        for pp in range(n_gparams):
            if grid_param_list[pp][0]>bounds[labels[pp]][0]:
                msg= "Grid has to extend beyond bounds for {}."
                msg+="\n Consider increasing the lower bound to >{}."
                raise ValueError(msg.format(labels[pp],grid_param_list[pp][0]))
            if grid_param_list[pp][-1]<bounds[labels[pp]][1]:
                msg= "Grid has to extend beyond bounds for {}."
                msg+="\n Consider decreasing the upper bound to <{}."
                raise ValueError(msg.format(labels[pp],grid_param_list[pp][1]))
                
    # Check initial state is within bounds for all params (not only grid)
    for pp in range(nparams):
        if initial_state[pp]<bounds[labels[pp]][0]:
            msg= "Initial state has to be within bounds for {}."
            msg+="\n Consider decreasing the lower bound to <{}."
            raise ValueError(msg.format(labels[pp],initial_state[pp]))            
        if initial_state[pp]>bounds[labels[pp]][1]:
            msg= "Initial state has to be within bounds for {}."
            msg+="\n Consider decreasing the upper bound to >{}."
            raise ValueError(msg.format(labels[pp],initial_state[pp]))
        
    # Prepare model grid: convolve+resample models as observations 
    if resamp_before and grid_param_list is not None:
        if isfile(output_dir+grid_name):
            model_grid = open_fits(output_dir+grid_name)
            # check its shape is consistent with grid_param_list
            if model_grid.shape[:n_gparams] != gp_dims:
                msg="the loaded model grid ({}) doesn't have expected dims ({})"
                raise TypeError(msg.format(model_grid.shape,gp_dims))
            elif model_grid.shape[-2] != len(lbda_obs):
                msg="the loaded model grid doesn't have expected WL dimension"
                raise TypeError(msg)
            elif model_grid.shape[-1] != 2:
                msg="the loaded model grid doesn't have expected last dimension"
                raise TypeError(msg)
            elif len(em_grid) > 0:
                if model_grid.shape[n_gparams:n_gparams+n_em] != em_dims:
                    msg="loaded model grid ({}) doesn't have expected dims ({})"
                    raise TypeError(msg.format(model_grid.shape,em_dims))
        else:
            model_grid = make_resampled_models(lbda_obs, grid_param_list, 
                                               model_grid, model_reader, 
                                               em_lines, em_grid, dlbda_obs, 
                                               instru_fwhm, instru_idx, 
                                               filter_reader, interp_nonexist)
            if output_dir and grid_name:
                write_fits(output_dir+grid_name, model_grid)
        # note: if model_grid is provided, it is still resampled to the 
        # same wavelengths as observed spectrum. However, if a fits name is 
        # provided in grid_name and that file exists, it is assumed the model 
        # grid in it is already resampled to match lbda_obs.

    if save and output_file is None:
        output_file = 'MCMC_results'

    
    # #########################################################################
    # Initialization of the MCMC variables                                    #
    # #########################################################################
    dim = len(labels)
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
    ac_count = 0
    chain = np.empty([nwalkers, 1, dim])
    ar_frac = np.empty([nwalkers, 1])
    nIterations = limit + supp
    rhat = np.zeros(dim)
    stop = np.inf
    
    # initialise ball of walkers
    if ini_ball == "uniform":
        pos = np.zeros([nwalkers, dim])
        for ii in range(dim):
            low_b = bounds[labels[ii]][0]
            up_b = (bounds[labels[ii]][1])
            pos[:,ii] = np.random.uniform(low_b + 0.01*(up_b-low_b), 
                                          up_b, nwalkers)
    elif isinstance(ini_ball,float):
        pos = initial_state + np.random.normal(0, ini_ball, (nwalkers, dim))
    else:
        raise ValueError("ini_ball must be string or float")
    
    sampler = emcee.EnsembleSampler(nwalkers, dim, spec_lnprob, a,
                                    args=([labels, bounds, grid_param_list, 
                                           lbda_obs, spec_obs, err_obs, dist,
                                           model_grid, model_reader, em_lines,
                                           em_grid, dlbda_obs, instru_corr, 
                                           instru_fwhm,instru_idx,filter_reader, 
                                           AV_bef_bb, units_obs, units_mod, 
                                           interp_order, priors, physical]),
                                    threads=nproc)
                                    
    start = datetime.datetime.now()

    # #########################################################################
    # Affine Invariant MCMC run
    # #########################################################################
    if verbosity > 1:
        print('\nStart of the MCMC run ...')
        print('Step  |  Duration/step (sec)  |  Remaining Estimated Time (sec)')
    
    for k, res in enumerate(sampler.sample(pos, iterations=nIterations,
                                           storechain=True)):
        elapsed = (datetime.datetime.now()-start).total_seconds()
        if verbosity > 1:
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
            empty = np.zeros([nwalkers, 2*s])
            ar_frac = np.concatenate((ar_frac, empty), axis=1)
        # Store the state of the chain
        chain[:, k] = res[0]
        ar_frac[:,k] = sampler.acceptance_fraction

        # ---------------------------------------------------------------------
        # If k meets the criterion, one tests the non-convergence.
        # ---------------------------------------------------------------------
        criterion = int(np.amin([np.ceil(itermin*(1+fraction)**geom),
                        lastcheck+np.floor(maxgap)]))
        if k == criterion:
            
            geom += 1
            lastcheck = k
            if display:
                spec_show_walk_plot(chain)
                
            # We only test the rhat if we have reached the min # of steps
            if (k+1) >= itermin and konvergence == np.inf:
                if verbosity > 1:
                    print('\n{} convergence test in progress...'.format(conv_test))
                if conv_test == 'gb':
                    thr0 = int(np.floor(burnin*k))
                    thr1 = int(np.floor((1-burnin)*k*0.25))
    
                    # We calculate the rhat for each model parameter.
                    for j in range(dim):
                        part1 = chain[:, thr0:thr0 + thr1, j].reshape(-1)
                        part2 = chain[:, thr0 + 3 * thr1:thr0 + 4 * thr1, j
                                     ].reshape(-1)
                        series = np.vstack((part1, part2))
                        rhat[j] = gelman_rubin(series)
                    if verbosity > 0:
                        print('   r_hat = {}'.format(rhat))
                        cond = rhat <= rhat_threshold
                        print('   r_hat <= threshold = {} \n'.format(cond))
                    # We test the rhat.
                    if (rhat <= rhat_threshold).all():
                        rhat_count += 1
                        if rhat_count < rhat_count_threshold:
                            if verbosity > 0:
                                msg = "Gelman-Rubin test OK {}/{}"
                                print(msg.format(rhat_count, 
                                                 rhat_count_threshold))
                        elif rhat_count >= rhat_count_threshold:
                            if verbosity > 0:
                                print('... ==> convergence reached')
                            konvergence = k
                            stop = konvergence + supp
                    else:
                        rhat_count = 0
                elif conv_test == 'ac':
                    # We calculate the auto-corr test for each model parameter.
                    for j in range(dim):
                        rhat[j] = autocorr_test(chain[:,:k,j])
                    thr = 1./ac_c
                    if verbosity > 0:
                        print('Auto-corr tau/N = {}'.format(rhat))
                        print('tau/N <= {} = {} \n'.format(thr, rhat<thr))
                    if (rhat <= thr).all():
                        ac_count+=1
                        if verbosity > 0:
                            msg = "Auto-correlation test passed for all params!"
                            msg+= "{}/{}".format(ac_count,ac_count_thr)
                            print(msg)
                        if ac_count >= ac_count_thr:
                            msg='\n ... ==> convergence reached'
                            print(msg)
                            stop = k
                    else:
                        ac_count = 0
                else:
                    raise ValueError('conv_test value not recognized')
                
                                
                if save and verbosity == 3:
                    ac_time = []
                    for j in range(dim):
                        ac_time.append(k*autocorr_test(chain[:,:k,j]))
                    fname = '{d}/{f}_temp_k{k}'.format(d=output_dir,
                                                       f=output_file, k=k)
                    data = {'chain': sampler.chain,
                            'lnprob': sampler.lnprobability,
                            'ac_time': ac_time,
                             'AR': ar_frac}
                    with open(fname, 'wb') as fileSave:
                        pickle.dump(data, fileSave)

        # We have reached convergence
        if k+1 >= stop:
            if verbosity > 0:
                print('We break the loop because we have reached convergence')
            break
        # We have reached the maximum number of steps for our Markov chain.
        if k+1 >= nIterations-1:
            # break to avoid a bug related to font type
            break
        
    isamples, ln_proba, ar = spec_chain_zero_truncated(chain, 
                                                       sampler.lnprobability,
                                                       ar_frac)   
    # update units in the chain if needed
    if len(em_grid)>0:
        for key, val in em_lines.items():
            idx_line = labels.index(key)
            if val[1] == 'L':
                R_si = isamples[:,:,idx_R]*con.R_jup.value
                isamples[:,:,idx_line] *= 4*np.pi*R_si**2
            elif val[1] == 'LogL':
                R_si = isamples[:,:,idx_R]*con.R_jup.value
                conv_fac = 4*np.pi*R_si**2/con.L_sun.value
                isamples[:,:,idx_line] = np.log10(isamples[:,:,idx_line]*conv_fac)
      
    if k == nIterations-1:
        if verbosity > 0:
            print("We have reached the limit # of steps without convergence")
            print("You may have to increase niteration_limit")
    # #########################################################################
    # Construction of the independent samples
    # #########################################################################
 

    if save:
        frame = inspect.currentframe()
        args, _, _, values = inspect.getargvalues(frame)
        input_parameters = {j: values[j] for j in args[1:]}
        ac_time = []
        for j in range(dim):
            ac_time.append(isamples.shape[1]*autocorr_test(isamples[:,:,j]))
        output = {'isamples': isamples,
                  #'chain': chain,
                  'input_parameters': input_parameters,
                  'AR': ar,
                  'ac_time': ac_time,
                  'lnprobability': ln_proba}
                  
        with open(output_dir+output_file, 'wb') as fileSave:
            pickle.dump(output, fileSave)
        
        msg = "\nThe file MCMC_results has been stored in the folder {}"
        print(msg.format(output_dir))

    if verbosity > 0:
        timing(start_time)
                                    
    return isamples, ln_proba

                                    
def spec_chain_zero_truncated(chain, ln_proba=None, ar=None):
    """
    Return the Markov chain with the dimension: walkers x steps* x parameters,
    where steps* is the last step before having 0 (not yet constructed chain).
    
    Parameters
    ----------
    chain: numpy.array
        The MCMC chain.
    ln_proba: numpy.array, opt
        Corresponding ln-probabilities.
     
    Returns
    -------
    out: numpy.array
        The truncated MCMC chain, that is to say, the chain which only contains
        relevant information.
    out_ln_proba: numpy.array
        If ln_proba is provided as input, out_ln_proba contains the 
        zero-truncated ln-proba (i.e. matching shape with non-zero samples)
    """
    try:
        idxzero = np.where(chain[0, :, 0] == 0.0)[0][0]
    except:
        idxzero = chain.shape[1]
    res = [chain[:, 0:idxzero, :]]
        
    if ln_proba is not None:
        res.append(ln_proba[:,0:idxzero])
    if ar is not None:
        res.append(ar[:,0:idxzero])
    if len(res) == 1:
        return res[0]
    else:
        return tuple(res)
 
   
def spec_show_walk_plot(chain, labels, save=False, output_dir='', ntrunc=100,
                        **kwargs):
    """
    Display/save a figure showing the path of each walker during the MCMC run.
    
    Parameters
    ----------
    chain: numpy.array
        The Markov chain. The shape of chain must be nwalkers x length x dim.
        If a part of the chain is filled with zero values, the method will
        discard these steps.
    labels: Tuple of strings
        Tuple of labels in the same order as initial_state, that is:
        - first all parameters related to loaded models (e.g. 'Teff', 'logg')
        - next the planet photometric radius 'R',
        - (optionally) the optical extinction 'Av'.
        - (optionally) 'Tbb1', 'Rbb1', 'Tbb2', 'Rbb2', etc. for each extra bb
        contribution
    save: boolean, default: False
        If True, a pdf file is created.
    ntrunc: int, opt
        max number of walkers plotted. If too many walkers the plot will become
        too voluminous and too crowded. Plot will be truncated to only ntrunc 
        first walkers
    kwargs:
        Additional attributes are passed to the matplotlib plot method.
                                                        
    Returns
    -------
    Display the figure or create a pdf file named walk_plot.pdf in the working
    directory.
    
    """
    npar = chain.shape[-1]
    if len(labels) != npar:
        raise ValueError("Labels length should match chain last dimension")
    temp = np.where(chain[0, :, 0] == 0.0)[0]
    if len(temp) != 0:
        chain = chain[:, :temp[0], :]


    color = kwargs.pop('color', 'k')
    alpha = kwargs.pop('alpha', 0.1)
    filename = kwargs.pop('filename','walk_plot.pdf')
    #labels = kwargs.pop('labels', ["$r$", r"$\theta$", "$f$"])
    fig, axes = plt.subplots(npar, 1, sharex=True,
                             figsize=kwargs.pop('figsize', (8, int(2*npar))))
    if npar>1:
        axes[-1].set_xlabel(kwargs.pop('xlabel', 'step number'))
        axes[-1].set_xlim(kwargs.pop('xlim', [0, chain.shape[1]]))
        for j in range(npar):
            axes[j].plot(chain[:ntrunc, :, j].T, color=color, alpha=alpha, **kwargs)
            axes[j].yaxis.set_major_locator(MaxNLocator(5))
            axes[j].set_ylabel(labels[j])
    else:
        axes.set_xlabel(kwargs.pop('xlabel', 'step number'))
        axes.set_xlim(kwargs.pop('xlim', [0, chain.shape[1]]))            
        for j in range(npar):
            axes.plot(chain[:ntrunc, :, j].T, color=color, alpha=alpha, **kwargs)
            axes.yaxis.set_major_locator(MaxNLocator(5))
            axes.set_ylabel(labels[j])
    fig.tight_layout(h_pad=0)
    if save:
        plt.savefig(output_dir+filename)
        plt.close(fig)
    else:
        plt.show()


def spec_show_corner_plot(chain, burnin=0.5, save=False, output_dir='', 
                          mcmc_res=None, units=None, ndig=None, 
                          labels_plot=None, plot_name='corner_plot.pdf', **kwargs):
    """
    Display/save a figure showing the corner plot (pdfs + correlation plots).
    
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
    mcmc_res: numpy array OR tuple of 2 dictionaries/np.array, opt
        Values to be printed on top of each 1d posterior distribution   
        - if numpy array: 
            - npar x 3 dimensions (where npar is the number of parameters), 
            containing the most likely value of each parameter and the lower 
            and upper uncertainties at the desired quantiles, resp. 
            - npar x 2 dimensions: same as above but with a single value of  
            uncertainty. E.g. output of spec_confidence() for a gaussian fit
        - if tuple of 2 dictionaries: output of spec_confidence without 
                                      gaussian fit
        - if tuple of 2 np.array: output of spec_confidence() with gaussian fit
    units: tuple, opt
        Tuple of strings containing units for each parameter. If provided,
        mcmc_res will be printed on top of each 1d posterior distribution along 
        with these units.
    ndig: tuple, opt
        Number of digits precision for each printed parameter
    labels_plot: tuple, opt
        Labels corresponding to parameter names, used for the plot. If None,
        will use "labels" passed in kwargs.
    kwargs:
        Additional attributes passed to the corner.corner() method.
        (e.g. 'labels', 'show_title')
                    
    Returns
    -------
    Display the figure or create a pdf file named walk_plot.pdf in the working
    directory.
        
    Raises
    ------
    ImportError
    
    """
    
    npar = chain.shape[-1]
    if "labels" in kwargs.keys():
        labels = kwargs.pop('labels')
        if labels_plot is None:
            labels_plot = labels
    else:
        labels = None
        
    # allows for different title/units on top of 1D posterior distributions
    labels_tit = kwargs.pop('labels_tit', labels_plot)
    labels_tit_unit = kwargs.pop('labels_tit_unit', units)
    
    try:
        temp = np.where(chain[0, :, 0] == 0.0)[0]
        if len(temp) != 0:
            chain = chain[:, :temp[0], :]
        length = chain.shape[1]
        indburn = int(np.floor(burnin*(length-1)))
        chain = chain[:, indburn:length, :].reshape((-1, npar))
    except IndexError:
        pass

    if chain.shape[0] == 0:
        raise ValueError("It seems the chain is empty. Have you run the MCMC?")
    else:
        fig = corner.corner(chain, labels=labels_plot, **kwargs)

        
    if mcmc_res is not None and labels is not None:
        title_kwargs = kwargs.pop('title_kwargs', None)
        if isinstance(mcmc_res,tuple):
            if len(mcmc_res) != 2:
                msg = "mcmc_res should have 2 elements"
                raise TypeError(msg)
            if len(mcmc_res[0]) != npar or len(mcmc_res[1]) != npar:
                msg = "dict should have as many entries as there are params"
                raise TypeError(msg)
            if labels is None:
                msg = "labels should be provided"
                raise TypeError(msg)                
        elif isinstance(mcmc_res,np.ndarray):
            if mcmc_res.shape[0] != npar:
                msg = "mcmc_res first dim should be equal to number of params"
                raise TypeError(msg)  
        else:
            msg = "type of mcmc_res not recognised"
            raise TypeError(msg)
        # Extract the axes
        axes = np.array(fig.axes).reshape((npar, npar))
                
        # Loop over the diagonal
        for i in range(npar):
            ax = axes[i, i]
            title = None
            if isinstance(mcmc_res,tuple):
                if isinstance(mcmc_res[0],dict):
                    q_50 = mcmc_res[0][labels[i]]
                    q_m = mcmc_res[1][labels[i]][0]
                    q_p = mcmc_res[1][labels[i]][1]
                else:
                    q_50 = mcmc_res[0][i][0]
                    q_m = mcmc_res[1][i][0]
                    q_p = q_m
            else:
                q_50 = mcmc_res[i,0]
                q_m = mcmc_res[i,1]
                q_p = mcmc_res[i,2]

            # Format the quantile display.
            try:
                fmt = "{{:.{0}f}}".format(ndig[i]).format
            except:
                fmt = "{{:.2f}}".format
            title = r"${{{0}}}_{{{1}}}^{{+{2}}}$"
            title = title.format(fmt(q_50), fmt(q_m), fmt(q_p))

            # Add in the column name if it's given.
            try:
                title = "{0} = {1} {2}".format(labels_tit[i], title, 
                                               labels_tit_unit[i])
            except:
                title = "{0} = {1}".format(labels_tit[i], title)
                
            ax.set_title(title, **title_kwargs)
    
    if save:
        plt.savefig(output_dir+plot_name)
        plt.close(fig)
    else:
        plt.show()


def spec_confidence(isamples, labels, cfd=68.27, bins=100, gaussian_fit=False, 
                    weights=None, verbose=True, save=False, output_dir='', 
                    bounds=None, priors=None, **kwargs):
    """
    Determine the highly probable value for each model parameter, as well as
    the 1-sigma confidence interval.
    
    Parameters
    ----------
    isamples: numpy.array
        The independent samples for each model parameter.
    labels: Tuple of strings
        Tuple of labels in the same order as initial_state, that is:
        - first all parameters related to loaded models (e.g. 'Teff', 'logg')
        - next the planet photometric radius 'R',
        - (optionally) the optical extinction 'Av'.
        - (optionally) 'Tbb1', 'Rbb1', 'Tbb2', 'Rbb2', etc. for each extra bb
        contribution
    cfd: float, optional
        The confidence level given in percentage.
    bins: int, optional
        The number of bins used to sample the posterior distributions.
    gaussian_fit: boolean, optional
        If True, a gaussian fit is performed in order to determine (\mu,\sigma)
    weights : (n, ) numpy ndarray or None, optional
        An array of weights for each sample.
    verbose: boolean, optional
        Display information in the shell.
    save: boolean, optional
        If "True", a txt file with the results is saved in the output
        repository.
    bounds: dictionary, opt
        Only used if a text file is saved summarizing results+bounds+priors.
        Should be the same bounds as provided to the MCMC.
    priors: dictionary, opt
        Only used if a text file is saved summarizing results+bounds+priors.
        Should be the same priors as provided to the MCMC.
    kwargs: optional
        Additional attributes are passed to the matplotlib hist() method.
        
    Returns
    -------
    out: tuple
        A 2 elements tuple with the highly probable solution and the confidence
        interval.
        
    """
    title = kwargs.pop('title', None)
        
    output_file = kwargs.pop('filename', 'confidence.txt')
    
    if gaussian_fit:
        output_pdf = kwargs.pop('pdfname','confi_hist_spec_params_gaussfit.pdf')
    else:
        output_pdf = kwargs.pop('pdfname','confi_hist_spec_params.pdf')
        
    try:
        l = isamples.shape[1]
    except:
        l = 1
     
    confidenceInterval = {}
    val_max = {}
    pKey = labels #['r', 'theta', 'f']
    
    if l != len(pKey):
        raise ValueError("Labels length should match chain last dimension")
    
    if cfd == 100:
        cfd = 99.9
        
    #########################################
    ##  Determine the confidence interval  ##
    #########################################
    if gaussian_fit:
        mu = np.zeros(l)
        sigma = np.zeros_like(mu)
    
    if gaussian_fit:
        fig, ax = plt.subplots(2, l, figsize=(int(4*l),8))
    else:
        fig, ax = plt.subplots(1, l, figsize=(int(4*l),4))
    
    for j in range(l):
        
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
            test = test + bins_width*n[int(jj)]
            pourcentage = test/surface_total*100
            if pourcentage > cfd:
                if verbose:
                    msg = 'percentage for {}: {}%'
                    print(msg.format(pKey[j], pourcentage))
                break
        n_arg_min = int(n_arg_sort[:k].min())
        n_arg_max = int(n_arg_sort[:k+1].max())
        
        if n_arg_min == 0:
            n_arg_min += 1
        if n_arg_max == bins:
            n_arg_max -= 1
        
        val_max[pKey[j]] = bin_vertices[int(n_arg_sort[0])]+bins_width/2.
        confidenceInterval[pKey[j]] = np.array([bin_vertices[n_arg_min-1],
                                               bin_vertices[n_arg_max+1]]
                                               - val_max[pKey[j]])
                        
        arg = (isamples[:, j] >= bin_vertices[n_arg_min - 1]) * \
              (isamples[:, j] <= bin_vertices[n_arg_max + 1])
        if gaussian_fit:
            ax[0][j].hist(isamples[arg,j], bins=bin_vertices,
                          facecolor='gray', edgecolor='darkgray',
                          histtype='stepfilled', alpha=0.5)
            ax[0][j].vlines(val_max[pKey[j]], 0, n[int(n_arg_sort[0])],
                            linestyles='dashed', color='red')
            ax[0][j].set_xlabel(pKey[j])
            if j == 0:
                ax[0][j].set_ylabel('Counts')

            mu[j], sigma[j] = norm.fit(isamples[:, j])
            n_fit, bins_fit = np.histogram(isamples[:, j], bins, normed=1,
                                           weights=weights)
            ax[1][j].hist(isamples[:, j], bins, density=1, weights=weights,
                          facecolor='gray', edgecolor='darkgray',
                          histtype='step')
            y = norm.pdf(bins_fit, mu[j], sigma[j])
            ax[1][j].plot(bins_fit, y, 'r--', linewidth=2, alpha=0.7)

            ax[1][j].set_xlabel(pKey[j])
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
            ax[j].vlines(val_max[pKey[j]], 0, n[int(n_arg_sort[0])],
                         linestyles='dashed', color='red')
            ax[j].set_xlabel(pKey[j])
            if j == 0:
                ax[j].set_ylabel('Counts')

            if title is not None:
                msg = r"{} - {:.3f} {:.3f} +{:.3f}"
                ax[1].set_title(msg.format(title, val_max[pKey[j]], 
                                           confidenceInterval[pKey[j]][0], 
                                           confidenceInterval[pKey[j]][1]),
                                fontsize=10)

        plt.tight_layout(w_pad=0.1)

    if save:
        plt.savefig(output_dir+output_pdf)


    if verbose:
        for j in range(l):
            print("******* Results for {} ***** ".format(pKey[j]))
            print('\n\nConfidence intervals:')
            print('{}: {} [{},{}]'.format(pKey[j], val_max[pKey[j]],
                                          confidenceInterval[pKey[j]][0],
                                          confidenceInterval[pKey[j]][1]))
            if gaussian_fit:
                print('Gaussian fit results:')
                print('{}: {} +-{}'.format(pKey[j], mu[j], sigma[j]))

    ##############################################
    ##  Write inference results in a text file  ##
    ##############################################
    if save:
        with open(output_dir+output_file, "w") as f:
            f.write('######################################\n')
            f.write('####   MCMC results (confidence)   ###\n')
            f.write('######################################\n')
            f.write(' \n')
            f.write('Results of the MCMC fit\n')
            f.write('----------------------- \n')
            f.write(' \n')
            if bounds is not None:
                f.write('Bounds\n')
                f.write('------')
                for j in range(l):
                    f.write('\n{}: [{},{}]'.format(pKey[j], bounds[pKey[j]][0],
                                                   bounds[pKey[j]][1]))
                f.write(' \n')
                f.write(' \n')
            f.write('Priors\n')
            f.write('------')
            if priors is not None:
                for key, prior in priors.items():
                    f.write('\n{}: {}+-{}'.format(key, prior[0], prior[1]))
            else:
                f.write('\n None')
            f.write(' \n')
            f.write(' \n')
            f.write('>> Spectral parameters for a ')
            f.write('{} % confidence interval:\n'.format(cfd))
            
            for j in range(l):
                f.write('{}: {} [{},{}]'.format(pKey[j], val_max[pKey[j]],
                                              confidenceInterval[pKey[j]][0],
                                              confidenceInterval[pKey[j]][1]))
                if gaussian_fit:
                    f.write('Gaussian fit results:')
                    f.write('{}: {} +-{}'.format(pKey[j], mu[j], sigma[j]))

    if gaussian_fit:
        return mu, sigma
    else:
        return val_max, confidenceInterval
