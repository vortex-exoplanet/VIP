#! /usr/bin/env python

"""
Functions useful for spectral fitting of companions, and model interpolation.
"""

__author__ = 'Valentin Christiaens'
__all__ = ['make_model_from_params',
           'make_resampled_models',
           'resample_model',
           'interpolate_model']

import numpy as np
import astropy.constants as con
from astropy.convolution import Gaussian1DKernel, convolve_fft
from astropy.stats import gaussian_fwhm_to_sigma
import itertools
from scipy.interpolate import InterpolatedUnivariateSpline
import pandas as pd
import pdb
from scipy.ndimage import map_coordinates
from ..fits import open_fits
from .utils_spec import (convert_F_units, blackbody, find_nearest, extinction,
                         inject_em_line)


def make_model_from_params(params, labels, grid_param_list, dist, lbda_obs=None, 
                           model_grid=None, model_reader=None, em_lines={}, 
                           em_grid={}, dlbda_obs=None, instru_fwhm=None, 
                           instru_idx=None, filter_reader=None, AV_bef_bb=False,
                           units_obs='si', units_mod='si', interp_order=1):
    """
    Routine to make the model from input parameters.
    
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
    dist :  float
        Distance in parsec, used for flux scaling of the models.
    lbda_obs : numpy 1d ndarray or list, opt
        Wavelength of observed spectrum. If provided, the model spectrum will 
        be resampled to match lbda_obs. If several instruments, should be 
        ordered per instrument, not necessarily as monotonically increasing 
        wavelength. Hereafter, n_ch = len(lbda_obs). 
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
        contains model values. See example routine in interpolate_model() 
        description.
    em_lines: dictionary, opt
        Dictionary of emission lines to be added on top of the model spectrum.
        Each dict entry should be the name of the line, assigned to a tuple of
        3 values: 1) the wavelength (in mu), 2) a string indicating whether line 
        intensity is expressed in flux ('F'), luminosity ('L') or log(L/LSun) 
        ("LogL"), and 3) the latter quantity. The intensity of the emission 
        lines can be sampled by MCMC, in that case the last element of the 
        tuple can be set to None. If not to be sampled, a value for the 
        intensity should be provided (in the same system of units as the model 
        spectra, multiplied by mu). Examples:
        em_lines = {'BrG':(2.1667,'F',263)};
        em_lines = {'BrG':(2.1667,'LogL',-5.1)}
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
    out: numpy array
        The model wavelength and spectrum
    """
    
    
    if 'Tbb1' in labels and grid_param_list is None and lbda_obs is None:
        raise ValueError("lbda_obs should be provided because there is no grid")
      
    if grid_param_list is None:
        lbda_mod = lbda_obs
        spec_mod = np.zeros_like(lbda_obs)
    else:
        npar_grid = len(grid_param_list)
        params_grid = [params[i] for i in range(npar_grid)]
        params_grid = tuple(params_grid)        
        if len(em_grid) == 0:
            p_em_grid = None
        else:
            p_em_grid = {}
            for key, _ in em_grid.items():
                j = labels.index(key)
                p_em_grid[key] = params[j]
        # interpolate model to requested parameters
        lbda_mod, spec_mod = interpolate_model(params_grid, grid_param_list, 
                                               p_em_grid, em_grid, em_lines,
                                               labels, model_grid, 
                                               model_reader, interp_order)
        
        # resample to lbda_obs if needed
        if lbda_obs is not None:
            cond = False
            if len(lbda_obs) != len(lbda_mod):
                cond = True
            elif not np.allclose(lbda_obs, lbda_mod):
                cond = True
            if cond:
                lbda_mod, spec_mod = resample_model(lbda_obs, lbda_mod, spec_mod, 
                                                    dlbda_obs, instru_fwhm, 
                                                    instru_idx, filter_reader)
            
        # convert model to same units as observed spectrum if necessary
        if units_mod != units_obs:
            spec_mod = convert_F_units(spec_mod, lbda_mod, in_unit=units_mod, 
                                       out_unit=units_obs)

        # scale by (R/dist)**2
        idx_R = labels.index("R")
        dilut_fac = ((params[idx_R]*con.R_jup.value)/(dist*con.pc.value))**2
        spec_mod *= dilut_fac

        
    # apply extinction if requested
    if 'Av' in labels and AV_bef_bb:
        ## so far only assume Cardelli extinction law
        idx_AV = labels.index("Av")
        if 'Rv' in labels:
            idx_RV = labels.index("Rv")
            RV = params[idx_RV]
        else:
            RV=3.1
        extinc_curve = extinction(lbda_mod, params[idx_AV], RV)
        flux_ratio_ext = np.power(10.,-extinc_curve/2.5)
        spec_mod *= flux_ratio_ext
        ## TBD: add more options

    # add n blackbody component(s) if requested 
    if 'Tbb1' in labels:
         n_bb = 0
         for label in labels:
             if 'Tbb' in label:
                 n_bb+=1
         idx_Tbb1 = labels.index("Tbb1")
         Rj = con.R_jup.value
         pc = con.pc.value
         for ii in range(n_bb):
             idx = ii*2
             Omega = np.pi*((params[idx_Tbb1+idx+1]*Rj)/(dist*pc))**2
             bb = Omega*blackbody(lbda_mod, params[idx_Tbb1+idx])
             spec_mod += bb

    # apply extinction if requested
    if 'Av' in labels and not AV_bef_bb:
        ## so far only assume Cardelli extinction law
        idx_AV = labels.index("Av")
        if 'Rv' in labels:
            idx_RV = labels.index("Rv")
            RV = params[idx_RV]
        else:
            RV=3.1
        extinc_curve = extinction(lbda_mod, params[idx_AV], RV)
        flux_ratio_ext = np.power(10.,-extinc_curve/2.5)
        spec_mod *= flux_ratio_ext
        ## TBD: add more options                
        
    return lbda_mod, spec_mod


def make_resampled_models(lbda_obs, grid_param_list, model_grid=None,
                          model_reader=None, em_lines={}, em_grid=None, 
                          dlbda_obs=None, instru_fwhm=None, instru_idx=None, 
                          filter_reader=None, interp_nonexist=True):
    """
    Returns a cube of models after convolution and resampling as in the 
    observations.
    
    Parameters:
    -----------   
    lbda_obs : numpy 1d ndarray or list
        Wavelength of observed spectrum. If several instruments, should be 
        ordered per instrument, not necessarily as monotonically increasing 
        wavelength. Hereafter, n_ch = len(lbda_obs).
    grid_param_list : list of 1d numpy arrays/lists
        Should contain list/numpy 1d arrays with available grid of model 
        parameters. Note: model grids shouldn't contain grids on radius and Av.
    model_grid: list of 1d numpy arrays, or list of lists.
        Available grid of model parameters (should only contain the parameter
        values, not the models themselves). The latter will be loaded. 
        Important: 1) Make sure the bounds are within the model grid to avoid 
        extrapolation. 2) All keywords that are neither 'R', 'Av' nor 'M' will 
        be considered model grid parameters.
        length of params, with the same order. OR '1bb' or '2bb' for black-body
        models. In that case the model will be created on the fly at each 
        iteration using 1 or 2 Planck functions respectively. There are 2 params 
        for each Planck function: Teff and radius.
    model_reader : python routine
        External routine that reads a model file, converts it to required 
        units and returns a 2D numpy array, where the first column corresponds
        to wavelengths, and the second contains model values. Example below.
    em_lines: dictionary, opt
        Dictionary of emission lines to be added on top of the model spectrum.
        Each dict entry should be the name of the line, assigned to a tuple of
        3 values: 1) the wavelength (in mu), 2) a string indicating whether line 
        intensity is expressed in flux ('F'), luminosity ('L') or log(L/LSun) 
        ("LogL"), and 3) the latter quantity. The intensity of the emission 
        lines can be sampled by MCMC, in that case the last element of the 
        tuple can be set to None. If not to be sampled, a value for the 
        intensity should be provided (in the same system of units as the model 
        spectra, multiplied by mu). Example:
        em_lines = {'BrG':(2.1667,'F',263)}
    em_grid: dictionary pointing to lists, opt
        Dictionary where each entry corresponds to an emission line and points
        to a list of values to inject for emission line fluxes. For computation 
        efficiency, interpolation will be performed between the points of this 
        grid during the MCMC sampling. Dict entries should match labels and 
        em_lines. Note: length of this dictionary can be different of em_lines;
        i.e. if a line is in em_lines but not in em_grid, it will not be 
        considered an MCMC parameter.
    lbda_mod : numpy 1d ndarray or list
        Wavelength of tested model. Should have a wider wavelength extent than 
        the observed spectrum.
    spec_mod : numpy 1d ndarray
        Model spectrum. It does not require the same wavelength sampling as the
        observed spectrum. If higher spectral resolution, it will be convolved
        with the instrumental spectral psf (if instru_fwhm is provided) and 
        then binned to the same sampling. If lower spectral resolution, a 
        linear interpolation is performed to infer the value at the observed 
        spectrum wavelength sampling.
    dlbda_obs: numpy 1d ndarray or list, optional
        Spectral channel width for the observed spectrum. It should be provided 
        IF one wants to weigh each point based on the spectral 
        resolution of the respective instruments (as in Olofsson et al. 2016).
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
    interp_nonexist: bool, opt
        Whether to interpolate if models do not exist, based on closest model(s)
        
    Returns
    -------
    resamp_mod: 1d numpy array
        Grid of model spectra resampled at wavelengths matching the observed 
        spectrum.
    """
    n_params = len(grid_param_list)
    n_mods = len(grid_param_list[0])
    dims = [len(grid_param_list[0])]
    
    if n_params>1:
        for pp in range(1,n_params):
            n_mods *= len(grid_param_list[pp])
            dims.append(len(grid_param_list[pp]))

    if em_grid is None:
        n_em = 0
        final_dims = dims+[len(lbda_obs)]+[2]
    else:
        n_em = len(em_grid)
        n_em_mods = 1
        dims_em = []
        for key, _ in em_grid.items():
            n_em_mods *= len(em_grid[key])
            dims_em.append(len(em_grid[key]))
        final_dims = dims+dims_em+[len(lbda_obs)]+[2]
        dims_em = tuple(dims_em)
     
    final_dims = tuple(final_dims)
    dims = tuple(dims)
    resamp_mod = []
    
    # Loop on all models whose parameters are provided in model grid
    for nn in range(n_mods):            
        if model_grid is not None:
            indices = []
            idx = np.unravel_index(nn,dims)
            for pp in range(n_params):
                indices.append(idx[pp])
            indices = tuple(indices)
            tmp = model_grid[indices]
            lbda_mod = tmp[:,0]
            spec_mod = tmp[:,1]             
        else:
            params_tmp = []
            idx = np.unravel_index(nn,dims)
            for pp in range(n_params):
                params_tmp.append(grid_param_list[pp][idx[pp]])  
            try:
                lbda_mod, spec_mod = model_reader(params_tmp)
                if np.sum(np.isnan(spec_mod))>0:
                    print("There are nan values in spec for params: ")
                    pdb.set_trace()
            except:
                msg= "Model does not exist for param combination ({})"
                print(msg.format(params_tmp))
                print("Press c if you wish to interpolate that model from neighbours")
                pdb.set_trace()
                if interp_nonexist:
                    # find for which dimension the model doesn't exist;
                    for qq in range(n_params):
                        interp_params1=[]
                        interp_params2=[]
                        for pp in range(n_params):
                            if pp == qq:
                                try:
                                    interp_params1.append(grid_param_list[pp][idx[pp]-1])
                                    interp_params2.append(grid_param_list[pp][idx[pp]+1])
                                except:
                                    continue
                            else:
                                interp_params1.append(grid_param_list[pp][idx[pp]])
                                interp_params2.append(grid_param_list[pp][idx[pp]])
                        try:
                            lbda_mod1, spec_mod1 = model_reader(interp_params1)
                            lbda_mod2, spec_mod2 = model_reader(interp_params2)
                            lbda_mod = np.mean([lbda_mod1,lbda_mod2],axis=0)
                            spec_mod = np.mean([spec_mod1,spec_mod2],axis=0)
                            msg= "Model was interpolated based on models: {} "
                            msg+="and {}"
                            print(msg.format(interp_params1,interp_params2))
                            break
                        except:
                            pass
                        if qq == n_params-1:
                            msg = "Impossible to interpolate model!" 
                            msg += "Consider reducing bounds."
                            raise ValueError(msg)
                else:
                    msg = "Model interpolation not allowed for non existing " 
                    msg += "models in the grid."
                    raise ValueError(msg)                    
                    
        # inject emission lines if any
        if n_em > 0:
            flux_grids = []
            wls = []
            widths = []
            for key, flux_grid in em_grid.items():
                flux_grids.append(flux_grid)
                wls.append(em_lines[key][0])
                widths.append(em_lines[key][2])
            # recursively inject em lines
            for fluxes in itertools.product(*flux_grids):
                for ff, flux in enumerate(fluxes):
                    spec_mod = inject_em_line(wls[ff], flux, lbda_mod, spec_mod,
                                              widths[ff])                
                # interpolate OR convolve+bin model spectrum if required
                if len(lbda_obs) != len(lbda_mod):
                    res = resample_model(lbda_obs, lbda_mod, spec_mod, 
                                         dlbda_obs, instru_fwhm, instru_idx, 
                                         filter_reader)
                elif not np.allclose(lbda_obs, lbda_mod):
                    res = resample_model(lbda_obs, lbda_mod, spec_mod, 
                                         dlbda_obs, instru_fwhm, instru_idx, 
                                         filter_reader)
                else:
                    res = np.array([lbda_obs, spec_mod])
                    
                resamp_mod.append(res)                
        else:
                    
            # interpolate OR convolve+bin model spectrum if not same sampling
            if len(lbda_obs) != len(lbda_mod):
                res = resample_model(lbda_obs, lbda_mod, spec_mod, dlbda_obs, 
                                     instru_fwhm, instru_idx, filter_reader)
            elif not np.allclose(lbda_obs, lbda_mod):
                res = resample_model(lbda_obs, lbda_mod, spec_mod, dlbda_obs, 
                                     instru_fwhm, instru_idx, filter_reader)
            else:
                res = np.array([lbda_obs, spec_mod])
                
            resamp_mod.append(res)

    resamp_mod = np.array(resamp_mod)
    resamp_mod = np.swapaxes(resamp_mod,-1,-2)
       
    return resamp_mod.reshape(final_dims)

 
def resample_model(lbda_obs, lbda_mod, spec_mod, dlbda_obs=None, 
                   instru_fwhm=None, instru_idx=None, filter_reader=None,
                   no_constraint=False, verbose=False):
    """
    Convolve, interpolate and resample a model spectrum to match observed 
    spectrum.

    Parameters:
    -----------
    lbda_obs : numpy 1d ndarray or list
        Wavelength of observed spectrum. If several instruments, should be 
        ordered per instrument, not necessarily as monotonically increasing 
        wavelength. Hereafter, n_ch = len(lbda_obs).
    lbda_mod : numpy 1d ndarray or list
        Wavelength of tested model. Should have a wider wavelength extent than 
        the observed spectrum.
    spec_mod : numpy 1d ndarray
        Model spectrum. It does not require the same wavelength sampling as the
        observed spectrum. If higher spectral resolution, it will be convolved
        with the instrumental spectral psf (if instru_fwhm is provided) and 
        then binned to the same sampling. If lower spectral resolution, a 
        linear interpolation is performed to infer the value at the observed 
        spectrum wavelength sampling.
    dlbda_obs: numpy 1d ndarray or list, optional
        Spectral channel width for the observed spectrum. It should be provided 
        IF one wants to weigh each point based on the spectral 
        resolution of the respective instruments (as in Olofsson et al. 2016).
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
    no_constraint: bool, optional
        If set to True, will not use 'floor' and 'ceil' constraints when
        cropping the model wavelength ranges, i.e. faces the risk of 
        extrapolation. May be useful, if the bounds of the wavelength ranges 
        are known to match exactly. 
    verbose: bool, optional
        Whether to print more information during resampling.
        
    Returns
    -------
    lbda_obs, spec_mod_res: 2x 1d numpy array
        Observed lambdas, and resampled model spectrum (at those lambdas)
    
    """
    
    def _default_file_reader(filter_name):
        """
        Default file reader if no filter file reader is provided.
        """
        filt_table = pd.read_csv(filter_name, sep=' ', header=0, 
                                 skipinitialspace=True)
        keys = filt_table.keys()
        lbda_filt = np.array(filt_table[keys[0]])
        if '(AA)' in keys[0]:
            lbda_filt /=10000
        elif '(mu)' in keys[0]:
            pass
        elif '(nm)' in keys[0]:
            lbda_filt /=10000
        else:
            raise ValueError('Wavelength unit not recognised in filter file')
        trans = np.array(filt_table[keys[1]])
        return lbda_filt, trans      
                    
    n_ch = len(lbda_obs)
    spec_mod_res = np.zeros_like(lbda_obs)
    
    if dlbda_obs is None:
        # this is only to trim out useless WL ranges, hence significantly 
        # improving speed for large (>1M pts) models (e.g. BT-SETTL).
        # 0.3 factor to consider possible broad-band filters.
        dlbda_obs1 = [min(0.3*lbda_obs[0],lbda_obs[1]-lbda_obs[0])]
        dlbda_obs2 = [(lbda_obs[i+2]-lbda_obs[i])/2 for i in range(n_ch-2)]
        dlbda_obs3 = [min(0.3*lbda_obs[-1],lbda_obs[-1]-lbda_obs[-2])]
        dlbda_obs = np.array(dlbda_obs1+dlbda_obs2+dlbda_obs3)

    if verbose:
        print("checking whether WL samplings are the same for obs and model")        
        
    if isinstance(instru_fwhm, float) or isinstance(instru_fwhm, int):
        instru_fwhm = [instru_fwhm]
    cond = False
    if len(lbda_obs) != len(lbda_mod):
        cond = True
    elif not np.allclose(lbda_obs, lbda_mod):
        cond = True
    if cond:
        lbda_min = lbda_obs[0]-dlbda_obs[0]
        lbda_max = lbda_obs[-1]+dlbda_obs[-1]
        try:
            lbda_min_tmp = min(lbda_min,lbda_obs[0]-3*np.amax(instru_fwhm)/1000)
            lbda_max_tmp = max(lbda_max,lbda_obs[-1]+3*np.amax(instru_fwhm)/1000)
            if lbda_min_tmp > 0:
                lbda_min = lbda_min_tmp
                lbda_max = lbda_max_tmp
        except:
            pass
            
        if no_constraint:
            idx_ini = find_nearest(lbda_mod, lbda_min)
            idx_fin = find_nearest(lbda_mod, lbda_max)
        else:
            idx_ini = find_nearest(lbda_mod, lbda_min, 
                                   constraint='floor')
            idx_fin = find_nearest(lbda_mod, lbda_max, 
                                   constraint='ceil')

        lbda_mod = lbda_mod[idx_ini:idx_fin+1]
        spec_mod = spec_mod[idx_ini:idx_fin+1]
        
        
    nmod = lbda_mod.shape[0]
    ## compute the wavelength sampling of the model
    dlbda_mod1 = [lbda_mod[1]-lbda_mod[0]]
    dlbda_mod2 = [(lbda_mod[i+1]-lbda_mod[i-1])/2 for i in range(1,nmod-1)]
    dlbda_mod3 = [lbda_mod[-1]-lbda_mod[-2]]
    dlbda_mod = np.array(dlbda_mod1+dlbda_mod2+dlbda_mod3)


    if verbose:
        print("testing whether observed spectral res could be > than model's ")
        print("(in at least parts of the spectrum)")
    dlbda_obs_min = np.amin(dlbda_obs)
    idx_obs_min = np.argmin(dlbda_obs)
    idx_near = find_nearest(lbda_mod, lbda_obs[idx_obs_min])
    dlbda_mod_tmp = (lbda_mod[idx_near+1]-lbda_mod[idx_near-1])/2
    do_interp = np.zeros(n_ch, dtype='int32')
    
    if dlbda_mod_tmp>dlbda_obs_min and dlbda_obs_min > 0:
        if verbose:
            print("checking where obs spec res is < or > than model's")
        ## check where obs spec res is < or > than model's
        nchunks_i = 0
        for ll in range(n_ch):
            idx_near = find_nearest(lbda_mod, lbda_obs[ll])
            do_interp[ll] = (dlbda_obs[ll] < dlbda_mod[idx_near])
            if ll > 0:
                if do_interp[ll] and not do_interp[ll-1]:
                    nchunks_i+=1
            elif do_interp[ll]:
                nchunks_i=1
            
    ## interpolate model if the observed spectrum has higher resolution 
    ## and is monotonically increasing
    if np.sum(do_interp) and dlbda_obs_min > 0:
        if verbose:
            print("interpolating model where obs spectrum has higher res")    
        idx_0=0
        for nc in range(nchunks_i):
            idx_1 = np.argmax(do_interp[idx_0:])+idx_0
            idx_0 = np.argmin(do_interp[idx_1:])+idx_1
            if idx_0==idx_1:
                idx_0=-1
                if nc != nchunks_i-1:
                    pdb.set_trace() # should not happen
            idx_ini = find_nearest(lbda_mod,lbda_obs[idx_1],
                                   constraint='floor')
            idx_fin = find_nearest(lbda_mod,lbda_obs[idx_0],
                                   constraint='ceil')


            spl = InterpolatedUnivariateSpline(lbda_mod[idx_ini:idx_fin], 
                                               spec_mod[idx_ini:idx_fin],
                                                   k=min(3,idx_fin-idx_ini-1))
            spec_mod_res[idx_1:idx_0] = spl(lbda_obs[idx_1:idx_0])

            
    ## convolve+bin where the model spectrum has higher resolution (most likely)
    if np.sum(do_interp) < n_ch or dlbda_obs_min > 0:
    # Note: if dlbda_obs_min < 0, it means several instruments are used with 
    # overlapping WL ranges. instru_fwhm should be provided!
        if instru_fwhm is None:
            msg = "Warning! No spectral FWHM nor filter file provided"
            msg+= " => binning without convolution"
            print(msg)
            for ll, lbda in enumerate(lbda_obs):
                mid_lbda_f = lbda_obs-dlbda_obs/2.
                mid_lbda_l = lbda_obs+dlbda_obs/2.
                i_f = find_nearest(lbda_mod,
                                   mid_lbda_f[ll])
                i_l = find_nearest(lbda_mod,
                                   mid_lbda_l[ll])
                spec_mod_res[ll] = np.mean(spec_mod[i_f:i_l+1])
        else:
            if verbose:
                print("convolving+binning where model spectrum has higher res")   
            if isinstance(instru_idx, list):
                instru_idx = np.array(instru_idx)
            elif not isinstance(instru_idx, np.ndarray):
                instru_idx = np.array([1]*n_ch)
    
            for i in range(1,len(instru_fwhm)+1):
                if isinstance(instru_fwhm[i-1], (float,int)):
                    ifwhm = instru_fwhm[i-1]/(1000*np.mean(dlbda_mod))
                    gau_ker = Gaussian1DKernel(stddev=ifwhm*gaussian_fwhm_to_sigma)
                    spec_mod_conv = convolve_fft(spec_mod, gau_ker)
                    tmp = np.zeros_like(lbda_obs[np.where(instru_idx==i)])
                    for ll, lbda in enumerate(lbda_obs[np.where(instru_idx==i)]):
                        mid_lbda_f = lbda_obs-dlbda_obs/2.
                        mid_lbda_l = lbda_obs+dlbda_obs/2.
                        i_f = find_nearest(lbda_mod,
                                           mid_lbda_f[np.where(instru_idx==i)][ll])
                        i_l = find_nearest(lbda_mod,
                                           mid_lbda_l[np.where(instru_idx==i)][ll])
                        tmp[ll] = np.mean(spec_mod_conv[i_f:i_l+1])
                    spec_mod_res[np.where(instru_idx==i)] = tmp  
                elif isinstance(instru_fwhm[i-1], str):
                    if filter_reader is not None:
                        lbda_filt, trans = filter_reader(instru_fwhm[i-1])
                    else:
                        lbda_filt, trans = _default_file_reader(instru_fwhm[i-1])
                    idx_ini = find_nearest(lbda_mod, lbda_filt[0], 
                                           constraint='ceil')
                    idx_fin = find_nearest(lbda_mod, lbda_filt[-1], 
                                           constraint='floor')
                    interp_trans = np.interp(lbda_mod[idx_ini:idx_fin], lbda_filt,
                                             trans)
                    num = np.sum(interp_trans*dlbda_mod[idx_ini:idx_fin]*spec_mod[idx_ini:idx_fin])
                    denom = np.sum(interp_trans*dlbda_mod[idx_ini:idx_fin])
                    spec_mod_res[np.where(instru_idx==i)] = num/denom
                else:
                    msg = "instru_fwhm is a {}, while it should be either a"
                    msg+= " scalar or a string"
                    raise TypeError(msg.format(type(instru_fwhm[i-1])))

    return np.array([lbda_obs, spec_mod_res])

    
def interpolate_model(params, grid_param_list, params_em={}, em_grid={}, 
                      em_lines={}, labels=None, model_grid=None, 
                      model_reader=None, interp_order=1, max_dlbda=2e-4,
                      verbose=False):
    """
    Parameters
    ----------
    params : tuple
        Set of models parameters for which the model grid has to be 
        interpolated.
    grid_param_list : list of 1d numpy arrays/lists
        - If list, should contain list/numpy 1d arrays with available grid of 
        model parameters.
        - Note1: model grids should not contain grids on radius and Av, but 
        these should still be passed in initial_state (Av optional).
    params_em : dictionary, opt
        Set of emission line parameters (typically fluxes) for which the model 
        grid has to be interpolated.
    em_grid: dictionary pointing to lists, opt
        Dictionary where each entry corresponds to an emission line and points
        to a list of values to inject for emission line fluxes. For computation 
        efficiency, interpolation will be performed between the points of this 
        grid during the MCMC sampling. Dict entries should match labels and 
        em_lines. Note: length of this dictionary can be different of em_lines;
        i.e. if a line is in em_lines but not in em_grid, it will not be 
        considered an MCMC parameter.
    em_lines: dictionary, opt
        Dictionary of emission lines to be added on top of the model spectrum.
        Each dict entry should be the name of the line, assigned to a tuple of
        3 values: 1) the wavelength (in mu), 2) a string indicating whether line 
        intensity is expressed in flux ('F'), luminosity ('L') or log(L/LSun) 
        ("LogL"), and 3) the latter quantity. The intensity of the emission 
        lines can be sampled by MCMC, in that case the last element of the 
        tuple can be set to None. If not to be sampled, a value for the 
        intensity should be provided (in the same system of units as the model 
        spectra, multiplied by mu). Example:
        em_lines = {'BrG':(2.1667,'F',263)}
    labels: Tuple of strings
        Tuple of labels in the same order as initial_state, that is:
        - first all parameters related to loaded models (e.g. 'Teff', 'logg')
        - next the planet photometric radius 'R', in Jupiter radius
        - (optionally) the flux of emission lines (labels should match those in
        the em_lines dictionary), in units of the model spectrum (times mu)
        - (optionally) the optical extinction 'Av', in mag
        - (optionally) the ratio of total to selective optical extinction 'Rv'
        - (optionally) 'Tbb1', 'Rbb1', 'Tbb2', 'Rbb2', etc. for each extra bb
        contribution.
        Note: only necessary if an emission list dictionary is provided.
    model_grid : numpy N-d array
        If provided, should contain the grid of model spectra for each
        free parameter of the given grid. I.e. for a grid of n_T values of Teff 
        and n_g values of Logg, the numpy array should be n_T x n_g x n_ch x 2, 
        where n_ch is the number of wavelengths for the observed spectrum.
        If provided, takes precedence over filename/file_reader which would 
        open and read models at each step of the MCMC.
    model_reader : python routine
        External routine that reads a model file, converts it to required 
        units and returns a 2D numpy array, where the first column corresponds
        to wavelengths, and the second contains model values. Example below.
    interp_order: int, opt, {0,1} 
        0: nearest neighbour model.
        1: Order 1 spline interpolation.
    max_dlbda: float, opt
        Maximum delta lbda in mu allowed if binning of lbda_model is necessary. 
        This is necessary for grids of models (e.g. BT-SETTL) where the wavelength
        sampling is not the same depending on parameters (e.g. between 4000K
        and 4100K models for BT-SETTL): resampling preserving original 
        resolution is too prohibitive computationally.
    verbose: bool, optional
        Whether to print more information during resampling.
        
    Returns
    -------
    model : 2d numpy array
        Interpolated model for input parameters. First column corresponds
        to wavelengths, and the second contains model values.
        
    Example file_reader:
    -------------------
    def _example_file_reader(params):
        '''This is a minimal example for the file_reader routine to be provided 
        as argument to model_interpolation. The routine should only take as 
        inputs grid parameters, and returns as output: both the wavelengths and 
        model values as a 2D numpy array.
        This example assumes the model is in a fits file, that is already a 2D
        numpy array, where the first column is the wavelength, and 2nd column 
        is the corresponding model values.'''
        
        model = open_fits(filename.format(params[0],params[1]))

        return model       
        
    """

    def _example_file_reader(filename):
        """ This is a minimal example for the file_reader routine to be provided 
        as argument to model_interpolation. The routine should take as input a 
        template filename format with blanks and parameters, and return as output 
        the wavelengths and model values as a 2D numpy array.
        This example assumes the model is in a fits file, that is already a 2D
        numpy array, where the first column is the wavelength, and second column 
        is the corresponding model values.
        """
        model = open_fits(filename.format(params[0],params[1]))

        return model

    def _den_to_bin(denary,ndigits=3):
        """
        Convert denary to binary number, keeping n digits for binary (i.e.
        padding with zeros if necessary)
        """
        binary=""  
        while denary>0:  
          #A left shift in binary means /2  
          binary = str(denary%2) + binary  
          denary = denary//2  
        if len(binary) < ndigits:
            pad = '0'*(ndigits-len(binary))
        else:
            pad=''
        return pad+binary  

    n_params = len(grid_param_list)
    n_em = len(em_grid)
    n_params_tot = n_params+n_em

    if interp_order == 0:
        if model_grid is None:
            params_tmp = np.zeros(n_params)    
            for nn in range(n_params):
                params_tmp[nn] = find_nearest(grid_param_list[nn], 
                                              params[nn], output='value')
            lbda, spec = model_reader(params_tmp)
            if n_em>0:
                for ll in range(len(labels)):
                    if labels[ll] in em_grid.keys():
                        key = labels[ll]
                        spec = inject_em_line(em_lines[key][0], params_em[key], 
                                              lbda, spec, em_lines[key][2])
            return lbda, spec
            
        else:
            idx_tmp = []
            counter = 0
            for nn in range(n_params_tot):
                if nn < n_params:
                    idx_tmp.append(find_nearest(grid_param_list[nn], params[nn],
                                                output='index'))
                else:
                    for ll in range(len(labels)):
                        if labels[counter+ll] in em_grid.keys():
                            key = labels[counter+ll]
                            counter+=1
                            break
                    idx_tmp.append(find_nearest(em_grid[key], params_em[key],
                                                output='index'))                    
            idx_tmp = tuple(idx_tmp)
            tmp = model_grid[idx_tmp]
            return tmp[:,0], tmp[:,1]
    
    
    elif abs(interp_order) == 1:
        # first compute new subgrid "coords" for interpolation
        if verbose:
            print("Computing new coords for interpolation")
        constraints = ['floor','ceil']
        new_coords = np.zeros([n_params_tot,1])
        sub_grid_param = np.zeros([n_params_tot,2])
        counter = 0
        for nn in range(n_params_tot):
            if nn < n_params:
                grid_tmp = grid_param_list[nn]
                params_tmp = params[nn]
            else:
                for ll in range(len(labels)):
                    if labels[counter+ll] in em_grid.keys():
                        key = labels[counter+ll]
                        grid_tmp = em_grid[key] 
                        params_tmp = params_em[key]
                        counter+=1
                        break
            for ii in range(2):
                try:
                    sub_grid_param[nn,ii] = find_nearest(grid_tmp, 
                                                         params_tmp,
                                                         constraint=constraints[ii], 
                                                         output='value')
                except:
                    pdb.set_trace()
                
            num = (params_tmp-sub_grid_param[nn,0])
            denom = (sub_grid_param[nn,1]-sub_grid_param[nn,0])
            new_coords[nn,0] = num/denom
            
        if verbose:
            print("Making sub-grid of models")
        sub_grid = []
        sub_grid_lbda = []        
        if model_grid is None:
            ntot_subgrid = 2**n_params_tot
            for dd in range(ntot_subgrid):
                str_indices = _den_to_bin(dd, n_params_tot)
                params_tmp = []
                for nn in range(n_params):
                    params_tmp.append(sub_grid_param[nn,int(str_indices[nn])])
                params_tmp=np.array(params_tmp)
                lbda, spec = model_reader(params_tmp)
                if n_em>0:
                    for nn in range(len(labels)):
                        if labels[nn] in em_grid.keys():
                            key = labels[nn]
                            spec = inject_em_line(em_lines[key][0], 
                                                  params_em[key], lbda, 
                                                  spec, em_lines[key][2])
                sub_grid.append(spec)
                sub_grid_lbda.append(lbda)
            
            # resample to match sparser sampling if required
            nch = np.amin([len(sub_grid_lbda[i]) for i in range(ntot_subgrid)])
            nch_max = np.amax([len(sub_grid_lbda[i]) for i in range(ntot_subgrid)])
            if nch_max != nch:
                min_i = np.argmin([len(sub_grid_lbda[i]) for i in range(ntot_subgrid)])
                min_dlbda = np.amin(sub_grid_lbda[min_i][1:]-sub_grid_lbda[min_i][:-1])
                if min_dlbda < max_dlbda:
                    bin_fac = int(max_dlbda/min_dlbda)
                    if verbose:
                        msg = "Models will be binned in WL by a factor {} to "
                        msg += "min dlbda = {}mu"
                        print(msg.format(bin_fac,max_dlbda))
                    
                    nch = int(len(sub_grid_lbda[min_i])/bin_fac)
                    tmp_spec = []
                    tmp_lbda = []
                    for bb in range(nch):
                        idx_ini = bb*bin_fac
                        idx_fin = (bb+1)*bin_fac
                        tmp_spec.append(np.mean(sub_grid[min_i][idx_ini:idx_fin]))
                        tmp_lbda.append(np.mean(sub_grid_lbda[min_i][idx_ini:idx_fin]))
                    sub_grid[min_i] = np.array(tmp_spec)
                    sub_grid_lbda[min_i] = np.array(tmp_lbda)
                for dd in range(ntot_subgrid):
                    cond = False
                    if len(sub_grid_lbda[dd]) != nch:
                        cond = True
                    else:
                        # np.allclose() or np.array_equal are TOO slow
                        dlbda = sub_grid_lbda[min_i][-1]-sub_grid_lbda[min_i][0]
                        dlbda/=nch
                        if np.sum(sub_grid_lbda[dd]-sub_grid_lbda[min_i])>dlbda:
                            cond = True
                    if cond:
                        if verbose:
                            msg = "Resampling model of different WL sampling. "
                            msg+= "This may take a while for high-res/large WL"
                            msg+= " ranges..."
                            print(msg)
                        res = resample_model(sub_grid_lbda[min_i], 
                                             sub_grid_lbda[dd], 
                                             sub_grid[dd],
                                             no_constraint=True,
                                             verbose=verbose)
                        sub_grid_lbda[dd], sub_grid[dd] = res

            # Create array with dimensions 'dims' for each wavelength
            final_dims = tuple([nch]+[2]*n_params_tot)
            sub_grid = np.array(sub_grid)
            sub_grid_lbda = np.array(sub_grid_lbda)
            sub_grid = np.swapaxes(sub_grid,0,1)            
            sub_grid_lbda = np.swapaxes(sub_grid_lbda,0,1)
            sub_grid = sub_grid.reshape(final_dims)
            sub_grid_lbda = sub_grid_lbda.reshape(final_dims)
        
        else:
            constraints = ['floor','ceil']
            sub_grid_idx = np.zeros([n_params_tot,2], dtype=np.int32)
            #list_idx = []
            counter = 0
            for nn in range(n_params_tot):
                if nn < n_params:
                    grid_tmp = grid_param_list[nn]
                    params_tmp = params[nn]
                else:
                    for ll in range(len(labels)):
                        if labels[counter+ll] in em_grid.keys():
                            key = labels[counter+ll]
                            grid_tmp = em_grid[key] 
                            params_tmp = params_em[key]
                            counter+=1
                            break
                for ii in range(2):
                    sub_grid_idx[nn,ii]=find_nearest(grid_tmp, params_tmp,
                                                     constraint=constraints[ii], 
                                                     output='index')
            for dd in range(2**n_params_tot):
                str_indices = _den_to_bin(dd, n_params_tot)
                idx_tmp = []
                for nn in range(n_params_tot):
                    idx_tmp.append(sub_grid_idx[nn,int(str_indices[nn])])
                    #idx_tmp = sub_grid_idx[nn,int(str_indices[nn])]
                #list_idx.append(idx_tmp)
                #list_idx=np.array(list_idx)
                sub_grid.append(model_grid[tuple(idx_tmp)])
            
            # first reshape
            sub_grid = np.array(sub_grid)
            dims = tuple([2]*n_params_tot+[sub_grid.shape[-2]]+[sub_grid.shape[-1]])
            sub_grid = sub_grid.reshape(dims)
            sub_grid = np.moveaxis(sub_grid,-1,0) # make last dim (lbda vs flux) come first
            sub_grid_lbda = sub_grid[0]
            sub_grid = sub_grid[1]
            # move again axis to have nch as first axis
            sub_grid = np.moveaxis(sub_grid,-1,0)
            sub_grid_lbda = np.moveaxis(sub_grid_lbda,-1,0)
            nch = sub_grid.shape[0]
            
        interp_model = np.zeros(nch)
        interp_lbdas = np.zeros(nch)
        
        if interp_order == -1:
            sub_grid = np.log10(sub_grid)
        
        for cc in range(nch):
            interp_model[cc] = map_coordinates(sub_grid[cc], new_coords, 
                                               order=abs(interp_order))
            interp_lbdas[cc] = map_coordinates(sub_grid_lbda[cc], new_coords, 
                                               order=abs(interp_order))
            
        if interp_order == -1:
            interp_model = np.power(10,interp_model)
            
        return interp_lbdas, interp_model
    
    else:
        msg = "Interpolation order not allowed. Only -1, 0 or 1 accepted"
        raise TypeError(msg)