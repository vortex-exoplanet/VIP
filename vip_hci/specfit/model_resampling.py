#! /usr/bin/env python

"""
Functions useful for spectral fitting of companions, and model interpolation.
"""

__author__ = 'Valentin Christiaens'
__all__ = ['make_resampled_models',
           'resample_model',
           'model_interpolation']

import numpy as np
from astropy.convolution import Gaussian1DKernel, convolve_fft
from astropy.stats import gaussian_fwhm_to_sigma
from scipy.interpolate import InterpolatedUnivariateSpline
import pandas as pd
from scipy.ndimage import map_coordinates
from ..fits import open_fits
from .utils_spec import find_nearest


def make_resampled_models(lbda_obs, grid_param_list, model_grid=None,
                          model_reader=None, dlbda_obs=None, instru_fwhm=None, 
                          instru_idx=None, filter_reader=None):
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
     
    final_dims = dims+[len(lbda_obs)]+[2]
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
            lbda_mod, spec_mod = model_reader(params_tmp)
        
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
                   instru_fwhm=None, instru_idx=None, filter_reader=None):
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
        ### compute it
        dlbda_obs1 = [lbda_obs[1]-lbda_obs[0]]
        dlbda_obs2 = [(lbda_obs[i+2]-lbda_obs[i])/2 for i in range(n_ch-2)]
        dlbda_obs3 = [lbda_obs[-1]-lbda_obs[-2]]
        dlbda_obs = np.array(dlbda_obs1+dlbda_obs2+dlbda_obs3)
        
    if len(lbda_obs) != len(lbda_mod):
        idx_ini = find_nearest(lbda_mod, lbda_obs[0]-dlbda_obs[0], 
                                   constraint='floor')
        idx_fin = find_nearest(lbda_mod, lbda_obs[-1]+dlbda_obs[-1], 
                               constraint='ceil')
        lbda_mod = lbda_mod[idx_ini:idx_fin+1]
        spec_mod = spec_mod[idx_ini:idx_fin+1]
    elif not np.allclose(lbda_obs, lbda_mod):
        idx_ini = find_nearest(lbda_mod, lbda_obs[0]-dlbda_obs[0], 
                               constraint='floor')
        idx_fin = find_nearest(lbda_mod, lbda_obs[-1]+dlbda_obs[-1], 
                               constraint='ceil')
        lbda_mod = lbda_mod[idx_ini:idx_fin+1]
        spec_mod = spec_mod[idx_ini:idx_fin+1]
        
        
    nmod = lbda_mod.shape[0]
    ## compute the wavelength sampling of the model
    dlbda_mod1 = [lbda_mod[1]-lbda_mod[0]]
    dlbda_mod2 = [(lbda_mod[i+1]-lbda_mod[i-1])/2 for i in range(1,nmod-1)]
    dlbda_mod3 = [lbda_mod[-1]-lbda_mod[-2]]
    dlbda_mod = np.array(dlbda_mod1+dlbda_mod2+dlbda_mod3)


    ## check where obs spec res is < or > than model's
    do_interp = np.zeros(n_ch, dtype='int32')
    nchunks_i = 0
    for ll in range(n_ch):
        idx_near = find_nearest(lbda_mod, lbda_obs[ll])
        do_interp[ll] = (dlbda_obs[ll] <= dlbda_mod[idx_near])
        if ll > 0:
            if do_interp[ll] and not do_interp[ll-1]:
                nchunks_i+=1
        elif do_interp[ll]:
            nchunks_i=1

                
    ## interpolate model where the observed spectrum has higher resolution
    if np.sum(do_interp):
        idx_0=0
        for nc in range(nchunks_i):
            idx_1 = np.argmax(do_interp[idx_0:])+idx_0
            idx_0 = np.argmin(do_interp[idx_1:])+idx_1
            idx_ini = find_nearest(lbda_mod,lbda_obs[idx_1],
                                   constraint='floor')
            idx_fin = find_nearest(lbda_mod,lbda_obs[idx_0-1],
                                   constraint='ceil')  
            spl = InterpolatedUnivariateSpline(lbda_mod[idx_ini:idx_fin], 
                                               spec_mod[idx_ini:idx_fin])
            spec_mod_res[idx_1:idx_0] = spl(lbda_obs[idx_1:idx_0])
            
    ## convolve+bin where the model spectrum has higher resolution (most likely)
    if np.sum(do_interp) < n_ch:
        if instru_fwhm is None:
            msg = "Spectral FWHM of the instrument should be provided"
            msg+= " for convolution"
            raise ValueError(msg)
        if isinstance(instru_fwhm, float) or isinstance(instru_fwhm, int):
            instru_fwhm = [instru_fwhm]
        if isinstance(instru_idx, list):
            instru_idx = np.array(instru_idx)
        elif not isinstance(instru_idx, np.ndarray):
            instru_idx = np.array([1])

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
                msg = "instru_fwhm is a {}, while it should be either scalar or string"
                raise TypeError(msg.format(type(instru_fwhm[i-1])))

    return np.array([lbda_obs, spec_mod_res])

    
def model_interpolation(params, grid_param_list, model_grid=None, 
                        model_reader=None, interp_order=1):
    """
    Parameters
    ----------
    params : dictionary
        Set of models parameters for which the model grid has to be 
        interpolated.
    grid_param_list : list of 1d numpy arrays/lists
        - If list, should contain list/numpy 1d arrays with available grid of 
        model parameters.
        - Note1: model grids should not contain grids on radius and Av, but 
        these should still be passed in initial_state (Av optional).
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

    if interp_order == 0:
        if model_grid is None:
            params_tmp = np.zeros(n_params)    
            for nn in range(n_params):
                params_tmp[nn] = find_nearest(grid_param_list[nn], 
                                              params[nn], output='value')
            # crop lbda_ini and lbda_fin?
            tmp = model_reader(params_tmp)
            return tmp[0], tmp[1]
            
        else:
            idx_tmp = []
            for nn in range(n_params):
                idx_tmp.append(find_nearest(grid_param_list[nn], params[nn],
                                            output='index'))
            idx_tmp = tuple(idx_tmp)
            tmp = model_grid[idx_tmp]
            return tmp[:,0], tmp[:,1]
    
    
    elif interp_order == 1:
        # first compute new subgrid "coords" for interpolation 
        constraints = ['floor','ceil']
        new_coords = np.zeros([n_params,1])
        sub_grid_param = np.zeros([n_params,2])  
        for nn in range(n_params):
            for ii in range(2):
                sub_grid_param[nn,ii] = find_nearest(grid_param_list[nn], 
                                                     params[nn],
                                                     constraint=constraints[ii], 
                                                     output='value')
            num = (params[nn]-sub_grid_param[nn,0])
            denom = (sub_grid_param[nn,1]-sub_grid_param[nn,0])
            new_coords[nn,0] = num/denom

        sub_grid = []
        sub_grid_lbda = []        
        if model_grid is None:  
            ntot_subgrid = 2**n_params
            
            for dd in range(ntot_subgrid):
                str_indices = _den_to_bin(dd, n_params)
                params_tmp = []
                for nn in range(n_params):
                    params_tmp.append(sub_grid_param[nn,int(str_indices[nn])])
                params_tmp=np.array(params_tmp)
                tmp = model_reader(params_tmp)
                sub_grid.append(tmp[1])
                sub_grid_lbda.append(tmp[0])
            
            # Crate array with dimensions 'dims' for each wavelength
            nch = len(sub_grid_lbda[0])
            final_dims = tuple([nch]+[2]*n_params)
            
            sub_grid = np.array(sub_grid)
            sub_grid = np.swapaxes(sub_grid,0,1)
            sub_grid = sub_grid.reshape(final_dims)
            sub_grid_lbda = np.array(sub_grid_lbda)
            sub_grid_lbda = np.swapaxes(sub_grid_lbda,0,1)
            sub_grid_lbda = sub_grid_lbda.reshape(final_dims)
        
        else:
            constraints = ['floor','ceil']
            sub_grid_idx = np.zeros([n_params,2], dtype=np.int32)
            #list_idx = []
            for nn in range(n_params):
                for ii in range(2):
                    sub_grid_idx[nn,ii]=find_nearest(grid_param_list[nn], 
                                                     params[nn],
                                                     constraint=constraints[ii], 
                                                     output='index')
            for dd in range(2**n_params):
                str_indices = _den_to_bin(dd, n_params)
                idx_tmp = []
                for nn in range(n_params):
                    idx_tmp.append(sub_grid_idx[nn,int(str_indices[nn])])
                    #idx_tmp = sub_grid_idx[nn,int(str_indices[nn])]
                #list_idx.append(idx_tmp)
                #list_idx=np.array(list_idx)
                sub_grid.append(model_grid[tuple(idx_tmp)])
            
            # first reshape
            sub_grid = np.array(sub_grid)
            dims = tuple([2]*n_params+[sub_grid.shape[-2]]+[sub_grid.shape[-1]])
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
        for cc in range(nch):
            interp_model[cc] = map_coordinates(sub_grid[cc], new_coords, 
                                               order=interp_order)
            interp_lbdas[cc] = map_coordinates(sub_grid_lbda[cc], new_coords, 
                                               order=interp_order)
            
        return interp_lbdas, interp_model
    
    else:
        msg = "Interpolation order not allowed. Only 0 or 1 accepted"
        raise TypeError(msg)