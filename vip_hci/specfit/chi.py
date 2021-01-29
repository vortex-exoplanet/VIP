#! /usr/bin/env python

"""
Function defining the goodness of fit.
"""

__author__ = 'Valentin Christiaens'
__all__ = ['goodness_of_fit',
           'gof_scal']

import numpy as np
from matplotlib import pyplot as plt
from ..conf import vip_figsize
from .model_resampling import resample_model
from .utils_spec import extinction

def goodness_of_fit(lbda_obs, spec_obs, err_obs, lbda_mod, spec_mod, 
                    dlbda_obs=None, instru_corr=None, instru_fwhm=None, 
                    instru_idx=None, filter_reader=None, plot=False, 
                    outfile=None):
    """ Function to estimate the goodness of fit indicator defined as 
    in Olofsson et al. 2016 (Eq. 8). In addition, if a spectral 
    correlation matrix is provided, it is used to take into account the 
    correlated noise between spectral channels (see Greco & Brandt 2016).
    The goodness of fit indicator is identical to a chi square when all
    points are obtained from the same instrument (no additional weighting).

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
    plot : bool, optional
        Whether to plot the 
    outfile : string, optional
        Path+filename for the plot to be saved if provided.
        
    Returns
    -------
    chi_sq : float
        Goodness of fit indicator.

    """

    lbda_obs = np.array(lbda_obs)
    spec_obs = np.array(spec_obs)
    err_obs = np.array(err_obs)
    lbda_mod = np.array(lbda_mod)
    spec_mod = np.array(spec_mod)
    
    if lbda_obs.ndim != 1 or spec_obs.ndim != 1:
        raise TypeError('Observed lbda and spec must be lists or 1d arrays')
    if lbda_obs.shape[0] != spec_obs.shape[0]:
        raise TypeError('Obs lbda and spec need same shape')        
    if lbda_obs.shape[0] != err_obs.shape[-1]:
        raise TypeError('Obs lbda and err need same shape')
    if lbda_mod.ndim != 1 or spec_mod.ndim != 1:
        raise TypeError('The input model lbda/spec must be lists or 1d arrays')
    else:
        if lbda_mod.shape != spec_mod.shape:
            raise TypeError('The input model lbda and spec need same shape')
    if dlbda_obs is not None:
        if isinstance(dlbda_obs, list):
            dlbda_obs = np.array(dlbda_obs)
        if lbda_obs.shape != dlbda_obs.shape:
            raise TypeError('The input lbda_obs and dlbda_obs need same shape')

    n_ch = lbda_obs.shape[0]
             
    # interpolate OR convolve+bin model spectrum if not same sampling
    if not np.allclose(lbda_obs, lbda_mod):
        spec_mod_fin = resample_model(lbda_obs, lbda_mod, spec_mod, dlbda_obs, 
                                      instru_fwhm, instru_idx, filter_reader)    
    else:
        spec_mod_fin = spec_mod
        
    # compute covariance matrix
    if err_obs.ndim == 2:
        err_obs = (np.absolute(err_obs[0])+np.absolute(err_obs[1]))/2.
    if instru_corr is None:
        instru_corr = np.diag(np.ones(n_ch))
    cov = np.ones_like(instru_corr)
    for ii in range(n_ch):
        for jj in range(n_ch):
            cov[ii,jj] = instru_corr[ii,jj]*err_obs[ii]*err_obs[jj] 

    delta = spec_obs-spec_mod_fin
    wi = np.ones_like(dlbda_obs)
    if dlbda_obs is not None:
        if np.sum(np.power((dlbda_obs[1:]/lbda_obs[1:])-(dlbda_obs[:-1]/lbda_obs[:-1]),2))!=0:
            # normalize weights for their sum to be equal to the number of points
            wi = np.sqrt(((dlbda_obs/lbda_obs)/np.sum(dlbda_obs/lbda_obs))*dlbda_obs.shape[0])    
   
    chi_sq = np.linalg.multi_dot((wi*delta,np.linalg.inv(cov),wi*delta))
                
    if plot:
        _, ax = plt.subplots(figsize=vip_figsize)

        ax.plot(lbda_obs, lbda_obs*spec_obs, 'o', alpha=0.6, color='#1f77b4',
                label="Measured spectrum")
        ax.plot(lbda_obs, lbda_mod*spec_mod, '-', alpha=0.4, color='#1f77b4',
                label="Model")
        plt.xlabel('Wavelength')
        plt.ylabel(r'Flux density ($\lambda F_{\lambda}$)')

        plt.xlim(xmin=0.9*lbda_obs[0], xmax=1.1*lbda_obs[-1])
        plt.minorticks_on()
        plt.legend(fancybox=True, framealpha=0.5, fontsize=12, loc='best')
        plt.grid(which='major', alpha=0.2)
        if outfile:
            plt.savefig(outfile, bbox_inches='tight')
        plt.show()

    return chi_sq


def gof_scal(params, lbda_obs, spec_obs, err_obs, lbda_mod, spec_mod, dlbda_obs, 
             instru_corr, instru_fwhm, instru_idx, filter_reader, ext_range):
    """ Wrapper for the goodness of fit routine to search for best template 
    library fitting spectrum. The only difference is the "params" argument, 
    
    Parameters
    ----------
    params: tuple
        Tuple of 1 or 2 elements: scaling factor and (optionally) differential
        optical extinction (i.e. Delta A_V can be negative if template spectra
        are not dereddened).
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
    ext_range: tuple or None, opt
        If None: differential extinction is not to be considered as a free 
        parameter. Elif a tuple of 3 floats is provided, differential extinction 
        will be considered, with the floats as lower limit, upper limit and step 
        of the grid search.
        Note: if simplex search, the range is still used to set a chi of 
        np.inf outside of the range.
        
    Returns
    -------
    chi_sq : float
        Goodness of fit indicator.
    """
    tmp_spec = spec_mod*params[0]
    
    if len(params) == 2:
        if ext_range is None:
            raise TypeError("ext_range should be a tuple of 3 elements")
        else:
            if params[1]<ext_range[0] or params[1]>ext_range[1]:
                return np.inf
            AV=params[1]
            thr = ext_range[-1]
            if abs(AV) < thr:
                AV = 0
            Albdas = extinction(lbda_obs,abs(AV))
            extinc_fac = np.power(10.,-Albdas/2.5)
            if AV>0:
                tmp_spec *= extinc_fac
            elif AV<0:
                tmp_spec /= extinc_fac
    elif len(params) > 2:
        raise TypeError("params tuple should have length 1 or 2")
    
    
    return goodness_of_fit(lbda_obs, spec_obs, err_obs, lbda_obs, tmp_spec, 
                           dlbda_obs=dlbda_obs, instru_corr=instru_corr, 
                           instru_fwhm=instru_fwhm, instru_idx=instru_idx, 
                           filter_reader=filter_reader)