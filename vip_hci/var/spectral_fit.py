#! /usr/bin/env python

"""
Functions useful for spectral fitting of companions.
"""

__author__ = 'Valentin Christiaens'
__all__ = ['goodness_of_fit',
           'combine_spec_corrs']

import numpy as np
from astropy.convolution import Gaussian1DKernel, convolve_fft
from astropy.stats import gaussian_fwhm_to_sigma
from scipy.interpolate import InterpolatedUnivariateSpline
from matplotlib import pyplot as plt
from ..conf import vip_figsize


def goodness_of_fit(lbda_obs, spec_obs, err_obs, lbda_mod, spec_mod, 
                    dlbda_obs=None, instru_corr=None, instru_fwhm=None, 
                    instru_idx=None, plot=False, outfile=None):
    """ Function to estimate the goodness of fit indicator (chi^2) defined as 
    in Olofsson et al. 2016 (Eq. 8). In addition, if a spectral 
    correlation matrix is provided, it is used to take into account the 
    correlated noise between spectral channels (see Greco & Brandt 2017).

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
    spec_mod : 
        Model spectrum. It does not require the same wavelength sampling as the
        observed spectrum. If higher spectral resolution, it will be convolved
        with the instrumental spectral psf (if instru_fwhm is provided) and 
        then binned to the same sampling. If lower spectral resolution, a 
        linear interpolation is performed to infer the value at the observed 
        spectrum wavelength sampling.
    dlbda_obs: numpy 1d ndarray or list
        Spectral channel width for the observed spectrum. If the observed 
        spectrum is obtained from a single instrument, no need to provide 
        dlbda_obs. It should be provided IF points are obtained from different 
        instruments AND one wants to weigh each point based on the spectral 
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
    instru_idx: numpy 1d array or list, optional
        List or 1d array containing an index representing each instrument used 
        to obtain the spectrum, label them from 0 to n_instru. Zero for points 
        that don't correspond to any instru_fwhm provided above, and i in 
        [1,n_instru] for points associated to instru_fwhm[i-1]. This parameter 
        must be provided if the spectrum consists of points obtained with 
        different instruments.
    plot : bool, optional
        Whether to plot the 
    outfile : string, optional
        Path+filename for the plot to be saved if provided.
        
    Returns
    -------
    chi_sq : float
        Goodness of fit indicator.

    """
    
    def find_nearest(array,value,output='index',constraint=None):
        """
        Function to find the (flattened) index, and optionally the value, of an
        array's closest element to a certain value.
        Possible outputs: 'index','value','both' 
        Possible constraints: 'ceil', 'floor', None. "ceil" will return the 
        closest element with a value greater than 'value', "floor" the opposite
        """
        idx = (np.abs(array-value)).argmin()
        if constraint == 'ceil' and np.ravel(array)[idx]-value < 0:
            idx+=1
        elif constraint == 'floor' and value-np.ravel(array)[idx] < 0:
            idx-=1
    
        if output=='index': return idx
        elif output=='value': return np.ravel(array)[idx]
        else: return np.ravel(array)[idx], idx

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
        if lbda_obs.shape != dlbda_obs.shape:
            raise TypeError('The input lbda_obs and dlbda_obs need same shape')

    n_ch = lbda_obs.shape[0]
             
    # interpolate OR convolve+bin model spectrum if not same sampling
    if lbda_obs != lbda_mod:
        spec_mod_fin = np.zeros_like(spec_obs)
                
        ## just limit the model to a section around the observed sepctrum
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
        
        if dlbda_obs is None:
            ### compute it
            dlbda_obs1 = [lbda_obs[1]-lbda_obs[0]]
            dlbda_obs2 = [(lbda_obs[i+2]-lbda_obs[i])/2 for i in range(n_ch-2)]
            dlbda_obs3 = [lbda_obs[-1]-lbda_obs[-2]]
            dlbda_obs = np.array(dlbda_obs1+dlbda_obs2+dlbda_obs3)

        ## check where obs spec res is < or > than model's
        do_interp = np.zeros(n_ch, dtype='int32')
        nchunks_i = 0        
        nchunks_c = 0
        for ll in range(n_ch):
            idx_near = find_nearest(lbda_mod, lbda_obs[ll])
            do_interp[ll] = (dlbda_obs[ll] < dlbda_mod[idx_near])
            if ll > 0:
                if do_interp[ll] and not do_interp[ll-1]:
                    nchunks_i+=1
                if not do_interp[ll] and do_interp[ll-1]:
                    nchunks_c+=1
        ## interpolate if needed
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
                spec_mod_fin[idx_1:idx_0] = spl(lbda_obs[idx_1:idx_0])
        ## convolve+bin if needed
        if np.sum(do_interp) < n_ch:
            if not isinstance(instru_fwhm, list):
                instru_fwhm = [instru_fwhm]
            for i in range(1,len(instru_fwhm)+1):
                ifwhm = instru_fwhm[i-1]/(1000*np.mean(dlbda_mod))
                gau_ker = Gaussian1DKernel(stddev=ifwhm*gaussian_fwhm_to_sigma)
                spec_mod_conv = convolve_fft(spec_mod, gau_ker)
                
                tmp = np.zeros_like(lbda_obs[np.where(instru_idx==i)])
                for ll, lbda in enumerate(lbda_obs[np.where(instru_idx==i)]):
                    i_f = find_nearest(lbda_mod,
                                       lbda_obs[np.where(instru_idx==i)][ll])
                    i_l = find_nearest(lbda_mod,
                                       lbda_obs[np.where(instru_idx==i)][ll+1])
                    tmp[ll] = np.mean(spec_mod_conv[i_f:i_l+1])
                spec_mod_fin[np.where(instru_idx==i)] = tmp       
    else:
        spec_mod_fin = spec_mod
        
    # compute covariance matrix
    if err_obs.ndim == 2:
        err_obs = (np.absolute(err_obs[0])+np.absolute(err_obs[1]))/2.
    cov = np.ones_like(instru_corr)
    for ii in range(n_ch):
        cov[ii,ii] = err_obs[ii]
    for ii in range(n_ch):
        for jj in range(n_ch):
            cov[ii,jj] = instru_corr[ii,jj]*np.sqrt(cov[ii,ii]*cov[jj,jj])    

    delta = spec_obs-spec_mod_fin
    chi_sq = np.linalg.multi_dot((delta,np.linalg.inv(cov),delta))  
                    
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

    
def combine_spec_corrs(arr_list):
    """ Combines the spectral correlation matrices of different instruments 
    into a single square matrix (required for input of spectral fit).
    
    Parameters
    ----------
    arr_list : list or tuple of numpy ndarrays
        List/tuple containing the distinct square spectral correlation matrices 
        OR ones (for independent photometric measurements). 

    Returns
    -------
    combi_corr : numpy 2d ndarray
        2d square ndarray representing the combined spectral correlation.
        
    """
    n_arr = len(arr_list)
    
    size = 0
    for nn in range(n_arr):
        if isinstance(arr_list[nn],np.ndarray):
            if arr_list[nn].ndim != 2:
                raise TypeError("Arrays of the tuple should be 2d")
            elif arr_list[nn].shape[0] != arr_list[nn].shape[1]:
                raise TypeError("Arrays of the tuple should be square")
            size+=arr_list[nn].shape[0]
        elif arr_list[nn] == 1:
            size+=1
        else:
            raise TypeError("Tuple can only have square 2d arrays or ones")
            
    combi_corr = np.zeros([size,size])
    
    size_tmp = 0
    for nn in range(n_arr):
        if isinstance(arr_list[nn],np.ndarray):
            mm = arr_list[nn].shape[0]
            combi_corr[size_tmp:size_tmp+mm,size_tmp:size_tmp+mm]=arr_list[nn]
            size_tmp+=mm
        elif arr_list[nn] == 1:
            combi_corr[size_tmp,size_tmp]=1
            size_tmp+=1      
        
    return combi_corr