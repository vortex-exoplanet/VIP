#! /usr/bin/env python

"""
Module for simplex or grid search of best fit spectrum in a template library.
TBD: implement multiprocessing version.
"""

__author__ = 'V. Christiaens'
__all__ = ['best_fit_tmp',
           'get_chi']

from datetime import datetime
import numpy as np
import os
from scipy.optimize import minimize
from ..conf import time_ini, timing, time_fin
from .chi import gof_scal
from .model_resampling import resample_model
from .utils_spec import extinction, find_nearest
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


def get_chi(lbda_obs, spec_obs, err_obs, tmp_name, tmp_reader, 
            search_mode='simplex', lambda_scal=None, scale_range=(0.1,10,0.01), 
            ext_range=None, dlbda_obs=None, instru_corr=None, instru_fwhm=None, 
            instru_idx=None, filter_reader=None, simplex_options=None,
            red_chi2=True, remove_nan=False, force_continue=False, 
            verbose=False, **kwargs):
    """ Routine calculating chi^2, optimal scaling factor and optimal 
    extinction for a given template spectrum to match an observed spectrum.
    
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
    tmp_name :  str
        Template spectrum filename.
    tmp_reader : python routine
        External routine that reads a model file and returns a 3D numpy array, 
        where the first column corresponds to wavelengths, the second 
        contains flux values, and the third the uncertainties on the flux.
    search_mode: str, opt {'simplex', 'grid'}
        How is the best fit template found? Simplex or grid search.
    lambda_scal: float, optional
        Wavelength where a first scaling will be performed between template
        and observed spectra. If not provided, the middle wavelength of the 
        osberved spectra will be considered.
    scale_range: tuple, opt
        If grid search, this parameter should be provided as a tuple of 3 
        floats: lower limit, upper limit and step of the grid search for the 
        scaling factor to be applied AFTER the first rough scaling (i.e.
        scale_range should always encompass 1).
    ext_range: tuple or None, opt
        If None: differential extinction is not to be considered as a free 
        parameter. Elif a tuple of 3 floats is provided, differential extinction 
        will be considered, with the floats as lower limit, upper limit and step 
        of the grid search.
        Note: if simplex search, the range is still used to set a chi of 
        np.inf outside of the range.
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
    red_chi2: bool, optional
        Whether to compute the reduced chi square. If False, considers chi^2.
    remove_nan: bool, optional
        Whether to remove NaN values from template spectrum BEFORE resampling
        to the wavelength sampling of the observed spectrum. Whether it is set
        to True or False, a check is made just before chi^2 is calculated 
        (after resampling), and only non-NaN values will be considered.
    simplex_options: dict, optional
        The scipy.optimize.minimize simplex (Nelder-Mead) options.
    force_continue: bool, optional
        In case of issue with the fit, whether to continue regardless (this may
        be useful in an uneven spectral library, where some templates have too
        few points for the fit to be performed).
    verbose: str, optional
        Whether to print more information when fit fails.
    **kwargs: optional
        Optional arguments to the scipy.optimize.minimize function.
        
    Returns
    -------
    best_chi: float
        goodness of fit scored by the template
    best_scal:
        best-fit scaling factor for the considered template
    best_ext:
        best-fit optical extinction for the considered template
        
    """
    # read template spectrum
    try:
        lbda_tmp, spec_tmp, spec_tmp_err = tmp_reader(tmp_name, 
                                                  verbose=bool(verbose))
    except:
        msg = "{} could not be opened. Corrupt file?".format(tmp_name)
        if force_continue:
            if verbose:
                print(msg)
            return np.inf, 1, 0, 1
        else:
            raise ValueError(msg)
            
    # look for any nan and replace
    if remove_nan:
        if np.isnan(spec_tmp).any() or np.isnan(spec_tmp_err).any():
            bad_idx1 = np.where(np.isnan(spec_tmp))[0]
            bad_idx2 = np.where(np.isnan(spec_tmp_err))[1]
            all_bad = np.concatenate(bad_idx1,bad_idx2)
            nch = len(lbda_tmp)
            new_lbda = [lbda_tmp[i] for i in range(nch) if i not in all_bad]
            new_spec = [spec_tmp[i] for i in range(nch) if i not in all_bad]
            new_err = [spec_tmp_err[i] for i in range(nch) if i not in all_bad]
            lbda_tmp = np.array(new_lbda)
            spec_tmp = np.array(new_spec)
            spec_tmp_err = np.array(new_err)
        
    # don't consider template spectra whose range is smaller than observed one
    if lbda_obs[0] < lbda_tmp[0] or lbda_obs[-1] > lbda_tmp[-1]:
        msg = "Wavelength range of template {} ({:.2f}, {:.2f})mu too short"
        msg+= " compared to that of observed spectrum ({:.2f}, {:.2f})mu"
        if force_continue:
            if verbose:
                print(msg.format(tmp_name, lbda_tmp[0],lbda_tmp[-1],
                                        lbda_obs[0],lbda_obs[-1]))
            return np.inf, 1, 0, len(lbda_tmp)-2
        else:
            raise ValueError(msg.format(tmp_name, lbda_tmp[0],lbda_tmp[-1],
                                        lbda_obs[0],lbda_obs[-1]))
    
    # resample as observed spectrum
    try:
        _, spec_tmp = resample_model(lbda_obs, lbda_tmp, spec_tmp, 
                                     dlbda_obs=dlbda_obs, 
                                     instru_fwhm=instru_fwhm, 
                                     instru_idx=instru_idx, 
                                     filter_reader=filter_reader)
        lbda_tmp, spec_tmp_err = resample_model(lbda_obs, lbda_tmp, 
                                                spec_tmp_err, 
                                                dlbda_obs=dlbda_obs, 
                                                instru_fwhm=instru_fwhm, 
                                                instru_idx=instru_idx, 
                                                filter_reader=filter_reader)
    except:
        msg = "Issue with resampling of template {}. Does the wavelength "
        msg+= "range extend far enough ({:.2f}, {:.2f})mu?"
        if force_continue:
            if verbose:
                print(msg.format(tmp_name, lbda_tmp[0],lbda_tmp[-1]))
            return np.inf, 1, 0, len(lbda_tmp)-2
        else:
            raise ValueError(msg.format(tmp_name, lbda_tmp[0],lbda_tmp[-1]))
    
    # first rescaling fac
    if not lambda_scal:
        lambda_scal = (lbda_obs[0]+lbda_obs[-1])/2
    idx_cen = find_nearest(lbda_obs, lambda_scal)
    scal_fac = spec_obs[idx_cen]/spec_tmp[idx_cen]
    spec_tmp*=scal_fac
    spec_tmp_err*=scal_fac
    
    # combine observed and template uncertainties
    # EDIT: Don't: the  best fit will be the most noisy tmp of the library!)
    #err_obs = np.sqrt(np.power(spec_tmp_err,2)+np.power(err_obs,2))
    
    
    # only consider non-zero and non-nan values for chi^2 calculation
    cond1 = np.where(np.isfinite(spec_tmp))[0]
    cond2 = np.where(np.isfinite(err_obs))[0] 
    cond3 = np.where(spec_tmp>0)[0]
    all_conds = np.sort(np.unique(np.concatenate((cond1,cond2,cond3))))
    ngood_ch = len(all_conds)
    good_ch = (all_conds,)
    lbda_obs = lbda_obs[good_ch]
    spec_obs = spec_obs[good_ch]
    err_obs = err_obs[good_ch]
    spec_tmp = spec_tmp[good_ch]
    
    n_dof = ngood_ch-1-(ext_range is not None)
    if n_dof <= 0:
        msg = "Not enough dof with remaining points of template spectrum {}"
        if force_continue:
            if verbose:
                print(msg.format(tmp_name))
            return np.inf, 1, 0, n_dof
        else:
            raise ValueError(msg.format(tmp_name))
    
    # simplex search
    if search_mode == 'simplex':
        if simplex_options is None:
            simplex_options = {'xatol': 1e-6, 'fatol': 1e-6, 'maxiter': 1000,
                               'maxfev': 5000}
        if not ext_range:
            p = (1,)
        else:
            AV_ini = (ext_range[0]+ext_range[1])/2
            p = (1,AV_ini)
        try:
            res = minimize(gof_scal, p, args=(lbda_obs, spec_obs, err_obs, 
                                              lbda_obs, spec_tmp, dlbda_obs, 
                                              instru_corr, instru_fwhm, 
                                              instru_idx, filter_reader,
                                              ext_range),
                           method='Nelder-Mead', options=simplex_options, 
                           **kwargs)
        except:
            msg = "Issue with simplex minimization for template {}. "
            msg+= "Try grid search?"
            if force_continue:
                if verbose:
                    print(msg.format(tmp_name))
                return np.inf, 1, 0, n_dof
            else:
                raise ValueError(msg.format(tmp_name))
        best_chi = res.fun
        if not ext_range:
            best_scal = res.x
            best_ext = 0
        else:
            best_scal, best_ext = res.x
        
    # or grid search        
    elif search_mode == 'grid':
        test_scale = np.arange(scale_range[0], scale_range[1], scale_range[2])
        n_test = len(test_scale)
        if ext_range is None:
            test_ext = np.array([0])
            n_ext = 1
        elif isinstance(ext_range, tuple) and len(ext_range)==3:
            test_ext = np.arange(ext_range[0], ext_range[1], ext_range[2])
            n_ext = len(test_ext)
        else:
            raise TypeError("ext_range can only be None or tuple of length 3")

        chi = np.zeros([n_test,n_ext])
        
        for cc, scal in enumerate(test_scale):
            for ee, AV in enumerate(test_ext):
                p = (scal,AV)
                chi[cc,ee] = gof_scal(p, lbda_obs, spec_obs, err_obs, lbda_obs, 
                                      spec_tmp, dlbda_obs=dlbda_obs, 
                                      instru_corr=instru_corr, 
                                      instru_fwhm=instru_fwhm, 
                                      instru_idx=instru_idx, 
                                      filter_reader=filter_reader,
                                      ext_range=ext_range) 
        best_chi = np.nanmin(chi)
        best_idx = np.nanargmin(chi)
        best_idx = np.unravel_index(best_idx,chi.shape)
        best_scal = test_scale[best_idx[0]]
        best_ext = test_ext[best_idx[1]]
    
    else:
        msg = "Search mode not recognised. Should be 'simplex' or 'grid'."
        raise TypeError(msg)
    
    if red_chi2:
        best_chi /= n_dof
        
    
    return best_chi, best_scal*scal_fac, best_ext, n_dof



def best_fit_tmp(lbda_obs, spec_obs, err_obs, tmp_reader, search_mode='simplex',
                 n_best=1, lambda_scal=None, scale_range=(0.1,10,0.01), 
                 ext_range=None, simplex_options=None, dlbda_obs=None, 
                 instru_corr=None, instru_fwhm=None, instru_idx=None, 
                 filter_reader=None, lib_dir='tmp_lib/', tmp_endswith='.fits', 
                 red_chi2=True, remove_nan=False, nproc=1, verbosity=0, 
                 force_continue=False, **kwargs):
    """ Finds the best fit template spectrum to a given observed spectrum, 
    within a spectral library.  By default, a single free parameter is 
    considered: the scaling factor of the spectrum. A first automatic scaling 
    is performed by comparing the flux of the observed and template spectra at 
    lambda_scal. Then a more refined scaling is performed, either through 
    simplex or grid search (within scale_range).
    If fit_extinction is set to True, the exctinction is also considered as a 
    free parameter.
    
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
    tmp_reader : python routine
        External routine that reads a model file and returns a 3D numpy array, 
        where the first column corresponds to wavelengths, the second 
        contains flux values, and the third the uncertainties on the flux.
    search_mode: str, optional, {'simplex','grid'}
        How is the best fit template found? Simplex or grid search.
    n_best: int, optional
        Number of best templates to be returned (default: 1)
    lambda_scal: float, optional
        Wavelength where a first scaling will be performed between template
        and observed spectra. If not provided, the middle wavelength of the 
        osberved spectra will be considered.
    scale_range: tuple, opt
        If grid search, this parameter should be provided as a tuple of 3 
        floats: lower limit, upper limit and step of the grid search for the 
        scaling factor to be applied AFTER the first rough scaling (i.e.
        scale_range should always encompass 1).
    ext_range: tuple or None, opt
        If None: differential extinction is not to be considered as a free 
        parameter. Elif a tuple of 3 floats is provided, differential extinction 
        will be considered, with the floats as lower limit, upper limit and step 
        of the grid search.
        Note: if simplex search, the range is still used to set a chi of 
        np.inf outside of the range.
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
    simplex_options: dict, optional
        The scipy.optimize.minimize simplex (Nelder-Mead) options.
    red_chi2: bool, optional
        Whether to compute the reduced chi square. If False, considers chi^2.
    remove_nan: bool, optional
        Whether to remove NaN values from template spectrum BEFORE resampling
        to the wavelength sampling of the observed spectrum. Whether it is set
        to True or False, a check is made just before chi^2 is calculated 
        (after resampling), and only non-NaN values will be considered.
    nproc: int, optional
        The number of processes to use for parallelization.
    verbosity: 0, 1 or 2, optional
        Verbosity level. 0 for no output and 2 for full information.
    force_continue: bool, optional
        In case of issue with the fit, whether to continue regardless (this may
        be useful in an uneven spectral library, where some templates have too
        few points for the fit to be performed).
    **kwargs: optional
        Optional arguments to the scipy.optimize.minimize function
         
    Returns
    -------
    final_tmpname: tuple of n_best str
        Best-fit template filenames
    final_tmp: tuple of n_best 3D numpy array
        Best-fit template spectra (3D: lbda+spec+spec_err)
    final_chi: 1D numpy array of length n_best
        Best-fit template chi^2
    final_params: 2D numpy array (2xn_best)
        Best-fit parameters (optimal scaling and optical extinction). Note if 
        extinction is not fitted, optimal AV will be set to 0.
        
    """
    # create list of template filenames
    tmp_filelist = [x for x in os.listdir(lib_dir) if x.endswith(tmp_endswith)]
    n_tmp = len(tmp_filelist)
    
    if verbosity > 0:
        start_time = time_ini()
        int_time = time_ini()
        print("{:.0f} template spectra will be tested. \n".format(n_tmp))
        print("****************************************\n")
    
    chi = np.ones(n_tmp)
    scal = np.ones_like(chi)
    ext = np.zeros_like(chi)
    n_dof =  np.ones_like(chi)
    counter = 0
    
    if nproc == 1:
        for tt in range(n_tmp):
            if verbosity>0 and search_mode=='simplex':
                print('Nelder-Mead minimization is running...')
            chi[tt], scal[tt], ext[tt], n_dof[tt] = get_chi(lbda_obs, spec_obs, err_obs, 
                                                 tmp_filelist[tt], tmp_reader, 
                                                 search_mode=search_mode, 
                                                 scale_range=scale_range, 
                                                 ext_range=ext_range,
                                                 lambda_scal=lambda_scal,
                                                 dlbda_obs=dlbda_obs, 
                                                 instru_corr=instru_corr, 
                                                 instru_fwhm=instru_fwhm, 
                                                 instru_idx=instru_idx, 
                                                 filter_reader=filter_reader,
                                                 simplex_options=simplex_options,
                                                 red_chi2=red_chi2,
                                                 remove_nan=remove_nan,
                                                 force_continue=force_continue,
                                                 verbose=verbosity,
                                                 **kwargs)
            
            if chi[tt]<np.inf:
                counter+=1
            elif verbosity>0:
                msg = "{:.0f}/{:.0f} ({}) FAILED"
                if np.isnan(chi[tt]):
                    msg += " (simplex did not converge)"
                print(msg.format(tt, n_tmp, tmp_filelist[tt]))

            if verbosity > 0 and tt==0:
                msg = "{:.0f}/{:.0f}: done in {}s"
                indiv_time = time_fin(start_time)
                print(msg.format(tt, n_tmp, indiv_time))
                now = datetime.now()
                delta_t = now.timestamp()-start_time.timestamp()
                tot_time = n_tmp*delta_t/60
                print("Fit may take a total of ~{:.0f}min \n".format(tot_time))
                int_time = time_ini(verbose=False)
            elif verbosity > 1:
                msg = "{:.0f}/{:.0f}: done in {}s \n"
                indiv_time = time_fin(int_time)
                int_time = time_ini(verbose=False)
                print(msg.format(tt, n_tmp, indiv_time))
    else:
        raise ValueError("multiprocessing mode yet to be implemented")
        
    if verbosity > 0:
        print("****************************************\n")
        msg = "{:.0f}/{:.0f} template spectra were fitted. \n"
        print(msg.format(counter, n_tmp))
        timing(start_time)
        
    return best_n_tmp(chi, scal, ext, n_dof, tmp_filelist, tmp_reader, n_best=n_best,
                      verbose=True)
    
    
def best_n_tmp(chi, scal, ext, n_dof, tmp_filelist, tmp_reader, n_best=1, 
               verbose=False):
    """
    Routine returning the n best template spectra.
    
    Parameters
    ----------
    lbda_obs : numpy 1d ndarray or list
        Wavelength of observed spectrum. If several instruments, should be 
        ordered per instrument, not necessarily as monotonically increasing 
        wavelength. Hereafter, n_ch = len(lbda_obs).

    Returns
    -------
    final_tmpname: tuple of n_best str
        Best-fit template filenames
    final_tmp: tuple of n_best 3D numpy array
        Best-fit template spectra (3D: lbda+spec+spec_err)
    final_chi: 1D numpy array of length n_best
        Best-fit template chi^2
    final_params: 2D numpy array (2xn_best)
        Best-fit parameters (optimal scaling and optical extinction). Note if 
        extinction is not fitted, optimal AV will be set to 0.
        
    """
    # sort from best to worst match
    order = np.argsort(chi)
    sort_chi = chi[order]
    sort_scal = scal[order]
    sort_ext = ext[order]
    sort_ndof = n_dof[order]
    sort_filelist = [tmp_filelist[i] for i in order]
    
    if verbose:
        print("best chi: ", sort_chi[:n_best])
        print("best scale fac: ", sort_scal[:n_best])
        print("n_dof: ", sort_ndof[:n_best])
    # take the n_best first ones
    best_tmp = []
    for n in range(n_best):
        lbda_tmp, spec_tmp, spec_tmp_err = tmp_reader(sort_filelist[n])
        Albdas = extinction(lbda_tmp,abs(sort_ext[n]))
        extinc_fac = np.power(10.,-Albdas/2.5)
        if sort_ext[n]>0:
            final_scal = sort_scal[n]*extinc_fac
        elif sort_ext[n]<0:
            final_scal = sort_scal[n]/extinc_fac
        else:
            final_scal = sort_scal[n]
        best_tmp.append(np.array([lbda_tmp, spec_tmp*final_scal, 
                                  spec_tmp_err*final_scal]))
        if verbose:
            msg = "The best template #{:.0f} is: {} "
            msg+="(Delta A_V={:.1f}mag)\n"
            print(msg.format(n, sort_filelist[n], sort_ext[n]))
            
    best_tmpname = tuple(sort_filelist[:n_best])
    best_tmp = tuple(best_tmp)
    best_params = np.array([sort_scal[:n_best],sort_ext[:n_best]])
    
    return best_tmpname, best_tmp, sort_chi[:n_best], best_params, sort_ndof[:n_best]