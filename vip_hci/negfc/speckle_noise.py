#! /usr/bin/env python

"""
Module with routines related to the estimation of the uncertainty on the 
parameters of an imaged companion due to speckle noise.
"""

__author__ = 'O. Wertz, C. A. Gomez Gonzalez, V. Christiaens'
__all__ = ['speckle_noise_uncertainty']

#import itertools as itt
from multiprocessing import cpu_count
import numpy as np
import matplotlib.pyplot as plt

from ..conf.utils_conf import pool_map, iterable #eval_func_tuple
from ..metrics import cube_inject_companions 
from .simplex_optim import firstguess_simplex
from .simplex_fmerit import get_mu_and_sigma
from .utils_negfc import cube_planet_free
from .mcmc_sampling import confidence


def speckle_noise_uncertainty(cube, p_true, angle_range, derot_angles, algo, 
                              psfn, plsc, fwhm, aperture_radius, cube_ref=None, 
                              fmerit='sum', algo_options={}, transmission=None, 
                              mu_sigma=None, wedge=None, weights=None, 
                              force_rPA=False, nproc=None, simplex_options=None, 
                              save=False, output=None, verbose=True, 
                              full_output=True, plot=False):
    """
    Step-by-step procedure used to determine the speckle noise uncertainty 
    associated to the parameters of a companion candidate.

      __ 
     |   The steps 1 to 3 need to be performed for each angle.
     | 
     | - 1 - At the true planet radial distance and for a given angle, we 
     |       inject a fake companion in our planet-free cube.
     |   
     | - 2 - Then, using the negative fake companion method, we determine the 
     |       position and flux of the fake companion thanks to a Simplex  
     |       Nelder-Mead minimization.
     |
     | - 3 - We calculate the offset between the true values of the position 
     |       and the flux of the fake companion, and those obtained from the 
     |       minimization. The results will be dependent on the angular 
     |       position of the fake companion. 
     |__
     
    The resulting distribution of deviations is then used to infer the 
    1-sigma uncertainty on each parameter by fitting a 1d-gaussian.
     
    Parameters
    ----------
    cube: numpy array
        The original ADI cube.    
    p_true: tuple or numpy array with 3 elements
        The radial separation, position angle (from x=0 axis) and flux 
        associated to a given companion candidate for which the speckle 
        uncertainty is to be evaluated. The planet will first 
        be subtracted from the cube, then used for test injections.
    angle_range: 1d numpy array
        Range of angles (counted from x=0 axis, counter-clockwise) at which the 
        fake companions will be injected, in [0,360[.
    derot_angles: 1d numpy array
        Derotation angles for ADI. Length should match input cube.
    algo: python routine
        Routine to be used to model and subtract the stellar PSF. From an input
        cube, derotation angles, and optional arguments, it should return a 
        post-processed frame.
    algo_options: dict
        Options for algo. To be provided as a dictionary. Can include ncomp 
        (for PCA), svd_mode, collapse, imlib, interpolation, scaling, delta_rot
    transmission: numpy array, optional
        Array with 2 columns. First column is the radial separation in pixels. 
        Second column is the off-axis transmission (between 0 and 1) at the 
        radial separation given in column 1.
    mu_sigma: tuple of 2 floats, bool or None, opt
        If set to None: not used, and falls back to original version of the 
        algorithm, using fmerit.
        If a tuple of 2 elements: should be the mean and standard deviation of 
        pixel intensities in an annulus centered on the lcoation of the 
        companion candidate, excluding the area directly adjacent to the CC.
        If set to anything else, but None/False/tuple: will compute said mean 
        and standard deviation automatically.
    force_rPA: bool, optional
        Whether to only search for optimal flux, provided (r,PA).
    fmerit: None
        Figure of merit to use, if mu_sigma is None.
    simplex_options: dict
        ALl the required simplex parameters, for instance {'tol':1e-08, 
        'max_iter':200}    
    output: str (optional)
        The name of the output file (if save is True)    
    save: boolean (optional)
        If True, the result are pickled.
    verbose: boolean (optional)
        If True, informations are displayed in the shell.
     
    Returns:
    --------
    
    """    
    
    if not nproc:   # Hyper-threading "duplicates" the cores -> cpu_count/2
        nproc = (cpu_count()/2)
    
    if verbose:
        print('')
        print('#######################################################')
        print('###            SPECKLE NOISE DETERMINATION          ###')
        print('#######################################################')
        print('')
    
    r_true, theta_true, f_true = p_true

    if angle_range[0]%360 == angle_range[-1]%360:
        angle_range = angle_range[:-1]
                  
    if verbose:              
        print('Number of steps: {}'.format(angle_range.shape[0]))
        print('')
    
    imlib = algo_options.get('imlib','opencv')
    interpolation = algo_options.get('interpolation','lanczos4')
    
    # FIRST SUBTRACT THE TRUE COMPANION CANDIDATE
    planet_parameter = np.array([[r_true, theta_true, f_true]])
    cube_pf = cube_planet_free(planet_parameter, cube, derot_angles, psfn, plsc, 
                               imlib=imlib, interpolation=interpolation,
                               transmission=transmission)

    # Measure mu and sigma once in the annulus (instead of each MCMC step)
    if isinstance(mu_sigma,tuple):
        if len(mu_sigma)!=2:
            raise TypeError("If a tuple, mu_sigma must have 2 elements")
    elif mu_sigma is not None:
        ncomp = algo_options.get('ncomp', None)
        annulus_width = algo_options.get('annulus_width', int(fwhm))
        if weights is not None:
            if not len(weights)==cube.shape[0]:
                raise TypeError("Weights should have same length as cube axis 0")
            norm_weights = weights/np.sum(weights)
        else:
            norm_weights=weights
        mu_sigma = get_mu_and_sigma(cube, derot_angles, ncomp, annulus_width, 
                                    aperture_radius, fwhm, r_true, theta_true, 
                                    cube_ref=cube_ref, wedge=wedge, algo=algo, 
                                    weights=norm_weights, 
                                    algo_options=algo_options)
      
    # MULTIPROCESSING - old deprecated way
#    pool = Pool(processes=nproc)
#    res = pool.map(eval_func_tuple, itt.izip(itt.repeat(_estimate_speckle_one_angle),
#                                             angle_range,
#                                             itt.repeat(cube_pf),
#                                             itt.repeat(psfn),
#                                             itt.repeat(derot_angles),
#                                             itt.repeat(r_true),
#                                             itt.repeat(f_true),
#                                             itt.repeat(plsc),
#                                             itt.repeat(fwhm),
#                                             itt.repeat(annulus_width),
#                                             itt.repeat(aperture_radius),
#                                             itt.repeat(cube_ref),
#                                             itt.repeat(fmerit),
#                                             itt.repeat(algo),
#                                             itt.repeat(algo_options),
#                                             itt.repeat(transmission),
#                                             itt.repeat(mu_sigma),
#                                             itt.repeat(weights),
#                                             itt.repeat(force_rPA),
#                                             itt.repeat(simplex_options),
#                                             itt.repeat(verbose)))
    res = pool_map(nproc, _estimate_speckle_one_angle, iterable(angle_range), 
                   cube_pf, psfn, derot_angles, r_true, f_true, plsc, fwhm,
                   aperture_radius, cube_ref, fmerit, algo, algo_options, 
                   transmission, mu_sigma, weights, force_rPA, simplex_options, 
                   verbose=verbose)
    residuals = np.array(res)
    #pool.close()
 
    if verbose:  
        print("residuals (offsets): ", residuals[:,3],residuals[:,4],
              residuals[:,5])

    p_simplex = np.transpose(np.array([residuals[:,0],residuals[:,1],
                                       residuals[:,2]]))
    #np.transpose(np.vstack((residuals[:,0],residuals[:,1],
                                        #residuals[:,2])))
    #offset = np.transpose(np.array([residuals[:,3],residuals[:,4],
                                   #residuals[:,5]]))
    offset = np.transpose(np.vstack((residuals[:,3],residuals[:,4],
                            residuals[:,5])))
    print(offset)
    chi2 = residuals[:,6]
    nit = residuals[:,7]
    success = residuals[:,8]
    chi2 = residuals[:,6]
    nit = residuals[:,7]
    success = residuals[:,8]


    if save:
        speckles = {'r_true':r_true,
                    'angle_range': angle_range,
                    'f_true':f_true,
                    'r_simplex':residuals[:,0],
                    'theta_simplex':residuals[:,1],
                    'f_simplex':residuals[:,2],
                    'offset': offset,
                    'chi2': chi2,
                    'nit': nit,
                    'success': success}
        
        if output is None:
            output = 'speckles_noise_result'

        from pickle import Pickler
        with open(output,'wb') as fileSave:
            myPickler = Pickler(fileSave)
            myPickler.dump(speckles)


    # Calculate 1 sigma of distribution of deviations
    print(offset.shape)
    if force_rPA:
        offset = offset[:,2]
        print(offset.shape)
    mean_dev, sp_unc = confidence(offset, cfd=68.27, bins=min(100,int(offset.shape[0]/2)), 
                                  gaussian_fit=True, verbose=True, save=False, 
                                  output_dir='')
    if plot:
        plt.show()

    if full_output:
        return sp_unc, mean_dev, p_simplex, offset, chi2, nit, success
    else:
        return sp_unc


def _estimate_speckle_one_angle(angle, cube_pf, psfn, angs, r_true, f_true, 
                                plsc, fwhm, aperture_radius, cube_ref, fmerit, 
                                algo, algo_options, transmission, mu_sigma, 
                                weights, force_rPA, simplex_options, 
                                verbose=True):
                         
    if verbose:
        print('Process is running for angle: {:.2f}'.format(angle))

    cube_fc = cube_inject_companions(cube_pf, psfn, angs, flevel=f_true, 
                                     plsc=plsc, rad_dists=[r_true], 
                                     n_branches=1, theta=angle, 
                                     transmission=transmission, verbose=False)
    
    ncomp = algo_options.get('ncomp', None)
    annulus_width = algo_options.get('annulus_width', int(fwhm))
    
    res_simplex = firstguess_simplex((r_true,angle,f_true), cube_fc, angs, psfn, 
                                     plsc, ncomp, fwhm, annulus_width, 
                                     aperture_radius, cube_ref=cube_ref,
                                     fmerit=fmerit, algo=algo, 
                                     algo_options=algo_options,
                                     transmission=transmission, 
                                     mu_sigma=mu_sigma, weights=weights, 
                                     force_rPA=force_rPA, 
                                     options=simplex_options, 
                                     verbose=False)

    if force_rPA:
        simplex_res_f, = res_simplex.x
        simplex_res_r, simplex_res_PA = r_true, angle
    else:
        simplex_res_r, simplex_res_PA, simplex_res_f = res_simplex.x
    offset_r = simplex_res_r - r_true
    offset_PA = simplex_res_PA - angle
    offset_f = simplex_res_f - f_true       
    chi2 = res_simplex.fun
    nit = res_simplex.nit
    success = res_simplex.success    
    
    return (simplex_res_r, simplex_res_PA, simplex_res_f, offset_r, offset_PA, 
            offset_f, chi2, nit, success)