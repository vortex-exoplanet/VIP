#! /usr/bin/env python

"""
Module with routines related to the estimation of the uncertainty on the 
parameters of an imaged companion due to speckle noise.
"""

__author__ = 'O. Wertz, Carlos Alberto Gomez Gonzalez, Valentin Christiaens'
__all__ = ['speckle_noise_uncertainty']

import numpy as np
from ..conf import eval_func_tuple
from ..metrics import cube_inject_companions 
from .simplex_optim import firstguess_simplex 
from .utils_negfc import cube_planet_free


def speckle_noise_uncertainty(cube, p_true, angle_range, angs, psfn, plsc, 
                              ncomp, fwhm, annulus_width, aperture_radius, 
                              simplex_options, ifs=False, output=None, 
                              save=False, verbose=True, nproc=None):
    """
    Step-by-step procedure used to determine the so-called speckles noise.

      __ 
     |   The steps 1 to 3 need to be performed for each angle.
     | 
     | - 1 - At the true planet radial distance and for a given angle, we 
     |       inject a fake companion in our free-planet cube.
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
     
    Parameters
    ----------
    cube: numpy array
        The cube (with any planets).    
    p_true: tuple (2,)
        The true position and flux associated to all injected fake companions.
        They will first be subtracted from cube, then used for test injections.
    angle_range: numpy array
        Range of angle at which the fake companions will be injected and where
        the speckle noise will be evaluated.   
    pipeline_parameters: dict
        All the required  pipeline parameters, for instance {'cube': cube, 
        'angs': angs, ...}    
    simplex_options: dict
        ALl the required simplex parameters, for instance {'tol':1e-08, 
        'max_iter':200}    
    output: str (optional)
        The name of the output file (if save is True)    
    save: boolean (optional)
        If True, the result are pickled.
    verbose: boolean (optional)
        If True, informations are displayed in the shell.
     
    """    
    
    if not nproc:   # Hyper-threading "duplicates" the cores -> cpu_count/2
        nproc = (cpu_count()/2)    
    
    if verbose:
        print('')
        print('#######################################################')
        print('###            SPECKLE NOISE DETERMINATION          ###')
        print('#######################################################')
        print('')
    
    r_true, f_true = p_true

    if angle_range[0]%360 == angle_range[-1]%360:
        angle_range = angle_range[:-1]
                  
    if verbose:              
        print('Number of steps: {}'.format(angle_range.shape[0]))
        print('')
    
    # FIRST SUBTRACT THE TRUE COMPANIONS
    cube_pf = cube_planet_free(cube)
    
    ############################
    ##  2, 3 and 4 in a loop  ##
    ############################
    ##  Parameters for the loop over the angle
#    n =  angle_range.shape[0]
#    
#    offset = np.zeros([n,3])
#    chi2 = np.zeros(n)
#    nit = np.zeros(n)
#    success = list()
#    
##    r_simplex = np.zeros(n)
##    theta_simplex = np.zeros(n)
##    f_simplex = np.zeros(n)
      
    # MULTIPROCESSING
    pool = Pool(processes=nproc)
    res = pool.map(eval_func_tuple, itt.izip(itt.repeat(_estimate_speckle_one_angle),
                                             angle_range,
                                             itt.repeat(cube_pf),
                                             itt.repeat(psfn),
                                             itt.repeat(angs),
                                             itt.repeat(r_true),
                                             itt.repeat(f_true),
                                             itt.repeat(PLSC),
                                             itt.repeat(ncomp),
                                             itt.repeat(fwhm),
                                             itt.repeat(annulus_width),
                                             itt.repeat(aperture_radius),
                                             itt.repeat(simplex_options),
                                             itt.repeat(fig_merit),
                                             itt.repeat(verbose),
                                             itt.repeat(ifs)))
    residuals = np.array(res)
    pool.close()
 
    if verbose:  
        if not ifs:
            print("residuals (offsets): ", residuals[:,3],residuals[:,4],
                  residuals[:,5])
        
            
    #r_true = np.ones_like(angle_range)*r_true
    #f_true = np.ones_like(angle_range)*f_true            
    #p_true = np.transpose(np.vstack((r_true,angle_range,f_true)))  

    if not ifs:
        p_simplex = np.transpose(np.vstack((residuals[:,0],residuals[:,1],residuals[:,2])))
        offset = np.transpose(np.vstack((residuals[:,3],residuals[:,4],residuals[:,5])))
        chi2 = residuals[:,6]
        nit = residuals[:,7]
        success = residuals[:,8]
    else:
        n_ch = cube_pf.shape[0]
        res_params = np.vstack((residuals[:,0],residuals[:,1]))
        offset = np.vstack((residuals[:,2+n_ch],residuals[:,3+n_ch]))
        for zz in range(cube_pf.shape[0]):
            res_params = np.vstack((res_params,residuals[:,2+zz]))
            offset = np.vstack((offset,residuals[:,4+n_ch+zz]))
        p_simplex = np.transpose(res_params)
        offset = np.transpose(offset)
    chi2 = residuals[:,6]
    nit = residuals[:,7]
    success = residuals[:,8]


    if save:
        if not ifs:
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
        else:
            speckles = {'r_true':r_true,
                        'angle_range': angle_range,
                        'f_true':f_true,
                        'r_simplex':residuals[:,0],
                        'theta_simplex':residuals[:,1],
                        'chi2': residuals[:,4+2*n_ch],
                        'nit': residuals[:,5+2*n_ch],
                        'success': residuals[:,6+2*n_ch]}
            for zz in range(n_ch):
                lab_f = 'f_ch{:}_simplex'.format(zz)
                speckles[lab_f] = residuals[:,2+zz]
            speckles['offset']=  offset
        
        if output is None:
            output = 'speckles_noise_result'

        from pickle import Pickler
        with open(output,'wb') as fileSave:
            myPickler = Pickler(fileSave)
            myPickler.dump(speckles)

    return p_simplex, offset, chi2, nit, success



def _estimate_speckle_one_angle(angle, cube_pf, psfn, angs, r_true, f_true, 
                                plsc, ncomp, fwhm, annulus_width, aperture_radius,
                                simplex_options, fig_merit, verbose=True, ifs=False):
                         
    if verbose:
        print('Process is running for angle: {:.2f}'.format(angle))

    if ifs:
        cube_fc = np.zeros_like(cube_pf)
        for zz in range(cube_pf.shape[0]):
            cube_fc[zz] = inject_fcs_cube(cube_pf[zz], psfn[zz], angs, flevel=f_true[zz], plsc=plsc,
                                          rad_dists=[r_true], n_branches=1, 
                                          theta=angle, verbose=False)
    else:
        cube_fc = inject_fcs_cube(cube_pf, psfn, angs, flevel=f_true, plsc=plsc,
                              rad_dists=[r_true], n_branches=1, 
                              theta=angle, verbose=False)
    ## 3 ##
    #print 'step 3: k = {}/{}, angle = {}'.format(k+1,n-1,angle)
    
    if ifs:
        res_simplex = firstguess_simplex_ifs((r_true,angle,f_true),
                             cube_fc, angs, psfn, plsc, ncomp, fwhm,
                             annulus_width, aperture_radius, 
                             options=simplex_options, 
                             verbose=False, fmerit=fig_merit)        
    else:
        res_simplex = firstguess_simplex((r_true,angle,f_true),
                             cube_fc, angs, psfn, plsc, ncomp, fwhm,
                             annulus_width, aperture_radius, 
                             options=simplex_options, 
                             verbose=False, fmerit=fig_merit)

                         
    simplex_res_r, simplex_res_PA, simplex_res_f = res_simplex.x
    offset_r = simplex_res_r - r_true
    offset_PA = simplex_res_PA - angle
    offset_f = simplex_res_f - f_true       
    chi2 = res_simplex.fun
    nit = res_simplex.nit
    success = res_simplex.success    
    
    if ifs:
        result = [simplex_res_r, simplex_res_PA]
        for zz in range(cube_pf.shape[0]):
            result.append(simplex_res_f[zz])
        result.append(offset_r)
        result.append(offset_PA)
        for zz in range(cube_pf.shape[0]):
            result.append(offset_f[zz])
        result.append(chi2)
        result.append(nit)
        result.append(success)
        results = tuple(result)
        return results  
    else:
        return simplex_res_r, simplex_res_PA, simplex_res_f, offset_r, offset_PA, offset_f, chi2, nit, success