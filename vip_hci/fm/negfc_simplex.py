#! /usr/bin/env python

"""
Module with simplex (Nelder-Mead) optimization for defining the flux and 
position of a companion using the Negative Fake Companion.
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from .negfc_fmerit import chisquare, get_mu_and_sigma
from ..psfsub import pca_annulus
from ..var import frame_center
from ..config import time_ini, timing
from ..config.utils_conf import sep


__author__ = 'O. Wertz, C. A. Gomez Gonzalez, V. Christiaens'
__all__ = ['firstguess']


def firstguess_from_coord(planet, center, cube, angs, PLSC, psf, fwhm,
                          annulus_width, aperture_radius, ncomp, cube_ref=None,
                          svd_mode='lapack', scaling=None, fmerit='sum',
                          imlib='vip-fft', interpolation='lanczos4',
                          collapse='median', algo=pca_annulus, delta_rot=1, 
                          algo_options={}, f_range=None, transmission=None, 
                          mu_sigma=None, weights=None, plot=False, 
                          verbose=True, save=False, debug=False):
    """ Determine a first guess for the flux of a companion at a given position
    in the cube by doing a simple grid search evaluating the reduced chi2.
    
    Parameters
    ----------
    planet: numpy.array
        The (x,y) position of the planet in the pca processed cube.
    center: numpy.array
        The (x,y) position of the cube center.
    cube: numpy.array
        The cube of fits images expressed as a numpy.array. 
    angs: numpy.array
        The parallactic angle fits image expressed as a numpy.array.         
    PLSC: float
        The platescale, in arcsec per pixel.
    psf: numpy.array
        The scaled psf expressed as a numpy.array. 
    fwhm : float
        The FHWM in pixels.           
    annulus_width: int, optional
        The width in pixels of the annulus on which the PCA is done.       
    aperture_radius: int, optional
        The radius of the circular aperture in terms of the FWHM.
    ncomp: int
        The number of principal components. 
    cube_ref : numpy ndarray, 3d, optional
        Reference library cube. For Reference Star Differential Imaging.
    svd_mode : {'lapack', 'randsvd', 'eigen', 'arpack'}, str optional
        Switch for different ways of computing the SVD and selected PCs.
    scaling : {'temp-mean', 'temp-standard'} or None, optional
        With None, no scaling is performed on the input data before SVD. With 
        "temp-mean" then temporal px-wise mean subtraction is done and with 
        "temp-standard" temporal mean centering plus scaling to unit variance 
        is done. 
    fmerit : {'sum', 'stddev'}, string optional
        Figure of merit to be used, if mu_sigma is set to None.
    imlib : str, optional
        See the documentation of the ``vip_hci.preproc.frame_rotate`` function.
    interpolation : str, optional
        See the documentation of the ``vip_hci.preproc.frame_rotate`` function.
    collapse : {'median', 'mean', 'sum', 'trimmean', None}, str or None, optional
        Sets the way of collapsing the frames for producing a final image. If
        None then the cube of residuals is used when measuring the function of
        merit (instead of a single final frame).
    algo: python routine, opt {pca_annulus, pca_annular, pca, custom}
        Routine to be used to model and subtract the stellar PSF. From an input
        cube, derotation angles, and optional arguments, it should return a 
        post-processed frame.
    delta_rot: float, optional
        If algo is set to pca_annular, delta_rot is the angular threshold used
        to select frames in the PCA library (see description of pca_annular).
    algo_options: dict, opt
        Dictionary with additional parameters for the pca algorithm (e.g. tol,
        min_frames_lib, max_frames_lib). Note: arguments such as svd_mode,
        scaling imlib, interpolation or collapse can also be included in this
        dict (the latter are also kept as function arguments for compatibility
        with older versions of vip). 
    f_range: numpy.array, optional
        The range of flux tested values. If None, 20 values between 0 and 5000
        are tested.
    transmission: numpy array, optional
        Array with 2 columns. First column is the radial separation in pixels. 
        Second column is the off-axis transmission (between 0 and 1) at the 
        radial separation given in column 1.
    mu_sigma: tuple of 2 floats or None, opt
        If set to None: not used, and falls back to original version of the 
        algorithm, using fmerit. Otherwise, should be a tuple of 2 elements,
        containing the mean and standard deviation of pixel intensities in an 
        annulus centered on the location of the companion, excluding the area 
        directly adjacent to the companion.
    weights : 1d array, optional
        If provided, the negative fake companion fluxes will be scaled according
        to these weights before injection in the cube. Can reflect changes in 
        the observing conditions throughout the sequence.
    plot: boolean, optional
        If True, the figure chi2 vs. flux is displayed.
    verbose: boolean
        If True, display intermediate info in the shell.
    save: boolean, optional
        If True, the figure chi2 vs. flux is saved as .pdf if plot is also True
        
    Returns
    -------
    out : numpy.array
        The radial coordinates and the flux of the companion.
                
    """  
    xy = planet-center
    r0 = np.sqrt(xy[0]**2 + xy[1]**2)
    theta0 = np.mod(np.arctan2(xy[1], xy[0]) / np.pi*180, 360)

    if f_range is not None:    
        n = f_range.shape[0]
    else:
        n = 30
        f_range = np.geomspace(1e-1, 1e4, n)
    
    chi2r = []
    if verbose:
        print('Step | flux    | chi2r')
        
    counter = 0
    for j, f_guess in enumerate(f_range):
        chi2r.append(chisquare((r0, theta0, f_guess), cube, angs, PLSC, psf,
                               fwhm, annulus_width, aperture_radius,
                               (r0, theta0), ncomp, cube_ref, svd_mode,
                               scaling, fmerit, collapse, algo, delta_rot,
                               imlib, interpolation, algo_options, transmission, 
                               mu_sigma, weights, debug))
        if chi2r[j] > chi2r[j-1]:
            counter += 1
        if counter == 4:
            break
        if verbose:
            print('{}/{}   {:.3f}   {:.3f}'.format(j+1, n, f_guess, chi2r[j]))

    chi2r = np.array(chi2r)
    f0 = f_range[chi2r.argmin()]  

    if plot:
        plt.figure(figsize=(8, 4))
        plt.title('$\chi^2_{r}$ vs flux')
        plt.xlim(f_range[0], f_range[:chi2r.shape[0]].max())
        plt.ylim(chi2r.min()*0.9, chi2r.max()*1.1)
        plt.plot(f_range[:chi2r.shape[0]], chi2r, linestyle='-', color='gray',
                 marker='.', markerfacecolor='r', markeredgecolor='r')
        plt.xlabel('flux')
        plt.ylabel(r'$\chi^2_{r}$')
        plt.grid('on')
    if save:
        plt.savefig('chi2rVSflux.pdf')
    if plot:
        plt.show()

    return r0, theta0, f0


def firstguess_simplex(p, cube, angs, psf, plsc, ncomp, fwhm, annulus_width, 
                       aperture_radius, cube_ref=None, svd_mode='lapack', 
                       scaling=None, fmerit='sum', imlib='vip-fft',
                       interpolation='lanczos4', collapse='median', 
                       algo=pca_annulus, delta_rot=1, algo_options={}, 
                       p_ini=None, transmission=None, mu_sigma=None, 
                       weights=None, force_rPA=False, options=None, 
                       verbose=False, **kwargs):
    """
    Determine the position of a companion using the negative fake companion 
    technique and a standard minimization algorithm (Default=Nelder-Mead) .
    
    Parameters
    ----------
    
    p : np.array
        Estimate of the candidate position.
    cube: numpy.array
        The cube of fits images expressed as a numpy.array. 
    angs: numpy.array
        The parallactic angle fits image expressed as a numpy.array. 
    psf: numpy.array
        The scaled psf expressed as a numpy.array.        
    plsc: float
        The platescale, in arcsec per pixel.
    ncomp: int or None
        The number of principal components.  
    fwhm : float
        The FHWM in pixels.   
    annulus_width: int, optional
        The width in pixels of the annulus on which the PCA is done.       
    aperture_radius: int, optional
        The radius of the circular aperture in terms of the FWHM.
    cube_ref : numpy ndarray, 3d, optional
        Reference library cube. For Reference Star Differential Imaging.
    svd_mode : {'lapack', 'randsvd', 'eigen', 'arpack'}, str optional
        Switch for different ways of computing the SVD and selected PCs.
    scaling : {'temp-mean', 'temp-standard'} or None, optional
        With None, no scaling is performed on the input data before SVD. With 
        "temp-mean" then temporal px-wise mean subtraction is done and with 
        "temp-standard" temporal mean centering plus scaling to unit variance 
        is done. 
    fmerit : {'sum', 'stddev'}, string optional
        Figure of merit to be used, if mu_sigma is set to None.
    imlib : str, optional
        See the documentation of the ``vip_hci.preproc.frame_rotate`` function.
    interpolation : str, optional
        See the documentation of the ``vip_hci.preproc.frame_rotate`` function.
    collapse : {'median', 'mean', 'sum', 'trimmean', None}, str or None, optional
        Sets the way of collapsing the frames for producing a final image. If
        None then the cube of residuals is used when measuring the function of
        merit (instead of a single final frame).
    algo: python routine, opt {pca_annulus, pca_annular, pca, custom}
        Routine to be used to model and subtract the stellar PSF. From an input
        cube, derotation angles, and optional arguments, it should return a 
        post-processed frame.
    delta_rot: float, optional
        If algo is set to pca_annular, delta_rot is the angular threshold used
        to select frames in the PCA library (see description of pca_annular).
    algo_options: dict, opt
        Dictionary with additional parameters for the pca algorithm (e.g. tol,
        min_frames_lib, max_frames_lib). Note: arguments such as svd_mode,
        scaling imlib, interpolation or collapse can also be included in this
        dict (the latter are also kept as function arguments for compatibility
        with older versions of vip). 
    p_ini : np.array
        Position (r, theta) of the circular aperture center.
    transmission: numpy array, optional
        Array with 2 columns. First column is the radial separation in pixels. 
        Second column is the off-axis transmission (between 0 and 1) at the 
        radial separation given in column 1.
    mu_sigma: tuple of 2 floats or None, opt
        If set to None: not used, and falls back to original version of the 
        algorithm, using fmerit. Otherwise, should be a tuple of 2 elements,
        containing the mean and standard deviation of pixel intensities in an 
        annulus centered on the location of the companion, excluding the area 
        directly adjacent to the companion.
    weights : 1d array, optional
        If provided, the negative fake companion fluxes will be scaled according
        to these weights before injection in the cube. Can reflect changes in 
        the observing conditions throughout the sequence.
    force_rPA: bool, optional
        Whether to only search for optimal flux, provided (r,PA).
    options: dict, optional
        The scipy.optimize.minimize options.
    verbose : boolean, optional
        If True, additional information is printed out.
    **kwargs: optional
        Optional arguments to the scipy.optimize.minimize function
        
    Returns
    -------
    out : scipy.optimize.minimize solution object
        The solution of the minimization algorithm.
        
    """    
    if verbose:
        print('\nNelder-Mead minimization is running...')
     
    if p_ini is None:
        p_ini = p

    if force_rPA:
        p_t = (p[-1],)
        p_ini = (p[0],p[1])
    else:
        p_t = p
    solu = minimize(chisquare, p_t, args=(cube, angs, plsc, psf, fwhm,
                                          annulus_width, aperture_radius, p_ini,
                                          ncomp, cube_ref, svd_mode, scaling,
                                          fmerit, collapse, algo, delta_rot, 
                                          imlib, interpolation, algo_options, 
                                          transmission, mu_sigma, weights, 
                                          force_rPA),
                    method='Nelder-Mead', options=options, **kwargs)

    if verbose:
        print(solu)
    return solu
    

def firstguess(cube, angs, psfn, ncomp, plsc, planets_xy_coord, fwhm=4, 
               annulus_width=4, aperture_radius=1, cube_ref=None, 
               svd_mode='lapack', scaling=None, fmerit='sum', imlib='vip-fft',
               interpolation='lanczos4', collapse='median', algo=pca_annulus,
               delta_rot=1, p_ini=None, f_range=None, transmission=None, 
               mu_sigma=None, wedge=None, weights=None, force_rPA= False, 
               algo_options={}, simplex=True, simplex_options=None, plot=False, 
               verbose=True, save=False):
    """ Determines a first guess for the position and the flux of a planet.
        
    We process the cube without injecting any negative fake companion. 
    This leads to the visual detection of the planet(s). For each of them,
    one can estimate the (x,y) coordinates in pixel for the position of the 
    star, as well as the planet(s). 

    From the (x,y) coordinates in pixels for the star and planet(s), we can 
    estimate a preliminary guess for the position and flux for each planet
    by using the method "firstguess_from_coord". The argument "f_range" allows
    to indicate prior limits for the flux (optional, default: None). 
    This step can be reiterate to refine the preliminary guess for the flux.

    We can go a step further by using a Simplex Nelder_Mead minimization to
    estimate the first guess based on the preliminary guess.
           
    Parameters
    ----------
    cube: numpy.array
        The cube of fits images expressed as a numpy.array. 
    angs: numpy.array
        The parallactic angle fits image expressed as a numpy.array.  
    psfn: numpy.array
        The centered and normalized (flux in a 1*FWHM aperture must equal 1) 
        PSF 2d-array.
    ncomp: int
        The number of principal components.         
    plsc: float
        The platescale, in arcsec per pixel.  
    planets_xy_coord: array or list
        The list of (x,y) positions of the planets.
    fwhm : float, optional
        The FHWM in pixels.
    annulus_width: int, optional
        The width in pixels of the annulus on which the PCA is done.       
    aperture_radius: int, optional
        The radius of the circular aperture in terms of the FWHM.
    cube_ref : numpy ndarray, 3d, optional
        Reference library cube. For Reference Star Differential Imaging.
    svd_mode : {'lapack', 'randsvd', 'eigen', 'arpack'}, str optional
        Switch for different ways of computing the SVD and selected PCs.
    scaling : {'temp-mean', 'temp-standard'} or None, optional
        With None, no scaling is performed on the input data before SVD. With 
        "temp-mean" then temporal px-wise mean subtraction is done and with 
        "temp-standard" temporal mean centering plus scaling to unit variance 
        is done. 
    fmerit : {'sum', 'stddev'}, string optional
        Figure of merit to be used, if mu_sigma is set to None.
    imlib : str, optional
        See the documentation of the ``vip_hci.preproc.frame_rotate`` function.
    interpolation : str, optional
        See the documentation of the ``vip_hci.preproc.frame_rotate`` function.
    collapse : {'median', 'mean', 'sum', 'trimmean', None}, str or None, opt
        Sets the way of collapsing the frames for producing a final image. If
        None then the cube of residuals is used when measuring the function of
        merit (instead of a single final frame).
    algo: python routine, opt {pca_annulus, pca_annular, pca, custom}
        Routine to be used to model and subtract the stellar PSF. From an input
        cube, derotation angles, and optional arguments, it should return a 
        post-processed frame.
    delta_rot: float, optional
        If algo is set to pca_annular, delta_rot is the angular threshold used
        to select frames in the PCA library (see description of pca_annular).
    p_ini: numpy.array
        Position (r, theta) of the circular aperture center.            
    f_range: numpy.array, optional
        The range of flux tested values. If None, 20 values between 0 and 5000
        are tested.
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
    wedge: tuple, opt
        Range in theta where the mean and standard deviation are computed in an 
        annulus defined in the PCA image. If None, it will be calculated 
        automatically based on initial guess and derotation angles to avoid.
        If some disc signal is present elsewhere in the annulus, it is 
        recommended to provide wedge manually. The provided range should be 
        continuous and >0. E.g. provide (270, 370) to consider a PA range 
        between [-90,+10].
    weights : 1d array, optional
        If provided, the negative fake companion fluxes will be scaled according
        to these weights before injection in the cube. Can reflect changes in 
        the observing conditions throughout the sequence.
    force_rPA: bool, optional
        Whether to only search for optimal flux, provided (r,PA).
    algo_options: dict, opt
        Dictionary with additional parameters for the pca algorithm (e.g. tol,
        min_frames_lib, max_frames_lib). Note: arguments such as svd_mode,
        scaling imlib, interpolation or collapse can also be included in this
        dict (the latter are also kept as function arguments for compatibility
        with older versions of vip). 
    simplex: bool, optional
        If True, the Nelder-Mead minimization is performed after the flux grid
        search.
    simplex_options: dict, optional
        The scipy.optimize.minimize options.
    plot: boolean, optional
        If True, the figure chi2 vs. flux is displayed.
    verbose: bool, optional
        If True, display intermediate info in the shell.
    save: bool, optional
        If True, the figure chi2 vs. flux is saved.

    Returns
    -------
    out : The radial coordinates and the flux of the companion.

    Notes
    -----
    Polar angle is not the conventional NORTH-TO-EAST P.A.
    """
    if verbose:
        start_time = time_ini()
        
    planets_xy_coord = np.array(planets_xy_coord)
    n_planet = planets_xy_coord.shape[0]
    center_xy_coord = np.array(frame_center(cube[0]))

    r_0 = np.zeros(n_planet)
    theta_0 = np.zeros_like(r_0)
    f_0 = np.zeros_like(r_0)

    if weights is not None:
        if not len(weights)==cube.shape[0]:
            raise TypeError("Weights should have same length as cube axis 0")
        norm_weights = weights/np.sum(weights)
    else:
        norm_weights=weights
    
    for index_planet in range(n_planet):    
        if verbose:
            print('\n'+sep)
            print('             Planet {}           '.format(index_planet))
            print(sep+'\n')
            msg2 = 'Planet {}: flux estimation at the position [{},{}], '
            msg2 += 'running ...'
            print(msg2.format(index_planet, planets_xy_coord[index_planet, 0],
                              planets_xy_coord[index_planet, 1]))
        # Measure mu and sigma once in the annulus (instead of each MCMC step)
        if isinstance(mu_sigma,tuple):
            if len(mu_sigma)!=2:
                raise TypeError("If a tuple, mu_sigma must have 2 elements")
        elif mu_sigma is not None:
            xy = planets_xy_coord[index_planet]-center_xy_coord
            r0 = np.sqrt(xy[0]**2 + xy[1]**2)
            theta0 = np.mod(np.arctan2(xy[1], xy[0]) / np.pi*180, 360)
            mu_sigma = get_mu_and_sigma(cube, angs, ncomp, annulus_width, 
                                        aperture_radius, fwhm, r0, 
                                        theta0, cube_ref=cube_ref, 
                                        wedge=wedge, svd_mode=svd_mode, 
                                        scaling=scaling, algo=algo, 
                                        delta_rot=delta_rot, imlib=imlib, 
                                        interpolation=interpolation,
                                        collapse=collapse, weights=norm_weights, 
                                        algo_options=algo_options)
        
        res_init = firstguess_from_coord(planets_xy_coord[index_planet],
                                         center_xy_coord, cube, angs, plsc,
                                         psfn, fwhm, annulus_width,
                                         aperture_radius, ncomp,
                                         f_range=f_range, cube_ref=cube_ref,
                                         svd_mode=svd_mode, scaling=scaling,
                                         fmerit=fmerit, imlib=imlib,
                                         collapse=collapse, algo=algo, 
                                         delta_rot=delta_rot,
                                         interpolation=interpolation, 
                                         algo_options=algo_options, 
                                         transmission=transmission, 
                                         mu_sigma=mu_sigma, weights=weights,
                                         plot=plot, verbose=verbose, save=save)
        r_pre, theta_pre, f_pre = res_init

        if verbose:
            msg3 = 'Planet {}: preliminary guess: (r, theta, f)=({:.1f}, '
            msg3 += '{:.1f}, {:.1f})'
            print(msg3.format(index_planet,r_pre, theta_pre, f_pre))
        
        if simplex or force_rPA:
            if verbose:
                msg4 = 'Planet {}: Simplex Nelder-Mead minimization, '
                msg4 += 'running ...'
                print(msg4.format(index_planet))

            if simplex_options is None:
                simplex_options = {'xatol': 1e-6, 'fatol': 1e-6, 'maxiter': 800,
                                   'maxfev': 2000}
                                                         
            res = firstguess_simplex((r_pre, theta_pre, f_pre), cube, angs,
                                     psfn, plsc, ncomp, fwhm, annulus_width,
                                     aperture_radius, cube_ref=cube_ref, 
                                     svd_mode=svd_mode, scaling=scaling,
                                     fmerit=fmerit, imlib=imlib,
                                     interpolation=interpolation,
                                     collapse=collapse, algo=algo, 
                                     delta_rot=delta_rot, algo_options=algo_options, 
                                     p_ini=p_ini, transmission=transmission,
                                     mu_sigma=mu_sigma, weights=weights, 
                                     force_rPA=force_rPA, 
                                     options=simplex_options, verbose=False)
            if force_rPA:
                r_0[index_planet], theta_0[index_planet] = (r_pre, theta_pre)
                f_0[index_planet], = res.x
            else:
                r_0[index_planet], theta_0[index_planet], f_0[index_planet] = res.x
            if verbose:
                msg5 = 'Planet {}: Success: {}, nit: {}, nfev: {}, chi2r: {}'
                print(msg5.format(index_planet, res.success, res.nit, res.nfev,
                                  res.fun))
                print('message: {}'.format(res.message))
            
        else:
            if verbose:
                msg4bis = 'Planet {}: Simplex Nelder-Mead minimization skipped.'
                print(msg4bis.format(index_planet))            
            r_0[index_planet] = r_pre
            theta_0[index_planet] = theta_pre
            f_0[index_planet] = f_pre                               

        if verbose:            
            centy, centx = frame_center(cube[0])
            posy = r_0 * np.sin(np.deg2rad(theta_0[index_planet])) + centy
            posx = r_0 * np.cos(np.deg2rad(theta_0[index_planet])) + centx
            msg6 = 'Planet {}: simplex result: (r, theta, f)=({:.3f}, {:.3f}'
            msg6 += ', {:.3f}) at \n          (X,Y)=({:.2f}, {:.2f})'
            print(msg6.format(index_planet, r_0[index_planet],
                              theta_0[index_planet], f_0[index_planet],
                              posx[0], posy[0]))
    
    if verbose:
        print('\n', sep, '\nDONE !\n', sep)
        timing(start_time)

    return r_0, theta_0, f_0