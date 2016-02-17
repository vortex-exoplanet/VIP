#! /usr/bin/env python

"""
Module with simplex (Nelder-Mead) optimization for defining the flux and 
position of a companion using the Negative Fake Companion.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from .func_merit import chisquare 
from ..var import frame_center
from ..phot import psf_norm

__all__ = ['firstguess_simplex',
           'firstguess_from_coord',
           'firstguess']


def firstguess_simplex(p, cube, angs, psf, plsc, ncomp, fwhm, annulus_width, 
                       aperture_radius, cube_ref=None, svd_mode='lapack', 
                       scaling='temp-mean', p_ini=None, options=None, 
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
    ncomp: int
        The number of principal components.  
    fwhm : float
        The FHWM in pixels.   
    annulus_width: int, optional
        The width in terms of the FWHM of the annulus on which the PCA is done.       
    aperture_radius: int, optional
        The radius of the circular aperture in terms of the FWHM.
    cube_ref : array_like, 3d, optional
        Reference library cube. For Reference Star Differential Imaging.
    svd_mode : {'lapack', 'randsvd', 'eigen', 'arpack'}, str optional
        Switch for different ways of computing the SVD and selected PCs.
    scaling : {'temp-mean', 'temp-standard'} or None, optional
        With None, no scaling is performed on the input data before SVD. With 
        "temp-mean" then temporal px-wise mean subtraction is done and with 
        "temp-standard" temporal mean centering plus scaling to unit variance 
        is done. 
    p_ini : np.array
        Position (r, theta) of the circular aperture center.
    options: dict, optional
        The scipy.optimize.minimize options.
    verbose : boolean, optional
        If True, informations are displayed in the shell.
        
    Returns
    -------
    out : scipy.optimize.minimize solution object
        The solution of the minimization algorithm.
        
    """    
    if verbose:
        print ''
        print '{} minimization is running...'.format(options.get('method','Nelder-Mead'))
     
    if p_ini is None:
        p_ini = p
        
    # We crop the PSF and check if PSF has been normalized (so that flux in 
    # 1*FWHM aperture = 1) and fix if needed
    psf = psf_norm(psf, size=3*fwhm, fwhm=fwhm)
        
    solu = minimize(chisquare, p, args=(cube,angs,plsc,psf,fwhm,annulus_width,
                                        aperture_radius,p_ini,ncomp,cube_ref,
                                        svd_mode,scaling), 
                    method = options.pop('method','Nelder-Mead'), 
                    options=options, **kwargs)                       

    if verbose:  print(solu)
    return solu
    
        
def firstguess_from_coord(planet, center, cube, angs, PLSC, psf, 
                          fwhm, annulus_width, aperture_radius, ncomp, 
                          cube_ref=None, svd_mode='lapack', scaling='temp-mean', 
                          f_range=None, display=False, verbose=True, save=False, 
                          **kwargs):
    """
    Determine a first guess for the flux of a companion at a given position 
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
        The width in terms of the FWHM of the annulus on which the PCA is done.       
    aperture_radius: int, optional
        The radius of the circular aperture in terms of the FWHM.
    ncomp: int
        The number of principal components. 
    cube_ref : array_like, 3d, optional
        Reference library cube. For Reference Star Differential Imaging.
    svd_mode : {'lapack', 'randsvd', 'eigen', 'arpack'}, str optional
        Switch for different ways of computing the SVD and selected PCs.
    scaling : {'temp-mean', 'temp-standard'} or None, optional
        With None, no scaling is performed on the input data before SVD. With 
        "temp-mean" then temporal px-wise mean subtraction is done and with 
        "temp-standard" temporal mean centering plus scaling to unit variance 
        is done. 
    f_range: numpy.array, optional
        The range of flux tested values. If None, 20 values between 0 and 5000
        are tested.
    display: boolean, optional
        If True, the figure chi2 vs. flux is displayed.
    verbose: boolean
        If True, display intermediate info in the shell.        
    save: boolean, optional
        If True, the figure chi2 vs. flux is saved. 
    kwargs: dict, optional
        Additional parameters are passed to the matplotlib plot method.
        
    Returns
    -------
    out : numpy.array
        The radial coordinates and the flux of the companion.
                
    """
    
    xy = planet-center
    r0= np.sqrt(xy[0]**2+xy[1]**2)
    theta0 = np.mod(np.arctan2(xy[1],xy[0])/np.pi*180,360) 

    # We crop the PSF and check if PSF has been normalized (so that flux in 
    # 1*FWHM aperture = 1) and fix if needed
    psf = psf_norm(psf, size=3*fwhm, fwhm=fwhm)

    if f_range is not None:    
        n = f_range.shape[0]
    else:
        n = 20
        f_range =  np.linspace(0,5000,n)
    
    chi2r = np.zeros(n)
    if verbose:
        print 'Step | flux    | chi2r'
    for j, f_guess in enumerate(f_range):
        chi2r[j] = chisquare((r0,theta0,f_guess), cube, angs, PLSC, psf, 
                             fwhm, annulus_width, aperture_radius,(r0,theta0),
                             ncomp, cube_ref=cube_ref, svd_mode=svd_mode, 
                             scaling=scaling)
        if verbose:
            print '{}/{}   {:.3f}   {:.3f}'.format(j+1,n,f_guess,chi2r[j])
         
    
    f0 = f_range[chi2r.argmin()]    
    
    if display:
        plt.figure(figsize=kwargs.pop('figsize',(8,4)))
        plt.title(kwargs.pop('title',''))
        plt.xlim(kwargs.pop('xlim',[0,f_range.max()]))
        plt.ylim(kwargs.pop('ylim',[chi2r.min()*0.9,chi2r.max()*1.1]))

        plt.plot(f_range,chi2r,linestyle = kwargs.pop('linestyle','-'),
                               color = kwargs.pop('color','r'),
                               marker = kwargs.pop('marker','s'),
                               markerfacecolor=kwargs.pop('markerfacecolor','r'),
                               markeredgecolor=kwargs.pop('markeredgecolor','r'),
                               **kwargs)
                       
        plt.xlabel('flux')
        plt.ylabel(r'$\chi^2_{r}$')
        plt.grid('on')
    if save:
        plt.savefig('chi2rVSflux.pdf')
    if display:
        plt.show()

    return (r0,theta0,f0)       



def firstguess(cube, angs, psfn, ncomp, plsc, planets_xy_coord, fwhm=4, 
               annulus_width=3, aperture_radius=4, cube_ref=None, 
               svd_mode='lapack', scaling='temp-mean', p_ini=None, f_range=None, 
               simplex=False, simplex_options=None, display=False, verbose=True, 
               save=False, figure_options=None):
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
    estimate the first guess based on the preliminary guess. This step may take
    some time !
           
    Parameters
    ----------
    cube: numpy.array
        The cube of fits images expressed as a numpy.array. 
    angs: numpy.array
        The parallactic angle fits image expressed as a numpy.array.  
    psfn: numpy.array
        The scaled psf expressed as a numpy.array.   
    ncomp: int
        The number of principal components.         
    plsc: float
        The platescale, in arcsec per pixel.  
    planet_xy_coord: array or list
        The list of (x,y) positions of the planets.
    fwhm : float, optional
        The FHWM in pixels.
    annulus_width: int, optional
        The width in terms of the FWHM of the annulus on which the PCA is done.       
    aperture_radius: int, optional
        The radius of the circular aperture in terms of the FWHM.
    cube_ref : array_like, 3d, optional
        Reference library cube. For Reference Star Differential Imaging.
    svd_mode : {'lapack', 'randsvd', 'eigen', 'arpack'}, str optional
        Switch for different ways of computing the SVD and selected PCs.
    scaling : {'temp-mean', 'temp-standard'} or None, optional
        With None, no scaling is performed on the input data before SVD. With 
        "temp-mean" then temporal px-wise mean subtraction is done and with 
        "temp-standard" temporal mean centering plus scaling to unit variance 
        is done. 
    p_ini: numpy.array
        Position (r, theta) of the circular aperture center.        
    f_range: numpy.array, optional
        The range of flux tested values. If None, 20 values between 0 and 5000
        are tested.
    simplex: boolean, optional
        If True, the Nelder-Mead minimization is performed after the flux grid
        search.
    simplex_options: dict, optional
        The scipy.optimize.minimize options.
    display: boolean, optional
        If True, the figure chi2 vs. flux is displayed.
    verbose: boolean
        If True, display intermediate info in the shell.        
    save: boolean, optional
        If True, the figure chi2 vs. flux is saved. 
    figure_options: dict, optional
        Additional parameters are passed to the matplotlib plot method.    

    Returns
    -------
    out : The radial coordinates and the flux of the companion.

    """
    planets_xy_coord = np.array(planets_xy_coord)
    n_planet = planets_xy_coord.shape[0]

    center_xy_coord = np.array([cube.shape[1]/2.,cube.shape[2]/2.])    

    # We crop the PSF and check if PSF has been normalized (so that flux in 
    # 1*FWHM aperture = 1) and fix if needed
    psfn = psf_norm(psfn, size=3*fwhm, fwhm=fwhm)

    if figure_options is None:
        figure_options = {'color':'b', 'marker':'o', 
                          'xlim': [f_range[0]-10,f_range[-1]+10], 
                          'title':r'$\chi^2_{r}$ vs flux'}
    if f_range is None:  
        f_range = np.linspace(0,2000,20)
    if simplex_options is None:  
        simplex_options = {'xtol':1e-1, 'maxiter':500, 'maxfev':1000}
        
    
    r_0 = np.zeros(n_planet)
    theta_0 = np.zeros_like(r_0)
    f_0 = np.zeros_like(r_0)
    
    for index_planet in range(n_planet):    
        if verbose:
            print ''
            print '************************************************************'
            print '             Planet {}           '.format(index_planet)
            print '************************************************************'
            print ''
            msg2 = 'Planet {}: flux estimation at the position [{},{}], running ...'
            print msg2.format(index_planet,planets_xy_coord[index_planet,0],
                              planets_xy_coord[index_planet,1])
        
        r_pre,theta_pre,f_pre = firstguess_from_coord(planets_xy_coord[index_planet],
                                            center_xy_coord, cube, angs, plsc,
                                            psfn, fwhm, annulus_width,  
                                            aperture_radius, ncomp, 
                                            f_range=f_range, cube_ref=cube_ref, 
                                            svd_mode=svd_mode, scaling=scaling, 
                                            display=display, verbose=verbose, 
                                            save=save, **figure_options)
                                                                                                                    
        if verbose:
            msg3 = 'Planet {}: preliminary guess: (r,theta,f)=({:.1f}, {:.1f}, {:.1f})'
            print msg3.format(index_planet,r_pre, theta_pre, f_pre)
        
        if simplex:
            if verbose:
                msg4 = 'Planet {}: Simplex Nelder-Mead minimization, running ...'
                print msg4.format(index_planet)
                                                         
            res = firstguess_simplex((r_pre,theta_pre,f_pre), cube, angs, psfn,
                                     plsc, ncomp, fwhm, annulus_width, 
                                     aperture_radius, cube_ref=cube_ref, 
                                     svd_mode=svd_mode, scaling=scaling, 
                                     p_ini=p_ini, options=simplex_options, 
                                     verbose=False)
            
            r_0[index_planet], theta_0[index_planet], f_0[index_planet] = res.x
            if verbose:
                msg5 = 'Planet {}: Success: {}, nit: {}, nfev: {}, chi2r: {}'
                print msg5.format(index_planet,res.success,res.nit,res.nfev, 
                                  res.fun)
                print 'message: {}'.format(res.message)
            
        else:
            if verbose:
                msg4bis = 'Planet {}: Simplex Nelder-Mead minimization skipped.'
                print msg4bis.format(index_planet)            
            r_0[index_planet] = r_pre
            theta_0[index_planet] = theta_pre
            f_0[index_planet] = f_pre                               

        if verbose:            
            centy, centx = frame_center(cube[0])
            posy = r_0 * np.sin(np.deg2rad(theta_0[index_planet])) + centy
            posx = r_0 * np.cos(np.deg2rad(theta_0[index_planet])) + centx
            msg6 = 'Planet {}: simplex guess: (r_0,theta_0,f_0)=({:.3f}, {:.3f}'
            msg6 += ', {:.3f}) at (X,Y)=({:.2f}, {:.2f})'
            print msg6.format(index_planet, r_0[index_planet],
                              theta_0[index_planet], f_0[index_planet], posx[0], posy[0])
    
    if verbose:
        print ''
        print '************************************************************'
        print 'DONE !'
        print '************************************************************'
        
    return (r_0,theta_0,f_0)


        