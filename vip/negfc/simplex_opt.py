#! /usr/bin/env python

"""
Module with simplex (Nelder-Mead) optimization for defining the flux and 
position of a companion using the Negative Fake Companion.
"""

import numpy as np
from scipy.optimize import minimize
from .func_merit import chisquare 

__all__ = ['firstguess_simplex',
           'firstguess_from_coord',
           'firstguess']


def firstguess_simplex(p, cube, angs, psf, plsc, ncomp, annulus_width, 
                       aperture_radius, cube_ref=None, svd_mode='randsvd', 
                       p_ini=None, options=None, verbose=False, **kwargs):               
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
    annulus_width: float
        The width in pixel of the annulus on wich the PCA is performed.
    aperture_radius: float
        The radius of the circular aperture.
    cube_ref : array_like, 3d, optional
        Reference library cube. For Reference Star Differential Imaging.
    svd_mode : {'randsvd', 'eigen', 'lapack', 'arpack', 'opencv'}, str optional
        Switch for different ways of computing the SVD and selected PCs.
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
        
    solu = minimize(chisquare, 
                    p, 
                    args=(cube,angs,plsc,psf,annulus_width,ncomp,
                          aperture_radius,p_ini,cube_ref,svd_mode), 
                    method = options.pop('method','Nelder-Mead'), 
                    options=options,
                    **kwargs)                       

    if verbose:
        print(solu)
        
    return solu
    
        
def firstguess_from_coord(planet, center, cube, angs, PLSC, psf_norm, 
                          annulus_width, ncomp, aperture_radius, cube_ref=None, 
                          svd_mode='randsvd',f_range=None, display=False, 
                          verbose=True, save=False, **kwargs):
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
    psf_norm: numpy.array
        The scaled psf expressed as a numpy.array.            
    annulus_width: float
        The width in pixel of the annulus on which the PCA is performed.
    ncomp: int
        The number of principal components.        
    aperture_radius: float
        The radius of the circular aperture.
    cube_ref : array_like, 3d, optional
        Reference library cube. For Reference Star Differential Imaging.
    svd_mode : {'randsvd', 'eigen', 'lapack', 'arpack', 'opencv'}, str optional
        Switch for different ways of computing the SVD and selected PCs.
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


    if f_range is not None:    
        n = f_range.shape[0]
    else:
        n = 20
        f_range =  np.linspace(0,5000,n)
    
    chi2r = np.zeros(n)
    if verbose:
        print 'Step | flux    | chi2r'
    for j, f_guess in enumerate(f_range):
        chi2r[j] = chisquare((r0,theta0,f_guess), cube, angs, PLSC, psf_norm, 
                             annulus_width, ncomp, aperture_radius,(r0,theta0),
                             cube_ref=cube_ref, svd_mode=svd_mode)
        if verbose:
            print '{}/{}   {:.3f}   {:.3f}'.format(j+1,n,f_guess,chi2r[j])
         
    
    f0 = f_range[chi2r.argmin()]    
    
    if display:
        import matplotlib.pyplot as plt
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
                       
        plt.xlabel("$f$")
        plt.ylabel("$\chi^2_{r}$")
    if save:
        plt.savefig('chi2rVSflux.pdf')
    if display:
        plt.show()

    return (r0,theta0,f0)       



def firstguess(cube, angs, psfn, ncomp, PLSC, annulus_width, aperture_radius, 
               planets_xy_coord, cube_ref=None, svd_mode='randsvd',  
               p_ini=None, f_range=None, simplex=False, simplex_options=None, 
               display=False, verbose=True, save=False, figure_options={}):
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
    PLSC: float
        The platescale, in arcsec per pixel.            
    annulus_width: float
        The width in pixel of the annulus on wich the PCA is performed.       
    aperture_radius: float
        The radius of the circular aperture.
    planet_xy_coord: numpy.array
        The (x,y) position of the planet in the pca processed frame. 
    cube_ref : array_like, 3d, optional
        Reference library cube. For Reference Star Differential Imaging.
    svd_mode : {'randsvd', 'eigen', 'lapack', 'arpack', 'opencv'}, str optional
        Switch for different ways of computing the SVD and selected PCs.
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
    n_planet = planets_xy_coord.shape[0]

    center_xy_coord = np.array([cube.shape[1]/2.,cube.shape[2]/2.])    

    if f_range is None:
        f_range = np.linspace(0,2000,20)
    
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
                                            center_xy_coord, cube, angs, PLSC,
                                            psfn, annulus_width, ncomp, 
                                            aperture_radius, f_range=f_range, 
                                            cube_ref=cube_ref, svd_mode=svd_mode,  
                                            display=display, verbose=verbose,
                                            save=save, **figure_options)
                                                                                                                    
        if verbose:
            msg3 = 'Planet {}: preliminary guess: (r,theta,f) = ({},{},{})'
            print msg3.format(index_planet,r_pre, theta_pre, f_pre)
        
        if simplex:
            if verbose:
                msg4 = 'Planet {}: Simplex Nelder-Mead minimization, running ...'
                print msg4.format(index_planet)
                                                         
            res = firstguess_simplex((r_pre,theta_pre,f_pre), cube, angs, psfn,
                                     PLSC, ncomp, annulus_width, aperture_radius,
                                     cube_ref=cube_ref, svd_mode=svd_mode,
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
            msg6 = 'Planet {}: first guess: (r_0, theta_0, f_0) = ({},{},{})'
            print msg6.format(index_planet,r_0[index_planet],
                              theta_0[index_planet],f_0[index_planet])
    
    if verbose:
        print ''
        print '************************************************************'
        print 'DONE !'
        print '************************************************************'
        
    return (r_0,theta_0,f_0)
        