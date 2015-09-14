#! /usr/bin/env python

"""
Module with simplex (Nelder-Mead) optimization for defining the flux and 
position of a companion using the Negative Fake Companion.
"""

import datetime
import numpy as np
from scipy.optimize import minimize
from .func_merit import chisquare 

__all__ = ['firstguess_simplex',
           'firstguess_from_coord',
           'firstguess']

<<<<<<< Updated upstream
def firstguess_simplex(p_ini, cube, angs, psf, plsc, fwhm, ncomp, annulus_width, 
                       aperture_radius, cube_ref=None, tol=1e-04, max_iter=300, 
                       timer=False, verbose=False):               
=======
<<<<<<< HEAD
def firstguess_simplex(p_ini, cube, angs, psf, plsc, ncomp, annulus_width, 
                       aperture_radius, options=None, timer=False, 
                       verbose=False):               
=======
def firstguess_simplex(p_ini, cube, angs, psf, plsc, fwhm, ncomp, annulus_width, 
                       aperture_radius, cube_ref=None, tol=1e-04, max_iter=300, 
                       timer=False, verbose=False):               
>>>>>>> origin/master
>>>>>>> Stashed changes
    """
    Determine the position of a planet using the negative fake companion 
    technique and Nelder-Mead minimization.
    
    Parameters
    ----------
    
    p_ini : np.array
        The initial position for the planet.
    cube: numpy.array
        The cube of fits images expressed as a numpy.array. 
    angs: numpy.array
        The parallactic angle fits image expressed as a numpy.array. 
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
    tol : float
        The scipy.optimize.minimize() tolerance.
    max_iter : int
        The scipy.optimize.minimize() maximum iteration.
    timer : boolean
        If True, the minimization duration is returned.
    verbose : boolean
        If True, informations are displayed in the shell.
        
    """
    #if isinstance(cube,str):
    #    if angs is None:
    #        cube, angs = open_adicube(cube)
    #    else:
    #        cube = open_fits(cube)
    #        angs = open_fits(parallactic_angle)
    #    psf = psf_norm(psf,-1)
    
    if verbose:
        print ''
        print '{} minimization is running'.format(options.get('method','Nelder-Mead'))
    

    start = datetime.datetime.now()  
    solu = minimize(chisquare, 
                    p_ini, 
                    args=(cube,angs,plsc,psf,annulus_width,ncomp,
<<<<<<< Updated upstream
=======
<<<<<<< HEAD
                          aperture_radius,p_ini), 
                    method = options.pop('method','Nelder-Mead'), 
                    options=options)                       
=======
>>>>>>> Stashed changes
                          aperture_radius,p_ini,cube_ref), 
                    method = 'Nelder-Mead', 
                    options={'xtol':tol, 'disp': True, 'maxiter': max_iter}) 
    
>>>>>>> origin/master
    
    end = datetime.datetime.now()
    elapsed_time = (end-start).total_seconds()

    if verbose:
        print("Duration (sec): {}".format(elapsed_time))
        print(solu)
        
    if timer:
        return (solu,elapsed_time)
    else:
        return solu
    
    
    
def firstguess_from_coord(planet, center, cube, angs, PLSC, psf_norm, 
                          annulus_width, ncomp, aperture_radius, f_range=None, 
                          display=False, verbose=True, save=False, **kwargs):
    """
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
                             annulus_width, ncomp, aperture_radius,(r0,theta0))
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



def firstguess(cube, angs, psfn, ncomp, PLSC, annulus_width, 
               aperture_radius, planets_xy_coord=None, f_range=None, 
               simplex=False, simplex_options=None, 
               display=False, verbose=True, save=False, 
               figure_options={}):
    """
    Determine a first guess for the position and the flux of a planet which appears
    into a cube.
        
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
    estimate the first guess based on the preliminary guess. This step may 
    some time !
           
    Parameters
    ----------
    
    """
# TODO: Au lieu de faire une minimization simplex, faire un fit d'une 
#       gaussienne
    
   
    #print kwargs

    #cube, angs, psfn, fwhm, ncomp, PLSC, annulus_width, aperture_radius
    #cube = pipeline_parameters['cube']
    #angs = pipeline_parameters['angs']
    #psfn = pipeline_parameters['psfn']
    ##fwhm = pipeline_parameters['fwhm']
    #ncomp = pipeline_parameters['ncomp']
    #PLSC = pipeline_parameters['PLSC']
    #annulus_width = pipeline_parameters['annulus_width']
    #aperture_radius = pipeline_parameters['aperture_radius']
    
    if planets_xy_coord is None:
        #pca_frame(cube,angs,ncomp,cube.shape[1]//2.,cube.shape[1]//2.,display=True)        
        #_ = pca_full_frame(cube,angs,ncomp)

        print ''
        n_planet = input('How many planet(s) ?\n')
        planets_xy_coord = np.zeros([n_planet,2])
        msg = 'Planet {}: please define the (x,y) coordinate of the pixel which' 
        msg += 'locate the planet. Example: [340,210]\n'
        for j in range(n_planet):
             planets_xy_coord[j,:] = input(msg.format(j))
    else:
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
        
        r_pre, theta_pre, f_pre = firstguess_from_coord(planets_xy_coord[index_planet],
                                                        center_xy_coord,
                                                        cube, angs, PLSC,
                                                        psfn, annulus_width,
                                                        ncomp, aperture_radius,
                                                        f_range, display=display,
                                                        verbose=verbose,
                                                        save=save, 
                                                        **figure_options)
                                                                                                                    
        if verbose:
            msg3 = 'Planet {}: preliminary guess: (r,theta,f) = ({},{},{})'
            print msg3.format(index_planet,r_pre, theta_pre, f_pre)
<<<<<<< HEAD
 
=======
            msg4 = 'Planet {}: Simplex Nelder-Mead minimization, running ...'
            print msg4.format(index_planet) 
        res = firstguess_simplex((r_pre,theta_pre,f_pre), cube, angs, psfn,
                                 PLSC, fwhm, ncomp, annulus_width, 
                                 aperture_radius, tol=tol, max_iter=max_iter,
                                 verbose=True)
        if verbose:                               
            print '   Done !'                                       
        
        r_0[index_planet], theta_0[index_planet], f_0[index_planet] = res.x
>>>>>>> origin/master
        
        if simplex:
            if verbose:
                msg4 = 'Planet {}: Simplex Nelder-Mead minimization, running ...'
                print msg4.format(index_planet)
                                                         
            res = firstguess_simplex((r_pre,theta_pre,f_pre), cube, angs, psfn,
                                     PLSC, ncomp, annulus_width, aperture_radius, 
                                     options = simplex_options, verbose=False)
            
            r_0[index_planet], theta_0[index_planet], f_0[index_planet] = res.x
            if verbose:
                msg5 = 'Planet {}: Success: {}, nit: {}, chi2r: {}'
                print msg5.format(index_planet,res.success,res.nit,res.fun)
            
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
        