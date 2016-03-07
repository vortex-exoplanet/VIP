#! /usr/bin/env python

"""
Module with the function of merit definitions.
"""

import numpy as np
from skimage.draw import circle
from ..phot import inject_fcs_cube
from ..var import frame_center
from ..pca import pca_annulus

def chisquare(modelParameters, cube, angs, plsc, psfs_norm, fwhm, annulus_width,  
              aperture_radius, initialState, ncomp, cube_ref=None, 
              svd_mode='lapack', scaling=None):
    """
    Calculate the reduced chi2:
    \chi^2_r = \frac{1}{N-3}\sum_{j=1}^{N} |I_j|,
    where N is the number of pixels within a circular aperture centered on the 
    first estimate of the planet position, and I_j the j-th pixel intensity.
    
    Parameters
    ----------    
    modelParameters: tuple
        The model parameters, typically (r, theta, flux).
    cube: numpy.array
        The cube of fits images expressed as a numpy.array.
    angs: numpy.array
        The parallactic angle fits image expressed as a numpy.array. 
    plsc: float
        The platescale, in arcsec per pixel.
    psfs_norm: numpy.array
        The scaled psf expressed as a numpy.array.    
    fwhm : float
        The FHWM in pixels.
    annulus_width: int, optional
        The width in terms of the FWHM of the annulus on which the PCA is done.       
    aperture_radius: int, optional
        The radius of the circular aperture in terms of the FWHM.
    initialState: numpy.array
        Position (r, theta) of the circular aperture center.
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
        
    Returns
    -------
    out: float
        The reduced chi squared.
        
    """    
    try:
        r, theta, flux = modelParameters
    except TypeError:
        print('paraVector must be a tuple, {} was given'.format(type(modelParameters)))    
    
    # Create the cube with the negative fake companion injected
    cube_negfc = inject_fcs_cube(cube, psfs_norm, angs, flevel=-flux, 
                                 plsc=plsc, rad_dists=[r], n_branches=1, 
                                 theta=theta, verbose=False)       
                                      
    # Perform PCA to generate the processed image and extract the zone of interest                                     
    values = get_values_optimize(cube_negfc, angs, ncomp, annulus_width*fwhm,
                                 aperture_radius*fwhm, initialState[0],
                                 initialState[1], cube_ref=cube_ref, 
                                 svd_mode=svd_mode, scaling=scaling)
    
    # Function of merit
    values = np.abs(values)
    
    chi2 = np.sum(values[values>0])
    N =len(values[values>0])    
    
    return chi2/(N-3)



def get_values_optimize(cube, angs, ncomp, annulus_width, aperture_radius, 
                        r_guess, theta_guess, cube_ref=None, svd_mode='lapack',
                        scaling=None, debug=False):
    """
    Extracts a PCA-ed annulus from the cube and returns the flux values of the 
    pixels included in a circular aperture centered at a given position.
    
    Parameters
    ----------
    cube: numpy.array
        The cube of fits images expressed as a numpy.array.
    angs: numpy.array
        The parallactic angle fits image expressed as a numpy.array.
    ncomp: int
        The number of principal component.
    annulus_width: float
        The width in pixel of the annulus on which the PCA is performed.
    aperture_radius: float
        The radius of the circular aperture.
    r_guess: float
        The radial position of the center of the circular aperture. This 
        parameter is NOT the radial position of the candidate associated to the 
        Markov chain, but should be the fixed initial guess.
    theta_guess: float
        The angular position of the center of the circular aperture. This 
        parameter is NOT the angular position of the candidate associated to the 
        Markov chain, but should be the fixed initial guess.  
    cube_ref : array_like, 3d, optional
        Reference library cube. For Reference Star Differential Imaging.
    svd_mode : {'lapack', 'randsvd', 'eigen', 'arpack'}, str optional
        Switch for different ways of computing the SVD and selected PCs.
    scaling : {None, 'temp-mean', 'temp-standard'}
        With None, no scaling is performed on the input data before SVD. With 
        "temp-mean" then temporal px-wise mean subtraction is done and with 
        "temp-standard" temporal mean centering plus scaling to unit variance 
        is done.
    debug: boolean
        If True, the cube is returned along with the values.        
        
    Returns
    -------
    out: numpy.array
        The pixel values in the circular aperture after the PCA process.
        
    """
    centy_fr, centx_fr = frame_center(cube[0])
    posy = r_guess * np.sin(np.deg2rad(theta_guess)) + centy_fr
    posx = r_guess * np.cos(np.deg2rad(theta_guess)) + centx_fr
    sizey, sizex = cube[0].shape
    inter = annulus_width/2
        
    #print 'candidate R,theta:', r_guess, theta_guess
    #print 'candidate X,Y', posx, posy
    #print 'restricted [', inter, sizex-inter,']'
    
    if posx > sizex-inter or posx < inter :
        msg = 'Try increasing the size of your frames, the annulus '
        msg += 'and/or circular aperture used by the NegFC falls outside the FOV'
        raise RuntimeError(msg)
    if posy > sizey-inter or posy < inter:
        msg = 'Try increasing the size of your frames, the annulus '
        msg += 'and/or circular aperture used by the NegFC falls outside the FOV'
        raise RuntimeError(msg)    
        
    pca_frame = pca_annulus(cube, angs, ncomp, annulus_width, r_guess, cube_ref,
                            svd_mode, scaling) 
    indices = circle(cy=posy, cx=posx, radius=aperture_radius)
    values = pca_frame[indices]
    
    if debug:
        return values, pca_frame
    else:
        return values

