#! /usr/bin/env python

"""
Module with the function for creating planet free images.
"""

import numpy as np
from ..phot import inject_fcs_cube



def cube_planet_free(planet_parameter, cube, angs, psfn, plsc):
    """
    Return a cube in which we have injected negative fake companion at the 
    position/flux given by planet_parameter.
    
    Parameters
    ----------
    planet_parameter: numpy.array or list
        The (r, theta, flux) for all known companions.    
    cube: numpy.array
        The cube of fits images expressed as a numpy.array.
    angs: numpy.array
        The parallactic angle fits image expressed as a numpy.array. 
    psfsn: numpy.array
        The scaled psf expressed as a numpy.array.        
    plsc: float
        The platescale, in arcsec per pixel.
        
    Returns
    -------
    cpf : numpy.array
        The cube with negative companions injected at the position given in
        planet_parameter.
       
    """    
    cpf = np.zeros_like(cube)
    
    planet_parameter = np.array(planet_parameter)
    
    for i in range(planet_parameter.shape[0]): 
        if i == 0:
            cube_temp = cube
        else:
            cube_temp = cpf
        
        cpf = inject_fcs_cube(cube_temp, psfn, angs, flevel=-planet_parameter[i,2],
                              plsc=plsc, rad_dists=[planet_parameter[i,0]],
                              n_branches=1, theta=planet_parameter[i,1], 
                              verbose=False)    
    return cpf


