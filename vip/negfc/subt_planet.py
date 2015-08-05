#! /usr/bin/env python

"""
Module with the function for creating planet free images.
"""

import numpy as np
from ..phot import inject_fcs_cube



def cube_planet_free(planet_parameter,pipeline_parameters):
    """
    Return a cube in which we have injected negative fake companion at the 
    position/flux given by planet_parameter.
    
    """
    cube = pipeline_parameters['cube'] 
    psfn = pipeline_parameters['psfn'] 
    angs = pipeline_parameters['angs'] 
    PLSC = pipeline_parameters['PLSC']    
    
    cpf = np.zeros_like(cube)
    
    if len(planet_parameter.shape) == 1:
        range_index = np.zeros(1) # There is only 1 planet.
    else:
        range_index = range(planet_parameter.shape[0]) # There are more than 1 planet.
    
    for planet_index in range_index: # At the end of this loop, "cpf" is free of planet(s)
        if planet_index == 0:
            cube_temp = cube
        else:
            cube_temp = cpf
        
        cpf = inject_fcs_cube(cube_temp, psfn, angs,
                                     flevel=-planet_parameter[planet_index,2],
                                     plsc=PLSC, 
                                     rad_arcs=[planet_parameter[planet_index,0]*PLSC],
                                     n_branches=1, 
                                     theta=planet_parameter[planet_index,1])    
    return cpf


