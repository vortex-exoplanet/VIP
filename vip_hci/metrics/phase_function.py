#! /usr/bin/env python
"""
Phase_function class definition
"""

__author__ = 'Julien Milli'
__all__ = []

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


class Phase_function(object):
    """ This class represents the scattering phase function (spf).
    It contains the attribute phase_function_calc that implements either a 
    single Henyey Greenstein phase function, a double Heyney Greenstein, 
    or any custom function (data interpolated from 
    an input list of phi, spf(phi)).
    """
    
    def __init__(self, spf_dico={'name': 'HG', 'g': 0., 'polar': False}):
        """
        Constructor of the Phase_function class. It checks whether the spf_dico
        contains a correct name and sets the attribute phase_function_calc
    
        Parameters
        ----------
        spf_dico :  dictionnary
            Parameters describing the scattering phase function to be
            implemented. By default, an isotropic phase function is implemented.
            It should at least contain the key "name" chosen between 'HG'
            (single Henyey Greenstein), 'DoubleHG' (double Heyney Greenstein) or
            'interpolated' (custom function). 
            The parameter "polar" enables to switch on the polarisation (if set 
            to True, the default is False). In this case it assumes a Rayleigh
            polarised fraction (1-(cos phi)^2) / (1+(cos phi)^2).
        """
        if not isinstance(spf_dico,dict):
            msg = 'The parameters describing the phase function must be a ' \
                  'Python dictionnary'
            raise TypeError(msg)
        if 'name' not in spf_dico.keys():
            msg = 'The dictionnary describing the phase function must contain' \
                  ' the key "name"'
            raise TypeError(msg)
        self.type = spf_dico['name']
        if 'polar' not in spf_dico.keys():
            self.polar = False
        elif not isinstance(spf_dico['polar'], bool):
            msg = 'The dictionnary describing the polarisation must be a ' \
                  'boolean'
            raise TypeError(msg)
        else: 
            self.polar = spf_dico['polar']
        if self.type == 'HG':
            self.phase_function_calc = HenyeyGreenstein_SPF(spf_dico)
        elif self.type == 'DoubleHG':
            self.phase_function_calc = DoubleHenyeyGreenstein_SPF(spf_dico)
        elif self.type == 'interpolated':
            self.phase_function_calc = Interpolated_SPF(spf_dico)
        else:
            msg = 'Type of phase function not understood: {0:s}'
            raise TypeError(msg.format(self.type))
                
    def compute_phase_function_from_cosphi(self, cos_phi):
        """
        Compute the phase function at (a) specific scattering scattering
        angle(s) phi. The argument is not phi but cos(phi) for optimization
        reasons.

        Parameters
        ----------
        cos_phi : float or array
            cosine of the scattering angle(s) at which the scattering function
            must be calculated.
        """
        phf = self.phase_function_calc.compute_phase_function_from_cosphi(cos_phi)
        if self.polar:
            return (1-cos_phi**2)/(1+cos_phi**2) * phf
        else:
            return phf

    def print_info(self):
        """
        Prints information on the type and parameters of the scattering phase 
        function.
        """
        print('----------------------------')
        print('Phase function parameters')
        print('----------------------------')
        print('Type of phase function: {0:s}'.format(self.type))
        print('Linear polarisation: {0!r}'.format(self.polar))
        self.phase_function_calc.print_info()        

    def plot_phase_function(self):
        """
        Plots the scattering phase function
        """
        phi = np.arange(0, 180, 1)
        phase_func = self.compute_phase_function_from_cosphi(np.cos(np.deg2rad(phi)))
        if self.polar:
            phase_func = (1-np.cos(np.deg2rad(phi))**2) / \
                         (1+np.cos(np.deg2rad(phi))**2) * phase_func

        plt.close(0)
        plt.figure(0)
        plt.plot(phi, phase_func)
        plt.xlabel('Scattering phase angle in degrees')
        plt.ylabel('Scattering phase function')
        plt.grid()
        plt.xlim(0, 180)
        plt.show()


class HenyeyGreenstein_SPF(object):
    """
    Implementation of a scattering phase function with a single Henyey
    Greenstein function.
    """

    def __init__(self, spf_dico={'g':0.}):
        """
        Constructor of a Heyney Greenstein phase function.
    
        Parameters
        ----------
        spf_dico :  dictionnary containing the key "g" (float)
            g is the Heyney Greenstein coefficient and should be between -1 
            (backward scattering) and 1 (forward scattering). 
        """
        # it must contain the key "g"
        if 'g' not in spf_dico.keys():
            raise TypeError('The dictionnary describing a Heyney Greenstein '
                            'phase function must contain the key "g"')
        # the value of "g" must be a float or a list of floats
        elif not isinstance(spf_dico['g'], (float, int)):
            raise TypeError('The key "g" of a Heyney Greenstein phase function'
                            ' dictionnary must be a float or an integer')
        self.set_phase_function(spf_dico['g'])

    def set_phase_function(self, g):
        """ 
        Set the value of g
        """
        if g >= 1:
            print('Warning the Henyey Greenstein parameter is greater than or '
                  'equal to 1')
            print('The value was changed from {0:6.2f} to 0.99'.format(g))
            g = 0.99
        elif g <= -1:
            print('Warning the Henyey Greenstein parameter is smaller than or '
                  'equal to -1')
            print('The value was changed from {0:6.2f} to -0.99'.format(g))
            g = -0.99
        self.g = float(g)
        
    def compute_phase_function_from_cosphi(self, cos_phi):
        """
        Compute the phase function at (a) specific scattering scattering
        angle(s) phi. The argument is not phi but cos(phi) for optimization
        reasons.

        Parameters
        ----------
        cos_phi : float or array
            cosine of the scattering angle(s) at which the scattering function
            must be calculated.
        """
        return 1./(4*np.pi)*(1-self.g**2)/(1+self.g**2-2*self.g*cos_phi)**(3./2.)         

    def print_info(self):
        """
        Prints the value of the HG coefficient
        """
        print('Heynyey Greenstein coefficient: {0:.2f}'.format(self.g))


class DoubleHenyeyGreenstein_SPF(object):
    """
    Implementation of a scattering phase function with a double Henyey
    Greenstein function.
    """
    
    def __init__(self, spf_dico={'g': [0.5,-0.3], 'weight': 0.7}):
        """
        """        
        # it must contain the key "g"
        if 'g' not in spf_dico.keys():
            raise TypeError('The dictionnary describing a Heyney Greenstein'
                            ' phase function must contain the key "g"')
        # the value of "g" must be a list of floats
        elif not isinstance(spf_dico['g'],(list,tuple,np.ndarray)):
            raise TypeError('The key "g" of a Heyney Greenstein phase '
                            'function dictionnary must be  a list of floats')
        # it must contain the key "weight"
        if 'weight' not in spf_dico.keys():
                raise TypeError('The dictionnary describing a multiple Henyey '
                                'Greenstein phase function must contain the '
                                'key "weight"')
        # the value of "weight" must be a list of floats
        elif not isinstance(spf_dico['weight'], (float, int)):
            raise TypeError('The key "weight" of a Heyney Greenstein phase '
                            'function dictionnary must be a float (weight of '
                            'the first HG coefficient between 0 and 1)')
        elif spf_dico['weight']<0 or spf_dico['weight']>1:
            raise ValueError('The key "weight" of a Heyney Greenstein phase'
                             ' function dictionnary must be between 0 and 1. It'
                             ' corresponds to the weight of the first HG '
                             'coefficient')
        if len(spf_dico['g']) != 2:
            raise TypeError('The keys "weight" and "g" must contain the same'
                            ' number of elements')
        self.g = spf_dico['g']
        self.weight = spf_dico['weight']   

    def print_info(self):
        """
        Prints the value of the HG coefficients and weights
        """
        print('Heynyey Greenstein first component : coeff {0:.2f} , '
              'weight {1:.1f}%'.format(self.g[0], self.weight*100))
        print('Heynyey Greenstein second component: coeff {0:.2f} , '
              'weight {1:.1f}%'.format(self.g[1], (1-self.weight)*100.))

    def compute_singleHG_from_cosphi(self, g, cos_phi):
        """
        Compute a single Heyney Greenstein phase function at (a) specific
        scattering scattering angle(s) phi. The argument is not phi but cos(phi)
        for optimization reasons.

        Parameters
        ----------
        g : float
            Heyney Greenstein coefficient
        cos_phi : float or array
            cosine of the scattering angle(s) at which the scattering function
            must be calculated.
        """
        return 1./(4*np.pi)*(1-g**2)/(1+g**2-2*g*cos_phi)**(3./2.)         

    def compute_phase_function_from_cosphi(self,cos_phi):
        """
        Compute the phase function at (a) specific scattering scattering
        angle(s) phi. The argument is not phi but cos(phi) for optimization
        reasons.

        Parameters
        ----------
        cos_phi : float or array
            cosine of the scattering angle(s) at which the scattering function
            must be calculated.
        """
        return self.weight * self.compute_singleHG_from_cosphi(self.g[0],
                                                               cos_phi) + \
                (1-self.weight) * self.compute_singleHG_from_cosphi(self.g[1],
                                                                    cos_phi)


class Interpolated_SPF(object):
    """
    Custom implementation of a scattering phase function by providing a list of 
    scattering phase angles and corresponding values of the phase function. 
    """
    
    def __init__(self, spf_dico={'phi':np.array([  0,  18,  36,  54,  72,  90,
                                                   108, 126, 144, 162]),
                                 'spf':np.array([3.580, 0.703, 0.141, 0.0489,
                                                 0.0233, 0.0136, 0.0091, 0.0069,
                                                 0.0056,0.005])}):
        """
        Constructor of the Interpolated_SPF class. It checks whether the spf_dico
        contains the keys 'phi' and 'spf'
    
        Parameters
        ----------
        spf_dico :  dict
            dictionnary containing at least the keys "phi" (list of scattering angles)
            and "spf" (list of corresponding scattering phase function values)
            Optionnaly it can specify the kind of interpolation requested (as 
            specified by the scipy.interpolate.interp1d function), by default
            it uses a quadratic interpolation.
        """
        for key in ['phi','spf']:
            if key not in spf_dico.keys():
                raise TypeError('The dictionnary describing a '
                                '"interpolated" phase function must contain '
                                'the key "{0:s}"'.format(key))
            elif not isinstance(spf_dico[key],(list,tuple,np.ndarray)):
                raise TypeError('The key "{0:s}" of a "interpolated" phase'
                                ' function dictionnary must be a list, np array'
                                ' or tuple'.format(key))
        if len(spf_dico['phi']) != len(spf_dico['spf']):
            raise TypeError('The keys "phi" and "spf" must contain the same'
                            ' number of elements')
        self.interpolate_phase_function(spf_dico)

    def print_info(self):
        """
        Prints the information of the spf
        """
        phi = np.linspace(0, 180, 19)
        spf = self.compute_phase_function_from_cosphi(np.cos(np.deg2rad(phi)))
        print('Scattering angle: ', phi)
        print('Interpolated scattering phase function: ', spf)

    def interpolate_phase_function(self, spf_dico):
        """
        Creates the function that returns the scattering phase function based
        on the scattering angle by interpolating the values given in the
        dictionnary. By default it uses a quadractic interpolation. 
    
        Parameters
        ----------
        spf_dico :  dict
            dictionnary containing at least the keys "phi" (list of scattering angles)
            and "spf" (list of corresponding scattering phase function values)
            Optionnaly it can specify the kind of interpolation requested (as 
            specified by the scipy.interpolate.interp1d function), by default
            it uses a quadratic interpolation.
        """
        if 'kind' in spf_dico.keys():
            if not isinstance(spf_dico['kind'],int) and spf_dico['kind'] not in \
                ['linear', 'nearest', 'zero', 'slinear', \
                 'quadratic', 'cubic', 'previous', 'next']:
                raise TypeError('The key "{0:s}" must be an integer '
                                'or a string ("linear", "nearest", "zero", "slinear", '
                                '"quadratic", "cubic", "previous",'
                                '"next"'.format(spf_dico['kind']))
            else:
                kind=spf_dico['kind']
        else:
            kind='quadratic'
        self.interpolation_function = interp1d(np.cos(np.deg2rad(spf_dico['phi'])),\
                                               spf_dico['spf'],kind=kind,\
                                               bounds_error=False,
                                               fill_value=np.nan)

    def compute_phase_function_from_cosphi(self, cos_phi):
        """
        Compute the phase function at (a) specific scattering scattering
        angle(s) phi. The argument is not phi but cos(phi) for optimization
        reasons.

        Parameters
        ----------
        cos_phi : float or array
            cosine of the scattering angle(s) at which the scattering function
            must be calculated.
        """
        return self.interpolation_function(cos_phi)
