# -*- coding: utf-8 -*-
"""
Created on Wed May  6 17:07:00 2015
"""
import numpy as np
from scipy.optimize import newton

_author__ = 'Julien Milli'


class Dust_distribution(object):
    """This class represents the dust distribution
    """
    def __init__(self,density_dico={'name':'2PowerLaws', 'ain':5, 'aout':-5,
                                    'a':60, 'e':0, 'ksi0':1., 'gamma':2.,
                                    'beta':1.,'amin':0.,'dens_at_r0':1.}):
        """ 
        Constructor for the Dust_distribution class.
        
        We assume the dust density is 0 radially after it drops below 0.5% 
        (the accuracy variable) of the peak density in 
        the midplane, and vertically whenever it drops below 0.5% of the 
        peak density in the midplane
        """
        self.accuracy = 5.e-3
        if not isinstance(density_dico, dict):
            errmsg = 'The parameters describing the dust density distribution' \
                     ' must be a Python dictionnary'
            raise TypeError(errmsg)
        if 'name' not in density_dico.keys():
            errmsg = 'The dictionnary describing the dust density ' \
                     'distribution must contain the key "name"'
            raise TypeError(errmsg)
        self.type = density_dico['name']
        if self.type == '2PowerLaws':
            self.dust_distribution_calc = DustEllipticalDistribution2PowerLaws(
                                                    self.accuracy, density_dico)
        else:
            errmsg = 'The only dust distribution implemented so far is the' \
                     ' "2PowerLaws"'
            raise TypeError(errmsg)

    def set_density_distribution(self,density_dico):
        """
        Update the parameters of the density distribution.
        """
        self.dust_distribution_calc.set_density_distribution(density_dico)

    def density_cylindrical(self, r, costheta, z):
        """ 
        Return the particule volume density at r, theta, z.
        """
        return self.dust_distribution_calc.density_cylindrical(r, costheta, z)

    def density_cartesian(self, x, y, z):
        """ 
        Return the particule volume density at x,y,z, taking into account the
        offset of the disk.
        """           
        return self.dust_distribution_calc.density_cartesian(x, y, z)

    def print_info(self, pxInAu=None):
        """
        Utility function that displays the parameters of the radial distribution
        of the dust

        Input:
            - pxInAu (optional): the pixel size in au
        """
        print('----------------------------')
        print('Dust distribution parameters')
        print('----------------------------')
        self.dust_distribution_calc.print_info(pxInAu)


class DustEllipticalDistribution2PowerLaws:
    """
    """
    def __init__(self, accuracy=5.e-3, density_dico={'ain':5,'aout':-5,\
                                                     'a':60,'e':0,'ksi0':1.,\
                                                     'gamma':2.,'beta':1.,\
                                                     'amin':0.,'dens_at_r0':1.}):
        """ 
        Constructor for the Dust_distribution class.
        
        We assume the dust density is 0 radially after it drops below 0.5% 
        (the accuracy variable) of the peak density in 
        the midplane, and vertically whenever it drops below 0.5% of the 
        peak density in the midplane
        """
        self.accuracy = accuracy
        self.set_density_distribution(density_dico)

    def set_density_distribution(self,density_dico):
        """
        """
        if 'ksi0' not in density_dico.keys():
            ksi0 = 1.
        else:
            ksi0 = density_dico['ksi0']
        if 'beta' not in density_dico.keys():
            beta = 1.
        else:
            beta = density_dico['beta']
        if 'gamma' not in density_dico.keys():
            gamma = 1.
        else:
            gamma = density_dico['gamma']
        if 'aout' not in density_dico.keys():
            aout = -5.
        else:
            aout = density_dico['aout']
        if 'ain' not in density_dico.keys():
            ain = 5.
        else:
            ain = density_dico['ain']
        if 'e' not in density_dico.keys():
            e = 0.
        else:
            e = density_dico['e']
        if 'a' not in density_dico.keys():
            a = 60.
        else:
            a = density_dico['a']
        if 'amin' not in density_dico.keys():
            amin = 0.
        else:
            amin = density_dico['amin']   
        if 'dens_at_r0' not in density_dico.keys():
            dens_at_r0=1.
        else:
            dens_at_r0=density_dico['dens_at_r0']
        self.set_vertical_density(ksi0=ksi0, gamma=gamma, beta=beta)
        self.set_radial_density(ain=ain, aout=aout, a=a, e=e,amin=amin,dens_at_r0=dens_at_r0)

    def set_vertical_density(self, ksi0=1., gamma=2., beta=1.):
        """ 
        Sets the parameters of the vertical density function    

        Parameters
        ----------
        ksi0 : float
            scale height in au at the reference radius (default 1 a.u.)
        gamma : float 
            exponent (2=gaussian,1=exponential profile, default 2)
        beta : float 
            flaring index (0=no flaring, 1=linear flaring, default 1)
        """    
        if gamma < 0.:
            print('Warning the vertical exponent gamma is negative')
            print('Gamma was changed from {0:6.2f} to 0.1'.format(gamma))
            gamma = 0.1
        if ksi0 < 0.:
            print('Warning the scale height ksi0 is negative')
            print('ksi0 was changed from {0:6.2f} to 0.1'.format(ksi0))
            ksi0 = 0.1
        if beta < 0.:
            print('Warning the flaring coefficient beta is negative')
            print('beta was changed from {0:6.2f} to 0 (flat disk)'.format(beta))
            beta = 0.
        self.ksi0 = float(ksi0)
        self.gamma = float(gamma)
        self.beta = float(beta)
        self.zmax = ksi0*(-np.log(self.accuracy))**(1./gamma)

    def set_radial_density(self, ain=5., aout=-5., a=60., e=0.,amin=0.,dens_at_r0=1.):
        """ 
        Sets the parameters of the radial density function

        Parameters
        ----------
        ain : float 
            slope of the power-low distribution in the inner disk. It 
            must be positive (default 5)
        aout : float 
            slope of the power-low distribution in the outer disk. It 
            must be negative (default -5)
        a : float 
            reference radius in au (default 60)
        e : float 
            eccentricity (default 0)
        amin: float 
            minimim semi-major axis: the dust density is 0 below this 
            value (default 0)
        """    
        if ain < 0.1:
            print('Warning the inner slope is greater than 0.1')
            print('ain was changed from {0:6.2f} to 0.1'.format(ain))
            ain = 0.1
        if aout > -0.1:
            print('Warning the outer slope is greater than -0.1')
            print('aout was changed from {0:6.2f} to -0.1'.format(aout))
            aout = -0.1           
        if e < 0:
            print('Warning the eccentricity is negative')
            print('e was changed from {0:6.2f} to 0'.format(e))
            e = 0.        
        if e >= 1:
            print('Warning the eccentricity is greater or equal to 1')
            print('e was changed from {0:6.2f} to 0.99'.format(e))
            e = 0.99  
        if a < 0:
            raise ValueError('Warning the semi-major axis a is negative')
        if amin < 0:
            raise ValueError('Warning the minimum radius a is negative')
            print('amin was changed from {0:6.2f} to 0.'.format(amin))
            amin = 0. 
        if dens_at_r0 <0:
            raise ValueError('Warning the reference dust density at r0 is negative')
            print('It was changed from {0:6.2f} to 1.'.format(dens_at_r0))
            dens_at_r0 = 1.  
        self.ain = float(ain)
        self.aout = float(aout)
        self.a = float(a)
        self.e = float(e)
        self.p = self.a*(1-self.e**2)
        self.amin = float(amin)
        self.pmin = self.amin*(1-self.e**2) ## we assume the inner hole is also elliptic (convention)
        self.dens_at_r0 = float(dens_at_r0)
        try:
            # maximum distance of integration, AU
            self.rmax = self.a*self.accuracy**(1/self.aout)
            if self.ain != self.aout:
                self.apeak = self.a * np.power(-self.ain/self.aout,
                                               1./(2.*(self.ain-self.aout)))
                Gamma_in = self.ain+self.beta
                Gamma_out = self.aout+self.beta
                self.apeak_surface_density = self.a * np.power(-Gamma_in/Gamma_out,
                                               1./(2.*(Gamma_in-Gamma_out)))
            else:
                self.apeak = self.a
                self.apeak_surface_density = self.a
        except OverflowError:
            print('The error occured during the calculation of rmax or apeak')
            print('Inner slope: {0:.6e}'.format(self.ain))
            print('Outer slope: {0:.6e}'.format(self.aout))
            print('Accuracy: {0:.6e}'.format(self.accuracy))
            raise OverflowError
        except ZeroDivisionError:
            print('The error occured during the calculation of rmax or apeak')
            print('Inner slope: {0:.6e}'.format(self.ain))
            print('Outer slope: {0:.6e}'.format(self.aout))
            print('Accuracy: {0:.6e}'.format(self.accuracy))
            raise ZeroDivisionError
        self.itiltthreshold = np.rad2deg(np.arctan(self.rmax/self.zmax))
                
    def print_info(self, pxInAu=None):
        """
        Utility function that displays the parameters of the radial distribution
        of the dust

        Input:
            - pxInAu (optional): the pixel size in au
        """
        def rad_density(r):
            return np.sqrt(2/(np.power(r/self.a,-2*self.ain) +
                       np.power(r/self.a,-2*self.aout)))
        half_max_density = lambda r:rad_density(r)/rad_density(self.apeak)-1./2.            
        try:
            if self.aout < -3:            
                a_plus_hwhm = newton(half_max_density,self.apeak*1.04)
            else:
                a_plus_hwhm = newton(half_max_density,self.apeak*1.1)
        except RuntimeError:
            a_plus_hwhm = np.nan
        try: 
            if self.ain < 2:
                a_minus_hwhm = newton(half_max_density,self.apeak*0.5)            
            else:
                a_minus_hwhm = newton(half_max_density,self.apeak*0.95)                            
        except RuntimeError:
            a_minus_hwhm = np.nan
        if pxInAu is not None:
            msg = 'Reference semi-major axis: {0:.1f}au or {1:.1f}px'
            print(msg.format(self.a,self.a/pxInAu))
            msg2 = 'Semi-major axis at maximum dust density in plane z=0: {0:.1f}au or ' \
                   '{1:.1f}px (same as ref sma if ain=-aout)'
            print(msg2.format(self.apeak,self.apeak/pxInAu))            
            msg3 = 'Semi-major axis at half max dust density in plane z=0: {0:.1f}au or ' \
                    '{1:.1f}px for the inner edge ' \
                    '/ {2:.1f}au or {3:.1f}px for the outer edge, with a FWHM of ' \
                    '{4:.1f}au or {5:.1f}px'
            print(msg3.format(a_minus_hwhm,a_minus_hwhm/pxInAu,a_plus_hwhm,\
                              a_plus_hwhm/pxInAu,a_plus_hwhm-a_minus_hwhm,\
                                  (a_plus_hwhm-a_minus_hwhm)/pxInAu))
            msg4 = 'Semi-major axis at maximum dust surface density: {0:.1f}au or ' \
                   '{1:.1f}px (same as ref sma if ain=-aout)'
            print(msg4.format(self.apeak_surface_density,self.apeak_surface_density/pxInAu))                            
            msg5 = 'Ellipse p parameter: {0:.1f}au or {1:.1f}px'
            print(msg5.format(self.p,self.p/pxInAu))
        else:
            print('Reference semi-major axis: {0:.1f}au'.format(self.a))
            msg = 'Semi-major axis at maximum dust density in plane z=0: {0:.1f}au (same ' \
                  'as ref sma if ain=-aout)'
            print(msg.format(self.apeak))
            msg3 = 'Semi-major axis at half max dust density: {0:.1f}au ' \
                    '/ {1:.1f}au for the inner/outer edge, or a FWHM of ' \
                    '{2:.1f}au'
            print(msg3.format(a_minus_hwhm,a_plus_hwhm,a_plus_hwhm-a_minus_hwhm))
            print('Ellipse p parameter: {0:.1f}au'.format(self.p))
        print('Ellipticity: {0:.3f}'.format(self.e))
        print('Inner slope: {0:.2f}'.format(self.ain))
        print('Outer slope: {0:.2f}'.format(self.aout))
        print('Density at the reference semi-major axis: {0:4.3e} (arbitrary unit'.format(self.dens_at_r0))
        if self.amin>0:
            print('Minimum radius (sma): {0:.2f}au'.format(self.amin))        
        if pxInAu is not None:
            msg = 'Scale height: {0:.1f}au or {1:.1f}px at {2:.1f}'
            print(msg.format(self.ksi0,self.ksi0/pxInAu,self.a))
        else:
            print('Scale height: {0:.2f} au at {1:.2f}'.format(self.ksi0,
                                                               self.a))
        print('Vertical profile index: {0:.2f}'.format(self.gamma))
        msg = 'Disc vertical FWHM: {0:.2f} at {1:.2f}'
        print(msg.format(2.*self.ksi0*np.power(np.log10(2.), 1./self.gamma),
                         self.a))
        print('Flaring coefficient: {0:.2f}'.format(self.beta))
        print('------------------------------------')
        print('Properties for numerical integration')
        print('------------------------------------')
        print('Requested accuracy {0:.2e}'.format(self.accuracy))
#        print('Minimum radius for integration: {0:.2f} au'.format(self.rmin))
        print('Maximum radius for integration: {0:.2f} au'.format(self.rmax))
        print('Maximum height for integration: {0:.2f} au'.format(self.zmax))
        msg = 'Inclination threshold: {0:.2f} degrees'
        print(msg.format(self.itiltthreshold))
        return
        
    def density_cylindrical(self, r, costheta, z):
        """ Returns the particule volume density at r, theta, z
        """
        radial_ratio = r/(self.p/(1-self.e*costheta))
        den = (np.power(radial_ratio, -2*self.ain) +
               np.power(radial_ratio,-2*self.aout))
        radial_density_term = np.sqrt(2./den)*self.dens_at_r0
        if self.pmin>0:
             radial_density_term[r/(self.pmin/(1-self.e*costheta)) <= 1]=0
        den2 = (self.ksi0*np.power(radial_ratio,self.beta))
        vertical_density_term = np.exp(-np.power(np.abs(z)/den2, self.gamma))
        return radial_density_term*vertical_density_term    

    def density_cartesian(self, x, y, z):
        """ Returns the particule volume density at x,y,z, taking into account
        the offset of the disk
        """
        r = np.sqrt(x**2+y**2)
        if r == 0:
            costheta=0
        else:
            costheta = x/r
        return self.density_cylindrical(r,costheta,z)


if __name__ == '__main__':
    """
    Main of the class for debugging
    """
    test = DustEllipticalDistribution2PowerLaws()
    test.set_radial_density(ain=5., aout=-5., a=60., e=0.)
    test.print_info()
    costheta = 0.
    z = 0.
    for a in np.linspace(60-5,60+5,11):
        t = test.density_cylindrical(a, costheta, z)
        print('r={0:.1f} density={1:.4f}'.format(a, t))