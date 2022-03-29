#! /usr/bin/env python
"""
Class definition for ScatteredLightDisk, Dust_distribution and Phase_function 
"""

__author__ = 'Julien Milli'
__all__ = ['ScatteredLightDisk',
           'Dust_distribution',
           'Phase_function']

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import newton
from scipy.interpolate import interp1d
from ..var import frame_center


class ScatteredLightDisk(object):
    """
    Class used to generate a synthetic disc, inspired from a light version of 
    the GRATER tool (GRenoble RAdiative TransfER, Augereau et al. 1999) 
    written originally in IDL and converted to Python by J. Milli.
    """
    
    def __init__(self, nx=200, ny=200, distance=50., itilt=60., omega=0.,
                 pxInArcsec=0.01225, pa=0., flux_max=None,
                 density_dico={'name': '2PowerLaws', 'ain':5, 'aout':-5,
                               'a':40, 'e':0, 'ksi0':1., 'gamma':2., 'beta':1.,\
                                'dens_at_r0':1.},
                 spf_dico={'name':'HG', 'g':0., 'polar':False}, xdo=0., ydo=0.):
        """
        Constructor of the Scattered_light_disk object, taking in input the 
        geometric parameters of the disk, the radial density distribution 
        and the scattering phase function.
        So far, only one radial distribution is implemented: a smoothed
        2-power-law distribution, but more complex radial profiles can be implemented
        on demand.
        The star is assumed to be centered at the frame center as defined in
        the vip_hci.var.frame_center function (geometric center of the image,
        e.g. either in the middle of the central pixel for odd-size images or 
        in between 4 pixel for even-size images).
 
        Parameters
        ----------
        nx : int
            number of pixels along the x axis of the image (default 200)
        ny : int
            number of pixels along the y axis of the image (default 200)
        distance : float
            distance to the star in pc (default 70.)
        itilt : float
            inclination wrt the line of sight in degrees (0 means pole-on,
            90 means edge-on, default 60 degrees)
        omega : float
            argument of the pericenter in degrees (0 by default) 
        pxInArcsec : float 
            pixel field of view in arcsec/px (default the SPHERE pixel 
            scale 0.01225 arcsec/px)
        pa : float
            position angle of the disc in degrees (default 0 degrees, e.g. North)
        flux_max : float
            the max flux of the disk in ADU. By default None, meaning that
            the disk flux is not normalized to any value.
        density_dico : dict
            Parameters describing the dust density distribution function 
            to be implemented. By default, it uses a two-power law dust
            distribution with a vertical gaussian distribution with 
            linear flaring. This dictionary should at least contain the key
            "name".
            For a to-power law distribution, you can set it with
            'name:'2PowerLaws' and with the following parameters:
                a : float 
                    reference radius in au (default 40)
                ksi0 : float
                    scale height in au at the reference radius (default 1 a.u.)
                gamma : float 
                    exponent (2=gaussian,1=exponential profile, default 2)
                beta : float 
                    flaring index (0=no flaring, 1=linear flaring, default 1)
                ain : float 
                    slope of the power-low distribution in the inner disk. It 
                    must be positive (default 5)
                aout : float 
                    slope of the power-low distribution in the outer disk. It 
                    must be negative (default -5)
                e : float 
                    eccentricity (default 0)
                amin: float 
                    minimim semi-major axis: the dust density is 0 below this 
                    value (default 0)
        spf_dico :  dictionnary
            Parameters describing the scattering phase function to be implemented.
            By default, an isotropic phase function is implemented. It should
            at least contain the key "name". 
        xdo : float 
            disk offset along the x-axis in the disk frame (=semi-major axis), 
            in a.u. (default 0)
        ydo : float 
            disk offset along the y-axis in the disk frame (=semi-minor axis), 
            in a.u. (default 0)           
        """
        self.nx = nx    # number of pixels along the x axis of the image
        self.ny = ny    # number of pixels along the y axis of the image
        self.distance = distance  # distance to the star in pc
        self.set_inclination(itilt)
        self.set_omega(omega)
        self.set_flux_max(flux_max)
        self.pxInArcsec = pxInArcsec  # pixel field of view in arcsec/px
        self.pxInAU = self.pxInArcsec*self.distance     # 1 pixel in AU
        # disk offset along the x-axis in the disk frame (semi-major axis), AU
        self.xdo = xdo
        # disk offset along the y-axis in the disk frame (semi-minor axis), AU
        self.ydo = ydo
        self.rmin = np.sqrt(self.xdo**2+self.ydo**2)+self.pxInAU 
        self.dust_density = Dust_distribution(density_dico)     
        # star center along the y- and x-axis, in pixels
        self.yc, self.xc = frame_center(np.ndarray((self.ny,self.nx)))
        self.x_vector = (np.arange(0, nx) - self.xc)*self.pxInAU  # x axis in au
        self.y_vector = (np.arange(0, ny) - self.yc)*self.pxInAU  # y axis in au
        self.x_map_0PA, self.y_map_0PA = np.meshgrid(self.x_vector,
                                                     self.y_vector)
        self.set_pa(pa)
        self.phase_function = Phase_function(spf_dico=spf_dico)
        self.scattered_light_map = np.ndarray((ny, nx))
        self.scattered_light_map.fill(0.)
        
    def set_inclination(self, itilt):
        """
        Sets the inclination of the disk.

        Parameters
        ----------
        itilt : float 
            inclination of the disk wrt the line of sight in degrees (0 means 
            pole-on, 90 means edge-on, default 60 degrees)      
        """
        self.itilt = float(itilt)  # inclination wrt the line of sight in deg
        self.cosi = np.cos(np.deg2rad(self.itilt))
        self.sini = np.sin(np.deg2rad(self.itilt))
        
    def set_pa(self, pa):
        """
        Sets the disk position angle

        Parameters
        ----------
        pa : float
            position angle in degrees
        """
        self.pa = pa    # position angle of the disc in degrees
        self.cospa = np.cos(np.deg2rad(self.pa))
        self.sinpa = np.sin(np.deg2rad(self.pa))
        # rotation to get the disk major axis properly oriented, x in AU
        self.y_map = (self.cospa*self.x_map_0PA + self.sinpa*self.y_map_0PA)
        # rotation to get the disk major axis properly oriented, y in AU
        self.x_map = (-self.sinpa*self.x_map_0PA + self.cospa*self.y_map_0PA)
        
    def set_omega(self, omega):
        """
        Sets the argument of pericenter 

        Parameters
        ----------
        omega : float
            angle in degrees
        """
        self.omega = float(omega)

    def set_flux_max(self, flux_max):
        """
        Sets the mas flux of the disk

        Parameters
        ----------
        flux_max : float
            the max flux of the disk in ADU
        """
        self.flux_max = flux_max

    def set_density_distribution(self, density_dico):
        """
        Sets or updates the parameters of the density distribution    

        Parameters
        ----------
        density_dico : dict
            Parameters describing the dust density distribution function 
            to be implemented. By default, it uses a two-power law dust
            distribution with a vertical gaussian distribution with 
            linear flaring. This dictionary should at least contain the key
            "name". For a to-power law distribution, you can set it with
            name:'2PowerLaws' and with the following parameters:
            - a : float.
            Reference radius in au (default 60)
            - ksi0 : float.
            Scale height in au at the reference radius (default 1 a.u.)
            - gamma : float.
            Exponent (2=gaussian,1=exponential profile, default 2)
            - beta : float.
            Flaring index (0=no flaring, 1=linear flaring, default 1)
            - ain : float.
            Slope of the power-low distribution in the inner disk. It 
            must be positive (default 5)
            - aout : float. 
            Slope of the power-low distribution in the outer disk. It 
            must be negative (default -5)
            - e : float.
            Eccentricity (default 0)
        """
        self.dust_density.set_density_distribution(density_dico)
    
    def set_phase_function(self, spf_dico):
        """
        Sets the phase function of the dust 

        Parameters
        ----------
        spf_dico :  dict
            Parameters describing the scattering phase function to be
            implemented. Three phase functions are implemented so far: single
            Heyney Greenstein, double Heyney Greenstein and custum phase
            functions through interpolation. Read the constructor of each of
            those classes to know which parameters must be set in the dictionary
            in each case.
        """
        self.phase_function = Phase_function(spf_dico=spf_dico)
    
    def print_info(self):
        """ 
        Prints the information of the disk and image parameters
        """
        print('-----------------------------------')
        print('Geometrical properties of the image')
        print('-----------------------------------')
        print('Image size: {0:d} px by {1:d} px'.format(self.nx,self.ny))
        msg1 = 'Pixel size: {0:.4f} arcsec/px or {1:.2f} au/px'
        print(msg1.format(self.pxInArcsec, self.pxInAU))
        msg2 = 'Distance of the star {0:.1f} pc'
        print(msg2.format(self.distance))
        msg3 = 'From {0:.1f} au to {1:.1f} au in X'
        print(msg3.format(self.x_vector[0],self.x_vector[self.nx-1]))
        msg4 = 'From {0:.1f} au to {1:.1f} au in Y'
        print(msg4.format(self.y_vector[0], self.y_vector[self.nx-1]))
        print('Position angle of the disc: {0:.2f} degrees'.format(self.pa))
        print('Inclination {0:.2f} degrees'.format(self.itilt))
        print('Argument of pericenter {0:.2f} degrees'.format(self.omega))
        if self.flux_max is not None:
            print('Maximum flux of the disk {0:.2f}'.format(self.flux_max))
        self.dust_density.print_info()
        self.phase_function.print_info()
        
    def check_inclination(self):
        """
        Checks whether the inclination set is close to edge-on and risks to 
        induce artefacts from the limited numerical accuracy. In such a case
        the inclination is changed to be less edge-on.
        """
        if np.abs(np.mod(self.itilt,180)-90) < np.abs(np.mod(self.dust_density.dust_distribution_calc.itiltthreshold,180)-90):
            print('Warning the disk is too close to edge-on')
            msg = 'The inclination was changed from {0:.2f} to {1:.2f}'
            print(msg.format(self.itilt,
                             self.dust_density.dust_distribution_calc.itiltthreshold))
            self.set_inclination(self.dust_density.dust_distribution_calc.itiltthreshold)
    
    def compute_scattered_light(self, halfNbSlices=25):
        """ 
        Computes the scattered lignt image of the disk.

        Parameters
        ----------
        halfNbSlices : integer 
            half number of distances along the line of sight l
        """        
        self.check_inclination()
        # dist along the line of sight to reach the disk midplane (z_D=0), AU:
        lz0_map = self.y_map * np.tan(np.deg2rad(self.itilt))
        # dist to reach +zmax, AU:
        lzp_map = self.dust_density.dust_distribution_calc.zmax/self.cosi + \
                  lz0_map
        # dist to reach -zmax, AU:
        lzm_map = -self.dust_density.dust_distribution_calc.zmax/self.cosi + \
                  lz0_map
        dl_map = np.absolute(lzp_map-lzm_map)  # l range, in AU
        # squared maximum l value to reach the outer disk radius, in AU^2:
        lmax2 = self.dust_density.dust_distribution_calc.rmax**2 - (self.x_map**2+self.y_map**2)
        # squared minimum l value to reach the inner disk radius, in AU^2:
        lmin2 = (self.x_map**2+self.y_map**2)-self.rmin**2 
        validPixel_map = (lmax2 > 0.) * (lmin2 > 0.)        
        lwidth = 100.  # control the distribution of distances along l
        nbSlices = 2*halfNbSlices-1  # total number of distances 
                                     # along the line of sight
        tmp = (np.exp(np.arange(halfNbSlices)*np.log(lwidth+1.) /
                      (halfNbSlices-1.))-1.)/lwidth  # between 0 and 1
        ll = np.concatenate((-tmp[:0:-1], tmp))
        # 1d array pre-calculated values, AU
        ycs_vector = self.cosi*self.y_map[validPixel_map] 
        # 1d array pre-calculated values, AU
        zsn_vector = -self.sini*self.y_map[validPixel_map]  
        xd_vector = self.x_map[validPixel_map]  # x_disk, in AU
        limage = np.ndarray((nbSlices, self.ny, self.nx))
        limage.fill(0.)
        
        for il in range(nbSlices):
            # distance along the line of sight to reach the plane z
            l_vector =lz0_map[validPixel_map] + ll[il]*dl_map[validPixel_map] 
            # rotation about x axis 
            yd_vector = ycs_vector + self.sini * l_vector  # y_Disk in AU
            zd_vector = zsn_vector + self.cosi * l_vector  # z_Disk, in AU
            # Dist and polar angles in the frame centered on the star position:
            # squared distance to the star, in AU^2
            d2star_vector = xd_vector**2+yd_vector**2+zd_vector**2
            dstar_vector = np.sqrt(d2star_vector)  # distance to the star, in AU
            # midplane distance to the star (r coordinate), in AU
            rstar_vector = np.sqrt(xd_vector**2+yd_vector**2)
            thetastar_vector = np.arctan2(yd_vector, xd_vector)
            # Phase angles:
            cosphi_vector = (rstar_vector*self.sini*np.sin(thetastar_vector)+zd_vector*self.cosi)/dstar_vector  # in radians
            # Polar coordinates in the disk frame, and semi-major axis:
            # midplane distance to the disk center (r coordinate), in AU
            r_vector = np.sqrt((xd_vector-self.xdo)**2+(yd_vector-self.ydo)**2)
            # polar angle in radians between 0 and pi
            theta_vector = np.arctan2(yd_vector-self.ydo,xd_vector-self.xdo)
            costheta_vector = np.cos(theta_vector-np.deg2rad(self.omega))
            # Scattered light:
            # volume density
            rho_vector = self.dust_density.density_cylindrical(r_vector,
                                                               costheta_vector,
                                                               zd_vector)
            phase_function = self.phase_function.compute_phase_function_from_cosphi(cosphi_vector)
            image = np.ndarray((self.ny, self.nx))
            image.fill(0.)
            image[validPixel_map] = rho_vector*phase_function/d2star_vector
            limage[il,:,:] = image
        self.scattered_light_map.fill(0.)
        for il in range(1,nbSlices):
            self.scattered_light_map += (ll[il]-ll[il-1]) * (limage[il-1,:,:] +
                                                             limage[il,:,:])
        if self.flux_max is not None:
            self.scattered_light_map *= (self.flux_max/np.nanmax(self.scattered_light_map))
        return self.scattered_light_map



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
            to True, the default is False). In this case it assumes either 
                - a Rayleigh polarised fraction (1-(cos phi)^2) / (1+(cos phi)^2).
                  if nothing else is specified
                - a polynomial if the keyword 'polar_polynom_coeff' is defined 
                  and corresponds to an array of polynomial coefficient, e.g. 
                  [p3,p2,p1,p0] evaluated as np.polyval([p3,p2,p1,p0],np.arange(0, 180, 1))
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
        else:
            if not isinstance(spf_dico['polar'], bool):
                msg = 'The dictionnary describing the polarisation must be a ' \
                      'boolean'
                raise TypeError(msg)
            self.polar = spf_dico['polar']
            if 'polar_polynom_coeff' in spf_dico.keys():
                self.polar_polynom = True
                if isinstance(spf_dico['polar_polynom_coeff'], (tuple,list,np.ndarray)):
                    self.polar_polynom_coeff = spf_dico['polar_polynom_coeff']
                else:
                    msg = 'The dictionnary describing the polarisation polynomial function must be an ' \
                          'array'
                    raise TypeError(msg)
            else:
                self.polar_polynom = False
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
            if self.polar_polynom:
                phi = np.rad2deg(np.arccos(cos_phi))
                return  np.polyval(self.polar_polynom_coeff, phi) * phf                
            else:
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
            if self.polar_polynom:
                phase_func =  np.polyval(self.polar_polynom_coeff, phi) * phase_func                
            else:
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
        self.interpolation_function = interp1d(spf_dico['phi'],\
                                               spf_dico['spf'],kind=kind,\
                                               bounds_error=False,
                                               fill_value=np.nan)
        # commented on 2022-02-16
        # self.interpolation_function = interp1d(np.cos(np.deg2rad(spf_dico['phi'])),\
        #                                        spf_dico['spf'],kind=kind,\
        #                                        bounds_error=False,
        #                                        fill_value=np.nan)

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
        return self.interpolation_function(np.rad2deg(np.arccos(cos_phi)))
        # return self.interpolation_function(cos_phi)


if __name__ == '__main__':
    """
    It is an example of use of the ScatteredLightDisk class. 
    """    

    # Example 1: creation of a disk at 20pc, with semi-major axis 20 a.u.
    #            inner and outer slopes 2 and -5, inclined by 70degrees wrt
    #            the line of sight and with a PA of 45degrees, an isotropic
    #            phase function
    Scattered_light_disk1 = ScatteredLightDisk(nx=201, ny=201, distance=20,
                                               itilt=70.,omega=0.,
                                               pxInArcsec=0.01225, pa=45.,
                                               density_dico={'name':'2PowerLaws',
                                                             'ain':5,'aout':-5,
                                                             'a':40,'e':0,
                                                             'ksi0':1.,
                                                             'gamma':2.,
                                                             'beta':1.},
                                               spf_dico={'name':'HG','g':0.,
                                                         'polar':False})
    disk1 = Scattered_light_disk1.compute_scattered_light()

    # If you prefer set the parameters in differen steps, you can also do that
    # with: (this is identical)
    Scattered_light_disk1 = ScatteredLightDisk(nx=201, ny=201, distance=20)
    Scattered_light_disk1.set_pa(45)
    Scattered_light_disk1.set_inclination(70)
    Scattered_light_disk1.set_density_distribution({'name':'2PowerLaws',
                                                    'ain':2,'aout':-5,'a':20,
                                                    'e':0,'ksi0':1.,
                                                    'gamma':2.,'beta':1.})
    Scattered_light_disk1.set_phase_function({'name':'HG','g':0.,'polar':False})
    disk1 = Scattered_light_disk1.compute_scattered_light()    

    # If you want to know what are the parameters you set for your disk:
    Scattered_light_disk1.print_info()

    # Example 2: Creation of a disk similar to example 1 but with a double 
    #           Heyney Greenstein phase function 

    Scattered_light_disk2 = ScatteredLightDisk(nx=201,ny=201,distance=20)
    Scattered_light_disk2.set_pa(45)
    Scattered_light_disk2.set_inclination(70)
    Scattered_light_disk2.set_density_distribution({'name':'2PowerLaws',
                                                    'ain':2,'aout':-5,'a':20,
                                                    'e':0,'ksi0':1.,
                                                    'gamma':2.,'beta':1.})
    Scattered_light_disk2.set_phase_function({'name':'DoubleHG',
                                              'g':[0.6,-0.6],'weight':0.7,
                                              'polar':False})
    Scattered_light_disk2.print_info()
    disk2 = Scattered_light_disk2.compute_scattered_light()    

    # Let's turn the polarisation on now:
    Scattered_light_disk2.set_phase_function({'name':'DoubleHG',
                                              'g':[0.6,-0.6],'weight':0.7,
                                              'polar':True})
    disk2_polar = Scattered_light_disk2.compute_scattered_light()    

