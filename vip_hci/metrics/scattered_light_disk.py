#! /usr/bin/env python
"""
Scattered_light_disk class definition
"""

__author__ = 'Julien Milli'
__all__ = []

import numpy as np
from ..var import frame_center
from .dust_distribution import Dust_distribution
from .phase_function import Phase_function


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
            a : float 
                reference radius in au (default 60)
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

