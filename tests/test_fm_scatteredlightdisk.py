"""
Tests for fm/scattered_light_disk.py

"""

from .helpers import aarc, parametrize, param
from vip_hci.fm.scattered_light_disk import DustEllipticalDistribution2PowerLaws
from vip_hci.fm import ScatteredLightDisk
import numpy as np


@parametrize("r",
             [
                param(55),
                param(60),
                param(65)
             ])
def test_dust_distribution(r):
    """
    Verify dust density at different radii, for a distribution centered at a=60
    """
    def t_expected(r):
        """
        Expected positions.
        """
        if r == 55:
            return 0.844237
        elif r == 60:
            return 1
        elif r == 65:
            return 0.864574
    test = DustEllipticalDistribution2PowerLaws()
    test.set_radial_density(ain=5., aout=-5., a=60., e=0.)
    costheta = 0.
    z = 0.
    t = test.density_cylindrical(r, costheta, z)

    aarc(t, t_expected(r))


@parametrize("i",
             [
                param(0),
                param(30),
                param(60)
             ])
def test_scattered_light_disk(i):
    """
    Verify that the scattered light image is normalized to a maximum value of 1
    for 3 different inclinations: 0, 30 and 60 degrees.
    """
    pixel_scale = 0.01225  # pixel scale in arcsec/px
    dstar = 80  # distance to the star in pc
    nx = 200  # number of pixels of your image in X
    ny = 200  # number of pixels of your image in Y
    pa = 30  # position angle of the disk in degrees (from north to east)
    a = 70  # semimajoraxis of the disk in au
    fake_disk = ScatteredLightDisk(
                            nx=nx, ny=ny, distance=dstar,
                            itilt=i, omega=0, pxInArcsec=pixel_scale, pa=pa,
                            density_dico={'name': '2PowerLaws', 'ain': 12, 'aout': -12,
                                          'a': a, 'e': 0.0, 'ksi0': 1., 'gamma': 2., 'beta': 1.},
                            spf_dico={'name': 'HG', 'g': 0., 'polar': False}, flux_max=1)
    fake_disk_map = fake_disk.compute_scattered_light()
    max_disk = np.max(fake_disk_map)
    aarc(max_disk, 1.0)
    
@parametrize("g",
             [
                param(0),
                param(0.4),
                param(0.9)
             ])
def test_HG_phase_function_and_print(g):
    """
    Verify that the scattered light image of a fake disk with a Heynyey-Greenstein
    phase function work nominally (test done on the sum of the 
    pixel values of the image) and that the print function and plot function 
    of the phase function submodule also works.
    """
    def t_expected(g):
        """
        Expected positions.
        """
        if g == 0:
            return 0.01451614524684395
        elif g == 0.4:
            return 0.012487996770272992
        elif g == 0.9:
            return 0.0037231833830253993
    pixel_scale=0.01225 # pixel scale in arcsec/px
    nx = 121 # number of pixels of your image in X
    ny = 121 # number of pixels of your image in Y
    dstar=85. # distance to the star in pc
    PA = -75. # position angle in degree
    itilt=80. # inclination in degree
    gamma=2.
    ksi0=1.
    r0=24.*pixel_scale*dstar
    ain=10.
    aout=-4.

    fake_disk = ScatteredLightDisk(\
                    nx=nx,ny=ny,distance=dstar,\
                    itilt=itilt,omega=0.,pxInArcsec=pixel_scale,pa=PA,\
                    density_dico={'name':'2PowerLaws','ain':ain,'aout':aout,\
                    'a':r0,'e':0.0,'ksi0':ksi0,'gamma':gamma,'beta':1.},\
                    spf_dico={'name':'HG', 'g':g, 'polar':True})
    fake_disk.print_info()
    fake_disk.phase_function.print_info()
    fake_disk.phase_function.plot_phase_function()
    fake_disk_map = fake_disk.compute_scattered_light()
    aarc(np.sum(fake_disk_map), t_expected(g))

@parametrize("g1",
             [
                param(0.2),
                param(0.4),
                param(0.9)
             ])
def test_double_HG_phase_function(g1):
    """
    Verify that the scattered light image of a fake disk with a double 
    Heynyey-Greenstein phase function works nominally (test done on the sum of the 
    pixel values of the image)
    """
    def t_expected(g1):
        """
        Expected positions.
        """
        if g1 == 0.2:
            return 1325.6664013360682
        elif g1 == 0.4:
            return 1256.5483780548618
        elif g1 == 0.9:
            return 1266.6841189508734
    pixel_scale=0.01225 # pixel scale in arcsec/px
    nx = 99 # number of pixels of your image in X
    ny = 99 # number of pixels of your image in Y
    dstar=80. # distance to the star in pc
    PA = -12. # position angle in degree
    itilt=10. # inclination in degree
    gamma=2.
    ksi0=1.
    r0=24.*pixel_scale*dstar
    ain=9.
    aout=-6.
    spf_dico={'name':'DoubleHG','g':[g1,-0.6],'weight':0.7,\
              'polar':False}

    fake_disk = ScatteredLightDisk(\
                    nx=nx,ny=ny,distance=dstar,\
                    itilt=itilt,omega=0.,pxInArcsec=pixel_scale,pa=PA,\
                    density_dico={'name':'2PowerLaws','ain':ain,'aout':aout,\
                    'a':r0,'e':0.0,'ksi0':ksi0,'gamma':gamma,'beta':1.},\
                    spf_dico=spf_dico,flux_max=1)
    fake_disk_map = fake_disk.compute_scattered_light()
    aarc(np.sum(fake_disk_map), t_expected(g1))

@parametrize("spf_50deg",
             [
                param(20),
                param(60),
                param(10)
             ])
def test_homemade_phase_function(spf_50deg):
    """
    Verify that the scattered light image of a fake disk with a home-made 
    phase function works nominally (test done on the sum of the 
    pixel values of the image)
    """
    def t_expected(spf_50deg):
        """
        Expected positions.
        """
        if spf_50deg == 20:
            return 9.540932
        elif spf_50deg == 60:
            return 18.271285
        elif spf_50deg == 10:
            return 7.355497
    pixel_scale=0.01225 # pixel scale in arcsec/px
    nx = 100 # number of pixels of your image in X
    ny = 100 # number of pixels of your image in Y
    dstar=77. # distance to the star in pc
    PA = 27. # position angle in degree
    itilt=78. # inclination in degree
    gamma=2.
    ksi0=1.
    r0=24.*pixel_scale*dstar
    ain=3.
    aout=-11.
    spf_angles = np.array([0.,50.,90.,130.,180.])
    spf_values = np.array([100,spf_50deg, 10, 5,10]) 
    spf_dico = {'phi':spf_angles,'spf':spf_values,'name':'interpolated',\
                'polar':False,'kind':'cubic'}
    fake_disk = ScatteredLightDisk(\
                    nx=nx,ny=ny,distance=dstar,\
                    itilt=itilt,omega=0.,pxInArcsec=pixel_scale,pa=PA,\
                    density_dico={'name':'2PowerLaws','ain':ain,'aout':aout,\
                    'a':r0,'e':0.0,'ksi0':ksi0,'gamma':gamma,'beta':1.},\
                    spf_dico=spf_dico)
    fake_disk_map = fake_disk.compute_scattered_light()
    fake_disk.phase_function.plot_phase_function()
    measured_value = np.sum(fake_disk_map)
    aarc(measured_value, t_expected(spf_50deg),rtol=1e-3,atol=1e-3)
