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
