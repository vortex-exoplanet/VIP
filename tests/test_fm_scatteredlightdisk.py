"""
Tests for fm/scattered_light_disk.py

"""

from .helpers import aarc, parametrize, param
from vip_hci.fm.scattered_light_disk import DustEllipticalDistribution2PowerLaws


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