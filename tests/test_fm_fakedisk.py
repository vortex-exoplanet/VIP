"""
Tests for fm/fakecomp.py

"""

from .helpers import aarc, np, fixture
from vip_hci.fm.fakedisk import cube_inject_fakedisk, cube_inject_trace


# ===== utility functions

@fixture(scope="module", params=["3D"])
def dataset(request):
    """
    Create 3D and 4D datasets for use with ``test_cube_inject_companions``.

    """
    if request.param == "3D":
        cube = np.zeros((3,25,25))
        psf = np.ones((1,1))

    angles = np.array([0, 90, 180])

    return cube, psf, angles


def test_cube_inject_fakedisk(dataset):
    """
    Verify position of injected disk image with 1 value, for 3D and 4D cases.
    """
    def _expected():
        """
        Expected positions.
        """
        return [(15, 12), (12, 9), (9, 12)]

    psf = np.zeros((25, 25))
    psf[15,12] = 1
    
    _, _, angles = dataset
    
    cube = cube_inject_fakedisk(psf, angle_list=angles)
    
    # find coords

    yx = np.unravel_index(np.argmax(cube, axis=0), cube[0].shape)    
    yx_expected = _expected()

    aarc(yx, yx_expected)


def test_cube_inject_trace(dataset):
    """
    Verify position of injected disk image with 1 value, for 3D and 4D cases.
    """
    def _expected():
        """
        Expected positions.
        """
        return [(15, 12), (12, 9), (9, 12)]

    cube, psf, angles = dataset
    
    rads = [3,3,3]
    thetas = [90,180,270]
    
    cube = cube_inject_trace(cube, psf, angles, flevel=1, 
                             rad_dists=rads, theta=thetas, 
                             plsc=0.01225, n_branches=1, imlib='vip-fft', 
                             interpolation='lanczos4', verbose=True)
    
    # find coords

    yx = np.unravel_index(np.argmax(cube, axis=0), cube[0].shape)    
    yx_expected = _expected()

    aarc(yx, yx_expected)
