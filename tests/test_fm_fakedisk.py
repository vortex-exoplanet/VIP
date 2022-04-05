"""
Tests for fm/fakecomp.py

"""

from .helpers import aarc, np, fixture
from vip_hci.fm import cube_inject_fakedisk, cube_inject_trace


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
    coords = []
    for i in range(cube.shape[0]):
        max_idx = np.argmax(cube[i])
        coords.append(np.unravel_index(max_idx, cube[0].shape))

    yx_expected = _expected()

    aarc(coords, yx_expected)


def test_cube_inject_trace(dataset):
    """
    Verify position of injected disk image with 1 value, for 3D and 4D cases.
    """
    def _expected(ang):
        """
        Expected positions.
        """
        if ang == 0:
            return [(7, 12), (12, 8), (15, 12)]
        elif ang == 90:
            return [(12, 7), (12, 15), (16, 12)]
        elif ang == 180:
            return [(9, 12), (12, 16), (17, 12)]

    cube, psf, angles = dataset
    
    rads = [3,4,5]
    thetas = [90,180,270]
    
    cube = cube_inject_trace(cube, psf, angles, flevel=1, rad_dists=rads, 
                             theta=thetas, plsc=0.01225, n_branches=1, 
                             imlib='vip-fft', interpolation='lanczos4', 
                             verbose=True)
    
    for i in range(cube.shape[0]):
        # find coords of trace in each image of the cube
        coords = []
        nspi = len(rads)
        frame_tmp = cube[i].copy()
        for s in range(nspi):
            max_idx = np.argmax(frame_tmp)
            coords_tmp = np.unravel_index(max_idx, frame_tmp.shape)
            coords.append(coords_tmp)
            frame_tmp[coords_tmp]=0
        idx_order = np.argsort(np.sum(coords,axis=1))
        coords_sort = [coords[i] for i in idx_order]
        
        yx_expected = _expected(angles[i])
    
        aarc(coords_sort, yx_expected)
