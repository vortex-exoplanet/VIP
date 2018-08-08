from __future__ import division, print_function, absolute_import

from helpers import np, pytest
from helpers import aarc


from vip_hci.metrics.fakecomp import cube_inject_companions


# ===== utility functions


@pytest.fixture(scope="module", params=["3D", "4D"])
def dataset(request):
    """
    Create 3D and 4D datasets for use with ``test_cube_inject_companions``.

    """
    if request.param == "3D":
        cube = np.zeros((3, 5, 5))
        psf = np.ones((1, 1))
    elif request.param == "4D":
        cube = np.zeros((2, 3, 5, 5))  # lambda, frames, width, height
        psf = np.ones((2, 1, 1))

    angles = np.array([0, 90, 180])

    return cube, psf, angles


@pytest.mark.parametrize("branches, dists",
                         [
                            pytest.param(1, 2, id="1br-2"),
                            pytest.param(2, 2, id="2br-2"),
                            pytest.param(2, [1, 2], id="2br-[1,2]")
                         ])
def test_cube_inject_companions(dataset, branches, dists):
    """
    Verify position of injected companions, for 3D and 4D cases.
    """
    def _expected(branches, dists):
        """
        Expected positions.
        """
        if branches == 1 and dists == 2:
            return [(2, 4)]
        elif branches == 2 and dists == 2:
            return [(2, 4), (2, 0)]
        elif branches == 2 and dists == [1, 2]:
            return [(2, 3), (2, 4), (2, 1), (2, 0)]

        else:
            raise ValueError("no expected result defined")

    cube, psf, angles = dataset

    c, yx = cube_inject_companions(cube, psf_template=psf, angle_list=angles,
                                   rad_dists=dists, n_branches=branches,
                                   flevel=3,
                                   full_output=True,
                                   plsc=1, verbose=True)
    yx_expected = _expected(branches, dists)

    aarc(yx, yx_expected)
