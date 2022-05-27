"""
Tests for fm/fakecomp.py

"""

from .helpers import aarc, np, param, parametrize, fixture, filterwarnings
from vip_hci.fm import cube_inject_companions, normalize_psf


# ===== utility functions

@fixture(scope="module", params=["3D", "4D"])
def dataset(request):
    """
    Create 3D and 4D datasets for use with ``test_cube_inject_companions``.

    """
    if request.param == "3D":
        cube = np.zeros((3, 25, 25))
        psf = np.ones((1, 1))
    elif request.param == "4D":
        cube = np.zeros((2, 3, 25, 25))  # lambda, frames, width, height
        psf = np.ones((2, 1, 1))

    angles = np.array([0, 90, 180])

    return cube, psf, angles


@parametrize("branches, dists",
             [
                param(1, 2, id="1br-2"),
                param(2, 2, id="2br-2"),
                param(2, [1, 2], id="2br-[1,2]")
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
            return [(12, 14)]
        elif branches == 2 and dists == 2:
            return [(12, 14), (12, 10)]
        elif branches == 2 and dists == [1, 2]:
            return [(12, 13), (12, 14), (12, 11), (12, 10)]
        else:
            raise ValueError("no expected result defined")

    cube, psf, angles = dataset

    c, yx = cube_inject_companions(cube, psf_template=psf, angle_list=angles,
                                   rad_dists=dists, n_branches=branches,
                                   flevel=3, full_output=True, plsc=0.999,
                                   verbose=True)
    yx_expected = _expected(branches, dists)

    aarc(yx, yx_expected)


@filterwarnings("ignore:invalid value encountered in true_divide")
def test_normalize_psf_shapes():
    """
    Test if normalize_psf produces the expected shapes.
    """
    # `Force_odd` is True therefore `size` was set to 19
    res_even = normalize_psf(np.ones((20, 20)), size=18)
    res_odd = normalize_psf(np.ones((21, 21)), size=18)
    assert res_even.shape == res_odd.shape == (19, 19)

    res_even = normalize_psf(np.ones((20, 20)), size=18, force_odd=False)
    res_odd = normalize_psf(np.ones((21, 21)), size=18, force_odd=False)
    assert res_even.shape == res_odd.shape == (18, 18)

    # set to odd size
    res_even = normalize_psf(np.ones((20, 20)), size=19)
    res_odd = normalize_psf(np.ones((21, 21)), size=19)
    assert res_even.shape == res_odd.shape == (19, 19)

    res_even = normalize_psf(np.ones((20, 20)), size=19, force_odd=False)
    res_odd = normalize_psf(np.ones((21, 21)), size=19, force_odd=False)
    assert res_even.shape == res_odd.shape == (19, 19)
