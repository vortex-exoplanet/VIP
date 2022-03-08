"""
Tests for metrics/stim.py

"""

import copy
from .helpers import fixture, np
from vip_hci.psfsub import pca
from vip_hci.hci_dataset import Frame
from vip_hci.metrics import stim_map, inverse_stim_map


@fixture(scope="module")
def get_frame(example_dataset_adi):
    """
    Inject a fake companion into an example cube.

    Parameters
    ----------
    example_dataset_adi : fixture
        Taken automatically from ``conftest.py``.

    Returns
    -------
    frame : VIP Frame
    planet_position : tuple(y, x)

    """
    dsi = copy.copy(example_dataset_adi)
    # we chose a shallow copy, as we will not use any in-place operations
    # (like +=). Using `deepcopy` would be safer, but consume more memory.

    print("producing a final frame...")
    res = pca(dsi.cube, dsi.angles, ncomp=10, full_output=True)
    res_frame = res[0]
    res_cube = res[-2]
    res_der_cube = res[-1]
    frame = Frame(res_frame, fwhm=dsi.fwhm)
    return frame, (63, 63), res_cube, res_der_cube, dsi.angles


atol = 2
plot = False

def test_stimmap(get_frame):
    frame, positions, _, res_der_cube, _ = get_frame
    y0, x0 = positions
    stimap = stim_map(res_der_cube)
    y1, x1 = np.where(stimap == stimap.max())
    assert np.allclose(x1, x0, atol=atol) and np.allclose(y1, y0, atol=atol)

def test_normstimmap(get_frame):
    frame, positions, res_cube, res_der_cube, angles = get_frame
    y0, x0 = positions
    stimap = stim_map(res_der_cube)
    inv_stimap = inverse_stim_map(res_cube, angles)
    norm_stimap = stimap/np.amax(inv_stimap)
    y1, x1 = np.where(norm_stimap == norm_stimap.max())
    assert np.allclose(x1, x0, atol=atol) and np.allclose(y1, y0, atol=atol)
