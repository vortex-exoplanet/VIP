"""Test for the object dedicated to the PostProc results."""

__author__ = "Thomas Bédrine"

import numpy as np
import pytest

from vip_hci.objects import PPResult


@pytest.mark.xdist_group(name="group1")
def test_results_object():
    """Create a PPResult object and test various actions on it."""
    results = PPResult()

    # Test if no session registered triggers an attribute error

    with pytest.raises(AttributeError) as excinfo:
        results.show_session_results()

    assert "No session" in str(excinfo.value)

    v_frame = np.zeros((10, 10))
    v_cube = np.zeros((5, 10, 10))
    v_params = {"param1": None, "param2": None, "param3": 1.0}
    f_params = {"param1": None, "param2": None, "param3": 1.0, "param4": v_cube}
    v_snr = np.ones((10, 10))
    v_algo = "pca"

    # Test if every information is registered correctly

    results.register_session(frame=v_frame, algo_name=v_algo, params=f_params)

    assert np.allclose(np.abs(results.sessions[-1].frame), np.abs(v_frame))
    assert results.sessions[-1].snr_map is None
    assert results.sessions[-1].parameters == v_params
    assert results.sessions[-1].algo_name == v_algo
    assert len(results.sessions) == 1

    # Test if registering new information does only update the corresponding session

    results.register_session(
        frame=v_frame, algo_name=v_algo, params=f_params, snr_map=v_snr
    )

    assert np.allclose(np.abs(results.sessions[-1].frame), np.abs(v_frame))
    assert np.allclose(np.abs(results.sessions[-1].snr_map), np.abs(v_snr))
    assert results.sessions[-1].parameters == v_params
    assert results.sessions[-1].algo_name == v_algo
    assert len(results.sessions) == 1

    # Test if giving a bad parameter to show_session_results triggers ValueError

    with pytest.raises(ValueError) as excinfo:
        results.show_session_results(session_id=1.5)

    assert "session ID isn't an integer" in str(excinfo.value)
