"""Test for the PostProc object."""

__author__ = "Thomas Bédrine"

import numpy as np
import pytest

from vip_hci.objects import PostProc, PPResult
from vip_hci.psfsub import median_sub


def test_postproc_object():
    """Create a PostProc objects and test various actions on it."""
    t_postproc = PostProc(verbose=False)

    # Test if trying to update the dataset without any specified triggers AttributeError

    with pytest.raises(AttributeError) as excinfo:
        t_postproc._update_dataset()

    assert "No dataset was specified" in str(excinfo.value)

    # Test if trying to get parameters from non-existing results triggers AttributeError

    with pytest.raises(AttributeError) as excinfo:
        t_postproc.get_params_from_results(session_id=-1)

    assert "No results were saved yet" in str(excinfo.value)

    # Test if trying to get parameters for non-existing session triggers ValueError

    results = PPResult()
    t_postproc.results = results

    with pytest.raises(ValueError) as excinfo:
        t_postproc.get_params_from_results(session_id=-1)

    assert "ID is higher" in str(excinfo.value)

    # Test if trying to get parameters for incompatible algorithm triggers ValueError

    v_frame = np.zeros((10, 10))
    v_params = {"param1": None, "param2": None, "param3": 1.0}
    v_algo = "pca"

    results.register_session(frame=v_frame, algo_name=v_algo, params=v_params)

    t_postproc._algo_name = "median_sub"

    with pytest.raises(ValueError) as excinfo:
        t_postproc.get_params_from_results(session_id=-1)

    assert "does not match" in str(excinfo.value)

    # Test if the setup_parameters filters unnecessary parameters

    initial_params = {"fwhm": 4, "imlib": "vip-fft", "annulus_width": 5}
    expected_params = {"fwhm": 4, "imlib": "vip-fft", "verbose": False}

    set_params = t_postproc._setup_parameters(median_sub, **initial_params)

    assert set_params == expected_params

    # Test if non-overloaded run method triggers NotImplementedError

    with pytest.raises(NotImplementedError) as excinfo:
        t_postproc.run()

    assert excinfo.type == NotImplementedError
