"""
Tests for the andromeda submodule.
"""

from __future__ import division, print_function, absolute_import

__author__ = "Ralf Farkas"

from helpers import np, aarc
import os
from vip_hci.andromeda.andromeda import andromeda
from vip_hci.andromeda.utils import (
    robust_std,
    idl_round,
    idl_where
)


CURRDIR = os.path.dirname(__file__)

try:
    IDL_DATA = np.load(os.path.join(CURRDIR, "andromeda_idl.npz"))
except FileNotFoundError:
    print("Could not load andromeda_idl.npz. Try running generate_test_data() "
          "to create it.")


def test_andromeda(example_dataset):
    # TODO: test `fast`
    # TODO: test `nsnmooth_snr`
    # TODO: test `homogeneous_variance`
    # TODO: test `filtering_fraction`
    # TODO: test `oversampling_fact`
    # TODO: test `min_sep`
    # TODO: test `opt_method`

    global IDL_DATA

    out = andromeda(example_dataset.cube,
                    angles=example_dataset.angles,
                    psf=example_dataset.psf,
                    oversampling_fact=1,
                    filtering_fraction=1,  # turn off high pass
                    min_sep=0.3,
                    opt_method="no",
                    nsmooth_snr=0,  # turn off smoothing
                    homogeneous_variance=True,
                    )
    contrast, snr, snr_norm, stdcontrast, stdcontrast_norm, likelihood, _ = out

    aarc(contrast, IDL_DATA["andromeda_contrast_1"], atol=1e-2)
    aarc(snr, IDL_DATA["andromeda_snr_1"], atol=1e-4)
    aarc(snr_norm, IDL_DATA["andromeda_snr_norm_1"], atol=1e-4)
    aarc(stdcontrast, IDL_DATA["andromeda_stdcontrast_1"])
    aarc(stdcontrast_norm, IDL_DATA["andromeda_stdcontrast_norm_1"])
    aarc(likelihood, IDL_DATA["andromeda_likelihood_1"], atol=1e-4)


def test_idl_round():
    assert idl_round(3.5) == 4
    assert idl_round(4.5) == 5
    assert idl_round(-3.5) == -4
    assert idl_round(-4.5) == -5


def test_idl_where():
    a = np.arange(3*3, dtype=int).reshape((3, 3))

    aarc(idl_where(a <= 3), [0, 1, 2, 3])  # WHERE(a LE 3)
    aarc(idl_where(a == 5), [5])  # WHERE(a EQ 5)
    aarc(idl_where(a > 20), [])  # WHERE(a GT 20) -> returns [-1]


def test_robust_std():
    assert robust_std([1, 1, 1]) == 0
    assert robust_std([1, 1, 1, 10]) == 0
    assert robust_std([1, 2, 3, 4]) == 1.4825796886582654
    # (was verified with other implementations)
