"""
Tests for pca/svd.py.

"""

import numpy as np

from vip_hci.psfsub.svd import svd_wrapper


def test_svd_recons():
    random_state = np.random.RandomState(42)
    mat = random_state.randn(20, 100)
    ncomp = 20
    U, S, V = svd_wrapper(mat, mode='lapack', ncomp=ncomp, verbose=False,
                          full_output=True)
    print(U.shape)
    print(S.shape)
    print(V.shape)
    rec_matrix = np.dot(U, np.dot(np.diag(S), V))
    assert np.allclose(np.abs(mat), np.abs(rec_matrix), atol=1.e-2)
