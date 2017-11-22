from __future__ import division, print_function

import numpy as np
import pytest

from vip_hci.pca import svd_wrapper


def test_svd_recons():
    random_state = np.random.RandomState(42)
    mat = random_state.randn(20,100)
    ncomp = 20
    U, S, V = svd_wrapper(mat, 'lapack', ncomp, False, False, usv=True)
    print(U.shape)
    print(S.shape)
    print(V.shape)
    rec_matrix = np.dot(U, np.dot(np.diag(S), V))
    #print(np.abs(mat))
    #print(np.abs(rec_matrix))
    assert np.allclose(np.abs(mat), np.abs(rec_matrix), atol=1.e-2)


#def test_svd_libs():
