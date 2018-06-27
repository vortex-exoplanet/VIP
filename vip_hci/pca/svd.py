#! /usr/bin/env python

"""
Module with functions for computing SVDs.
"""

from __future__ import division
from __future__ import print_function

__author__ = 'Carlos Alberto Gomez Gonzalez'
__all__ = ['svd_wrapper',
           'randomized_svd_gpu',
           'get_eigenvectors']

import warnings
try:
    import cupy
    no_cupy = False
except ImportError:
    msg = "Cupy not found. Have a GPU? Consider setting up a CUDA environment "
    msg += "and installing cupy >= 2.0.0"
    warnings.warn(msg, ImportWarning)
    no_cupy = True
try:
    import torch
    no_torch = False
except ImportError:
    msg = "Pytorch not found. Have a GPU? Consider setting up a CUDA "
    msg += "environment and installing pytorch"
    warnings.warn(msg, ImportWarning)
    no_torch = True

import numpy as np
from numpy import linalg
from matplotlib import pyplot as plt 
from scipy.sparse.linalg import svds
from sklearn.decomposition import randomized_svd
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.utils import check_random_state
from ..var import matrix_scaling, prepare_matrix


def svd_wrapper(matrix, mode, ncomp, debug, verbose, usv=False,
                random_state=None, to_numpy=True):
    """ Wrapper for different SVD libraries (CPU and GPU). 
      
    Parameters
    ----------
    matrix : array_like, 2d
        2d input matrix.
    mode : {'lapack', 'arpack', 'eigen', 'randsvd', 'cupy', 'eigencupy',
            'randcupy', 'pytorch', 'eigenpytorch', 'randpytorch'}, str optional
        Switch for the SVD method/library to be used. ``lapack`` uses the LAPACK 
        linear algebra library through Numpy and it is the most conventional way 
        of computing the SVD (deterministic result computed on CPU). ``arpack`` 
        uses the ARPACK Fortran libraries accessible through Scipy (computation
        on CPU). ``eigen`` computes the singular vectors through the 
        eigendecomposition of the covariance M.M' (computation on CPU).
        ``randsvd`` uses the randomized_svd algorithm implemented in Sklearn 
        (computation on CPU). ``cupy`` uses the Cupy library for GPU computation
        of the SVD as in the LAPACK version. ``eigencupy`` offers the same 
        method as with the ``eigen`` option but on GPU (through Cupy). 
        ``randcupy`` is an adaptation f the randomized_svd algorithm, where all
        the computations are done on a GPU (through Cupy). ``pytorch`` uses the
        Pytorch library for GPU computation of the SVD. ``eigenpytorch`` offers
        the same method as with the ``eigen`` option but on GPU (through
        Pytorch). ``randpytorch`` is an adaptation of the randomized_svd
        algorithm, where all the linear algebra computations are done on a GPU
        (through Pytorch).
    ncomp : int
        Number of singular vectors to be obtained. In the cases when the full
        SVD is computed (LAPACK, ARPACK, EIGEN, CUPY), the matrix of singular 
        vectors is truncated. 
    debug : bool
        If True the explained variance ratio is computed and displayed.
    verbose: bool
        If True intermediate information is printed out.
    usv : bool optional
        If True the 3 terms of the SVD factorization are returned.
    random_state : int, RandomState instance or None, optional
        If int, random_state is the seed used by the random number generator.
        If RandomState instance, random_state is the random number generator.
        If None, the random number generator is the RandomState instance used
        by np.random. Used for ``randsvd`` mode.
    to_numpy : bool, optional
        If True (by default) the arrays computed in GPU are transferred from
        VRAM and converted to numpy ndarrays.

    Returns
    -------
    V : array_like
        The right singular vectors of the input matrix. If ``usv`` is True it
        returns the left and right singular vectors and the singular values of
        the input matrix.
    
    References
    ----------
    * For ``lapack`` SVD mode see:
        https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.linalg.svd.html
        http://www.netlib.org/lapack/
    * For ``eigen`` mode see:
        https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.linalg.eigh.html
    * For ``arpack`` SVD mode see:
        https://docs.scipy.org/doc/scipy-0.19.1/reference/generated/scipy.sparse.linalg.svds.html
        http://www.caam.rice.edu/software/ARPACK/
    * For ``randsvd`` SVD mode see:
        https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/utils/extmath.py
        Finding structure with randomness: Stochastic algorithms for constructing
        approximate matrix decompositions
        Halko, et al., 2009 http://arxiv.org/abs/arXiv:0909.4061
    * For ``cupy`` SVD mode see:
        https://docs-cupy.chainer.org/en/stable/reference/generated/cupy.linalg.svd.html
    * For ``eigencupy`` mode see:
        https://docs-cupy.chainer.org/en/master/reference/generated/cupy.linalg.eigh.html
    * For ``pytorch`` SVD mode see:
        http://pytorch.org/docs/master/torch.html#torch.svd
    * For ``eigenpytorch`` mode see:
        http://pytorch.org/docs/master/torch.html#torch.eig

    """

    def reconstruction(ncomp, U, S, V, var=1):
        if mode == 'lapack':
            rec_matrix = np.dot(U[:, :ncomp],
                                np.dot(np.diag(S[:ncomp]), V[:ncomp]))
            rec_matrix = rec_matrix.T
            print('  Matrix reconstruction with {} PCs:'.format(ncomp))
            print('  Mean Absolute Error =', MAE(matrix, rec_matrix))
            print('  Mean Squared Error =', MSE(matrix, rec_matrix))

            # see https://github.com/scikit-learn/scikit-learn/blob/c3980bcbabd9d2527548820581725df2904e4a0d/sklearn/decomposition/pca.py
            exp_var = (S ** 2) / (S.shape[0] - 1)
            full_var = np.sum(exp_var)
            explained_variance_ratio = exp_var / full_var   # % of variance explained by each PC
            ratio_cumsum = np.cumsum(explained_variance_ratio)
        elif mode == 'eigen':
            exp_var = (S ** 2) / (S.shape[0] - 1)
            full_var = np.sum(exp_var)
            explained_variance_ratio = exp_var / full_var   # % of variance explained by each PC
            ratio_cumsum = np.cumsum(explained_variance_ratio)
        else:
            rec_matrix = np.dot(U, np.dot(np.diag(S), V))
            print('  Matrix reconstruction MAE =', MAE(matrix, rec_matrix))
            exp_var = (S ** 2) / (S.shape[0] - 1)
            full_var = np.var(matrix, axis=0).sum()
            explained_variance_ratio = exp_var / full_var   # % of variance explained by each PC
            if var == 1:
                pass
            else:
                explained_variance_ratio = explained_variance_ratio[::-1]
            ratio_cumsum = np.cumsum(explained_variance_ratio)
            msg = '  This info makes sense when the matrix is mean centered '
            msg += '(temp-mean scaling)'
            print(msg)

        lw = 2; alpha = 0.4
        fig = plt.figure(figsize=vip_figsize)
        fig.subplots_adjust(wspace=0.4)
        ax1 = plt.subplot2grid((1, 3), (0, 0), colspan=2)
        ax1.step(range(explained_variance_ratio.shape[0]),
                 explained_variance_ratio, alpha=alpha, where='mid',
                 label='Individual EVR', lw=lw)
        ax1.plot(ratio_cumsum, '.-', alpha=alpha,
                 label='Cumulative EVR', lw=lw)
        ax1.legend(loc='best', frameon=False, fontsize='medium')
        ax1.set_ylabel('Explained variance ratio (EVR)')
        ax1.set_xlabel('Principal components')
        ax1.grid(linestyle='solid', alpha=0.2)
        ax1.set_xlim(-10, explained_variance_ratio.shape[0] + 10)
        ax1.set_ylim(0, 1)

        trunc = 20
        ax2 = plt.subplot2grid((1, 3), (0, 2), colspan=1)
        # plt.setp(ax2.get_yticklabels(), visible=False)
        ax2.step(range(trunc), explained_variance_ratio[:trunc], alpha=alpha,
                 where='mid', lw=lw)
        ax2.plot(ratio_cumsum[:trunc], '.-', alpha=alpha, lw=lw)
        ax2.set_xlabel('Principal components')
        ax2.grid(linestyle='solid', alpha=0.2)
        ax2.set_xlim(-2, trunc + 2)
        ax2.set_ylim(0, 1)

        msg = '  Cumulative explained variance ratio for {} PCs = {:.5f}'
        # plt.savefig('figure.pdf', dpi=300, bbox_inches='tight')
        print(msg.format(ncomp, ratio_cumsum[ncomp - 1]))

    # --------------------------------------------------------------------------

    if matrix.ndim != 2:
        raise TypeError('Input matrix is not a 2d array')

    if usv:
        if mode not in ('lapack', 'arpack', 'randsvd', 'cupy', 'randcupy',
                        'pytorch', 'randpytorch'):
            msg = "Returning USV is supported with modes lapack, arpack, "
            msg += "randsvd, cupy, randcupy, pytorch or randpytorch"
            raise ValueError(msg)

    if ncomp > min(matrix.shape[0], matrix.shape[1]):
        msg = '{} PCs cannot be obtained from a matrix with size [{},{}].'
        msg += ' Increase the size of the patches or request less PCs'
        raise RuntimeError(msg.format(ncomp, matrix.shape[0], matrix.shape[1]))

    if mode == 'eigen':
        # building the covariance as np.dot(matrix.T,matrix) is slower and takes more memory
        C = np.dot(matrix, matrix.T)        # covariance matrix
        e, EV = linalg.eigh(C)              # eigenvalues and eigenvectors
        pc = np.dot(EV.T, matrix)           # PCs using a compact trick when cov is MM'
        V = pc[::-1]                        # reverse since last eigenvectors are the ones we want
        S = np.sqrt(e)[::-1]                # reverse since eigenvalues are in increasing order
        if debug:
            reconstruction(ncomp, None, S, None)
        for i in range(V.shape[1]):
            V[:, i] /= S                    # scaling by the square root of eigenvalues
        V = V[:ncomp]
        if verbose:
            print('Done PCA with numpy linalg eigh functions')

    elif mode == 'lapack':
        # n_frames is usually smaller than n_pixels. In this setting taking the SVD of M'
        # and keeping the left (transposed) SVs is faster than taking the SVD of M (right SVs)
        U, S, V = linalg.svd(matrix.T, full_matrices=False)
        if debug:
            reconstruction(ncomp, U, S, V)
        V = V[:ncomp]                       # we cut projection matrix according to the # of PCs
        U = U[:, :ncomp]
        S = S[:ncomp]
        if verbose:
            print('Done SVD/PCA with numpy SVD (LAPACK)')

    elif mode == 'arpack':
        U, S, V = svds(matrix, k=ncomp)
        if debug:
            reconstruction(ncomp, U, S, V, -1)
        if verbose:
            print('Done SVD/PCA with scipy sparse SVD (ARPACK)')

    elif mode == 'randsvd':
        U, S, V = randomized_svd(matrix, n_components=ncomp, n_iter=2,
                                 transpose='auto', random_state=random_state)
        if debug:
            reconstruction(ncomp, U, S, V)
        if verbose:
            print('Done SVD/PCA with randomized SVD')

    elif mode == 'cupy':
        if no_cupy:
            raise RuntimeError('Cupy is not installed')
        a_gpu = cupy.array(matrix)
        a_gpu = cupy.asarray(a_gpu)  # move the data to the current device
        u_gpu, s_gpu, vh_gpu = cupy.linalg.svd(a_gpu, full_matrices=True,
                                               compute_uv=True)
        V = vh_gpu[:ncomp]
        if to_numpy:
            V = cupy.asnumpy(V)
        if usv:
            S = s_gpu[:ncomp]
            if to_numpy:
                S = cupy.asnumpy(S)
            U = u_gpu[:, :ncomp]
            if to_numpy:
                U = cupy.asnumpy(U)
        if verbose:
            print('Done SVD/PCA with cupy (GPU)')

    elif mode == 'randcupy':
        if no_cupy:
            raise RuntimeError('Cupy is not installed')
        U, S, V = randomized_svd_gpu(matrix, ncomp, n_iter=2, lib='cupy')
        if to_numpy:
            V = cupy.asnumpy(V)
            S = cupy.asnumpy(S)
            U = cupy.asnumpy(U)
        if debug:
            reconstruction(ncomp, U, S, V)
        if verbose:
            print('Done randomized SVD/PCA with cupy (GPU)')

    elif mode == 'eigencupy':
        if no_cupy:
            raise RuntimeError('Cupy is not installed')
        a_gpu = cupy.array(matrix)
        a_gpu = cupy.asarray(a_gpu)         # move the data to the current device
        C = cupy.dot(a_gpu, a_gpu.T)        # covariance matrix
        e, EV = cupy.linalg.eigh(C)         # eigenvalues and eigenvectors
        pc = cupy.dot(EV.T, a_gpu)          # PCs using a compact trick when cov is MM'
        V = pc[::-1]                        # reverse since last eigenvectors are the ones we want
        S = cupy.sqrt(e)[::-1]              # reverse since eigenvalues are in increasing order
        if debug:
            reconstruction(ncomp, None, S, None)
        for i in range(V.shape[1]):
            V[:, i] /= S                    # scaling by the square root of eigenvalues
        V = V[:ncomp]
        if to_numpy:
            V = cupy.asnumpy(V)
        if verbose:
            print('Done PCA with cupy eigh function (GPU)')

    elif mode == 'pytorch':
        if no_torch:
            raise RuntimeError('Pytorch is not installed')
        a_gpu = torch.Tensor.cuda(torch.from_numpy(matrix.astype('float32').T))
        u_gpu, s_gpu, vh_gpu = torch.svd(a_gpu)
        V = vh_gpu[:ncomp]
        S = s_gpu[:ncomp]
        U = torch.transpose(u_gpu, 0, 1)[:ncomp]
        if to_numpy:
            V = np.array(V)
            S = np.array(S)
            U = np.array(U)
        if verbose:
            print('Done SVD/PCA with pytorch (GPU)')

    elif mode == 'eigenpytorch':
        if no_torch:
            raise RuntimeError('Pytorch is not installed')
        a_gpu = torch.Tensor.cuda(torch.from_numpy(matrix.astype('float32')))
        C = torch.mm(a_gpu, torch.transpose(a_gpu, 0, 1))
        e, EV = torch.eig(C, eigenvectors=True)
        V = torch.mm(torch.transpose(EV, 0, 1), a_gpu)
        S = torch.sqrt(e[:, 0])
        if debug:
            reconstruction(ncomp, None, S, None)
        for i in range(V.shape[1]):
            V[:, i] /= S
        V = V[:ncomp]
        if to_numpy:
            V = np.array(V)
        if verbose:
            print('Done PCA with pytorch eig function')

    elif mode == 'randpytorch':
        if no_torch:
            raise RuntimeError('Pytorch is not installed')
        U, S, V = randomized_svd_gpu(matrix, ncomp, n_iter=2, lib='pytorch')
        if to_numpy:
            V = np.array(V)
            S = np.array(S)
            U = np.array(U)
        if debug:
            reconstruction(ncomp, U, S, V)
        if verbose:
            print('Done randomized SVD/PCA with randomized pytorch (GPU)')

    else:
        raise ValueError('The SVD mode is not available')

    if usv:
        if mode == 'lapack':
            return V.T, S, U.T
        elif mode == 'pytorch':
            if to_numpy:
                return V.T, S, U.T
            else:
                return torch.transpose(V, 0, 1), S, torch.transpose(U, 0, 1)
        else:
            return U, S, V
    else:
        if mode == 'lapack':
            return U.T
        elif mode == 'pytorch':
            return U
        else:
            return V


def get_eigenvectors(ncomp, data, svd_mode, mode='noise', noise_error=1e-3,
                     cevr=0.9, max_evs=None, data_ref=None, debug=False,
                     collapse=False):
    """ Getting ``ncomp`` eigenvectors. Choosing the size of the PCA truncation
    when ``ncomp`` is set to None.
    """
    no_dataref = False
    if data_ref is None:
        no_dataref = True
        data_ref = data

    if max_evs is None:
        max_evs = min(data_ref.shape[0], data_ref.shape[1])

    if ncomp == 'auto':
        ncomp = 0
        V_big = svd_wrapper(data_ref, svd_mode, max_evs, False, False)

        if mode=='noise':
            if not collapse:
                data_ref_sc = matrix_scaling(data_ref, 'temp-mean')
                data_sc = matrix_scaling(data, 'temp-mean')
            else:
                data_ref_sc = matrix_scaling(data_ref, 'temp-standard')
                data_sc = matrix_scaling(data, 'temp-standard')

            V_sc = svd_wrapper(data_ref_sc, svd_mode, max_evs, False, False)

            px_noise = []
            px_noise_decay = 1
            # Noise (px stddev of residuals) to be lower than a given threshold
            while px_noise_decay >= noise_error:
                ncomp += 1
                V = V_sc[:ncomp]
                if no_dataref:
                    transformed = np.dot(data_sc, V.T)
                    reconstructed = np.dot(transformed, V)
                else:
                    transformed = np.dot(V, data_sc)
                    reconstructed = np.dot(transformed.T, V).T
                residuals = data_sc - reconstructed
                if not collapse:
                    curr_noise = np.std(residuals)
                else:
                    curr_noise = np.std((np.median(residuals, axis=0)))
                px_noise.append(curr_noise)
                if ncomp > 1:
                    px_noise_decay = px_noise[-2] - curr_noise
                # print '{} {:.4f} {:.4f}'.format(ncomp, curr_noise, px_noise_decay)
            V = V_big[:ncomp]

        elif mode=='cevr':
            data_sc = matrix_scaling(data, 'temp-mean')
            _, S, _ = svd_wrapper(data_sc, svd_mode, min(data_sc.shape[0],
                                                         data_sc.shape[1]),
                                  False, False, usv=True)
            exp_var = (S ** 2) / (S.shape[0] - 1)
            full_var = np.sum(exp_var)
            # % of variance explained by each PC
            explained_variance_ratio = exp_var / full_var
            ratio_cumsum = np.cumsum(explained_variance_ratio)
            ncomp = np.searchsorted(ratio_cumsum, cevr) + 1
            V = V_big[:ncomp]

        if debug:
            print('ncomp', ncomp)

    else:
        # Performing SVD/PCA according to "svd_mode" flag
        ncomp = min(ncomp, min(data_ref.shape[0], data_ref.shape[1]))
        V = svd_wrapper(data_ref, svd_mode, ncomp, debug=False, verbose=False)

    return V


def _get_cumexpvar(cube, expvar_mode, inrad, outrad, size_patch, k_list=None,
                   verbose=True):
    """ Calculated the cumulative explained variance ratio for the SVD of a
    cube (either full frames or a single annulus could be used).

    # TODO : Documentation
    """
    n_frames = cube.shape[0]
    ann_width = outrad - inrad
    cent_ann = inrad + int(np.round(ann_width / 2.))
    ann_width += size_patch + 2

    if expvar_mode == 'annular':
        matrix_svd = prepare_matrix(cube, 'temp-mean', None, mode=expvar_mode,
                                    annulus_radius=cent_ann,
                                    annulus_width=ann_width, verbose=False)[0]
        U, S, V = svd_wrapper(matrix_svd, 'lapack', min(matrix_svd.shape[0],
                                                        matrix_svd.shape[1]),
                              False, False, True)
    elif expvar_mode == 'fullfr':
        matrix_svd = prepare_matrix(cube, 'temp-mean', None, mode=expvar_mode,
                                    verbose=False)
        U, S, V = svd_wrapper(matrix_svd, 'lapack', n_frames, False, False,
                              True)

    exp_var = (S ** 2) / (S.shape[0] - 1)
    full_var = np.sum(exp_var)
    # % of variance explained by each PC
    explained_variance_ratio = exp_var / full_var
    ratio_cumsum = np.cumsum(explained_variance_ratio)

    if k_list is not None:
        ratio_cumsum_klist = []
        for k in k_list:
            ratio_cumsum_klist.append(ratio_cumsum[k - 1])

        if verbose:
            print("SVD on input matrix (annulus from cube)")
            print("  Number of PCs :")
            print("  ", k_list)
            print("  Cum. explained variance ratios :")
            print("  ", ", ".join("{:.2f}".format(i) for i in
                                  ratio_cumsum_klist))
            print("")
    else:
        ratio_cumsum_klist = ratio_cumsum

    return ratio_cumsum, ratio_cumsum_klist


def randomized_svd_gpu(M, n_components, n_oversamples=10, n_iter='auto',
                       transpose='auto', random_state=0, lib='cupy'):
    """Computes a truncated randomized SVD on GPU. Adapted from Sklearn.

    Parameters
    ----------
    M : ndarray or sparse matrix
        Matrix to decompose
    n_components : int
        Number of singular values and vectors to extract.
    n_oversamples : int (default is 10)
        Additional number of random vectors to sample the range of M so as
        to ensure proper conditioning. The total number of random vectors
        used to find the range of M is n_components + n_oversamples. Smaller
        number can improve speed but can negatively impact the quality of
        approximation of singular vectors and singular values.
    n_iter : int or 'auto' (default is 'auto')
        Number of power iterations. It can be used to deal with very noisy
        problems. When 'auto', it is set to 4, unless `n_components` is small
        (< .1 * min(X.shape)) `n_iter` in which case is set to 7.
        This improves precision with few components.
    transpose : True, False or 'auto' (default)
        Whether the algorithm should be applied to M.T instead of M. The
        result should approximately be the same. The 'auto' mode will
        trigger the transposition if M.shape[1] > M.shape[0] since this
        implementation of randomized SVD tend to be a little faster in that
        case.
    random_state : int, RandomState instance or None, optional (default=None)
        The seed of the pseudo random number generator to use when shuffling
        the data.  If int, random_state is the seed used by the random number
        generator; If RandomState instance, random_state is the random number
        generator; If None, the random number generator is the RandomState
        instance used by `np.random`.
    lib : {'cupy', 'pytorch'}, str optional
        Chooses the GPU library to be used.

    Notes
    -----
    This algorithm finds a (usually very good) approximate truncated
    singular value decomposition using randomization to speed up the
    computations. It is particularly fast on large matrices on which
    you wish to extract only a small number of components. In order to
    obtain further speed up, `n_iter` can be set <=2 (at the cost of
    loss of precision).

    References
    ----------
    * Finding structure with randomness: Stochastic algorithms for constructing
      approximate matrix decompositions
      Halko, et al., 2009 http://arxiv.org/abs/arXiv:0909.4061
    * A randomized algorithm for the decomposition of matrices
      Per-Gunnar Martinsson, Vladimir Rokhlin and Mark Tygert
    * An implementation of a randomized algorithm for principal component
      analysis
      A. Szlam et al. 2014
    """
    random_state = check_random_state(random_state)
    n_random = n_components + n_oversamples
    n_samples, n_features = M.shape

    if n_iter == 'auto':
        # Checks if the number of iterations is explicitly specified
        n_iter = 7 if n_components < .1 * min(M.shape) else 4

    if transpose == 'auto':
        transpose = n_samples < n_features
    if transpose:
        M = M.T # this implementation is a bit faster with smaller shape[1]

    if lib == 'cupy':
        M = cupy.array(M)
        M = cupy.asarray(M)

        # Generating normal random vectors with shape: (M.shape[1], n_random)
        Q = random_state.normal(size=(M.shape[1], n_random))
        Q = cupy.array(Q)
        Q = cupy.asarray(Q)

        # Perform power iterations with Q to further 'imprint' the top
        # singular vectors of M in Q
        for i in range(n_iter):
            Q = cupy.dot(M, Q)
            Q = cupy.dot(M.T, Q)

        # Sample the range of M using by linear projection of Q. Extract an orthonormal basis
        Q, _ = cupy.linalg.qr(cupy.dot(M, Q), mode='reduced')

        # project M to the (k + p) dimensional space using the basis vectors
        B = cupy.dot(Q.T, M)

        B = cupy.array(B)
        Q = cupy.array(Q)
        # compute the SVD on the thin matrix: (k + p) wide
        Uhat, s, V = cupy.linalg.svd(B, full_matrices=False, compute_uv=True)
        del B
        U = cupy.dot(Q, Uhat)

        if transpose:
            # transpose back the results according to the input convention
            return V[:n_components, :].T, s[:n_components], U[:,
                                                            :n_components].T
        else:
            return U[:, :n_components], s[:n_components], V[:n_components, :]

    elif lib == 'pytorch':
        M_gpu = torch.Tensor.cuda(torch.from_numpy(M.astype('float32')))

        # Generating normal random vectors with shape: (M.shape[1], n_random)
        Q = torch.cuda.FloatTensor(M_gpu.shape[1], n_random).normal_()

        # Perform power iterations with Q to further 'imprint' the top
        # singular vectors of M in Q
        for i in range(n_iter):
            Q = torch.mm(M_gpu, Q)
            Q = torch.mm(torch.transpose(M_gpu, 0, 1), Q)

        # Sample the range of M using by linear projection of Q. Extract an orthonormal basis
        Q, _ = torch.qr(torch.mm(M_gpu, Q))

        # project M to the (k + p) dimensional space using the basis vectors
        B = torch.mm(torch.transpose(Q, 0, 1), M_gpu)

        # compute the SVD on the thin matrix: (k + p) wide
        Uhat, s, V = torch.svd(B)
        del B
        U = torch.mm(Q, Uhat)

        if transpose:
            # transpose back the results according to the input convention
            return (torch.transpose(V[:n_components, :], 0, 1),
                    s[:n_components],
                    torch.transpose(U[:, :n_components], 0, 1))
        else:
            return U[:, :n_components], s[:n_components], V[:n_components, :]


