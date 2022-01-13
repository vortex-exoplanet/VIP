#! /usr/bin/env python

"""
Module with functions for computing SVDs.
"""

__author__ = 'Carlos Alberto Gomez Gonzalez'
__all__ = ['SVDecomposer']

import warnings
try:
    import cupy
    no_cupy = False
except ImportError:
    msg = "Cupy not found. Do you have a GPU? Consider setting up a CUDA "
    msg += "environment and installing cupy >= 2.0.0"
    warnings.warn(msg, ImportWarning)
    no_cupy = True
try:
    import torch
    no_torch = False
except ImportError:
    msg = "Pytorch not found. Do you have a GPU? Consider setting up a CUDA "
    msg += "environment and installing pytorch"
    warnings.warn(msg, ImportWarning)
    no_torch = True

import numpy as np
from numpy import linalg
from matplotlib import pyplot as plt 
from scipy.sparse.linalg import svds
from sklearn.decomposition import randomized_svd
from sklearn.utils import check_random_state
from pandas import DataFrame
from ..config import timing, time_ini, sep, Progressbar
from ..var import matrix_scaling, prepare_matrix
from ..preproc import check_scal_vector, cube_crop_frames
from ..preproc import cube_rescaling_wavelengths as scwave
from ..config import vip_figsize, check_array


class SVDecomposer:
    """
    Class for SVD decomposition of 2d, 3d or 4d HCI arrays.

    Parameters
    ----------
    data : numpy ndarray
        Input array (2d, 3d or 4d).
    mode : {'fullfr', 'annular'}, optional
        Whether to use the whole frames or a single annulus.
    inrad : None or int, optional
        [mode='annular'] Inner radius.
    outrad : None or int, optional
        [mode='annular'] Outer radius.
    svd_mode : {'lapack', 'arpack', 'eigen', 'randsvd', 'cupy', 'eigencupy',
        'randcupy', 'pytorch', 'eigenpytorch', 'randpytorch'}, str optional
        Switch for the SVD method/library to be used.

        ``lapack``: uses the LAPACK linear algebra library through Numpy
        and it is the most conventional way of computing the SVD
        (deterministic result computed on CPU).

        ``arpack``: uses the ARPACK Fortran libraries accessible through
        Scipy (computation on CPU).

        ``eigen``: computes the singular vectors through the
        eigendecomposition of the covariance M.M' (computation on CPU).

        ``randsvd``: uses the randomized_svd algorithm implemented in
        Sklearn (computation on CPU).

        ``cupy``: uses the Cupy library for GPU computation of the SVD as in
        the LAPACK version. `

        `eigencupy``: offers the same method as with the ``eigen`` option
        but on GPU (through Cupy).

        ``randcupy``: is an adaptation of the randomized_svd algorithm,
        where all the computations are done on a GPU (through Cupy). `

        `pytorch``: uses the Pytorch library for GPU computation of the SVD.

        ``eigenpytorch``: offers the same method as with the ``eigen``
        option but on GPU (through Pytorch).

        ``randpytorch``: is an adaptation of the randomized_svd algorithm,
        where all the linear algebra computations are done on a GPU
        (through Pytorch).

    scaling : {None, "temp-mean", spat-mean", "temp-standard",
               "spat-standard"}, None or str optional
        Pixel-wise scaling mode using ``sklearn.preprocessing.scale``
        function. If set to None, the input matrix is left untouched.
        Otherwise:

        ``temp-mean``: temporal px-wise mean is subtracted.

        ``spat-mean``: spatial mean is subtracted.

        ``temp-standard``: temporal mean centering plus scaling pixel values
        to unit variance.

        ``spat-standard``: spatial mean centering plus scaling pixel values
        to unit variance.

    wavelengths : numpy ndarray, optional
        Wavelengths in case of a 4d HCI cube. These are used to compute
        scaling factors for re-scaling the spectral channels and aligning
        the speckles.

    verbose : bool, optional
        If True intermediate messages and timing are printed.

    Notes
    -----
    For info on CEVR search: # Get variance explained by singular values in
    https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/decomposition/pca.py
    """
    def __init__(self, data, mode='fullfr', inrad=None, outrad=None,
                 svd_mode='lapack', scaling='temp-standard', wavelengths=None,
                 verbose=True):
        """
        """
        check_array(data, (2, 3, 4), msg='data')
        self.data = data
        self.mode = mode
        self.svd_mode = svd_mode
        self.inrad = inrad
        self.outrad = outrad
        self.scaling = scaling
        self.wavelengths = wavelengths
        self.verbose = verbose

        if self.mode == 'annular':
            if inrad is None:
                raise ValueError("`inrad` must be a positive integer")
            if outrad is None:
                raise ValueError("`outrad` must be a positive integer")

        if self.verbose:
            print(sep)

    def generate_matrix(self):
        """
        Generate a matrix from the input ``data``. Pixel values in the matrix
        are scaled. Depending on ``mode``, the matrix can come from an annulus
        instead of the whole frames.
        """
        start_time = time_ini(False)
        if self.data.ndim == 2:
            print("`data` is already a 2d array")
            self.matrix = matrix_scaling(self.data, self.scaling)

        elif self.data.ndim in [3, 4]:
            if self.data.ndim == 3:
                cube_ = self.data

            elif self.data.ndim == 4:
                if self.wavelengths is None:
                    raise ValueError("`wavelengths` must be provided when "
                                     "`data` is a 4D array")
                z, n_frames, y_in, x_in = self.data.shape
                scale_list = check_scal_vector(self.wavelengths)
                if not scale_list.shape[0] == z:
                    raise ValueError("`wavelengths` length is {} instead of "
                                     "{}".format(scale_list.shape[0], z))
                big_cube = []
                # Rescaling the spectral channels to align the speckles
                if self.verbose:
                    print('Rescaling the spectral channels to align the '
                          'speckles')
                for i in Progressbar(range(n_frames), verbose=self.verbose):
                    cube_resc = scwave(self.data[:, i, :, :], scale_list)[0]
                    cube_resc = cube_crop_frames(cube_resc, size=y_in,
                                                 verbose=False)
                    big_cube.append(cube_resc)
                big_cube = np.array(big_cube)
                cube_ = big_cube.reshape(z * n_frames, y_in, x_in)
                self.cube4dto3d_shape = cube_.shape

            result = prepare_matrix(cube_, self.scaling, mode=self.mode,
                                    inner_radius=self.inrad,
                                    outer_radius=self.outrad,
                                    verbose=self.verbose)
            if self.mode == 'annular':
                self.matrix = result[0]
                pxind = result[1]
                self.yy, self.xx = pxind  # pixel coords in the annulus
            elif self.mode == 'fullfr':
                self.matrix = result

        if self.verbose:
            timing(start_time)

    def run(self):
        """
        Decompose the input data.
        """
        start_time = time_ini(False)
        if not hasattr(self, 'matrix'):
            self.generate_matrix()

        max_pcs = min(self.matrix.shape[0], self.matrix.shape[1])

        results = svd_wrapper(self.matrix, self.svd_mode, max_pcs,
                              verbose=self.verbose, full_output=True)
        if len(results) == 3:
            self.u, self.s, self.v = results
        elif len(results) == 2:
            self.s, self.v = results

        if self.verbose:
            timing(start_time)

    def get_cevr(self, ncomp_list=None, plot=True, plot_save=False, plot_dpi=90,
                 plot_truncation=None):
        """
        Calculate the cumulative explained variance ratio for the SVD of a
        cube/matrix (either full frames or a single annulus could be used).

        Parameters
        ----------
        ncomp_list : None, list or tuple, optional
            If provided the list is used to filter the vector of CEVR.
        plot : bool, optional
            If True, the CEVR is plotted.
        plot_save : bool, optional
            If True, the plot is saved as ./figure.pdf.
        plot_dpi : int, optional
            The DPI of the figure.
        plot_truncation : None or int, optional
            If provided, it created a second panel in the plot, focusing on the
            CEVR curve up to ``plot_truncation`` components.

        Returns
        -------
        df_allks : pandas dataframe
            [ncomp_list is None] A table with the explained varaince ratio and
            the CEVR for all ncomps.
        df_klist : pandas dataframe
            [ncomp_list is not None] A table with the ncomp_list, the explained
            varaince ratio and the CEVR.
        """
        start_time = time_ini(False)
        if not hasattr(self, 'v'):
            self.run()

        if self.verbose:
            print("Computing the cumulative explained variance ratios")

        self.ncomp_list = ncomp_list
        exp_var = (self.s ** 2) / (self.s.shape[0] - 1)
        full_var = np.sum(exp_var)
        # % of variance explained by each PC
        self.explained_variance_ratio = exp_var / full_var
        self.cevr = np.cumsum(self.explained_variance_ratio)

        df_allks = DataFrame({'ncomp': range(1, self.s.shape[0] + 1),
                              'expvar_ratio': self.explained_variance_ratio,
                              'cevr': self.cevr})
        self.table_cevr = df_allks

        if plot:
            lw = 2;
            alpha = 0.4
            fig = plt.figure(figsize=vip_figsize, dpi=plot_dpi)
            fig.subplots_adjust(wspace=0.4)
            ax1 = plt.subplot2grid((1, 3), (0, 0), colspan=2)
            ax1.step(range(self.explained_variance_ratio.shape[0]),
                     self.explained_variance_ratio, alpha=alpha, where='mid',
                     label='Individual EVR', lw=lw)
            ax1.plot(self.cevr, '.-', alpha=alpha,
                     label='Cumulative EVR', lw=lw)
            ax1.legend(loc='best', frameon=False, fontsize='medium')
            ax1.set_ylabel('Explained variance ratio (EVR)')
            ax1.set_xlabel('Principal components')
            ax1.grid(linestyle='solid', alpha=0.2)
            ax1.set_xlim(-10, self.explained_variance_ratio.shape[0] + 10)
            ax1.set_ylim(0, 1)

            if plot_truncation is not None:
                ax2 = plt.subplot2grid((1, 3), (0, 2), colspan=1)
                ax2.step(range(plot_truncation),
                         self.explained_variance_ratio[:plot_truncation],
                         alpha=alpha, where='mid', lw=lw)
                ax2.plot(self.cevr[:plot_truncation], '.-', alpha=alpha, lw=lw)
                ax2.set_xlabel('Principal components')
                ax2.grid(linestyle='solid', alpha=0.2)
                ax2.set_xlim(-2, plot_truncation + 2)
                ax2.set_ylim(0, 1)

            if plot_save:
                plt.savefig('figure.pdf', dpi=300, bbox_inches='tight')

        if self.ncomp_list is not None:
            cevr_klist = []
            expvar_ratio_klist = []
            for k in self.ncomp_list:
                cevr_klist.append(self.cevr[k - 1])
                expvar_ratio_klist.append(self.explained_variance_ratio[k - 1])

            df_klist = DataFrame({'ncomp': self.ncomp_list,
                                  'exp_var_ratio': expvar_ratio_klist,
                                  'cevr': cevr_klist})
            self.cevr_ncomp = cevr_klist
            self.table_cevr_ncomp = df_klist
            if self.verbose:
                timing(start_time)
            return df_klist
        else:
            if self.verbose:
                timing(start_time)
            return df_allks

    def cevr_to_ncomp(self, cevr=0.9):
        """
        Infer the number of principal components for a given CEVR.

        Parameters
        ----------
        cevr : float or tuple of floats, optional
            The desired CEVR.

        Returns
        -------
        ncomp : int or list of ints
            The found number(s) of PCs.

        """
        if not hasattr(self, 'cevr'):
            self.get_cevr(plot=False)

        if isinstance(cevr, float):
            ncomp = np.searchsorted(self.cevr, cevr) + 1
        elif isinstance(cevr, tuple):
            ncomp = [np.searchsorted(self.cevr, c) + 1 for c in cevr]

        return ncomp


def svd_wrapper(matrix, mode, ncomp, verbose, full_output=False,
                random_state=None, to_numpy=True):
    """ Wrapper for different SVD libraries (CPU and GPU). 
      
    Parameters
    ----------
    matrix : numpy ndarray, 2d
        2d input matrix.
    mode : {'lapack', 'arpack', 'eigen', 'randsvd', 'cupy', 'eigencupy',
        'randcupy', 'pytorch', 'eigenpytorch', 'randpytorch'}, str optional
        Switch for the SVD method/library to be used.

        ``lapack``: uses the LAPACK linear algebra library through Numpy
        and it is the most conventional way of computing the SVD
        (deterministic result computed on CPU).

        ``arpack``: uses the ARPACK Fortran libraries accessible through
        Scipy (computation on CPU).

        ``eigen``: computes the singular vectors through the
        eigendecomposition of the covariance M.M' (computation on CPU).

        ``randsvd``: uses the randomized_svd algorithm implemented in
        Sklearn (computation on CPU).

        ``cupy``: uses the Cupy library for GPU computation of the SVD as in
        the LAPACK version. `

        `eigencupy``: offers the same method as with the ``eigen`` option
        but on GPU (through Cupy).

        ``randcupy``: is an adaptation of the randomized_svd algorithm,
        where all the computations are done on a GPU (through Cupy). `

        `pytorch``: uses the Pytorch library for GPU computation of the SVD.

        ``eigenpytorch``: offers the same method as with the ``eigen``
        option but on GPU (through Pytorch).

        ``randpytorch``: is an adaptation of the randomized_svd algorithm,
        where all the linear algebra computations are done on a GPU
        (through Pytorch).

    ncomp : int
        Number of singular vectors to be obtained. In the cases when the full
        SVD is computed (LAPACK, ARPACK, EIGEN, CUPY), the matrix of singular 
        vectors is truncated.
    verbose: bool
        If True intermediate information is printed out.
    full_output : bool optional
        If True the 3 terms of the SVD factorization are returned. If ``mode``
        is eigen then only S and V are returned.
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
    V : numpy ndarray
        The right singular vectors of the input matrix. If ``full_output`` is
        True it returns the left and right singular vectors and the singular
        values of the input matrix. If ``mode`` is set to eigen then only S and
        V are returned.
    
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
    if matrix.ndim != 2:
        raise TypeError('Input matrix is not a 2d array')

    if ncomp > min(matrix.shape[0], matrix.shape[1]):
        msg = '{} PCs cannot be obtained from a matrix with size [{},{}].'
        msg += ' Increase the size of the patches or request less PCs'
        raise RuntimeError(msg.format(ncomp, matrix.shape[0], matrix.shape[1]))

    if mode == 'eigen':
        # building C as np.dot(matrix.T,matrix) is slower and takes more memory
        C = np.dot(matrix, matrix.T)    # covariance matrix
        e, EV = linalg.eigh(C)          # EVals and EVs
        pc = np.dot(EV.T, matrix)       # PCs using a compact trick when cov is MM'
        V = pc[::-1]                    # reverse since we need the last EVs
        S = np.sqrt(np.abs(e))          # SVals = sqrt(EVals)
        S = S[::-1]                     # reverse since EVals go in increasing order
        for i in range(V.shape[1]):
            V[:, i] /= S    # scaling EVs by the square root of EVals
        V = V[:ncomp]
        if verbose:
            print('Done PCA with numpy linalg eigh functions')

    elif mode == 'lapack':
        # n_frames is usually smaller than n_pixels. In this setting taking
        # the SVD of M' and keeping the left (transposed) SVs is faster than
        # taking the SVD of M (right SVs)
        U, S, V = linalg.svd(matrix.T, full_matrices=False)
        V = V[:ncomp]       # we cut projection matrix according to the # of PCs
        U = U[:, :ncomp]
        S = S[:ncomp]
        if verbose:
            print('Done SVD/PCA with numpy SVD (LAPACK)')

    elif mode == 'arpack':
        U, S, V = svds(matrix, k=ncomp)
        if verbose:
            print('Done SVD/PCA with scipy sparse SVD (ARPACK)')

    elif mode == 'randsvd':
        U, S, V = randomized_svd(matrix, n_components=ncomp, n_iter=2,
                                 transpose='auto', random_state=random_state)
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
        if full_output:
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
        if verbose:
            print('Done randomized SVD/PCA with cupy (GPU)')

    elif mode == 'eigencupy':
        if no_cupy:
            raise RuntimeError('Cupy is not installed')
        a_gpu = cupy.array(matrix)
        a_gpu = cupy.asarray(a_gpu)     # move the data to the current device
        C = cupy.dot(a_gpu, a_gpu.T)    # covariance matrix
        e, EV = cupy.linalg.eigh(C)     # eigenvalues and eigenvectors
        pc = cupy.dot(EV.T, a_gpu)      # using a compact trick when cov is MM'
        V = pc[::-1]                    # reverse to get last eigenvectors
        S = cupy.sqrt(e)[::-1]          # reverse since EVals go in increasing order
        for i in range(V.shape[1]):
            V[:, i] /= S                # scaling by the square root of eigvals
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
        if verbose:
            print('Done randomized SVD/PCA with randomized pytorch (GPU)')

    else:
        raise ValueError('The SVD `mode` is not recognized')

    if full_output:
        if mode == 'lapack':
            return V.T, S, U.T
        elif mode == 'pytorch':
            if to_numpy:
                return V.T, S, U.T
            else:
                return torch.transpose(V, 0, 1), S, torch.transpose(U, 0, 1)
        elif mode in ('eigen', 'eigencupy', 'eigenpytorch'):
            return S, V
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
                     collapse=False, scaling=None):
    """ Getting ``ncomp`` eigenvectors. Choosing the size of the PCA truncation
    when ``ncomp`` is set to ``auto``. Used in ``pca_annular`` and ``llsg``.
    """
    no_dataref = False
    if data_ref is None:
        no_dataref = True
        data_ref = data

    if max_evs is None:
        max_evs = min(data_ref.shape[0], data_ref.shape[1])

    if ncomp is None:
        raise ValueError('ncomp must be an integer or `auto`')

    if ncomp == 'auto':
        ncomp = 0
        V_big = svd_wrapper(data_ref, svd_mode, max_evs, False)

        if mode == 'noise':
            if not collapse:
                data_ref_sc = matrix_scaling(data_ref, scaling)
                data_sc = matrix_scaling(data, scaling)
            else:
                data_ref_sc = matrix_scaling(data_ref, scaling)
                data_sc = matrix_scaling(data, scaling)

            V_sc = svd_wrapper(data_ref_sc, svd_mode, max_evs, False)

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

        elif mode == 'cevr':
            data_sc = matrix_scaling(data, scaling)
            _, S, _ = svd_wrapper(data_sc, svd_mode, min(data_sc.shape[0],
                                                         data_sc.shape[1]),
                                  False, full_output=True)
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
        V = svd_wrapper(data_ref, svd_mode, ncomp, verbose=False)

    return V


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

        # Sample the range of M by linear projection of Q.
        # Extract an orthonormal basis
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
            return (V[:n_components, :].T, s[:n_components],
                    U[:,:n_components].T)
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

        # Sample the range of M by linear projection of Q.
        # Extract an orthonormal basis
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


