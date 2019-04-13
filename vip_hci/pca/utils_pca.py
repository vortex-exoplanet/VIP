#! /usr/bin/env python

"""
Module with helping functions.
"""

__author__ = 'Carlos Alberto Gomez Gonzalez'
__all__ = []

import numpy as np
from sklearn.decomposition import IncrementalPCA
from ..conf import timing, time_ini, get_available_memory
from ..var import prepare_matrix, reshape_matrix
from ..preproc import cube_derotate, cube_collapse, check_pa_vector
from ..fits import open_fits
from .svd import svd_wrapper


def pca_incremental(cube, angle_list, batch=0.25, ncomp=1, imlib='opencv',
                    interpolation='lanczos4', collapse='median', verbose=True,
                    full_output=False, return_residuals=False, start_time=None):
    """ Computes the full-frame PCA-ADI algorithm in batches, for processing
    fits files larger than the available system memory. It uses the incremental
    PCA algorithm from Sklearn. There is no ``scaling`` parameter as in other
    PCA algorithms in ``VIP``, but by default this implementation returns a
    temporally mean-centered frame ("temp-mean").

    Parameters
    ----------
    cube : str or numpy ndarray
        Input cube as numpy array or string with the path to the fits file to be
        opened in memmap mode.
    angle_list : str or numpy ndarray
        Corresponding parallactic angle for each frame.
    batch : int or float, optional
        When int it corresponds to the number of frames in each batch. If a
        float (0, 1] is passed then it is the size of the batch is computed wrt
        the available memory in the system.
    ncomp : int, optional
        How many PCs are used as a lower-dimensional subspace to project the
        target frames.
    imlib : str, optional
        See the documentation of the ``vip_hci.preproc.frame_rotate`` function.
    interpolation : str, optional
        See the documentation of the ``vip_hci.preproc.frame_rotate`` function.
    collapse : {'median', 'mean', 'sum', 'trimmean'}, str optional
        Sets the way of collapsing the frames for producing a final image.
    verbose : {True, False}, bool optional
        If True prints intermediate info and timing.
    full_output : boolean, optional
        Whether to return the final median combined image only or with other
        intermediate arrays.
    return_residuals : bool, optional
        If True, only the cube of residuals is returned (before de-rotating).
    start_time : None or datetime.datetime, optional
        Used when embedding this function in the main ``pca`` function. The
        object datetime.datetime is the global starting time. If None, it
        initiates its own counter.

    Returns
    -------
    frame : numpy ndarray
        [return_residuals=False] Final frame (2d array).
    ipca : scikit-learn model
        [full_output=True, return_residuals=False] The incremental PCA model of
        scikit-learn.
    pcs : numpy ndarray
        [full_output=True, return_residuals=False] Principal components reshaped
        into images.
    medians : numpy ndarray
        [full_output=True, return_residuals=False] The median of the derotated
        residuals for each batch.
    cube_residuals : numpy ndarray
        [return_residuals=True] Cube of residuals.


    """
    if start_time is None:
        start_time = time_ini(verbose)
        verbose_memcheck = True
    else:
        verbose_memcheck = False

    # checking cube and angle_list data types
    if not isinstance(cube, (np.ndarray, str)):
        raise TypeError('`cube` must be a str (full path on disk) or a numpy '
                        'array')
    if not isinstance(angle_list, (np.ndarray, str)):
        raise TypeError('`angle_list` must be a str (full path on disk) or a '
                        'numpy array')

    # opening data
    if isinstance(cube, str):
        # assuming the first HDULIST contains the datacube
        hdulist = open_fits(cube, n=0, return_memmap=True)
        cube = hdulist.data
    if not cube.ndim > 2:
        raise TypeError('Input array is not a 3d array')
    n_frames, y, x = cube.shape

    # checking angles length and ncomp
    if isinstance(angle_list, str):
        angle_list = open_fits(angle_list)
    angle_list = check_pa_vector(angle_list)
    if not n_frames == angle_list.shape[0] and not return_residuals:
        raise TypeError('`angle_list` vector has wrong length. It must be the '
                        'same as the number of frames in the cube')
    if not isinstance(ncomp, (int, float)):
        raise TypeError("`ncomp` must be an int or a float in the ADI case")
    if ncomp > n_frames:
        ncomp = min(ncomp, n_frames)
        msg = 'Number of PCs too high (max PCs={}), using {} PCs instead.'
        print(msg.format(n_frames, ncomp))

    # checking memory and determining batch size
    cube_size = cube.nbytes
    aval_mem = get_available_memory(verbose_memcheck)
    if isinstance(batch, int):      # the batch size in n_fr
        batch_size = batch
    elif isinstance(batch, float):  # the batch ratio wrt available memory
        if 1 > batch > 0:
            batch_size = min(int(n_frames * (batch * aval_mem) / cube_size),
                             n_frames)
    else:
        raise TypeError("`batch` must be an int or float")

    if verbose:
        msg1 = "Cube size = {:.3f} GB ({} frames)"
        print(msg1.format(cube_size / 1e9, n_frames))
        msg2 = "Batch size = {} frames ({:.3f} GB)\n"
        print(msg2.format(batch_size, cube[:batch_size].nbytes / 1e9))

    n_batches = n_frames // batch_size      # floor/int division
    remaining_frames = n_frames % batch_size
    if remaining_frames > 0:
        n_batches += 1

    # computing the PCA model for each batch
    ipca = IncrementalPCA(n_components=ncomp)

    for i in range(n_batches):
        intini = i * batch_size
        intfin = (i + 1) * batch_size
        batch = cube[intini:min(n_frames, intfin)]
        msg = 'Batch {}/{}\tshape: {}\tsize: {:.1f} MB'
        if verbose:
            print(msg.format(i+1, n_batches, batch.shape, batch.nbytes / 1e6))
        matrix = prepare_matrix(batch, verbose=False)
        ipca.partial_fit(matrix)

    if verbose:
        timing(start_time)

    # getting PCs and the mean in order to center each batch
    V = ipca.components_
    mean = ipca.mean_.reshape(y, x)

    if verbose:
        print('\nReconstructing and obtaining residuals')

    if return_residuals:
        cube_residuals = np.empty((n_frames, y, x))
    else:
        medians = []

    for i in range(n_batches):
        intini = i * batch_size
        intfin = (i + 1) * batch_size
        batch = cube[intini:min(n_frames, intfin)] - mean
        matrix = prepare_matrix(batch, verbose=False)
        reconst = np.dot(np.dot(matrix, V.T), V)
        resid = matrix - reconst
        resid_reshaped = resid.reshape(batch.shape)
        if return_residuals:
            cube_residuals[intini:intfin] = resid_reshaped
        else:
            resid_der = cube_derotate(resid_reshaped, angle_list[intini:intfin],
                                      imlib=imlib, interpolation=interpolation)
            medians.append(cube_collapse(resid_der, mode=collapse))

    del matrix
    del batch

    if return_residuals:
        return cube_residuals

    else:
        medians = np.array(medians)
        frame = np.median(medians, axis=0)

        if verbose:
            timing(start_time)

        if full_output:
            pcs = reshape_matrix(V, y, x)
            return frame, ipca, pcs, medians
        else:
            return frame


def pca_annulus(cube, angs, ncomp, annulus_width, r_guess, cube_ref=None,
                svd_mode='lapack', scaling=None, collapse='median',
                imlib='opencv', interpolation='lanczos4'):
    """
    PCA process the cube only for an annulus of a given width and at a given
    radial distance to the frame center. It returns a PCA processed frame with 
    only non-zero values at the positions of the annulus.
    
    Parameters
    ----------
    cube : numpy ndarray
        The cube of fits images expressed as a numpy.array.
    angs : numpy ndarray
        The parallactic angles expressed as a numpy.array.
    ncomp : int
        The number of principal component.
    annulus_width : float
        The annulus width in pixel on which the PCA is performed.
    r_guess : float
        Radius of the annulus in pixels.
    cube_ref : numpy ndarray, 3d, optional
        Reference library cube. For Reference Star Differential Imaging.
    svd_mode : {'lapack', 'randsvd', 'eigen', 'arpack'}, str optional
        Switch for different ways of computing the SVD and selected PCs.
    scaling : {None, 'temp-mean', 'spat-mean', 'temp-standard', 'spat-standard'}
        With None, no scaling is performed on the input data before SVD. With
        "temp-mean" then temporal px-wise mean subtraction is done, with
        "spat-mean" then the spatial mean is subtracted, with "temp-standard"
        temporal mean centering plus scaling to unit variance is done and with
        "spat-standard" spatial mean centering plus scaling to unit variance is
        performed.
    collapse : {'median', 'mean', 'sum', 'trimmean', None}, str or None, optional
        Sets the way of collapsing the frames for producing a final image. If
        None then the cube of residuals is returned.
    imlib : str, optional
        See the documentation of the ``vip_hci.preproc.frame_rotate`` function.
    interpolation : str, optional
        See the documentation of the ``vip_hci.preproc.frame_rotate`` function.
    
    Returns
    -------
    Depending on ``collapse`` parameter a final collapsed frame or the cube of
    residuals is returned.
    """
    data, ind = prepare_matrix(cube, scaling, mode='annular',
                               annulus_radius=r_guess, verbose=False,
                               annulus_width=annulus_width)
    yy, xx = ind

    if cube_ref is not None:
        data_svd, _ = prepare_matrix(cube_ref, scaling, mode='annular',
                                     annulus_radius=r_guess, verbose=False,
                                     annulus_width=annulus_width)
    else:
        data_svd = data
        
    V = svd_wrapper(data_svd, svd_mode, ncomp, verbose=False)
        
    transformed = np.dot(data, V.T)
    reconstructed = np.dot(transformed, V)                           
    residuals = data - reconstructed
    cube_zeros = np.zeros_like(cube)
    cube_zeros[:, yy, xx] = residuals

    if angs is not None:
        cube_res_der = cube_derotate(cube_zeros, angs, imlib=imlib,
                                     interpolation=interpolation)
        if collapse is not None:
            pca_frame = cube_collapse(cube_res_der, mode=collapse)
            return pca_frame
        else:
            return cube_res_der

    else:
        if collapse is not None:
            pca_frame = cube_collapse(cube_zeros, mode=collapse)
            return pca_frame
        else:
            return cube_zeros




