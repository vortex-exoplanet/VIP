#! /usr/bin/env python

"""
Module with sky subtraction functionalities.
"""

from __future__ import division

__author__ = 'C. Gomez @ ULg'
__all__ = ['cube_subtract_sky_pca']

import numpy as np
from ..pca import prepare_matrix, svd_wrapper


def cube_subtract_sky_pca(sci_cube, sky_cube, mask, ref_cube=None, ncomp=2):
    """ PCA based sky subtraction.

    Notes
    -----
    MSF : masked science frame
    MPC_1,...,MPC_k : PCs masked (the same way)
    MPC : matrix whose columns are MPC_i

    The coefficients c_1,...,c_k are obtained by solving in the least square
    sense.
    MSF = sum_i(c_i * MPC_i)

    the solution vector c = (c_1,...,c_k)' is given by
    c = inv(MPC' * MPC) * MPC' * MSF,
    where MSF is in vector column form, ' denotes the matrix transpose and * the
    matrix product.

    Note that MPC' * MSF is equal to PC' * MSF, but the masked PCs are not
    orthonormal, hence MPC' * MPC is not the identity, therefore
    inv(MPC' * MPC) * MPC' * MSF does not reduce to PC' * MSF.

    Parameters
    ----------
    sci_cube : array_like
        3d array of science frames.
    sky_cube : array_like
        3d array of sky frames.
    mask : array_like
        Mask indicating the region for the analysis. Can be created with the
        function vip.var.create_ringed_spider_mask.
    ref_cube : array_like or None
        Reference cube.
    ncomp : int
        Sets the number of PCs you want to use in the sky subtraction.

    Returns
    -------
    Sky subtracted cube.

    """
    if sci_cube.shape[1] != sky_cube.shape[1] or sci_cube.shape[2] != \
            sky_cube.shape[2]:
        raise TypeError('Science and Sky frames sizes do not match')
    if ref_cube is not None:
        if sci_cube.shape[1] != ref_cube.shape[1] or sci_cube.shape[2] != \
                ref_cube.shape[2]:
            raise TypeError('Science and Reference frames sizes do not match')

    # Getting the EVs from the sky cube
    Msky = prepare_matrix(sky_cube, scaling=None, verbose=False)
    sky_pcs = svd_wrapper(Msky, 'lapack', sky_cube.shape[0], False,
                                  False)
    sky_pcs_cube = sky_pcs.reshape(sky_cube.shape[0], sky_cube.shape[1],
                                   sky_cube.shape[1])

    # Masking the science cube
    sci_cube_masked = np.zeros_like(sci_cube)
    ind_masked = np.where(mask == 0)
    for i in xrange(sci_cube.shape[0]):
        masked_image = np.copy(sci_cube[i])
        masked_image[ind_masked] = 0
        sci_cube_masked[i] = masked_image
    Msci_masked = prepare_matrix(sci_cube_masked, scaling=None,
                                         verbose=False)

    # Masking the PCs learned from the skies
    sky_pcs_cube_masked = np.zeros_like(sky_pcs_cube)
    for i in xrange(sky_pcs_cube.shape[0]):
        masked_image = np.copy(sky_pcs_cube[i])
        masked_image[ind_masked] = 0
        sky_pcs_cube_masked[i] = masked_image

    # Project the masked frames onto the sky PCs to get the coefficients
    transf_sci = np.zeros((sky_cube.shape[0], Msci_masked.shape[0]))
    for i in xrange(Msci_masked.shape[0]):
        transf_sci[:, i] = np.inner(sky_pcs, Msci_masked[i].T)

    Msky_pcs_masked = prepare_matrix(sky_pcs_cube_masked, scaling=None,
                                             verbose=False)
    mat_inv = np.linalg.inv(np.dot(Msky_pcs_masked, Msky_pcs_masked.T))
    transf_sci_scaled = np.dot(mat_inv, transf_sci)

    # Obtaining the optimized sky and subtraction
    sci_cube_skysub = np.zeros_like(sci_cube)
    for i in xrange(Msci_masked.shape[0]):
        sky_opt = np.array([np.sum(
            transf_sci_scaled[j, i] * sky_pcs_cube[j] for j in range(ncomp))])
        sci_cube_skysub[i] = sci_cube[i] - sky_opt

        # Processing the reference cube (if any)
    if ref_cube is not None:
        ref_cube_masked = np.zeros_like(ref_cube)
        for i in xrange(ref_cube.shape[0]):
            masked_image = np.copy(ref_cube[i])
            masked_image[ind_masked] = 0
            ref_cube_masked[i] = masked_image
        Mref_masked = prepare_matrix(ref_cube_masked, scaling=None,
                                             verbose=False)
        transf_ref = np.zeros((sky_cube.shape[0], Mref_masked.shape[0]))
        for i in xrange(Mref_masked.shape[0]):
            transf_ref[:, i] = np.inner(sky_pcs, Mref_masked[i].T)

        transf_ref_scaled = np.dot(mat_inv, transf_ref)

        ref_cube_skysub = np.zeros_like(ref_cube)
        for i in xrange(Mref_masked.shape[0]):
            sky_opt = np.array([np.sum(
                transf_ref_scaled[j, i] * sky_pcs_cube[j] for j in
                range(ncomp))])
            ref_cube_skysub[i] = ref_cube[i] - sky_opt

        return sci_cube_skysub, ref_cube_skysub
    else:
        return sci_cube_skysub
