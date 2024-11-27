#! /usr/bin/env python
"""
Module with sky subtraction function.

.. [GOM17]
   | Gomez-Gonzalez et al. 2017
   | **VIP: Vortex Image Processing Package for High-contrast Direct Imaging**
   | *The Astronomical Journal, Volume 154, p. 7*
   | `https://arxiv.org/abs/1705.06184
     <https://arxiv.org/abs/1705.06184>`_

.. [HUN18]
   | Hunziker et al. 2018
   | **PCA-based approach for subtracting thermal background emission in
     high-contrast imaging data**
   | *Astronomy & Astrophysics, Volume 611, p. 23*
   | `https://arxiv.org/abs/1706.10069
     <https://arxiv.org/abs/1706.10069>`_

.. [REN23]
   | Ren 2023
   | **Karhunen-Loève data imputation in high-contrast imaging**
   | *Astronomy & Astrophysics, Volume 679, p. 8*
   | `https://arxiv.org/abs/2308.16912
     <https://arxiv.org/abs/2308.16912>`_

"""

__author__ = 'Sandrine Juillard'
__all__ = ['cube_subtract_sky_pca']

import numpy as np
from ..var import prepare_matrix


def cube_subtract_sky_pca(sci_cube, sky_cube, masks, ref_cube=None, ncomp=2,
                          full_output=False):
    """PCA-based sky subtraction as explained in [REN23]_ (see also\
    [GOM17]_ and [HUN18]_).

    Parameters
    ----------
    sci_cube : numpy ndarray
        3d array of science frames.
    sky_cube : numpy ndarray
        3d array of sky frames.
    masks : tuple of two numpy ndarray or one single numpy ndarray
        Masks indicating the anchor and boat regions for the analysis.
        If two masks are provided, they will be assigned to mask_anchor and
        mask_boat, respectively.
        If only one mask is provided, it will be used as the anchor, and the
        boat images will not be masked (i.e., full frames used).
    ref_cube : numpy ndarray or None, opt
        Reference cube.
    ncomp : int, opt
        Sets the number of PCs you want to use in the sky subtraction.
    full_output: bool, opt
        Whether to also output pcs, reconstructed cube, residuals cube and
        derotated residual cube.


    Notes
    -----
    Masks can be created with the function
    ``vip_hci.var.create_ringed_spider_mask`` or
    ``vip_hci.var.get_annulus_segments`` (see Usage Example below).


    Returns
    -------
    sci_cube_skysub : numpy ndarray
        Sky-subtracted science cube
    ref_cube_skysub : numpy ndarray
        [If ref_cube is not None] Also returns sky-subtracted reference cube.
    If full_output is set to True, returns (in the following order):
         - sky-subtracted science cube,
         - sky-subtracted reference cube (if any provided),
         - boat principal components,
         - anchor principal components, and
         - reconstructed cube.


    Usage Example
    -------------

    You can create the masks using `get_annulus_segments` from `vip_hci.var`.

    .. code-block:: python

        from vip_hci.var import get_annulus_segments

    The function must be used as follows, where `ring_out`, `ring_in`, and
    `coro` define the radius of the different annulus masks. They must have
    the same shape as a frame of the science cube.


    .. code-block:: python

        ones = np.ones(cube[0].shape)
        boat = get_annulus_segments(ones,coro,ring_out-coro, mode="mask")[0]
        anchor = get_annulus_segments(ones,ring_in,ring_out-ring_in,
                                      mode="mask")[0]


    Masks should be provided as 'mask_rdi' argument when using PCA.

    .. code-block:: python

        res = pca(cube, angles, ref, mask_rdi=(boat, anchor), ncomp=2)


    """
    try:
        from ..psfsub.svd import svd_wrapper
    except BaseException:
        from ..pca.svd import svd_wrapper

    if sci_cube.shape[1] != sky_cube.shape[1] or sci_cube.shape[2] != \
            sky_cube.shape[2]:
        raise TypeError('Science and Sky frames sizes do not match')

    if ref_cube is not None:
        if sci_cube.shape[1] != ref_cube.shape[1] or sci_cube.shape[2] != \
                ref_cube.shape[2]:
            raise TypeError('Science and Reference frames sizes do not match')
    if type(masks) not in (list, tuple):
        # If only one mask is provided, the second mask is generated
        mask_anchor = masks
        mask_boat = np.ones(masks.shape)
    elif len(masks) != 2:
        raise TypeError('Science and Reference frames sizes do not match')
    else:
        mask_anchor, mask_boat = masks

    # -- Generate boat and anchor matrices
    # Masking the sky cube with anchor
    sky_cube_masked = np.zeros_like(sky_cube)
    ind_masked = np.where(mask_anchor == 0)
    for i in range(sky_cube.shape[0]):
        masked_image = np.copy(sky_cube[i])
        masked_image[ind_masked] = 0
        sky_cube_masked[i] = masked_image
    sky_anchor = sky_cube_masked.reshape(sky_cube.shape[0],
                                         sky_cube.shape[1]*sky_cube.shape[2])

    # Masking the science cube with anchor
    sci_cube_anchor = np.zeros_like(sci_cube)
    ind_masked = np.where(mask_anchor == 0)
    for i in range(sci_cube.shape[0]):
        masked_image = np.copy(sci_cube[i])
        masked_image[ind_masked] = 0
        sci_cube_anchor[i] = masked_image
    Msci_masked_anchor = prepare_matrix(sci_cube_anchor, scaling=None,
                                        verbose=False)

    # Masking the science cube with boat
    sci_cube_boat = np.zeros_like(sci_cube)
    ind_masked = np.where(mask_boat == 0)
    for i in range(sci_cube.shape[0]):
        masked_image = np.copy(sci_cube[i])
        masked_image[ind_masked] = 0
        sci_cube_boat[i] = masked_image
    Msci_masked = prepare_matrix(sci_cube_boat, scaling=None, verbose=False)

    # Masking the sky cube with boat
    sky_cube_boat = np.zeros_like(sky_cube)
    ind_masked = np.where(mask_boat == 0)
    for i in range(sky_cube.shape[0]):
        masked_image = np.copy(sky_cube[i])
        masked_image[ind_masked] = 0
        sky_cube_boat[i] = masked_image
    sky_boat = sky_cube_boat.reshape(sky_cube.shape[0],
                                     sky_cube.shape[1]*sky_cube.shape[2])

    # -- Generate eigenvectors of R(a)T R(a)

    sky_kl = np.dot(sky_anchor, sky_anchor.T)
    Msky_kl = prepare_matrix(sky_kl, scaling=None, verbose=False)
    sky_pcs = svd_wrapper(Msky_kl, 'lapack', sky_kl.shape[0], False)
    sky_pcs_kl = sky_pcs.reshape(sky_kl.shape[0], sky_kl.shape[1])

    # -- Generate Kl and Dikl transform

    sky_pc_anchor = np.dot(sky_pcs_kl, sky_anchor)
    sky_anchor_cube = sky_pc_anchor.reshape(sky_cube.shape[0],
                                            sky_cube.shape[1],
                                            sky_cube.shape[2])

    sky_boat_cube = np.dot(sky_pcs_kl, sky_boat).reshape(sky_cube.shape[0],
                                                         sky_cube.shape[1],
                                                         sky_cube.shape[2])

    # -- Generate Kl projection to get coeff

    transf_sci = np.zeros((sky_cube.shape[0], Msci_masked_anchor.shape[0]))
    for i in range(Msci_masked_anchor.shape[0]):
        transf_sci[:, i] = np.inner(sky_pc_anchor, Msci_masked_anchor[i].T)

    Msky_pcs_anchor = prepare_matrix(sky_anchor_cube, scaling=None,
                                     verbose=False)

    mat_inv = np.linalg.inv(np.dot(Msky_pcs_anchor, Msky_pcs_anchor.T))
    transf_sci_scaled = np.dot(mat_inv, transf_sci)

    # -- Subtraction Dikl projection using anchor coeff to sci cube

    sci_cube_skysub = np.zeros_like(sci_cube)
    sky_opt = sci_cube.copy()
    for i in range(Msci_masked.shape[0]):
        sky_opt[i] = np.sum([transf_sci_scaled[j, i]*sky_boat_cube[j]
                             for j in range(ncomp)], axis=0)
        sci_cube_skysub[i] = sci_cube_boat[i] - sky_opt[i]

    # -- Processing the reference cube (if any)
    if ref_cube is not None:

        # Masking the ref cube with anchor
        ref_cube_anchor = np.zeros_like(sci_cube)
        ind_masked = np.where(mask_anchor == 0)
        for i in range(sci_cube.shape[0]):
            masked_image = np.copy(sci_cube[i])
            masked_image[ind_masked] = 0
            ref_cube_anchor[i] = masked_image
        Mref_masked_anchor = prepare_matrix(ref_cube_anchor, scaling=None,
                                            verbose=False)

        # Masking the ref cube with boat
        ref_cube_boat = np.zeros_like(sci_cube)
        ind_masked = np.where(mask_boat == 0)
        for i in range(sci_cube.shape[0]):
            masked_image = np.copy(sci_cube[i])
            masked_image[ind_masked] = 0
            ref_cube_boat[i] = masked_image
        Mref_masked = prepare_matrix(ref_cube_boat, scaling=None, verbose=False)

        transf_ref = np.zeros((sky_cube.shape[0], Mref_masked.shape[0]))
        for i in range(Mref_masked.shape[0]):
            transf_ref[:, i] = np.inner(sky_pc_anchor, Mref_masked_anchor[i].T)

        transf_ref_scaled = np.dot(mat_inv, transf_ref)

        ref_cube_skysub = np.zeros_like(ref_cube)
        for i in range(Mref_masked.shape[0]):
            tmp_sky = [np.sum(transf_ref_scaled[j, i]*sky_boat_cube[j]
                              for j in range(ncomp))]
            sky_opt = np.array(tmp_sky)
            ref_cube_skysub[i] = ref_cube_boat[i] - sky_opt

        if full_output:
            return (sci_cube_skysub, ref_cube_skysub, sky_anchor_cube,
                    sky_boat_cube, sky_opt)
        else:
            return sci_cube_skysub, ref_cube_skysub
    else:
        if full_output:
            return (sci_cube_skysub, sky_anchor_cube, sky_boat_cube,
                    sky_opt)
        else:
            return sci_cube_skysub
