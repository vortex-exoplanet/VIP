#! /usr/bin/env python

"""
Forward model matched filter relying on either KLIP [SOU12]_ / [PUE16]_ or LOCI 
[LAF07] for the PSF reference approximation. The original concept of matched
 filter applied to KLIP has been first proposed in [RUF17] and then adapted in 
[DAH21a] to use the LOCI framework. For both PSF-subtraction techniques, a 
forward model of the PSF is computed for each pixel contained in the field of
view and each frame to account for the over-subtraction and self-subtraction
of potential planetary signal due to the reference PSF subtraction. The 
obtained model is then compared to the pixels intensities within each frame of
the residual cube. The SNR associated to each pixel contained in the field of
view, as well as its estimated contrast is then obtained via a Gaussian maximum 
likelihood approach.

.. [DAH21a]
   | Dahlqvist et al. 2021a
   | **Improving the RSM map exoplanet detection algorithm. PSF forward 
     modelling and optimal selection of PSF subtraction techniques**
   | *The Astrophysical Journal Letters, Volume 646, p. 49*
   | `https://arxiv.org/abs/astro-ph/2012.05094
     <https://arxiv.org/abs/astro-ph/2012.05094>`_    

.. [LAF07]
   | Lafreniere et al. 2007
   | **A New Algorithm for Point-Spread Function Subtraction in High-Contrast 
     Imaging: A Demonstration with Angular Differential Imaging**
   | *The Astrophysical Journal, Volume 660, Issue 4, pp. 770-780*
   | `https://arxiv.org/abs/astro-ph/0702697
     <https://arxiv.org/abs/astro-ph/0702697>`_

.. [PUE16]
   | Pueyo 2016
   | **Detection and Characterization of Exoplanets using Projections on 
     Karhunen Loeve Eigenimages: Forward Modeling**
   | *The Astrophysical Journal, Volume 824, Issue 2, p. 117*
   | `https://arxiv.org/abs/astro-ph/1604.06097
     <https://arxiv.org/abs/astro-ph/1604.06097>`_
     
.. [RUF17]
   | Ruffio et al. 2017
   | **Improving and Assessing Planet Sensitivity of the GPI Exoplanet Survey 
     with a Forward Model Matched Filter**
   | *The Astrophysical Journal, Volume 842, p. 14*
   | `https://arxiv.org/abs/astro-ph/1705.05477
     <https://arxiv.org/abs/astro-ph/1705.05477>`_
     
.. [SOU12]
   | Soummer et al. 2012
   | **Detection and Characterization of Exoplanets and Disks Using Projections 
     on Karhunen-Loève Eigenimages**
   | *The Astrophysical Journal Letters, Volume 755, Issue 2, p. 28*
   | `https://arxiv.org/abs/astro-ph/1207.4197
     <https://arxiv.org/abs/astro-ph/1207.4197>`_ 

"""

__author__ = 'Carl-Henrik Dahlqvist'
__all__ = ['fmmf']

from multiprocessing import cpu_count
import numpy as np
import numpy.linalg as la
from skimage.draw import disk
from ..var import get_annulus_segments, frame_center
from ..preproc import frame_crop, cube_crop_frames, cube_derotate
from ..config.utils_conf import pool_map, iterable
from ..config import time_ini, timing
from ..fm import cube_inject_companions
from ..preproc.derotation import _find_indices_adi


def fmmf(cube, pa, psf, fwhm, min_r=None, max_r=None, model='KLIP', var='FR',
         param={'ncomp': 20, 'tolerance': 5e-3, 'delta_rot': 0.5}, crop=5,
         imlib='vip-fft', interpolation='lanczos4', nproc=1, verbose=True):
    """
    Forward model matched filter generating SNR map and contrast map, using
    either KLIP or LOCI as PSF subtraction techniques, as implemented in [RUF17] 
    and [DAH21a]_.

    Parameters
    ----------
    cube : numpy ndarray, 3d
        Input cube (ADI sequences), Dim 1 = temporal axis, Dim 2-3 =
        spatial axis
    pa : numpy ndarray, 1d
        Parallactic angles for each frame of the ADI sequences.
    psf : numpy ndarray 2d
        2d array with the normalized PSF template, with an odd shape.
        The PSF image must be centered wrt to the array! Therefore, it is
        recommended to run the function ``normalize_psf`` to generate a
        centered and flux-normalized PSF template.
    fwhm: int
        Full width at half maximum for the instrument PSF
    min_r : int,optional
        Center radius of the first annulus considered in the FMMF detection
        map estimation. The radius should be larger than half
        the value of the 'crop' parameter . Default is None which
        corresponds to one FWHM.
    max_r : int
        Center radius of the last annulus considered in the FMMF detection
        map estimation. The radius should be smaller or equal to half the
        size of the image minus half the value of the 'crop' parameter.
        Default is None which corresponds to half the size of the image
        minus half the value of the 'crop' parameter.
    model: string, optional
        Selected PSF-subtraction technique for the computation of the FMMF
        detection map. FMMF work either with KLIP or LOCI. Default is 'KLIP'.
    var: str, optional
        Model used for the residual noise variance estimation used in the
        matched filtering (maximum likelihood estimation of the flux and SNR).
        Three different approaches are proposed: 'FR', 'FM', and 'TE'.

        * 'FR': consider the pixels in the selected annulus with a width equal
        to asize but separately for every frame.

        * 'FM': consider the pixels in the selected annulus with a width
        equal to asize but separately for every frame. Apply a mask one FWHM
        on the selected pixel and its surrounding.

        * 'TE': rely on the method developped in PACO to estimate the
        residual noise variance (take the pixels in a region of one FWHM
        arround the selected pixel, considering every frame in the
        derotated cube of residuals except for the selected frame)

    param: dict, optional
        Dictionnary regrouping the parameters used by the KLIP (ncomp and
        delta_rot) or LOCI (tolerance and delta_rot) PSF-subtraction
        technique.

        * ncomp : int, optional. Number of components used for the low-rank
        approximation of the speckle field. Default is 20.

        * tolerance: float, optional. Tolerance level for the approximation of
        the speckle field via a linear combination of the reference images in
        the LOCI algorithm. Default is 5e-3.

        * delta_rot : float, optional. Factor for tunning the parallactic angle
        threshold, expressed in FWHM. Default is 0.5 (excludes 0.5xFHWM on each
        side of the considered frame).

    crop: int, optional
        Part of the PSF template considered in the estimation of the FMMF
        detection map. Default is 5.
    imlib : str, optional
        Parameter used for the derotation of the residual cube. See the
        documentation of the ``vip_hci.preproc.frame_rotate`` function.
    interpolation : str, optional
        Parameter used for the derotation of the residual cube. See the
        documentation of the ``vip_hci.preproc.frame_rotate`` function.
    nproc : int or None, optional
        Number of processes for parallel computing. By default ('nproc=1')
        the algorithm works in single-process mode. If set to None, nproc
        is automatically set to half the number of available CPUs.
    verbose: bool, optional
        If True provide a message each time an annulus has been treated.
        Default True.

    Returns
    -------
    flux_matrix : 2d ndarray
        Maximum likelihood estimate of the contrast for each pixel in the field
        of view
    snr_matrix : 2d ndarray
        Signal to noise ratio map (defined as the estimated contrast divided by
        the estimated standard deviation of the contrast).

    """
    start_time = time_ini(verbose)
    
    if crop >= 2*round(fwhm)+1:
        raise ValueError("Maximum cropsize should be lower or equal to two" +
                         " FWHM,please change accordingly the value of 'crop'")
    if min_r is None:
        min_r = int(round(fwhm))
    if max_r is None:
        max_r = cube.shape[-1]//2-(crop//2+1)
    if nproc is None:
        nproc = cpu_count()//2

    res_full = pool_map(nproc, _snr_contrast_esti, iterable(range(min_r, max_r)),
                        cube, pa, psf, fwhm, model, var, param, crop, imlib,
                        interpolation, verbose)

    flux_matrix = np.zeros((cube.shape[1], cube.shape[2]))
    snr_matrix = np.zeros((cube.shape[1], cube.shape[2]))

    for res_temp in res_full:

        indices = get_annulus_segments(cube[0], res_temp[2], 1)
        flux_matrix[indices[0][0], indices[0][1]] = res_temp[0]
        snr_matrix[indices[0][0], indices[0][1]] = res_temp[1]

    if verbose:
        timing(start_time)

    return flux_matrix, snr_matrix


def _snr_contrast_esti(ann_center, cube, pa, psf, fwhm, model, var, param, crop,
                       imlib, interpolation, verbose):
    """
    Computation of the SNR and contrast associated to the pixels contained
    in a given annulus via the foward model matched filter
    """

    n, y, x = cube.shape

    evals_matrix = []
    evecs_matrix = []
    KL_basis_matrix = []
    refs_mean_sub_matrix = []
    sci_mean_sub_matrix = []
    resicube_klip = None

    ind_ref_list = None
    coef_list = None

    ncomp = param['ncomp']
    tolerance = param['tolerance']
    delta_rot = param['delta_rot']

    # Computation of the reference PSF, and the matrices
    # required for the computation of the PSF forward models

    pa_threshold = np.rad2deg(
        2 * np.arctan(delta_rot * fwhm / (2 * (ann_center))))
    mid_range = np.abs(np.amax(pa) - np.amin(pa)) / 2
    if pa_threshold >= mid_range - mid_range * 0.1:
        pa_threshold = float(mid_range - mid_range * 0.1)

    if model == 'KLIP':

        resicube_klip = np.zeros_like(cube)

        indices = get_annulus_segments(cube[0], ann_center-int(round(fwhm)/2),
                                       int(round(fwhm)), 1)

        for k in range(0, cube.shape[0]):

            res_temp = KLIP_patch(k, cube[:, indices[0][0], indices[0][1]],
                                  ncomp, pa, int(round(fwhm)), pa_threshold,
                                  ann_center)
            evals_temp = res_temp[0]
            evecs_temp = res_temp[1]
            KL_basis_temp = res_temp[2]
            sub_img_rows_temp = res_temp[3]
            refs_mean_sub_temp = res_temp[4]
            sci_mean_sub_temp = res_temp[5]
            resicube_klip[k, indices[0][0], indices[0][1]] = sub_img_rows_temp

            evals_matrix.append(evals_temp)
            evecs_matrix.append(evecs_temp)
            KL_basis_matrix.append(KL_basis_temp)
            refs_mean_sub_matrix.append(refs_mean_sub_temp)
            sci_mean_sub_matrix.append(sci_mean_sub_temp)

        mcube = cube_derotate(resicube_klip, pa, imlib=imlib,
                              interpolation=interpolation)

    elif model == 'LOCI':

        resicube, ind_ref_list, coef_list = LOCI_FM(cube, psf, ann_center, pa,
                                                    int(round(fwhm)), fwhm,
                                                    tolerance, delta_rot,
                                                    pa_threshold)
        mcube = cube_derotate(resicube, pa, imlib=imlib,
                              interpolation=interpolation)

    ceny, cenx = frame_center(cube[0])
    indices = get_annulus_segments(mcube[0], ann_center, 1, 1)
    indicesy = indices[0][0]
    indicesx = indices[0][1]

    flux_esti = np.zeros_like(indicesy)
    prob_esti = np.zeros_like(indicesy)

    var_f = _var_esti(mcube, pa, var, crop, ann_center)

    for i in range(0, len(indicesy)):

        psfm_temp = None
        poscenty = indicesy[i]
        poscentx = indicesx[i]

        indices = get_annulus_segments(cube[0], ann_center-int(round(fwhm)/2),
                                       int(round(fwhm)), 1)

        an_dist = np.sqrt((poscenty-ceny)**2 + (poscentx-cenx)**2)
        theta = np.degrees(np.arctan2(poscenty-ceny, poscentx-cenx))

        model_matrix = cube_inject_companions(np.zeros_like(cube), psf, pa,
                                              flevel=1, rad_dists=an_dist,
                                              theta=theta, n_branches=1,
                                              verbose=False, imlib=imlib,
                                              interpolation=interpolation)

        # PSF forward model computation for KLIP

        if model == 'KLIP':

            psf_map = np.zeros_like(model_matrix)

            for b in range(0, n):
                psf_map_temp = _perturb(b, model_matrix[:, indices[0][0],
                                                        indices[0][1]],
                                        ncomp, evals_matrix, evecs_matrix,
                                        KL_basis_matrix,
                                        sci_mean_sub_matrix,
                                        refs_mean_sub_matrix, pa, fwhm,
                                        pa_threshold, ann_center)

                psf_map[b, indices[0][0], indices[0][1]] = psf_map_temp
                psf_map[b, indices[0][0], indices[0]
                        [1]] -= np.mean(psf_map_temp)

            psf_map_der = cube_derotate(psf_map, pa, imlib=imlib,
                                        interpolation=interpolation)
            psfm_temp = cube_crop_frames(psf_map_der, int(2*round(fwhm)+1),
                                         xy=(poscentx, poscenty), verbose=False)

        # PSF forward model computation for LOCI

        if model == 'LOCI':

            values_fc = model_matrix[:, indices[0][0], indices[0][1]]

            cube_res_fc = np.zeros_like(model_matrix)

            matrix_res_fc = np.zeros((values_fc.shape[0],
                                      indices[0][0].shape[0]))

            for e in range(values_fc.shape[0]):

                recon_fc = np.dot(coef_list[e], values_fc[ind_ref_list[e]])
                matrix_res_fc[e] = values_fc[e] - recon_fc

            cube_res_fc[:, indices[0][0], indices[0][1]] = matrix_res_fc
            cube_der_fc = cube_derotate(cube_res_fc-np.mean(cube_res_fc),
                                        pa, imlib=imlib,
                                        interpolation=interpolation)
            psfm_temp = cube_crop_frames(cube_der_fc, int(2*round(fwhm)+1),
                                         xy=(poscentx, poscenty), verbose=False)

        num = []
        denom = []

        # Matched Filter

        for j in range(n):

            if var == 'FR':
                svar = var_f[j]

            elif var == 'FM':
                svar = var_f[i, j]

            elif var == 'TE':
                svar = var_f[i, j]

            if psfm_temp.shape[1] == crop:
                psfm = psfm_temp[j]
            else:
                psfm = frame_crop(psfm_temp[j],
                                  crop, cenxy=[int(psfm_temp.shape[-1]/2),
                                               int(psfm_temp.shape[-1]/2)],
                                  verbose=False)

            num.append(np.multiply(frame_crop(mcube[j], crop,
                                              cenxy=[poscentx, poscenty],
                                              verbose=False), psfm).sum()/svar)
            denom.append(np.multiply(psfm, psfm).sum()/svar)

        flux_esti[i] = sum(num)/np.sqrt(sum(denom))
        prob_esti[i] = sum(num)/sum(denom)

    if verbose == True:
        print("Radial distance "+"{}".format(ann_center)+" done!")

    return prob_esti, flux_esti, ann_center


def _var_esti(mcube, pa, var, crop, ann_center):
    """
    Computation of the residual noise variance
    """

    n, y, x = mcube.shape

    if var == 'FR':

        var_f = np.zeros(n)

        indices = get_annulus_segments(
            mcube[0], ann_center-int(crop/2), crop, 1)

        poscentx = indices[0][1]
        poscenty = indices[0][0]

        for a in range(n):

            var_f[a] = np.var(mcube[a, poscenty, poscentx])

    elif var == 'FM':

        indices = get_annulus_segments(mcube[0], ann_center, 1, 1)
        indicesy = indices[0][0]
        indicesx = indices[0][1]

        var_f = np.zeros((len(indicesy), n))

        indices = get_annulus_segments(
            mcube[0], ann_center-int(crop/2), crop, 1)

        for a in range(len(indicesy)):

            indc = disk((indicesy[a], indicesx[a]), 3)
            positionx = []
            positiony = []

            for k in range(0, len(indices[0][1])):
                cond1 = set(np.where(indices[0][1][k] == indc[1])[0])
                cond2 = set(np.where(indices[0][0][k] == indc[0])[0])
                if len(cond1 & cond2) == 0:
                    positionx.append(indices[0][1][k])
                    positiony.append(indices[0][0][k])

            for b in range((n)):

                var_f[a, b] = np.var(mcube[b, positiony, positionx])

    elif var == 'TE':

        indices = get_annulus_segments(mcube[0], ann_center, 1, 1)
        indicesy = indices[0][0]
        indicesx = indices[0][1]

        var_f = np.zeros((len(indicesy), n))

        mcube_derot = cube_derotate(mcube, -pa)

        for a in range(0, len(indicesy)):

            radist = np.sqrt((indicesx[a]-int(x/2)) **
                             2+(indicesy[a]-int(y/2))**2)

            if (indicesy[a]-int(y/2)) >= 0:
                ang_s = np.arccos((indicesx[a]-int(x/2))/radist)/np.pi*180
            else:
                ang_s = 360-np.arccos((indicesx[a]-int(x/2))/radist)/np.pi*180

            for b in range(n):

                twopi = 2*np.pi
                sigposy = int(y/2 + np.sin((ang_s-pa[b])/360*twopi)*radist)
                sigposx = int(x/2 + np.cos((ang_s-pa[b])/360*twopi)*radist)

                y0 = int(sigposy - int(crop/2))
                y1 = int(sigposy + int(crop/2)+1)  # +1 cause endpoint is
                # excluded when slicing
                x0 = int(sigposx - int(crop/2))
                x1 = int(sigposx + int(crop/2)+1)

                mask = np.ones(mcube_derot.shape[0], dtype=bool)
                mask[b] = False
                mcube_sel = mcube_derot[mask, y0:y1, x0:x1]

                var_f[a, b] = np.var(np.asarray(mcube_sel))

    return var_f


def _perturb(frame, model_matrix, numbasis, evals_matrix, evecs_matrix,
             KL_basis_matrix, sci_mean_sub_matrix, refs_mean_sub_matrix,
             angle_list, fwhm, pa_threshold, ann_center):
    """
    Function allowing the estimation of the PSF forward model when relying on
    KLIP for the computation of the speckle field. The code is based on the
    PyKLIP library considering only the ADI case with a singlle number of
    principal components considered. For more details about the code, consider
    the PyKLIP library or the original articles (Pueyo, L. 2016, ApJ, 824, 117
    or Ruffio, J.-B., Macintosh, B., Wang, J. J., & Pueyo, L. 2017, ApJ, 842)
    """

    # Selection of the reference library based on the given parallactic angle
    # threshold

    if pa_threshold != 0:
        indices_left = _find_indices_adi(angle_list, frame,
                                         pa_threshold, truncate=False)

        models_ref = model_matrix[indices_left]

    else:
        models_ref = model_matrix

    # Computation of the self-subtraction and over-subtraction for the current
    # frame

    model_sci = model_matrix[frame]
    KL_basis = KL_basis_matrix[frame]
    sci_mean_sub = sci_mean_sub_matrix[frame]
    refs_mean_sub = refs_mean_sub_matrix[frame]
    evals = evals_matrix[frame]
    evecs = evecs_matrix[frame]

    max_basis = KL_basis.shape[0]
    N_pix = KL_basis.shape[1]

    models_msub = models_ref - np.nanmean(models_ref, axis=1)[:, None]
    models_msub[np.where(np.isnan(models_msub))] = 0

    model_sci_msub = model_sci - np.nanmean(model_sci)
    model_sci_msub[np.where(np.isnan(model_sci_msub))] = 0
    model_sci_msub_rows = np.reshape(model_sci_msub, (1, N_pix))
    sci_mean_sub_rows = np.reshape(sci_mean_sub, (1, N_pix))

    delta_KL = np.zeros([max_basis, N_pix])

    proj_models_T = models_msub.dot(refs_mean_sub.transpose())

    for k in range(max_basis):
        Zk = np.reshape(KL_basis[k, :], (1, KL_basis[k, :].size))
        Vk = (evecs[:, k])[:, None]

        diagVk_T = (Vk.T).dot(proj_models_T)
        proj_models_Vk = proj_models_T.dot(Vk)

        fac = -(1/(2*np.sqrt(evals[k])))
        term1 = (diagVk_T.dot(Vk) + ((Vk.T).dot(proj_models_Vk))).dot(Zk)
        term2 = (Vk.T).dot(models_msub)
        DeltaZk = fac*term1 + term2

        for j in range(k):
            Zj = KL_basis[j, :][None, :]
            Vj = evecs[:, j][:, None]
            fac = np.sqrt(evals[j])/(evals[k]-evals[j])
            t1 = diagVk_T.dot(Vj)
            t2 = (Vj.T).dot(proj_models_Vk)
            DeltaZk += fac*(t1 + t2).dot(Zj)
        for j in range(k+1, max_basis):
            Zj = KL_basis[j, :][None, :]
            Vj = evecs[:, j][:, None]
            fac = np.sqrt(evals[j])/(evals[k]-evals[j])
            t1 = diagVk_T.dot(Vj)
            t2 = (Vj.T).dot(proj_models_Vk)
            DeltaZk += fac*(t1 + t2).dot(Zj)

        delta_KL[k] = DeltaZk/np.sqrt(evals[k])

    oversubtraction_inner_products = np.dot(model_sci_msub_rows, KL_basis.T)

    selfsubtraction_1_inner_products = np.dot(sci_mean_sub_rows, delta_KL.T)
    selfsubtraction_2_inner_products = np.dot(sci_mean_sub_rows, KL_basis.T)

    oversubtraction_inner_products[max_basis::] = 0
    klipped_oversub = np.dot(oversubtraction_inner_products, KL_basis)

    selfsubtraction_1_inner_products[0, max_basis::] = 0
    selfsubtraction_2_inner_products[0, max_basis::] = 0
    klipped_selfsub = np.dot(selfsubtraction_1_inner_products, KL_basis) + \
        np.dot(selfsubtraction_2_inner_products, delta_KL)

    return model_sci[None, :] - klipped_oversub - klipped_selfsub


def KLIP_patch(frame, matrix, numbasis, angle_list, fwhm, pa_threshold,
               ann_center, nframes=None):
    """
    Function allowing the computation of the reference PSF via KLIP for a
    given sub-region of the original ADI sequence. Code inspired by the
    PyKLIP librabry
    """

    max_frames_lib = 200

    if pa_threshold != 0:
        if ann_center > fwhm*20:
            indices_left = _find_indices_adi(angle_list, frame, pa_threshold,
                                             truncate=True,
                                             max_frames=max_frames_lib)
        else:
            indices_left = _find_indices_adi(angle_list, frame, pa_threshold,
                                             truncate=False, nframes=nframes)

        refs = matrix[indices_left]

    else:
        refs = matrix

    sci = matrix[frame]
    sci_mean_sub = sci - np.nanmean(sci)
    #sci_mean_sub[np.where(np.isnan(sci_mean_sub))] = 0
    refs_mean_sub = refs - np.nanmean(refs, axis=1)[:, None]
    #refs_mean_sub[np.where(np.isnan(refs_mean_sub))] = 0

    # Covariance matrix definition
    covar_psfs = np.cov(refs_mean_sub)
    covar_psfs *= (np.size(sci)-1)

    tot_basis = covar_psfs.shape[0]

    numbasis = np.clip(numbasis - 1, 0, tot_basis-1)
    max_basis = np.max(numbasis) + 1

    # Computation of the eigenvectors/values of the covariance matrix
    evals, evecs = la.eigh(covar_psfs)
    evals = np.copy(evals[int(tot_basis-max_basis):int(tot_basis)])
    evecs = np.copy(evecs[:, int(tot_basis-max_basis):int(tot_basis)])
    evals = np.copy(evals[::-1])
    evecs = np.copy(evecs[:, ::-1])

    # Computation of the principal components

    KL_basis = np.dot(refs_mean_sub.T, evecs)
    KL_basis = KL_basis * (1. / np.sqrt(evals))[None, :]
    KL_basis = KL_basis.T

    N_pix = np.size(sci_mean_sub)
    sci_rows = np.reshape(sci_mean_sub, (1, N_pix))

    inner_products = np.dot(sci_rows, KL_basis.T)
    inner_products[0, int(max_basis)::] = 0

    # Projection of the science image on the selected prinicpal component
    # to generate the speckle field model

    klip_reconstruction = np.dot(inner_products, KL_basis)

    # Subtraction of the speckle field model from the riginal science image
    # to obtain the residual frame

    sub_img_rows = sci_rows - klip_reconstruction

    return (evals, evecs, KL_basis, np.reshape(sub_img_rows, (N_pix)),
            refs_mean_sub, sci_mean_sub)


def LOCI_FM(cube, psf, ann_center, angle_list, asize, fwhm, Tol, delta_rot,
            pa_threshold):
    """
    Computation of the optimal factors weigthing the linear combination of
    reference frames used to obtain the modeled speckle field for each frame
    and allowing the determination of the forward modeled PSF. Estimation of
    the cube of residuals based on the modeled speckle field.
    """

    cube_res = np.zeros_like(cube)
    ceny, cenx = frame_center(cube[0])
    radius_int = ann_center-int(1.5*asize)
    if radius_int <= 0:
        radius_int = 1

    for ann in range(3):
        n_segments_ann = 1
        inner_radius_ann = radius_int + ann*asize

        indices = get_annulus_segments(cube[0], inner_radius=inner_radius_ann,
                                       width=asize, nsegm=n_segments_ann)
        ind_opt = get_annulus_segments(cube[0], inner_radius=inner_radius_ann,
                                       width=asize, nsegm=n_segments_ann,
                                       optim_scale_fact=2)

        ayxyx = [inner_radius_ann, pa_threshold, indices[0][0], indices[0][1],
                 ind_opt[0][0], ind_opt[0][1]]

        matrix_res, ind_ref, coef, yy, xx = _leastsq_patch_fm(ayxyx, angle_list,
                                                              fwhm, cube, 100,
                                                              Tol, psf=psf)

        if ann == 1:
            ind_ref_list = ind_ref
            coef_list = coef

        cube_res[:, yy, xx] = matrix_res

    return cube_res, ind_ref_list, coef_list


def _leastsq_patch_fm(ayxyx, angle_list, fwhm, cube, dist_threshold,
                      tol, psf=None):
    """
    Function allowing th estimation of the optimal factors for the modeled
    speckle field estimation via the LOCI framework. The code has been
    developped based on the VIP python function _leastsq_patch, but return
    additionnaly the set of coefficients used for the speckle field computation.
    """

    ann_center, pa_threshold, yy, xx, yy_opti, xx_opti = ayxyx

    ind_ref_list = []
    coef_list = []

    yy_opt = []
    xx_opt = []

    for j in range(0, len(yy_opti)):
        if not any(x in np.where(yy == yy_opti[j])[0]
                   for x in np.where(xx == xx_opti[j])[0]):
            xx_opt.append(xx_opti[j])
            yy_opt.append(yy_opti[j])

    values = cube[:, yy, xx]
    matrix_res = np.zeros((values.shape[0], yy.shape[0]))
    values_opt = cube[:, yy_opti, xx_opti]
    n_frames = cube.shape[0]

    for i in range(n_frames):

        ind_fr_i = _find_indices_adi(angle_list, i,
                                     pa_threshold, truncate=False)
        if len(ind_fr_i) > 0:
            A = values_opt[ind_fr_i]
            b = values_opt[i]
            coef = np.linalg.lstsq(A.T, b, rcond=tol)[0]     # SVD method
        else:
            msg = "No frames left in the reference set. Try increasing "
            msg += "`dist_threshold` or decreasing `delta_rot`."
            raise RuntimeError(msg)

        ind_ref_list.append(ind_fr_i)
        coef_list.append(coef)
        recon = np.dot(coef, values[ind_fr_i])
        matrix_res[i] = values[i] - recon

    return matrix_res, ind_ref_list, coef_list, yy, xx
