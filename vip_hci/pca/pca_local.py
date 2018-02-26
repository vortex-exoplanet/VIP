#! /usr/bin/env python

"""
Module with local smart pca (annulus-wise) serial and parallel implementations.
"""

from __future__ import division, print_function

__author__ = 'Carlos Alberto Gomez Gonzalez'
__all__ = ['pca_adi_annular',
           'pca_rdi_annular']

import numpy as np
import itertools as itt
from scipy import stats
from multiprocessing import Pool, cpu_count
from ..preproc import cube_derotate, cube_collapse, check_pa_vector
from ..conf import time_ini, timing
from ..conf.utils_conf import eval_func_tuple as EFT
from ..var import get_annulus_segments, matrix_scaling, get_annulus
from ..madi.adi_utils import _find_indices, _define_annuli
from ..stats import descriptive_stats
from .svd import get_eigenvectors


def pca_rdi_annular(cube, angle_list, cube_ref, radius_int=0, asize=1, 
                    ncomp=1, svd_mode='randsvd', min_corr=0.9, fwhm=4, 
                    scaling='temp-standard', imlib='opencv',
                    interpolation='lanczos4', collapse='median',
                    full_output=False, verbose=True):
    """ Annular PCA with Reference Library + Correlation + standardization
    
    In the case of having a large number of reference images, e.g. for a survey 
    on a single instrument, we can afford a better selection of the library by 
    constraining the correlation with the median of the science dataset and by 
    working on an annulus-wise way. As with other local PCA algorithms in VIP
    the number of principal components can be automatically adjusted by the
    algorithm by minmizing the residuals in the given patch (a la LOCI). 
    
    Parameters
    ----------
    cube : array_like, 3d
        Input science cube.
    angle_list : array_like, 1d
        Corresponding parallactic angle for each frame.
    cube_ref : array_like, 3d
        Reference library cube. For Reference Star Differential Imaging.
    radius_int : int, optional
        The radius of the innermost annulus. By default is 0, if >0 then the 
        central circular area is discarded.
    asize : int, optional
        The size of the annuli, in FWHM. Default is 3.
    ncomp : int, optional
        How many PCs are kept. If none it will be automatically determined.
    svd_mode : {'lapack', 'arpack', 'eigen', 'randsvd', 'cupy', 'eigencupy', 'randcupy'}, str
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
        ``randcupy`` is an adaptation of the randomized_svd algorith, where all 
        the computations are done on a GPU. 
    min_corr : int, optional
        Level of linear correlation between the library patches and the median 
        of the science. Deafult is 0.9.
    fwhm : float, optional
        Known size of the FHWM in pixels to be used. Deafult is 4.
    scaling : {None, 'temp-mean', 'spat-mean', 'temp-standard', 'spat-standard'}
        With None, no scaling is performed on the input data before SVD. With 
        "temp-mean" then temporal px-wise mean subtraction is done, with 
        "spat-mean" then the spatial mean is subtracted, with "temp-standard" 
        temporal mean centering plus scaling to unit variance is done and with
        "spat-standard" spatial mean centering plus scaling to unit variance is
        performed.
    imlib : str, optional
        See the documentation of the ``vip_hci.preproc.frame_rotate`` function.
    interpolation : str, optional
        See the documentation of the ``vip_hci.preproc.frame_rotate`` function.
    collapse : {'median', 'mean', 'sum', 'trimmean'}, str optional
        Sets the way of collapsing the frames for producing a final image.
    full_output: boolean, optional
        Whether to return the final median combined image only or with other 
        intermediate arrays.  
    verbose : {True, False}, bool optional
        If True prints to stdout intermediate info.
    
    Returns
    -------
    frame : array_like, 2d    
        Median combination of the de-rotated cube.
    If full_output is True:  
    array_out : array_like, 3d 
        Cube of residuals.
    array_der : array_like, 3d
        Cube residuals after de-rotation.    
    
    """
    def define_annuli(angle_list, ann, n_annuli, fwhm, radius_int, annulus_width, 
                      verbose):
        """ Defining the annuli """
        if ann == n_annuli-1:
            inner_radius = radius_int + (ann*annulus_width-1)
        else:                                                                                         
            inner_radius = radius_int + ann*annulus_width
        ann_center = (inner_radius+(annulus_width/2.0))
        
        if verbose:
            msg2 = 'Annulus {:}, Inn radius = {:.2f}, Ann center = {:.2f} '
            print(msg2.format(int(ann+1),inner_radius, ann_center))
        return inner_radius, ann_center
    
    def fr_ref_correlation(vector, matrix):
        """ Getting the correlations """
        lista = []
        for i in range(matrix.shape[0]):
            pears, _ = stats.pearsonr(vector, matrix[i])
            lista.append(pears)
        
        return lista

    def do_pca_annulus(ncomp, matrix, svd_mode, noise_error, data_ref):
        """ Actual PCA for the annulus """
        #V = svd_wrapper(data_ref, svd_mode, ncomp, debug=False, verbose=False)
        V = get_eigenvectors(ncomp, matrix, svd_mode, noise_error=noise_error, 
                             data_ref=data_ref, debug=False)
        # new variables as linear combinations of the original variables in 
        # matrix.T with coefficientes from EV
        transformed = np.dot(V, matrix.T) 
        reconstructed = np.dot(V.T, transformed)
        residuals = matrix - reconstructed.T  
        return residuals, V.shape[0]

    #---------------------------------------------------------------------------
    array = cube
    array_ref = cube_ref
    if not array.ndim == 3:
        raise TypeError('Input array is not a cube or 3d array.')
    if not array.shape[0] == angle_list.shape[0]:
        raise TypeError('Input vector or parallactic angles has wrong length.')
    
    n, y, _ = array.shape
    if verbose:  start_time = time_ini()
    
    angle_list = check_pa_vector(angle_list)

    annulus_width = asize * fwhm                     # equal size for all annuli
    n_annuli = int(np.floor((y/2-radius_int)/annulus_width))
    if verbose:
        msg = '# annuli = {:}, Ann width = {:}, FWHM = {:.3f}\n'
        print(msg.format(n_annuli, annulus_width, fwhm))
        print('PCA will be done locally per annulus and per quadrant.\n')
     
    cube_out = np.zeros_like(array)
    for ann in range(n_annuli):
        inner_radius, _ = define_annuli(angle_list, ann, n_annuli, fwhm, 
                                        radius_int, annulus_width, verbose) 
        indices = get_annulus(array[0], inner_radius, annulus_width,
                              output_indices=True)
        yy = indices[0]
        xx = indices[1]
                    
        matrix = array[:, yy, xx]                 # shape [nframes x npx_ann] 
        matrix_ref = array_ref[:, yy, xx]
        
        corr = fr_ref_correlation(np.median(matrix, axis=0), matrix_ref)
        indcorr = np.where(np.abs(corr)>=min_corr)
        #print indcorr
        data_ref = matrix_ref[indcorr]
        nfrslib = data_ref.shape[0]
                
        if nfrslib<5:
            msg = 'Too few frames left (<5) fulfill the given correlation level.'
            msg += 'Try decreasing it'
            raise RuntimeError(msg)
        
        matrix = matrix_scaling(matrix, scaling)
        data_ref = matrix_scaling(data_ref, scaling)
        
        residuals, ncomps = do_pca_annulus(ncomp, matrix, svd_mode, 10e-3, data_ref)  
        cube_out[:, yy, xx] = residuals  
            
        if verbose:
            print('# frames in LIB = {}'.format(nfrslib))
            print('# PCs = {}'.format(ncomps))
            print('Done PCA with {:} for current annulus'.format(svd_mode))
            timing(start_time)      
         
    cube_der = cube_derotate(cube_out, angle_list, imlib=imlib,
                             interpolation=interpolation)
    frame = cube_collapse(cube_der, mode=collapse)
    if verbose:
        print('Done derotating and combining.')
        timing(start_time)
    if full_output:
        return cube_out, cube_der, frame 
    else:
        return frame           


def pca_adi_annular(cube, angle_list, radius_int=0, fwhm=4, asize=3,
                    n_segments=1, delta_rot=1, ncomp=1, svd_mode='randsvd',
                    nproc=1, min_frames_lib=10, max_frames_lib=200, tol=1e-1,
                    scaling=None, imlib='opencv', interpolation='lanczos4',
                    collapse='median', full_output=False, verbose=True):
    """ Annular (smart) ADI PCA. The PCA model is computed locally in each
    annulus (optionally quadrants of each annulus). For each annulus we discard
    reference images taking into account a parallactic angle threshold
    (set by ``delta_rot``).
     
    Depending on parameter ``nproc`` the algorithm can work with several cores.
    It's been tested on a Linux and OSX. The ACCELERATE library for linear 
    algebra calcularions, which comes by default in every OSX system, is broken 
    for multiprocessing. Avoid using this function unless you have compiled 
    Python against other linear algebra library. An easy fix is to install 
    (ana)conda and the openblas or MKL libraries (replacing ACCELERATE).
    
    Parameters
    ----------
    cube : array_like, 3d
        Input cube.
    angle_list : array_like, 1d
        Corresponding parallactic angle for each frame.
    radius_int : int, optional
        The radius of the innermost annulus. By default is 0, if >0 then the 
        central circular area is discarded.
    fwhm : float, optional
        Known size of the FHWM in pixels to be used. Deafult is 4.
    asize : int, optional
        The size of the annuli, in FWHM. Default is 3.
    n_segments : int or list of ints, optional
        The number of segments for each annulus. When a single integer is given
        it is used for all annuli.
    delta_rot : int, optional
        Factor for increasing the parallactic angle threshold, expressed in FWHM.
        Default is 1 (excludes 1 FHWM on each side of the considered frame).
        According to Absil+13, a slightly better contrast can be reached for the 
        innermost annuli if we consider a ``delta_rot`` condition as small as 
        0.1 lambda/D. This is because at very small separation, the effect of 
        speckle correlation is more significant than self-subtraction.
    ncomp : int or list or 1d numpy array, optional
        How many PCs are kept. If none it will be automatically determined. If a
        list is provided and it matches the number of annuli then a different
        number of PCs will be used for each annulus (starting with the innermost
        one).
    svd_mode : {'lapack', 'arpack', 'eigen', 'randsvd', 'cupy', 'eigencupy', 'randcupy'}, str
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
        ``randcupy`` is an adaptation of the randomized_svd algorith, where all 
        the computations are done on a GPU. 
    nproc : None or int, optional
        Number of processes for parallel computing. If None the number of 
        processes will be set to (cpu_count()/2). By default the algorithm works
        in single-process mode.
    min_frames_lib : int, optional
        Minimum number of frames in the PCA reference library. Be careful, when
        ``min_frames_pca`` < ``ncomp``, then for certain frames the subtracted
        low-rank approximation is not possible. It is recommended to decrease
        ``delta_rot`` and have enough frames in the libraries to allow getting 
        ``ncomp`` PCs.
    max_frames_lib : int, optional
        Maximum number of frames in the PCA reference library for annuli beyond
        10*FWHM. The more distant/decorrelated frames are removed from the
        library.
    tol : float, optional
        Stopping criterion for choosing the number of PCs when ``ncomp`` is None.
        Lower values will lead to smaller residuals and more PCs.
    imlib : str, optional
        See the documentation of the ``vip_hci.preproc.frame_rotate`` function.
    interpolation : str, optional
        See the documentation of the ``vip_hci.preproc.frame_rotate`` function.
    collapse : {'median', 'mean', 'sum', 'trimmean'}, str optional
        Sets the way of collapsing the frames for producing a final image.
    full_output: boolean, optional
        Whether to return the final median combined image only or with other 
        intermediate arrays.  
    scaling : {None, 'temp-mean', 'spat-mean', 'temp-standard', 'spat-standard'}
        With None, no scaling is performed on the input data before SVD. With 
        "temp-mean" then temporal px-wise mean subtraction is done, with 
        "spat-mean" then the spatial mean is subtracted, with "temp-standard" 
        temporal mean centering plus scaling to unit variance is done and with
        "spat-standard" spatial mean centering plus scaling to unit variance is
        performed.  
    verbose : {True, False}, bool optional
        If True prints to stdout intermediate info.
     
    Returns
    -------
    frame : array_like, 2d    
        Median combination of the de-rotated cube.
    If full_output is True:  
    array_out : array_like, 3d 
        Cube of residuals.
    array_der : array_like, 3d
        Cube residuals after de-rotation.
     
    """
    array = cube
    if not array.ndim == 3:
        raise TypeError('Input array is not a cube or 3d array')
    if not array.shape[0] == angle_list.shape[0]:
        raise TypeError('Input vector or parallactic angles has wrong length')

    n, y, _ = array.shape
     
    if verbose:
        start_time = time_ini()
    
    angle_list = check_pa_vector(angle_list)

    annulus_width = asize * fwhm                     # equal size for all annuli
    n_annuli = int(np.floor((y/2 - radius_int) / annulus_width))

    if isinstance(n_segments, int):
        n_segments = [n_segments for _ in range(n_annuli)]
    elif n_segments == 'auto':
        n_segments = []
        n_segments.append(2)  # for first annulus
        n_segments.append(3)  # for second annulus
        ld = 2 * np.tan(360 / 4 / 2) * annulus_width
        for i in range(2, n_annuli):  # rest of annuli
            radius = i * annulus_width
            ang = np.rad2deg(2 * np.arctan(ld / (2 * radius)))
            n_segments.append(int(np.ceil(360 / ang)))

    if verbose:
        msg = '# annuli = {:}, Ann width = {:}, FWHM = {:.3f} \n'
        print(msg.format(n_annuli, annulus_width, fwhm))
        print('PCA per annulus (or annular segment)\n')

    if nproc is None:   # Hyper-threading "duplicates" the cores -> cpu_count/2
        nproc = (cpu_count() / 2)
    
    #***************************************************************************
    # The annuli are built, and the corresponding PA thresholds for frame 
    # rejection are calculated (at the center of the annulus)
    #***************************************************************************
    cube_out = np.zeros_like(array)
    for ann in range(n_annuli):
        if isinstance(ncomp, list) or isinstance(ncomp, np.ndarray):
            ncomp = list(ncomp)
            if len(ncomp) == n_annuli:
                ncompann = ncomp[ann]
            else:
                msge = 'If ncomp is a list, it must match the number of annuli'
                raise TypeError(msge)
        else:
            ncompann = ncomp

        n_segments_ann = n_segments[ann]
        res_ann_par = _define_annuli(angle_list, ann, n_annuli, fwhm,
                                     radius_int, annulus_width, delta_rot,
                                     n_segments_ann, verbose)
        pa_thr, inner_radius, ann_center = res_ann_par
        indices = get_annulus_segments(array[0], inner_radius,
                                       annulus_width, n_segments_ann, 0)

        # **********************************************************************
        # Library matrix is created for each segment and scaled if needed
        # **********************************************************************
        for j in range(n_segments_ann):
            yy = indices[j][0]
            xx = indices[j][1]
            matrix_segm = array[:, yy, xx]  # shape [nframes x npx_segment]
            matrix_segm = matrix_scaling(matrix_segm, scaling)

            # ***************************************************************
            # We loop the frames and do the PCA to obtain the residuals cube
            # ***************************************************************
            residuals = do_pca_loop(matrix_segm, nproc, angle_list, fwhm,
                                    pa_thr, ann_center, svd_mode, ncompann,
                                    min_frames_lib, max_frames_lib, tol,
                                    verbose)
            for frame in range(n):
                cube_out[frame][yy, xx] = residuals[frame]

        if verbose:
            print('Done PCA with {:} for current annulus'.format(svd_mode))
            timing(start_time)      
         
    #***************************************************************************
    # Cube is derotated according to the parallactic angle and median combined.
    #***************************************************************************
    cube_der = cube_derotate(cube_out, angle_list, imlib=imlib,
                             interpolation=interpolation)
    frame = cube_collapse(cube_der, mode=collapse)
    if verbose:
        print('Done derotating and combining.')
        timing(start_time)
    if full_output:
        return cube_out, cube_der, frame 
    else:
        return frame 
        
    
    
################################################################################
### Help functions (encapsulating portions of the main algos)
################################################################################
    
def do_pca_loop(matrix, nproc, angle_list, fwhm, pa_threshold, ann_center,
                svd_mode, ncomp, min_frames_lib, max_frames_lib, tol, verbose):
    """
    """
    matrix_ann = matrix
    n = matrix.shape[0]
    #***************************************************************
    # For each frame we call the subfunction do_pca_patch that will 
    # do PCA on the small matrix, where some frames are discarded 
    # according to the PA threshold, and return the residuals
    #***************************************************************          
    ncomps = []
    nfrslib = []          
    if nproc == 1:
        residualarr = []
        for frame in range(n):
            res = do_pca_patch(matrix_ann, frame, angle_list, fwhm,
                               pa_threshold, ann_center, svd_mode, ncomp,
                               min_frames_lib, max_frames_lib, tol)
            residualarr.append(res[0])
            ncomps.append(res[1])
            nfrslib.append(res[2])
        residuals = np.array(residualarr)
        
    elif nproc > 1:
        #***********************************************************
        # A multiprocessing pool is created to process the frames in 
        # a parallel way. SVD/PCA is done in do_pca_patch function
        #***********************************************************            
        pool = Pool(processes=int(nproc))
        res = pool.map(EFT, zip(itt.repeat(do_pca_patch),
                                itt.repeat(matrix_ann), range(n),
                                itt.repeat(angle_list), itt.repeat(fwhm),
                                itt.repeat(pa_threshold),
                                itt.repeat(ann_center), itt.repeat(svd_mode),
                                itt.repeat(ncomp), itt.repeat(min_frames_lib),
                                itt.repeat(max_frames_lib), itt.repeat(tol)))
        res = np.array(res)
        residuals = np.array(res[:,0])
        ncomps = res[:, 1]
        nfrslib = res[:, 2]
        pool.close()                         

    # number of frames in library printed for each annular quadrant
    if verbose:
        descriptive_stats(nfrslib, verbose=verbose, label='Size LIB: ')
    # number of PCs printed for each annular quadrant     
    if ncomp is None and verbose:  
        descriptive_stats(ncomps, verbose=verbose, label='Numb PCs: ')
        
    return residuals


def do_pca_patch(matrix, frame, angle_list, fwhm, pa_threshold, ann_center,
                 svd_mode, ncomp, min_frames_lib, max_frames_lib,  tol):
    """
    Does the SVD/PCA for each frame patch (small matrix). For each frame we 
    find the frames to be rejected depending on the amount of rotation. The
    library is also truncated on the other end (frames too far or which have 
    rotated more) which are more decorrelated to keep the computational cost 
    lower. This truncation is done on the annuli after 10*FWHM and the goal is
    to keep min(num_frames/2, 200) in the library. 
    """
    if pa_threshold != 0:
        if ann_center > fwhm*10:    # TODO: 10*FWHM optimal? new parameter?
            indices_left = _find_indices(angle_list, frame, pa_threshold,
                                         truncate=True,
                                         max_frames=max_frames_lib)
        else:
            indices_left = _find_indices(angle_list, frame, pa_threshold,
                                         truncate=False)
         
        data_ref = matrix[indices_left]
        
        if data_ref.shape[0] <= min_frames_lib:
            msg = 'Too few frames left in the PCA library. '
            msg += 'Try decreasing either delta_rot or min_frames_lib.'
            raise RuntimeError(msg)
    else:
        data_ref = matrix
       
    data = data_ref
    #data = data_ref - data_ref.mean(axis=0)
    curr_frame = matrix[frame]                     # current frame
    
    V = get_eigenvectors(ncomp, data, svd_mode, noise_error=tol, debug=False)        
    
    transformed = np.dot(curr_frame, V.T)
    reconstructed = np.dot(transformed.T, V)                        
    residuals = curr_frame - reconstructed     
    return residuals, V.shape[0], data_ref.shape[0]  




