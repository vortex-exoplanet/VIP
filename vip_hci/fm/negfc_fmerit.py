#! /usr/bin/env python

"""
Module with the function of merit definitions for the NEGFC optimization.
"""

__author__ = 'O. Wertz, Carlos Alberto Gomez Gonzalez, Valentin Christiaens'
__all__ = []

import numpy as np
from hciplot import plot_frames
from skimage.draw import disk
from ..fm import cube_inject_companions
from ..var import (frame_center, get_annular_wedge, cube_filter_highpass, dist,
                   get_circle)
from ..psfsub import pca_annulus, pca_annular, pca
from ..preproc import cube_crop_frames
from ..config import check_array

def chisquare(modelParameters, cube, angs, plsc, psfs_norm, fwhm, annulus_width,  
              aperture_radius, initialState, ncomp, cube_ref=None, 
              svd_mode='lapack', scaling=None, fmerit='sum', collapse='median',
              algo=pca_annulus, delta_rot=1, imlib='vip-fft', 
              interpolation='lanczos4', algo_options={}, transmission=None, 
              mu_sigma=None, weights=None, force_rPA=False, debug=False):
    r"""
    Calculate the reduced :math:`\chi^2`:
    .. math:: \chi^2_r = \frac{1}{N-3}\sum_{j=1}^{N} |I_j|,
    where N is the number of pixels within a circular aperture centered on the 
    first estimate of the planet position, and :math:`I_j` the j-th pixel 
    intensity.
    
    Parameters
    ----------    
    modelParameters: tuple
        The model parameters, typically (r, theta, flux).
    cube: numpy.array
        The cube of fits images expressed as a numpy.array.
    angs: numpy.array
        The parallactic angle fits image expressed as a numpy.array. 
    plsc: float
        The platescale, in arcsec per pixel.
    psfs_norm: numpy.array
        The scaled psf expressed as a numpy.array.    
    fwhm : float
        The FHWM in pixels.
    annulus_width: int, optional
        The width in pixels of the annulus on which the PCA is done.       
    aperture_radius: int, optional
        The radius of the circular aperture in terms of the FWHM.
    initialState: numpy.array
        Position (r, theta) of the circular aperture center.
    ncomp: int or None
        The number of principal components for PCA-based algorithms.
    cube_ref : numpy ndarray, 3d, optional
        Reference library cube. For Reference Star Differential Imaging.
    svd_mode : {'lapack', 'randsvd', 'eigen', 'arpack'}, str optional
        Switch for different ways of computing the SVD and selected PCs.         
    scaling : {'temp-mean', 'temp-standard'} or None, optional
        With None, no scaling is performed on the input data before SVD. With 
        "temp-mean" then temporal px-wise mean subtraction is done and with 
        "temp-standard" temporal mean centering plus scaling to unit variance 
        is done. 
    fmerit : {'sum', 'stddev'}, string optional
        Chooses the figure of merit to be used. stddev works better for close in
        companions sitting on top of speckle noise.
    collapse : {'median', 'mean', 'sum', 'trimmean', None}, str or None, optional
        Sets the way of collapsing the frames for producing a final image. If
        None then the cube of residuals is used when measuring the function of
        merit (instead of a single final frame).
    algo: python routine, opt {pca_annulus, pca_annular, pca, custom}
        Routine to be used to model and subtract the stellar PSF. From an input
        cube, derotation angles, and optional arguments, it should return a 
        post-processed frame.
    delta_rot: float, optional
        If algo is set to pca_annular, delta_rot is the angular threshold used
        to select frames in the PCA library (see description of pca_annular).
    imlib : str, optional
        See the documentation of the ``vip_hci.preproc.frame_shift`` function.
    interpolation : str, optional
        See the documentation of the ``vip_hci.preproc.frame_shift`` function.
    algo_options: dict, opt
        Dictionary with additional parameters related to the algorithm 
        (e.g. tol, min_frames_lib, max_frames_lib). If 'algo' is not a vip
        routine, this dict should contain all necessary arguments apart from
        the cube and derotation angles. Note: arguments such as ncomp, svd_mode,
        scaling, imlib, interpolation or collapse can also be included in this
        dict (the latter are also kept as function arguments for consistency
        with older versions of vip). 
    transmission: numpy array, optional
        Array with 2 columns. First column is the radial separation in pixels. 
        Second column is the off-axis transmission (between 0 and 1) at the 
        radial separation given in column 1.
    mu_sigma: tuple of 2 floats or None, opt
        If set to None: not used, and falls back to original version of the 
        algorithm, using fmerit.
        If set to anything but None: will compute the mean and standard 
        deviation of pixel intensities in an annulus centered on the location 
        of the companion, excluding the area directly adjacent to the companion.
    weights : 1d array, optional
        If provided, the negative fake companion fluxes will be scaled according
        to these weights before injection in the cube. Can reflect changes in 
        the observing conditions throughout the sequence.
    force_rPA: bool, optional
        Whether to only search for optimal flux, provided (r,PA).
    debug: bool, opt
        Whether to debug and plot the post-processed frame after injection of 
        the negative fake companion.
        
    Returns
    -------
    out: float
        The reduced chi squared.
        
    """
    if force_rPA:
        r, theta = initialState
        flux_tmp = modelParameters[0]
    else:
        try:
            r, theta, flux_tmp = modelParameters
        except TypeError:
            msg = 'modelParameters must be a tuple, {} was given'
            print(msg.format(type(modelParameters)))

    ## set imlib for rotation and shift
    if imlib == 'opencv':
        imlib_sh = imlib
        imlib_rot = imlib
    elif imlib == 'skimage' or imlib == 'ndimage-interp':
        imlib_sh = 'ndimage-interp'
        imlib_rot = 'skimage'
    elif imlib == 'vip-fft' or imlib == 'ndimage-fourier':
        imlib_sh = 'ndimage-fourier'
        imlib_rot = 'vip-fft'
    else:
        raise TypeError("Interpolation not recognized.")

    if weights is None:   
        flux = -flux_tmp
        norm_weights=weights
    else:
        flux = -flux_tmp*weights
        norm_weights = weights/np.sum(weights)

    # Create the cube with the negative fake companion injected
    cube_negfc = cube_inject_companions(cube, psfs_norm, angs, flevel=flux,
                                        plsc=plsc, rad_dists=[r], n_branches=1,
                                        theta=theta, imlib=imlib_sh, 
                                        verbose=False, 
                                        interpolation=interpolation, 
                                        transmission=transmission)
                                      
    # Perform PCA and extract the zone of interest
    res = get_values_optimize(cube_negfc, angs, ncomp, annulus_width,
                              aperture_radius, fwhm, initialState[0],
                              initialState[1], cube_ref=cube_ref,
                              svd_mode=svd_mode, scaling=scaling, algo=algo,
                              delta_rot=delta_rot, collapse=collapse, 
                              algo_options=algo_options, weights=norm_weights, 
                              imlib=imlib_rot, interpolation=interpolation, 
                              debug=debug)
    
    if debug and collapse is not None:
        values, frpca = res
        plot_frames(frpca)
    else:
        values = res
    
    # Function of merit
    if mu_sigma is None:
        # old version - delete?
        if fmerit == 'sum':
            chi = np.sum(np.abs(values))
        elif fmerit == 'stddev':
            values = values[values != 0]
            chi = np.std(values)*values.size # TODO: test std**2
        else:
            raise RuntimeError('fmerit choice not recognized.')
    else:
        # true expression of a gaussian log probability
        mu = mu_sigma[0]
        sigma = mu_sigma[1]
        chi = np.sum(np.power(mu-values,2)/sigma**2)   
        
    return chi


def get_values_optimize(cube, angs, ncomp, annulus_width, aperture_radius, 
                        fwhm, r_guess, theta_guess, cube_ref=None, 
                        svd_mode='lapack', scaling=None, algo=pca_annulus, 
                        delta_rot=1, imlib='vip-fft', interpolation='lanczos4',
                        collapse='median', algo_options={}, weights=None, 
                        debug=False):
    """ Extracts a PCA-ed annulus from the cube and returns the flux values of
    the pixels included in a circular aperture centered at a given position.
    
    Parameters
    ----------
    cube: numpy.array
        The cube of fits images expressed as a numpy.array.
    angs: numpy.array
        The parallactic angle fits image expressed as a numpy.array.
    ncomp: int or None
        The number of principal components for PCA-based algorithms.
    annulus_width: float
        The width in pixels of the annulus on which the PCA is performed.
    aperture_radius: float
        The radius in fwhm of the circular aperture.
    fwhm: float
        Value of the FWHM of the PSF.
    r_guess: float
        The radial position of the center of the circular aperture. This 
        parameter is NOT the radial position of the candidate associated to the 
        Markov chain, but should be the fixed initial guess.
    theta_guess: float
        The angular position of the center of the circular aperture. This 
        parameter is NOT the angular position of the candidate associated to the 
        Markov chain, but should be the fixed initial guess.  
    cube_ref : numpy ndarray, 3d, optional
        Reference library cube. For Reference Star Differential Imaging.
    svd_mode : {'lapack', 'randsvd', 'eigen', 'arpack'}, str optional
        Switch for different ways of computing the SVD and selected PCs.
    scaling : {None, 'temp-mean', 'temp-standard'}
        With None, no scaling is performed on the input data before SVD. With 
        "temp-mean" then temporal px-wise mean subtraction is done and with 
        "temp-standard" temporal mean centering plus scaling to unit variance 
        is done.
    algo: python routine, opt {pca_annulus, pca_annular, pca, custom}
        Routine to be used to model and subtract the stellar PSF. From an input
        cube, derotation angles, and optional arguments, it should return a 
        post-processed frame.
    delta_rot: float, optional
        If algo is set to pca_annular, delta_rot is the angular threshold used
        to select frames in the PCA library (see description of pca_annular).
    imlib : str, optional
        See the documentation of the ``vip_hci.preproc.frame_rotate`` function.
    interpolation : str, optional
        See the documentation of the ``vip_hci.preproc.frame_rotate`` function.
    collapse : {'median', 'mean', 'sum', 'trimmean', None}, str or None, optional
        Sets the way of collapsing the frames for producing a final image. If
        None then the cube of residuals is returned.
    algo_options: dict, opt
        Dictionary with additional parameters related to the algorithm 
        (e.g. tol, min_frames_lib, max_frames_lib). If 'algo' is not a vip
        routine, this dict should contain all necessary arguments apart from
        the cube and derotation angles. Note: arguments such as ncomp, svd_mode,
        scaling, imlib, interpolation or collapse can also be included in this
        dict (the latter are also kept as function arguments for consistency
        with older versions of vip). 
    weights : 1d array, optional
        If provided, the negative fake companion fluxes will be scaled according
        to these weights before injection in the cube. Can reflect changes in 
        the observing conditions throughout the sequence.
    debug: boolean
        If True, the cube is returned along with the values.        
        
    Returns
    -------
    values: numpy.array
        The pixel values in the circular aperture after the PCA process.

    If debug is True and collapse non-None, the PCA frame is also returned.
        
    """
    centy_fr, centx_fr = frame_center(cube[0])
    posy = r_guess * np.sin(np.deg2rad(theta_guess)) + centy_fr
    posx = r_guess * np.cos(np.deg2rad(theta_guess)) + centx_fr
    halfw = max(aperture_radius*fwhm, annulus_width/2)

    # Checking annulus/aperture sizes. Assuming square frames
    msg = 'The annulus and/or the circular aperture used by the NegFC falls '
    msg += 'outside the FOV. Try increasing the size of your frames or '
    msg += 'decreasing the annulus or aperture size.'
    msg += 'rguess: {:.0f}px; centx_fr: {:.0f}px'.format(r_guess,centx_fr)
    msg += 'halfw: {:.0f}px'.format(halfw)
    if r_guess > centx_fr-halfw:# or r_guess <= halfw:
        raise RuntimeError(msg)

    ncomp = algo_options.get('ncomp', ncomp)
    svd_mode = algo_options.get('svd_mode', svd_mode)
    scaling = algo_options.get('scaling', scaling)
    imlib = algo_options.get('imlib', imlib)
    interpolation = algo_options.get('interpolation', interpolation)
    collapse = algo_options.get('collapse', collapse)


    if algo == pca_annulus:
        pca_res = pca_annulus(cube, angs, ncomp, annulus_width, r_guess, 
                              cube_ref, svd_mode, scaling, imlib=imlib,
                              interpolation=interpolation, collapse=collapse,
                              weights=weights)
        
    elif algo == pca_annular:
                
        tol = algo_options.get('tol',1e-1)
        min_frames_lib=algo_options.get('min_frames_lib',2)
        max_frames_lib=algo_options.get('max_frames_lib',200)
        radius_int = max(1,int(np.floor(r_guess-annulus_width/2)))
        radius_int = algo_options.get('radius_int', radius_int)
        # crop cube to just be larger than annulus => FASTER PCA
        crop_sz = int(2*np.ceil(radius_int+annulus_width+1))
        if not crop_sz %2:
            crop_sz+=1
        if crop_sz < cube.shape[1] and crop_sz < cube.shape[2]:
            pad = int((cube.shape[1]-crop_sz)/2)
            crop_cube = cube_crop_frames(cube, crop_sz, verbose=False)
        else:
            crop_cube = cube


        pca_res_tmp = pca_annular(crop_cube, angs, radius_int=radius_int, 
                                  fwhm=fwhm, asize=annulus_width, 
                                  delta_rot=delta_rot, ncomp=ncomp, 
                                  svd_mode=svd_mode, scaling=scaling, 
                                  imlib=imlib, interpolation=interpolation,
                                  collapse=collapse, weights=weights, tol=tol, 
                                  min_frames_lib=min_frames_lib, 
                                  max_frames_lib=max_frames_lib, 
                                  full_output=False, verbose=False)
        # pad again now                      
        pca_res = np.pad(pca_res_tmp,pad,mode='constant',constant_values=0)
        
    elif algo == pca: 
        scale_list = algo_options.get('scale_list',None)
        ifs_collapse_range = algo_options.get('ifs_collapse_range','all')
        nproc = algo_options.get('nproc','all')
        pca_res = pca(cube, angs, cube_ref, scale_list, ncomp, 
                      svd_mode=svd_mode, scaling=scaling, imlib=imlib,
                      interpolation=interpolation, collapse=collapse,
                      ifs_collapse_range=ifs_collapse_range, nproc=nproc,
                      weights=weights, verbose=False)
    else:
        algo_args = algo_options
        pca_res = algo(cube, angs, **algo_args)
                                  
    indices = disk((posy, posx), radius=aperture_radius*fwhm)
    yy, xx = indices

    if collapse is None:
        values = pca_res[:, yy, xx].ravel()
    else:
        values = pca_res[yy, xx].ravel()

    if debug and collapse is not None:
        return values, pca_res
    else:
        return values


def get_mu_and_sigma(cube, angs, ncomp, annulus_width, aperture_radius, fwhm, 
                     r_guess, theta_guess, cube_ref=None, wedge=None, 
                     svd_mode='lapack', scaling=None, algo=pca_annulus, 
                     delta_rot=1, imlib='vip-fft', interpolation='lanczos4',
                     collapse='median', weights=None, algo_options={}):
    """ Extracts the mean and standard deviation of pixel intensities in an
    annulus of the PCA-ADI image obtained with 'algo', in the part of a defined 
    wedge that is not overlapping with PA_pl+-delta_PA.
    
    Parameters
    ----------
    cube: numpy.array
        The cube of fits images expressed as a numpy.array.
    angs: numpy.array
        The parallactic angle fits image expressed as a numpy.array.
    ncomp: int or None
        The number of principal components for PCA-based algorithms.
    annulus_width: float
        The width in pixels of the annulus on which the PCA is performed.
    aperture_radius: float
        The radius in fwhm of the circular aperture.
    fwhm: float
        Value of the FWHM of the PSF.
    r_guess: float
        The radial position of the center of the circular aperture. This 
        parameter is NOT the radial position of the candidate associated to the 
        Markov chain, but should be the fixed initial guess.
    theta_guess: float
        The angular position of the center of the circular aperture. This 
        parameter is NOT the angular position of the candidate associated to the 
        Markov chain, but should be the fixed initial guess.  
    cube_ref : numpy ndarray, 3d, optional
        Reference library cube. For Reference Star Differential Imaging.
    wedge: tuple, opt
        Range in theta where the mean and standard deviation are computed in an 
        annulus defined in the PCA image. If None, it will be calculated 
        automatically based on initial guess and derotation angles to avoid.
        If some disc signal is present elsewhere in the annulus, it is 
        recommended to provide wedge manually. The provided range should be 
        continuous and >0. E.g. provide (270, 370) to consider a PA range 
        between [-90,+10].
    svd_mode : {'lapack', 'randsvd', 'eigen', 'arpack'}, str optional
        Switch for different ways of computing the SVD and selected PCs.
    scaling : {None, 'temp-mean', 'temp-standard'}
        With None, no scaling is performed on the input data before SVD. With 
        "temp-mean" then temporal px-wise mean subtraction is done and with 
        "temp-standard" temporal mean centering plus scaling to unit variance 
        is done.
    algo: python routine, opt {pca_annulus, pca_annular, pca, custom}
        Routine to be used to model and subtract the stellar PSF. From an input
        cube, derotation angles, and optional arguments, it should return a 
        post-processed frame.
    delta_rot: float, optional
        If algo is set to pca_annular, delta_rot is the angular threshold used
        to select frames in the PCA library (see description of pca_annular).
    imlib : str, optional
        See the documentation of the ``vip_hci.preproc.frame_rotate`` function.
    interpolation : str, optional
        See the documentation of the ``vip_hci.preproc.frame_rotate`` function.
    collapse : {'median', 'mean', 'sum', 'trimmean', None}, str or None, optional
        Sets the way of collapsing the frames for producing a final image. If
        None then the cube of residuals is returned.
    weights: 1d numpy array or list, optional
        Weights to be applied for a weighted mean. Need to be provided if 
        collapse mode is 'wmean'.
    algo_options: dict, opt
        Dictionary with additional parameters related to the algorithm 
        (e.g. tol, min_frames_lib, max_frames_lib). If 'algo' is not a vip
        routine, this dict should contain all necessary arguments apart from
        the cube and derotation angles. Note: arguments such as ncomp, svd_mode,
        scaling, imlib, interpolation or collapse can also be included in this
        dict (the latter are also kept as function arguments for consistency
        with older versions of vip). 
        
    Returns
    -------
    values: numpy.array
        The pixel values in the circular aperture after the PCA process.
        
    """
    centy_fr, centx_fr = frame_center(cube[0])
    halfw = max(aperture_radius*fwhm, annulus_width/2)

    # Checking annulus/aperture sizes. Assuming square frames
    msg = 'The annulus and/or the circular aperture used by the NegFC falls '
    msg += 'outside the FOV. Try increasing the size of your frames or '
    msg += 'decreasing the annulus or aperture size.'
    msg += 'rguess: {:.0f}px; centx_fr: {:.0f}px'.format(r_guess,centx_fr)
    msg += 'halfw: {:.0f}px'.format(halfw)
    if r_guess > centx_fr-halfw:# or r_guess <= halfw:
        raise RuntimeError(msg)

    ncomp = algo_options.get('ncomp',ncomp)                       
    svd_mode = algo_options.get('svd_mode',svd_mode)
    scaling = algo_options.get('scaling',scaling)
    imlib = algo_options.get('imlib',imlib)
    interpolation = algo_options.get('interpolation',interpolation)
    collapse = algo_options.get('collapse',collapse)

    radius_int = max(1,int(np.floor(r_guess-annulus_width/2)))
    radius_int = algo_options.get('radius_int', radius_int)


    # not recommended, except if large-scale residual sky present (NIRC2-L')
    hp_filter = algo_options.get('hp_filter',None)
    hp_kernel = algo_options.get('hp_kernel',None)
    if hp_filter is not None:
        if 'median' in hp_filter:
            cube = cube_filter_highpass(cube, mode=hp_filter, 
                                        median_size=hp_kernel)
        elif "gauss" in hp_filter:
            cube = cube_filter_highpass(cube, mode=hp_filter, 
                                        fwhm_size=hp_kernel)
        else:
            cube = cube_filter_highpass(cube, mode=hp_filter, 
                                        kernel_size=hp_kernel)
    
    if algo == pca_annulus:
        pca_res = pca_annulus(cube, angs, ncomp, annulus_width, r_guess, 
                              cube_ref, svd_mode, scaling, imlib=imlib,
                              interpolation=interpolation, collapse=collapse, 
                              weights=weights)
        pca_res_inv = pca_annulus(cube, -angs, ncomp, annulus_width, r_guess, 
                              cube_ref, svd_mode, scaling, imlib=imlib,
                              interpolation=interpolation, collapse=collapse, 
                              weights=weights)
        
    elif algo == pca_annular:                
        tol = algo_options.get('tol',1e-1)
        min_frames_lib=algo_options.get('min_frames_lib',2)
        max_frames_lib=algo_options.get('max_frames_lib',200)       
        # crop cube to just be larger than annulus => FASTER PCA
        crop_sz = int(2*np.ceil(radius_int+annulus_width+1))
        if not crop_sz %2:
            crop_sz+=1
        if crop_sz < cube.shape[1] and crop_sz < cube.shape[2]:
            pad = int((cube.shape[1]-crop_sz)/2)
            crop_cube = cube_crop_frames(cube, crop_sz, verbose=False)
        else:
            crop_cube = cube

        pca_res_tmp = pca_annular(crop_cube, angs, radius_int=radius_int, 
                                  fwhm=fwhm, asize=annulus_width, 
                                  delta_rot=delta_rot, ncomp=ncomp, 
                                  svd_mode=svd_mode, scaling=scaling, 
                                  imlib=imlib, interpolation=interpolation,
                                  collapse=collapse, tol=tol, 
                                  min_frames_lib=min_frames_lib, 
                                  max_frames_lib=max_frames_lib, 
                                  full_output=False, verbose=False, 
                                  weights=weights)
        pca_res_tinv = pca_annular(crop_cube, -angs, radius_int=radius_int, 
                                  fwhm=fwhm, asize=annulus_width, 
                                  delta_rot=delta_rot, ncomp=ncomp, 
                                  svd_mode=svd_mode, scaling=scaling, 
                                  imlib=imlib, interpolation=interpolation,
                                  collapse=collapse, tol=tol, 
                                  min_frames_lib=min_frames_lib, 
                                  max_frames_lib=max_frames_lib, 
                                  full_output=False, verbose=False, 
                                  weights=weights)
        # pad again now                      
        pca_res = np.pad(pca_res_tmp,pad,mode='constant',constant_values=0)
        pca_res_inv = np.pad(pca_res_tinv,pad,mode='constant',
                              constant_values=0)
        
    elif algo == pca:
        scale_list = algo_options.get('scale_list',None)
        ifs_collapse_range = algo_options.get('ifs_collapse_range','all')
        nproc = algo_options.get('nproc','all')
        
        pca_res = pca(cube, angs, cube_ref, scale_list, ncomp, 
                      svd_mode=svd_mode, scaling=scaling, imlib=imlib,
                      interpolation=interpolation, collapse=collapse,
                      ifs_collapse_range=ifs_collapse_range, nproc=nproc,
                      weights=weights, verbose=False)
        pca_res_inv = pca(cube, -angs, cube_ref, scale_list, ncomp, 
                      svd_mode=svd_mode, scaling=scaling, imlib=imlib,
                      interpolation=interpolation, collapse=collapse,
                      ifs_collapse_range=ifs_collapse_range, nproc=nproc,
                      weights=weights, verbose=False)
        
    else:
        algo_args = algo_options
        pca_res = algo(cube, angs, **algo_args)
        #pca_res_inv = algo(cube, -angs, **algo_args)
        
    if wedge is None:
        delta_theta = np.amax(angs)-np.amin(angs)
        if delta_theta>150:
            delta_theta = 150 # if too much rotation, be less conservative
            
        theta_ini = (theta_guess+delta_theta)%360
        theta_fin = theta_ini+delta_theta
        wedge = (theta_ini,theta_fin)
    elif len(wedge)==2:
        if wedge[0]>wedge[1]:
            msg = '2nd value of wedge smaller than first one => 360 was added'
            print(msg)
            wedge = (wedge[0],wedge[1]+360)
    else:
        raise TypeError("Wedge should have exactly 2 values")
        
    
    indices = get_annular_wedge(pca_res, radius_int, 2*fwhm,
                                wedge=wedge)
    yy, xx = indices
    indices_inv = get_annular_wedge(pca_res_inv, radius_int, 2*fwhm,
                                    wedge=wedge)
    yyi, xxi = indices_inv
    all_res = np.concatenate((pca_res[yy, xx], pca_res_inv[yyi, xxi]))
    mu = np.mean(all_res)
    all_res-=mu
    npx = len(yy)+len(yyi)
    area = np.pi*(fwhm/2)**2
    ddof = min(int(npx*(1.-(1./area)))+1, npx-1) 
    sigma = np.std(all_res, ddof=ddof)   
    
    return mu, sigma


def get_sigma(array, source_xy, fwhm, aperture, full_output=False, 
              exclude_negative_lobes=False, verbose=False):
    """
    Estimate the average standard deviation in apertures similar as the one
    used for NEGFC minimization. This is done through the mean of as many 
    independent apertures of the same size at the same radius.
    
    Parameters
    ----------
    array : numpy ndarray, 2d
        Post-processed frame where we want to measure S/N.
    source_xy : tuple of floats
        X and Y coordinates of the planet or test speckle.
    fwhm : float
        Size in pixels of the FWHM.
    aperture: float
        Size of aperture in FWHM.
    full_output : bool, optional
        If True returns back the S/N value, the y, x input coordinates, noise
        and flux.
    exclude_negative_lobes : bool, opt
        Whether to include the adjacent aperture lobes to the tested location 
        or not. Can be set to True if the image shows significant neg lobes.
    verbose: bool, optional
        Chooses whether to print some output or not.

    Returns
    -------
    [if full_output=True:]
    sourcey : numpy ndarray
        [full_output=True] Input coordinates (``source_xy``) in Y.
    sourcex : numpy ndarray
        [full_output=True] Input coordinates (``source_xy``) in X.
    f_source : float
        [full_output=True] Flux in test elemnt.
    fluxes : numpy ndarray
        [full_output=True] Background apertures fluxes.
    [always:]    
    snr_vale : float
        Value of the S/N for the given test resolution element.
    """
    check_array(array, dim=2, msg='array')
    if not isinstance(source_xy, tuple):
        raise TypeError("`source_xy` must be a tuple of floats")

    sourcex, sourcey = source_xy
    centery, centerx = frame_center(array)
    sep = dist(centery, centerx, float(sourcey), float(sourcex))

    if not sep > (aperture*fwhm)+1:
        raise RuntimeError('`source_xy` is too close to the frame center')

    sens = 'clock'  # counterclock

    angle = np.arcsin(aperture*fwhm/sep)*2
    number_apertures = int(np.floor(2*np.pi/angle))
    yy = np.zeros((number_apertures))
    xx = np.zeros((number_apertures))
    cosangle = np.cos(angle)
    sinangle = np.sin(angle)
    xx[0] = sourcex - centerx
    yy[0] = sourcey - centery
    for i in range(number_apertures-1):
        if sens == 'clock':
            xx[i+1] = cosangle*xx[i] + sinangle*yy[i] 
            yy[i+1] = cosangle*yy[i] - sinangle*xx[i] 
        elif sens == 'counterclock':
            xx[i+1] = cosangle*xx[i] - sinangle*yy[i] 
            yy[i+1] = cosangle*yy[i] + sinangle*xx[i]

    xx += centerx
    yy += centery
    rad = aperture*fwhm
    if exclude_negative_lobes:
        xx = np.concatenate(([xx[0]], xx[2:-1]))
        yy = np.concatenate(([yy[0]], yy[2:-1]))

    sigmas = np.zeros_like(yy)
    for i in range(1,len(sigmas)):
        sigmas[i] = np.std(get_circle(array, rad, cy=yy[i], cx=xx[i], 
                                      mode="val"), ddof=1)

    sigmas = sigmas[1:] # do not consider companion itself
    sigma = np.mean(sigmas)

    if full_output:
        return sigmas, sigma
    else:
        return sigma