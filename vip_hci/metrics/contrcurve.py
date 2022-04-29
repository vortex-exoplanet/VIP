#! /usr/bin/env python

"""
Module with contrast curve generation function.
"""

__author__ = 'C. Gomez, C.H. Dahlqvist, O. Absil @ ULg'
__all__ = ['contrast_curve',
           'noise_per_annulus',
           'throughput',
           'aperture_flux',
           'completeness_curve',
           'completeness_map']

import numpy as np
import pandas as pd
import photutils
import inspect
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy import stats
from scipy.signal import savgol_filter
from skimage.draw import disk
from matplotlib import pyplot as plt
from ..fm import (cube_inject_companions, frame_inject_companion,
                  normalize_psf)
from ..config import time_ini, timing
from ..config.utils_conf import pool_map, iterable,sep
from ..var import get_annulus_segments,frame_center, dist
from ..preproc import cube_crop_frames
from ..metrics.snr_source import snrmap,_snr_approx, snr
from astropy.convolution import convolve, Tophat2DKernel
import math

def contrast_curve(cube, angle_list, psf_template, fwhm, pxscale, starphot,
                   algo, sigma=5, nbranch=1, theta=0, inner_rad=1,
                   wedge=(0,360), fc_snr=100, student=True, transmission=None,
                   smooth=True, interp_order=2, plot=True, dpi=100, debug=False, 
                   verbose=True, full_output=False, save_plot=None, 
                   object_name=None, frame_size=None, fix_y_lim=(), 
                   figsize=(8, 4), **algo_dict):
    """ Computes the contrast curve at a given SIGMA (``sigma``) level for an
    ADI cube or ADI+IFS cube. The contrast is calculated as
    sigma*noise/throughput. This implementation takes into account the small
    sample statistics correction proposed in Mawet et al. 2014.

    Parameters
    ----------
    cube : numpy ndarray
        The input cube, 3d (ADI data) or 4d array (IFS data), without fake
        companions.
    angle_list : numpy ndarray
        Vector with the parallactic angles.
    psf_template : numpy ndarray
        Frame with the psf template for the fake companion(s).
        PSF must be centered in array. Normalization is done internally.
    fwhm: int or float or 1d array, optional
        The the Full Width Half Maximum in pixels. It can handle a different
        FWHM value for different wavelengths (IFS data).
    pxscale : float
        Plate scale or pixel scale of the instrument.
    starphot : int or float or 1d array
        If int or float it corresponds to the aperture photometry of the
        non-coronagraphic PSF which we use to scale the contrast. If a vector
        is given it must contain the photometry correction for each frame.
    algo : callable or function
        The post-processing algorithm, e.g. vip_hci.pca.pca.
    sigma : int
        Sigma level for contrast calculation. Note this is a "Gaussian sigma"
        regardless of whether Student t correction is performed (set by the 
        'student' parameter). E.g. setting sigma to 5 will yield the contrast 
        curve corresponding to a false alarm probability of 3e-7.
    nbranch : int, optional
        Number of branches on which to inject fakes companions. Each branch
        is tested individually.
    theta : float, optional
        Angle in degrees for rotating the position of the first branch that by
        default is located at zero degrees. Theta counts counterclockwise from
        the positive x axis. When working on a wedge, make sure that theta is
        located inside of it.
    inner_rad : int, optional
        Innermost radial distance to be considered in terms of FWHM.
    wedge : tuple of floats, optional
        Initial and Final angles for using a wedge. For example (-90,90) only
        considers the right side of an image.
    fc_snr: float optional
        Signal to noise ratio of injected fake companions (w.r.t a Gaussian 
        distribution).
    student : bool, optional
        If True uses Student t correction to inject fake companion.
    transmission : tuple of 2 1d arrays, optional
        If not None, then the tuple contains a vector with the factors to be
        applied to the sensitivity and a vector of the radial distances [px]
        where it is sampled (in this order).
    smooth : bool, optional
        If True the radial noise curve is smoothed with a Savitzky-Golay filter
        of order 2.
    interp_order : int or None, optional
        If True the throughput vector is interpolated with a spline of order
        ``interp_order``. Takes values from 1 to 5. If None, then the
        throughput is not interpolated.
    plot : bool, optional
        Whether to plot the final contrast curve or not. True by default.
    dpi : int optional
        Dots per inch for the plots. 100 by default. 300 for printing quality.
    imlib : {'opencv', 'ndimage-fourier', 'ndimage-interp', 'vip-fft'}, str opt
        Library or method used for image operations (rotations). Opencv is the
        default for being the fastest. See description of
        `vip_hci.preproc.frame_rotate`.
    interpolation: str, opt
        See description of ``vip_hci.preproc.frame_rotate`` function
    debug : bool, optional
        Whether to print and plot additional info such as the noise, throughput,
        the contrast curve with different X axis and the delta magnitude instead
        of contrast.
    verbose : {True, False, 0, 1, 2}, optional
        If True or 1 the function prints to stdout intermediate info and timing,
        if set to 2 more output will be shown. 
    full_output : bool, optional
        If True returns intermediate arrays.
    save_plot: string
        If provided, the contrast curve will be saved to this path.
    object_name: string
        Target name, used in the plot title.
    frame_size: int
        Frame size used for generating the contrast curve, used in the plot
        title.
    fix_y_lim: tuple
        If provided, the y axis limits will be fixed, for easier comparison
        between plots.
    **algo_dict
        Any other valid parameter of the post-processing algorithms can be
        passed here, including e.g. imlib and interpolation.

    Returns
    -------
    datafr : pandas dataframe
        Dataframe containing the sensitivity (Gaussian and Student corrected if
        Student parameter is True), the interpolated throughput, the distance in
        pixels, the noise and the sigma corrected (if Student is True).

    If full_output is True then the function returns: 
        datafr, cube_fc_all, frame_fc_all, frame_nofc and fc_map_all.

    frame_fc_all : numpy ndarray
        3d array with the 3 frames of the 3 (patterns) processed cubes with
        companions.
    frame_nofc : numpy ndarray
        2d array, PCA processed frame without companions.
    fc_map_all : numpy ndarray
        3d array with 3 frames containing the position of the companions in the
        3 patterns.
    """
    if cube.ndim != 3 and cube.ndim != 4:
        raise TypeError('The input array is not a 3d or 4d cube')
    if cube.ndim == 3 and (cube.shape[0] != angle_list.shape[0]):
        raise TypeError('Input parallactic angles vector has wrong length')
    if cube.ndim == 4 and (cube.shape[1] != angle_list.shape[0]):
        raise TypeError('Input parallactic angles vector has wrong length')
    if cube.ndim == 3 and psf_template.ndim != 2:
        raise TypeError('Template PSF is not a frame (for ADI case)')
    if cube.ndim == 4 and psf_template.ndim != 3:
        raise TypeError('Template PSF is not a cube (for ADI+IFS case)')
    if transmission is not None:
        if not isinstance(transmission, tuple) or not len(transmission) == 2:
            raise TypeError('transmission must be a tuple with 2 1d vectors')

    if isinstance(fwhm, (np.ndarray,list)):
        fwhm_med = np.median(fwhm)
    else:
        fwhm_med = fwhm

    if verbose:
        start_time = time_ini()
        if isinstance(starphot, float) or isinstance(starphot, int):
            msg0 = 'ALGO : {}, FWHM = {}, # BRANCHES = {}, SIGMA = {},'
            msg0 += ' STARPHOT = {}'
            print(msg0.format(algo.__name__, fwhm_med, nbranch, sigma, starphot))
        else:
            msg0 = 'ALGO : {}, FWHM = {}, # BRANCHES = {}, SIGMA = {}'
            print(msg0.format(algo.__name__, fwhm_med, nbranch, sigma))
        print(sep)

    # throughput
    verbose_thru = False
    if verbose == 2:
        verbose_thru = True
    res_throug = throughput(cube, angle_list, psf_template, fwhm, pxscale,
                            nbranch=nbranch, theta=theta, inner_rad=inner_rad,
                            wedge=wedge, fc_snr=fc_snr, full_output=True,
                            algo=algo, verbose=verbose_thru, **algo_dict)
    vector_radd = res_throug[3]
    if res_throug[0].shape[0] > 1:
        thruput_mean = np.nanmean(res_throug[0], axis=0)
    else:
        thruput_mean = res_throug[0][0]
    frame_fc_all = res_throug[4]
    frame_nofc = res_throug[5]
    fc_map_all = res_throug[6]

    if verbose:
        print('Finished the throughput calculation')
        timing(start_time)

    if interp_order is not None:
        # noise measured in the empty frame with better sampling, every px
        # starting from 1*FWHM
        noise_samp, res_lev_samp, rad_samp = noise_per_annulus(frame_nofc, 
                                                               separation=1,
                                                               fwhm=fwhm_med, 
                                                               init_rad=fwhm_med,
                                                               wedge=wedge)
        radmin = vector_radd.astype(int).min()
        cutin1 = np.where(rad_samp.astype(int) == radmin)[0][0]
        noise_samp = noise_samp[cutin1:]
        res_lev_samp = res_lev_samp[cutin1:]
        rad_samp = rad_samp[cutin1:]
        radmax = vector_radd.astype(int).max()
        cutin2 = np.where(rad_samp.astype(int) == radmax)[0][0]
        noise_samp = noise_samp[:cutin2 + 1]
        res_lev_samp = res_lev_samp[:cutin2 + 1]
        rad_samp = rad_samp[:cutin2 + 1]

        # interpolating the throughput vector, spline order 2
        f = InterpolatedUnivariateSpline(vector_radd, thruput_mean,
                                         k=interp_order)
        thruput_interp = f(rad_samp)

        # interpolating the transmission vector, spline order 1
        if transmission is not None:
            trans = transmission[0]
            radvec_trans = transmission[1]
            f2 = InterpolatedUnivariateSpline(radvec_trans, trans, k=1)
            trans_interp = f2(rad_samp)
            thruput_interp *= trans_interp
    else:
        rad_samp = vector_radd
        noise_samp = res_throug[1]
        res_lev_samp = res_throug[2]
        thruput_interp = thruput_mean
        if transmission is not None:
            if not transmission[0].shape == thruput_interp.shape[0]:
                msg = 'Transmiss. and throughput vectors have different length'
                raise ValueError(msg)
            thruput_interp *= transmission[0]

    rad_samp_arcsec = rad_samp * pxscale

    # take abs value of the mean residual fluxes otherwise the more 
    # oversubtraction (negative res_lev_samp), the better the contrast!!
    res_lev_samp = np.abs(res_lev_samp) 

    if smooth:
        # smoothing the noise vector using a Savitzky-Golay filter
        win = min(noise_samp.shape[0]-2, int(2*fwhm_med))
        if win % 2 == 0:
            win += 1
        noise_samp_sm = savgol_filter(noise_samp, polyorder=2, mode='nearest',
                                      window_length=win)
        res_lev_samp_sm = savgol_filter(res_lev_samp, polyorder=2, 
                                        mode='nearest', window_length=win)
    else:
        noise_samp_sm = noise_samp
        res_lev_samp_sm = res_lev_samp

    # calculating the contrast
    if isinstance(starphot, float) or isinstance(starphot, int):
        cont_curve_samp = ((sigma * noise_samp_sm + res_lev_samp_sm
                            )/ thruput_interp) / starphot
    else:
        cont_curve_samp = ((sigma * noise_samp_sm + res_lev_samp_sm
                            ) / thruput_interp) / np.median(starphot)
    cont_curve_samp[np.where(cont_curve_samp < 0)] = 1
    cont_curve_samp[np.where(cont_curve_samp > 1)] = 1

    # calculating the Student corrected contrast
    if student:
        n_res_els = np.floor(rad_samp/fwhm_med*2*np.pi)
        ss_corr = np.sqrt(1 + 1/n_res_els)
        sigma_corr = stats.t.ppf(stats.norm.cdf(sigma), n_res_els-1)*ss_corr
        if isinstance(starphot, float) or isinstance(starphot, int):
            cont_curve_samp_corr = ((sigma_corr*noise_samp_sm + res_lev_samp_sm
                                     )/thruput_interp)/starphot
        else:
            cont_curve_samp_corr = ((sigma_corr*noise_samp_sm + res_lev_samp_sm
                                     )/thruput_interp) / np.median(starphot)
        cont_curve_samp_corr[np.where(cont_curve_samp_corr < 0)] = 1
        cont_curve_samp_corr[np.where(cont_curve_samp_corr > 1)] = 1

    if debug:
        plt.rc("savefig", dpi=dpi)
        plt.figure(figsize=figsize, dpi=dpi)
        # throughput
        plt.plot(vector_radd * pxscale, thruput_mean, '.', label='computed',
                 alpha=0.6)
        plt.plot(rad_samp_arcsec, thruput_interp, ',-', label='interpolated',
                 lw=2, alpha=0.5)
        plt.grid('on', which='both', alpha=0.2, linestyle='solid')
        plt.xlabel('Angular separation [arcsec]')
        plt.ylabel('Throughput')
        plt.legend(loc='best')
        plt.xlim(0, np.max(rad_samp*pxscale))
        # noise
        plt.figure(figsize=figsize, dpi=dpi)
        plt.plot(rad_samp_arcsec, noise_samp, '.', label='computed', alpha=0.6)
        if smooth:
            plt.plot(rad_samp_arcsec, noise_samp_sm, ',-', 
                     label='noise smoothed', lw=2, alpha=0.5)
        plt.grid('on', alpha=0.2, linestyle='solid')
        plt.xlabel('Angular separation [arcsec]')
        plt.ylabel('Noise')
        plt.legend(loc='best')
        plt.xlim(0, np.max(rad_samp_arcsec))
        # mean residual level
        plt.figure(figsize=figsize, dpi=dpi)
        plt.plot(rad_samp_arcsec, res_lev_samp, '.', 
                 label='computed residual level', alpha=0.6)
        if smooth:
            plt.plot(rad_samp_arcsec, res_lev_samp_sm, ',-', 
                     label='smoothed residual level', lw=2, alpha=0.5)
        plt.grid('on', alpha=0.2, linestyle='solid')
        plt.xlabel('Angular separation [arcsec]')
        plt.ylabel('Mean residual level')
        plt.legend(loc='best')
        plt.xlim(0, np.max(rad_samp_arcsec))

    # plotting
    if plot or debug:
        if student:
            label = ['Sensitivity (Gaussian)',
                     'Sensitivity (Student-t correction)']
        else:
            label = ['Sensitivity (Gaussian)']

        plt.rc("savefig", dpi=dpi)
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax1 = fig.add_subplot(111)
        con1, = ax1.plot(rad_samp_arcsec, cont_curve_samp, '-',
                         alpha=0.2, lw=2, color='green')
        con2, = ax1.plot(rad_samp_arcsec, cont_curve_samp, '.',
                         alpha=0.2, color='green')
        if student:
            con3, = ax1.plot(rad_samp_arcsec, cont_curve_samp_corr, '-',
                             alpha=0.4, lw=2, color='blue')
            con4, = ax1.plot(rad_samp_arcsec, cont_curve_samp_corr, '.',
                             alpha=0.4, color='blue')
            lege = [(con1, con2), (con3, con4)]
        else:
            lege = [(con1, con2)]
        plt.legend(lege, label, fancybox=True, fontsize='medium')
        plt.xlabel('Angular separation [arcsec]')
        plt.ylabel(str(sigma)+' sigma contrast')
        plt.grid('on', which='both', alpha=0.2, linestyle='solid')
        ax1.set_yscale('log')
        ax1.set_xlim(0, np.max(rad_samp_arcsec))

        # Give a title to the contrast curve plot
        if object_name is not None and frame_size is not None:
            # Retrieve ncomp and pca_type info to use in title
            ncomp = algo_dict['ncomp']
            if algo_dict['cube_ref'] is None:
                pca_type = 'ADI'
            else:
                pca_type = 'RDI'
            title = "{} {} {}pc {} + {}".format(pca_type, object_name, ncomp,
                                                frame_size, inner_rad)
            plt.title(title, fontsize=14)

        # Option to fix the y-limit
        if len(fix_y_lim) == 2:
            min_y_lim = min(fix_y_lim[0], fix_y_lim[1])
            max_y_lim = max(fix_y_lim[0], fix_y_lim[1])
            ax1.set_ylim(min_y_lim, max_y_lim)

        # Optionally, save the figure to a path
        if save_plot is not None:
            fig.savefig(save_plot, dpi=dpi)
            
        if debug:
            fig2 = plt.figure(figsize=figsize, dpi=dpi)
            ax3 = fig2.add_subplot(111)
            cc_mags = -2.5*np.log10(cont_curve_samp)
            con4, = ax3.plot(rad_samp_arcsec, cc_mags, '-',
                             alpha=0.2, lw=2, color='green')
            con5, = ax3.plot(rad_samp_arcsec, cc_mags, '.', alpha=0.2,
                             color='green')
            if student:
                cc_mags_corr = -2.5*np.log10(cont_curve_samp_corr)
                con6, = ax3.plot(rad_samp_arcsec, cc_mags_corr, '-',
                                 alpha=0.4, lw=2, color='blue')
                con7, = ax3.plot(rad_samp_arcsec, cc_mags_corr, '.',
                                 alpha=0.4, color='blue')
                lege = [(con4, con5), (con6, con7)]
            else:
                lege = [(con4, con5)]
            plt.legend(lege, label, fancybox=True, fontsize='medium')
            plt.xlabel('Angular separation [arcsec]')
            plt.ylabel('Delta magnitude')
            plt.gca().invert_yaxis()
            plt.grid('on', which='both', alpha=0.2, linestyle='solid')
            ax3.set_xlim(0, np.max(rad_samp*pxscale))
            ax4 = ax3.twiny()
            ax4.set_xlabel('Distance [pixels]')
            ax4.plot(rad_samp, cc_mags, '', alpha=0.)
            ax4.set_xlim(0, np.max(rad_samp))

    if student:
        datafr = pd.DataFrame({'sensitivity_gaussian': cont_curve_samp,
                               'sensitivity_student': cont_curve_samp_corr,
                               'throughput': thruput_interp,
                               'distance': rad_samp,
                               'distance_arcsec': rad_samp_arcsec,
                               'noise': noise_samp_sm,
                               'residual_level': res_lev_samp_sm,
                               'sigma corr': sigma_corr})
    else:
        datafr = pd.DataFrame({'sensitivity_gaussian': cont_curve_samp,
                               'throughput': thruput_interp,
                               'distance': rad_samp,
                               'distance_arcsec': rad_samp_arcsec,
                               'noise': noise_samp_sm,
                               'residual_level': res_lev_samp_sm})

    if full_output:
        return datafr, frame_fc_all, frame_nofc, fc_map_all
    else: 
        return datafr


def throughput(cube, angle_list, psf_template, fwhm, pxscale, algo, nbranch=1,
               theta=0, inner_rad=1, fc_rad_sep=3, wedge=(0,360), fc_snr=100,
               full_output=False, verbose=True, **algo_dict):
    """ Measures the throughput for chosen algorithm and input dataset (ADI or
    ADI+mSDI). The final throughput is the average of the same procedure
    measured in ``nbranch`` azimutally equidistant branches.

    Parameters
    ---------_
    cube : numpy ndarray
        The input cube, 3d (ADI data) or 4d array (IFS data), without fake
        companions.
    angle_list : numpy ndarray
        Vector with the parallactic angles.
    psf_template : numpy ndarray
        Frame with the psf template for the fake companion(s).
        PSF must be centered in array. Normalization is done internally.
    fwhm: int or float or 1d array, optional
        The the Full Width Half Maximum in pixels. It can handle a different
        FWHM value for different wavelengths (IFS data).
    pxscale : float
        Plate scale in arcsec/px.
    algo : callable or function
        The post-processing algorithm, e.g. vip_hci.pca.pca. Third party Python
        algorithms can be plugged here. They must have the parameters: 'cube',
        'angle_list' and 'verbose'. Optionally a wrapper function can be used.
    nbranch : int optional
        Number of branches on which to inject fakes companions. Each branch
        is tested individually.
    theta : float, optional
        Angle in degrees for rotating the position of the first branch that by
        default is located at zero degrees. Theta counts counterclockwise from
        the positive x axis.
    inner_rad : int, optional
        Innermost radial distance to be considered in terms of FWHM.
    fc_rad_sep : int optional
        Radial separation between the injected companions (in each of the
        patterns) in FWHM. Must be large enough to avoid overlapping. With the
        maximum possible value, a single fake companion will be injected per
        cube and algorithm post-processing (which greatly affects computation
        time).
    wedge : tuple of floats, optional
        Initial and Final angles for using a wedge. For example (-90,90) only
        considers the right side of an image.
    fc_snr: float optional
        Signal to noise ratio of injected fake companions (w.r.t a Gaussian 
        distribution).
    full_output : bool, optional
        If True returns intermediate arrays.
    verbose : bool, optional
        If True prints out timing and information.
    **algo_dict
        Parameters of the post-processing algorithms must be passed here,
        including imlib and interpolation.

    Returns
    -------
    thruput_arr : numpy ndarray
        2d array whose rows are the annulus-wise throughput values for each
        branch.
    vector_radd : numpy ndarray
        1d array with the distances in FWHM (the positions of the annuli).

    If full_output is True then the function returns: thruput_arr, noise,
    vector_radd, cube_fc_all, frame_fc_all, frame_nofc and fc_map_all.

    noise : numpy ndarray
        1d array with the noise per annulus.
    frame_fc_all : numpy ndarray
        3d array with the 3 frames of the 3 (patterns) processed cubes with
        companions.
    frame_nofc : numpy ndarray
        2d array, PCA processed frame without companions.
    fc_map_all : numpy ndarray
        3d array with 3 frames containing the position of the companions in the
        3 patterns.

    """
    array = cube
    parangles = angle_list
    imlib = algo_dict.get('imlib', 'vip-fft')
    interpolation = algo_dict.get('interpolation', 'lanczos4')

    if array.ndim != 3 and array.ndim != 4:
        raise TypeError('The input array is not a 3d or 4d cube')
    else:
        if array.ndim == 3:
            if array.shape[0] != parangles.shape[0]:
                msg = 'Input parallactic angles vector has wrong length'
                raise TypeError(msg)
            if psf_template.ndim != 2:
                raise TypeError('Template PSF is not a frame or 2d array')
            maxfcsep = int((array.shape[1]/2.)/fwhm)-1
            if fc_rad_sep < 3 or fc_rad_sep > maxfcsep:
                msg = 'Too large separation between companions in the radial '
                msg += 'patterns. Should lie between 3 and {}'
                raise ValueError(msg.format(maxfcsep))

        elif array.ndim == 4:
            if array.shape[1] != parangles.shape[0]:
                msg = 'Input vector or parallactic angles has wrong length'
                raise TypeError(msg)
            if psf_template.ndim != 3:
                raise TypeError('Template PSF is not a frame, 3d array')
            if 'scale_list' not in algo_dict:
                raise ValueError('Vector of wavelength not found')
            else:
                if algo_dict['scale_list'].shape[0] != array.shape[0]:
                    raise TypeError('Input wavelength vector has wrong length')
                if isinstance(fwhm, float) or isinstance(fwhm, int):
                    maxfcsep = int((array.shape[2] / 2.) / fwhm) - 1
                else:
                    maxfcsep = int((array.shape[2] / 2.) / np.amin(fwhm)) - 1
                if fc_rad_sep < 3 or fc_rad_sep > maxfcsep:
                    msg = 'Too large separation between companions in the '
                    msg += 'radial patterns. Should lie between 3 and {}'
                    raise ValueError(msg.format(maxfcsep))

        if psf_template.shape[1] % 2 == 0:
            raise ValueError("Only odd-sized PSF is accepted")
        if not hasattr(algo, '__call__'):
            raise TypeError('Parameter `algo` must be a callable function')
        if not isinstance(inner_rad, int):
            raise TypeError('inner_rad must be an integer')
        angular_range = wedge[1] - wedge[0]
        if nbranch > 1 and angular_range < 360:
            msg = 'Only a single branch is allowed when working on a wedge'
            raise RuntimeError(msg)

    if isinstance(fwhm, (np.ndarray,list)):
        fwhm_med = np.median(fwhm)
    else:
        fwhm_med = fwhm

    if verbose:
        start_time = time_ini()
    #***************************************************************************
    # Compute noise in concentric annuli on the "empty frame"
    argl = inspect.getargspec(algo).args
    if 'cube' in argl and 'angle_list' in argl and 'verbose' in argl:
        if 'fwhm' in argl:
            frame_nofc = algo(cube=array, angle_list=parangles, fwhm=fwhm_med,
                              verbose=False, **algo_dict)
            if algo_dict.pop('scaling',None):
                new_algo_dict = algo_dict.copy()
                new_algo_dict['scaling'] = None
                frame_nofc_noscal = algo(cube=array, angle_list=parangles, 
                                         fwhm=fwhm_med, verbose=False, 
                                         **new_algo_dict)
            else:
                frame_nofc_noscal = frame_nofc
        else:
            frame_nofc = algo(array, angle_list=parangles, verbose=False,
                              **algo_dict)
            if algo_dict.pop('scaling',None):
                new_algo_dict = algo_dict.copy()
                new_algo_dict['scaling'] = None
                frame_nofc_noscal = algo(cube=array, angle_list=parangles,
                                         verbose=False, **new_algo_dict)
            else:
                frame_nofc_noscal = frame_nofc
                
    if verbose:
        msg1 = 'Cube without fake companions processed with {}'
        print(msg1.format(algo.__name__))
        timing(start_time)

    noise, res_level, vector_radd = noise_per_annulus(frame_nofc, 
                                                      separation=fwhm_med,
                                                      fwhm=fwhm_med, 
                                                      wedge=wedge)
    noise_noscal, _, _ = noise_per_annulus(frame_nofc_noscal, 
                                           separation=fwhm_med, fwhm=fwhm_med, 
                                           wedge=wedge)                                       
    vector_radd = vector_radd[inner_rad-1:]
    noise = noise[inner_rad-1:]
    res_level = res_level[inner_rad-1:]
    noise_noscal = noise_noscal[inner_rad-1:]
    if verbose:
        print('Measured annulus-wise noise in resulting frame')
        timing(start_time)

    # We crop the PSF and check if PSF has been normalized (so that flux in
    # 1*FWHM aperture = 1) and fix if needed
    new_psf_size = int(round(3 * fwhm_med))
    if new_psf_size % 2 == 0:
        new_psf_size += 1

    if cube.ndim == 3:
        n, y, x = array.shape
        psf_template = normalize_psf(psf_template, fwhm=fwhm, verbose=verbose,
                                     size=min(new_psf_size,
                                              psf_template.shape[1]))

        # Initialize the fake companions
        angle_branch = angular_range / nbranch
        thruput_arr = np.zeros((nbranch, noise.shape[0]))
        fc_map_all = np.zeros((nbranch * fc_rad_sep, y, x))
        frame_fc_all = np.zeros((nbranch * fc_rad_sep, y, x))
        cy, cx = frame_center(array[0])

        # each branch is computed separately
        for br in range(nbranch):
            # each pattern is computed separately. For each one the companions
            # are separated by "fc_rad_sep * fwhm", interleaving the injections
            for irad in range(fc_rad_sep):
                radvec = vector_radd[irad::fc_rad_sep]
                cube_fc = array.copy()
                # filling map with small numbers
                fc_map = np.ones_like(array[0]) * 1e-6
                fcy = []
                fcx = []
                for i in range(radvec.shape[0]):
                    flux = fc_snr * noise_noscal[irad + i * fc_rad_sep]
                    cube_fc = cube_inject_companions(cube_fc, psf_template,
                                                     parangles, flux, pxscale,
                                                     rad_dists=[radvec[i]],
                                                     theta=br*angle_branch +
                                                           theta,
                                                     imlib=imlib, verbose=False,
                                                     interpolation=
                                                        interpolation)
                    y = cy + radvec[i] * np.sin(np.deg2rad(br * angle_branch +
                                                           theta))
                    x = cx + radvec[i] * np.cos(np.deg2rad(br * angle_branch +
                                                           theta))
                    fc_map = frame_inject_companion(fc_map, psf_template, y, x,
                                                    flux, imlib, interpolation)
                    fcy.append(y)
                    fcx.append(x)

                if verbose:
                    msg2 = 'Fake companions injected in branch {} '
                    msg2 += '(pattern {}/{})'
                    print(msg2.format(br+1, irad+1, fc_rad_sep))
                    timing(start_time)

                #***************************************************************
                arg = inspect.getargspec(algo).args
                if 'cube' in arg and 'angle_list' in arg and 'verbose' in arg:
                    if 'fwhm' in arg:
                        frame_fc = algo(cube=cube_fc, angle_list=parangles,
                                        fwhm=fwhm_med, verbose=False, 
                                        **algo_dict)
                    else:
                        frame_fc = algo(cube=cube_fc, angle_list=parangles,
                                        verbose=False, **algo_dict)
                else:
                    msg = 'Input algorithm must have at least 3 parameters: '
                    msg += 'cube, angle_list and verbose'
                    raise ValueError(msg)

                if verbose:
                    msg3 = 'Cube with fake companions processed with {}'
                    msg3 += '\nMeasuring its annulus-wise throughput'
                    print(msg3.format(algo.__name__))
                    timing(start_time)

                #**************************************************************
                injected_flux = aperture_flux(fc_map, fcy, fcx, fwhm_med)
                recovered_flux = aperture_flux((frame_fc - frame_nofc), fcy,
                                               fcx, fwhm_med)
                thruput = recovered_flux / injected_flux
                thruput[np.where(thruput < 0)] = 0

                thruput_arr[br, irad::fc_rad_sep] = thruput
                fc_map_all[br*fc_rad_sep+irad, :, :] = fc_map
                frame_fc_all[br*fc_rad_sep+irad, :, :] = frame_fc

    elif cube.ndim == 4:
        w, n, y, x = array.shape
        if isinstance(fwhm, (int, float)):
            fwhm = [fwhm] * w
        psf_template = normalize_psf(psf_template, fwhm=fwhm, verbose=verbose,
                                     size=min(new_psf_size,
                                              psf_template.shape[1]))

        # Initialize the fake companions
        angle_branch = angular_range / nbranch
        thruput_arr = np.zeros((nbranch, noise.shape[0]))
        fc_map_all = np.zeros((nbranch * fc_rad_sep, w, y, x))
        frame_fc_all = np.zeros((nbranch * fc_rad_sep, y, x))
        cy, cx = frame_center(array[0, 0])

        # each branch is computed separately
        for br in range(nbranch):
            # each pattern is computed separately. For each pattern the
            # companions are separated by "fc_rad_sep * fwhm"
            # radius = vector_radd[irad::fc_rad_sep]
            for irad in range(fc_rad_sep):
                radvec = vector_radd[irad::fc_rad_sep]
                thetavec = range(int(theta), int(theta) + 360,
                                 360 // len(radvec))
                cube_fc = array.copy()
                # filling map with small numbers
                fc_map = np.ones_like(array[:, 0]) * 1e-6
                fcy = []
                fcx = []
                for i in range(radvec.shape[0]):
                    flux = fc_snr * noise_noscal[irad + i * fc_rad_sep]
                    cube_fc = cube_inject_companions(cube_fc, psf_template,
                                                     parangles, flux, pxscale,
                                                     rad_dists=[radvec[i]],
                                                     theta=thetavec[i],
                                                     verbose=False,
                                                     imlib=imlib, 
                                                     interpolation=
                                                        interpolation)
                    y = cy + radvec[i] * np.sin(np.deg2rad(br * angle_branch +
                                                           thetavec[i]))
                    x = cx + radvec[i] * np.cos(np.deg2rad(br * angle_branch +
                                                           thetavec[i]))
                    fc_map = frame_inject_companion(fc_map, psf_template, y, x,
                                                    flux)
                    fcy.append(y)
                    fcx.append(x)

                if verbose:
                    msg2 = 'Fake companions injected in branch {} '
                    msg2 += '(pattern {}/{})'
                    print(msg2.format(br + 1, irad + 1, fc_rad_sep))
                    timing(start_time)

                # **************************************************************
                arg = inspect.getargspec(algo).args
                if 'cube' in arg and 'angle_list' in arg and 'verbose' in arg:
                    if 'fwhm' in arg:
                        frame_fc = algo(cube=cube_fc, angle_list=parangles,
                                        fwhm=fwhm_med, verbose=False, 
                                        **algo_dict)
                    else:
                        frame_fc = algo(cube=cube_fc, angle_list=parangles,
                                        verbose=False, **algo_dict)

                if verbose:
                    msg3 = 'Cube with fake companions processed with {}'
                    msg3 += '\nMeasuring its annulus-wise throughput'
                    print(msg3.format(algo.__name__))
                    timing(start_time)

                # *************************************************************
                injected_flux = [aperture_flux(fc_map[i], fcy, fcx, fwhm[i])
                                 for i in range(array.shape[0])]
                injected_flux = np.mean(injected_flux, axis=0)
                recovered_flux = aperture_flux((frame_fc - frame_nofc), fcy,
                                               fcx, fwhm_med)
                thruput = recovered_flux / injected_flux
                thruput[np.where(thruput < 0)] = 0

                thruput_arr[br, irad::fc_rad_sep] = thruput
                fc_map_all[br * fc_rad_sep + irad, :, :] = fc_map
                frame_fc_all[br * fc_rad_sep + irad, :, :] = frame_fc

    if verbose:
        msg = 'Finished measuring the throughput in {} branches'
        print(msg.format(nbranch))
        timing(start_time)

    if full_output:
        return (thruput_arr, noise, res_level, vector_radd, frame_fc_all, 
                frame_nofc, fc_map_all)
    else:
        return thruput_arr, vector_radd


def noise_per_annulus(array, separation, fwhm, init_rad=None, wedge=(0, 360),
                      verbose=False, debug=False):
    """ Measures the noise and mean residual level as the standard deviation 
    and mean, respectively, of apertures defined in each annulus with a given 
    separation.
    
    The annuli start at init_rad (== fwhm by default) and stop 2*separation
    before the edge of the frame.

    Parameters
    ----------
    array : numpy ndarray
        Input frame.
    separation : float
        Separation in pixels of the centers of the annuli measured from the
        center of the frame.
    fwhm : float
        FWHM in pixels.
    init_rad : float
        Initial radial distance to be used. If None then the init_rad = FWHM.
    wedge : tuple of floats, optional
        Initial and Final angles for using a wedge. For example (-90,90) only
        considers the right side of an image. Be careful when using small
        wedges, this leads to computing a standard deviation of very small
        samples (<10 values).
    verbose : bool, optional
        If True prints information.
    debug : bool, optional
        If True plots the positioning of the apertures.

    Returns
    -------
    noise : numpy ndarray
        Vector with the noise value per annulus.
    res_level : numpy ndarray
        Vector with the mean residual level per annulus.        
    vector_radd : numpy ndarray
        Vector with the radial distances values.

    """
    def find_coords(rad, sep, init_angle, fin_angle):
        angular_range = fin_angle-init_angle
        npoints = (np.deg2rad(angular_range)*rad)/sep   #(2*np.pi*rad)/sep
        ang_step = angular_range/npoints   #360/npoints
        x = []
        y = []
        for i in range(int(npoints)):
            newx = rad * np.cos(np.deg2rad(ang_step * i + init_angle))
            newy = rad * np.sin(np.deg2rad(ang_step * i + init_angle))
            x.append(newx)
            y.append(newy)
        return np.array(y), np.array(x)
    ###

    if array.ndim != 2:
        raise TypeError('Input array is not a frame or 2d array')
    if not isinstance(wedge, tuple):
        raise TypeError('Wedge must be a tuple with the initial and final '
                        'angles')

    if init_rad is None:
        init_rad = fwhm

    init_angle, fin_angle = wedge
    centery, centerx = frame_center(array)
    n_annuli = int(np.floor((centery - init_rad)/separation)) - 1
    noise = []
    res_level = []
    vector_radd = []
    if verbose:
        print('{} annuli'.format(n_annuli))

    if debug:
        _, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(array, origin='lower', interpolation='nearest',
                  alpha=0.5, cmap='gray')

    for i in range(n_annuli):
        y = centery + init_rad + separation * i
        rad = dist(centery, centerx, y, centerx)
        yy, xx = find_coords(rad, fwhm, init_angle, fin_angle)
        yy += centery
        xx += centerx

        apertures = photutils.CircularAperture(np.array((xx, yy)).T, fwhm/2)
        fluxes = photutils.aperture_photometry(array, apertures)
        fluxes = np.array(fluxes['aperture_sum'])

        noise_ann = np.std(fluxes)
        mean_ann = np.mean(fluxes)
        noise.append(noise_ann)
        res_level.append(mean_ann)
        vector_radd.append(rad)

        if debug:
            for j in range(xx.shape[0]):
                # Circle takes coordinates as (X,Y)
                aper = plt.Circle((xx[j], yy[j]), radius=fwhm/2, color='r',
                                  fill=False, alpha=0.8)
                ax.add_patch(aper)
                cent = plt.Circle((xx[j], yy[j]), radius=0.8, color='r',
                                  fill=True, alpha=0.5)
                ax.add_patch(cent)

        if verbose:
            print('Radius(px) = {}, Noise = {:.3f} '.format(rad, noise_ann))

    return np.array(noise), np.array(res_level), np.array(vector_radd)



def aperture_flux(array, yc, xc, fwhm, ap_factor=1, mean=False, verbose=False):
    """ Returns the sum of pixel values in a circular aperture centered on the
    input coordinates. The radius of the aperture is set as (ap_factor*fwhm)/2.

    Parameters
    ----------
    array : numpy ndarray
        Input frame.
    yc, xc : list or 1d arrays
        List of y and x coordinates of sources.
    fwhm : float
        FWHM in pixels.
    ap_factor : int, optional
        Diameter of aperture in terms of the FWHM.

    Returns
    -------
    flux : list of floats
        List of fluxes.

    Note
    ----
    From Photutils documentation, the aperture photometry defines the aperture
    using one of 3 methods:

    'center': A pixel is considered to be entirely in or out of the aperture
              depending on whether its center is in or out of the aperture.
    'subpixel': A pixel is divided into subpixels and the center of each
                subpixel is tested (as above).
    'exact': (default) The exact overlap between the aperture and each pixel is
             calculated.

    """
    n_obj = len(yc)
    flux = np.zeros((n_obj))
    for i, (y, x) in enumerate(zip(yc, xc)):
        if mean:
            ind = disk((y, x),  (ap_factor*fwhm)/2)
            values = array[ind]
            obj_flux = np.mean(values)
        else:
            aper = photutils.CircularAperture((x, y), (ap_factor*fwhm)/2)
            obj_flux = photutils.aperture_photometry(array, aper,
                                                     method='exact')
            obj_flux = np.array(obj_flux['aperture_sum'])
        flux[i] = obj_flux

        if verbose:
            print('Coordinates of object {} : ({},{})'.format(i, y, x))
            print('Object Flux = {:.2f}'.format(flux[i]))

    return flux

def estimate_snr_fc(a,b,level,n_fc,cube,psf,pa,fwhm,algo,algo_dict,
                        snrmap_empty,starphot=1,approximated=True):
                        
    cubefc = cube_inject_companions(cube, psf, pa,
                                    flevel=level*starphot,plsc=0.1, 
                                    rad_dists=a, theta=b/n_fc*360, 
                                    n_branches=1,verbose=False)
    
    if isinstance(fwhm, (np.ndarray,list)):
        fwhm_med = np.median(fwhm)
    else:
        fwhm_med = fwhm
    
    if cube.ndim==4:
        cy, cx = frame_center(cube[0,0,:,:])
    else:
        cy, cx = frame_center(cube[0])
         
    argl = inspect.getargspec(algo).args
    if 'cube' in argl and 'angle_list' in argl and 'verbose' in argl:
        if 'fwhm' in argl:
            if  'radius_int' in argl:
                
                if algo_dict.get('asize') is None:
                    annulus_width = int(np.ceil(fwhm))
                elif isinstance(algo_dict.get('asize'), int):
                    annulus_width = algo_dict.get('asize')

                if a> 2*annulus_width:
                    n_annuli = 5
                    radius_int=(a//annulus_width-2)*annulus_width 
                else:
                    n_annuli = 4 
                    radius_int=(a//annulus_width-1)*annulus_width
                if 2*(radius_int+n_annuli*annulus_width)<cube.shape[-1]:
                    
                    cubefc_crop=cube_crop_frames(cubefc,
                                    int(2*(radius_int+n_annuli*annulus_width)),
                                         xy=(cx,cy),
                                         verbose=False)
                else:
                    cubefc_crop=cubefc
        
                frame_temp = algo(cube=cubefc_crop, angle_list=pa,
                                 fwhm=fwhm_med,radius_int=radius_int,
                                 verbose=False,**algo_dict)
                frame_fin=np.zeros((cube.shape[-2],cube.shape[-1]))
                indices=get_annulus_segments(frame_fin, 0,
                                radius_int+n_annuli*annulus_width,1)
                sub=(frame_fin.shape[0]-frame_temp.shape[0])//2
                frame_fin[indices[0][0],
                          indices[0][1]]=frame_temp[indices[0][0]-sub,
                          indices[0][1]-sub]
                
            else:
                frame_fin = algo(cube=cubefc, angle_list=pa, fwhm=fwhm_med,
                              verbose=False, **algo_dict)
        else:
            frame_fin = algo(cubefc, angle_list=pa, verbose=False,
                              **algo_dict)
    
    snrmap_temp = np.zeros_like(frame_fin) 

    cy, cx = frame_center(frame_fin)
      
    mask = get_annulus_segments(frame_fin, a-(fwhm//2), fwhm+1,
                                    mode="mask")[0]
    mask = np.ma.make_mask(mask)
    yy, xx = np.where(mask)
            
    if approximated:
        coords = [(int(x), int(y)) for (x, y) in zip(xx, yy)] 
        tophat_kernel = Tophat2DKernel(fwhm / 2)
        frame_fin = convolve(frame_fin, tophat_kernel)
        res = pool_map(1, _snr_approx, frame_fin, iterable(coords),fwhm,
                       cy, cx)
        res = np.array(res, dtype=object)
        yy = res[:, 0]
        xx = res[:, 1]
        snr_value = res[:, 2]
        snrmap_temp[yy.astype(int), xx.astype(int)] = snr_value

    else:
        coords = zip(xx, yy)
        res = pool_map(1, snr, frame_fin, iterable(coords), fwhm, True)
        res = np.array(res, dtype=object)
        yy = res[:, 0]
        xx = res[:, 1]
        snr_value = res[:, -1]
        snrmap_temp[yy.astype('int'), xx.astype('int')] = snr_value
    
    snrmap_fin = np.where(abs(np.nan_to_num(snrmap_temp))>0.000001,0,
                       snrmap_empty)+np.nan_to_num(snrmap_temp)

    y,x=frame_fin.shape
    twopi=2*np.pi
    sigposy=int(y/2 + np.sin(b/n_fc*twopi)*a)
    sigposx=int(x/2+ np.cos(b/n_fc*twopi)*a)

    indc = disk((sigposy, sigposx),4)
    max_target=np.nan_to_num(snrmap_fin[indc[0],indc[1]]).max()
    snrmap_fin[indc[0],indc[1]]=0
    max_map=np.nan_to_num(snrmap_fin).max()

    return max_target-max_map,b
                        
global cc_SPHERE

cc_SPHERE=np.array([0,0,0,0,0,2.09750151e-03, 1.56692211e-03,
                    1.11612303e-03,7.63798249e-04,
       5.07941326e-04, 3.32963180e-04, 2.18449547e-04, 1.45503146e-04,
       9.95338289e-05, 7.04730484e-05, 5.18426383e-05, 3.96388202e-05,
       3.14320215e-05, 2.57516126e-05, 2.16999428e-05, 1.87207689e-05,
       1.64630132e-05, 1.47008536e-05, 1.32864330e-05, 1.21213674e-05,
       1.11391798e-05, 1.02942292e-05, 9.55463058e-06, 8.89770541e-06,
       8.30706713e-06, 7.77076101e-06, 7.28006511e-06, 6.82868514e-06,
       6.41216677e-06, 6.02742378e-06, 5.67233835e-06, 5.34542632e-06,
       5.04557146e-06, 4.77182716e-06, 4.52327586e-06, 4.29893356e-06,
       4.09769298e-06, 3.91830568e-06, 3.75940597e-06, 3.61957269e-06,
       3.49741299e-06, 3.39164391e-06, 3.30114768e-06, 3.22498521e-06,
       3.16236357e-06, 3.11256423e-06, 3.07484875e-06, 3.04836497e-06,
       3.03207801e-06, 3.02474241e-06, 3.02491751e-06, 3.03101580e-06,
       3.04136960e-06, 3.05430558e-06, 3.06822342e-06, 3.08167728e-06,
       3.09345532e-06, 3.10264506e-06, 3.10866843e-06, 3.11127543e-06,
       3.11049720e-06, 3.10657199e-06, 3.09986177e-06, 3.09077415e-06,
       3.07969675e-06, 3.06694545e-06, 3.05272591e-06, 3.03710824e-06,
       3.02001552e-06, 3.00122541e-06, 2.98038181e-06, 2.95701108e-06,
       2.93053755e-06, 2.90029448e-06, 2.86552996e-06, 2.82540902e-06,
       2.77901540e-06, 2.72535785e-06, 2.66338722e-06, 2.59203256e-06,
       2.51026574e-06, 2.41720414e-06, 2.31225656e-06, 2.19530607e-06,
       2.06690302e-06, 1.92841676e-06, 1.78207927e-06, 1.63086729e-06,
       1.47821920e-06, 1.32764891e-06, 1.18236117e-06])
                        
def completeness_curve(cube,angle_list,psf,fwhm,algo,an_dist=None,ini_contrast=None,
                            starphot=1,pxscale=1.0,n_fc=20,completeness=0.95,
                            snr_approximation=True,nproc=1,
                            algo_dict={'ncomp':20}, plot=True, dpi=100, 
                            save_plot=None, object_name=None,
                            fix_y_lim=(),figsize=(8, 4)):
        
    """
    Function allowing the computation of contrast curves with all the psf-
    subtraction algorithms provided by VIP, inspired bythe framework developped
    by Jenssen Clemm et al. (2017), the code relies on the approach proposed by
    Dahlqvist et al. (2021) which relies on the computation of the contrast 
    associated to a completeness level achieved at a level defined as the first
    false positive in the original SNR map (brightest speckle observed in the 
    empty map) instead of the computation of the local noise and throughput 
    (see the function contrast curve above). The computation of the 
    completeness level associated to a contrast is done via the sequential 
    injection of multiple fake companions. The algorithm uses multiple 
    interpolations to find the contrast associated to the selected completeness
    level (0.95 by default). More information about the algorithm can be found
    in Dahlqvist et al. (2021).
            
    Parameters
    ----------
    cube : numpy ndarray
        The input cube, 3d (ADI data) or 4d array (IFS data), without fake
        companions.
    angle_list : numpy ndarray
        Vector with the parallactic angles.
    psf : numpy ndarray
        Frame with the psf template for the fake companion(s).
        PSF must be centered in array. Normalization is done internally.
    fwhm: int or float or 1d array, optional
        The the Full Width Half Maximum in pixels. It can handle a different
        FWHM value for different wavelengths (IFS data).
    algo : callable or function
        The post-processing algorithm, e.g. vip_hci.pca.pca.
    an_dist: list or ndarray
        List of angular separations for which a contrast has to be estimated.
        Default is None which corresponds to a range of spanning between 2
        FWHM and half the size of the provided cube - PSF size //2 with a 
        step of 5 pixels
    ini_contrast: list or ndarray
        Initial contrast for the range of angular separations included in 
        an_dist.The number of initial contrasts shoul be equivalent to the 
        number of angular separations. Default is None which corresponds to the
        mean contrast achieved with the RSM approach (Dahlqvist et al. 2020)
        applied to the SHARDS survey (using the VLT/SPHERE instrument). One 
        can rely instead on the VIP contrast_curve function to get a first
        estimate. If ini_contrast=None, starphot should be provided.
    starphot : int or float or 1d array
        If int or float it corresponds to the aperture photometry of the
        non-coronagraphic PSF which we use to scale the contrast.
        Default is 1 which corresponds to the contrast expressed in ADU.
    pxscale : float
        Plate scale or pixel scale of the instrument. Default =1.0
    n_fc: int, optional
        Number of azimuths considered for the computation of the True 
        positive rate/completeness,(number of fake companions injected 
        sequentially). The number of azimuths is defined such that the 
        selected completeness is reachable (e.g. 95% of completeness 
        requires at least 20 fake companion injections). Default 20.
    completeness: float, optional
        The completeness level to be achieved when computing the contrasts,
        i.e. the True positive rate reached at the threshold associated to 
        the first false positive (the first false positive is defined as 
        the brightest speckle present in the entire detection map). 
        Default 95.
    snr_approximated : bool, optional
        If True, an approximated S/N map is generated. If False the
        approach of Mawett et al. is used (2014). Default is True 
    nproc : int or None
        Number of processes for parallel computing.
    algo_dict
        Any other valid parameter of the post-processing algorithms can be
        passed here, including e.g. imlib and interpolation.
    plot : bool, optional
        Whether to plot the final contrast curve or not. True by default.
    dpi : int optional
        Dots per inch for the plots. 100 by default. 300 for printing quality.
    save_plot: string
        If provided, the contrast curve will be saved to this path.
    object_name: string
        Target name, used in the plot title.
    fix_y_lim: tuple
        If provided, the y axis limits will be fixed, for easier comparison
        between plots.
        
    Returns
    ----------
    1D numpy ndarray containg the contrasts for the considered angular 
    distance at the selected completeness level.            
    """
    
    if (100*completeness)%(100/n_fc)>0:
        n_fc=int(100/math.gcd(int(100*completeness), 100))
    
    if cube.ndim != 3 and cube.ndim != 4:
        raise TypeError('The input array is not a 3d or 4d cube')
    if cube.ndim == 3 and (cube.shape[0] != angle_list.shape[0]):
        raise TypeError('Input parallactic angles vector has wrong length')
    if cube.ndim == 4 and (cube.shape[1] != angle_list.shape[0]):
        raise TypeError('Input parallactic angles vector has wrong length')
    if cube.ndim == 3 and psf.ndim != 2:
        raise TypeError('Template PSF is not a frame (for ADI case)')
    if cube.ndim == 4 and psf.ndim != 3:
        raise TypeError('Template PSF is not a cube (for ADI+IFS case)')
        
    if an_dist is None:
        an_dist=np.array(range(2*round(fwhm),
                               cube.shape[-1]//2-psf.shape[-1]//2-1,5))
    elif an_dist[-1]>cube.shape[-1]//2-psf.shape[-1]//2-1:
        raise TypeError('Please decrease the maximum annular distance')
        
    if ini_contrast is None:
        if starphot==1:
           raise TypeError('A star phtotmetry should be provided!') 
        ini_contrast=cc_SPHERE
        if an_dist[-1]>95:
            range_log=-(np.log10(cc_SPHERE[-1])-np.log10(5e-7))/105
            range_log*=np.array(range(0,
                       an_dist[-1]-96))
            app_contrast=np.power(10,
                                  range_log+np.log10(cc_SPHERE[-1]))
            ini_contrast=np.append(cc_SPHERE,app_contrast)
        
        ini_contrast=ini_contrast[an_dist]
            

    pa=angle_list
    
    if isinstance(fwhm, (np.ndarray,list)):
        fwhm_med = np.median(fwhm)
    else:
        fwhm_med = fwhm
       
    argl = inspect.getargspec(algo).args
    if 'cube' in argl and 'angle_list' in argl and 'verbose' in argl:
        if 'fwhm' in argl:    
            frame_fin = algo(cube=cube, angle_list=pa, fwhm=fwhm_med,
                              verbose=False, **algo_dict)
        else:
            frame_fin = algo(cube, angle_list=pa, verbose=False,
                              **algo_dict)
    
    snrmap_empty=snrmap(frame_fin, fwhm, approximated=snr_approximation,
                        plot=False,known_sources=None,nproc=nproc,
                        array2=None,use2alone=False, 
                        exclude_negative_lobes=False,verbose=False)
    
    cont_curve=np.zeros((len(an_dist)))
    
    for k in range(0,len(an_dist)):
    
        a=an_dist[k]
        level=ini_contrast[k]
        pos_detect=[]
        
        detect_bound=[None,None]
        level_bound=[None,None]
        
        while len(pos_detect)==0:
            pos_detect=[] 
            pos_non_detect=[]
            val_detect=[] 
            val_non_detect=[] 

            res=pool_map(nproc, estimate_snr_fc,a,iterable(range(0,n_fc)),
                         level,n_fc,cube,psf,pa,fwhm,algo,algo_dict,
                         snrmap_empty,starphot,approximated=True)
            
            for res_i in res:
                
               if res_i[0]>0:
                   pos_detect.append(res_i[1])
                   val_detect.append(res_i[0])
               else:
                   pos_non_detect.append(res_i[1])
                   val_non_detect.append(res_i[0])
                    
                    
            if len(pos_detect)==0:
                level=level*1.5
                
        if len(pos_detect)>round(completeness*n_fc):
            detect_bound[1]=len(pos_detect)
            level_bound[1]=level
        elif len(pos_detect)<round(completeness*n_fc):
            detect_bound[0]=len(pos_detect)
            level_bound[0]=level            
            pos_non_detect_temp=pos_non_detect.copy()
            val_non_detect_temp=val_non_detect.copy()
            pos_detect_temp=pos_detect.copy()
            val_detect_temp=val_detect.copy()
        
        cond1=(detect_bound[0]==None or detect_bound[1]==None)
        cond2=(len(pos_detect)!=round(completeness*n_fc))
            
        while cond1 and cond2:
            
            if detect_bound[0]==None:
                
                level=level*0.5
                pos_detect=[] 
                pos_non_detect=[]
                val_detect=[] 
                val_non_detect=[] 
                
                res=pool_map(nproc, estimate_snr_fc,a,
                             iterable(range(0,n_fc)),level,n_fc,cube,psf,
                             pa,fwhm,algo,algo_dict,snrmap_empty,starphot,
                             approximated=True)
            
                for res_i in res:
                    
                   if res_i[0]>0:
                       pos_detect.append(res_i[1])
                       val_detect.append(res_i[0])
                   else:
                       pos_non_detect.append(res_i[1])
                       val_non_detect.append(res_i[0])
                    
                comp_temp=round(completeness*n_fc)
                if len(pos_detect)>comp_temp and level_bound[1]>level:
                    detect_bound[1]=len(pos_detect)
                    level_bound[1]=level
                elif len(pos_detect)<comp_temp:
                    detect_bound[0]=len(pos_detect)
                    level_bound[0]=level 
                    pos_non_detect_temp=pos_non_detect.copy()
                    val_non_detect_temp=val_non_detect.copy()
                    pos_detect_temp=pos_detect.copy()
                    val_detect_temp=val_detect.copy()
                    
            elif detect_bound[1]==None:
                
                level=level*1.5
                res=pool_map(nproc, estimate_snr_fc,a,
                             iterable(-np.sort(-np.array(pos_non_detect))),
                             level,n_fc,cube,psf,pa,fwhm,algo,algo_dict,
                             snrmap_empty,starphot,approximated=True)
            
                it=len(pos_non_detect)-1        
                for res_i in res:
                    
                    if res_i[0]>0:
                       pos_detect.append(res_i[1])
                       val_detect.append(res_i[0])
                       del pos_non_detect[it]
                       del val_non_detect[it]
                    it-=1
                
                comp_temp=round(completeness*n_fc)      
                if len(pos_detect)>comp_temp:
                    detect_bound[1]=len(pos_detect)
                    level_bound[1]=level
                elif len(pos_detect)<comp_temp  and level_bound[0]<level:
                    detect_bound[0]=len(pos_detect)
                    level_bound[0]=level
                    pos_non_detect_temp=pos_non_detect.copy()
                    val_non_detect_temp=val_non_detect.copy()
                    pos_detect_temp=pos_detect.copy()
                    val_detect_temp=val_detect.copy()
                    
            cond1=(detect_bound[0]==None or detect_bound[1]==None)
            cond2=(len(pos_detect)!=comp_temp)
                    
        if len(pos_detect)!=round(completeness*n_fc):
            
            pos_non_detect=pos_non_detect_temp.copy()
            val_non_detect=val_non_detect_temp.copy()
            pos_detect=pos_detect_temp.copy()
            val_detect=val_detect_temp.copy()
        
        while len(pos_detect)!=round(completeness*n_fc):
            fact=(level_bound[1]-level_bound[0])/(detect_bound[1]-detect_bound[0])              
            level=level_bound[0]+fact*(completeness*n_fc-detect_bound[0])
            
            res=pool_map(nproc, estimate_snr_fc,a,
                         iterable(-np.sort(-np.array(pos_non_detect))),
                         level,n_fc,cube,psf,pa,fwhm,algo,algo_dict,
                         snrmap_empty,starphot,approximated=True)

            it=len(pos_non_detect)-1      
            for res_i in res:
                
                if res_i[0]>0:
                   pos_detect.append(res_i[1])
                   val_detect.append(res_i[0])
                   del pos_non_detect[it]
                   del val_non_detect[it]
                it-=1
                   
            comp_temp=round(completeness*n_fc)
            if len(pos_detect)>comp_temp:
                detect_bound[1]=len(pos_detect)
                level_bound[1]=level
            elif len(pos_detect)<comp_temp and level_bound[0]<level:
                detect_bound[0]=len(pos_detect)
                level_bound[0]=level
                pos_non_detect_temp=pos_non_detect.copy()
                val_non_detect_temp=val_non_detect.copy()
                pos_detect_temp=pos_detect.copy()
                val_detect_temp=val_detect.copy()               
            
            if len(pos_detect)!=comp_temp:
                
                pos_non_detect=pos_non_detect_temp.copy()
                val_non_detect=val_non_detect_temp.copy()
                pos_detect=pos_detect_temp.copy()
                val_detect=val_detect_temp.copy()

      
        print("Distance: "+"{}".format(a)+" Final contrast "+
              "{}".format(level))  
        cont_curve[k]=level   

    an_dist_arcsec=np.asarray(an_dist)*pxscale
    
    # plotting
    if plot:
        label = ['Sensitivity']

        plt.rc("savefig", dpi=dpi)
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax1 = fig.add_subplot(111)
        con1, = ax1.plot(an_dist_arcsec, cont_curve, '-',
                         alpha=0.2, lw=2, color='green')
        con2, = ax1.plot(an_dist_arcsec, cont_curve, '.',
                         alpha=0.2, color='green')
        
        lege = [(con1, con2)]
        
        plt.legend(lege, label, fancybox=True, fontsize='medium')
        plt.xlabel('Angular separation [arcsec]')
        plt.ylabel(str(int(completeness*10))+' percent completeness contrast')
        plt.grid('on', which='both', alpha=0.2, linestyle='solid')
        ax1.set_yscale('log')
        ax1.set_xlim(0, np.max(an_dist_arcsec)+20)

        # Give a title to the contrast curve plot
        if object_name is not None:
            # Retrieve ncomp and pca_type info to use in title
            ncomp = algo_dict['ncomp']
            if algo_dict['cube_ref'] is None:
                pca_type = 'ADI'
            else:
                pca_type = 'RDI'
            title = "{} {} {}pc".format(pca_type, object_name, ncomp)
            plt.title(title, fontsize=14)

        # Option to fix the y-limit
        if len(fix_y_lim) == 2:
            min_y_lim = min(fix_y_lim[0], fix_y_lim[1])
            max_y_lim = max(fix_y_lim[0], fix_y_lim[1])
            ax1.set_ylim(min_y_lim, max_y_lim)

        # Optionally, save the figure to a path
        if save_plot is not None:
            fig.savefig(save_plot, dpi=dpi)        
    
    return an_dist,cont_curve                              
                                                   
def completeness_map(cube,angle_list,psf,fwhm,algo,an_dist,ini_contrast,
                            starphot=1,pxscale=1.0,n_fc=20, 
                            snr_approximation=True,nproc=1,
                            algo_dict={'ncomp':20},verbose=True):
        
    """
    Function allowing the computation of three dimensional contrast curves 
    with all the psf-subtraction algorithms provided by VIP, inspired by the 
    framework developped by Jenssen Clemm et al. (2017) and the code relies on 
    the approach proposed byDahlqvist et al. (2021) which relies on the 
    computation of the contrast associated to a completeness level achieved at 
    a level defined as the first false positive in the original SNR map 
    (brightest speckle observed in the empty map). The computation of the 
    completeness level associated to a contrast is done via the sequential 
    injection of multiple fake companions. The algorithm uses multiple 
    interpolations to find the contrast associated to the selected
    completeness level (0.95by default). The function allows the computation 
    of three dimensional completeness map, with contrasts computed for multiple
    completeness level, allowing the reconstruction of the 
    contrast/completeness distribution for every considered angular
    separations.(for more details see Dahlqvist et al. 2021)
            
    Parameters
    ----------
    cube : numpy ndarray
        The input cube, 3d (ADI data) or 4d array (IFS data), without fake
        companions.
    angle_list : numpy ndarray
        Vector with the parallactic angles.
    psf : numpy ndarray
        Frame with the psf template for the fake companion(s).
        PSF must be centered in array. Normalization is done internally.
    fwhm: int or float or 1d array, optional
        The the Full Width Half Maximum in pixels. It can handle a different
        FWHM value for different wavelengths (IFS data).
    algo : callable or function
        The post-processing algorithm, e.g. vip_hci.pca.pca.
    an_dist: list or ndarray
        List of angular separations for which a contrast has to be estimated.
        Default is None which corresponds to a range of spanning between 2
        FWHM and half the size of the provided cube - PSF size //2 with a 
        step of 5 pixels
    ini_contrast: list or ndarray
        Initial contrast for the range of angular separations included in 
        an_dist.The number of initial contrasts shoul be equivalent to the 
        number of angular separations. Default is None which corresponds to the
        mean contrast achieved with the RSM approach (Dahlqvist et al. 2020)
        applied to the SHARDS survey (using the VLT/SPHERE instrument). One 
        can rely instead on the VIP contrast_curve function to get a first
        estimate. If ini_contrast=None, starphot should be provided.
    starphot : int or float or 1d array
        If int or float it corresponds to the aperture photometry of the
        non-coronagraphic PSF which we use to scale the contrast.
        Default is 1 which corresponds to the contrast expressed in ADU.
    pxscale : float
        Plate scale or pixel scale of the instrument. Default =1.0
    n_fc: int, optional
        Number of azimuths considered for the computation of the True 
        positive rate/completeness,(number of fake companions injected 
        separately). The range of achievable completenness depends on the
        number of considered azimuths (the minimum completeness is defined 
        as 1/n_fc an the maximum is 1-1/n_fc). Default 20.
    snr_approximated : bool, optional
        If True, an approximated S/N map is generated. If False the
        approach of Mawett et al. is used (2014). Default is True 
    nproc : int or None
        Number of processes for parallel computing.
    algo_dict
        Any other valid parameter of the post-processing algorithms can be
        passed here, including e.g. imlib and interpolation.
    verbose : Boolean, optional
        If True the function prints intermediate info about the comptation of
        the completeness map. Default is True.
        
    Returns
    ----------
    2D numpy ndarray providing the contrast with the first axis associated
    to the angular distance and the second axis associated to the 
    completeness level.
    """
    
    if cube.ndim != 3 and cube.ndim != 4:
        raise TypeError('The input array is not a 3d or 4d cube')
    if cube.ndim == 3 and (cube.shape[0] != angle_list.shape[0]):
        raise TypeError('Input parallactic angles vector has wrong length')
    if cube.ndim == 4 and (cube.shape[1] != angle_list.shape[0]):
        raise TypeError('Input parallactic angles vector has wrong length')
    if cube.ndim == 3 and psf.ndim != 2:
        raise TypeError('Template PSF is not a frame (for ADI case)')
    if cube.ndim == 4 and psf.ndim != 3:
        raise TypeError('Template PSF is not a cube (for ADI+IFS case)')

    if an_dist is None:
        an_dist=np.array(range(2*round(fwhm),
                               cube.shape[-1]//2-psf.shape[-1]//2-1,5))
    elif an_dist[-1]>cube.shape[-1]//2-psf.shape[-1]//2-1:
        raise TypeError('Please decrease the maximum annular distance')
        
    if ini_contrast is None:
        if starphot==1:
           raise TypeError('A star phtotmetry should be provided!') 
        ini_contrast=cc_SPHERE
        if an_dist[-1]>95:
            range_log=-(np.log10(cc_SPHERE[-1])-np.log10(5e-7))/105
            range_log*=np.array(range(0,
                       an_dist[-1]-96))
            app_contrast=np.power(10,
                                  range_log+np.log10(cc_SPHERE[-1]))
            ini_contrast=np.append(cc_SPHERE,app_contrast) 
        ini_contrast=ini_contrast[an_dist]
        
    pa=angle_list

    if isinstance(fwhm, (np.ndarray,list)):
        fwhm_med = np.median(fwhm)
    else:
        fwhm_med = fwhm
        
    argl = inspect.getargspec(algo).args
    if 'cube' in argl and 'angle_list' in argl and 'verbose' in argl:
        if 'fwhm' in argl:    
            frame_fin = algo(cube=cube, angle_list=pa, fwhm=fwhm_med,
                              verbose=False, **algo_dict)
        else:
            frame_fin = algo(cube, angle_list=pa, verbose=False,
                              **algo_dict)
    
    snrmap_empty=snrmap(frame_fin, fwhm, approximated=snr_approximation,
                        plot=False,known_sources=None,nproc=nproc,
                        array2=None,use2alone=False, 
                        exclude_negative_lobes=False,verbose=False)
    
    
    contrast_matrix=np.zeros((len(an_dist),n_fc+1))
    detect_pos_matrix=[[]]*(n_fc+1)
    
    for k in range(0,len(an_dist)):
              
        a=an_dist[k]
        level=ini_contrast[k]
        pos_detect=[] 
        detect_bound=[None,None]
        level_bound=[None,None]
        
        print("Starting annulus "+"{}".format(a))
        
        while len(pos_detect)==0:
            pos_detect=[] 
            pos_non_detect=[]
            res=pool_map(nproc, estimate_snr_fc,a,iterable(range(0,n_fc)),
                         level,n_fc,cube,psf,pa,fwhm,algo,algo_dict,
                         snrmap_empty,starphot,approximated=True)
            
            for res_i in res:
                
               if res_i[0]>0:
                   pos_detect.append(res_i[1])
               else:
                   pos_non_detect.append(res_i[1])
                    
            contrast_matrix[k,len(pos_detect)]=level
            detect_pos_matrix[len(pos_detect)]=[list(pos_detect.copy()),
                              list(pos_non_detect.copy())]
            if len(pos_detect)==0:
                level=level*1.5

       
        while contrast_matrix[k,0]==0:
            
            level=level*0.75
            res=pool_map(nproc, estimate_snr_fc,a,
                         iterable(-np.sort(-np.array(pos_detect))),level,
                         n_fc,cube,psf,pa,fwhm,algo,algo_dict,snrmap_empty,
                         starphot,approximated=True)
            
            it=len(pos_detect)-1        
            for res_i in res:
                
                if res_i[0]<0:
                   pos_non_detect.append(res_i[1])
                   del pos_detect[it]
                it-=1

            contrast_matrix[k,len(pos_detect)]=level
            detect_pos_matrix[len(pos_detect)]=[list(pos_detect.copy()),
                              list(pos_non_detect.copy())]

        if verbose:
            print("Lower boundary found") 
        
        level=contrast_matrix[k,np.where(contrast_matrix[k,:]>0)[0][-1]]
        
        pos_detect=[] 
        pos_non_detect=list(np.arange(0,n_fc))
        
        while contrast_matrix[k,n_fc]==0:
            
            level=level*1.25
            
            res=pool_map(nproc, estimate_snr_fc,a,
                         iterable(-np.sort(-np.array(pos_non_detect))),
                         level,n_fc,cube,psf,pa,fwhm,algo,algo_dict,
                         snrmap_empty,starphot,approximated=True)
            
            it=len(pos_non_detect)-1        
            for res_i in res:
                
                if res_i[0]>0:
                   pos_detect.append(res_i[1])
                   del pos_non_detect[it]
                it-=1

            contrast_matrix[k,len(pos_detect)]=level
            detect_pos_matrix[len(pos_detect)]=[list(pos_detect.copy()),
                              list(pos_non_detect.copy())]

        if verbose:
            print("Upper boundary found") 

        missing=np.where(contrast_matrix[k,:]==0)[0]
        computed=np.where(contrast_matrix[k,:]>0)[0]
        while len(missing)>0:
            
            pos_temp=np.argmax((computed-missing[0])[computed<missing[0]])
            detect_bound[0]= computed[pos_temp]
            level_bound[0]=contrast_matrix[k,detect_bound[0]]
            sort_temp=np.sort((missing[0]-computed))
            sort_temp=sort_temp[np.sort((missing[0]-computed))<0]
            detect_bound[1]= -np.sort(-computed)[np.argmax(sort_temp)]
            level_bound[1]=contrast_matrix[k,detect_bound[1]]
            it=0    
            while len(pos_detect)!=missing[0]:
                
                if np.argmin([len(detect_pos_matrix[detect_bound[1]][0]),
                        len(detect_pos_matrix[detect_bound[0]][1])])==0:
                    

                    pos_detect=list(np.sort(detect_pos_matrix[detect_bound[1]][0]))
                    pos_non_detect=list(np.sort(detect_pos_matrix[detect_bound[1]][1]))
                    fact=(level_bound[1]-level_bound[0])/(detect_bound[1]-detect_bound[0])
                    level=level_bound[1]+fact*(missing[0]-detect_bound[1])
                    
                    res=pool_map(nproc, estimate_snr_fc,a,
                                 iterable(-np.sort(-np.array(pos_detect))),
                                 level,n_fc,cube,psf,pa,fwhm,algo,
                                 algo_dict,snrmap_empty,starphot,
                                 approximated=True)
            
                    it=len(pos_detect)-1      
                    for res_i in res:
                        
                        if res_i[0]<0:
                           pos_non_detect.append(res_i[1])
                           del pos_detect[it]
                        it-=1   

                else:
                    
                    pos_detect=list(np.sort(detect_pos_matrix[detect_bound[0]][0]))
                    pos_non_detect=list(np.sort(detect_pos_matrix[detect_bound[0]][1]))
                    fact=(level_bound[1]-level_bound[0])/(detect_bound[1]-detect_bound[0])          
                    level=level_bound[0]+fact*(missing[0]-detect_bound[0])
                    
                    res=pool_map(nproc, estimate_snr_fc,a,
                            iterable(-np.sort(-np.array(pos_non_detect))),
                            level,n_fc,cube,psf,pa,fwhm,algo,algo_dict,
                            snrmap_empty,starphot,approximated=True)
            
                    it=len(pos_non_detect)-1      
                    for res_i in res:
                        
                        if res_i[0]>0:
                           pos_detect.append(res_i[1])
                           del pos_non_detect[it]
                        it-=1
                    
                if len(pos_detect)>missing[0]:
                    detect_bound[1]=len(pos_detect)
                    level_bound[1]=level
                elif len(pos_detect)<missing[0]  and level_bound[0]<level:
                    detect_bound[0]=len(pos_detect)
                    level_bound[0]=level
    
                contrast_matrix[k,len(pos_detect)]=level
                detect_pos_matrix[len(pos_detect)]=[list(pos_detect.copy()),
                                  list(pos_non_detect.copy())]
                
                if len(pos_detect)==missing[0]:
                    if verbose:
                        print("Data point "+"{}".format(len(pos_detect)/n_fc)+
                              " found. Still "+"{}".format(len(missing)-it-1)+
                              " data point(s) missing") 
                
            computed=np.where(contrast_matrix[k,:]>0)[0]
            missing=np.where(contrast_matrix[k,:]==0)[0]
            
       
    return an_dist,contrast_matrix 
