#! /usr/bin/env python

"""
Module with contrast curve generation function.
"""

__author__ = 'C. Gomez, O. Absil @ ULg'
__all__ = ['contrast_curve',
           'noise_per_annulus',
           'throughput',
           'aperture_flux']

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
from ..config.utils_conf import sep
from ..var import frame_center, dist


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
