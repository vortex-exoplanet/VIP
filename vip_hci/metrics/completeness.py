#! /usr/bin/env python

"""
Module with completeness curve and map generation function.
"""

__author__ = 'C.H. Dahlqvist'
__all__ = ['completeness_curve',
           'completeness_map']

from multiprocessing import cpu_count
import numpy as np
from inspect import getfullargspec
from skimage.draw import disk
from matplotlib import pyplot as plt
from ..fm import cube_inject_companions, normalize_psf
from ..config.utils_conf import pool_map, iterable, vip_figsize, vip_figdpi
from ..var import get_annulus_segments, frame_center
from ..fm.utils_negfc import find_nearest
from ..preproc import cube_crop_frames
from .snr_source import snrmap, _snr_approx, snr
from .contrcurve import contrast_curve
from astropy.convolution import convolve, Tophat2DKernel
import math


def _estimate_snr_fc(a, b, level, n_fc, cube, psf, angle_list, fwhm, algo,
                     algo_dict, snrmap_empty, starphot=1, approximated=True):

    cubefc = cube_inject_companions(cube, psf, angle_list, flevel=level*starphot,
                                    plsc=0.1, rad_dists=a, theta=b/n_fc*360,
                                    n_branches=1, verbose=False)

    if isinstance(fwhm, (np.ndarray, list)):
        fwhm_med = np.median(fwhm)
    else:
        fwhm_med = fwhm

    if cube.ndim == 4:
        cy, cx = frame_center(cube[0, 0, :, :])
    else:
        cy, cx = frame_center(cube[0])

    argl = getfullargspec(algo).args
    if 'cube' in argl and 'angle_list' in argl and 'verbose' in argl:
        if 'fwhm' in argl:
            if 'radius_int' in argl:

                if algo_dict.get('asize') is None:
                    annulus_width = int(np.ceil(fwhm))
                elif isinstance(algo_dict.get('asize'), int):
                    annulus_width = algo_dict.get('asize')

                if a > 2*annulus_width:
                    n_annuli = 5
                    radius_int = (a//annulus_width-2)*annulus_width
                else:
                    n_annuli = 4
                    radius_int = (a//annulus_width-1)*annulus_width
                if 2*(radius_int+n_annuli*annulus_width) < cube.shape[-1]:

                    cubefc_crop = cube_crop_frames(cubefc,
                                                   int(2*(radius_int +
                                                       n_annuli*annulus_width)),
                                                   xy=(cx, cy),
                                                   verbose=False)
                else:
                    cubefc_crop = cubefc

                frame_temp = algo(cube=cubefc_crop, angle_list=angle_list,
                                  fwhm=fwhm_med, radius_int=radius_int,
                                  verbose=False, **algo_dict)
                frame_fin = np.zeros((cube.shape[-2], cube.shape[-1]))
                indices = get_annulus_segments(frame_fin, 0,
                                               radius_int+n_annuli*annulus_width,
                                               1)
                sub = (frame_fin.shape[0]-frame_temp.shape[0])//2
                frame_fin[indices[0][0],
                          indices[0][1]] = frame_temp[indices[0][0]-sub,
                                                      indices[0][1]-sub]

            else:
                frame_fin = algo(cube=cubefc, angle_list=angle_list,
                                 fwhm=fwhm_med, verbose=False, **algo_dict)
        else:
            frame_fin = algo(cubefc, angle_list=angle_list, verbose=False,
                             **algo_dict)

    snrmap_temp = np.zeros_like(frame_fin)

    cy, cx = frame_center(frame_fin)

    mask = get_annulus_segments(frame_fin, a-(fwhm_med//2), fwhm_med+1,
                                mode="mask")[0]
    mask = np.ma.make_mask(mask)
    yy, xx = np.where(mask)

    if approximated:
        coords = [(int(x), int(y)) for (x, y) in zip(xx, yy)]
        tophat_kernel = Tophat2DKernel(fwhm / 2)
        frame_fin = convolve(frame_fin, tophat_kernel)
        res = pool_map(1, _snr_approx, frame_fin, iterable(coords), fwhm, cy,
                       cx)
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

    snrmap_fin = np.where(abs(np.nan_to_num(snrmap_temp)) > 0.000001, 0,
                          snrmap_empty)+np.nan_to_num(snrmap_temp)

    y, x = frame_fin.shape
    twopi = 2*np.pi
    sigposy = int(y/2 + np.sin(b/n_fc*twopi)*a)
    sigposx = int(x/2 + np.cos(b/n_fc*twopi)*a)

    indc = disk((sigposy, sigposx), 4)
    max_target = np.nan_to_num(snrmap_fin[indc[0], indc[1]]).max()
    snrmap_fin[indc[0], indc[1]] = 0
    max_map = np.nan_to_num(snrmap_fin).max()

    return max_target-max_map, b


def completeness_curve(cube, angle_list, psf, fwhm, algo, an_dist=None,
                       ini_contrast=None, starphot=1, pxscale=0.1, n_fc=20,
                       completeness=0.95, snr_approximation=True, nproc=1,
                       algo_dict={'ncomp': 20}, plot=True, dpi=vip_figdpi,
                       save_plot=None, object_name=None, fix_y_lim=(),
                       figsize=vip_figsize):
    """
    Function allowing the computation of completeness-based contrast curves with
    any of the psf-subtraction algorithms provided by VIP. The code relies on
    the approach proposed in Dahlqvist et al. (2021), itself inspired by the
    framework developed in Jensen-Clem et al. (2018). It relies on the
    computation of the contrast associated to a completeness level achieved at
    a level defined as the first false positive in the original SNR map
    (brightest speckle observed in the empty map) instead of the computation of
    the local noise and throughput (see the `vip_hci.metrics.contrast_curve`
    function). The computation of the completeness level associated to a
    contrast is done via the sequential injection of multiple fake companions.
    The algorithm uses multiple interpolations to find the contrast associated
    to the selected completeness level (0.95 by default). More information about
    the algorithm can be found in Dahlqvist et al. (2021).

    Parameters
    ----------
    cube : 3d or 4d numpy ndarray
        The input cube, 3d (ADI data) or 4d array (IFS data), without fake
        companions.
    angle_list : 1d numpy ndarray
        Vector with parallactic angles.
    psf : 2d or 3d numpy ndarray
        Frame with the psf template for the fake companion(s).
        PSF must be centered in array. Normalization is done internally.
    fwhm: int or float or 1d array, optional
        The the Full Width Half Maximum in pixels. It can handle a different
        FWHM value for different wavelengths (IFS data).
    algo : callable or function
        The post-processing algorithm, e.g. vip_hci.pca.pca.
    an_dist: list or ndarray, optional
        List of angular separations for which a contrast has to be estimated.
        Default is None which corresponds to a range spanning 2 FWHM to half
        the size of the provided cube - PSF size //2 with a step of 5 pixels
    ini_contrast: list, 1d ndarray or None, optional
        Initial contrast for the range of angular separations included in
        an_dist. The number of initial contrasts should be equivalent to the
        number of angular separations. Default is None which corresponds to the
        mean contrast achieved with the RSM approach (Dahlqvist et al. 2020)
        applied to the SHARDS survey (using the VLT/SPHERE instrument). One
        can rely instead on the VIP contrast_curve function to get a first
        estimate.
    starphot : int or float or 1d array, optional
        If int or float it corresponds to the aperture photometry of the
        non-coronagraphic PSF which we use to scale the contrast.
        Default is 1, which corresponds to an output contrast expressed in ADU.
    pxscale : float, optional
        Plate scale or pixel scale of the instrument. Only used for plots.
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
    nproc : int or None, optional
        Number of processes for parallel computing.
    algo_dict: dictionary, optional
        Any other valid parameter of the post-processing algorithms can be
        passed here, including e.g. imlib and interpolation.
    plot : bool, optional
        Whether to plot the final contrast curve or not. True by default.
    dpi : int optional
        Dots per inch for the plots. 100 by default. 300 for printing quality.
    save_plot: string or None, optional
        If provided, the contrast curve will be saved to this path.
    object_name: string or None, optional
        Target name, used in the plot title.
    fix_y_lim: tuple, optional
        If provided, the y axis limits will be fixed, for easier comparison
        between plots.
    fig_size: tuple, optional
        Figure size

    Returns
    -------
    an_dist: 1d numpy ndarray
        Radial distances where the contrasts are calculated
    cont_curve: 1d numpy ndarray
        Contrasts for the considered radial distances and selected completeness
        level.
    """

    if (100*completeness) % (100/n_fc) > 0:
        n_fc = int(100/math.gcd(int(100*completeness), 100))

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
    if nproc is None:
        nproc = cpu_count()//2

    if isinstance(fwhm, (np.ndarray, list)):
        fwhm_med = np.median(fwhm)
    else:
        fwhm_med = fwhm

    if an_dist is None:
        an_dist = np.array(range(2*round(fwhm_med),
                                 int(cube.shape[-1]//2-2*fwhm_med), 5))
        print("an_dist not provided, the following list will be used:", an_dist)
    elif an_dist[-1] > cube.shape[-1]//2-2*fwhm_med:
        raise TypeError('Please decrease the maximum annular distance')

    if ini_contrast is None:
        print("Contrast curve not provided => will be computed first...")
        ini_cc = contrast_curve(cube, angle_list, psf, fwhm_med, pxscale,
                                starphot, algo, sigma=3, nbranch=1, theta=0,
                                inner_rad=1, wedge=(0, 360), fc_snr=100,
                                plot=False, **algo_dict)
        ini_rads = np.array(ini_cc['distance'])
        ini_cc = np.array(ini_cc['sensitivity_student'])

        if np.amax(an_dist) > np.amax(ini_rads):
            msg = 'Max requested annular distance larger than covered by '
            msg += 'contrast curve. Please decrease the maximum annular distance'
            raise ValueError(msg)

        # find closest contrast values to requested radii
        ini_contrast = []
        for aa, ad in enumerate(an_dist):
            idx = find_nearest(ini_rads, ad)
            ini_contrast.append(ini_cc[idx])

    argl = getfullargspec(algo).args
    if 'cube' in argl and 'angle_list' in argl and 'verbose' in argl:
        if 'fwhm' in argl:
            frame_fin = algo(cube=cube, angle_list=angle_list, fwhm=fwhm_med,
                             verbose=False, **algo_dict)
        else:
            frame_fin = algo(cube, angle_list=angle_list, verbose=False,
                             **algo_dict)

    snrmap_empty = snrmap(frame_fin, fwhm, approximated=snr_approximation,
                          plot=False, known_sources=None, nproc=nproc,
                          array2=None, use2alone=False,
                          exclude_negative_lobes=False, verbose=False)

    cont_curve = np.zeros((len(an_dist)))

    # We crop the PSF and check if PSF has been normalized (so that flux in
    # 1*FWHM aperture = 1) and fix if needed
    new_psf_size = int(round(3 * fwhm_med))
    if new_psf_size % 2 == 0:
        new_psf_size += 1
    # Normalize psf
    psf = normalize_psf(psf, fwhm=fwhm, verbose=False, size=min(new_psf_size,
                                                                psf.shape[1]))

    for k in range(len(an_dist)):

        a = an_dist[k]
        level = ini_contrast[k]
        pos_detect = []

        detect_bound = [None, None]
        level_bound = [None, None]

        while len(pos_detect) == 0:
            pos_detect = []
            pos_non_detect = []
            val_detect = []
            val_non_detect = []

            res = pool_map(nproc, _estimate_snr_fc, a, iterable(range(0, n_fc)),
                           level, n_fc, cube, psf, angle_list, fwhm, algo,
                           algo_dict, snrmap_empty, starphot, approximated=True)

            for res_i in res:

                if res_i[0] > 0:
                    pos_detect.append(res_i[1])
                    val_detect.append(res_i[0])
                else:
                    pos_non_detect.append(res_i[1])
                    val_non_detect.append(res_i[0])

            if len(pos_detect) == 0:
                level = level*1.5

        if len(pos_detect) > round(completeness*n_fc):
            detect_bound[1] = len(pos_detect)
            level_bound[1] = level
        elif len(pos_detect) < round(completeness*n_fc):
            detect_bound[0] = len(pos_detect)
            level_bound[0] = level
            pos_non_detect_temp = pos_non_detect.copy()
            val_non_detect_temp = val_non_detect.copy()
            pos_detect_temp = pos_detect.copy()
            val_detect_temp = val_detect.copy()

        cond1 = (detect_bound[0] is None or detect_bound[1] is None)
        cond2 = (len(pos_detect) != round(completeness*n_fc))

        while cond1 and cond2:

            if detect_bound[0] is None:

                level = level*0.5
                pos_detect = []
                pos_non_detect = []
                val_detect = []
                val_non_detect = []

                res = pool_map(nproc, _estimate_snr_fc, a, iterable(range(0, n_fc)),
                               level, n_fc, cube, psf, angle_list, fwhm, algo,
                               algo_dict, snrmap_empty, starphot,
                               approximated=True)

                for res_i in res:

                    if res_i[0] > 0:
                        pos_detect.append(res_i[1])
                        val_detect.append(res_i[0])
                    else:
                        pos_non_detect.append(res_i[1])
                        val_non_detect.append(res_i[0])

                comp_temp = round(completeness*n_fc)
                if len(pos_detect) > comp_temp and level_bound[1] > level:
                    detect_bound[1] = len(pos_detect)
                    level_bound[1] = level
                elif len(pos_detect) < comp_temp:
                    detect_bound[0] = len(pos_detect)
                    level_bound[0] = level
                    pos_non_detect_temp = pos_non_detect.copy()
                    val_non_detect_temp = val_non_detect.copy()
                    pos_detect_temp = pos_detect.copy()
                    val_detect_temp = val_detect.copy()

            elif detect_bound[1] is None:

                level = level*1.5
                res = pool_map(nproc, _estimate_snr_fc, a,
                               iterable(-np.sort(-np.array(pos_non_detect))),
                               level, n_fc, cube, psf, angle_list, fwhm, algo,
                               algo_dict, snrmap_empty, starphot,
                               approximated=True)

                it = len(pos_non_detect)-1
                for res_i in res:

                    if res_i[0] > 0:
                        pos_detect.append(res_i[1])
                        val_detect.append(res_i[0])
                        del pos_non_detect[it]
                        del val_non_detect[it]
                    it -= 1

                comp_temp = round(completeness*n_fc)
                if len(pos_detect) > comp_temp:
                    detect_bound[1] = len(pos_detect)
                    level_bound[1] = level
                elif len(pos_detect) < comp_temp and level_bound[0] < level:
                    detect_bound[0] = len(pos_detect)
                    level_bound[0] = level
                    pos_non_detect_temp = pos_non_detect.copy()
                    val_non_detect_temp = val_non_detect.copy()
                    pos_detect_temp = pos_detect.copy()
                    val_detect_temp = val_detect.copy()

            cond1 = (detect_bound[0] is None or detect_bound[1] is None)
            cond2 = (len(pos_detect) != comp_temp)

        if len(pos_detect) != round(completeness*n_fc):

            pos_non_detect = pos_non_detect_temp.copy()
            val_non_detect = val_non_detect_temp.copy()
            pos_detect = pos_detect_temp.copy()
            val_detect = val_detect_temp.copy()

        while len(pos_detect) != round(completeness*n_fc):
            fact = (level_bound[1]-level_bound[0]) / \
                    (detect_bound[1]-detect_bound[0])
            level = level_bound[0]+fact*(completeness*n_fc-detect_bound[0])

            res = pool_map(nproc, _estimate_snr_fc, a,
                           iterable(-np.sort(-np.array(pos_non_detect))),
                           level, n_fc, cube, psf, angle_list, fwhm, algo,
                           algo_dict, snrmap_empty, starphot, approximated=True)

            it = len(pos_non_detect)-1
            for res_i in res:

                if res_i[0] > 0:
                    pos_detect.append(res_i[1])
                    val_detect.append(res_i[0])
                    del pos_non_detect[it]
                    del val_non_detect[it]
                it -= 1

            comp_temp = round(completeness*n_fc)
            if len(pos_detect) > comp_temp:
                detect_bound[1] = len(pos_detect)
                level_bound[1] = level
            elif len(pos_detect) < comp_temp and level_bound[0] < level:
                detect_bound[0] = len(pos_detect)
                level_bound[0] = level
                pos_non_detect_temp = pos_non_detect.copy()
                val_non_detect_temp = val_non_detect.copy()
                pos_detect_temp = pos_detect.copy()
                val_detect_temp = val_detect.copy()

            if len(pos_detect) != comp_temp:

                pos_non_detect = pos_non_detect_temp.copy()
                val_non_detect = val_non_detect_temp.copy()
                pos_detect = pos_detect_temp.copy()
                val_detect = val_detect_temp.copy()

        print("Distance: "+"{}".format(a)+" Final contrast " +
              "{}".format(level))
        cont_curve[k] = level

    # plotting
    if plot:
        an_dist_arcsec = np.asarray(an_dist)*pxscale
        label = ['Sensitivity']

        plt.rc("savefig", dpi=dpi)
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax1 = fig.add_subplot(111)
        con1, = ax1.plot(an_dist_arcsec, cont_curve, '-', alpha=0.2, lw=2,
                         color='green')
        con2, = ax1.plot(an_dist_arcsec, cont_curve, '.', alpha=0.2,
                         color='green')

        lege = [(con1, con2)]

        plt.legend(lege, label, fancybox=True, fontsize='medium')
        plt.xlabel('Angular separation [arcsec]')
        plt.ylabel(str(int(completeness*100))+'% completeness contrast')
        plt.grid('on', which='both', alpha=0.2, linestyle='solid')
        ax1.set_yscale('log')
        ax1.set_xlim(0, 1.1*np.max(an_dist_arcsec))

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

    return an_dist, cont_curve


def completeness_map(cube, angle_list, psf, fwhm, algo, an_dist, ini_contrast,
                     starphot=1, n_fc=20, snr_approximation=True, nproc=1,
                     algo_dict={'ncomp': 20}, verbose=True):
    """
    Function allowing the computation of two dimensional (radius and 
    completeness) contrast curves with any psf-subtraction algorithm provided by
    VIP. The code relies on the approach proposed by Dahlqvist et al. (2021), 
    itself inspired by the framework developped by Jensen-Clem et al. (2018). 
    It relies on the computation of the contrast associated to a completeness 
    level achieved at
    a level defined as the first false positive in the original SNR map
    (brightest speckle observed in the empty map). The computation of the
    completeness level associated to a contrast is done via the sequential
    injection of multiple fake companions. The algorithm uses multiple
    interpolations to find the contrast associated to the selected
    completeness level (0.95by default). The function allows the computation
    of three dimensional completeness map, with contrasts computed for multiple
    completeness level, allowing the reconstruction of the
    contrast/completeness distribution for every considered angular
    separations (for more details see Dahlqvist et al. 2021).

    Parameters
    ----------
    cube : 3d or 4d numpy ndarray
        The input cube, 3d (ADI data) or 4d array (IFS data), without fake
        companions.
    angle_list : 1d numpy ndarray
        Vector with parallactic angles.
    psf : 2d or 3d numpy ndarray
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
        an_dist. The number of initial contrasts shoul be equivalent to the
        number of angular separations. Default is None which corresponds to the
        mean contrast achieved with the RSM approach (Dahlqvist et al. 2020)
        applied to the SHARDS survey (using the VLT/SPHERE instrument). One
        can rely instead on the VIP contrast_curve function to get a first
        estimate. If ini_contrast=None, starphot should be provided.
    starphot : int or float or 1d array
        If int or float it corresponds to the aperture photometry of the
        non-coronagraphic PSF which we use to scale the contrast.
        Default is 1 which corresponds to the contrast expressed in ADU.
    n_fc: int, optional
        Number of azimuths considered for the computation of the True
        positive rate/completeness, (number of fake companions injected
        separately). The range of achievable completeness depends on the
        number of considered azimuths (the minimum completeness is defined
        as 1/n_fc and the maximum is 1-1/n_fc). Default 20.
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
    -------
    an_dist: 1d numpy ndarray
        Radial distances where the contrasts are calculated
    cont_curve: 2d numpy ndarray
        Contrast matrix, with the first axis associated to the radial distances
        and the second axis associated to the completeness level.

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
    if nproc is None:
        nproc = cpu_count()//2
        
    if isinstance(fwhm, (np.ndarray, list)):
        fwhm_med = np.median(fwhm)
    else:
        fwhm_med = fwhm

    if an_dist is None:
        an_dist = np.array(range(2*round(fwhm),
                                 cube.shape[-1]//2-2*fwhm_med, 5))
    elif an_dist[-1] > cube.shape[-1]//2-2*fwhm_med:
        raise TypeError('Please decrease the maximum annular distance')

    if ini_contrast is None:
        print("Contrast curve not provided => will be computed first...")
        # pxscale unused if plot=False
        ini_cc = contrast_curve(cube, angle_list, psf, fwhm_med, pxscale=0.1,
                                starphot=starphot, algo=algo, sigma=3,
                                plot=False, **algo_dict)
        ini_rads = np.array(ini_cc['distance'])
        ini_cc = np.array(ini_cc['sensitivity_student'])

        if np.amax(an_dist) > np.amax(ini_rads):
            msg = 'Max requested annular distance larger than covered by '
            msg += 'contrast curve. Please decrease the maximum annular distance'
            raise ValueError(msg)

        # find closest contrast values to requested radii
        ini_contrast = []
        for aa, ad in enumerate(an_dist):
            idx = find_nearest(ini_rads, ad)
            ini_contrast.append(ini_cc[idx])

    argl = getfullargspec(algo).args
    if 'cube' in argl and 'angle_list' in argl and 'verbose' in argl:
        if 'fwhm' in argl:
            frame_fin = algo(cube=cube, angle_list=angle_list, fwhm=fwhm_med,
                             verbose=False, **algo_dict)
        else:
            frame_fin = algo(cube, angle_list=angle_list, verbose=False,
                             **algo_dict)

    snrmap_empty = snrmap(frame_fin, fwhm_med, approximated=snr_approximation,
                          plot=False, known_sources=None, nproc=nproc,
                          array2=None, use2alone=False,
                          exclude_negative_lobes=False, verbose=False)

    contrast_matrix = np.zeros((len(an_dist), n_fc+1))
    detect_pos_matrix = [[]]*(n_fc+1)

    for k in range(0, len(an_dist)):

        a = an_dist[k]
        level = ini_contrast[k]
        pos_detect = []
        det_bound = [None, None]
        lvl_bound = [None, None]

        print("Starting annulus "+"{}".format(a))

        while len(pos_detect) == 0:
            pos_detect = []
            pos_non_detect = []
            res = pool_map(nproc, _estimate_snr_fc, a, iterable(range(0, n_fc)),
                           level, n_fc, cube, psf, angle_list, fwhm, algo,
                           algo_dict, snrmap_empty, starphot, approximated=True)

            for res_i in res:

                if res_i[0] > 0:
                    pos_detect.append(res_i[1])
                else:
                    pos_non_detect.append(res_i[1])

            contrast_matrix[k, len(pos_detect)] = level
            detect_pos_matrix[len(pos_detect)] = [list(pos_detect.copy()),
                                                  list(pos_non_detect.copy())]
            if len(pos_detect) == 0:
                level = level*1.5

        while contrast_matrix[k, 0] == 0:

            level = level*0.75
            res = pool_map(nproc, _estimate_snr_fc, a,
                           iterable(-np.sort(-np.array(pos_detect))), level,
                           n_fc, cube, psf, angle_list, fwhm, algo, algo_dict,
                           snrmap_empty, starphot, approximated=True)

            it = len(pos_detect)-1
            for res_i in res:

                if res_i[0] < 0:
                    pos_non_detect.append(res_i[1])
                    del pos_detect[it]
                it -= 1

            contrast_matrix[k, len(pos_detect)] = level
            detect_pos_matrix[len(pos_detect)] = [list(pos_detect.copy()),
                                                  list(pos_non_detect.copy())]

        if verbose:
            print("Lower boundary found")

        level = contrast_matrix[k, np.where(contrast_matrix[k, :] > 0)[0][-1]]

        pos_detect = []
        pos_non_detect = list(np.arange(0, n_fc))

        while contrast_matrix[k, n_fc] == 0:

            level = level*1.25

            res = pool_map(nproc, _estimate_snr_fc, a,
                           iterable(-np.sort(-np.array(pos_non_detect))),
                           level, n_fc, cube, psf, angle_list, fwhm, algo,
                           algo_dict, snrmap_empty, starphot, approximated=True)

            it = len(pos_non_detect)-1
            for res_i in res:

                if res_i[0] > 0:
                    pos_detect.append(res_i[1])
                    del pos_non_detect[it]
                it -= 1

            contrast_matrix[k, len(pos_detect)] = level
            detect_pos_matrix[len(pos_detect)] = [list(pos_detect.copy()),
                                                  list(pos_non_detect.copy())]

        if verbose:
            print("Upper boundary found")

        missing = np.where(contrast_matrix[k, :] == 0)[0]
        computed = np.where(contrast_matrix[k, :] > 0)[0]
        while len(missing) > 0:

            pos_temp = np.argmax((computed-missing[0])[computed < missing[0]])
            det_bound[0] = computed[pos_temp]
            lvl_bound[0] = contrast_matrix[k, det_bound[0]]
            sort_temp = np.sort((missing[0]-computed))
            sort_temp = sort_temp[np.sort((missing[0]-computed)) < 0]
            det_bound[1] = -np.sort(-computed)[np.argmax(sort_temp)]
            lvl_bound[1] = contrast_matrix[k, det_bound[1]]
            it = 0
            while len(pos_detect) != missing[0]:

                if np.argmin([len(detect_pos_matrix[det_bound[1]][0]),
                              len(detect_pos_matrix[det_bound[0]][1])]) == 0:

                    pos_detect = np.sort(detect_pos_matrix[det_bound[1]][0])
                    pos_non_detect = np.sort(detect_pos_matrix[det_bound[1]][1])
                    pos_detect = list(pos_detect)
                    pos_non_detect = list(pos_non_detect)
                    num = lvl_bound[1]-lvl_bound[0]
                    denom = det_bound[1]-det_bound[0]
                    level = lvl_bound[1]+num*(missing[0]-det_bound[1])/denom

                    res = pool_map(nproc, _estimate_snr_fc, a,
                                   iterable(-np.sort(-np.array(pos_detect))),
                                   level, n_fc, cube, psf, angle_list, fwhm, algo,
                                   algo_dict, snrmap_empty, starphot,
                                   approximated=True)

                    it = len(pos_detect)-1
                    for res_i in res:

                        if res_i[0] < 0:
                            pos_non_detect.append(res_i[1])
                            del pos_detect[it]
                        it -= 1

                else:

                    pos_detect = np.sort(detect_pos_matrix[det_bound[0]][0])
                    pos_non_detect = np.sort(detect_pos_matrix[det_bound[0]][1])
                    pos_detect = list(pos_detect)
                    pos_non_detect = list(pos_non_detect)
                    num = lvl_bound[1]-lvl_bound[0]
                    denom = det_bound[1]-det_bound[0]
                    level = lvl_bound[0]+num*(missing[0]-det_bound[0])/denom

                    res = pool_map(nproc, _estimate_snr_fc, a,
                                   iterable(-np.sort(-np.array(pos_non_detect))),
                                   level, n_fc, cube, psf, angle_list, fwhm, algo,
                                   algo_dict, snrmap_empty, starphot,
                                   approximated=True)

                    it = len(pos_non_detect)-1
                    for res_i in res:

                        if res_i[0] > 0:
                            pos_detect.append(res_i[1])
                            del pos_non_detect[it]
                        it -= 1

                if len(pos_detect) > missing[0]:
                    det_bound[1] = len(pos_detect)
                    lvl_bound[1] = level
                elif len(pos_detect) < missing[0] and lvl_bound[0] < level:
                    det_bound[0] = len(pos_detect)
                    lvl_bound[0] = level

                contrast_matrix[k, len(pos_detect)] = level
                detect_pos_matrix[len(pos_detect)] = [list(pos_detect.copy()),
                                                      list(pos_non_detect.copy())]

                if len(pos_detect) == missing[0]:
                    if verbose:
                        print("Data point "+"{}".format(len(pos_detect)/n_fc) +
                              " found. Still "+"{}".format(len(missing)-it-1) +
                              " data point(s) missing")

            computed = np.where(contrast_matrix[k, :] > 0)[0]
            missing = np.where(contrast_matrix[k, :] == 0)[0]

    return an_dist, contrast_matrix
