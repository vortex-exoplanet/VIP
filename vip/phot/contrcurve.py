#! /usr/bin/env python

"""
Module with contrast curve generation function.
"""

from __future__ import division
from __future__ import print_function

__author__ = 'C. Gomez, O. Absil @ ULg'
__all__ = ['contrast_curve',
           'noise_per_annulus',
           'throughput',
           'aperture_flux']

import numpy as np
import photutils
import inspect
from scipy.interpolate import interp1d
from scipy import stats
from scipy.signal import savgol_filter
from skimage.draw import circle
from matplotlib import pyplot as plt
from .fakecomp import inject_fcs_cube, inject_fc_frame, psf_norm
from ..conf import timeInit, timing
from ..var import frame_center, dist



def contrast_curve(cube, angle_list, psf_template, fwhm, pxscale, starphot, 
                   algo, sigma=5, nbranch=1, student=True, plot=True, 
                   dpi=100, debug=False, verbose=True, **algo_dict):
    """ Computes the contrast curve for a given SIGMA (*sigma*) level. The 
    contrast is calculated as sigma*noise/throughput. This implementation takes
    into account the small sample statistics correction proposed in Mawet et al.
    2014. 
    
    Parameters
    ----------
    cube : array_like
        The input cube without fake companions.
    angle_list : array_like
        Vector with the parallactic angles.
    psf_template : array_like
        Frame with the psf template for the fake companion(s).
        PSF must be centered in array. Normalization is done internally.
    fwhm : float
        FWHM in pixels.
    pxscale : float
        Plate scale or pixel scale of the instrument. 
    starphot : int or float or 1d array
        If int or float it corresponds to the aperture photometry of the 
        non-coronagraphic PSF which we use to scale the contrast. If a vector 
        is given it must contain the photometry correction for each frame.
    algo : callable or function
        The post-processing algorithm, e.g. vip.pca.pca.
    sigma : int
        Sigma level for contrast calculation.
    nbranch : int optional
        Number of branches on which to inject fakes companions. Each branch
        is tested individually.  
    student : {True, False}, bool optional
        If True uses Student t correction to inject fake companion.      
    plot : {True, False}, bool optional 
        Whether to plot the final contrast curve or not. True by default.
    dpi : int optional 
        Dots per inch for the plots. 100 by default. 300 for printing quality.
    debug : {False, True}, bool optional
        Whether to print and plot additional info such as the noise, throughput,
        the contrast curve with different X axis and the delta magnitude intead
        of constrast.
    verbose : {True, False}, bool optional
        If True prints to stdout intermediate info and timing. 
    **algo_dict
        Any other valid parameter of the post-processing algorithms can be 
        passed here.
    
    Returns
    -------
    cont_curve_samp : array_like
        Final contrast for the input sigma, as seen in the plot.
    cont_curve_samp_t : array_like
        Contrast with the student t correction.
    rad_samp : array_like
        Vector of distances in arcseconds. 
        
    """  
    if not cube.ndim == 3:
        raise TypeError('The input array is not a cube')
    if not cube.shape[0] == angle_list.shape[0]:
        raise TypeError('Input vector or parallactic angles has wrong length')
    if not psf_template.ndim==2:
        raise TypeError('Template PSF is not a frame')    

    if isinstance(starphot, float) or isinstance(starphot, int):  pass
    else:
        if not starphot.shape[0] == cube.shape[0]:
            raise TypeError('Correction vector has bad size')
        cube = cube.copy()
        for i in xrange(cube.shape[0]):
            cube[i] = cube[i] / starphot[i]

    fwhm = int(np.round(fwhm))

    if verbose:  start_time = timeInit()
    
    # throughput
    res_throug = throughput(cube, angle_list, psf_template, fwhm, pxscale, 
                            nbranch=nbranch, full_output=True, algo=algo,
                            verbose=False, **algo_dict)
    vector_radd = res_throug[2] 
    if res_throug[0].shape[0]>1:
        thruput_mean = np.mean(res_throug[0], axis=0)
    else:
        thruput_mean = res_throug[0][0]
    frame_nofc = res_throug[5]
    
    if verbose:
        msg1 = 'Finished the throughput calculation'
        print(msg1)
        timing(start_time)
    
    if thruput_mean[-1]==0:
        thruput_mean = thruput_mean[:-1]
        vector_radd = vector_radd[:-1]
    
    # noise measured in the empty PCA-frame with better sampling, every px
    # starting from 1*FWHM
    noise_samp, rad_samp = noise_per_annulus(frame_nofc, 1, fwhm, False)        
    cutin1 = np.where(rad_samp.astype(int)==vector_radd.astype(int).min())[0]
    noise_samp = noise_samp[cutin1:]
    rad_samp = rad_samp[cutin1:]
    cutin2 = np.where(rad_samp.astype(int)==vector_radd.astype(int).max())[0]
    noise_samp = noise_samp[:cutin2+1]
    rad_samp = rad_samp[:cutin2+1]
        
    # interpolating the throughput vector
    f = interp1d(vector_radd, thruput_mean, bounds_error=False, fill_value=0)
    thruput_interp = f(rad_samp)      
    
    # smoothing the throughput and noise vectors using a Savitzky-Golay filter
    win1 = int(thruput_interp.shape[0]*0.1)
    win2 = int(noise_samp.shape[0]*0.1)
    if win1%2==0.:  win1 += 1
    if win2%2==0.:  win2 += 1
    thruput_interp_sm = savgol_filter(thruput_interp, polyorder=1, mode='nearest',
                                      window_length=win1)
    noise_samp_sm = savgol_filter(noise_samp, polyorder=1, mode='nearest',
                                  window_length=win2)
    
    if debug:
        print('SIGMA={}'.format(sigma))
        if isinstance(starphot, float) or isinstance(starphot, int):
            print('STARPHOT={}'.format(starphot))
        
        plt.rc("savefig", dpi=dpi)
        plt.figure(figsize=(8,4))
        plt.plot(rad_samp*pxscale, thruput_interp_sm, ',-', label='smoothed', 
                 lw=2, alpha=0.5)
        plt.plot(rad_samp*pxscale, thruput_interp, '.', label='interpolated')
        plt.plot(vector_radd*pxscale, thruput_mean, 'o', label='computed', 
                 alpha=0.4, color='blue')
        plt.grid('on', which='both', alpha=0.2, linestyle='solid')
        plt.xlabel('Angular separation [arcsec]')
        plt.ylabel('Throughput')
        plt.legend(loc='best')
        plt.xlim(0, np.max(rad_samp*pxscale))
        
        plt.figure(figsize=(8,4))
        plt.plot(rad_samp*pxscale, noise_samp, '.', label='computed')
        plt.plot(rad_samp*pxscale, noise_samp_sm, ',-', label='noise smoothed', 
                 lw=2, alpha=0.5)
        plt.grid('on', alpha=0.2, linestyle='solid')
        plt.xlabel('Angular separation [arcsec]')
        plt.ylabel('Noise')
        plt.legend(loc='best')
        plt.yscale('log')
        plt.xlim(0, np.max(rad_samp*pxscale))
    
    sig_label = sigma
    # student t correction
    if student:
        n_res_els = np.floor(rad_samp/fwhm*2*np.pi)
        ss_corr = np.sqrt(1 + 1/(n_res_els-1))
        sigma = stats.t.ppf(stats.norm.cdf(sigma), n_res_els)/ss_corr
    
    # calculating the contrast
    if isinstance(starphot, float) or isinstance(starphot, int):
        cont_curve_samp = ((sigma * noise_samp_sm)/thruput_interp_sm)/starphot
    else:
        cont_curve_samp = ((sigma * noise_samp_sm)/thruput_interp_sm)
        
    # plotting
    if plot or debug:
        if student:  label = 'CC (student-t correction)'
        else:  label = 'CC (gaussian)'
        
        plt.rc("savefig", dpi=dpi)
        fig = plt.figure(figsize=(8,4))
        ax1 = fig.add_subplot(111)
        con1, = ax1.plot(rad_samp*pxscale, cont_curve_samp, '-', alpha=0.8)
        con2, = ax1.plot(rad_samp*pxscale, cont_curve_samp, '.', alpha=0.4)
        plt.xlabel('Angular separation [arcsec]')
        plt.ylabel(str(sig_label)+' sigma contrast')
        plt.legend([(con1, con2)], [label], fancybox=True, fontsize='medium')
        plt.grid('on', which='both', alpha=0.2, linestyle='solid')
        plt.yscale('log')
        ax1.set_xlim(0, np.max(rad_samp*pxscale))
        ax2 = ax1.twiny()
        ax2.set_xlabel('Distance [pixels]')
        ax2.plot(rad_samp, cont_curve_samp, '',alpha=0.)
        ax2.set_xlim(0, np.max(rad_samp))                                 
        
        if debug:        
            plt.figure(figsize=(8,4))
            con3, = plt.plot(rad_samp*pxscale, -2.5*np.log10(cont_curve_samp), '-', 
                             alpha=0.8)
            con4, = plt.plot(rad_samp*pxscale, -2.5*np.log10(cont_curve_samp), '.', 
                             alpha=0.4)
            plt.xlabel('Angular separation [arcsec]')
            plt.ylabel('Delta magnitude')
            plt.legend([(con3, con4)], [label], fancybox=True, fontsize='medium')
            plt.gca().invert_yaxis()
            plt.grid('on', which='both', alpha=0.2, linestyle='solid')
            plt.xlim(0, np.max(rad_samp*pxscale))
    
    if verbose:  timing(start_time)
    
    return cont_curve_samp, rad_samp*pxscale


def throughput(cube, angle_list, psf_template, fwhm, pxscale, algo, 
               nbranch=3, fc_rad_sep=3, full_output=False, verbose=True, 
               **algo_dict):
    """ Measures the throughput for chosen algorithm and input dataset. The 
    final throughput is the average of the same procedure measured in *nbranch* 
    azimutally equidistant branches.
    
    Parameters
    ----------
    cube : array_like
        The input cube without fake companions.
    angle_list : array_like
        Vector with the parallactic angles.
    psf_template : array_like
        Frame with the psf template for the fake companion(s).
        PSF must be centered in array. Normalization is done internally.
    fwhm : float
        FWHM in pixels.
    pxscale : float
        Plate scale in arcsec/px.
    algo : callable or function
        The post-processing algorithm, e.g. vip.pca.pca. Third party Python 
        algorithms can be plugged here. They must have the parameters: 'cube', 
        'angle_list' and 'verbose'. Optionally a wrapper function can be used.
    nbranch : int optional
        Number of branches on which to inject fakes companions. Each branch
        is tested individually.
    fc_rad_sep : int optional
        Radial separation between the injection companions (in each of the 
        patterns) in FWHM. Must be large enough to avoid overlapping.  
    full_output : {False, True}, bool optional
        If True returns intermediate arrays.
    verbose : {True, False}, bool optional
        If True prints out timing and information.
    **algo_dict
        Parameters of the post-processing algorithms must be passed here.
    
    Returns
    -------
    thruput_arr : array_like
        2d array whose rows are the annulus-wise throughput values for each 
        branch.
    vector_radd : array_like
        1d array with the distances in FWHM (the positions of the annuli).
        
    If full_output is True then the function returns: thruput_arr, noise, 
    vector_radd, cube_fc_all, frame_fc_all, frame_nofc and fc_map_all.
    
    noise : array_like
        1d array with the noise per annulus.
    cube_fc_all : array_like
        4d array, with the 3 different pattern cubes with the injected fake 
        companions.
    frame_fc_all : array_like
        3d array with the 3 frames of the 3 (patterns) processed cubes with 
        companions.
    frame_nofc : array_like
        2d array, PCA processed frame without companions.
    fc_map_all : array_like
        3d array with 3 frames containing the position of the companions in the
        3 patterns.

    """
    array = cube
    parangles = angle_list
    fwhm = int(np.round(fwhm))
    
    if not array.ndim == 3:
        raise TypeError('The input array is not a cube')
    if not array.shape[0] == parangles.shape[0]:
        raise TypeError('Input vector or parallactic angles has wrong length')
    if not psf_template.ndim==2:
        raise TypeError('Template PSF is not a frame or 2d array')
    if not hasattr(algo, '__call__'):
        raise TypeError('Parameter *algo* must be a callable function')

    if verbose:  start_time = timeInit()
    #***************************************************************************
    # Compute noise in concentric annuli on the "empty frame"  
    if 'cube' and 'angle_list' and 'verbose' in inspect.getargspec(algo).args:
        if 'fwhm' in inspect.getargspec(algo).args:
            frame_nofc = algo(cube=array, angle_list=parangles, fwhm=fwhm, 
                              verbose=False, **algo_dict)
        else:
            frame_nofc = algo(array, angle_list=parangles, verbose=False, 
                              **algo_dict)
    
    if verbose:
        msg1 = 'Cube without fake companions processed with {:}'
        print(msg1.format(algo.func_name))
        timing(start_time)
    
    noise, vector_radd = noise_per_annulus(frame_nofc,fwhm,fwhm,verbose=False)
    if verbose:
        print('Measured annulus-wise noise in resulting frame')
        timing(start_time)
    
    # We crop the PSF and check if PSF has been normalized (so that flux in 
    # 1*FWHM aperture = 1) and fix if needed
    psf_template = psf_norm(psf_template, size=3*fwhm, fwhm=fwhm)
    
    #***************************************************************************
    # Initialize the fake companions
    angle_branch = 360.0/nbranch        
    # signal-to-noise ratio of injected fake companions                                                     
    snr_level = 10.0 * np.ones_like(noise)         
    
    thruput_arr = np.zeros((nbranch, noise.shape[0]))
    fc_map_all = np.zeros((nbranch*fc_rad_sep, array.shape[1], array.shape[2]))
    frame_fc_all = fc_map_all.copy()
    cube_fc_all = np.zeros((nbranch*fc_rad_sep, array.shape[0], array.shape[1], 
                            array.shape[2]))
    cy, cx = frame_center(array[0])

    for br in range(nbranch):
        for irad in range(fc_rad_sep):
            # contains companions separated by "fc_rad_sep * fwhm"
            radvec = vector_radd[irad::fc_rad_sep]                              
            cube_fc = array.copy()
            # filling map with small numbers
            fc_map = np.ones_like(array[0]) * min(noise) * 1e-6                 
            fcy = []
            fcx = []
            for i in range(radvec.shape[0]):
                cube_fc = inject_fcs_cube(cube_fc, psf_template, parangles,
                        snr_level[irad+i*fc_rad_sep] * noise[irad+i*fc_rad_sep],
                        pxscale, [radvec[i]], theta=br*angle_branch,
                        verbose=False)
                y = cy + radvec[i] * np.sin(np.deg2rad(br*angle_branch))
                x = cx + radvec[i] * np.cos(np.deg2rad(br*angle_branch))
                fc_map = inject_fc_frame(fc_map, psf_template, y, x,
                        snr_level[irad+i*fc_rad_sep] * noise[irad+i*fc_rad_sep])
                fcy.append(y)
                fcx.append(x)
            
            if verbose: 
                msg2 = 'Fake companions injected in branch {:} (pattern {:}/{:})'
                print(msg2.format(br+1, irad+1, fc_rad_sep))
                timing(start_time)

            #*******************************************************************
            if 'cube' and 'angle_list' and 'verbose' in inspect.getargspec(algo).args:
                if 'fwhm' in inspect.getargspec(algo).args:
                    frame_fc = algo(cube=cube_fc, angle_list=parangles, 
                                    fwhm=fwhm, verbose=False, **algo_dict)
                else:
                    frame_fc = algo(cube=cube_fc, angle_list=parangles, 
                                    verbose=False, **algo_dict)               
                        
            if verbose:
                msg3 = 'Cube with fake companions processed with {:}'
                print(msg3.format(algo.func_name))
                timing(start_time)

            #*******************************************************************
            ratio = (frame_fc - frame_nofc) / fc_map
            thruput = aperture_flux(ratio, fcy, fcx, fwhm, ap_factor=1,
                                    mean=True, verbose=False)
                        
            if verbose:
                msg4 = 'Measured the annulus-wise throughput of {:}'
                print(msg4.format(algo.func_name))
                timing(start_time)
            
            thruput_arr[br, irad::fc_rad_sep] = thruput
            fc_map_all[br*fc_rad_sep+irad, :, :] = fc_map
            frame_fc_all[br*fc_rad_sep+irad, :, :] = frame_fc
            cube_fc_all[br*fc_rad_sep+irad, :, :, :] = cube_fc
            
    if verbose:
        print('Finished measuring the throughput in {:} branches'.format(nbranch))
        timing(start_time)
    
    if full_output:
        return (thruput_arr, noise, vector_radd, cube_fc_all, frame_fc_all, 
                frame_nofc, fc_map_all)
    else:
        return thruput_arr, vector_radd
    


def noise_per_annulus(array, separation, fwhm, verbose=False):
    """ Measures the noise as the standard deviation of apertures defined in
    each annulus with a given separation.
    
    Parameters
    ----------
    array : array_like
        Input frame.
    separation : float
        Separation in pixels of the centers of the annuli measured from the 
        center of the frame.
    fwhm : float
        FWHM in pixels.
    verbose : {False, True}, bool optional
        If True prints information.
    
    Returns
    -------
    noise : array_like
        Vector with the noise value per annulus.
    vector_radd : array_like
        Vector with the radial distances values.
    
    """
    if not array.ndim==2:
        raise TypeError('Input array is not a frame or 2d array.')
    
    def find_coords(rad, sep):
        npoints = (2*np.pi*rad)/sep
        ang_step = 360/npoints
        x = []
        y = []
        for i in range(int(npoints)): 
            newx = rad * np.cos(np.deg2rad(ang_step * i))
            newy = rad * np.sin(np.deg2rad(ang_step * i))
            x.append(newx)
            y.append(newy)
        return np.array(y), np.array(x)
    
    centery, centerx = frame_center(array)
    n_annuli = int(np.floor((centery)/separation))

    x = centerx
    y = centery
    noise = []
    vector_radd = []
    if verbose:  print('{} annuli'.format(n_annuli-1))
        
    for _ in range(n_annuli-1):
        y -= separation
        rad = dist(centery, centerx, y, x)
        if rad>=fwhm:
            yy, xx = find_coords(rad, sep=fwhm)
            yy += centery
            xx += centerx
                 
            fluxes = []
            apertures = photutils.CircularAperture((xx, yy), fwhm/2.)
            fluxes = photutils.aperture_photometry(array, apertures)
            fluxes = np.array(fluxes['aperture_sum'])
            
            noise_ann = np.std(fluxes)
            noise.append(noise_ann) 
            vector_radd.append(rad)
            if verbose:
                print('Radius(px) = {:} // Noise = {:.3f} '.format(rad, noise_ann))
     
    return np.array(noise), np.array(vector_radd)
    

def aperture_flux(array, yc, xc, fwhm, ap_factor=1, mean=False, verbose=False):
    """ Returns the sum of pixel values in a circular aperture centered on the
    input coordinates. The radius of the aperture is set as (ap_factor*fwhm)/2.
    
    Parameters
    ----------
    array : array_like
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
            ind = circle(y, x,  (ap_factor*fwhm)/2.)
            values = array[ind]
            obj_flux = np.mean(values)
        else:
            aper = photutils.CircularAperture((x, y), (ap_factor*fwhm)/2.)
            obj_flux = photutils.aperture_photometry(array, aper, method='exact')
            obj_flux = np.array(obj_flux['aperture_sum'])
        flux[i] = obj_flux
        
        if verbose:
            print('Coordinates of object {:} : ({:},{:})'.format(i, y, x))
            print('Object Flux = {:.2f}'.format(flux[i]))

    return flux
    
