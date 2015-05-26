#! /usr/bin/env python

"""
Module with contrast curve generation function.
"""

from __future__ import division
from __future__ import print_function

__author__ = 'C. Gomez, O. Absil @ ULg'
__all__ = ['noise_per_annulus',
           'throughput',
           'throughput_single_branch']

import numpy as np
import photutils
from scipy.interpolate import interp1d
from scipy import stats
from skimage.draw import circle
from matplotlib import pyplot as plt
from .fakecomp import inject_fcs_cube, inject_fc_frame
from ..conf import timeInit, timing, VLT_NACO, LBT
from ..var import frame_center, dist
from ..calib import frame_crop
from ..pca import annular_pca, pca, subannular_pca


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
    separation = np.round(separation)     
    n_annuli = int(np.floor((centery)/separation))    
    
    x = centerx
    y = centery
    noise = []
    vector_radd = []    
    for _ in range(n_annuli-1):
        y -= separation
        rad = dist(centery, centerx, y, x)
        yy, xx = find_coords(rad, sep=fwhm)
        yy += centery
        xx += centerx
             
        fluxes = []
        apertures = photutils.CircularAperture((xx, yy), fwhm/2.)
        fluxes = photutils.aperture_photometry(array, apertures)
        fluxes = np.array(fluxes['aperture_sum'])
        
        noise_ann = np.std(fluxes)
        noise.append(noise_ann) 
        vector_radd.append(int(round(rad)))
        if verbose:
            print('Radius(px) = {:} // Noise = {:.3f} '.format(int(round(rad)), 
                                                             noise_ann))
     
    return np.array(noise), np.array(vector_radd)


def throughput(array, parangles, psf_template, fwhm, n_comp, algo='spca',
               nbranch=3, instrument='naco27', student=True, full_output=False,
               **kwargs):
    """ Measures the throughput for chosen and input dataset. The final 
    throughput is the average of the same procedure measured in nbranch 
    azimutally equidistant branches.
    
    Parameters
    ----------
    array : array_like
        The input cube without fake companions.
    parangles : array_like
        Vector with the parallactic angles.
    psf_template : array_like
        Frame with the psf template for the fake companion(s).
        PSF must be centered in array. Normalization is done internally.
    fwhm : float
        FWHM in pixels.
    n_comp : int
        Number of principal components if PCA or other subspace projection 
        technique is used.
    instrument : {'naco27', 'lmircam'}, string optional
        The instrument that acquired the dataset.
    algo : {'spca', 'pca', 'subspca', 'fd'}, string optional
        The post-processing algorithm.
    nbranch : int
        Number of branches on which to inject fakes companions. Each branch
        is tested individually.
    student {True, False}, bool optional
        If True uses Student correction to inject fake companion.
    full_output : {False, True}, bool optional
        If True returns intermediate arrays.
    **kwargs
        Any other valid parameter of the pca algorithms can be passed here.
    
    Returns
    -------
    throughput : array_like
        2d array whose rows are the annulus-wise throughput values for each 
        branch.

    """
    if not array.ndim == 3:
        raise TypeError('The input array is not a cube.')
    if not array.shape[0] == parangles.shape[0]:
        raise TypeError('Input vector or parallactic angles has wrong length.')
    if not psf_template.ndim==2:
        raise TypeError('Template PSF has wrong shape.')
    if not psf_template.shape[0]%2==0:
        psf_template = frame_crop(psf_template, psf_template.shape[0]-1)
    
    if instrument == 'naco27':
        plsc = VLT_NACO['plsc']
    elif instrument == 'lmircam':
        plsc = LBT['plsc']
    else:
        raise TypeError('Instrument not recognized.')

    start_time = timeInit()

    #***************************************************************************    
    if algo=='spca':
        function = annular_pca
        frame_nofc = function(array, angle_list=parangles, fwhm=fwhm, 
                              ncomp=n_comp, verbose=False, **kwargs)
    elif algo=='pca':
        function = pca
        frame_nofc = function(array, angle_list=parangles, ncomp=n_comp, 
                              verbose=False, **kwargs)
    
    elif algo=='subspca':
        function = subannular_pca
        frame_nofc = function(array, angle_list=parangles, fwhm=fwhm, 
                              ncomp=n_comp, verbose=False, **kwargs) 

    else:
        raise TypeError('Algorithm not recognized.')
    print('Cube without fake companions processed with {:}.'.format(function.\
                                                                    func_name))
    timing(start_time)
    
    #***************************************************************************
    # Compute noise in concentric annuli
    noise, vector_radd = noise_per_annulus(frame_nofc, fwhm, fwhm, verbose=False)
    print('Measured annulus-wise noise in resulting frame.')
    timing(start_time)
    
    #***************************************************************************
    # Prepare PSF template and normalize it so that flux in aperture = 1
    aperture = photutils.CircularAperture((psf_template.shape[0]/2,
                                           psf_template.shape[1]/2), fwhm/2.)
    ap_phot = photutils.aperture_photometry(psf_template, aperture, 
                                                  method='exact')
    psf_template /= np.array(ap_phot['aperture_sum']) 

    # Initialize the fake companions
    angle_branch = 360.0/nbranch
    fc_rad_sep = 3                                                              # radial separation between fake companions in terms of FWHM (must be integer)
    snr_level = 7.0 * np.ones_like(noise)                                       # signal-to-noise ratio of injected fake companions
    if student:
        snr_level = stats.t.ppf(stats.norm.cdf(snr_level), 
                                np.floor(vector_radd/fwhm*2*np.pi)) * \
                                np.sqrt(1 + 1 / (np.floor(vector_radd/fwhm*2*np.pi)-1))

    thruput_arr = np.zeros((nbranch, noise.shape[0]))
    fc_map_all = np.zeros((nbranch*fc_rad_sep, array.shape[1], array.shape[2]))
    frame_fc_all = fc_map_all.copy()
    cube_fc_all = np.zeros((nbranch*fc_rad_sep, array.shape[0], array.shape[1], 
                            array.shape[2]))
    cy, cx = frame_center(array[0])

    for br in range(nbranch):
        for irad in range(fc_rad_sep):
            radvec = vector_radd[irad::fc_rad_sep]                              # contains companions separated by "fc_rad_sep * fwhm"
            cube_fc = array.copy()
            fc_map = np.ones_like(array[0]) * min(noise) * 1e-6                 # fill map with small numbers
            fcy = []
            fcx = []
            for i in range(radvec.shape[0]):
                cube_fc = inject_fcs_cube(cube_fc, psf_template, parangles,
                                          snr_level[irad+i*fc_rad_sep] * noise[irad+i*fc_rad_sep],
                                          plsc, [radvec[i]*plsc], theta=br*angle_branch)
                y = cy + radvec[i] * np.sin(np.deg2rad(br*angle_branch))
                x = cx + radvec[i] * np.cos(np.deg2rad(br*angle_branch))
                fc_map = inject_fc_frame(fc_map, psf_template, y, x,
                                         snr_level[irad+i*fc_rad_sep] * noise[irad+i*fc_rad_sep])
                fcy.append(y)
                fcx.append(x)
            print('Fake companions injected in branch {:} (pattern {:}/{:}).'.format(br+1, irad+1, fc_rad_sep))
            timing(start_time)

            #***********************************************************************
            if algo=='spca':
                frame_fc = function(cube_fc, angle_list=parangles, fwhm=fwhm,
                                    ncomp=n_comp, verbose=False, **kwargs)
            elif algo=='pca':
                frame_fc = function(cube_fc, angle_list=parangles, ncomp=n_comp,
                                    verbose=False, **kwargs)
            
            elif algo=='subspca':
                frame_fc = function(cube_fc, angle_list=parangles, fwhm=fwhm, 
                                      ncomp=n_comp, verbose=False, **kwargs)     
            
            else:
                raise TypeError('Algorithm not recognized.')
            print('Cube with fake companions processed with {:}.'.format(function.\
                                                                         func_name))
            timing(start_time)

            #***********************************************************************
            ratio = (frame_fc - frame_nofc) / fc_map
            thruput = aperture_flux(ratio, fcy, fcx, fwhm, ap_factor=0.5,
                                       mean=True, verbose=False)
            print('Measured the annulus-wise throughput of {:}.'.format(function.\
                                                                        func_name))
            timing(start_time)
            thruput_arr[br, irad::fc_rad_sep] = thruput
            fc_map_all[br*fc_rad_sep+irad, :, :] = fc_map
            frame_fc_all[br*fc_rad_sep+irad, :, :] = frame_fc
            cube_fc_all[br*fc_rad_sep+irad, :, :, :] = cube_fc

    print('Finished measuring the throughput in {:} branches.'.format(nbranch))
    timing(start_time)
    
    if full_output:
        return thruput_arr, noise, vector_radd, cube_fc_all, frame_fc_all, frame_nofc, fc_map_all
    else:
        return thruput_arr
    
    
def throughput_single_branch(array, parangles, psf_template, fwhm, n_comp, 
                           algo='spca', instrument='naco27', full_output=False):
    """ Measures the throughput for chosen and input dataset.
    
    Parameters
    ----------
    array : array_like
        The input cube without fake companions.
    parangles : array_like
        Vector with the parallactic angles.
    psf_template : array_like
        Frame with the psf template for the fake companion(s).
    fwhm : float
        FWHM in pixels.
    n_comp : int
        Number of principal components if PCA or other subspace projection 
        technique is used.
    instrument : {'naco27', 'lmircam'}, string optional
        The instrument that acquired the dataset.
    algo : {'spca', 'pca'}, string optional
        The post-processing algorithm.
    full_output : {False, True}, bool optional
        If True returns all the intermidiate arrays. If False only the 
        throughput.
        
    Returns
    -------
    Depends on argument full_output. If False:
    throughput : array_like
        Vector with the throughput value for each considered annulus.
    
    If True it returns these intermediate arrays:
    cube_fc , frame_fc, frame_nofc, fc_map, ratio
    
    """
    if not array.ndim == 3:
        raise TypeError('The input array is not a cube.')
    if not array.shape[0] == parangles.shape[0]:
        raise TypeError('Input vector or parallactic angles has wrong length.')
    if not psf_template.ndim==2:
        raise TypeError('Fake companion array has wrong shape.')
    if not psf_template.shape[0]%2==0:
        psf_template = frame_crop(psf_template, psf_template.shape[0]-1)
    
    start_time = timeInit()
    #***************************************************************************    
    if algo=='spca':
        function = annular_pca
        frame_nofc = function(array, angle_list=parangles, fwhm=fwhm, 
                              ncomp=n_comp, verbose=False)
    elif algo=='pca':
        function = pca
        frame_nofc = function(array, angle_list=parangles, ncomp=n_comp, 
                              verbose=False)
    else:
        raise TypeError('Algorithm not recognized.')
    print('Cube without fake companions processed with {:}.'.format(function.\
                                                                    func_name))
    timing(start_time)
    
    #***************************************************************************    
    noise, vector_radd = noise_per_annulus(frame_nofc, 3*fwhm, fwhm, verbose=False)
    print('Measured annulus-wise noise in resulting frame.')
    timing(start_time)
    
    #***************************************************************************
    if instrument == 'naco27':
        plsc = VLT_NACO['plsc']
    elif instrument == 'lmircam':
        plsc = LBT['plsc']
    else:
        raise TypeError('Instrument not recognized.')
    cube_fc = array.copy()
    psf_template[np.where(psf_template < 0.01)] = 0
    fc_map = np.ones_like(array[0])
    cy, cx = frame_center(fc_map)
    fcy = []
    fcx = []
    for i in range(noise.shape[0]):
        cube_fc = inject_fcs_cube(cube_fc, psf_template, parangles, 20*noise[i], 
                                  plsc, [vector_radd[i]*plsc])
        fc_map = inject_fc_frame(fc_map, psf_template, cy, cx+vector_radd[i], 
                                 20*noise[i])
        fcy.append(cy)
        fcx.append(cx+vector_radd[i])
    print('Fake companions injected in frame.')
    timing(start_time)

    #***************************************************************************
    if algo=='spca':
        frame_fc = function(cube_fc, angle_list=parangles, fwhm=fwhm, 
                            ncomp=n_comp, verbose=False)
    elif algo=='pca':
        frame_fc = function(cube_fc, angle_list=parangles, ncomp=n_comp, 
                            verbose=False)
    else:
        raise TypeError('Algorithm not recognized.')
    print('Cube with fake companions processed with {:}.'.format(function.\
                                                                    func_name))
    timing(start_time)

    #***************************************************************************
    ratio = (frame_fc - frame_nofc)/ fc_map
    throughput = aperture_flux(ratio, fcy, fcx, fwhm, ap_factor=0.5, mean=True,
                               verbose=False)
    
    print('Measured the annulus-wise throughput of {:}.'.format(function.\
                                                                    func_name))
    timing(start_time)
    
    if full_output:
        print(throughput)
        return cube_fc, frame_fc, frame_nofc, fc_map, ratio
    else:
        return throughput
    

def aperture_flux(array, yc, xc, fwhm, ap_factor=0.6, mean=False, verbose=False):
    """ Returns the sum of pixel values in a circular aperture centered on the
    input coordinates. 
    
    Parameters
    ----------
    array : array_like
        Input frame.
    yc, xc : list or 1d arrays
        List of y and x coordinates of sources.
    
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
    
