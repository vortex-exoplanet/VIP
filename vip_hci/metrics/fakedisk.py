#! /usr/bin/env python

"""
Module with fake disk injection functions.
"""

__author__ = 'Julien Milli @ ESO, Valentin Christiaens @ ULg/UChile'
__all__ = ['cube_inject_fakedisk',
           'cube_inject_trace']

import numpy as np
from scipy import signal
from ..preproc import cube_derotate, frame_shift
from ..var import frame_center


def cube_inject_fakedisk(fakedisk, angle_list, psf=None, **rot_options):
    """
    Rotates an ADI cube to a common north given a vector with the corresponding
    parallactic angles for each frame of the sequence. By default bicubic
    interpolation is used (opencv).

    Parameters
    ----------
    fakedisk : numpy ndarray
        Input image of a fake disc
    angle_list : list
        Vector containing the parallactic angles.
    psf : (optional) the PSF to convolve the disk image with. It can be a
        small numpy.ndarray (we advise to use odd sizes to make sure the center
        s not shifted through the convolution). It forces normalization of the
        PSF to preserve the flux. It can also be a float representing
        the FWHM of the gaussian to be used for convolution.
    imlib : str, optional
        See the documentation of the ``vip_hci.preproc.frame_rotate`` function.
    interpolation : str, optional
        See the documentation of the ``vip_hci.preproc.frame_rotate`` function.
    cxy : tuple of int, optional
        Coordinates X,Y  of the point with respect to which the rotation will be
        performed. By default the rotation is done with respect to the center
        of the frames, as it is returned by the function
        vip_hci.var.frame_center.
    nproc : int, optional
        Whether to rotate the frames in the sequence in a multi-processing
        fashion. Only useful if the cube is significantly large (frame size and
        number of frames).
    border_mode : str, optional
        See the documentation of the ``vip_hci.preproc.frame_rotate`` function.
    rot_options: dictionary, optional
        Dictionary with optional keyword values for "nproc", "cxy", "imlib", 
        "interpolation, "border_mode", "mask_val",  "edge_blend", 
        "interp_zeros", "ker" (see documentation of 
        ``vip_hci.preproc.frame_rotate``)
        
    Returns
    -------
    fakedisk_cube : numpy ndarray
        Resulting cube with the fake disc inserted at the correct angles and
        convolved with the psf if a psf was provided.

    Notes
    -----
    .. code-block:: python

        import numpy as np

        fakedisk = np.zeros((200,200))
        fakedisk[:,99:101] = 1
        angle_list = np.arange(10)
        c = create_fakedisk_cube(fakedisk, angle_list, psf=None, imlib='opencv',
                                 interpolation='lanczos4',cxy=None, nproc=1,
                                 border_mode='constant')

    """
    if not fakedisk.ndim == 2:
        raise TypeError('Fakedisk is not a frame or a 2d array.')
    if not angle_list.ndim == 1:
        raise TypeError('Input parallactic angle is not a 1d array')
    nframes = len(angle_list)
    ny, nx = fakedisk.shape
    fakedisk_cube = np.repeat(fakedisk[np.newaxis, :, :], nframes, axis=0)
    fakedisk_cube = cube_derotate(fakedisk_cube, angle_list, **rot_options)

    if psf is not None:
        if isinstance(psf, np.ndarray):
            if psf.ndim != 2:
                raise TypeError('Input PSF is not a frame or 2d array.')
            if np.abs(np.sum(psf)-1) > 1e-4:
                print('Warning the PSF is not normalized to a total of 1. '
                      'Normalization was forced.')
                psf = psf/np.sum(psf)
        elif isinstance(psf, (int, float)):
            # assumes psf is equal to the FWHM of the PSF. We create a synthetic
            # PSF in that case
            # with a size of 2 times the FWHM.
            psf_size = 2*int(np.round(psf))+1  # to make sure this is odd.
            xarrray, yarray = np.meshgrid(np.arange(-(psf_size//2),
                                                    psf_size//2+1),
                                          np.arange(-(psf_size//2),
                                                    psf_size//2+1))
            d = np.sqrt(xarrray**2+yarray**2)
            sigma = psf/(2*np.sqrt(2*np.log(2)))
            psf = np.exp(-(d**2 / (2.0*sigma**2)))
            psf = psf/np.sum(psf)
        else:
            raise TypeError('The type of the psf is unknown. '
                            'create_fakedisk_cube accepts ndarray, int or '
                            'float.')
        for i in range(nframes):
            # fakedisk_cube[i, :, :] = signal.convolve2d(fakedisk_cube[i, :, :],
            #                                            psf, mode='same')
            # much faster
            fakedisk_cube[i, :, :] = signal.fftconvolve(fakedisk_cube[i, :, :],\
                                                        psf, mode='same')
    return fakedisk_cube


def cube_inject_trace(array, psf_template, angle_list, flevel, rad_dists, theta, 
                      plsc=0.01225, n_branches=1, imlib='vip-fft', 
                      interpolation='lanczos4', verbose=True):
    """ Injects fake companions along a trace, such as a spiral. The trace is 
    provided by 2 arrays corresponding to the polar coordinates where the 
    companions will be located in the final derotated frame.
    Note: for a continuous-looking trace, and for an easier scaling using 
    parameter 'flevel', it is recommended to separate the points of the trace
    by a distance of FWHM/2.
    
    Parameters
    ----------
    array : numpy ndarray
        Input 3D cube in which the extended feature is injected.
    psf_template : numpy ndarray 
        2d array with the normalized psf template. It should have an odd shape.
        It is recommended to run the function psf_norm to get a proper PSF
        template.
    flevel : float
        Flux at which the fake companions are injected into the cube along the 
        trace.
    rad_dists : list or array 1d
        Vector of radial distances where the trace is to be injected.
    theta : list or array 1d
        Vector of angles (deg) where the trace is to be injected (trigonometric
        angles, NOT PA East from North).
    plsc : float, opt
        Value of the plate scale in arcsec/pixel (optional, will only be used 
        if verbose is True).
    n_branches : int, optional
        Number of azimutal branches on which the trace is injected.
    imlib : {'opencv', 'ndimage-fourier', 'ndimage-interp', 'vip-fft'}, str opt
        Library or method used for image operations (shifts).
    interpolation: str, optional
        Interpolation method. Check documentation of the function
        ``vip_hci.preproc.frame_shift``.
    verbose : {True, False}, bool optional
        If True prints out additional information. 
    
    Returns
    -------
    array_out : numpy ndarray
        Output array with the injected fake companions.
        
    """
    if not array.ndim==3: 
        raise TypeError('Array is not a cube or 3d array')
    
    ceny, cenx = frame_center(array[0])
    ceny = int(ceny)
    cenx = int(cenx)
    rad_dists = np.array(rad_dists)
    if not rad_dists[-1]<array[0].shape[0]/2.:
        msg = 'rad_dists last location is at the border or outside of the field'
        raise ValueError(msg)
    
    size_fc = psf_template.shape[0]
    nframes, ny, nx = array.shape
    n_fc_rad = rad_dists.shape[0]

    w = int(np.floor(size_fc/2.))

    array_out = np.zeros_like(array)
    for fr in range(nframes):
        tmp = np.zeros_like(array[0])
        for branch in range(n_branches):
            for i in range(n_fc_rad):
                fc_fr = np.zeros_like(array[0], dtype=np.float64)
                ang = (branch * 2 * np.pi / n_branches) + np.deg2rad(theta[i])
                rad = rad_dists[i]                                         
                y = rad * np.sin(ang - np.deg2rad(angle_list[fr]))
                x = rad * np.cos(ang - np.deg2rad(angle_list[fr]))
                # deal with exceptions
                y0_fr = max(0,ceny+(int(y)-w))
                y0_psf = max(0,-(ceny+int(y)-w))
                x0_fr = max(0,cenx+(int(x)-w))
                x0_psf = max(0,-(cenx+int(x)-w))
                yn_fr = min(ny,ceny+(int(y)+w+1))     
                yn_psf = min(size_fc,size_fc-(ceny+(int(y)+w+1)-ny))        
                xn_fr = min(nx,cenx+(int(x)+w+1))     
                xn_psf = min(size_fc,size_fc-(cenx+(int(x)+w+1)-nx))     
                if x > 0:
                    mod_x = x%1.
                else:
                    mod_x = (x%1.)-1
                if y > 0:
                    mod_y = y%1.
                else:
                    mod_y = (y%1.)-1
                try:
                    psf_tmp = flevel*psf_template[y0_psf:yn_psf,x0_psf:xn_psf]
                    fc_fr[y0_fr:yn_fr, x0_fr:xn_fr] = frame_shift(psf_tmp, 
                                                                  mod_y, mod_x, 
                                                                  imlib=imlib,
                                                                  interpolation=interpolation)
                except:
                    raise TypeError('Problem with the coordinates of the trace')
                tmp += fc_fr
        array_out[fr] = array[fr] + tmp
    
    if verbose:
        for branch in range(n_branches):
            print('Branch '+str(branch+1)+':')
            for i in range(n_fc_rad):
                ang = (branch * 2 * np.pi / n_branches) + np.deg2rad(theta[i])
                posy = rad_dists[i] * np.sin(ang) + ceny
                posx = rad_dists[i] * np.cos(ang) + cenx
                rad_arcs = rad_dists[i]*plsc
                msg ='\t(X,Y)=({:.2f}, {:.2f}) at {:.2f} arcsec ({:.2f} pxs)'
                print(msg.format(posx, posy, rad_arcs, rad_dists[i]))
        
    return array_out