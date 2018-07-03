#! /usr/bin/env python

"""
Module with fake disk injection functions.
"""

from __future__ import division, print_function

__author__ = 'Julien Milli'
__all__ = ['create_fakedisk_cube']

import numpy as np
import photutils
from scipy import signal
from ..preproc import cube_derotate
from ..var import frame_center


def create_fakedisk_cube(fakedisk, angle_list, psf=None, imlib='opencv',
                         interpolation='lanczos4', cxy=None, nproc=1,
                         border_mode='constant'):
    """ Rotates an ADI cube to a common north given a vector with the 
    corresponding parallactic angles for each frame of the sequence. By default
    bicubic interpolation is used (opencv). 
    
    Parameters
    ----------
    fakedisk : array_like 
        Input image of a fake disc
    angle_list : list
        Vector containing the parallactic angles.
    psf :
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
        
    Returns
    -------
    fakedisk_cube : array_like
        Resulting cube with the fake disc inserted at the correct angles and 
        convolved with the psf if a psf was provided.

    Notes
    -----
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
    nframes=len(angle_list)
    ny, nx = fakedisk.shape
    fakedisk_cube = np.repeat(fakedisk[np.newaxis,:,:], nframes, axis=0)
    fakedisk_cube = cube_derotate(fakedisk_cube, angle_list, imlib=imlib,
                                  interpolation=interpolation, cxy=cxy,
                                  nproc=nproc, border_mode=border_mode)

    if psf is not None:
        if not psf.ndim == 2:
            raise TypeError('Input PSF is not a frame or 2d array.')
        if np.abs(np.sum(psf)-1) > 1e-4:
            print('Warning the PSF is not normalized to a total of 1.')
            print('Normalization was forced.')
            psf = psf/np.sum(psf)
        for i in range(nframes):
            fakedisk_cube[i,:,:] = signal.convolve2d(fakedisk_cube[i,:,:],psf,mode='same')             
    return fakedisk_cube


