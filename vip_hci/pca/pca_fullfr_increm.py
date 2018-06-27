#! /usr/bin/env python

"""
Incremental full-frame PCA for big (larger than available memory) cubes.
"""

from __future__ import division, print_function

__author__ = 'Carlos Alberto Gomez Gonzalez'
__all__ = ['pca_incremental']

import numpy as np
from astropy.io import fits
from sklearn.decomposition import IncrementalPCA
from ..preproc import cube_derotate, cube_collapse
from ..conf import timing, time_ini, get_available_memory
from ..var import prepare_matrix, reshape_matrix


def pca_incremental(cubepath, angle_list=None, n=0, batch_size=None, 
                    batch_ratio=0.1, ncomp=10, imlib='opencv',
                    interpolation='lanczos4', collapse='median',
                    verbose=True, full_output=False):
    """ Computes the full-frame PCA-ADI algorithm in batches, for processing 
    fits files larger than the available system memory. It uses the incremental 
    PCA algorithm from scikit-learn. 
    
    Parameters
    ----------
    cubepath : str
        String with the path to the fits file to be opened in memmap mode.
    angle_list : array_like, 1d
        Corresponding parallactic angle for each frame. If None the parallactic
        angles are obtained from the same fits file (extension). 
    n : int optional
        The index of the HDULIST contaning the data/cube.
    batch_size : int optional
        The number of frames in each batch. If None the size of the batch is 
        computed wrt the available memory in the system.
    batch_ratio : float
        If batch_size is None, batch_ratio indicates the % of the available 
        memory that should be used by every batch.
    ncomp : int, optional
        How many PCs are used as a lower-dimensional subspace to project the
        target frames.
    imlib : str, optional
        See the documentation of the ``vip_hci.preproc.frame_rotate`` function.
    interpolation : str, optional
        See the documentation of the ``vip_hci.preproc.frame_rotate`` function.
    collapse : {'median', 'mean', 'sum', 'trimmean'}, str optional
        Sets the way of collapsing the frames for producing a final image.
    verbose : {True, False}, bool optional
        If True prints intermediate info and timing. 
    full_output: boolean, optional
        Whether to return the final median combined image only or with other 
        intermediate arrays.  
        
    Returns
    -------
    If full_output is True the algorithm returns the incremental PCA model of
    scikit-learn, the PCs reshaped into images, the median of the derotated 
    residuals for each batch, and the final frame. If full_output is False then
    the final frame is returned.
    
    """
    if verbose:  start = time_ini()
    if not isinstance(cubepath, str):
        raise TypeError('Cubepath must be a string with the full path of your '
                        'fits file')
      
    fitsfilename = cubepath
    hdulist = fits.open(fitsfilename, memmap=True)
    if not hdulist[n].data.ndim>2:
        raise TypeError('Input array is not a 3d or 4d array')
    
    n_frames = hdulist[n].data.shape[0]
    y = hdulist[n].data.shape[1]
    x = hdulist[n].data.shape[2]
    if angle_list is None:
        try:  
            angle_list = hdulist[n+1].data
        except:  
            raise RuntimeError('Parallactic angles were not provided')
    if not n_frames==angle_list.shape[0]:
        raise TypeError('Angle list vector has wrong length. It must equal the '
                        'number of frames in the cube.')
    
    ipca = IncrementalPCA(n_components=ncomp)
    
    if batch_size is None:
        aval_mem = get_available_memory(verbose)
        total_size = hdulist[n].data.nbytes
        batch_size = int(n_frames/(total_size/(batch_ratio*aval_mem)))
    
    if verbose:
        msg1 = "Cube with {} frames ({:.3f} GB)"
        print(msg1.format(n_frames, hdulist[n].data.nbytes/1e9))
        msg2 = "Batch size set to {} frames ({:.3f} GB)\n"
        print(msg2.format(batch_size, hdulist[n].data[:batch_size].nbytes/1e9))
                
    res = n_frames % batch_size
    for i in range(0, n_frames//batch_size):
        intini = i*batch_size
        intfin = (i+1)*batch_size
        batch = hdulist[n].data[intini:intfin]
        msg = 'Processing batch [{},{}] with shape {}'
        if verbose:
            print(msg.format(intini, intfin, batch.shape))
            print('Batch size in memory = {:.3f} MB'.format(batch.nbytes/1e6))
        matrix = prepare_matrix(batch, verbose=False)
        ipca.partial_fit(matrix)
    if res > 0:
        batch = hdulist[n].data[intfin:]
        msg = 'Processing batch [{},{}] with shape {}'
        if verbose:
            print(msg.format(intfin, n_frames, batch.shape))
            print('Batch size in memory = {:.3f} MB'.format(batch.nbytes/1e6))
        matrix = prepare_matrix(batch, verbose=False)
        ipca.partial_fit(matrix)
    
    if verbose:
        timing(start)
    
    V = ipca.components_
    mean = ipca.mean_.reshape(batch.shape[1], batch.shape[2])
    
    if verbose:
        print('\nReconstructing and obtaining residuals')
    medians = []
    for i in range(0, n_frames//batch_size):
        intini = i*batch_size
        intfin = (i+1)*batch_size
        batch = hdulist[n].data[intini:intfin]
        batch = batch - mean 
        matrix = prepare_matrix(batch, verbose=False)
        reconst = np.dot(np.dot(matrix, V.T), V)
        resid = matrix - reconst
        resid_der = cube_derotate(resid.reshape(batch.shape[0], 
                                                batch.shape[1],
                                                batch.shape[2]), 
                                  angle_list[intini:intfin], imlib=imlib,
                                  interpolation=interpolation)
        medians.append(cube_collapse(resid_der, mode=collapse))
    if res > 0:
        batch = hdulist[n].data[intfin:]
        batch = batch - mean
        matrix = prepare_matrix(batch, verbose=False)
        reconst = np.dot(np.dot(matrix, V.T), V)
        resid = matrix - reconst
        resid_der = cube_derotate(resid.reshape(batch.shape[0], 
                                                batch.shape[1],
                                                batch.shape[2]), 
                                  angle_list[intfin:], imlib=imlib,
                                  interpolation=interpolation)
        medians.append(cube_collapse(resid_der, mode=collapse))
    del matrix
    del batch

    medians = np.array(medians)
    frame = np.median(medians, axis=0)
    
    if verbose:
        timing(start)

    if full_output:
        pcs = reshape_matrix(V, y, x)
        return ipca, pcs, medians, frame
    else:
        return frame
    
    
