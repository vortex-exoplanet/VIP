#! /usr/bin/env python

"""
Module with various fits handling functions.
"""
from __future__ import division, print_function

__author__ = 'Carlos Alberto Gomez Gonzalez'
__all__ = ['open_fits',
           'open_adicube',
           'info_fits',
           'write_fits',
           'append_extension',
           'verify_fits',
           'byteswap_array']


import os
import numpy as np
from astropy.io import fits as ap_fits


def open_fits(fitsfilename, n=0, header=False, ignore_missing_end=False,
              precision=np.float32, verbose=True):
    """Loads a fits file into a memory as numpy array.
    
    Parameters
    ----------
    fitsfilename : string or pathlib.Path
        Name of the fits file or ``pathlib.Path`` object
    n : int
        It chooses which HDU to open. Default is the first one.
    header : bool, optional
        Whether to return the header along with the data or not.
    precision : numpy dtype
        Float precision, by default np.float32 or single precision float.
    ignore_missing_end : {False, True}, bool optional
        Allows to open fits files with a header missing END card.
    verbose : bool, optional
        If True prints message of completion.
    
    Returns
    -------
    data : array_like
        Array containing the frames of the fits-cube.
    If header is True:
    header : dict
        Dictionary containing the fits header.
    """
    fitsfilename = str(fitsfilename)
    if not fitsfilename.endswith('.fits'):
        fitsfilename = fitsfilename+'.fits'

    with ap_fits.open(fitsfilename, memmap=True,
                      ignore_missing_end=ignore_missing_end) as hdulist:
        data = hdulist[n].data
        data = np.array(data, dtype=precision)

        if header:
            header = hdulist[0].header
            if verbose:
                print("Fits HDU-{} data and header successfully loaded. "
                      "Data shape: {}".format(n, data.shape))
            return data, header
        else:
            if verbose:
                print("Fits HDU-{} data successfully loaded. "
                      "Data shape: {}".format(n, data.shape))
            return data


def open_adicube(fitsfilename, verbose=True):
    """ Opens an ADI cube with the parallactic angles appended (see function
    append_par_angles).
    
    Parameters
    ----------
    fitsfilename : string or pathlib.Path
        Name of the fits file or ``pathlib.Path`` object
    verbose : {True, False}, bool optional
        If True prints message.
        
    Returns
    -------
    data : array_like
        Array containing the frames of the fits-cube.
    parangles : array_like
        1d array containing the corresponding parallactic angles.
          
    """
    fitsfilename = str(fitsfilename)
    if not fitsfilename.endswith('.fits'):
        fitsfilename = fitsfilename+'.fits'
    with ap_fits.open(fitsfilename, memmap=True) as hdulist:
        data = hdulist[0].data
        parangles = hdulist[1].data
    
    if data.ndim != 3:
        raise TypeError('Input fits file does not contain a cube or 3d array.')

    if verbose:
        print("Fits HDU-0 data successfully loaded. "
              "Data shape: {}".format(data.shape))
        print("Fits HDU-1 data successfully loaded. "
              "Data shape: {}".format(parangles.shape))
    
    return data, parangles


def byteswap_array(array):
    """ FITS files are stored in big-endian byte order. All modern CPUs are 
    little-endian byte order, so at some point you have to byteswap the data. 
    Some FITS readers (cfitsio, the fitsio python module) do the byteswap when 
    reading the data from disk to memory, so we get numpy arrays in native 
    (little-endian) byte order. Unfortunately, astropy.io.fits does not 
    byteswap for us, and we get numpy arrays in non-native byte order. 
    However, most of the time we never notice this because when you do any 
    numpy operations on such arrays, numpy uses an intermediate buffer to 
    byteswap the array behind the scenes and returns the result as a native 
    byte order array. Some operations require the data to be byteswaped 
    before and will complain about it. This function will help in this cases.
    
    Parameters
    ----------
    array : array_like
        2d input array.
        
    Returns
    -------
    array_out : array_like
        2d resulting array after the byteswap operation.


    Notes
    -----
    http://docs.scipy.org/doc/numpy-1.10.1/user/basics.byteswapping.html

    """
    array_out = array.byteswap().newbyteorder()
    return array_out


def info_fits(fitsfilename):
    """Prints the information about a fits file. 

    Parameters
    ----------
    fitsfilename : str
        Path to the fits file.
    """
    with ap_fits.open(fitsfilename, memmap=True) as hdulist:
        hdulist.info()

         
def verify_fits(fitsfilename):
    """Verifies "the FITS standard" of a fits file or list of fits.

    Parameters
    ----------
    fitsfilename : string or list
        Path to the fits file or list with fits filename paths.
    """
    if isinstance(fitsfilename, list):
        for ffile in fitsfilename:
            with ap_fits.open(ffile) as f:
                f.verify()
    else:
        with ap_fits.open(fitsfilename) as f:
            f.verify()
    
    
def write_fits(fitsfilename, array, header=None, precision=np.float32,
               verbose=True):
    """Writes array and header into FTIS file, if there is a previous file with
    the same filename then it's replaced.
    
    Parameters
    ----------
    fitsfilename : string
        Full path of the fits file to be written.
    array : array_like
        Array to be written into a fits file.
    header : array_like, optional
        Array with header. 
    precision : numpy dtype, optional
        Float precision, by default np.float32 or single precision float.
    verbose : bool, optional
        If True prints message.

    """
    array = array.astype(precision, copy=False)
    if os.path.exists(fitsfilename):
        os.remove(fitsfilename)                                     
        ap_fits.writeto(fitsfilename, array, header)
        if verbose:
            print("Fits file successfully overwritten")
    else:
        ap_fits.writeto(fitsfilename, array, header)
        if verbose:
            print("Fits file successfully saved")


def append_extension(fitsfilename, array, verbose=True):
    """Appends an extension to fits file. 

    Parameters
    ----------
    fitsfilename : str
        Path to the fits file.
    array : array_like
        Data to append.
    verbose : bool, optional
        Print success message.
    """
    ap_fits.append(fitsfilename, array)
    if verbose:
        print("Fits extension appended")
        
        
    