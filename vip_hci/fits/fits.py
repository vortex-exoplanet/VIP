#! /usr/bin/env python

"""
Module with various fits handling functions.
"""
from __future__ import print_function

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
    fitsfilename : string
        Name of the fits file.
    n : int
        It chooses which HDU to open. Default is the first one.
    header : {False, True}, bool optional
        Whether to return the header along with the data or not.
    precision : numpy dtype
        Float precision, by default np.float32 or single precision float.
    ignore_missing_end : {False, True}, bool optional
        Allows to open fits files with a header missing END card.
    verbose : {True, False}, bool optional
        If True prints message of completion.
    
    Returns
    -------
    data : array_like
        Array containing the frames of the fits-cube.
    If header is True:
    header : dictionary
        Dictionary containing the fits header.
    """
    if not fitsfilename.endswith('.fits'):
        fitsfilename = str(fitsfilename+'.fits')
    hdulist = ap_fits.open(fitsfilename, memmap=True,
                           ignore_missing_end=ignore_missing_end)
    data = hdulist[n].data
    data = np.array(data, dtype=precision)
    
    if header:
        header = hdulist[0].header
        if verbose:
            if len(data.shape)==1:
                msg = "\nFits HDU:{:} data and header successfully loaded. Data"
                msg += " shape: [{:}]"
                print(msg.format(n, data.shape[0]))
            if len(data.shape)==2:  
                msg = "\nFits HDU:{:} data and header successfully loaded. Data"
                msg += " shape: [{:},{:}]"
                print(msg.format(n, data.shape[0],data.shape[1]))
            if len(data.shape)==3:
                msg = "\nFits HDU:{:} data and header successfully loaded. Data"
                msg += " shape: [{:},{:},{:}]"
                print(msg.format(n, data.shape[0],data.shape[1],data.shape[2]))
            if len(data.shape)==4:
                msg = "\nFits HDU:{:} data and header successfully loaded. Data"
                msg += " shape: [{:},{:},{:},{:}]"
                print(msg.format(n, data.shape[0], data.shape[1], data.shape[2],
                                 data.shape[3]))
        hdulist.close()
        return data, header 
    else:
        if verbose:
            if len(data.shape)==1:
                msg = "\nFits HDU:{:} data successfully loaded. Data"
                msg += " shape: [{:}]"
                print(msg.format(n, data.shape[0]))
            if len(data.shape)==2:  
                msg = "\nFits HDU:{:} data successfully loaded. Data"
                msg += " shape: [{:},{:}]"
                print(msg.format(n, data.shape[0],data.shape[1]))
            if len(data.shape)==3:
                msg = "\nFits HDU:{:} data successfully loaded. Data"
                msg += " shape: [{:},{:},{:}]"
                print(msg.format(n, data.shape[0],data.shape[1],data.shape[2]))
            if len(data.shape)==4:
                msg = "\nFits HDU:{:} data successfully loaded. Data"
                msg += " shape: [{:},{:},{:},{:}]"
                print(msg.format(n, data.shape[0], data.shape[1], data.shape[2],
                                 data.shape[3]))
        hdulist.close()
        return data        


def open_adicube(fitsfilename, verbose=True):
    """ Opens an ADI cube with the parallactic angles appended (see function
    append_par_angles).
    
    Parameters
    ----------
    fitsfilename : string
        Name of the fits file.
    verbose : {True, False}, bool optional
        If True prints message.
        
    Returns
    -------
    data : array_like
        Array containing the frames of the fits-cube.
    parangles : array_like
        1d array containing the corresponding parallactic angles.
          
    """
    if not fitsfilename.endswith('.fits'):
        fitsfilename = str(fitsfilename+'.fits')
    hdulist = ap_fits.open(fitsfilename, memmap=True)
    data = hdulist[0].data
    if not data.ndim ==3:
        raise TypeError('Input fits file does not contain a cube or 3d array.')
    parangles = hdulist[1].data 
    if verbose:
        msg1 = "\nFits HDU:{:} data successfully loaded. Data shape: [{:},{:},{:}]"
        msg2 = "\nFits HDU:{:} data successfully loaded. Data shape: [{:}]"
        print(msg1.format(0, data.shape[0],data.shape[1],data.shape[2]))
        print(msg2.format(1, parangles.shape[0]))
    
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
    """
    hdulist = ap_fits.open(fitsfilename, memmap=True)
    hdulist.info()

         
def verify_fits(fitspath):
    """Verifies "the FITS standard" of a fits file or list of fits.

    Parameters
    ----------
    fitspath : string
        Path to the fits file or list with fits filename paths.

    """
    if isinstance(fitspath, list):
        for ffile in fitspath:
            f = ap_fits.open(ffile)
            f.verify()
    else:
        f = ap_fits.open(fitspath)
        f.verify()
    
    
def write_fits(filename, array, header=None, dtype32=True, verbose=True):
    """Writes array and header into FTIS file, if there is a previous file with
    the same filename then it's replaced.
    
    Parameters
    ----------
    filename : string
        Filename of the fits file to be written.
    array : array_like
        Array to be written into a fits file.
    header : array_like, optional
        Array with header. 
    dtype32 : {True, False}, bool optional
        If True the array is casted as a float32
    verbose : {True, False}, bool optional
        If True prints message.

    """
    if dtype32:
        array = array.astype('float32', copy=False)
    if os.path.exists(filename):
        os.remove(filename)                                     
        if verbose:
            print("\nFits file successfully overwritten")
        ap_fits.writeto(filename, array, header)
    else:
        ap_fits.writeto(filename, array, header)
        if verbose:
            print("\nFits file successfully saved")    


def append_extension(filename, array):
    """Appends an extension to fits file. 
    """
    ap_fits.append(filename, array)
    print("\nFits extension appended")
        
        
    