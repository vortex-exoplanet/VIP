#! /usr/bin/env python

"""
Module with various fits handling functions.
"""

__author__ = 'C. Gomez @ ULg'
__all__ = ['open_fits',
           'open_adicube',
           'info_fits',
           'write_fits',
           'verify_fits',
           'display_fits_ds9',
           'display_array_ds9']

import glob
import os
import numpy as np
from astropy.io import fits as fits
from ..exlib.ds9 import DS9Win


def open_fits(fitsfilename, n=0, header=False, verbose=True):
    """Loads a fits file into a memory as numpy array.
    
    Parameters
    ----------
    fitsfilename : string
        Name of the fits file.
    n : int
        It chooses which HDU to open. Default is the first one.
    header : {False, True}, bool optional
        Whether to return the header along with the data or not.
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
    hdulist = fits.open(fitsfilename, memmap=True)
    data = hdulist[n].data
    #if data.dtype.name == 'float32':  data = np.array(data, dtype='float64')
    
    if header:
        header = hdulist[0].header
        if verbose:
            if len(data.shape)==1:
                msg = "\nFits HDU:{:} data and header successfully loaded. Data"+\
                    " shape: [{:}]"
                print msg.format(n, data.shape[0])
            if len(data.shape)==2:  
                msg = "\nFits HDU:{:} data and header successfully loaded. Data"+\
                    " shape: [{:},{:}]"
                print msg.format(n, data.shape[0],data.shape[1])
            if len(data.shape)==3:
                msg = "\nFits HDU:{:} data and header successfully loaded. Data"+\
                    " shape: [{:},{:},{:}]"
                print msg.format(n, data.shape[0],data.shape[1],data.shape[2])
        return data, header 
    else:
        if verbose:
            if len(data.shape)==1:
                msg = "\nFits HDU:{:} data successfully loaded. Data"+\
                    " shape: [{:}]"
                print msg.format(n, data.shape[0])
            if len(data.shape)==2:  
                msg = "\nFits HDU:{:} data successfully loaded. Data"+\
                    " shape: [{:},{:}]"
                print msg.format(n, data.shape[0],data.shape[1])
            if len(data.shape)==3:
                msg = "\nFits HDU:{:} data successfully loaded. Data"+\
                    " shape: [{:},{:},{:}]"
                print msg.format(n, data.shape[0],data.shape[1],data.shape[2])
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
    hdulist = fits.open(fitsfilename, memmap=True)
    data = hdulist[0].data
    if not data.ndim ==3:
        raise TypeError('Input fits file does not contain a cube or 3d array.')
    parangles = hdulist[1].data 
    msg1 = "\nFits HDU:{:} data successfully loaded. Data shape: [{:},{:},{:}]"
    msg2 = "\nFits HDU:{:} data successfully loaded. Data shape: [{:}]"
    print msg1.format(0, data.shape[0],data.shape[1],data.shape[2])
    print msg2.format(1, parangles.shape[0])
    
    return data, parangles


def info_fits(fitsfilename):
    """Prints the information about a fits file. 
    """
    hdulist = fits.open(fitsfilename, memmap=True)
    hdulist.info()
         
         
def display_fits_ds9(fitsfilename):
    """Displays fits file in ds9 (which should be already installed in your
    system along with XPA).
    """
    ds9 = DS9Win()
    ds9.showFITSFile(fitsfilename)
         
         
def display_array_ds9(*args):
    """ Displays arrays listed in args in ds9 (which should be already installed 
    in your system along with XPA).
    """
    ds9 = DS9Win()
    ds9.xpaset('frame delete all')
    ds9.xpaset('frame new')
    ds9.xpaset('tile')
    for i, array in enumerate(args):
        if i==0: 
            ds9.showArray(array)
        else:    
            ds9.xpaset('frame new')
            ds9.showArray(array)

         
def verify_fits(fits):
    """Verifies "the FITS standard" of a fits file or list of fits.
    """
    if isinstance(fits, list):
        for ffile in fits:
            f = fits.open(ffile)
            f.verify()
    else:
        f = fits.open(fits)
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
            print "\nFits file successfully overwritten"
        fits.writeto(filename, array, header)                       
    else:
        fits.writeto(filename, array, header)                       
        if verbose:
            print "\nFits file successfully saved"    
    
    