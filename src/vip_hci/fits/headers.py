#! /usr/bin/env python
"""
Module with conversion utilities from dictionaries to headers, and reversely.
"""

__author__ = "Thomas Bedrine, Iain Hammond"
__all__ = ["dict_to_fitsheader",
           "fitsheader_to_dict",
           "open_header",
           "seeing_from_header"]

from os.path import isfile
from typing import Tuple

from astropy.io.fits.convenience import getheader
from astropy.io.fits.header import Header


def dict_to_fitsheader(initial_dict: dict) -> Header:
    """
    Convert a dictionnary into a fits Header object.

    Parameters
    ----------
    initial_dict: dict
        Dictionnary of parameters to convert to a Header object.

    Returns
    -------
    fits_header: Header
        Converted set of parameters.
    """
    fits_header = Header()
    for key, value in initial_dict.items():
        fits_header[key] = value

    return fits_header


def fitsheader_to_dict(
    initial_header: Header, sort_by_prefix: str = ""
) -> Tuple[dict, str]:
    """
    Extract a dictionary of parameters and a string from a FITS Header.

    The string is supposedly the name of the algorithm that was used to obtain
    the results that go with the Header.

    Parameters
    ----------
    initial_header : Header
        HDU Header that contains parameters used for the run of an algorithm
        through PostProc, and some unwanted parameters as well.
    sort_by_prefix : str
        String that will help filter keys of the header that don't start with
        that same string. By default, it doesn't filter out anything.

    Returns
    -------
    parameters : dict
        The set of parameters saved in PPResult that was used for a run of
        an algorithm.
    algo_name : str
        The name of the algorithm that was saved alongside its parameters.
    """
    head_dict = dict(initial_header)
    # Some parameters get their keys converted to uppercase
    lowercase_dict = {key.lower(): value for key, value in head_dict.items()}
    # Sorting parameters that don't belong in the dictionnary
    parameters = {
        key[len(sort_by_prefix):]: value
        for key, value in lowercase_dict.items()
        if key.startswith(sort_by_prefix)
    }
    algo_name = parameters["algo_name"]
    del parameters["algo_name"]
    return parameters, algo_name


def open_header(fitsfilename: str, n: int = 0, extname: str = None,
                verbose: bool = False) -> Header:
    """
    Load a FITS header into memory to avoid loading the data.

    This function is a simple wrapper of astropy.io.fits.convenience.getheader
    designed to substitute `open_fits` when only a FITS header is needed.
    This is ~ 40 times faster than using `open_fits` with header=True on
    an average sized VLT/SPHERE data set.

    Parameters
    ----------
    fitsfilename : string
        Name of the FITS file.
    n : int, optional
        Which HDU ext to open. Default is the first ext (zero based indexing).
    extname : str, optional
        Opens the HDU ext by name, rather than by HDU number. Overrides `n` and
        is not case-sensitive.
    verbose : bool, optional
        If True prints message of completion.

    Returns
    -------
    header : `Header` dictionary
        Astropy Header class with both a dict-like and list-like interface.
    """

    fitsfilename = str(fitsfilename)
    if not isfile(fitsfilename):
        fitsfilename += ".fits"

    # if extname is a non-empty string, provide that instead. Else use HDU number
    if extname and not extname.isspace():
        n = extname
        header = getheader(fitsfilename, extname=n, ignore_missing_end=True)
    else:
        header = getheader(fitsfilename, ext=n, ignore_missing_end=True)

    if verbose:
        print(f"FITS HDU-{n} header successfully loaded.")

    return header


def seeing_from_header(fitsfilename, verbose: bool = False) -> float:
    """
    Extract the average seeing values from FITS headers.

    Parameters
    ----------
    fitsfilename : string or list of string
        FITS files.
    verbose : bool, optional
        If True prints result.

    Returns
    -------
    seeing : float
        The average seeing extracted from the FITS header.
    """

    if isinstance(fitsfilename, str):
        fitsfilename = [fitsfilename]

    seeing = []
    for fitsfile in fitsfilename:  # loop and get the seeing values
        header = open_header(fitsfile)
        seeing.append(header["HIERARCH ESO TEL AMBI FWHM START"])
        seeing.append(header["HIERARCH ESO TEL AMBI FWHM END"])

    seeing = sum(seeing)/len(seeing)
    if verbose:
        print(f"Average seeing is {seeing} arcseconds", flush=True)
    return seeing
