#! /usr/bin/env python

"""Module with a dictionary and variables for storing constant parameters.

Usage
-----
from param import VLT_NACO
VLT_NACO['diam']

"""


VLT_NACO = {
    'plsc': 0.027190,                       # plate scale [arcsec]/px
    'diam': 8.2,                            # telescope diameter [m]
    'lambdal' : 3.8e-6,                     # filter central wavelength [m] L band
    'camera_filter' : 'Ll',
    'npix_x': 1024,
    'npix_y': 1024,
    # header keywords
    'kw_categ' : 'HIERARCH ESO DPR CATG',   # header keyword for calibration frames: 'CALIB' / 'SCIENCE'
    'kw_type' : 'HIERARCH ESO DPR TYPE'     # header keyword for frame type: 'FLAT,SKY' / 'DARK' / 'OBJECT' / 'SKY'
}


VLT_SINFONI = {
    'plsc': 0.0125,                         # plate scale [arcsec]/px
    'diam': 8.2,                            # telescope diameter [m]
    'lambdabrg' : 2.166e-6,                 # wavelength of the Brackett gamma line
    'camera_filter' : 'H+K',
    'spec_res':5e-4,                        # spectral resolution in um (for H+K)
    # header keywords
    'kw_categ' : 'HIERARCH ESO DPR CATG',   # header keyword for calibration frames: 'CALIB' / 'SCIENCE'
    'kw_type' : 'HIERARCH ESO DPR TYPE'     # header keyword for frame type: 'FLAT,SKY' / 'DARK' / 'OBJECT' / 'SKY'
}


LBT = {
    'latitude' : 32.70131,                  # LBT's latitude in degrees
    'longitude' : -109.889064,              # LBT's longitude in degrees
    'lambdal' : 3.47e-6,                    # central wavelenght L cont2' band [m]
    'plsc' : 0.0106,                        # plate scale [arcsec]/px
    'diam' : 8.4,                           # telescope diameter [m]
    # header keywords
    'lst': 'LBT_LST',                       # Local sidereal Time header keyword
    'ra' : 'LBT_RA',                        # right ascension header keyword
    'dec' : 'LBT_DEC',                      # declination header keyword
    'altitude' : 'LBT_ALT',                 # altitude header keyword
    'azimuth' : 'LBT_AZ',                   # azimuth header keyword
    'exptime' : 'EXPTIME',                  # nominal total integration time per pixel header keyword
    'acqtime' : 'ACQTIME',                  # total controller acquisition time header keyword    
    'filter' : 'LMIR_FW2'                   # filter
}




