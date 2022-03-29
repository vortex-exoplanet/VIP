#! /usr/bin/env python

"""
Module with frame parallactica angles calculations and de-rotation routines for
ADI.
"""


__author__ = 'V. Christiaens @ UChile/ULg, Carlos Alberto Gomez Gonzalez'
__all__ = ['compute_paral_angles',
           'compute_derot_angles_pa',
           'compute_derot_angles_cd',
           'check_pa_vector']

import math
import numpy as np
import os
from ..fits import open_fits
from astropy.coordinates import FK5
from astropy.coordinates import sky_coordinate
from astropy.time import Time
from astropy.units import hourangle, degree 


def compute_paral_angles(header, latitude, ra_key, dec_key, lst_key, 
                         acqtime_key, date_key='DATE-OBS'):
    """Calculates the parallactic angle for a frame, taking coordinates and
    local sidereal time from fits-headers (frames taken in an alt-az telescope 
    with the image rotator off).
    
    The coordinates in the header are assumed to be J2000 FK5 coordinates.
    The spherical trigonometry formula for calculating the parallactic angle
    is taken from Astronomical Algorithms (Meeus, 1998).
    
    Parameters
    ----------
    header : dictionary
        Header of current frame.
    latitude : float
        Latitude of the observatory in degrees. The dictionaries in 
        vip_hci/conf/param.py can be used like: latitude=LBT['latitude'].
    ra_key, dec_key, lst_key, acqtime_key, date_key : strings
        Keywords where the values are stored in the header.
        
    Returns
    -------
    pa.value : float
        Parallactic angle in degrees for current header (frame).
    """                                    
    obs_epoch = Time(header[date_key], format='iso', scale='utc')
       
    # equatorial coordinates in J2000
    ra = header[ra_key]                                                         
    dec = header[dec_key]   
    coor = sky_coordinate.SkyCoord(ra=ra, dec=dec, unit=(hourangle,degree),
                                   frame=FK5, equinox='J2000.0')
    # recalculate for DATE-OBS (precession)
    coor_curr = coor.transform_to(FK5(equinox=obs_epoch))
    
    # new ra and dec in radians
    ra_curr = coor_curr.ra                                                      
    dec_curr = coor_curr.dec
        
    lst_split = header[lst_key].split(':')
    lst = float(lst_split[0])+float(lst_split[1])/60+float(lst_split[2])/3600
    exp_delay = (header[acqtime_key] * 0.5) / 3600
    # solar to sidereal time
    exp_delay = exp_delay*1.0027                                                
    
    # hour angle in degrees
    hour_angle = (lst + exp_delay) * 15 - ra_curr.deg                           
    hour_angle = np.deg2rad(hour_angle)                                         
    latitude = np.deg2rad(latitude)                                             
    
    # PA formula from Astronomical Algorithms 
    pa = -np.rad2deg(np.arctan2(-np.sin(hour_angle), np.cos(dec_curr) * \
                 np.tan(latitude) - np.sin(dec_curr) * np.cos(hour_angle)))     
  
    #if dec_curr.value > latitude:  pa = (pa.value + 360) % 360
    
    return pa.value


def compute_derot_angles_pa(objname_tmp_A, digit_format=3, objname_tmp_B='',
                            inpath='./', writing=False, outpath='./',
                            list_obj=None, 
                            PosAng_st_key='HIERARCH ESO ADA POSANG',
                            PosAng_nd_key='HIERARCH ESO ADA POSANG END',
                            verbose=False):
    """
    Function that returns a numpy vector of angles to derotate datacubes so as
    to match North up, East left, based on the mean of the Position Angle at
    the beginning and the end of the exposure.
    => It is twice more precise than function derot_angles_CD (there can be
    >1deg difference in the resulting angle vector returned for fast rotators
    with long exposures!), but it requires:

    1. a header keyword for both the position angle at start and end of exposure
    2. no skewness of the frames

    The output is in appropriate format for the pca algorithm in the sense that:

    1. all angles of the output are in degrees
    2. all angles of the ouput  are positive
    3. there is no jump of more than 180 deg between consecutive values (e.g. no
       jump like [350deg,355deg,0deg,5deg] => replaced by
       [350deg,355deg,360deg,365deg])

    Parameters
    ----------
    objname_tmp_A: string
        Contains the common name of the cubes BEFORE the digits
    digit_format: int, optional
        Number of digits in the name of the cube. The digits are supposed to be
        the only changing part in the name of one cube to another.
    objname_tmp_B: string, optional
        Contains the name of the cubes AFTER the digits
    inpath: string, optional
        Contains the full path of the directory with the data
    writing: bool, optional
        True if you want to write the derotation angles in a txt file.
    outpath: string, optional
        Contains the full path of the directory where you want the txt file to
        be saved.
    list_obj: integer list or 1-D array, optional
        List of the digits corresponding to the cubes to be considered.
        If not provided, the function will consider automatically all the cubes
        with objname_tmp_A+digit+objname_tmp_B+'.fits' name structure in the
        provided "inpath".
    PosAng_st_key, PosAng_nd_key: strings, optional
        Name of the keywords to be looked up in the header, to provide the PA
        from North at start and end of integration.
    verbose: bool, optional
        True if you want more info to be printed.

    Examples
    --------
    If your cubes are: ``/home/foo/out_cube_obj_HK_025_000_sorted.fits``,
    ``/home/foo/out_cube_obj_HK_025_001_sorted.fits``,
    ``/home/foo/out_cube_obj_HK_025_002_sorted.fits``, etc,
    the first arguments should be:

    .. code-block:: python

        objname_tmp_A = 'out_cube_obj_HK_025_'
        digit_format = 3
        objname_tmp_B = '_sorted'
        inpath = '/home/foo/'

    Returns
    -------
    angle_list: 1-D numpy ndarray
        vector of angles corresponding to the angular difference between the
        positive y axis and the North in the image.
        sign convention: positive angles in anti-clockwise direction.
        Opposite values are applied when rotating the image to match North up.
    """
   

    posang_st = []
    posang_nd = []

    def _fitsfile(ii):
        return "{}{}{:0{}d}{}.fits".format(inpath, objname_tmp_A, ii,
                                           digit_format, objname_tmp_B)

    if list_obj is None:
        list_obj = []
        for ii in range(10**digit_format):
            if os.path.exists(_fitsfile(ii)):   
                list_obj.append(ii)
                _, header = open_fits(_fitsfile(ii), verbose=False, header=True)
                posang_st.append(header[PosAng_st_key])
                posang_nd.append(header[PosAng_nd_key])
    else:
        for ii in list_obj:
            _, header = open_fits(_fitsfile(ii), verbose=False, header=True)
            posang_st.append(header[PosAng_st_key])
            posang_nd.append(header[PosAng_nd_key])


    # Write the vector containing parallactic angles
    rot = np.zeros(len(list_obj))
    for ii in range(len(list_obj)):
        rot[ii]=-(posang_st[ii]+posang_nd[ii])/2

    # Check and correct to output at the right format
    rot = check_pa_vector(rot,'deg')

    if verbose:
        print("This is the list of angles to be applied: ")
        for ii in range(len(list_obj)):
            print(ii, ' -> ', rot[ii])

    if writing:
        if outpath == '' or outpath is None: outpath=inpath
        f=open(outpath+'Parallactic_angles.txt','w')
        for ii in range(len(list_obj)):
            print(rot[ii], file=f)
        f.close()

    return rot


def compute_derot_angles_cd(objname_tmp_A, digit_format=3, objname_tmp_B='',
                            inpath='./', skew=False, writing=False, 
                            outpath='./', list_obj=None, cd11_key='CD1_1', 
                            cd12_key='CD1_2', cd21_key='CD2_1', 
                            cd22_key='CD2_2', verbose=False):
    """
    Function that returns a numpy vector of angles to derotate datacubes so as 
    to match North up, East left, based on the CD matrix information contained
    in the header.
    In case the PosAng keyword is present in the header and there is no skewness
    between x and y axes, favor the use of function compute_derot_angles_PA
    (more precise as it averages for the middle of the exposure).
    The output is in appropriate format for the pca algorithm in the sense that:

    1. all angles of the output are in degrees
    2. all angles of the ouput  are positive
    3. there is no jump of more than 180 deg between consecutive values (e.g. no
       jump like [350deg,355deg,0deg,5deg] => replaced by
       [350deg,355deg,360deg,365deg])

    Parameters
    ----------
    objname_tmp_A: string
        Contains the common name of the cubes BEFORE the digits
    digit_format: int, optional
        Number of digits in the name of the cube. The digits are supposed to be 
        the only changing part in the name of one cube to another.
    objname_tmp_B: string, optional
        Contains the name of the cubes AFTER the digits
    inpath: string, optional
        Contains the full path of the directory with the data
    skew: bool, optional
        True if you know there is a different rotation between y- and x- axes. 
        The code also detects automatically if there is >1deg skew between y and
        x axes. In case of skewing, 2 vectors of derotation angles are returned:
        one for x and one for y, instead of only one vector.
    writing: bool, optional
        True if you want to write the derotation angles in a txt file.
    outpath: string, opt
        Contains the full path of the directory where you want the txt file to 
        be saved.
    list_obj: integer list or 1-D array or None, optional
        List of the digits corresponding to the cubes to be considered.
        If not provided, the function will consider automatically all the cubes 
        with objname_tmp_A+digit+objname_tmp_B+'.fits' name structure in the 
        provided "inpath".
    cd11_key,cd12_key,cd21_key,cd22_key: strings, optional
        Name of the keywords to be looked up in the header, to provide the:

        - partial of first axis coordinate w.r.t. x   (cd11_key)
        - partial of first axis coordinate w.r.t. y   (cd12_key)
        - partial of second axis coordinate w.r.t. x  (cd21_key)
        - partial of second axis coordinate w.r.t. y  (cd22_key)

        Default values are the ones in the headers of ESO or HST fits files.
        For more information, go to:
        http://www.stsci.edu/hst/HST_overview/documents/multidrizzle/ch44.html
    verbose: bool, optional
        True if you want more info to be printed.


    Examples
    --------
    If your cubes are: ``/home/foo/out_cube_obj_HK_025_000_sorted.fits``,
    ``/home/foo/out_cube_obj_HK_025_001_sorted.fits``,
    ``/home/foo/out_cube_obj_HK_025_002_sorted.fits``, etc,
    the first arguments should be:

    .. code:: python

        objname_tmp_A = 'out_cube_obj_HK_025_'
        digit_format = 3
        objname_tmp_B = '_sorted'
        inpath = '/home/foo/'

    Returns
    -------
    angle_list: 1-D numpy ndarray
        vector of angles corresponding to the angular difference between the
        positive y axis and the North in the image.
        sign convention: positive angles in anti-clockwise direction.
        Opposite values are applied when rotating the image to match North up.
        **Note:** if skew is set to True, there are 2 angle_list vectors
        returned; the first to rotate the x-axis and the second for the y-axis.

    """

    cd1_1 = []
    cd1_2 = []
    cd2_1 = []
    cd2_2 = []

    def _fitsfile(ii):
        return "{}{}{:0{}d}{}.fits".format(inpath, objname_tmp_A, ii,
                                           digit_format, objname_tmp_B)

    if list_obj is None:
        list_obj = []
        for ii in range(10**digit_format):
            if os.path.exists(_fitsfile(ii)):   
                list_obj.append(ii)
                _, header = open_fits(_fitsfile(ii), verbose=False, header=True)
                cd1_1.append(header[cd11_key])
                cd1_2.append(header[cd12_key])
                cd2_1.append(header[cd21_key])
                cd2_2.append(header[cd22_key])
    else:
        for ii in list_obj:
            _, header = open_fits(_fitsfile(ii), verbose=False, header=True)
            cd1_1.append(header[cd11_key])
            cd1_2.append(header[cd12_key])
            cd2_1.append(header[cd21_key])
            cd2_2.append(header[cd22_key])

    # Determine if it's a right- or left-handed coord system from first cube
    det=cd1_1[0]*cd2_2[0]-cd1_2[0]*cd2_1[0]
    if det<0:  sgn = -1
    else: sgn = 1
    
    # Write the vector containing parallactic angles
    rot = np.zeros(len(list_obj))
    rot2 = np.zeros(len(list_obj))
    for ii in range(len(cd1_1)):
        if cd2_1[ii]==0 and cd1_2[ii]==0:
            rot[ii]=0
            rot2[ii]=0
        else:
            rot[ii]=-np.arctan2(sgn*cd1_2[ii],sgn*cd1_1[ii])
            rot2[ii]=-np.arctan2(-cd2_1[ii],cd2_2[ii])
            if rot2[ii] < 0:
                rot2[ii] = 2*math.pi + rot2[ii]
        if np.floor(rot[ii]) != np.floor(rot2[ii]):
            msg = "There is more than 1deg skewness between y and x! "
            msg2 = "Please re-run the function with argument skew=True"
            raise ValueError(msg+msg2)

    # Check and correct to output at the right format
    rot = check_pa_vector(rot,'rad')
    if skew:
        rot2 = check_pa_vector(rot2,'rad')

    if verbose:
        print("This is the list of angles to be applied: ")
        for ii in range(len(cd1_1)):
            print(ii, ' -> ', rot[ii])
            if skew: print('rot2: ', ii, ' -> ', rot2[ii])

    if writing:
        if outpath == '' or outpath is None: outpath=inpath
        f=open(outpath+'Parallactic_angles.txt','w')
        if skew:
            for ii in range(len(cd1_1)):
                print(rot[ii], rot2[ii], file=f)
        else:
            for ii in range(len(cd1_1)):
                print(rot[ii], file=f)
        f.close()

    if skew:
        return rot, rot2
    else:
        return rot


def check_pa_vector(angle_list, unit='deg'):
    """ Checks if the angle list has the right format to avoid any bug in the 
    pca-adi algorithm. The right format complies to 3 criteria:

    1. angles are expressed in degree
    2. the angles are positive
    3. there is no jump of more than 180 deg between consecutive values (e.g.
        no jump like [350deg,355deg,0deg,5deg] => replaced by
        [350deg,355deg,360deg,365deg])
       
    Parameters
    ----------
    angle_list: 1D-numpy ndarray
        Vector containing the derotation angles
    unit: string, {'deg','rad'}, optional
        The unit type of the input angle list
    
    Returns
    -------
    angle_list: 1-D numpy ndarray 
        Vector containing the derotation angles (after correction to comply with
        the 3 criteria, if needed)
    """
    angle_list = angle_list.copy()
    if unit != 'rad' and unit != 'deg':
        raise ValueError("The input unit should either be 'deg' or 'rad'")

    npa = angle_list.shape[0]

    for ii in range(npa):
        if unit == 'rad':
            angle_list[ii] = np.rad2deg(angle_list[ii])
        if angle_list[ii] < 0:
            angle_list[ii] = 360+angle_list[ii]

    correct = False
    #sorted_rot = np.sort(angle_list)

    # Check if there is a jump > 180deg  within the angle list
    for ii in range(npa-1):
        #if abs(sorted_rot[ii+1]-sorted_rot[ii]) > 180:
        if abs(angle_list[ii+1]-angle_list[ii]) > 180:
            correct = True
            break

    # In the previous case, correct for it by adding 360deg to angles < 180deg
    if correct:
        for ii in range(npa):
            if angle_list[ii] < 180:
                angle_list[ii] = 360+angle_list[ii]

    return angle_list


