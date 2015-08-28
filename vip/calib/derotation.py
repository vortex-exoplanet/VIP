#! /usr/bin/env python

"""
Module with frame de-rotation routine for ADI.
"""
__author__ = 'C. Gomez @ ULg, V. Christiaens @ UChile/ULg'
__all__ = ['cube_derotate',
           'frame_rotate',
           'derot_angles',
           'check_PA_vector']

import math
import numpy as np
import os
import cv2
from ..var import frame_center


def frame_rotate(array, angle, interpolation='bicubic', cy=None, cx=None):
    """ Rotates a frame.
    
    Parameters
    ----------
    array : array_like 
        Input frame, 2d array.
    angle : float
        Rotation angle.
    interpolation : {'bicubic', 'bilinear', 'nearneig'}, optional
        'nneighbor' stands for nearest-neighbor interpolation,
        'bilinear' stands for bilinear interpolation,
        'bicubic' for interpolation over 4x4 pixel neighborhood.
        The 'bicubic' is the default. The 'nearneig' is the fastest method and
        the 'bicubic' the slowest of the three. The 'nearneig' is the poorer
        option for interpolation of noisy astronomical images.
    cy, cx : float, optional
        Coordinates X,Y  of the point with respect to which the rotation will be 
        performed. By default the rotation is done with respect to the center 
        of the frame; central pixel if frame has odd size.
        
    Returns
    -------
    array_out : array_like
        Resulting frame.
        
    """
    if not array.ndim == 2:
        raise TypeError('Input array is not a frame or 2d array.')
    array = np.float32(array)
    y, x = array.shape
    
    if not cy and not cx:  cy, cx = frame_center(array)
    
    if interpolation == 'bilinear':
        intp = cv2.INTER_LINEAR
    elif interpolation == 'bicubic':
        intp= cv2.INTER_CUBIC
    elif interpolation == 'nearneig':
        intp = cv2.INTER_NEAREST
    else:
        raise TypeError('Interpolation method not recognized.')
    
    M = cv2.getRotationMatrix2D((cx,cy), angle, 1)
    array_out = cv2.warpAffine(array.astype(np.float32), M, (x, y), flags=intp)
             
    return array_out
    
    
def cube_derotate(array, angle_list, cy=None, cx=None):
    """ Rotates an ADI cube to a common north given a vector with the 
    corresponding parallactic angles for each frame of the sequence. By default
    bicubic interpolation is used (opencv). 
    
    Parameters
    ----------
    array : array_like 
        Input 3d array, cube.
    angle_list : list or 1D-array
        Vector containing the parallactic angles.
    cy, cx : float, optional
        Coordinates X,Y  of the point with respect to which the rotation will be 
        performed. By default the rotation is done with respect to the center 
        of the frames; central pixel if the frames have odd size.
        
    Returns
    -------
    array_der : array_like
        Resulting cube with de-rotated frames.
    array_out : array_like
        Median combined image of the de-rotated cube.
        
    """
    if not array.ndim == 3:
        raise TypeError('Input array is not a cube or 3d array.')
    array_der = np.zeros_like(array) 
    y, x = array[0].shape
    
    if not cy and not cx:  cy, cx = frame_center(array[0])
    
    for i in xrange(array.shape[0]): 
        M = cv2.getRotationMatrix2D((cx,cy), -angle_list[i], 1)
        array_der[i] = cv2.warpAffine(array[i].astype(np.float32), M, (x, y))
            
    array_out = np.median(array_der, axis=0)              
    return array_der, array_out


def derot_angles(objname_tmp_A, digit_format=3,objname_tmp_B='',inpath='~/', skew=False, writing=False, outpath='~/', list_obj=None, cd11_key='CD1_1', cd12_key='CD1_2', cd21_key='CD2_1', cd22_key='CD2_2', verbose=False):
    """
    Function that provides a numpy vector of angles to derotate datacubes so as to match North up, East left, based on header information.
    It is robust for the pca algorithm (which computes variables based on this vector), in the sense that:
    1) all angles of the output are in degrees
    2) all angles of the ouput list are positive
    3) there is no jump of more than 180 deg between consecutive values (e.g. no jump like [350deg,355deg,0deg,5deg] => replaced by [350deg,355deg,360deg,365deg])

    Parameters:
    -----------
    objname_tmp_A: string
        Contains the common name of the cubes BEFORE the digits
    digit_format: int
        Number of digits in the name of the cube. The digits are supposed to be the only changing part in the name of one cube to another.
    objname_tmp_B: string, optional
        Contains the name of the cubes AFTER the digits
    inpath: string
        Contains the full path of the directory with the data
    skew: bool, optional {False,True}
        True if you know there is a different rotation between y- and x- axes. 
        The code also detects automatically if there is >1deg skew between y and x axes. 
        In case of skewing, 2 vectors of derotation angles are returned: one for x and one for y, instead of only one vector.
    writing: bool, optional {False,True}
        True if you want to write the derotation angles in a txt file.
    outpath: string, optional 
        Contains the full path of the directory where you want the txt file to be saved.
    list_obj: integer list or 1-D array
        List of the digits corresponding to the cubes to be considered.
        If not provided, the function will consider automatically all the cubes with objname_tmp_A+digit+objname_tmp_B+'.fits' name structure in the provided "inpath"
    cd11_key,cd12_key,cd21_key,cd22_key: strings
        Name of the keywords to be looked up in the header, to provide the:
        - partial of first axis coordinate w.r.t. x   (cd11_key)
        - partial of first axis coordinate w.r.t. y   (cd12_key)
        - partial of second axis coordinate w.r.t. x  (cd21_key)
        - partial of second axis coordinate w.r.t. y  (cd22_key)
        Default values are the ones in the headers of ESO or HST fits files.
        See http://www.stsci.edu/hst/HST_overview/documents/multidrizzle/ch44.html for more information.
    verbose: boolean, optional {False,True}
        True if you want more info to be printed.

    Example:
    -------
    If your cubes are: '/home/foo/out_cube_obj_HK_025_000_sorted.fits',
                       '/home/foo/out_cube_obj_HK_025_001_sorted.fits',
                       '/home/foo/out_cube_obj_HK_025_002_sorted.fits', etc,
    the first arguments should be:
                       objname_tmp_A = 'out_cube_obj_HK_025_'
                       digit_format = 3
                       objname_tmp_B = '_sorted'
                       inpath = /home/foo/'

    Return:
    -------
    angle_list: 1-D array_like
        vector of angles corresponding to the angular difference between the positive y axis and the North in the image.
        sign convention: positive angles in anti-clockwise direction.
        tip: apply the opposite value when rotating the image to match North up.
    """

    def numberToString(n, digits):
        """ 
        Converts an int in a string according to the number of desired digits
        
        Parameters:
        -----------
        n: int
            Number to be converted into string
        digits: int
            Number of characters in the string. If less than the number of digits of n, it is filled with zeros.

        Examples:
        ---------
        >>> numberToString(23, 3)
        023
        >>> numberToString(8, 5)
        00008

        Returns:
        --------
        number: string
            The string representing "n", with "digits" characters.
        """

        number = str(n)
        for i in range(digits - len(number)):
            number = "0" + number
        return number
   

    cd1_1 = []
    cd1_2 = []
    cd2_1 = []
    cd2_2 = []

    if list_obj == None:
        list_obj = []
        for ii in range(10**digit_format):
            digits_ii = numberToString(ii,digit_format)
            if os.path.exists(inpath+objname_tmp_A+digits_ii+objname_tmp_B+'.fits'):   
                list_obj.append(ii)
                _, header = vip.fits.open_fits(inpath+objname_tmp_A+digits_ii+objname_tmp_B+'.fits', verbose=False, header=True)
                cd1_1.append(header[cd11_key])
                cd1_2.append(header[cd12_key])
                cd2_1.append(header[cd21_key])
                cd2_2.append(header[cd22_key])
    else:
        for ii in list_obj:
            digits_ii = numberToString(ii,digit_format)
            _, header = vip.fits.open_fits(inpath+objname_tmp_A+digits_ii+objname_tmp_B+'.fits', verbose=False, header=True)
            cd1_1.append(header[cd11_key])
            cd1_2.append(header[cd12_key])
            cd2_1.append(header[cd21_key])
            cd2_2.append(header[cd22_key])

    # Determine if it's a right- or left-handed coordinate system from the first cube
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
            rot[ii]=0.-np.arctan2(sgn*cd1_2[ii],sgn*cd1_1[ii])
            rot2[ii]=0.-np.arctan2(-cd2_1[ii],cd2_2[ii])
            if rot2[ii] < 0:
                rot2[ii] = (2*math.pi)+rot2[ii]
        if np.floor(rot[ii]) != np.floor(rot2[ii]):
            print "There is more than 1deg differential rotation between axes y and x!"
            skew = True

    # Check and correct to output at the right format
    rot = check_PA_vector(rot,'rad')
    if skew: rot2 = check_PA_vector(rot2,'rad')

    if verbose:
        print "This is the list of angles to be applied: "
        for ii in range(len(cd1_1)):
            print ii, ' -> ', rot[ii]
            if skew: print 'rot2: ', ii, ' -> ', rot2[ii]

    if writing:
        if outpath == '' or outpath == None: outpath=inpath
        f=open(outpath+'Parallactic_angles.txt','w')
        for ii in range(len(cd1_1)):
            print >>f, rot[ii]
        f.close()


    if skew: return rot, rot2
    else: return rot



def check_PA_vector(angle_list, unit='deg'):
    """
    Function to check if the angle list has the right format to avoid any bug in the pca algorithm.
    The right format complies to 3 criteria:
       1) angles are expressed in degree
       2) the angles are positive
       3) there is no jump of more than 180 deg between consecutive values (e.g. no jump like [350deg,355deg,0deg,5deg] => replaced by [350deg,355deg,360deg,365deg])

    Parameter:
    ----------
    angle_list: 1D-array_like
        Vector containing the derotation angles
    unit: String, {'deg','rad'}
        The unit type of the input angle list

    Returns:
    --------
    angle_list: 1-D array_like 
        Vector containing the derotation angles (after correction to comply with the 3 criteria, if needed)

    """

    if unit != 'rad' and unit != 'deg':
        raise ValueError("The input unit should either be 'deg' or 'rad'")

    npa = angle_list.shape[0]

    for ii in range(npa):
        if unit == 'rad':
            angle_list[ii] = np.rad2deg(angle_list[ii])
        if angle_list[ii] < 0:
            angle_list[ii] = 360+angle_list[ii]

    correct = False
    sorted_rot = np.sort(angle_list)

    # Check if there is a jump > 180deg  within the angle list (e.g. from 355deg to 5deg)
    for ii in range(npa-1):
        if abs(sorted_rot[ii+1]-sorted_rot[ii]) > 180:
            correct = True
            break

    # In the previous case, correct for it by adding 360deg to angles whose value is < 180deg
    if correct:
        for ii in range(npa):
           if angle_list[ii] < 180:
               angle_list[ii] = 360+angle_list[ii]

    return angle_list
