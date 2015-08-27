#! /usr/bin/env python

"""
Module with frame de-rotation routine for ADI.
"""
__author__ = 'C. Gomez @ ULg'
__all__ = ['cube_derotate',
           'frame_rotate',
           'derot_angles',
           'numberToString']

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
    angle_list : list
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


def derot_angles(objname_tmp_A, digit_format=3,objname_tmp_B='',inpath='~/',verbose=False, skew = False, writing=True, outpath='~/',unit='deg',list_obj=None):
    """
    FUNCTION TO PROVIDE A LIST OF ANGLES TO DEROTATE DATACUBES FROM ANY HEADER OF ESO (not tested on other headers).
    The header should contain the following keywords: 'CD1_1','CD1_2','CD2_1','CD2_2' and 'CUNIT1'.
    ROBUST, in the sense that:
    1) all angles of the ouput list are positive
    2) there is no jump of more than 180 deg between consecutive values (e.g. no jump like [350deg,355deg,0deg,5deg] => replaced by [350deg,355deg,360deg,365deg])

    Parameters:
    -----------
    objname_tmp_A: string containing the name of the cubes BEFORE the digits (e.g.  'out_cube_obj_HK_025_')
    digit_format: number of digits (e.g. if digits are like 088, digit_format = 3)
    objname_tmp_B: string containing the name of the cubes AFTER the digits (e.g.  '')
    inpath: string containing the full path of the directory with the data
    verbose: Bool.
             True if you want more info to be printed.
    skew: Bool. 
          True if you know there is a different rotation between y- and x- axes. The code also detects automatically if there is >1deg skew between y and x axes. In such cases, 2 lists of derotation angles are returned: one for x and one for y.
    writing: Bool.
             True if you want to write the derotation angles in a txt file.
    outpath:  string containing the full path of the directory where we want the list to be created.
    unit: string with either "deg" or "rad" depending on the units you want for the angles
    list_obj: list of digits corresponding to the cubes to be considered, if not provided, it will find all of the cubes in the inpath  by itself

    Return:
    -------
    A LIST OF ANGLES corresponding to the difference between the positive y axis and the North in the image.
    Sign convention: positive angles in trigonometric or anti-clockwise direction.
    Apply the opposite value when rotating the image to match North up.
    """
   
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
                if header['CUNIT1'] == 'deg':
                    cd1_1.append(3600*header['CD1_1'])
                    cd1_2.append(3600*header['CD1_2'])
                    cd2_1.append(3600*header['CD2_1'])
                    cd2_2.append(3600*header['CD2_2'])
                else:
                    cd1_1.append(header['CD1_1'])
                    cd1_2.append(header['CD1_2'])
                    cd2_1.append(header['CD2_1'])
                    cd2_2.append(header['CD2_2'])
    else:
        for ii in list_obj:
            digits_ii = numberToString(ii,digit_format)
            _, header = vip.fits.open_fits(inpath+objname_tmp_A+digits_ii+objname_tmp_B+'.fits', verbose=False, header=True)
            if header['CUNIT1'] == 'deg':
                cd1_1.append(3600*header['CD1_1'])
                cd1_2.append(3600*header['CD1_2'])
                cd2_1.append(3600*header['CD2_1'])
                cd2_2.append(3600*header['CD2_2'])
            else:
                cd1_1.append(header['CD1_1'])
                cd1_2.append(header['CD1_2'])
                cd2_1.append(header['CD2_1'])
                cd2_2.append(header['CD2_2'])

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
            if rot[ii] < 0:
                rot[ii] = (2*math.pi)+rot[ii]
            rot2[ii]=0.-np.arctan2(-cd2_1[ii],cd2_2[ii])
            if rot2[ii] < 0:
                rot2[ii] = (2*math.pi)+rot2[ii]
        if np.floor(rot[ii]) != np.floor(rot2[ii]):
            print "There is more than 1deg differential rotation between axes y and x!"
            skew = True

    # Check if there is a jump within the angle list like from 355deg to 5deg (typically if the abs difference is > 180deg)
    cycle = False
    sorted_rot = np.sort(rot)
    for ii in range(len(cd1_1)-1):
        if abs(sorted_rot[ii+1]-sorted_rot[ii]) > math.pi:
            cycle = True
            break

    # In the previous case, correct for it by adding 360deg to angles whose value is < 180deg
    if cycle:
        for ii in range(len(cd1_1)):
           if rot[ii] < math.pi:
               rot[ii] = (2*math.pi)+rot[ii]
           if rot2[ii] < math.pi:
               rot2[ii] = (2*math.pi)+rot2[ii]

    if verbose:
        print "This is the list of angles to be applied: "
    for ii in range(len(cd1_1)):
        if unit == 'deg':
            rot[ii] = np.rad2deg(rot[ii])
            rot2[ii] = np.rad2deg(rot2[ii])
        if verbose:
            print ii, ' -> ', rot[ii]

    if writing:
        if outpath == '' or outpath == None: outpath=inpath
        f=open(outpath+'Parallactic_angles.txt','w')
        for ii in range(len(cd1_1)):
            print >>f, rot[ii]
        f.close()


    if skew: return rot, rot2
    else: return rot


def numberToString(n, digits):
    """ Convierte un int en un string de acuerdo a la cantidad de digitos
    Ejemplo:
        >>> numberToString(23, 3)
        023
        >>> numberToString(8, 5)
        00008
    """

    number = str(n)
    for i in range(digits - len(number)):
        number = "0" + number
    return number
