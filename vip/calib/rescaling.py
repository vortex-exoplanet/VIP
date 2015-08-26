#! /usr/bin/env python

"""
Module with frame rescaling routine for SDI.
"""
__author__ = 'V. Christiaens @ ULg'
__all__ = ['scale_cube',
           'cube_rescaling',
           'frame_rescaling',
           'scale_func']

import cv2
import numpy as np
from scipy.ndimage.interpolation import geometric_transform
from ..var import frame_center, get_square_robust


def scale_cube(cube,scal_list,full_output=True,inverse=False,y_in=1,x_in=1):

    """
    Wrapper to scale or descale a cube by factors given in scal_list, without any loss of information (zero-padding if scaling > 1).
    Important: in case of ifs data, the scaling factors in var_list should be >= 1 (ie. provide the scaling factors as for scaling to the longest wavelength channel)
    """

    #First pad the cube with zeros appropriately to not loose info when scaling the cube.
    # Next step: pad with random gaussian noise instead of zeros??  Too many zeros might alter the svd or not?

    n, y, x = cube.shape

    max_sc = np.amax(scal_list)

    if not inverse and max_sc > 1:
        new_y = np.ceil(max_sc*y)
        new_x = np.ceil(max_sc*x)
        if (new_y - y)%2 != 0: new_y = new_y+1
        if (new_x - x)%2 != 0: new_x = new_x+1
        pad_len_y = (new_y - y)/2
        pad_len_x = (new_x - x)/2
        big_cube = np.pad(cube, ((0,0), (pad_len_y, pad_len_y), (pad_len_x, pad_len_x)), 'constant', constant_values=(0,))
    else: 
        big_cube = cube.copy()

    n, y, x = big_cube.shape
    cy,cx = frame_center(big_cube[0])
    var_list = scal_list

    if inverse:
        var_list = 1./scal_list[:]
        cy,cx = frame_center(cube[0])


    # (de)scale the cube, so that a planet would now move radially
    cube,frame = cube_rescaling(big_cube,var_list,ref_y=cy, ref_x=cx)


    if inverse:
        if max_sc > 1:
            frame = get_square_robust(frame,max(y_in,x_in), cy,cx,strict=False)
            if full_output:
                n_z = cube.shape[0]
                array_old = cube.copy()
                cube = np.zeros([n_z,max(y_in,x_in),max(y_in,x_in)])
                for zz in range(n_z):
                    cube[zz]=get_square_robust(array_old[zz],max(y_in,x_in), cy,cx,strict=False)


    if full_output: return cube,frame,y,x,cy,cx
    else: return frame


def scale_func(output_coords,ref_y=0,ref_x=0, scaling=1.0):    
    """
    For each coordinate point in a new scaled image (output_coords), 
    coordinates in the image before the scaling are returned. 
    Typically this function is used within geometric_transform, 
    which, for each point in the output image, will compute the (spline)
    interpolated value at the corresponding frame coordinates before the scaling.
    """
    return (ref_y+((output_coords[0]-ref_y)/scaling), ref_x+((output_coords[1]-ref_x)/scaling))


def frame_rescaling(array, ref_y=0,ref_x=0, gamma=1.0, method = 'geometric_transform'):
    """
    Function to rescale a frame by a factor gamma,
    with respect to a reference point ref_pt (typically the exact location of the star).
    However, it keeps the same dimensions.
    It uses spline interpolation of order 3 to find the new values in the output array.
    
    Parameters
    ----------
    array : array_like 
        Input frame, 2d array.
    ref_y, ref_x : float, optional
        Coordinates X,Y  of the point with respect to which the rotation will be 
        performed. By default the rotation is done with respect to the center 
        of the frame; central pixel if frame has odd size.
    gamma : float
        Scaling factor. If > 1, it will expand the input array.        
    method: {geometric_transform,cv2.warp_affine}, optional
        String determining which method to apply to rescale. 
        Both use a spline of order 3 for interpolation.
        geometric transform (default) seems to work fine, from a test on a numpy array in ipython.
        cv2.warp_affine works good as well, although it yields slightly different results for the same ipython test. 
        From this test, it seems it is not as good as geometric_transform for edge values. 
        More tests might be needed. No comparison of speed was done between both algorithms.

    Returns
    -------
    array_out : array_like
        Resulting frame.
        
    """

    array_out = np.zeros_like(array)

    if method == 'geometric_transform':
        geometric_transform(array, scale_func, output_shape=array.shape, output = array_out, extra_keywords={'ref_y':ref_y,'ref_x':ref_x,'scaling':gamma})

    elif method == 'cv2.warp_affine':
        M = np.array([[gamma,0,(1.-gamma)*ref_x],[0,gamma,(1.-gamma)*ref_y]])
        array_out = cv2.warpAffine(array, M, (array.shape[1],array.shape[0]), flags=cv2.INTER_LINEAR)

    else: raise ValueError("Please choose a valid method between 'geometric transform' and 'cv2.warp_affine' ")

    return array_out

    
    
def cube_rescaling(array, scaling_list, ref_y=None, ref_x=None, method = 'geometric_transform'):
    """ Function to rescale a cube, frame by frame, 
        by a factor gamma, with respect to position (cy,cx).
        It calls frame_rescaling.
    
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
    method: {geometric_transform,cv2.warp_affine}, optional
        String determining which method to apply to rescale. 
        Both use a spline of order 3 for interpolation.
        geometric transform (default) seems to work fine, from a test on a numpy array in ipython.
        cv2.warp_affine works good as well, although it yields slightly different results for the same ipython test. 
        From this test, it seems it is not as good as geometric_transform for edge values. 
        More tests might be needed. No comparison of speed was done between both algorithms.
        
    Returns
    -------
    array_der : array_like
        Resulting cube with rescaled frames.
    array_out : array_like
        Median combined image of the rescaled cube.
        
    """
    if not array.ndim == 3:
        raise TypeError('Input array is not a cube or 3d array.')
    array_sc = np.zeros_like(array) 
    y, x = array[0].shape
    
    if not ref_y and not ref_x:  ref_y, ref_x = frame_center(array[0])
    
    for i in xrange(array.shape[0]): 
        array_sc[i] = frame_rescaling(array[i], ref_y=ref_y,ref_x=ref_x, gamma=scaling_list[i], method=method)
            
    array_out = np.median(array_sc, axis=0)              
    return array_sc, array_out

