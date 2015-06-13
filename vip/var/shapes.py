#! /usr/bin/env python

"""
Module with various functions.
"""

from __future__ import division

__author__ = 'C. Gomez @ ULg'
__all__ = ['dist',
           'frame_center',
           'get_square',
           'get_circle',
           'get_annulus',
           'get_annulus_quad',
           'get_annulus_cube',
           'mask_circle']

import numpy as np    

def dist(yc,xc,y1,x1):
    """ Returns the distance between two points.
    """
    return np.sqrt((yc-y1)**2+(xc-x1)**2)


def frame_center(array, verbose=False):
    """ Returns the coordinates y,x of a frame central pixel if the sides are 
    odd numbers. Python uses 0-based indexing, so the coordinates of the center
    of a 5x5 pixels frame are (2,2). 
    When the frame sides are even values, the coordinates returned are:
    (side_length/2) - 1.  
    """
    y = array.shape[0]/2.       
    x = array.shape[1]/2.
    # side length/2 - 1, because python has 0-based indexing
    cy = np.ceil(y) - 1
    cx = np.ceil(x) - 1
    if verbose:
        print 'Half image at x,y = ({:.3f},{:.3f})'.format(y, x)
        print 'Center px coordinates at x,y = ({:},{:})'.format(cy, cx)
    return cy, cx

    
def get_square(array, size, y, x, position=False):                 
    """ Returns an square subframe. If size is an even value it'll be increased
    by one to make it odd.
    
    Parameters
    ----------
    array : array_like
        Input frame.
    size : int, odd
        Size of the subframe.
    y, x : int
        Coordinates of the center of the subframe.
    position : {False, True}, optional
        If set to True return also the coordinates of the left upper vertex.
        
    Returns
    -------
    array_view : array_like
        Sub array.
        
    """
    if not array.ndim==2:
        raise TypeError('Input array is not a frame or 2d array.')
    
    if size%2!=0:  size -= 1 # making it odd to get the wing
    wing = size/2    
    # wing is added to the sides of the subframe center. Note the +1 when 
    # closing the interval (python doesn't include the endpoint)
    array_view = array[y-wing:y+wing+1, x-wing:x+wing+1].copy()
    
    if position:
        return array_view, y-wing, x-wing
    else:
        return array_view


def get_circle(array, radius, output_values=False, cy=None, cx=None):           
    """Returns a centered circular region from a 2d ndarray. All the rest 
    pixels are set to zeros. 
    
    Parameters
    ----------
    array : array_like
        Input 2d array or image. 
    radius : int
        The radius of the circular region.
    output_values : {False, True}
        Sets the type of output.
    cy, cx : int
        Coordinates of the circle center.
        
    Returns
    -------
    values : array_like
        1d array with the values of the pixels in the circular region.
    array_masked : array_like
        Input array with the circular mask applied.
    """
    if not array.ndim == 2:
        raise TypeError('Input array is not a frame or 2d array.')
    sy, sx = array.shape
    if not cy and not cx:
        cy, cx = frame_center(array, verbose=False)
         
    yy, xx = np.ogrid[:sx, :sy]                                                 # ogrid is a multidim mesh creator (faster than mgrid)
    circle = (yy - cy)**2 + (xx - cx)**2                                        # eq of circle. squared distance to the center                                        
    circle_mask = circle < radius**2                                            # mask of 1's and 0's                                       
    if output_values:
        values = array[circle_mask]
        return values
    else:
        array_masked = array*circle_mask
        return array_masked


def get_annulus(array, inner_radius, width, output_values=False, 
                output_indices=False):                                          
    """Returns a centerered annulus from a 2d ndarray. All the rest pixels are 
    set to zeros. 
    
    Parameters
    ----------
    array : array_like
        Input 2d array or image. 
    inner_radius : int
        The inner radius of the donut region.
    width : int
        The size of the annulus.
    output_values : {False, True}, optional
        If True returns the values of the pixels in the annulus.
    output_indices : {False, True}, optional
        If True returns the indices inside the annulus.
    
    Returns
    -------
    Depending on output_values, output_indices:
    values : array_like
        1d array with the values of the pixels in the annulus.
    array_masked : array_like
        Input array with the annular mask applied.
    y, x : array_like
        Coordinates of pixels in annulus.
        
    """
    if not array.ndim == 2:
        raise TypeError('Input array is not a frame or 2d array.')
    cy, cx = frame_center(array)
    xx, yy = np.mgrid[:array.shape[0], :array.shape[1]]
    circle = (xx - cx)**2 + (yy - cy)**2                                                                               
    donut_mask = (circle <= (inner_radius + width)**2) & (circle >= inner_radius**2)
    if output_values and not output_indices:
        values = array[donut_mask]
        return values
    elif output_indices and not output_values:      
        indices = np.array(np.where(donut_mask))
        y = indices[0]
        x = indices[1]
        return y, x
    elif output_indices and output_values:
        raise ValueError('output_values and output_indices cannot be both True.')
    else:
        array_masked = array*donut_mask
        return array_masked
    

def get_annulus_quad(array, inner_radius, width, output_values=False):                                          
    """ Returns indices or values in quadrants of a centerered annulus from a 
    2d ndarray. 
    
    Parameters
    ----------
    array : array_like
        Input 2d array or image. 
    inner_radius : int
        The inner radius of the donut region.
    width : int
        The size of the annulus.
    output_values : {False, True}, optional
        If True returns the values of the pixels in the each quadrant instead
        of the indices.
    
    Returns
    -------
    Depending on output_values:
    values : array_like with shape [4, npix]
        Array with the values of the pixels in each quadrant of annulus.
    ind : array_like with shape [4,2,npix]
        Coordinates of pixels for each quadrant in annulus.
        
    """
    if not array.ndim == 2:
        raise TypeError('Input array is not a frame or 2d array.')
    
    cy, cx = frame_center(array)
    xx, yy = np.mgrid[:array.shape[0], :array.shape[1]]
    circle = (xx - cx)**2 + (yy - cy)**2                                                                               
    q1 = (circle >= inner_radius**2) & (circle <= (inner_radius + width)**2) & (xx >= cx) & (yy <= cy)  
    q2 = (circle >= inner_radius**2) & (circle <= (inner_radius + width)**2) & (xx <= cx) & (yy <= cy)
    q3 = (circle >= inner_radius**2) & (circle <= (inner_radius + width)**2) & (xx <= cx) & (yy >= cy)
    q4 = (circle >= inner_radius**2) & (circle <= (inner_radius + width)**2) & (xx >= cx) & (yy >= cy)
    
    if output_values:
        values = [array[mask] for mask in [q1,q2,q3,q4]]
        return np.array(values)
    else:      
        ind = [np.array(np.where(mask)) for mask in [q1,q2,q3,q4]]          
        return np.array(ind)

    
def get_annulus_cube(array, inner_radius, width, output_values=False):     
    """ Returns a centerered annulus from a 3d ndarray. All the rest pixels are 
    set to zeros. 
    
    Parameters
    ----------
    array : array_like
        Input 2d array or image. 
    inner_radius : int
        The inner radius of the donut region.
    width : int
        The size of the annulus.
    output_values : {False, True}, optional
        If True returns the values of the pixels in the annulus.
    
    Returns
    -------
    Depending on output_values:
    values : array_like
        1d array with the values of the pixels in the circular region.
    array_masked : array_like
        Input array with the circular mask applied.

    """
    if not array.ndim == 3:
        raise TypeError('Input array is not a cube or 3d array.')
    arr_annulus = np.empty_like(array)
    if output_values:
        values = [get_annulus(array[i], inner_radius, width, output_values=True) for i in xrange(array.shape[0])]
        return np.array(values)
    else:
        for i in xrange(array.shape[0]):
            arr_annulus[i] = get_annulus(array[i], inner_radius, width)
        return arr_annulus
    

def mask_circle(array, radius):                                      
    """ Masks (sets pixels to zero) a centered circle from a frame or cube. 
    
    Parameters
    ----------
    array : array_like
        Input frame or cube.
    radius : int
        Radius of the circular aperture.
    
    Returns
    -------
    array_masked : array_like
        Masked frame or cube.
        
    """
    if len(array.shape) == 2:
        sy, sx = array.shape
        cy = sy/2
        cx = sx/2
        xx, yy = np.ogrid[:sy, :sx]
        circle = (xx - cx)**2 + (yy - cy)**2               # squared distance to the center                                        
        hole_mask = circle > radius**2                                             
        array_masked = array*hole_mask
        
    if len(array.shape) == 3:
        n, sy, sx = array.shape
        cy = sy/2
        cx = sx/2
        xx, yy = np.ogrid[:sy, :sx]
        circle = (xx - cx)**2 + (yy - cy)**2               # squared distance to the center                                        
        hole_mask = circle > radius**2      
        array_masked = np.empty_like(array)
        for i in xrange(n):
            array_masked[i] = array[i]*hole_mask
        
    return array_masked    
    

        

