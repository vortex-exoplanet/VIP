#! /usr/bin/env python

"""
Module with various functions.
"""

from __future__ import division

__author__ = 'C. Gomez @ ULg', 'O. Wertz'
__all__ = ['pp_subplots',
           'plot_surface',
           'lines_of_code',
           'px2mas']

import os
import numpy as np
from matplotlib.pyplot import (figure, subplot, show, colorbar, rc, axes)
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from ..var import fit_2dgaussian, dist, frame_center
from ..conf import VLT_NACO, LBT


def pp_subplots(*args, **kwargs):
    """ Creating pyplot subplots dynamically. Friendly with jupyter notebooks.
    
    Parameters
    ----------
    rows : how many rows (to make a grid)
    cmap : colormap
    colorb : to attach a colorbar, off by default
    vmax : for stretching the displayed pixels values
    vmin : for stretching the displayed pixels values
    dpi : dots per inch, for plot quality
    title : title of the plot(s), None by default
    noaxis : to remove the axis, on by default
    
    """   
    if kwargs.has_key('rows'):
        rows = kwargs['rows']
    else:
        rows = 1
    if kwargs.has_key('cmap'):
        custom_cmap = kwargs['cmap']
    else:
        custom_cmap = 'CMRmap'
    if kwargs.has_key('colorb'):
        colorb = kwargs['colorb']
    else:
        colorb = False
    if kwargs.has_key('vmax'):
        vmax = kwargs['vmax']
    else:
        vmax = None
    if kwargs.has_key('vmin'):
        vmin = kwargs['vmin']
    else:
        vmin = None
    if kwargs.has_key('dpi'):
        rc("savefig", dpi=kwargs['dpi']) 
    else:
        rc("savefig", dpi=90) 
    if kwargs.has_key('title'):
        tit = kwargs['title']
    else:
        tit = None
    if kwargs.has_key('noaxis'):
        noax = kwargs['noaxis']
    else:
        noax = False
    
    if not isinstance(rows, int):
        raise(TypeError('Rows must be an integer'))
    num_plots = len(args)

    if num_plots%rows==0:
        cols = num_plots/rows
    else:
        cols = (num_plots/rows) + 1
    
    min_size = 4
    max_hor_size = 13
    if rows==0:
        raise(TypeError('Rows must be greater than zero'))
    elif rows==1:
        if cols==1:
            fig = figure(figsize=(min_size, min_size))
        elif cols>1:
            fig = figure(figsize=(max_hor_size, min_size*rows))
    elif rows>1:
        if cols==1:
            fig = figure(figsize=(min_size, 10))
        elif cols>1:
            fig = figure(figsize=(max_hor_size, 10))
    
    if tit is not None:  fig.suptitle(tit, fontsize=14)
    fig.subplots_adjust(wspace=0.1)
    for i,v in enumerate(xrange(num_plots)):
        v += 1
        ax = subplot(rows,cols,v)
        im = ax.imshow(args[i], cmap=custom_cmap, interpolation='nearest', 
                       origin='lower', vmin=vmin, vmax=vmax)      
        if colorb: 
            # create an axes on the right side of ax. The width of cax will be 5%
            # of ax and the padding between cax and ax will be fixed at 0.05 inch. 
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            colorbar(im, ax=ax, cax=cax)
        ax.grid('off')
        if noax:  ax.set_axis_off()
    show()
    


def plot_surface(image, center=None, size=15, output=False, ds9_indexing=False, 
                 **kwargs):
    """
    Create a surface plot from image.
    
    By default, the whole image is plotted. The 'center' and 'size' attributes 
    allow to crop the image.
        
    Parameters
    ----------
    image : numpy.array
        The image as a numpy.array.
    center : tuple of 2 int (optional, default=None)
        If None, the whole image will be plotted. Otherwise, it grabs a square
        subimage at the 'center' (Y,X) from the image.
    size : int (optional, default=15)
        It corresponds to the size of a square in the image.
    output : {False, True}, bool optional
        Whether to output the grids and intensities or not.
    ds9_indexing : {False, True}, bool optional 
        If True the coordinates are in X,Y convention and in 1-indexed format.
    kwargs:
        Additional attributes are passed to the matplotlib figure() and 
        plot_surface() method.        
    
    Returns
    -------
    out : tuple of 3 numpy.array
        x and y for the grid, and the intensity
        
    """        
    if not center:
        size = image.shape[0]
        x = np.outer(np.arange(0,size,1), np.ones(size))
        y = x.copy().T 
        z = image
    else: 
        if ds9_indexing:
            center = (center[0]-1,center[1]-1) 
            cx, cy = center
        else:
            cy, cx = center
        if size % 2:            # if size is odd             
            x = np.outer(np.arange(0,size,1), np.ones(size))
        else:                   # otherwise, size is even
            x = np.outer(np.arange(0,size+1,1), np.ones(size+1))
        y = x.copy().T            
        z = image[cy-size//2:cy+size//2+1,cx-size//2:cx+size//2+1]           
    
    figure(figsize=kwargs.pop('figsize',(5,5)))
    ax = axes(projection='3d')
    ax.plot_surface(x, y, z, rstride=1, cstride=1, linewidth=0, **kwargs) 
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_zlabel('$I(x,y)$')
    ax.set_title('Data')
    show()
    
    if output:
        return (x,y,z)
 

def lines_of_code():
    """ Calculates the lines of code for VORTEX pipeline. Not objective measure 
    of developer's work! (note to self). 
    """
    cur_path = os.path.dirname(os.path.abspath(__file__))
    path = cur_path[:-len('var')]

    ignore_set = set(["__init__.py"])
    
    loclist = []

    for pydir, _, pyfiles in os.walk(path):
        if 'mod_ext_lib/' not in pydir:
            for pyfile in pyfiles:
                if pyfile not in ignore_set and pyfile.endswith(".py"):
                    totalpath = os.path.join(pydir, pyfile)
                    loclist.append((len(open(totalpath,"r").read().splitlines()),
                                       totalpath.split(path)[1]) )

    for linenumbercount, filename in loclist: 
        print "%05d lines in %s" % (linenumbercount, filename)

    msg = "\nTotal: {:} lines in ({:}) excluding external libraries."
    print msg.format(sum([x[0] for x in loclist]), path)


#TODO: move later to a separate module
def px2mas(array, y, x, instrument='naco', fwhm=None):
    """ Returns the distance in milliarcseconds of a source in a frame for a 
    given instrument and pixel coordinates. 
    """
    if fwhm:
        fy, fx = fit_2dgaussian(array, y, x, fwhm=fwhm)
    else:
        if instrument=='naco': 
            INST = VLT_NACO
        elif instrument=='lmircam':
            INST = LBT
        else:
            raise TypeError('Instrument not recognized.')
        fwhm = INST['lambdal']/INST['diam']*206265/INST['plsc'] 
        fy, fx = fit_2dgaussian(array, y, x, fwhm)
    
    cy, cx = frame_center(array)
    dist_px = dist(cy,cx,fy,fx)
    dist_arcs = dist_px*INST['plsc']
    dist_mas = dist_arcs/0.001
    return dist_mas


 



    
