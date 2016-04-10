#! /usr/bin/env python

"""
Module with various functions.
"""

from __future__ import division

__author__ = 'C. Gomez @ ULg', 'O. Wertz'
__all__ = ['pp_subplots',
           'plot_surface',
           'get_fwhm',
           'lines_of_code']

import os
import numpy as np
from matplotlib.pyplot import (figure, subplot, show, colorbar, rc, axes)
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from .shapes import frame_center


def get_fwhm(lambd, diameter, pxscale):
    """ Returnes the instrument FWHM [px] given the wavelenght [m], diameter [m] 
    and plate/pixel scale [arcs/px]. In vip/conf/param.py can be found
    dictionaries with the parameters for different instruments.                           
    """
    fwhm = lambd/diameter*206265/pxscale                    
    return fwhm 


def pp_subplots(*args, **kwargs):
    """ Creates pyplot subplots dynamically. It allows displaying images in 
    jupyter notebooks easily, with shortcuts to functions for controlling the
    settings of the plot. The function also fixes some annoying defaults of 
    pyplot that do not work well for astro-data.
    
    Parameters in **kwargs
    ----------------------
    arrow : show an arrow pointing to input px coordinates
    cmap : colormap to be used, CMRmap by default
    cbar : to attach a colorbar, on by default
    dpi : dots per inch, for plot quality
    grid : for showing a grid over the image, off by default
    horsp : horizontal gap between subplots
    label : text for annotating on subplots
    labelpad : padding of the label from the left bottom corner
    labelsize : size of the labels
    noaxis : to remove the axis, on by default
    rows : how many rows (subplots in a grid)
    title : title of the plot(s), None by default
    vmax : for stretching the displayed pixels values
    vmin : for stretching the displayed pixels values
    versp : vertical gap between subplots

    """
    parlist = ['arrow', 'cmap', 'cbar', 'dpi', 'grid', 'horsp', 'label', 'labelpad', 
               'labelsize', 'noaxis', 'rows', 'title', 'vmax', 'vmin', 'versp']
    
    for key in kwargs.iterkeys():
        if key not in parlist:
            print "Parameter '{:}' not recognized".format(key)
            print "Available parameters are: {:}".format(parlist)
    
    if kwargs.has_key('label'):  
        label = kwargs['label']
        if len(label) != len(args):
            print "The number of labels doesn't match the number of subplots"
            label = None
    else:  label = None
    
    if kwargs.has_key('arrow'):
        if not isinstance(kwargs['arrow'], tuple):
            print "Arrow must be a tuple (X,Y)"
            show_arrow = False
        else:
            coor_arrow = kwargs['arrow']
            show_arrow = True
    else:  
        show_arrow = False
    
    if kwargs.has_key('labelsize'):  labelsize = kwargs['labelsize']
    else:  labelsize = 12
    
    if kwargs.has_key('labelpad'):  labelpad = kwargs['labelpad']
    else:  labelpad = 5
    
    if kwargs.has_key('rows'):  rows = kwargs['rows']
    else:  rows = 1
    
    if kwargs.has_key('cmap'):  custom_cmap = kwargs['cmap']
    else:  custom_cmap = 'CMRmap'
    
    if kwargs.has_key('cbar'):  colorb = kwargs['cbar']
    else:  colorb = True
    
    if kwargs.has_key('grid'):  grid = kwargs['grid']
    else:  grid = False
    
    if kwargs.has_key('vmax'):  vmax = kwargs['vmax']
    else:  vmax = None
    
    if kwargs.has_key('vmin'):  vmin = kwargs['vmin']
    else:  vmin = None
    
    if kwargs.has_key('dpi'):  rc("savefig", dpi=kwargs['dpi']) 
    else:  rc("savefig", dpi=90) 
    
    if kwargs.has_key('title'):  tit = kwargs['title']
    else:  tit = None
    
    if kwargs.has_key('noaxis'):  noax = kwargs['noaxis']
    else:  noax = False
    
    if kwargs.has_key('horsp'):  hor_spacing = kwargs['horsp']
    else:  hor_spacing = 0.3
    
    if kwargs.has_key('versp'):  ver_spacing = kwargs['versp']
    else:  ver_spacing = 0
    
    if not isinstance(rows, int):
        raise(TypeError('Rows must be an integer'))
    num_plots = len(args)

    if num_plots%rows==0:
        cols = num_plots/rows
    else:
        cols = (num_plots/rows) + 1
             
    subplot_size = 4
    if rows==0:
        raise(TypeError('Rows must be greater than zero'))
    fig = figure(figsize=(cols*subplot_size, rows*subplot_size))
    
    if tit is not None:  fig.suptitle(tit, fontsize=14)
    
    for i,v in enumerate(xrange(num_plots)):
        v += 1
        ax = subplot(rows,cols,v)
        ax.set_aspect('equal')
        im = ax.imshow(args[i], cmap=custom_cmap, interpolation='nearest', 
                       origin='lower', vmin=vmin, vmax=vmax)  
        if show_arrow:
            leng = 20
            ax.arrow(coor_arrow[0]+2*leng, coor_arrow[1], -leng, 0, color='white', 
                     head_width=6, head_length=4, width=2, length_includes_head=True)
            
        if label is not None:
            ax.annotate(label[i], xy=(labelpad,labelpad), color='white', 
                        xycoords='axes pixels', weight='bold', size=labelsize)    
        if colorb: 
            # create an axes on the right side of ax. The width of cax will be 5%
            # of ax and the padding between cax and ax will be fixed at 0.05 inch. 
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cb = colorbar(im, ax=ax, cax=cax, drawedges=False)
            cb.outline.set_linewidth(0.1)
            cb.ax.tick_params(labelsize=8) 
        if grid:  
            ax.grid('on', which='both', color='w')
        else:
            ax.grid('off')
        if noax:  ax.set_axis_off()
    
    fig.subplots_adjust(wspace=hor_spacing, hspace=ver_spacing)
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
    if center is not None: 
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
    else:
        cy, cx = frame_center(image)
        if size is not None:
            if size % 2:
                x = np.outer(np.arange(0,size,1), np.ones(size))
            else: 
                x = np.outer(np.arange(0,size+1,1), np.ones(size+1))
            y = x.copy().T
            z = image[cy-size//2:cy+size//2+1,cx-size//2:cx+size//2+1]
        else:
            size = image.shape[0]
            x = np.outer(np.arange(0,size,1), np.ones(size))
            y = x.copy().T 
            z = image        
    
    figure(figsize=kwargs.pop('figsize',(6,6)))
    ax = axes(projection='3d')
    ax.dist = 12
    ax.plot_surface(x, y, z, rstride=1, cstride=1, linewidth=0, **kwargs) 
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_zlabel('$I(x,y)$')
    ax.xaxis._axinfo['label']['space_factor'] = 2
    ax.yaxis._axinfo['label']['space_factor'] = 2
    ax.zaxis._axinfo['label']['space_factor'] = 3.5
    #ax.zaxis._axinfo['ticklabel']['space_factor'] = 1.5
    ax.set_title('Data')
    show()
    
    if output:
        return (x,y,z)
 

def lines_of_code():
    """ Calculates the lines of code for VIP pipeline. Not objective measure 
    of developer's work! (note to self). 
    """
    cur_path = os.path.dirname(os.path.abspath(__file__))
    path = cur_path[:-len('var')]

    ignore_set = set(["__init__.py"])
    
    loclist = []

    for pydir, _, pyfiles in os.walk(path):
        if 'exlib/' not in pydir:
            for pyfile in pyfiles:
                if pyfile not in ignore_set and pyfile.endswith(".py"):
                    totalpath = os.path.join(pydir, pyfile)
                    loclist.append((len(open(totalpath,"r").read().splitlines()),
                                       totalpath.split(path)[1]) )

    for linenumbercount, filename in loclist: 
        print "%05d lines in %s" % (linenumbercount, filename)

    msg = "\nTotal: {:} lines in ({:}) excluding external libraries."
    print msg.format(sum([x[0] for x in loclist]), path)




 



    
