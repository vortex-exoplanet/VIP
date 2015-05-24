#! /usr/bin/env python

"""
Module with various functions.
"""

from __future__ import division

__author__ = 'C. Gomez @ ULg'
__all__ = ['pp_subplots',
           'lines_of_code',
           'px2mas']

import os
import glob
import re
import numpy as np
from matplotlib.pyplot import (figure, subplot, show, colorbar, close, imshow,
                               xlim, ylim, matshow)
from subprocess import call     
from ..var import (fit_2dgaussian, dist, frame_center, get_circle, 
                        get_annulus)
from ..conf import VLT_NACO, LBT


def pp_subplots(*args, **kwargs):
    """ Creating pyplot subplots dynamically. Friendly with jupyter notebooks.
    """   
    if kwargs.has_key('rows'):
        rows = kwargs['rows']
    else:
        rows = 1
    if kwargs.has_key('cmap'):
        custom_cmap = kwargs['cmap']
    else:
        custom_cmap = 'CMRmap'
    if kwargs.has_key('size'):
        min_size = kwargs['size']
    else:
        min_size = 4
    
    if not isinstance(rows, int):
        raise(TypeError('Rows must be an integer'))
    num_plots = len(args)

    if num_plots%rows==0:
        cols = num_plots/rows
    else:
        cols = (num_plots/rows) + 1
    
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
            
    fig.subplots_adjust(wspace=0.1)
    for i,v in enumerate(xrange(num_plots)):
        v += 1
        ax = subplot(rows,cols,v)
        im = ax.imshow(args[i], cmap=custom_cmap, interpolation='nearest', 
                       origin='lower')
        colorbar(im, ax=ax)
        ax.grid('off')
    show()


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



    
