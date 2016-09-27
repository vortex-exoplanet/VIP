#! /usr/bin/env python

"""
Module with various DS9 related functions.
"""

__author__ = 'C. Gomez @ ULg'

import numpy as np
from ..exlib.ds9 import setup 

# tests is DS9/XPA are already installed in your system
error = setup(doRaise=False)

if not error:
    from ..exlib.ds9 import DS9Win
    __all__ = ['vipDS9']
else:
    __all__ = []
    print error
    print 'vipDS9 class that controls DS9 will not be available \n'
    


class vipDS9(object):
    """ Creates a DS9 window. The methods of this class allow interacting with
    the instance of the DS9 window. The methods xpaset and xpaget will allow
    fine control of DS9 (more info: http://ds9.si.edu/doc/ref/xpa.html).
    
    Methods
    -------
    clear_frames : Clears all frames
    create_frame : Creates a new frame (if there's no window it creates one)
    crosshair : Sets the crosshair in given coordinates
    delete_frame : Deletes last frame or all frames
    delete_region : Deletes selected region or all regions
    display : Displays one or multiple arrays
    file : Displays a FITS file
    get_crosshair : Gets the physical coordinates of the crosshair
    lock : Locks all frames to the current one
    pan : Sets the mode to pan or pans to given coordinates
    region : Manipulates regions in VIP ds9 window
    rotate : Rotates with a given angle
    save_array : Saves current ds9 frame array in given path as FITS
    scale : Scales to a given value
    tile : Sets tile mode
    zoom : Zooms to a given value or to fit
    xpaget : XPA get commands
    xpaset : XPA set commands
    
    """
    
    def __init__(self, name='VIP', gridon=True):
        self.name = name
        self.win = DS9Win(self.name, doOpen=True)
        self.gridon = gridon
        if self.gridon:
            self.tile('grid')

    def clear_frames(self):
        """ Clears all frames. """       
        # creates/detects the VIP window
        self.win = DS9Win(self.name, doOpen=True)
        self.win.xpaset('frame clear all')
        
    def create_frame(self):
        """ Creates a new frame (if no window it creates one). """
        self.win = DS9Win(self.name, doOpen=True)
        self.win.xpaset('frame new')

    def crosshair(self, x=None, y=None):
        """ Sets the crosshair in given coordinates. """
        self.win = DS9Win(self.name, doOpen=True)
        if x and y:
            self.win.xpaset('crosshair '+str(x)+' '+str(y)+' physical')
        else:
            self.win.xpaset('mode crosshair')

    def delete_frame(self, allfr=False):
        """ Deletes last frame or all. """       
        self.win = DS9Win(self.name, doOpen=True)
        if allfr:
            self.win.xpaset('frame delete all')
        else:
            self.win.xpaset('frame delete')
            
    def delete_region(self, allreg=False):
        """ Deletes selected region or all. """
        self.win = DS9Win(self.name, doOpen=True)
        if allreg:
            self.win.xpaset('regions delete all')
        else:
            self.win.xpaset('regions delete select')      
         
    def display(self, *arrays, **kwargs):
        """ Displays one or multiple arrays (listed in *arrays). There are 
        several possible parameters in **kwargs as listed below.
        
        When *keepwin* exists and is equal to True, the new arrays are displayed
        in an existing window (creating new ds9 frames). If no *keepwin* is 
        given or if it's equal to False, then the existing ds9 frames are not
        preserved.
        """
        self.win = DS9Win(self.name, doOpen=True) # creates/detects the VIP ds9
        if kwargs.has_key('keepwin'):
            if kwargs['keepwin']:  pass
            else:  self.delete_frame(allfr=True)
        else: 
            self.delete_frame(allfr=True)
            
        self.create_frame()
        self.tile()
        for i, array in enumerate(arrays):
            if i==0: 
                self.win.showArray(array)                   
            else:    
                self.create_frame()
                self.win.showArray(array)
            
    def file(self, fitsfilename):
        """ Displays a fits file. """
        self.win = DS9Win(self.name, doOpen=True)
        self.win.showFITSFile(fitsfilename)
        
    def get_crosshair(self):
        """ Gets the physical coordinates of the crosshair. """
        self.win = DS9Win(self.name, doOpen=True)
        return self.win.xpaget('crosshair')
  
    def lock(self, scale=True, colorbar=True, crosshair=True, slices=True):
        """ Locks all frames to the current one. """
        self.win = DS9Win(self.name, doOpen=True)
        if scale:
            self.win.xpaset('lock scale yes') 
        if colorbar:
            self.win.xpaset('lock colorbar yes')        
        if crosshair:
            self.win.xpaset('lock crosshair image') 
        if slices:
            self.win.xpaset('lock slice image') 
   
    def pan(self, x=None, y=None):
        """ Sets the mode to pan or pans to given coordinates. """
        self.win = DS9Win(self.name, doOpen=True)
        if x and y:
            self.win.xpaset('pan to '+str(x)+' '+str(y))
        else:
            self.win.xpaset('mode pan') 
  
    def region(self, cmd=None, savein=None, loadfrom=None, liston=False, 
               get=False):
        """ Manipulates regions in VIP ds9 window. 
        
        *cmd* is a command to send to ds9, example: 
        >>> ds9.region(cmd='circle 2 2 20 # color=red')
        See: http://ds9.si.edu/doc/ref/xpa.html#regions
        for the full list of arguments and options.  
            
        *savein* argument saves regions in the current frame to a given path.
        
        *loadfrom* argument loads regions from the given file.
        
        When *liston* is True it will list the existing regions in the frame. 
        
        When *get* is True it will get the regions from ds9 into a python 
        variable.
        
        """
        self.win = DS9Win(self.name, doOpen=True)
        self.win.xpaset('mode region')
        if cmd is not None:
            print 'Executing region command'
            self.win.xpaset('regions command "{'+str(cmd)+'}"')
        if liston:
            self.win.xpaset('regions list')
        if savein is not None:
            print 'Saving region file in {}'.format(savein)
            self.win.xpaset('regions save '+str(savein))
        if loadfrom is not None:
            print 'Loading region file from {}'.format(loadfrom)
            self.win.xpaset('regions load '+str(loadfrom))
        if get:
            print 'Getting regions'
            return self.win.xpaget('regions') 
        
    def rotate(self, value=None):
        """ Rotates with a given angle. """
        self.win = DS9Win(self.name, doOpen=True)
        if value:
            self.win.xpaset('rotate '+str(value))
        else:
            self.win.xpaset('rotate open') 
        
    def save_array(self, fname='./ds9array.fits'):
        """ Saves current ds9 frame array in given path as FITS. """
        self.win = DS9Win(self.name, doOpen=True)
        return np.array(self.win.xpaset('save '+fname)) 

    def scale(self, value=None):
        """ Scales to a given value. """
        self.win = DS9Win(self.name, doOpen=True)
        if value:
            self.win.xpaset('scale '+str(value))
        else:
            self.win.xpaset('scale open') 
        
    def tile(self, mode='column'):
        """ Sets tile frames. Mode can be: off, row, column or grid. """       
        self.win = DS9Win(self.name, doOpen=True)
        if mode=='off':
            self.win.xpaset('tile off')
        else:
            self.win.xpaset('tile on')
            self.win.xpaset('tile mode '+mode)

    def unlock(self, scale=True, colorbar=True, crosshair=True, slices=True):
        """ Locks all frames to the current one. """
        self.win = DS9Win(self.name, doOpen=True)
        if scale:
            self.win.xpaset('lock scale no') 
        if colorbar:
            self.win.xpaset('lock colorbar no')        
        if crosshair:
            self.win.xpaset('lock crosshair none') 
        if slices:
            self.win.xpaset('lock slice none') 

    def zoom(self, value='to fit'):
        """ Zooms to a given value or to fit. """
        self.win = DS9Win(self.name, doOpen=True)
        self.win.xpaset('zoom '+str(value)) 
    
    def xpaget(self, cmd):
        """ XPA get commands. For fine control of DS9 see: 
        http://ds9.si.edu/doc/ref/xpa.html
        """
        self.win = DS9Win(self.name, doOpen=True)       
        return self.win.xpaget(cmd)
    
    def xpaset(self, cmd, data=None, dataFunc=None):
        """ XPA set commands. For fine control of DS9 see:
        http://ds9.si.edu/doc/ref/xpa.html
        """
        self.win = DS9Win(self.name, doOpen=True)       
        self.win.xpaset(cmd, data, dataFunc)
        

        
        
            