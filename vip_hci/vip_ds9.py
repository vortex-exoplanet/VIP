#! /usr/bin/env python

"""
Module with a class for creating a DS9 window through pyds9.
"""
__author__ = 'Carlos Alberto Gomez Gonzalez'

import warnings
from .hci_dataset import Dataset, Frame
try:
    import pyds9
    no_pyds9 = False
    __all__ = ['Ds9Window']
except ImportError:
    msg = "Pyds9 not available."
    warnings.warn(msg, ImportWarning)
    no_pyds9 = True
    __all__ = []


class Ds9Window(object):
    """ Creates a DS9 window (named 'VIP_ds9') using pyDS9 functionality.

    The methods of this class allow interacting with the DS9 window. It uses
    XPA under the hood, which mean that using the ``set`` and ``get`` methods
    one can have fine control of the DS9 window. More info here:
    http://ds9.si.edu/doc/ref/xpa.html.

    """
    def __init__(self,wait=10):
        """ __init__ method.
        """
        self.window_name = 'VIP_ds9'
        self.window = pyds9.DS9(self.window_name,wait=wait)

    def clear_frames(self):
        """ Clears all frames. """
        self.window = pyds9.DS9(self.window_name)
        self.window.set('frame clear all')

    def create_frame(self):
        """ Creates a new frame (if no window it creates one). """
        self.window = pyds9.DS9(self.window_name)
        self.window.set('frame new')

    def cmap(self, value=None):
        """ Controls the colormap for the current frame. The colormap name is
        not case sensitive. A valid contrast value is  from 0 to 10 and bias
        value from 0 to 1.

        Parameters
        ----------
        value : str, optional
            Value to be passed to the cmap command.

        Notes
        -----
        syntax:

        .. code-block:: none

            cmap [<colormap>]
                 [file]
                 [load <filename>]
                 [save <filename>]
                 [invert yes|no]
                 [value <constrast> <bias>]
                 [tag [load|save] <filename>]
                 [tag delete]
                 [match]
                 [lock [yes|no]]
                 [open|close]
        """
        self.window = pyds9.DS9(self.window_name)
        self.window.set('cmap ' + str(value))

    def crosshair_get(self):
        """ Gets the physical coordinates (x,y) of the crosshair.

        Returns
        -------
        x : float, optional
            Crosshair x coordinate.
        y : float, optional
            Crosshair y coordinate.
        """
        self.window = pyds9.DS9(self.window_name)
        coor = self.window.get('crosshair')
        x = float(coor.split()[0])
        y = float(coor.split()[1])
        return x, y

    def crosshair_set(self, x=None, y=None):
        """ Sets the crosshair in given coordinates.

        Parameters
        ----------
        x : float, optional
            Crosshair x coordinate.
        y : float, optional
            Crosshair y coordinate.
        """
        self.window = pyds9.DS9(self.window_name)
        if x and y:
            self.window.set('crosshair '+str(x)+' '+str(y)+' physical')
        else:
            self.window.set('mode crosshair')

    def delete_frame(self, all_frames=False):
        """ Deletes last frame or all.

        Parameters
        ----------
        all_frames : bool, optional
            If True all frames are deleted, otherwise only the active one.
        """
        self.window = pyds9.DS9(self.window_name)
        if all_frames:
            self.window.set('frame delete all')
        else:
            self.window.set('frame delete')

    def display(self, *arrays, **kwargs):
        """ Displays one or multiple arrays (listed in ``*arrays``).

        Parameters
        ----------
        *arrays : list of arrays
            The arrays to be displayed.
        **kwargs : dictionary, optional
            Only one parameter is admitted: ``keep_frames``. When
            ``keep_frames`` is set to True (default), the new arrays are
            displayed in the existing window (creating new ds9 frames).
            Otherwise the existing ds9 frames are not preserved.
        """
        self.window = pyds9.DS9(self.window_name)

        if kwargs.get("keep_frames", True):
            self.delete_frame(all_frames=True)

        self.tile('grid')
        for i, array in enumerate(arrays):
            if i == 0:
                self.create_frame()
                if isinstance(array, Dataset):
                    self.window.set_np2arr(array.cube)
                if isinstance(array, Frame):
                    self.window.set_np2arr(array.image)
                else:
                    self.window.set_np2arr(array)
            else:
                self.window.set('frame new')
                if isinstance(array, Dataset):
                    self.window.set_np2arr(array.cube)
                if isinstance(array, Frame):
                    self.window.set_np2arr(array.image)
                else:
                    self.window.set_np2arr(array)

    def get(self, paramlist):
        """ XPA get command. Gets data from ds9. See:
        http://ds9.si.edu/doc/ref/xpa.html
        """
        self.window = pyds9.DS9(self.window_name)
        return self.window.get(paramlist)

    def lock(self, scale=True, colorbar=True, crosshair=True, slices=True):
        """ Locks the scaling, colorbar, crosshair position, or slice in cube
        for all existing ds9 frames (wrt the active one). """
        self.window = pyds9.DS9(self.window_name)
        if scale:
            self.window.set('lock scale yes')
        if colorbar:
            self.window.set('lock colorbar yes')
        if crosshair:
            self.window.set('lock crosshair image')
        if slices:
            self.window.set('lock slice image')

    def pan(self, x=None, y=None):
        """ Sets the mode to pan or pans to given coordinates.

        Parameters
        ----------
        x : float, optional
            X coordinate.
        y : float, optional
            Y coordinate.
        """
        self.window = pyds9.DS9(self.window_name)
        if x and y:
            self.window.set('pan to '+str(x)+' '+str(y))
        else:
            self.window.set('mode pan')

    def rotate(self, value=None):
        """ Rotates with a given angle.

        Parameters
        ----------
        value : float, optional
            Angle.
        """
        self.window = pyds9.DS9(self.window_name)
        if value:
            self.window.set('rotate '+str(value))
        else:
            self.window.set('rotate open')

    def scale(self, value=None):
        """ Controls the limits and color scale distribution.

        Parameters
        ----------
        value : str, optional
            Controls the scaling.

        Notes
        -----
        Syntax:

        .. code-block:: none

            scale [linear|log|pow|sqrt|squared|asinh|sinh|histequ]
                  [log exp <value>]
                  [datasec yes|no]
                  [limits <minvalue> <maxvalue>]
                  [mode minmax|<value>|zscale|zmax]
                  [scope local|global]
                  [match]
                  [match limits]
                  [lock [yes|no]]
                  [lock limits [yes|no]]
                  [open|close]
        """
        self.window = pyds9.DS9(self.window_name)
        if value:
            self.window.set('scale '+str(value))
        else:
            self.window.set('scale open')

    def set(self, paramlist, data=None, data_func=-1):
        """ XPA set command. Sends data or commands to ds9. See:
        http://ds9.si.edu/doc/ref/xpa.html
        """
        self.window = pyds9.DS9(self.window_name)
        self.window.set(paramlist, data, data_func)

    def tile(self, mode='column'):
        """ Controls the tile display mode.

        Parameters
        ----------
        mode : str, optional
            The mode used for tiling the frames.

        Notes
        -----
        Syntax:

        .. code-block:: none

            tile []
                 [yes|no]
                 [mode grid|column|row]
                 [grid]
                 [grid mode automatic|manual]
                 [grid direction x|y]
                 [grid layout <col> <row>]
                 [grid gap <pixels>]
                 [row]
                 [column]
        """
        self.window = pyds9.DS9(self.window_name)
        if mode == 'off':
            self.window.set('tile off')
        else:
            self.window.set('tile on')
            self.window.set('tile mode ' + mode)

    def unlock(self, scale=True, colorbar=True, crosshair=True, slices=True):
        """ The opposite of the ``lock`` method.
        """
        self.window = pyds9.DS9(self.window_name)
        if scale:
            self.window.set('lock scale no')
        if colorbar:
            self.window.set('lock colorbar no')
        if crosshair:
            self.window.set('lock crosshair none')
        if slices:
            self.window.set('lock slice none')

    def zoom(self, value='to fit'):
        """ Controls the current zoom value for the current frame.

        Parameters
        ----------
        value : str or int
            Parameters of the zoom command.

        Notes
        -----
        syntax:

        .. code-block:: none

            zoom [<value>]
                 [<value> <value>]
                 [to <value>]
                 [to <value> <value>]
                 [in]
                 [out]
                 [to fit]
                 [open|close]
        """
        self.window = pyds9.DS9(self.window_name)
        self.window.set('zoom ' + str(value))



