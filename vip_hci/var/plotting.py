#! /usr/bin/env python

"""
Module with 2d/3d plotting functions.
"""

from __future__ import division, print_function

__author__ = 'Carlos Alberto Gomez Gonzalez, O. Wertz'
__all__ = ['pp_subplots',
           'plot_surface',
           'save_animation']

import os
import shutil
import numpy as np
from subprocess import Popen
from matplotlib.pyplot import (figure, subplot, show, colorbar, axes, Circle,
                               savefig, close)
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from .shapes import frame_center


def save_animation(data, anim_path=None, data_step_range=None, label=None,
                   labelpad=10, label_step_range=None, delay=50, format='gif',
                   **kwargs):
    """ Generates a matplotlib animation from a ``data`` 3d array and saves it
    to disk using ImageMagick's convert command (it must be installed otherwise
    a ``FileNotFoundError`` will be raised).

    Parameters
    ----------
    data : array_like
        3d array.
    anim_path : str, optional
        The animation filename/path. If None then the animation will be called
        animation.``format`` and will be saved in the current directory.
    data_step_range : tuple, optional
        Tuple of 1, 2 or 3 values that creates a range for slicing the ``data``
        cube.
    label : str, optional
        Label to be overlaid on top of each frame of the animation. If None,
        then 'frame #' will be used.
    labelpad : int, optional
        Padding of the label from the left bottom corner. 10 by default.
    label_step_range : tuple, optional
        Tuple of 1, 2 or 3 values that creates a range for customizing the label
        overlaid on top of the image.
    delay : int, optional
        Delay for displaying the frames in the animation sequence.
    format : str, optional
        Format of the saved animation. By default 'gif' is used. Other formats
        supported by ImageMagick are valid, such as 'mp4'.
    **kwargs : dictionary, optional
        Arguments to be passed to ``pp_subplots`` to customize the plot.

    """
    if not (isinstance(data, np.ndarray) and data.ndim == 3):
        raise ValueError('`data` must be a 3d numpy array')

    dir_path = './animation_temp/'
    if anim_path is None:
        anim_path = './animation'

    if data_step_range is None:
        data_step_range = range(0, data.shape[0], 1)
    else:
        if not isinstance(data_step_range, tuple):
            msg = '`data_step_range` must be a tuple with 1, 2 or 3 values'
            raise ValueError(msg)
        if len(data_step_range) == 1:
            data_step_range = range(data_step_range)
        elif len(data_step_range) == 2:
            data_step_range = range(data_step_range[0], data_step_range[1])
        elif len(data_step_range) == 3:
            data_step_range = range(data_step_range[0],
                                    data_step_range[1],
                                    data_step_range[2])

    if label_step_range is None:
        label_step_range = data_step_range
    else:
        if not isinstance(label_step_range, tuple):
            msg = '`label_step_range` must be a tuple with 1, 2 or 3 values'
            raise ValueError(msg)
        if len(label_step_range) == 1:
            label_step_range = range(label_step_range)
        elif len(label_step_range) == 2:
            label_step_range = range(label_step_range[0], label_step_range[1])
        elif len(label_step_range) == 3:
            label_step_range = range(label_step_range[0],
                                     label_step_range[1],
                                     label_step_range[2])

    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
        print('Overwriting content in ' + dir_path)
    os.mkdir(dir_path)

    for i, labstep in zip(data_step_range, list(label_step_range)):
        if label is None:
            label = 'frame '
        savelabel = dir_path + label + str(i + 100)
        pp_subplots(data[i], save=savelabel, label=[label+str(labstep + 1)],
                    labelpad=labelpad, **kwargs)
    try:
        filename = anim_path + '.' + format
        Popen(['convert', '-delay', str(delay), dir_path + '*.png', filename])
        print('Animation successfully saved to disk as ' + filename)
    except FileNotFoundError:
        print('ImageMagick convert command could not be found')


def pp_subplots(*data, **kwargs):
    """ Wrapper for easy creation of pyplot subplots. It is convenient for
    displaying VIP images in jupyter notebooks.

    Parameters
    ----------
    data : list
        List of 2d arrays or a single 3d array to be plotted.

    Parameters in **kwargs
    ----------------------
    angscale : bool
        If True, the axes are displayed in angular scale (arcsecs).
    angticksep : int
        Separation for the ticks when using axis in angular scale.
    arrow : bool
        To show an arrow pointing to input px coordinates.
    arrowalpha : float
        Alpha transparency for the arrow.
    arrowlength : int
        Length of the arrow, 20 px by default.
    arrowshiftx : int
        Shift in x of the arrow pointing position, 5 px by default.
    axis : bool
        Show the axis, on by default.
    circle : tuple or list of tuples
        To show a circle at given px coordinates. The circles are shown on all
        subplots.
    circlealpha : float or list of floats
        Alpha transparencey for each circle.
    circlelabel : bool
        Whether to show the coordinates of each circle.
    circlerad : int
        Radius of the circle, 6 px by default.
    cmap : str
        Colormap to be used, 'viridis' by default.
    colorb : bool
        To attach a colorbar, on by default.
    cross : tuple of float
        If provided, a crosshair is displayed at given px coordinates.
    crossalpha : float
        Alpha transparency of thr crosshair.
    dpi : int
        Dots per inch, for plot quality.
    getfig : bool
        Returns the matplotlib figure.
    grid : bool
        If True, a grid is displayed over the image, off by default.
    gridalpha : float
        Alpha transparency of the grid.
    gridcolor : str
        Color of the grid lines.
    gridspacing : int
        Separation of the grid lines in pixels.
    horsp : float
        Horizontal gap between subplots.
    label : str or list of str
        Text for annotating on subplots.
    labelpad : int
        Padding of the label from the left bottom corner. 5 by default.
    labelsize : int
        Size of the labels.
    log : bool
        Log colorscale.
    maxplots : int
        When the input (*args) is a 3d array, maxplots sets the number of
        cube slices to be displayed.
    pxscale : float
        Pixel scale in arcseconds/px. Default 0.01 for Keck/NIRC2.
    rows : int
        How many rows (subplots in a grid).
    save : str
        If a string is provided the plot is saved using this as the path.
    showcent : bool
        To show a big crosshair at the center of the frame.
    title : str
        Title of the plot(s), None by default.
    vmax : int
        For stretching the displayed pixels values.
    vmin : int
        For stretching the displayed pixels values.
    versp : float
        Vertical gap between subplots.

    """
    parlist = ['angscale', 'angticksep', 'arrow', 'arrowalpha', 'arrowlength',
               'arrowshiftx', 'axis', 'circle', 'circlealpha', 'circlerad',
               'circlelabel', 'cmap', 'colorb', 'cross', 'crossalpha', 'dpi',
               'getfig', 'grid', 'gridalpha', 'gridcolor', 'gridspacing',
               'horsp', 'label', 'labelpad', 'labelsize', 'log', 'maxplots',
               'pxscale', 'rows', 'save', 'showcent', 'title', 'vmax', 'vmin',
               'versp']

    for key in kwargs.keys():
        if key not in parlist:
            print("Parameter '{}' not recognized".format(key))
            print("Available parameters are: {}".format(parlist))

    # GEOM ---------------------------------------------------------------------
    num_plots = len(data)
    if num_plots == 1:
        if data[0].ndim == 3:
            data = data[0]
            maxplots = kwargs.get("maxplots", 10)
            num_plots = min(data.shape[0], maxplots)
    elif num_plots > 1:
        for i in range(num_plots):
            if data[i].ndim != 2:
                msg = "Wrong input. Must be either several 2d arrays (images) "
                msg += "or a single 3d array"
                raise TypeError(msg)

    rows = kwargs.get("rows", 1)

    if num_plots % rows == 0:
        cols = num_plots / rows
    else:
        cols = (num_plots / rows) + 1

    # CIRCLE -------------------------------------------------------------------
    if 'circle' in kwargs:
        coor_circle = kwargs['circle']
        if isinstance(coor_circle, (list, tuple)):
            show_circle = True
            if isinstance(coor_circle[0], tuple):
                n_circ = len(coor_circle)
            else:
                n_circ = 1
                coor_circle = [coor_circle] * n_circ
        else:
            print("Circle must be a tuple (X,Y) or tuple/list of tuples (X,Y)")
            show_circle = False
    else:
        show_circle = False

    if 'circlerad' in kwargs:
        # single value is provided, used for all circles
        if isinstance(kwargs['circlerad'], (float, int)):
            circle_rad = [kwargs['circlerad']] * n_circ
        # a different value for each circle
        elif isinstance(kwargs['circlerad'], tuple):
            circle_rad = kwargs['circlerad']
        else:
            print("Circlerad must be a float or tuple of floats")
    else:
        if show_circle:
            circle_rad = [6] * n_circ

    if 'circlealpha' in kwargs:
        circle_alpha = kwargs['circlealpha']
        # single value is provided, used for all the circles
        if isinstance(circle_alpha, (float, int)):
            circle_alpha = [circle_alpha] * n_circ
    else:
        if show_circle:
            # no alpha is provided, 0.8 is used for all of them
            circle_alpha = [0.8] * n_circ

    circle_label = kwargs.get('circlelabel', False)

    # ARROW --------------------------------------------------------------------
    if 'arrow' in kwargs:
        if not isinstance(kwargs['arrow'], tuple):
            print("Arrow must be a tuple (X,Y)")
            show_arrow = False
        else:
            coor_arrow = kwargs['arrow']
            show_arrow = True
    else:
        show_arrow = False

    arrow_shiftx = kwargs.get('arrowshiftx', 5)
    arrow_length = kwargs.get('arrowlength', 20)
    arrow_alpha = kwargs.get('arrowalpha', 0.8)

    # LABEL --------------------------------------------------------------------
    if 'label' in kwargs:
        label = kwargs['label']
        if len(label) != num_plots:
            print("Label list does not have enough items")
            label = None
    else:
        label = None

    labelsize = kwargs.get('labelsize', 12)
    labelpad = kwargs.get('labelpad', 5)

    # GRID ---------------------------------------------------------------------
    grid = kwargs.get('grid', False)
    grid_color = kwargs.get('gridcolor', '#f7f7f7')
    grid_spacing = kwargs.get('gridspacing', None)
    grid_alpha = kwargs.get('gridalpha', 0.4)

    # VMAX-VMIN ----------------------------------------------------------------
    if 'vmax' in kwargs:
        if isinstance(kwargs['vmax'], (tuple, list)):
            if len(kwargs['vmax']) != num_plots:
                print("Vmax does not list enough items, setting all to None")
                vmax = [None] * num_plots
            else:
                vmax = kwargs['vmax']
        else:
            vmax = [kwargs['vmax']]*num_plots
    else:
        vmax = [None] * num_plots

    if 'vmin' in kwargs:
        if isinstance(kwargs['vmin'], (tuple, list)):
            if len(kwargs['vmin']) != num_plots:
                print("Vmin does not list enough items, setting all to None")
                vmin = [None]*num_plots
            else:
                vmin = kwargs['vmin']
        else:
            vmin = [kwargs['vmin']] * num_plots
    else:
        vmin = [None]*num_plots

    # CROSS --------------------------------------------------------------------
    if 'cross' in kwargs:
        if not isinstance(kwargs['cross'], tuple):
            print("Crosshair must be a tuple (X,Y)")
            show_cross = False
        else:
            coor_cross = kwargs['cross']
            show_cross = True
    else:
        show_cross = False

    cross_alpha = kwargs.get('crossalpha', 0.4)

    # AXIS - ANGSCALE ----------------------------------------------------------
    angticksep = kwargs.get('angticksep', 50)
    pxscale = kwargs.get('pxscale', 0.01)  # default for Keck/NIRC2
    angscale = kwargs.get('angscale', False)

    if angscale:
        print("`Pixel scale set to {}`".format(pxscale))

    show_axis = kwargs.get('axis', True)

    # --------------------------------------------------------------------------

    show_center = kwargs.get("showcent", False)
    getfig = kwargs.get('getfig', False)

    save = kwargs.get("save", False)

    # Defaults previously used: 'magma','CMRmap','RdBu_r'
    if 'cmap' in kwargs:
        custom_cmap = kwargs['cmap']
        if not isinstance(custom_cmap, list):
            custom_cmap = [kwargs['cmap']] * num_plots
        else:
            if not len(custom_cmap) == num_plots:
                raise RuntimeError('Cmap list does not have enough items')
    else:
        custom_cmap = ['viridis'] * num_plots

    logscale = kwargs.get('log', False)
    colorb = kwargs.get('colorb', True)
    dpi = kwargs.get('dpi', 90)
    title = kwargs.get('title', None)
    hor_spacing = kwargs.get('horsp', 0.4)
    ver_spacing = kwargs.get('versp', 0.2)

    # --------------------------------------------------------------------------

    subplot_size = 4
    if rows == 0:
        raise ValueError('Rows must be a positive integer')
    fig = figure(figsize=(cols * subplot_size, rows * subplot_size), dpi=dpi)

    if title is not None:
        fig.suptitle(title, fontsize=14)

    for i, v in enumerate(range(num_plots)):
        image = data[i].copy()
        frame_size = image.shape[0]  # assuming square frames
        cy, cx = frame_center(image)
        if grid_spacing is None:
            if cy < 10:
                grid_spacing = 1
            elif cy >= 10:
                if cy % 2 == 0:
                    grid_spacing = 4
                else:
                    grid_spacing = 5

        v += 1
        ax = subplot(rows, cols, v)
        ax.set_aspect('equal')
        if logscale:
            image += np.abs(image.min())
            norm = colors.LogNorm(vmin=max(image.min(), 0.1), vmax=image.max())
        else:
            norm = None

        if image.dtype == bool:
            image = image.astype(int)

        im = ax.imshow(image, cmap=custom_cmap[i], interpolation='nearest',
                       origin='lower', vmin=vmin[i], vmax=vmax[i],
                       norm=norm)

        if show_circle:
            for j in range(n_circ):
                circle = Circle(coor_circle[j], radius=circle_rad[j],
                                color='white', fill=False,
                                alpha=circle_alpha[j])
                ax.add_artist(circle)
                if circle_label:
                    x = coor_circle[j][0]
                    y = coor_circle[j][1]
                    cirlabel = str(int(x))+','+str(int(y))
                    ax.text(x, y+1.8*circle_rad[j], cirlabel, fontsize=8,
                            color='white', family='monospace', ha='center',
                            va='top', weight='bold', alpha=circle_alpha[j])

        if show_cross:
            ax.scatter([coor_cross[0]], [coor_cross[1]], marker='+',
                       color='white', alpha=cross_alpha)

        if show_center:
            ax.axhline(cx, xmin=0, xmax=frame_size, alpha=0.5,
                       linestyle='dashed', color='white', lw=0.8)
            ax.axvline(cy, ymin=0, ymax=frame_size, alpha=0.5,
                       linestyle='dashed', color='white', lw=0.8)

        if show_arrow:
            ax.arrow(coor_arrow[0]+arrow_length+arrow_shiftx, coor_arrow[1],
                     -arrow_length, 0, color='white', head_width=10,
                     head_length=8, width=3, length_includes_head=True,
                     alpha=arrow_alpha)

        if label is not None:
            ax.annotate(label[i], xy=(labelpad, labelpad), color='white',
                        xycoords='axes pixels', weight='bold', size=labelsize)

        if colorb:
            # create an axes on the right side of ax. The width of cax is 5%
            # of ax and the padding between cax and ax wis fixed at 0.05 inch
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cb = colorbar(im, ax=ax, cax=cax, drawedges=False)
            cb.outline.set_linewidth(0.1)
            cb.ax.tick_params(labelsize=8)

        if grid:
            ax.tick_params(axis='both', which='minor')
            minor_ticks = np.arange(0, data[i].shape[0], grid_spacing)
            ax.set_xticks(minor_ticks, minor=True)
            ax.set_yticks(minor_ticks, minor=True)
            ax.grid(True, which='minor', color=grid_color, linewidth=0.5,
                    alpha=grid_alpha, linestyle='dashed')
        else:
            ax.grid(False)

        if angscale:
            # Converting axes from pixels to arcseconds
            half_num_ticks = int(np.round(cy // angticksep))

            # Calculate the pixel locations at which to put ticks
            ticks = []
            for t in range(half_num_ticks, -half_num_ticks-1, -1):
                # Avoid ticks not showing on the last pixel
                if not cy - t * angticksep == frame_size:
                    ticks.append(cy - t * angticksep)
                else:
                    ticks.append((cy - t * angticksep) - 1)
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)

            # Corresponding distance in arcseconds, measured from the center
            labels = []
            for t in range(half_num_ticks, -half_num_ticks-1, -1):
                labels.append(-t * (angticksep * pxscale))
            ax.set_xticklabels(labels)
            ax.set_yticklabels(labels)
            ax.set_xlabel("arcseconds", fontsize=12)
            ax.set_ylabel("arcseconds", fontsize=12)
            ax.tick_params(axis='both', which='major', labelsize=10)

        if not show_axis:
            ax.set_axis_off()

    fig.subplots_adjust(wspace=hor_spacing, hspace=ver_spacing)
    if save:
        savefig(save, dpi=dpi, bbox_inches='tight', pad_inches=0,
                transparent=True)
        close()
        if getfig:
            return fig
    else:
        show()
        if getfig:
            return fig


def plot_surface(image, center_xy=None, size=15, output=False, title=None,
                 zlim=None, **kwargs):
    """
    Create a surface plot from image.

    By default, the whole image is plotted. The 'center' and 'size' attributes
    allow to crop the image.

    Parameters
    ----------
    image : numpy.array
        The image as a numpy.array.
    center_xy : tuple of 2 int or None, optional
        If None, the whole image will be plotted. Otherwise, it grabs a square
        subimage at the position X,Y from the image.
    size : int (optional, default=15)
        It corresponds to the size of a square in the image.
    output : {False, True}, bool optional
        Whether to output the grids and intensities or not.
    title : str, optional
        Sets the title for the surface plot.
    zlim : tuple, optional
        Tuple with the range of values (min,max) for the Z axis.
    kwargs:
        Additional attributes are passed to the matplotlib figure() and
        plot_surface() method.

    Returns
    -------
    out : tuple of 3 numpy.array
        x and y for the grid, and the intensity

    """
    if not isinstance(center_xy, tuple):
        raise ValueError('`center_xy` must be a tuple')

    if center_xy is not None:
        cx = int(center_xy[0])
        cy = int(center_xy[1])
        if size % 2:            # if size is odd
            x = np.outer(np.arange(0, size, 1), np.ones(size))
        else:                   # otherwise, size is even
            x = np.outer(np.arange(0, size+1, 1), np.ones(size+1))
        y = x.copy().T
        z = image[cy - size//2: cy + size//2 + 1,
                  cx - size//2: cx + size//2 + 1]
    else:
        cy, cx = frame_center(image)
        if size is not None:
            if size % 2:
                x = np.outer(np.arange(0, size, 1), np.ones(size))
            else:
                x = np.outer(np.arange(0, size+1, 1), np.ones(size+1))
            y = x.copy().T
            z = image[cy-size//2:cy+size//2+1, cx-size//2:cx+size//2+1]
        else:
            size = image.shape[0]
            x = np.outer(np.arange(0, size, 1), np.ones(size))
            y = x.copy().T
            z = image

    figure(figsize=kwargs.pop('figsize', (6, 6)))
    ax = axes(projection='3d')
    ax.dist = 10
    ax.plot_surface(x, y, z, rstride=1, cstride=1, linewidth=0, **kwargs)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('I(x,y)')
    if zlim is not None:
        if isinstance(zlim, tuple):
            ax.set_zlim(zlim[0], zlim[1])
    if title is not None:
        ax.set_title(title)
    show()

    if output:
        return x, y, z
