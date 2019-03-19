#! /usr/bin/env python

"""
Module with 2d/3d plotting functions.
"""

from __future__ import division, print_function

__author__ = 'Carlos Alberto Gomez Gonzalez, O. Wertz'
__all__ = ['pp_subplots']

import numpy as np
from matplotlib.pyplot import (figure, subplot, show, colorbar, Circle, savefig,
                               close)
import matplotlib.colors as colors
import matplotlib.cm as mplcm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.cm import register_cmap
from .shapes import frame_center


# Registering heat and cool colormaps from DS9
# taken from: https://gist.github.com/adonath/c9a97d2f2d964ae7b9eb
ds9cool = {'red': lambda v: 2 * v - 1,
           'green': lambda v: 2 * v - 0.5,
           'blue': lambda v: 2 * v}
ds9heat = {'red': lambda v: np.interp(v, [0, 0.34, 1], [0, 1, 1]),
           'green': lambda v: np.interp(v, [0, 1], [0, 1]),
           'blue': lambda v: np.interp(v, [0, 0.65, 0.98, 1], [0, 0, 1, 1])}
register_cmap('ds9cool', data=ds9cool)
register_cmap('ds9heat', data=ds9heat)
vip_default_cmap = 'viridis'


def pp_subplots(*data, **kwargs):
    """ Wrapper for easy creation of pyplot subplots. It is convenient for
    displaying VIP images in jupyter notebooks.

    Parameters
    ----------
    data : list
        List of 2d arrays or a single 3d array to be plotted.
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
    circlecolor : str
        Color or circle(s). White by default.
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
        When the input (``*args``) is a 3d array, maxplots sets the number of
        cube slices to be displayed.
    pxscale : float
        Pixel scale in arcseconds/px. Default 0.01 for Keck/NIRC2.
    rows : int
        How many rows (subplots in a grid).
    save : str
        If a string is provided the plot is saved using this as the path.
    showcent : bool
        To show a big crosshair at the center of the frame.
    spsize : int
        Determines the size of the plot. Figsize=(spsize*ncols, spsize*nrows).
    title : str
        Title of the plot(s), None by default.
    vmax : int
        For stretching the displayed pixels values.
    vmin : int
        For stretching the displayed pixels values.
    versp : float
        Vertical gap between subplots.

    """
    parlist = ['angscale',
               'angticksep',
               'arrow',
               'arrowalpha',
               'arrowlength',
               'arrowshiftx',
               'axis',
               'circle',
               'circlealpha',
               'circlecolor',
               'circlerad',
               'circlelabel',
               'cmap',
               'colorb',
               'cross',
               'crossalpha',
               'dpi',
               'getfig',
               'grid',
               'gridalpha',
               'gridcolor',
               'gridspacing',
               'horsp',
               'label',
               'labelpad',
               'labelsize',
               'log',
               'maxplots',
               'pxscale',
               'rows',
               'save',
               'showcent',
               'spsize',
               'title',
               'vmax',
               'vmin',
               'versp']

    for key in kwargs.keys():
        if key not in parlist:
            print("Parameter '{}' not recognized".format(key))
            print("Available parameters are: {}".format(parlist))

    # GEOM ---------------------------------------------------------------------
    num_plots = len(data)
    if num_plots == 1:
        if data[0].ndim == 3 and data[0].shape[2] != 3:
            data = data[0]
            maxplots = kwargs.get("maxplots", 10)
            num_plots = min(data.shape[0], maxplots)
    elif num_plots > 1:
        for i in range(num_plots):
            if data[i].ndim != 2 and data[i].shape[2] != 3:
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

    if 'circlerad' in kwargs and show_circle:
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

    if 'circlecolor' in kwargs:
        circle_col = kwargs['circlecolor']
    else:
        circle_col = 'white'

    if 'circlealpha' in kwargs:
        circle_alpha = kwargs['circlealpha']
        # single value is provided, used for all the circles
        if isinstance(circle_alpha, (float, int)) and show_circle:
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

    if 'cmap' in kwargs:
        custom_cmap = kwargs['cmap']
        if not isinstance(custom_cmap, (list, tuple)):
            custom_cmap = [kwargs['cmap']] * num_plots
        else:
            if not len(custom_cmap) == num_plots:
                raise RuntimeError('Cmap does not contain enough items')
    else:
        custom_cmap = [vip_default_cmap] * num_plots

    if 'log' in kwargs:
        # Showing bad/nan pixels with the darkest color in current colormap
        current_cmap = mplcm.get_cmap()
        current_cmap.set_bad(current_cmap.colors[0])
        logscale = kwargs['log']
        if not isinstance(logscale, (list, tuple)):
            logscale = [kwargs['log']] * num_plots
        else:
            if not len(logscale) == num_plots:
                raise RuntimeError('Logscale does not contain enough items')
    else:
        logscale = [False] * num_plots

    colorb = kwargs.get('colorb', True)
    dpi = kwargs.get('dpi', 90)
    title = kwargs.get('title', None)
    hor_spacing = kwargs.get('horsp', 0.4)
    ver_spacing = kwargs.get('versp', 0.2)

    # --------------------------------------------------------------------------

    if 'spsize' in kwargs:
        spsize = kwargs['spsize']
    else:
        spsize = 4

    if rows == 0:
        raise ValueError('Rows must be a positive integer')
    fig = figure(figsize=(cols * spsize, rows * spsize), dpi=dpi)

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

        if logscale[i]:
            image += np.abs(image.min())
            if vmin[i] is None:
                linthresh = 1e-2
            else:
                linthresh = vmin[i]
            norm = colors.SymLogNorm(linthresh)
        else:
            norm = None

        if image.dtype == bool:
            image = image.astype(int)

        im = ax.imshow(image, cmap=custom_cmap[i], interpolation='nearest',
                       origin='lower', vmin=vmin[i], vmax=vmax[i], norm=norm)

        if show_circle:
            for j in range(n_circ):
                circle = Circle(coor_circle[j], radius=circle_rad[j],
                                color=circle_col, fill=False, lw=2,
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
            ax.axhline(cx, xmin=0, xmax=frame_size, alpha=0.3,
                       linestyle='dashed', color='white', lw=0.6)
            ax.axvline(cy, ymin=0, ymax=frame_size, alpha=0.3,
                       linestyle='dashed', color='white', lw=0.6)

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

