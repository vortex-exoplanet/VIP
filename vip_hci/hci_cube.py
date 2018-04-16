#! /usr/bin/env python

"""
Module with HCICube class.
"""

from __future__ import division, print_function

__author__ = 'Carlos Alberto Gomez Gonzalez'
__all__ = ['HCICube',
           'HCIFrame']

import numpy as np
from .fits import open_fits, write_fits, append_extension
from .preproc import frame_crop, frame_px_resampling, frame_rotate, frame_shift
from .preproc import (cube_collapse, cube_crop_frames, cube_derotate,
                      cube_drop_frames, cube_detect_badfr_correlation,
                      cube_detect_badfr_pxstats, cube_px_resampling,
                      cube_subsample, cube_recenter_2dfit,
                      cube_recenter_dft_upsampling)
from .var import frame_filter_lowpass, frame_filter_highpass, frame_center
from .var import (cube_filter_highpass, cube_filter_lowpass, mask_circle,
                  pp_subplots)
from .stats import frame_basic_stats, frame_histo_stats
from .stats import cube_basic_stats, cube_distance
from .phot import (frame_quick_report, cube_inject_companions, snr_ss,
                   snr_peakstddev, snrmap, snrmap_fast, detection,
                   normalize_psf)
from .conf.utils_conf import check_array


class HCIFrame:
    """ High-contrast imaging frame (2d array).

    Parameters
    ----------
    image : array_like
        2d array.
    hdu : int, optional
        If ``cube`` is a String, ``hdu`` indicates the HDU from the FITS file.
        By default the first HDU is used.
    fwhm : float, optional
        The FWHM associated with this dataset (instrument dependent). Required
        for several methods (operations on the cube).
    """
    def __init__(self, image, hdu=0, fwhm=None):
        """ HCIFrame object initialization. """
        if isinstance(image, str):
            self.image = open_fits(image, hdu, verbose=False)
        elif isinstance(image, np.ndarray):
            if not image.ndim == 2:
                raise ValueError('`Image` array has wrong dimensions')
            self.image = image
        else:
            raise ValueError('`Image` has a wrong type')
        print('Frame shape: {}'.format(self.image.shape))

        self.fwhm = fwhm
        if self.fwhm is not None:
            print('FWHM: {}'.format(self.fwhm))

    def crop(self, size, xy=None, force=False):
        """ Cropping the frame.
        """
        self.image = frame_crop(self.image, size, xy, force, verbose=True)

    def detect_blobs(self, psf, bkg_sigma=1, method='lpeaks',
                     matched_filter=False, mask=True, snr_thresh=5, plot=True,
                     debug=False, verbose=False, save_plot=None,
                     plot_title=None, angscale=False):
        """ Detecting blobs on the 2d array.
        """
        self.detection_results = detection(self.image, psf, bkg_sigma, method,
                                           matched_filter, mask, snr_thresh,
                                           plot, debug, True, verbose,
                                           save_plot, plot_title, angscale)

    def filter(self, method, mode, median_size=5, kernel_size=5, fwhm_size=5,
               btw_cutoff=0.2, btw_order=2, gauss_mode='conv'):
        """ High/low pass filtering the frames of the image.

        Parameters
        ----------
        method : {'lp', 'hp'}

        mode : {'median', 'gauss'}
        {'laplacian', 'laplacian-conv', 'median-subt', 'gauss-subt', 'fourier-butter'}
        """
        if method == 'hp':
            self.image = frame_filter_highpass(self.image, mode, median_size,
                                               kernel_size, fwhm_size,
                                               btw_cutoff, btw_order)
        elif method == 'lp':
            self.image = frame_filter_lowpass(self.image, mode, median_size,
                                              fwhm_size, gauss_mode)
        else:
            raise ValueError('Filtering mode not recognized')
        print('Image successfully filtered')

    def get_center(self, verbose=True):
        """ Getting the coordinates of the center of the image.
        """
        cent = frame_center(self.image, verbose)
        return cent

    def plot(self, **kwargs):
        """ Plotting the 2d array.

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
        circle : list of tuples
            To show a circle at given px coordinates, list of tuples.
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
            Padding of the label from the left bottom corner.
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
        pp_subplots(self.image, **kwargs)

    def rescale(self, scale, imlib='ndimage', interpolation='bicubic',
                verbose=True):
        """ Resampling the image (upscaling or downscaling).
        """
        self.image = frame_px_resampling(self.image, scale, imlib, interpolation,
                                         verbose)

    def rotate(self, angle, imlib='opencv', interpolation='lanczos4', cxy=None):
        """ Rotating the image.
        """
        self.image = frame_rotate(self.image, angle, imlib, interpolation, cxy)
        print('Image successfully rotated')

    def save(self, path):
        """ Writing to FITS file.
        """
        write_fits(path, self.image)

    def shift(self, shift_y, shift_x, imlib='opencv', interpolation='lanczos4'):
        """ Shifting the image.
        """
        self.image = frame_shift(self.image, shift_y, shift_x, imlib,
                                 interpolation)
        print('Image successfully shifted')

    def snr(self, source_xy, method='student', plot=False, verbose=True):
        """ Calculating the S/N for a test resolution element ``source_xy``.

        Parameters
        ----------
        source_xy : tuple of floats
            X and Y coordinates of the planet or test speckle.
        method : {'student', 'classic'}, str optional
            With 'student' the small sample statistics (Mawet et al. 2014) is
            used. With 'classic', the S/N is estimated with the old approach
            using the standard deviation of independent pixels.
        plot : bool, optional
            Plots the frame and the apertures considered for clarity.
        verbose : bool, optional
            Chooses whether to print some output or not.

        Returns
        -------
        snr_val : float
            Value of the S/N for ``source_xy``.
        """
        if self.fwhm is None:
            raise ValueError('FWHM has not been set')

        if method == 'student':
            snr_val = snr_ss(self.image, source_xy, self.fwhm, False, plot,
                             verbose)
        elif method == 'classic':
            snr_val = snr_peakstddev(self.image, source_xy, self.fwhm, False,
                                     plot, verbose)
        else:
            raise ValueError('S/N estimation method not recognized')
        return snr_val

    def snrmap(self, method='student', approx=False, plot=True,
               source_mask=None, nproc=None, verbose=True):
        """ Generating the S/N map for the image.

        Parameters
        ----------
        method : {'student', 'classic'}, str optional
            With 'student' the small sample statistics (Mawet et al. 2014) is
            used. With 'classic', the S/N is estimated with the old approach
            using the standard deviation of independent pixels.
        approx : bool, optional
            If True, the function ``vip_hci.phot.snrmap_fast`` is used instead
            of ``vip_hci.phot.snrmap``.
        plot : bool, optional
            If True plots the S/N map. True by default.
        source_mask : array_like, optional
            If exists, it takes into account existing sources. The mask is a
            ones 2d array, with the same size as the input frame. The centers
            of the known sources have a zero value.
        nproc : int or None
            Number of processes for parallel computing.

        Returns
        -------
        map : HCIFrame object
            S/N map.
        """
        if self.fwhm is None:
            raise ValueError('FWHM has not been set')

        if approx:
            map = snrmap_fast(self.image, self.fwhm, nproc, plot, verbose)
        else:
            if method == 'student':
                mode = 'sss'
            elif method == 'classic':
                mode = 'peakstddev'
            map = snrmap(self.image, self.fwhm, plot, mode, source_mask, nproc,
                         verbose=verbose)
        return HCIFrame(map)

    def stats(self, region='circle', radius=5, xy=None, annulus_inner_radius=0,
              annulus_width=5, source_xy=None, verbose=True, plot=True):
        """ Calculating statistics on the image, both in the full-frame and in
        a region (circular aperture or annulus). Also, the S/N of the either
        ``source_xy`` or the max pixel is calculated.

        Parameters
        ----------
        region : {'circle', 'annulus'}, str optional
            Region in which basic statistics (mean, stddev, median and max) are
            calculated.
        radius : int, optional
            Radius of the circular aperture.
        xy : tuple of floats, optional
            Center of the circular aperture.
        annulus_inner_radius : int, optional
            Inner radius of the annular region.
        annulus_width : int, optional
            Width of the annular region.
        source_xy : tuple of floats, optional
            Coordinates for which the S/N information will be obtained. If None,
            the S/N is estimated for the pixel with the maximum value.
        verbose : bool, optional
            Whether to print out the values of the calculated statistics.
        plot : bool, optional
            Whether to plot the frame, histograms and region.
        """
        res_region = frame_basic_stats(self.image, region, radius, xy,
                                       annulus_inner_radius, annulus_width,
                                       plot, True)
        if verbose:
            if region == 'circle':
                msg = 'Stats in circular aperture of radius: {}pxs'
                print(msg.format(radius))
            elif region == 'annulus':
                msg = 'Stats in annulus. Inner_rad: {}pxs, width: {}pxs'
                print(msg.format(annulus_inner_radius, annulus_width))
            mean, std_dev, median, maxi = res_region
            msg = 'Mean: {:.3f}, Stddev: {:.3f}, Median: {:.3f}, Max: {:.3f}'
            print(msg.format(mean, std_dev, median, maxi))

        res_ff = frame_histo_stats(self.image, plot)
        if verbose:
            mean, median, std, maxim, minim = res_ff
            print('Stats in the whole frame:')
            msg = 'Mean: {:.3f}, Stddev: {:.3f}, Median: {:.3f}, Max: {:.3f}, '
            msg += 'Min: {:.3f}'
            print(msg.format(mean, std, median, maxim, minim))

        print('\nS/N info:')
        _ = frame_quick_report(self.image, self.fwhm, source_xy, verbose)


class HCICube:
    """ High-contrast imaging sequence/cube class.

    Parameters
    ----------
    cube : str or numpy array
        3d or 4d high-contrast image sequence. If a string is provided, cube is
        interpreted as the path of the FITS file containing the sequence.
    hdu : int, optional
        If ``cube`` is a String, ``hdu`` indicates the HDU from the FITS file.
        By default the first HDU is used.
    angles : list or numpy array, optional
        The vector of parallactic angles.
    wavelengths : list or numpy array, optional
        The vector of wavelengths (to be used as scaling factors).
    fwhm : float, optional
        The FWHM associated with this dataset (instrument dependent). Required
        for several methods (operations on the cube).
    px_scale : float, optional
        The pixel scale associated with this dataset (instrument dependent).
    psf : numpy array, optional
        The PSF template associated with this dataset.
    psfn : numpy array, optional
        Normalized/cropped/centered version of the PSF template associated with
        this dataset.
    cuberef : str or numpy array
        3d or 4d high-contrast image sequence. To be used as a reference cube.
    """
    def __init__(self, cube, hdu=0, angles=None, wavelengths=None, fwhm=None,
                 px_scale=None, psf=None, psfn=None, cuberef=None):
        """ Initialization of the HCICube object.
        """
        # Loading the 3d/4d cube or image sequence
        if isinstance(cube, str):
            self.cube = open_fits(cube, hdu, verbose=False)
        elif isinstance(cube, np.ndarray):
            if not (cube.ndim == 3 or cube.ndim == 4):
                raise ValueError('`Cube` array has wrong dimensions')
            self.cube = cube
        else:
            raise TypeError('`Cube` has a wrong type')
        print('Cube array shape: {}'.format(self.cube.shape))
        if self.cube.ndim == 3:
            self.n, self.y, self.x = self.cube.shape
        elif self.cube.ndim == 4:
            self.w, self.n, self.y, self.x = self.cube.shape

        # Loading the reference cube
        if cuberef is not None:
            if isinstance(cuberef, str):
                self.cuberef = open_fits(cuberef, hdu, verbose=False)
            elif isinstance(cuberef, np.ndarray):
                msg = '`Cuberef` array has wrong dimensions'
                if not cuberef.ndim == 3:
                    raise ValueError(msg)
                if not cuberef.shape[1] == self.y:
                    raise ValueError(msg)
                self.cuberef = cuberef
            elif isinstance(cuberef, HCICube):
                msg = '`Cuberef` array has wrong dimensions'
                if not cuberef.cube.ndim == 3:
                    raise ValueError(msg)
                if not cuberef.cube.shape[1] == self.y:
                    raise ValueError(msg)
                self.cuberef = cuberef.cube
            else:
                raise TypeError('`Cuberef` has a wrong type')
            print('Cuberef array shape: {}'.format(self.cuberef.shape))

        # Loading the angles (ADI)
        if isinstance(angles, str):
            self.angles = open_fits(angles, verbose=False)
        else:
            self.angles = angles
        if self.angles is not None:
            print('Angles array shape: {}'.format(self.angles.shape))
            # Checking the shape of the angles vector
            self.angles = check_array(self.angles, dim=1,
                                      name='Parallactic angles vector')
            if not self.angles.shape[0] == self.n:
                raise ValueError('Parallactic angles vector has a wrong shape')

        # Loading the scaling factors (mSDI)
        if isinstance(wavelengths, str):
            self.wavelengths = open_fits(wavelengths, verbose=False)
        else:
            self.wavelengths = wavelengths
        if self.wavelengths is not None:
            print('Wavelengths array shape: {}'.format(self.wavelengths.shape))
            # Checking the shape of the scaling vector
            self.wavelengths = check_array(self.wavelengths, dim=1,
                                           name='Wavelengths vector')
            if not self.wavelengths.shape[0] == self.w:
                raise ValueError('Wavelengths vector has a wrong shape')

        # Loading the PSF
        if isinstance(psf, str):
            self.psf = open_fits(psf, verbose=False)
        else:
            self.psf = psf
        if self.psf is not None:
            print('PSF array shape: {}'.format(self.psf.shape))
            # Checking the shape of the PSF array
            if not self.psf.ndim == self.cube.ndim - 1:
                msg = 'PSF array has a wrong shape. Must have {} dimensions, '
                msg += 'got {} instead'
                raise ValueError(msg.format(self.cube.ndim - 1, self.psf.ndim))

        # Loading the normalized PSF
        if isinstance(psfn, str):
            self.psfn = open_fits(psfn, verbose=False)
        else:
            self.psfn = psfn
        if self.psfn is not None:
            print('Normalized PSF array shape: {}'.format(self.psfn.shape))
            # Checking the shape of the PSF array
            if not self.psfn.ndim == self.cube.ndim - 1:
                msg = 'Normalized PSF array has a wrong shape. Must have {} '
                msg += 'dimensions, got {} instead'
                raise ValueError(msg.format(self.cube.ndim - 1, self.psfn.ndim))

        self.fwhm = fwhm
        if self.fwhm is not None:
            if self.cube.ndim == 4:
                self.fwhm = check_array(self.fwhm, 1, 'FHWM')
            elif self.cube.ndim == 3:
                print('FWHM: {}'.format(self.fwhm))
        self.px_scale = px_scale
        if self.px_scale is not None:
            print('Pixel/plate scale: {}'.format(self.px_scale))

    def collapse(self, mode='median', n=50):
        """ Collapsing the sequence into a 2d array.

        # TODO: support 4d case.
        """
        frame = cube_collapse(self.cube, mode, n)
        print('Cube successfully collapsed')
        return HCIFrame(frame)

    def crop_frames(self, size, xy=None, force=False):
        """ Cropping the frames of the sequence (3d or 4d cube).

        Parameters
        ----------
        size : int
            New size of the (square) frames.
        xy : tuple of ints
            X, Y coordinates of new frame center. If you are getting the
            coordinates from ds9 subtract 1, python has 0-based indexing.
        force : bool, optional
            ``Size`` and the original size of the frames must be both even or odd.
            With ``force`` set to True this condition can be avoided.
        """
        self.cube = cube_crop_frames(self.cube, size, xy, force, verbose=True)

    def derotate(self, imlib='opencv', interpolation='lanczos4', cxy=None,
                 nproc=1):
        """ Derotating the frames of the sequence according to the parallactic
        angles.
        """
        if self.angles is None:
            raise ValueError('Parallactic angles vector has not been set')

        self.cube = cube_derotate(self.cube, self.angles, imlib,
                                  interpolation, cxy, nproc)
        print('Cube successfully derotated')

    def drop_frames(self, n, m):
        """ Slicing the cube using the `n` (initial) and `m` (final) indices in
        a 1-indexed fashion.

        # TODO: support 4d case.
        """
        self.cube = cube_drop_frames(self.cube, n, m, self.angles)

    def filter(self, method, mode, median_size=5, kernel_size=5, fwhm_size=5,
               btw_cutoff=0.2, btw_order=2, gauss_mode='conv', verbose=True):
        """ High/low pass filtering the frames of the cube.

        # TODO: support 4d case.

        Parameters
        ----------
        method : {'lp', 'hp'}

        mode : {'median', 'gauss'}
        {'laplacian', 'laplacian-conv', 'median-subt', 'gauss-subt', 'fourier-butter'}
        """
        if method == 'hp':
            self.cube = cube_filter_highpass(self.cube, mode, median_size,
                                              kernel_size, fwhm_size,
                                              btw_cutoff, btw_order, verbose)
        elif method == 'lp':
            self.cube = cube_filter_lowpass(self.cube, mode, median_size,
                                             fwhm_size, gauss_mode, verbose)
        else:
            raise ValueError('Filtering mode not recognized')

    def frame_distances(self, frame, region='full', dist='sad',
                        inner_radius=None, width=None, plot=True):
        """ Calculating the frame distance/correlation with respect to a
        reference image.

        # TODO: support 4d case.
        """
        _ = cube_distance(self.cube, frame, region, dist, inner_radius, width,
                          plot)

    # TODO: support 4d case.
    def frame_stats(self, region='circle', radius=5, xy=None,
                    annulus_inner_radius=0, annulus_width=5, plot=False):
        """ Calculating statistics on a ``region`` (circular aperture or
        annulus) of each image of the sequence.

        Parameters
        ----------
        region : {'circle', 'annulus'}, str optional
            Region in which basic statistics (mean, stddev, median and max) are
            calculated.
        radius : int, optional
            Radius of the circular aperture.
        xy : tuple of floats, optional
            Center of the circular aperture.
        annulus_inner_radius : int, optional
            Inner radius of the annular region.
        annulus_width : int, optional
            Width of the annular region.
        plot : bool, optional
            Whether to plot the frame, histograms and region.
        """
        _ = cube_basic_stats(self.cube, region, radius, xy,
                             annulus_inner_radius, annulus_width, plot, False)

    def inject_companions(self, flux, rad_dists, n_branches=1,
                          theta=0, imlib='opencv', interpolation='lanczos4',
                          verbose=True):
        """ Injection of fake companions in 3d or 4d cubes.

        # TODO: support the injection of a Gaussian/Moffat kernel.
        # TODO: return array/HCICube object instead?
        """
        if self.psf is None:
            raise ValueError('PSf array has not been set')
        if self.px_scale is None:
            raise ValueError('Pixel/plate scale has not been set')

        self.cube = cube_inject_companions(self.cube, self.psf, self.angles,
                                            flux, self.px_scale, rad_dists,
                                            n_branches, theta, imlib,
                                            interpolation, verbose)

    def load_angles(self, angles, hdu):
        """ Loads the PA vector from a FITS file. It is possible to specify the
        HDU.
        """
        if isinstance(angles, str):
            self.angles = open_fits(angles, hdu)
        elif isinstance(angles, (list, np.ndarray)):
            self.angles = angles
        else:
            msg = '`Angles` has a wrong type. Must be a list or 1d np.ndarray'
            raise ValueError(msg)

    def load_wavelengths(self, wavelengths, hdu):
        """ Loads the scaling factors vector from a FITS file. It is possible to
        specify the HDU.

        Parameters
        ----------
        wavelengths :
        """
        if isinstance(wavelengths, str):
            self.wavelengths = open_fits(wavelengths, hdu)
        elif isinstance(wavelengths, (list, np.ndarray)):
            self.wavelengths = wavelengths
        else:
            msg = '`wavelengths` has a wrong type. Must be a list or np.ndarray'
            raise ValueError(msg)

    def mask_center(self, radius, fillwith=0, mode='in'):
        """ Masking the values inside/outside a centered circular aperture.

        # TODO: support 4d case.
        """
        self.cube = mask_circle(self.cube, radius, fillwith, mode)

    def normalize_psf(self, fwhm='fit', size=None, threshold=None,
                      mask_core=None, model='gauss', imlib='opencv',
                      interpolation='lanczos4', force_odd=True, verbose=True):
        """ Normalizes a PSF (2d or 3d array), to have the flux in a 1xFWHM
        aperture equal to one. It also allows to crop the array and center the
        PSF at the center of the frame(s).
        """
        if self.psf is None:
            raise ValueError('PSF array has not been loaded')
        self.psfn = normalize_psf(self.psf, fwhm, size, threshold, mask_core,
                                  model, imlib, interpolation, force_odd,
                                  False, verbose)
        print('Normalized PSF array shape: {}'.format(self.psfn.shape))

    def plot(self, **kwargs):
        """ Plotting ``maxplots`` frames of the cube (3D case).

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
        circle : list of tuples
            To show a circle at given px coordinates, list of tuples.
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
            Padding of the label from the left bottom corner.
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
        # TODO: argument to slice the array?

        if self.cube.ndim == 3:
            pp_subplots(self.cube, **kwargs)
        elif self.cube.ndim == 4:
            for i in range(0, self.cube.shape[0], 10):
                tits = 'Wavelength '+str(i + 1)+', first 5 frames'
                pp_subplots(self.cube[i], title=tits,
                            maxplots=5, **kwargs)

    def recenter(self, method='2dfit', xy=None, subi_size=5, model='gauss',
                 nproc=1, imlib='opencv', interpolation='lanczos4',
                 offset=None, negative=False, threshold=False,
                 save_shifts=False, cy_1=None, cx_1=None, upsample_factor=100,
                 verbose=True, debug=False, plot=True):
        """ Frame to frame recentering.
        """
        if self.fwhm is None:
            raise ValueError('FWHM has not been set')

        if method == '2d_fitting':
            self.cube = cube_recenter_2dfit(self.cube, xy, self.fwhm,
                                subi_size, model, nproc, imlib, interpolation,
                                offset, negative, threshold, save_shifts, False,
                                verbose, debug, plot)
        elif method == 'dft_upsampling':
            self.cube = cube_recenter_dft_upsampling(self.cube, cy_1, cx_1,
                                negative, self.fwhm, subi_size, upsample_factor,
                                imlib, interpolation, False, verbose,
                                save_shifts, debug)
        else:
            # TODO support other recentering methods from vip_hci.preproc
            raise ValueError('Method not recognized')

    def remove_badframes(self, method='corr', frame_ref=None, crop_size=30,
                         dist='pearson', percentile=20, stat_region='annulus',
                         inner_radius=10, width=10, top_sigma=1.0,
                         low_sigma=1.0, window=None, plot=True, verbose=True):
        """ Finding outlying/bad frames and slicing the cube accordingly.

        Parameters
        ----------
        method : {'corr', 'pxstats'}, str optional

        """
        if method == 'corr':
            if frame_ref is None:
                print("Correlation method selected but `frame_ref` is missing")
                print("Setting the 1st frame as the reference")
                frame_ref = 0

            self.good_indices, _ = cube_detect_badfr_correlation(self.cube,
                                            frame_ref, crop_size, dist,
                                            percentile, plot, verbose)
        elif method == 'pxstats':
            self.good_indices, _ = cube_detect_badfr_pxstats(self.cube,
                                            stat_region, inner_radius, width,
                                            top_sigma, low_sigma, window, plot,
                                            verbose)
        else:
            raise ValueError('Bad frames detection method not recognized')

        self.cube = self.cube[self.good_indices]
        print("New cube shape: {}".format(self.cube.shape))
        if self.angles is not None:
            self.angles = self.angles[self.good_indices]
            msg = "New parallactic angles vector shape: {}"
            print(msg.format(self.angles.shape))

    def rescale(self, scale, imlib='ndimage', interpolation='bicubic',
                verbose=True):
        """ Resampling the pixels (upscaling or downscaling the frames).
        """
        self.cube = cube_px_resampling(self.cube, scale, imlib, interpolation,
                                        verbose)

    def save(self, path, dtype32=True):
        """ Writing to FITS file. If self.angles is present, then the angles
        are appended to the FITS file.

        Parameters
        ----------
        filename : string
            Full path of the fits file to be written.
        dtype32 : bool, optional
            If True the array is casted as a float32. When False, the array is
            usually saved in float64 precision.
        """
        write_fits(path, self.cube, dtype32)
        if self.angles is not None:
            append_extension(path, self.angles)

    def subsample(self, window, mode='mean'):
        """ Temporally sub-sampling the sequence (3d or 4d cube).

        Parameters
        ----------
        window : int
            Window for mean/median.
        mode : {'mean','median'}, optional
            Switch for choosing mean or median.
        """
        if self.angles is not None:
            self.cube, self.angles = cube_subsample(self.cube, window,
                                                     mode, self.angles)
        else:
            self.cube = cube_subsample(self.cube, window, mode)


