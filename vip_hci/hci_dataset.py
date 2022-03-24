#! /usr/bin/env python

"""
Module with Dataset and Frame classes.
"""

__author__ = 'Carlos Alberto Gomez Gonzalez'
__all__ = ['Dataset',
           'Frame']

import numpy as np
import copy
import hciplot as hp
from .fits import open_fits
from .fm import (cube_inject_companions, generate_cube_copies_with_injections,
                 normalize_psf)
from .preproc import (frame_crop, frame_px_resampling, frame_rotate,
                      frame_shift, frame_center_satspots, frame_center_radon)
from .preproc import (cube_collapse, cube_crop_frames, cube_derotate,
                      cube_drop_frames, cube_detect_badfr_correlation,
                      cube_detect_badfr_pxstats, cube_detect_badfr_ellipticity,
                      cube_px_resampling, cube_subsample, cube_recenter_2dfit,
                      cube_recenter_satspots, cube_recenter_radon,
                      cube_recenter_dft_upsampling, cube_recenter_via_speckles)
from .var import (frame_filter_lowpass, frame_filter_highpass, frame_center,
                  cube_filter_highpass, cube_filter_lowpass, mask_circle)
from .stats import (frame_basic_stats, frame_histo_stats,
                    frame_average_radprofile, cube_basic_stats, cube_distance)
from .metrics import (frame_report, snr, snrmap, detection)

from .config.utils_conf import check_array, Saveable, print_precision
from .config.mem import check_enough_memory


class Frame(object):
    """ High-contrast imaging frame (2d array).

    Parameters
    ----------
    data : numpy ndarray
        2d array.
    hdu : int, optional
        If ``cube`` is a String, ``hdu`` indicates the HDU from the FITS file.
        By default the first HDU is used.
    fwhm : float, optional
        The FWHM associated with this dataset (instrument dependent). Required
        for several methods (operations on the cube).
    """
    def __init__(self, data, hdu=0, fwhm=None):
        """ Frame object initialization. """
        if isinstance(data, str):
            self.data = open_fits(data, hdu, verbose=False)
        else:
            self.data = data

        check_array(self.data, dim=2, msg='Image.data')
        print('Frame shape: {}'.format(self.data.shape))

        self.fwhm = fwhm
        if self.fwhm is not None:
            print('FWHM: {}'.format(self.fwhm))

    def crop(self, size, xy=None, force=False):
        """ Cropping the frame.

        Parameters
        ----------
        size : int, odd
            Size of the subframe.
        cenxy : tuple, optional
            Coordinates of the center of the subframe.
        force : bool, optional
            Size and the size of the 2d array must be both even or odd. With
            ``force`` set to True this condition can be avoided.

        """
        self.data = frame_crop(self.data, size, xy, force, verbose=True)

    def detect_blobs(self, psf, bkg_sigma=1, method='lpeaks',
                     matched_filter=False, mask=True, snr_thresh=5, plot=True,
                     debug=False, verbose=False, save_plot=None,
                     plot_title=None, angscale=False):
        """ Detecting blobs on the 2d array.
        """
        self.detection_results = detection(self.data, psf, bkg_sigma, method,
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
            self.data = frame_filter_highpass(self.data, mode, median_size,
                                              kernel_size, fwhm_size,
                                              btw_cutoff, btw_order)
        elif method == 'lp':
            self.data = frame_filter_lowpass(self.data, mode, median_size,
                                             fwhm_size, gauss_mode)
        else:
            raise ValueError('Filtering mode not recognized')
        print('Image successfully filtered')

    def get_center(self, verbose=True):
        """ Getting the coordinates of the center of the image.

        Parameters
        ----------
        verbose : bool optional
            If True the center coordinates are printed out.
        """
        return frame_center(self.data, verbose)

    def plot(self, **kwargs):
        """ Plotting the 2d array.

        Parameters
        ----------
        **kwargs : dict, optional
            Parameters passed to the function ``plot_frames`` of the package
            ``HCIplot``.
        """
        hp.plot_frames(self.data, **kwargs)

    def radial_profile(self, sep=1):
        """ Calculates the average radial profile of an image.

        Parameters
        ----------
        sep : int, optional
            The average radial profile is recorded every ``sep`` pixels.
        """
        radpro = frame_average_radprofile(self.data, sep=sep, plot=True)
        return radpro

    def recenter(self, method='satspots', xy=None, subi_size=19, sigfactor=6,
                 imlib='vip-fft', interpolation='lanczos4', debug=False,
                 verbose=True):
        """ Recentering the frame using the satellite spots or a radon
        transform.

        Parameters
        ----------
        method : {'satspots', 'radon'}, str optional
            Method for recentering the frame.
        xy : tuple, optional
            Tuple with coordinates X,Y of the satellite spots. When the spots
            are in an X configuration, the order is the following: top-left,
            top-right, bottom-left and bottom-right. When the spots are in an
            + (cross-like) configuration, the order is the following: top,
            right, left, bottom.
        """
        if method == 'satspots':
            if xy is None:
                raise ValueError('`xy` must be a tuple of 4 tuples')
            self.data, _, _ = frame_center_satspots(self.data, xy, subi_size,
                                                    sigfactor, True, imlib,
                                                    interpolation, debug,
                                                    verbose)
        elif method == 'radon':
            pass
            # self.data = frame_center_radon()
        else:
            raise ValueError('Recentering method not recognized')

    def rescale(self, scale, imlib='vip-fft', interpolation='bicubic',
                verbose=True):
        """ Resampling the image (upscaling or downscaling).

        Parameters
        ----------
        scale : int, float or tuple
            Scale factor for upsampling or downsampling the frames in the cube.
            If a tuple it corresponds to the scale along x and y.
        imlib : {'ndimage', 'opencv', 'vip-fft'}, str optional
            Library used for image transformations. ndimage is the default.
        interpolation : str, optional
            For 'ndimage' library: 'nearneig', bilinear', 'bicuadratic',
            'bicubic', 'biquartic', 'biquintic'. The 'nearneig' interpolation
            is the fastest and the 'biquintic' the slowest. The 'nearneig' is
            the worst option for interpolation of noisy astronomical images.
            For 'opencv' library: 'nearneig', 'bilinear', 'bicubic', 'lanczos4'.
            The 'nearneig' interpolation is the fastest and the 'lanczos4' the
            slowest and accurate.
        verbose : bool, optional
            Whether to print out additional info such as the new image shape.
        """
        self.data = frame_px_resampling(self.data, scale, imlib, interpolation,
                                        verbose)

    def rotate(self, angle, imlib='opencv', interpolation='lanczos4', cxy=None):
        """ Rotating the image by a given ``angle``.

        Parameters
        ----------
        imlib : {'opencv', 'skimage', 'vip-fft'}, str optional
            Library used for image transformations. Opencv is faster than
            ndimage or skimage.
        interpolation : str, optional
            For 'skimage' library: 'nearneig', bilinear', 'bicuadratic',
            'bicubic', 'biquartic', 'biquintic'. The 'nearneig' interpolation
            is the fastest and the 'biquintic' the slowest. The 'nearneig' is
            the poorer option for interpolation of noisy astronomical images.
            For 'opencv' library: 'nearneig', 'bilinear', 'bicubic', 'lanczos4'.
            The 'nearneig' interpolation is the fastest and the 'lanczos4' the
            slowest and accurate. 'lanczos4' is the default.
        cxy : tuple of int, optional
            Coordinates X,Y  of the point with respect to which the rotation
            will be performed. By default the rotation is done with respect to
            the center of the frames, as it is returned by the function
            vip_hci.var.frame_center.

        """
        self.data = frame_rotate(self.data, angle, imlib, interpolation, cxy)
        print('Image successfully rotated')

    def shift(self, shift_y, shift_x, imlib='vip-fft', interpolation='lanczos4'):
        """ Shifting the image.

        Parameters
        ----------
        shift_y, shift_x: float
            Shifts in x and y directions.
        imlib : {'opencv', 'ndimage-fourier', 'ndimage-interp', 'vip-fft'}, str opt
            Library or method used for performing the image shift.
            'ndimage-fourier', does a fourier shift operation and preserves
            better the pixel values (therefore the flux and photometry).
            Interpolation based shift ('opencv' and 'ndimage-interp') is faster
            than the fourier shift. 'opencv' is recommended when speed is
            critical.
        interpolation : {'bicubic', 'bilinear', 'nearneig'}, optional
            Only used in case of imlib is set to 'opencv' or 'ndimage-interp',
            where the images are shifted via interpolation.
            For 'ndimage-interp' library: 'nearneig', bilinear', 'bicuadratic',
            'bicubic', 'biquartic', 'biquintic'. The 'nearneig' interpolation is
            the fastest and the 'biquintic' the slowest. The 'nearneig' is the
            poorer option for interpolation of noisy astronomical images.
            For 'opencv' library: 'nearneig', 'bilinear', 'bicubic', 'lanczos4'.
            The 'nearneig' interpolation is the fastest and the 'lanczos4' the
            slowest and accurate. 'lanczos4' is the default.
        """
        self.data = frame_shift(self.data, shift_y, shift_x, imlib,
                                interpolation)
        print('Image successfully shifted')

    def snr(self, source_xy, plot=False, verbose=True):
        """ Calculating the S/N for a test resolution element ``source_xy``.

        Parameters
        ----------
        source_xy : tuple of floats
            X and Y coordinates of the planet or test speckle.
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
        return snr(self.data, source_xy, self.fwhm, False, plot, verbose)

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
        res_region = frame_basic_stats(self.data, region, radius, xy,
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

        res_ff = frame_histo_stats(self.data, plot)
        if verbose:
            mean, median, std, maxim, minim = res_ff
            print('Stats in the whole frame:')
            msg = 'Mean: {:.3f}, Stddev: {:.3f}, Median: {:.3f}, Max: {:.3f}, '
            msg += 'Min: {:.3f}'
            print(msg.format(mean, std, median, maxim, minim))

        print('\nS/N info:')
        _ = frame_report(self.data, self.fwhm, source_xy, verbose)


class Dataset(Saveable):
    """ High-contrast imaging dataset class.

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

    _saved_attributes = ["cube", "psf", "psfn", "angles", "fwhm", "wavelengths",
                         "px_scale", "cuberef", "injections_yx"]

    def __init__(self, cube, hdu=0, angles=None, wavelengths=None, fwhm=None,
                 px_scale=None, psf=None, psfn=None, cuberef=None):
        """ Initialization of the Dataset object.
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
            self.w = 1
        elif self.cube.ndim == 4:
            self.w, self.n, self.y, self.x = self.cube.shape

        # Loading the reference cube
        if isinstance(cuberef, str):
            self.cuberef = open_fits(cuberef, hdu, verbose=False)
        elif isinstance(cuberef, np.ndarray):
            msg = '`Cuberef` array has wrong dimensions'
            if not cuberef.ndim == 3:
                raise ValueError(msg)
            if not cuberef.shape[1] == self.y:
                raise ValueError(msg)
            self.cuberef = cuberef
        elif isinstance(cuberef, Dataset):
            msg = '`Cuberef` array has wrong dimensions'
            if not cuberef.cube.ndim == 3:
                raise ValueError(msg)
            if not cuberef.cube.shape[1] == self.y:
                raise ValueError(msg)
            self.cuberef = cuberef.cube
        else:
            self.cuberef = None
        if self.cuberef is not None:
            print('Cuberef array shape: {}'.format(self.cuberef.shape))

        # Loading the angles (ADI)
        if isinstance(angles, str):
            self.angles = open_fits(angles, verbose=False)
        else:
            self.angles = angles
        if self.angles is not None:
            print('Angles array shape: {}'.format(self.angles.shape))
            # Checking the shape of the angles vector
            check_array(self.angles, dim=1, msg='Parallactic angles vector')
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
            check_array(self.wavelengths, dim=1, msg='Wavelengths vector')
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
                check_array(self.fwhm, dim=1, msg='FHWM')
            elif self.cube.ndim == 3:
                print('FWHM: {}'.format(self.fwhm))
        self.px_scale = px_scale
        if self.px_scale is not None:
            print('Pixel/plate scale: {}'.format(self.px_scale))

        self.injections_yx = None

    def collapse(self, mode='median', n=50):
        """ Collapsing the sequence into a 2d array.

        """
        frame = cube_collapse(self.cube, mode, n)
        print('Cube successfully collapsed')
        return Frame(frame)

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
            ``Size`` and the original size of the frames must be both even or
            odd. With ``force`` set to True this condition can be avoided.
        """
        self.cube = cube_crop_frames(self.cube, size, xy, force, verbose=True)

    def derotate(self, imlib='vip-fft', interpolation='lanczos4', cxy=None,
                 nproc=1, border_mode='constant', mask_val=np.nan,
                 edge_blend=None, interp_zeros=False, ker=1):
        """ Derotating the frames of the sequence according to the parallactic
        angles.

        Parameters
        ----------
        imlib : {'opencv', 'skimage', 'vip-fft'}, str optional
            Library used for image transformations. Opencv is faster than
            ndimage or skimage.
        interpolation : str, optional
            For 'skimage' library: 'nearneig', bilinear', 'bicuadratic',
            'bicubic', 'biquartic', 'biquintic'. The 'nearneig' interpolation
            is the fastest and the 'biquintic' the slowest. The 'nearneig' is
            the poorer option for interpolation of noisy astronomical images.
            For 'opencv' library: 'nearneig', 'bilinear', 'bicubic', 'lanczos4'.
            The 'nearneig' interpolation is the fastest and the 'lanczos4' the
            slowest and accurate. 'lanczos4' is the default.
        cxy : tuple of int, optional
            Coordinates X,Y  of the point with respect to which the rotation
            will be performed. By default the rotation is done with respect to
            the center of the frames, as it is returned by the function
            vip_hci.var.frame_center.
        nproc : int, optional
            Whether to rotate the frames in the sequence in a multi-processing
            fashion. Only useful if the cube is significantly large (frame size
            and number of frames).
        border_mode : str, optional
            See the documentation of the ``vip_hci.preproc.frame_rotate`` 
            function.
        mask_val : float, optional
            See the documentation of the ``vip_hci.preproc.frame_rotate`` 
            function.
        edge_blend : str, optional
            See the documentation of the ``vip_hci.preproc.frame_rotate`` 
            function.
        interp_zeros : str, optional
            See the documentation of the ``vip_hci.preproc.frame_rotate`` 
            function.
        ker: int, optional
            See the documentation of the ``vip_hci.preproc.frame_rotate`` 
            function.
        """
        if self.angles is None:
            raise ValueError('Parallactic angles vector has not been set')

        self.cube = cube_derotate(self.cube, self.angles, imlib,
                                  interpolation, cxy, nproc, border_mode, 
                                  mask_val, edge_blend, interp_zeros, ker)
        print('Cube successfully derotated')

    def drop_frames(self, n, m, verbose=True):
        """
        Slice the cube so that all frames between ``n``and ``m`` are kept.

        The indices ``n`` and ``m`` are included and 1-based.

        Examples
        --------
        For a cube which has 5 frames numbered ``1, 2, 3, 4, 5``, calling
        ``ds.drop_frames(2, 4)`` would result in the frames ``2, 3, 4`` to be
        kept, so the first and the last frame would be discarded.

        """
        res = cube_drop_frames(self.cube, n, m, self.angles, verbose=verbose)
        if self.angles:
            self.cube, self.angles = res
        else:
            self.cube = res

    def filter(self, method, mode, median_size=5, kernel_size=5, fwhm_size=5,
               btw_cutoff=0.2, btw_order=2, gauss_mode='conv', verbose=True):
        """ High/low pass filtering the frames of the cube.

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

        Parameters
        ----------
        frame : int or 2d array
            Reference frame in the cube or 2d array.
        region : {'full', 'annulus'}, string optional
            Whether to use the full frames or a centered annulus.
        dist : {'sad','euclidean','mse','pearson','spearman', 'ssim'}, str optional
            Which criterion to use.
        inner_radius : None or int, optional
            The inner radius when mode is 'annulus'.
        width : None or int, optional
            The width when mode is 'annulus'.
        plot : bool, optional
            Whether to plot the distances or not.

        """
        _ = cube_distance(self.cube, frame, region, dist, inner_radius, width,
                          plot)

    def frame_stats(self, region='circle', radius=5, xy=None,
                    annulus_inner_radius=0, annulus_width=5, wavelength=0,
                    plot=True):
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
        wavelength : int, optional
            Index of the wavelength to be analyzed in the case of a 4d cube.
        plot : bool, optional
            Whether to plot the frame, histograms and region.

        """
        if self.cube.ndim == 3:
            _ = cube_basic_stats(self.cube, region, radius, xy,
                                 annulus_inner_radius, annulus_width, plot,
                                 False)
        elif self.cube.ndim == 4:
            print('Stats for wavelength {}'.format(wavelength + 1))
            _ = cube_basic_stats(self.cube[wavelength], region, radius, xy,
                                 annulus_inner_radius, annulus_width, plot,
                                 False)

    def inject_companions(self, flux, rad_dists, n_branches=1, theta=0,
                          imlib='vip-fft', interpolation='lanczos4',
                          full_output=False, verbose=True):
        """ Injection of fake companions in 3d or 4d cubes.

        Parameters
        ----------
        flux : float or list
            Factor for controlling the brightness of the fake companions.
        rad_dists : float, list or array 1d
            Vector of radial distances of fake companions in pixels.
        n_branches : int, optional
            Number of azimutal branches.
        theta : float, optional
            Angle in degrees for rotating the position of the first branch that
            by default is located at zero degrees. Theta counts
            counterclockwise from the positive x axis.
        imlib : {'opencv', 'ndimage-fourier', 'ndimage-interp', 'vip-fft'}, str opt
            Library or method used for performing the image shift.
            'ndimage-fourier', does a fourier shift operation and preserves
            better the pixel values (therefore the flux and photometry).
            Interpolation based shift ('opencv' and 'ndimage-interp') is faster
            than the fourier shift. 'opencv' is recommended when speed is
            critical.
        interpolation : {'bicubic', 'bilinear', 'nearneig'}, optional
            Only used in case of imlib is set to 'opencv' or 'ndimage-interp',
            where the images are shifted via interpolation. For 'ndimage-interp'
            library: 'nearneig', bilinear', 'bicuadratic', 'bicubic',
            'biquartic', 'biquintic'. The 'nearneig' interpolation is the
            fastest and the 'biquintic' the slowest. The 'nearneig' is the
            poorer option for interpolation of noisy astronomical images.
            For 'opencv' library: 'nearneig', 'bilinear', 'bicubic', 'lanczos4'.
            The 'nearneig' interpolation is the fastest and the 'lanczos4' the
            slowest and accurate. 'lanczos4' is the default.
        full_output : bool, optional
            Return the coordinates of the injected companions.
        verbose : bool, optional
            If True prints out additional information.

        Returns
        -------
        yx : list of tuple(y,x)
            [full_output=True] Pixel coordinates of the injections in the first
            frame (and first wavelength for 4D cubes). These are only the new
            injections - all injections (from multiple calls to this function)
            are stored in ``self.injections_yx``.

        """
        if self.angles is None:
            raise ValueError('The PA angles have not been set')
        if self.psfn is None:
            raise ValueError('The normalized PSF array cannot be found')
        if self.px_scale is None:
            raise ValueError('Pixel/plate scale has not been set')
        if self.cube.ndim == 4:
            if self.wavelengths is None:
                raise ValueError('The wavelengths vector has not been set')

        self.cube, yx = cube_inject_companions(
            self.cube, self.psfn, self.angles, flux, self.px_scale,
            rad_dists, n_branches, theta, imlib, interpolation,
            full_output=True, verbose=verbose
        )

        if self.injections_yx is None:
            self.injections_yx = []

        self.injections_yx += yx

        if verbose:
            print("Coordinates of the injections stored in self.injections_yx")

        if full_output:
            return yx

    def generate_copies_with_injections(self, n_copies, inrad=8, outrad=12,
                                        dist_flux=("uniform", 2, 500)):
        """
        Create copies of this dataset, containing different random injections.

        Parameters
        ----------
        n_copies : int
            This is the number of 'cube copies' returned.
        inrad,outrad : float
            Inner and outer radius of the injections. The actual injection
            position is chosen randomly.
        dist_flux : tuple('method', *params)
            Tuple describing the flux selection. Method can be a function, the
            ``*params`` are passed to it. Method can also be a string, for a
            pre-defined random function:

                ``('skewnormal', skew, mean, var)``
                    uses scipy.stats.skewnorm.rvs
                ``('uniform', low, high)``
                    uses np.random.uniform
                ``('normal', loc, scale)``
                    uses np.random.normal

        Yields
        -------
        fake_dataset : Dataset
            Copy of the original Dataset, with injected companions.

        """

        for data in generate_cube_copies_with_injections(
            self.cube, self.psf, self.angles, self.px_scale, n_copies=n_copies,
            inrad=inrad, outrad=outrad, dist_flux=dist_flux
        ):

            dsi = self.copy()
            dsi.cube = data["cube"]
            dsi.injections_yx = data["positions"]
            # data["dist"], data["theta"], data["flux"] are not used.

            yield dsi

    def get_nbytes(self):
        """
        Return the total number of bytes the Dataset consumes.
        """
        return sum(arr.nbytes for arr in [self.cube, self.cuberef, self.angles,
                                          self.wavelengths, self.psf, self.psfn]
                   if arr is not None)

    def copy(self, deep=True, check_mem=True):
        """
        Create an in-memory copy of this Dataset.

        This is especially useful for keeping a backup copy of the original
        dataset before modifying if (e.g. with injections).

        Parameters
        ----------
        deep : bool, optional
            By default, a deep copy is created. That means every (sub)attribute
            is copied in memory. While this requires more memory, one can safely
            modify the attributes without touching the original Dataset. When
            ``deep=False``, a shallow copy of the Dataset is returned
            instead. That means all attributes (e.g. ``self.cube``) point back
            to the original object's attributes. Pay attention when modifying
            such a shallow copy!
        check_mem : bool, optional
            [deep=True] If True, verifies that the system has enough memory to
            store the result.

        Returns
        -------
        new_dataset : Dataset
            (deep) copy of this Dataset.

        """
        if deep:
            if check_mem and not check_enough_memory(self.get_nbytes(), 1.5,
                                                     verbose=False):
                raise RuntimeError("copy would require more memory than "
                                   "available.")

            return copy.deepcopy(self)
        else:
            return copy.copy(self)

    def load_angles(self, angles, hdu=0):
        """ Loads the PA vector from a FITS file. It is possible to specify the
        HDU.

        Parameters
        ----------
        angles : str or 1d numpy ndarray
            List or vector with the parallactic angles.
        hdu : int, optional
            If ``angles`` is a String, ``hdu`` indicates the HDU from the FITS
            file. By default the first HDU is used.
        """
        if isinstance(angles, str):
            self.angles = open_fits(angles, hdu)
        elif isinstance(angles, (list, np.ndarray)):
            self.angles = angles
        else:
            msg = '`Angles` has a wrong type. Must be a list or 1d np.ndarray'
            raise ValueError(msg)

    def load_wavelengths(self, wavelengths, hdu=0):
        """ Loads the scaling factors vector from a FITS file. It is possible to
        specify the HDU.

        Parameters
        ----------
        wavelengths : str or 1d numpy ndarray
            List or vector with the wavelengths.
        hdu : int, optional
            If ``wavelengths`` is a String, ``hdu`` indicates the HDU from the
            FITS file. By default the first HDU is used.

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

        Parameters
        ----------
        radius : int
            Radius of the circular aperture.
        fillwith : int, float or np.nan, optional
            Value to put instead of the masked out pixels.
        mode : {'in', 'out'}, optional
            When set to 'in' then the pixels inside the radius are set to
            ``fillwith``. When set to 'out' the pixels outside the circular
            mask are set to ``fillwith``.

        """
        self.cube = mask_circle(self.cube, radius, fillwith, mode)

    def normalize_psf(self, fit_fwhm=True, size=None, threshold=None,
                      mask_core=None, model='gauss', imlib='vip-fft',
                      interpolation='lanczos4', force_odd=True, verbose=True):
        """ Normalizes a PSF (2d or 3d array), to have the flux in a 1xFWHM
        aperture equal to one. It also allows to crop the array and center the
        PSF at the center of the frame(s).

        Parameters
        ----------
        fit_fwhm: bool, optional
            Whether to fit a ``model`` to estimate the FWHM instead of using the
            self.fwhm attribute.
        size : int or None, optional
            If int it will correspond to the size of the squared subimage to be
            cropped form the psf array.
        threshold : None of float, optional
            Sets to zero small values, trying to leave only the core of the PSF.
        mask_core : None of float, optional
            Sets the radius of a circular aperture for the core of the PSF,
            everything else will be set to zero.
        imlib : {'opencv', 'ndimage-fourier', 'ndimage-interp', 'vip-fft'}, str opt
            Library or method used for performing the image shift.
            'ndimage-fourier', does a fourier shift operation and preserves
            better the pixel values (therefore the flux and photometry).
            Interpolation based shift ('opencv' and 'ndimage-interp') is faster
            than the fourier shift. 'opencv' is recommended when speed is
            critical.
        interpolation : {'bicubic', 'bilinear', 'nearneig'}, optional
            Only used in case of imlib is set to 'opencv' or 'ndimage-interp',
            where the images are shifted via interpolation. For 'ndimage-interp'
            library: 'nearneig', bilinear', 'bicuadratic', 'bicubic',
            'biquartic', 'biquintic'. The 'nearneig' interpolation is the
            fastest and the 'biquintic' the slowest. The 'nearneig' is the
            poorer option for interpolation of noisy astronomical images.
            For 'opencv' library: 'nearneig', 'bilinear', 'bicubic', 'lanczos4'.
            The 'nearneig' interpolation is the fastest and the 'lanczos4' the
            slowest and accurate. 'lanczos4' is the default.
        force_odd : str, optional
            If True the resulting array will have odd size (and the PSF will be
            placed at its center). If False, and the frame size is even, then
            the PSF will be put at the center of an even-sized frame.
        verbose : bool, optional
            If True intermediate results are printed out.
        """
        if not fit_fwhm and self.fwhm is None:
            raise ValueError('FWHM has not been set')
        if self.psf is None:
            raise ValueError('PSF array has not been loaded')

        if not fit_fwhm:
            fwhm = self.fwhm
        else:
            fwhm = 'fit'
        res = normalize_psf(self.psf, fwhm, size, threshold, mask_core, model,
                            imlib, interpolation, force_odd, True, verbose)
        self.psfn, self.aperture_flux, self.fwhm = res
        print('Normalized PSF array shape: {}'.format(self.psfn.shape))
        print('The attribute `psfn` contains the normalized PSF')
        print("`fwhm` attribute set to")
        print_precision(self.fwhm)

    def plot(self, **kwargs):
        """ Plotting the frames of a 3D or 4d cube.

        Parameters
        ----------
        **kwargs : dict, optional
            Parameters passed to the function ``plot_cubes`` of the package
            ``HCIplot``.
        """
        hp.plot_cubes(self.cube, **kwargs)

    def recenter(self, method='2dfit', xy=None, subi_size=5, model='gauss',
                 nproc=1, imlib='vip-fft', interpolation='lanczos4',
                 offset=None, negative=False, threshold=False,
                 save_shifts=False, cy_1=None, cx_1=None, upsample_factor=100,
                 alignment_iter=5, gamma=1, min_spat_freq=0.5, max_spat_freq=3,
                 recenter_median=False, sigfactor=6, cropsize=101, hsize=0.4,
                 step=0.01, mask_center=None, verbose=True, debug=False,
                 plot=True):
        """ Frame to frame recentering.

        Parameters
        ----------
        method : {'2dfit', 'dftups', 'dftupspeckles', 'satspots', 'radon'}, optional
            Recentering method.
        xy : tuple or ints or tuple of 4 tuples of ints, optional
            For the 2dfitting, ``xy`` are the coordinates of the center of the
            subimage (wrt the original frame). For the satellite spots method,
            it is a tuple with coordinates X,Y of the 4 satellite spots. When
            the spots are in an X configuration, the order is the following:
            top-left, top-right, bottom-left and bottom-right. When the spots
            are in an + (cross-like) configuration, the order is the following:
            top, right, left, bottom.
        subi_size : int, optional
            Size of the square subimage sides in pixels.
        model : str, optional
            [method=2dfit] Sets the type of fit to be used.
            'gauss' for a 2d Gaussian fit and 'moff' for a 2d Moffat fit.
        nproc : int or None, optional
            Number of processes (>1) for parallel computing. If 1 then it runs
            in serial. If None the number of processes will be set to
            (cpu_count()/2).
        imlib : {'opencv', 'ndimage-fourier', 'ndimage-interp', 'vip-fft'}, str opt
            Library or method used for performing the image shift.
            'ndimage-fourier', does a fourier shift operation and preserves
            better the pixel values (therefore the flux and photometry).
            Interpolation based shift ('opencv' and 'ndimage-interp') is faster
            than the fourier shift. 'opencv' is recommended when speed is
            critical.
        interpolation : {'bicubic', 'bilinear', 'nearneig'}, optional
            Only used in case of imlib is set to 'opencv' or 'ndimage-interp',
            where the images are shifted via interpolation.
            For 'ndimage-interp' library: 'nearneig', bilinear', 'bicuadratic',
            'bicubic', 'biquartic', 'biquintic'. The 'nearneig' interpolation is
            the fastest and the 'biquintic' the slowest. The 'nearneig' is the
            poorer option for interpolation of noisy astronomical images.
            For 'opencv' library: 'nearneig', 'bilinear', 'bicubic', 'lanczos4'.
            The 'nearneig' interpolation is the fastest and the 'lanczos4' the
            slowest and accurate. 'lanczos4' is the default.
        offset : tuple of floats, optional
            [method=2dfit] If None the region of the frames
            used for the 2d Gaussian/Moffat fit is shifted to the center of the
            images (2d arrays). If a tuple is given it serves as the offset of
            the fitted area wrt the center of the 2d arrays.
        negative : bool, optional
            [method=2dfit/dftups/dftupspeckles] If True a negative 2d
            Gaussian/Moffat fit is performed.
        threshold : bool, optional
            [method=2dfit] If True the background pixels
            (estimated using sigma clipped statistics) will be replaced by
            small random Gaussian noise.
        save_shifts : bool, optional
            [method=2dfit/dftups] Whether to save the shifts to a file in disk.
        cy_1, cx_1 : int, optional
            [method=dftups] Coordinates of the center of the
            subimage for fitting a 2d Gaussian and centroiding the 1st frame.
        upsample_factor : int, optional
            [method=dftups] Upsampling factor (default 100).
            Images will be registered to within 1/upsample_factor of a pixel.
        alignment_iter : int, optional
            [method=dftupspeckles] Number of alignment
            iterations (recomputes median after each iteration).
        gamma : int, optional
            [method=dftupspeckles] Applies a gamma correction
            to emphasize speckles (useful for faint stars).
        min_spat_freq : float, optional
            [method=dftupspeckles] Spatial frequency for high
            pass filter.
        max_spat_freq : float, optional
            [method=dftupspeckles] Spatial frequency for low
            pass filter.
        recenter_median : bool, optional
            [method=dftupspeckles] Recenter the frames at each
            iteration based on the gaussian fit.
        sigfactor : int, optional
            [method=satspots] The background pixels will
            be thresholded before fitting a 2d Gaussian to the data using sigma
            clipped statistics. All values smaller than (MEDIAN +
            sigfactor*STDDEV) will be replaced by small random Gaussian noise.
        cropsize : odd int, optional
            [method=radon] Size in pixels of the cropped central area of the
            input array that will be used. It should be large enough to contain
            the satellite spots.
        hsize : float, optional
            [method=radon] Size of the box for the grid search. The frame is
            shifted to each direction from the center in a hsize length with a
            given step.
        step : float, optional
            [method=radon] The step of the coordinates change.
        mask_center : None or int, optional
            [method=radon] If None the central area of the frame is kept. If int
            a centered zero mask will be applied to the frame. By default the
            center isn't masked.
        verbose : bool, optional
            Whether to print to stdout the timing and aditional info.
        debug : bool, optional
            If True debug information is printed and plotted.
        plot : bool, optional
            Whether to plot the shifts.

        """

        if method == '2dfit':
            if self.fwhm is None:
                raise ValueError('FWHM has not been set')
            self.cube = cube_recenter_2dfit(
                self.cube, xy, self.fwhm, subi_size, model, nproc, imlib,
                interpolation, offset, negative, threshold, save_shifts, False,
                verbose, debug, plot
            )
        elif method == 'dftups':
            if self.fwhm is None:
                raise ValueError('FWHM has not been set')
            self.cube = cube_recenter_dft_upsampling(
                self.cube, cy_1, cx_1, negative, self.fwhm, subi_size,
                upsample_factor, imlib, interpolation, False, verbose,
                save_shifts, debug, plot
            )
        elif method == 'dftupspeckles':
            if self.fwhm is None:
                raise ValueError('FWHM has not been set')
            res = cube_recenter_via_speckles(
                self.cube, self.cuberef, alignment_iter, gamma, min_spat_freq,
                max_spat_freq, self.fwhm, debug, negative, recenter_median,
                subi_size, imlib, interpolation, plot
            )
            if self.cuberef is None:
                self.cube = res[0]
            else:
                self.cube = res[0]
                self.cuberef = res[1]
        elif method == 'satspots':
            self.cube, _, _ = cube_recenter_satspots(
                self.cube, xy, subi_size, sigfactor, plot, debug, verbose
            )
        elif method == 'radon':
            self.cube = cube_recenter_radon(
                self.cube, full_output=False, verbose=verbose, imlib=imlib,
                interpolation=interpolation, cropsize=cropsize, hsize=hsize,
                step=step, mask_center=mask_center, nproc=nproc, debug=debug
            )
        else:
            raise ValueError('Method not recognized')

    def remove_badframes(self, method='corr', frame_ref=None, crop_size=30,
                         dist='pearson', percentile=20, stat_region='annulus',
                         inner_radius=10, width=10, top_sigma=1.0,
                         low_sigma=1.0, window=None, roundlo=-0.2, roundhi=0.2,
                         lambda_ref=0, plot=True, verbose=True):
        """
        Find outlying/bad frames and slice the cube accordingly.

        Besides modifying ``self.cube`` and ``self.angles``, also sets a
        ``self.good_indices`` which contain the indices of the angles which were
        kept.

        Parameters
        ----------
        method : {'corr', 'pxstats', 'ellip'}, optional
            Method which is used to determine bad frames. Refer to the
            ``preproc.badframes`` submodule for explanation of the different
            methods.
        frame_ref : int, 2d array or None, optional
            [method=corr] Index of the frame that will be used as a reference or
            2d reference array.
        crop_size : int, optional
            [method=corr] Size in pixels of the square subframe to be analyzed.
        dist : {'sad','euclidean','mse','pearson','spearman'}, optional
            [method=corr] One of the similarity or dissimilarity measures from
            function vip_hci.stats.distances.cube_distance().
        percentile : float, optional
            [method=corr] The percentage of frames that will be discarded
            [0..100].
        stat_region : {'annulus', 'circle'}, optional
            [method=pxstats] Whether to take the statistics from a circle or an
            annulus.
        inner_radius : int, optional
            [method=pxstats] If stat_region is 'annulus' then 'in_radius' is the
            inner radius of the annular region. If stat_region is 'circle' then
            'in_radius' is the radius of the aperture.
        width : int, optional
            [method=pxstats] Size of the annulus. Ignored if mode is 'circle'.
        top_sigma : int, optional
            [method=pxstats] Top boundary for rejection.
        low_sigma : int, optional
            [method=pxstats] Lower boundary for rejection.
        window : int, optional
            [method=pxstats] Window for smoothing the median and getting the
            rejection statistic.
        roundlo,roundhi : float, optional
            [method=ellip] : Lower and higher bounds for the ellipticipy.
        lambda_ref : int, optional
            [4D cube] Which wavelength to consider when determining bad frames
            on a 4D cube.
        plot : bool, optional
            If true it plots the mean fluctuation as a function of the frames
            and the boundaries.
        verbose : bool, optional
            Show debug output.

        """
        if self.cube.ndim == 4:
            tcube = self.cube[lambda_ref]
        else:
            tcube = self.cube

        if method == 'corr':
            if frame_ref is None:
                print("Correlation method selected but `frame_ref` is missing")
                print("Setting the 1st frame as the reference")
                frame_ref = 0

            self.good_indices, _ = cube_detect_badfr_correlation(tcube,
                                        frame_ref, crop_size, dist, percentile,
                                        plot, verbose)
        elif method == 'pxstats':
            self.good_indices, _ = cube_detect_badfr_pxstats(tcube, stat_region,
                                        inner_radius, width, top_sigma,
                                        low_sigma, window, plot, verbose)
        elif method == 'ellip':
            if self.cube.ndim == 4:
                fwhm = self.fwhm[lambda_ref]
            else:
                fwhm = self.fwhm
            self.good_indices, _ = cube_detect_badfr_ellipticity(tcube, fwhm,
                                        crop_size, roundlo, roundhi, plot,
                                        verbose)
        else:
            raise ValueError('Bad frames detection method not recognized')

        if self.cube.ndim == 4:
            self.cube = self.cube[:, self.good_indices]
        else:
            self.cube = self.cube[self.good_indices]

        if verbose:
            print("New cube shape: {}".format(self.cube.shape))

        if self.angles is not None:
            self.angles = self.angles[self.good_indices]
            if verbose:
                msg = "New parallactic angles vector shape: {}"
                print(msg.format(self.angles.shape))

    def rescale(self, scale, imlib='ndimage', interpolation='bicubic',
                verbose=True):
        """ Resampling the pixels (upscaling or downscaling the frames).

        Parameters
        ----------
        scale : int, float or tuple
            Scale factor for upsampling or downsampling the frames in the cube.
            If a tuple it corresponds to the scale along x and y.
        imlib : {'ndimage', 'opencv', 'vip-fft'}, str optional
            Library used for image transformations. ndimage is the default.
        interpolation : str, optional
            For 'ndimage' library: 'nearneig', bilinear', 'bicuadratic',
            'bicubic', 'biquartic', 'biquintic'. The 'nearneig' interpolation
            is the fastest and the 'biquintic' the slowest. The 'nearneig' is
            the worst option for interpolation of noisy astronomical images.
            For 'opencv' library: 'nearneig', 'bilinear', 'bicubic', 'lanczos4'.
            The 'nearneig' interpolation is the fastest and the 'lanczos4' the
            slowest and accurate.
        verbose : bool, optional
            Whether to print out additional info such as the new cube shape.

        """
        self.cube = cube_px_resampling(self.cube, scale, imlib, interpolation,
                                       verbose)

    def subsample(self, window, mode='mean'):
        """ Temporally sub-sampling the sequence (3d or 4d cube).

        Parameters
        ----------
        window : int
            Window for mean/median.
        mode : {'mean', 'median'}, optional
            Switch for choosing mean or median.
        """
        if self.angles is not None:
            self.cube, self.angles = cube_subsample(self.cube, window,
                                                    mode, self.angles)
        else:
            self.cube = cube_subsample(self.cube, window, mode)
