#! /usr/bin/env python

"""
Module with HCICube class.
"""

from __future__ import division, print_function

__author__ = 'Carlos Alberto Gomez Gonzalez'
__all__ = ['HCICube']

from .fits import open_fits, write_fits, append_extension
from .preproc import (cube_collapse, cube_crop_frames, cube_derotate,
                      cube_drop_frames, cube_detect_badfr_correlation,
                      cube_detect_badfr_pxstats, cube_px_resampling,
                      cube_subsample, cube_recenter_2dfit,
                      cube_recenter_dft_upsampling)
from .var import cube_filter_highpass, cube_filter_lowpass, mask_circle


class HCICube:
    """ High-contrast imaging sequence/cube class.
    """
    def __init__(self, cube, hdu=0, angles=None, scaling=None, fwhm=None,
                 psf=None):
        """ Initialization of the HCICube object.
        """
        if isinstance(cube, str):
            self.array = open_fits(cube, hdu, verbose=False)
        else:
            self.array = cube
        print('Cube array shape: {}'.format(self.array.shape))

        # Loading the angles (ADI)
        if isinstance(angles, str):
            self.angles = open_fits(angles, verbose=False)
        else:
            self.angles = angles
        if self.angles is not None:
            print('Angles array shape: {}'.format(self.angles.shape))
            # Checking the shape of the angles vector
            if not self.angles.shape[0] == self.array.shape[0] or \
               not self.angles.ndim == 1:
                raise ValueError('Parallactic angles vector has a wrong shape')

        # Loading the scaling factors (mSDI)
        if isinstance(scaling, str):
            self.scaling = open_fits(scaling, verbose=False)
        else:
            self.scaling = scaling
        if self.scaling is not None:
            print('Scaling array shape: {}'.format(self.scaling.shape))
            # Checking the shape of the scaling vector
            if not self.scaling.shape[0] == self.array.shape[0] or \
               not self.scaling.shape == 1:
                raise ValueError('Scaling factors vector has a wrong shape')

        # Loading the PSF
        if isinstance(psf, str):
            self.psf = open_fits(psf, verbose=False)
        else:
            self.psf = psf
        if self.psf is not None:
            print('PSF array shape: {}'.format(self.psf.shape))
            # Checking the shape of the PSF array
            if not self.psf.ndim == self.array.ndim - 1:
                msg = 'PSF array has a wrong shape. Must have {} dimensions, '
                msg += 'got {} instead'
                raise ValueError(msg.format(self.array.ndim - 1, self.psf.ndim))

        self.fwhm = fwhm

    def collapse(self, mode='median', n=50):
        """ Collapsing the sequence into a 2d array.
        """
        frame = cube_collapse(self.array, mode, n)
        print('Cube successfully collapsed')
        return frame

    def crop_frames(self, size, xy=None, force=False):
        """ Cropping the frames of the sequence.
        """
        self.array = cube_crop_frames(self.array, size, xy, force)

    def derotate(self, imlib='opencv', interpolation='lanczos4', cxy=None,
                 nproc=1):
        """ Derotating the frames of the sequence according to the parallactic
        angles.
        """
        if self.angles is None:
            raise ValueError('Parallactic angles vector has not been set')

        self.array = cube_derotate(self.array, self.angles, imlib,
                                   interpolation, cxy, nproc)
        print('Cube successfully derotated')

    def drop_frames(self, n, m):
        """ Slicing the cube using the `n` (initial) and `m` (final) indices in
        a 1-indexed fashion.
        """
        self.array = cube_drop_frames(self.array, n, m, self.angles)

    def filter(self, method, mode, median_size=5, kernel_size=5, fwhm_size=5,
               btw_cutoff=0.2, btw_order=2, gauss_mode='conv', verbose=True):
        """
        Parameters
        ----------
        method : {'lp', 'hp'}

        mode : {'median', 'gauss'}
        {'laplacian', 'laplacian-conv', 'median-subt', 'gauss-subt', 'fourier-butter'}
        """
        if method == 'hp':
            self.array = cube_filter_highpass(self.array, mode, median_size,
                                              kernel_size, fwhm_size,
                                              btw_cutoff, btw_order, verbose)
        elif method == 'lp':
            self.array = cube_filter_lowpass(self.array, mode, median_size,
                                             fwhm_size, gauss_mode, verbose)
        else:
            raise ValueError('Filtering mode not recognized')

    def inject_companions(self):
        """
        # TODO: inject companions
        """
        pass

    def load_angles(self, angles, hdu):
        """ Loads the PA vector from a FITS file. It is possible to specify the
        HDU.
        """
        if isinstance(angles, str):
            self.angles = open_fits(angles, hdu)
        else:
            self.angles = angles

    def mask_center(self, radius, fillwith=0, mode='in'):
        """ Masking the values inside/outside a centered circular aperture.
        """
        self.array = mask_circle(self.array, radius, fillwith, mode)

    def recenter(self, method='2dfit', xy=None, subi_size=5, model='gauss',
                 nproc=1, imlib='opencv', interpolation='lanczos4',
                 offset=None, negative=False, threshold=False,
                 save_shifts=False, cy_1=None, cx_1=None, upsample_factor=100,
                 verbose=True, debug=False, plot=True):
        """ Frame to frame recentering.

        # TODO: cover the full_output=False case
        """
        if self.fwhm is None:
            raise ValueError('FWHM has not been set')

        if method == '2d_fitting':
            self.array = cube_recenter_2dfit(self.array, xy, self.fwhm,
                                subi_size, model, nproc, imlib, interpolation,
                                offset, negative, threshold, save_shifts, False,
                                verbose, debug, plot)
        elif method == 'dft_upsampling':
            self.array = cube_recenter_dft_upsampling(self.array, cy_1, cx_1,
                                negative, self.fwhm, subi_size, upsample_factor,
                                imlib, interpolation, False, verbose,
                                save_shifts, debug)
        else:
            # TODO support other recentering methods from vip_hci.preproc
            raise ValueError('Method not recognized')

    def remove_badframes(self, method='corr', frame_ref=None, crop_size=30,
                       dist='pearson', percentile=20, stat_region='annulus',
                       in_radius=10, width=10, top_sigma=1.0, low_sigma=1.0,
                       window=None, plot=True, verbose=True):
        """ Finding outlying/bad frames and slicing the cube accordingly.

        Parameters
        ----------
        method : str, optional
            corr, pxstats
        """
        if method == 'corr':
            if frame_ref is None:
                print("Correlation method selected but `frame_ref` is missing")
                print("Setting the 1st frame as the reference")
                frame_ref = 0

            self.good_indices, _ = cube_detect_badfr_correlation(self.array,
                                            frame_ref, crop_size, dist,
                                            percentile, plot, verbose)
        elif method == 'pxstats':
            self.good_indices, _ = cube_detect_badfr_pxstats(self.array,
                                            stat_region, in_radius, width,
                                            top_sigma, low_sigma, window, plot,
                                            verbose)
        else:
            raise ValueError('Bad frames detection method not recognized')

        self.array = self.array[self.good_indices]
        print("New cube shape: {}".format(self.array.shape))
        if self.angles is not None:
            self.angles = self.angles[self.good_indices]
            msg = "New parallactic angles vector shape: {}"
            print(msg.format(self.angles.shape))

    def rescale(self, scale, imlib='ndimage', interpolation='bicubic',
                verbose=True):
        """ Resampling the pixels (upscaling or downscaling the frames).
        """
        self.array = cube_px_resampling(self.array, scale, imlib, interpolation,
                                        verbose)

    def save(self, path):
        """ Writing to FITS file. If present, the angles are appended to the
        FITS file.
        """
        write_fits(path, self.array)
        if self.angles is not None:
            append_extension(path, self.angles)

    def stats(self, region='circle'):
        """
        # TODO: stats in region
        """
        pass


    def subsample(self, window, mode='mean'):
        """ Temporally sub-sampling the sequence.
        """
        if self.array.ndim == 3:
            if self.angles is not None:
                self.array, self.angles = cube_subsample(self.array, window,
                                                         mode, self.angles)
            else:
                self.array = cube_subsample(self.array, window, mode)
        else:
            # TODO: ADI+mSDI
            pass