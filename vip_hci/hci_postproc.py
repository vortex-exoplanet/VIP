#! /usr/bin/env python

"""
Module with the HCI<post-processing algorithms> classes.
"""

from __future__ import division, print_function

__author__ = 'Carlos Alberto Gomez Gonzalez'
__all__ = ['HCIMedianSub',
           'HCIPca']

from sklearn.base import BaseEstimator
from vip_hci import HCIDataset
from vip_hci.madi import adi
from vip_hci.pca import pca


class HCIPostProcAlgo(BaseEstimator):
    """ Base HCI post-processing algorithm class.
    """
    def print_parameters(self):
        """ Printing out the parameters of the algorithm.
        """
        dicpar = self.get_params()
        for key in dicpar.keys():
            print("{}: {}".format(key, dicpar[key]))


class HCIMedianSub(HCIPostProcAlgo):
    """
    """
    def __init__(self, mode='fullfr', radius_int=0, asize=1, delta_rot=1,
                 nframes=4, imlib='opencv', interpolation='lanczos4',
                 collapse='median', nproc=1):
        """
        """
        self.mode = mode
        self.radius_int = radius_int
        self.asize = asize
        self.delta_rot = delta_rot
        self.nframes = nframes
        self.imlib = imlib
        self.interpolation = interpolation
        self.collapse = collapse
        self.nproc = nproc
        self.print_parameters()

    def run(self, dataset, full_output=False, verbose=True):
        """
        """
        if not isinstance(dataset, HCIDataset):
            raise ValueError('`Dataset` must be a HCIDataset object')

        # TODO: 4d support ADI medsub function

        res = adi(dataset.cube, dataset.angles, dataset.fwhm, self.radius_int,
                  self.asize, self.delta_rot, self.mode, self.nframes,
                  self.imlib, self.interpolation, self.collapse, self.nproc,
                  full_output, verbose)

        if full_output:
            cube_residuals, cube_residuals_der, frame_final = res
            self.cube_residuals = cube_residuals
            self.cube_residuals_der = cube_residuals_der
            self.frame_final = frame_final
            return cube_residuals, cube_residuals_der, frame_final
        else:
            frame_final = res
            self.frame_final = frame_final
            return frame_final


class HCIPca(HCIPostProcAlgo):
    """
    """
    def __init__(self, ncomp=1, ncomp2=1, svd_mode='lapack', scaling=None,
                 adimsdi='double', mask_central_px=None, source_xy=None,
                 delta_rot=1, imlib='opencv', interpolation='lanczos4',
                 collapse='median', check_mem=True):
        """
        """
        self.svd_mode = svd_mode
        self.ncomp = ncomp
        self.ncomp2 = ncomp2
        self.adimsdi = adimsdi
        self.scaling = scaling
        self.mask_central_px = mask_central_px
        self.source_xy = source_xy
        self.delta_rot = delta_rot
        self.imlib = imlib
        self.interpolation = interpolation
        self.collapse = collapse
        self.check_mem = check_mem
        self.print_parameters()

    def run(self, dataset, full_output=False, verbose=True, debug=False):
        """
        """
        if not isinstance(dataset, HCIDataset):
            raise ValueError('`Dataset` must be a HCIDataset object')

        res = pca(dataset.cube, dataset.angles, dataset.cuberef,
                  dataset.wavelengths, self.ncomp, self.ncomp2, self.svd_mode,
                  self.scaling, self.adimsdi, self.mask_central_px,
                  self.source_xy, self.delta_rot, dataset.fwhm, self.imlib,
                  self.interpolation, self.collapse, self.check_mem,
                  full_output, verbose, debug)

        if dataset.cube.ndim == 3:
            if full_output:
                if self.source_xy is not None:
                    cuberecon, cuberes, cuberesder, frame = res
                    self.cube_reconstructed = cuberecon
                    self.cube_residuals = cuberes
                    self.cube_residuals_der = cuberesder
                    self.frame_final = frame
                    return cuberecon, cuberes, cuberesder, frame
                else:
                    pcs, cuberecon, cuberes, cuberesder, frame = res
                    self.pcs = pcs
                    self.cube_reconstructed = cuberecon
                    self.cube_residuals = cuberes
                    self.cube_residuals_der = cuberesder
                    self.frame_final = frame
                    return pcs, cuberecon, cuberes, cuberesder, frame
            else:
                frame = res
                self.frame_final = frame
                return frame
        elif dataset.cube.ndim == 4:
            if full_output:
                if self.adimsdi == 'double':
                    cubereschan, cubereschander, frame = res
                    self.cube_residuals_per_channel = cubereschan
                    self.cube_residuals_per_channel_der = cubereschander
                    self.frame_final = frame
                    return cubereschan, cubereschander, frame
                elif self.adimsdi == 'single':
                    cuberes, cuberesresc, frame = res
                    self.cube_residuals = cuberes
                    self.cube_residuals_resc = cuberesresc
                    self.frame = frame
                    return cuberes, cuberesresc, frame
            else:
                frame = res
                self.frame = frame
                return frame