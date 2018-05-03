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
    """ HCI median subtraction algorithm.

    Parameters
    ----------
    mode : {"fullfr","annular"}, str optional
        In "simple" mode only the median frame is subtracted, in "annular" mode
        also the 4 closest frames given a PA threshold (annulus-wise) are
        subtracted.
    radius_int : int, optional
        The radius of the innermost annulus. By default is 0, if >0 then the
        central circular area is discarded.
    asize : int, optional
        The size of the annuli, in FWHM. Default is 2.
    delta_rot : int, optional
        Factor for increasing the parallactic angle threshold, expressed in
        FWHM. Default is 1 (excludes 1 FHWM on each side of the considered
        frame).
    nframes : even int optional
        Number of frames to be used for building the optimized reference PSF
        when working in annular mode.
    imlib : {'opencv', 'skimage'}, str optional
        Library used for image transformations. Opencv is faster than ndimage or
        skimage.
    interpolation : str, optional
        For 'skimage' library: 'nearneig', bilinear', 'bicuadratic',
        'bicubic', 'biquartic', 'biquintic'. The 'nearneig' interpolation
        is the fastest and the 'biquintic' the slowest. The 'nearneig' is
        the poorer option for interpolation of noisy astronomical images.
        For 'opencv' library: 'nearneig', 'bilinear', 'bicubic', 'lanczos4'.
        The 'nearneig' interpolation is the fastest and the 'lanczos4' the
        slowest and accurate. 'lanczos4' is the default.
    collapse : {'median', 'mean', 'sum', 'trimmean'}, str optional
        Sets the way of collapsing the frames for producing a final image.
    nproc : None or int, optional
        Number of processes for parallel computing. If None the number of
        processes will be set to cpu_count()/2. By default the algorithm works
        in single-process mode.

    """
    def __init__(self, mode='fullfr', radius_int=0, asize=1, delta_rot=1,
                 nframes=4, imlib='opencv', interpolation='lanczos4',
                 collapse='median', nproc=1):
        """ """
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
        """ Running the HCI median subtraction algorithm for model PSF
        subtraction.

        Parameters
        ----------
        dataset : HCIDataset object
            An HCIDataset object to be processed.
        full_output: bool, optional
            Whether to return the final median combined image only or with other
            intermediate arrays.
        verbose : bool, optional
            If True prints to stdout intermediate info.
            
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
    """ HCI PCA algorithm.

    Parameters
    ----------
    ncomp : int, optional
        How many PCs are used as a lower-dimensional subspace to project the
        target frames. For an ADI cube, ``ncomp`` is the number of PCs extracted
        from ``cube``. For the RDI case, when ``cube`` and ``cube_ref`` are
        provided, ``ncomp`` is the number of PCs obtained from ``cube_ref``.
        For an ADI+mSDI cube (e.g. SPHERE/IFS), if ``adimsdi`` is ``double``
        then ``ncomp`` is the number of PCs obtained from each multi-spectral
        frame (if ``ncomp`` is None then this stage will be skipped and the
        spectral channels will be combined without subtraction). If ``adimsdi``
        is ``single``, then ``ncomp`` is the number of PCs obtained from the
        whole set of frames (n_channels * n_adiframes).
    ncomp2 : int, optional
        Only used for ADI+mSDI cubes, when ``adimsdi`` is set to ``double``.
        ``ncomp2`` sets the number of PCs used in the second PCA stage (ADI
        fashion, using the residuals of the first stage). If None then the
        second PCA stage is skipped and the residuals are de-rotated and
        combined.
    svd_mode : {'lapack', 'arpack', 'eigen', 'randsvd', 'cupy', 'eigencupy', 'randcupy'}, str
        Switch for the SVD method/library to be used. ``lapack`` uses the LAPACK
        linear algebra library through Numpy and it is the most conventional way
        of computing the SVD (deterministic result computed on CPU). ``arpack``
        uses the ARPACK Fortran libraries accessible through Scipy (computation
        on CPU). ``eigen`` computes the singular vectors through the
        eigendecomposition of the covariance M.M' (computation on CPU).
        ``randsvd`` uses the randomized_svd algorithm implemented in Sklearn
        (computation on CPU). ``cupy`` uses the Cupy library for GPU computation
        of the SVD as in the LAPACK version. ``eigencupy`` offers the same
        method as with the ``eigen`` option but on GPU (through Cupy).
        ``randcupy`` is an adaptation of the randomized_svd algorithm, where all
        the computations are done on a GPU.
    scaling : {None, 'temp-mean', 'spat-mean', 'temp-standard', 'spat-standard'}
        With None, no scaling is performed on the input data before SVD. With
        "temp-mean" then temporal px-wise mean subtraction is done, with
        "spat-mean" then the spatial mean is subtracted, with "temp-standard"
        temporal mean centering plus scaling to unit variance is done and with
        "spat-standard" spatial mean centering plus scaling to unit variance is
        performed.
    adimsdi : {'double', 'single'}, str optional
        In the case ``cube`` is a 4d array, ``adimsdi`` determines whether a
        single or double pass PCA is going to be computed. In the ``single``
        case, the multi-spectral frames are rescaled wrt the largest wavelength
        to align the speckles and all the frames are processed with a single
        PCA low-rank approximation. In the ``double`` case, a firt stage is run
        on the rescaled spectral frames, and a second PCA frame is run on the
        residuals in an ADI fashion.
    mask_center_px : None or int
        If None, no masking is done. If an integer > 1 then this value is the
        radius of the circular mask.
    source_xy : tuple of int, optional
        For ADI PCA, this triggers a frame rejection in the PCA library.
        source_xy are the coordinates X,Y of the center of the annulus where the
        PA criterion will be used to reject frames from the library.
    delta_rot : int, optional
        Factor for tunning the parallactic angle threshold, expressed in FWHM.
        Default is 1 (excludes 1xFHWM on each side of the considered frame).
    imlib : {'opencv', 'skimage'}, str optional
        Library used for image transformations. Opencv is faster than ndimage or
        skimage.
    interpolation : str, optional
        For 'skimage' library: 'nearneig', bilinear', 'bicuadratic',
        'bicubic', 'biquartic', 'biquintic'. The 'nearneig' interpolation
        is the fastest and the 'biquintic' the slowest. The 'nearneig' is
        the poorer option for interpolation of noisy astronomical images.
        For 'opencv' library: 'nearneig', 'bilinear', 'bicubic', 'lanczos4'.
        The 'nearneig' interpolation is the fastest and the 'lanczos4' the
        slowest and accurate. 'lanczos4' is the default.
    collapse : {'median', 'mean', 'sum', 'trimmean'}, str optional
        Sets the way of collapsing the frames for producing a final image.
    check_mem : bool, optional
        If True, it check that the input cube(s) are smaller than the available
        system memory.
    """
    def __init__(self, ncomp=1, ncomp2=1, svd_mode='lapack', scaling=None,
                 adimsdi='double', mask_central_px=None, source_xy=None,
                 delta_rot=1, imlib='opencv', interpolation='lanczos4',
                 collapse='median', check_mem=True):
        """ """
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
        """ Running the HCI PCA algorithm for model PSF subtraction.

        Parameters
        ----------
        dataset : HCIDataset object
            An HCIDataset object to be processed.
        full_output: bool, optional
            Whether to return the final median combined image only or with other
            intermediate arrays.
        verbose : bool, optional
            If True prints intermediate info and timing.
        debug : bool, optional
            Whether to print debug information or not.

        """
        if not isinstance(dataset, HCIDataset):
            raise ValueError('`Dataset` must be a HCIDataset object')
        if self.source_xy is not None and self.fwhm is None:
            raise ValueError('`FWHM` has not been set')

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