#! /usr/bin/env python

"""
Module with the HCI<post-processing algorithms> classes.
"""

from __future__ import division, print_function

__author__ = 'Carlos Alberto Gomez Gonzalez, Ralf Farkas'
__all__ = ['HCIMedianSub',
           'HCIPca']

from sklearn.base import BaseEstimator
from .hci_dataset import HCIDataset
from .medsub import median_sub
from .metrics import snrmap_fast, snrmap
from .pca import pca
import pickle
import numpy as np


class HCIPostProcAlgo(BaseEstimator):
    """
    Base HCI post-processing algorithm class.
    """

    def __init__(self, locals_dict, *skip):
        """
        Set up the algorithm parameters.

        This does multiple things:

        - verify that ``dataset`` is a HCIDataset object or ``None`` (it could
          also be provided to ``run``)
        - store all the keywords (from ``locals_dict``) as object attributes, so
          they can be accessed e.g. in the ``run()`` method
        - print out the full algorithm settings (user provided parameters +
          default ones) if ``verbose=True``

        Parameters
        ----------
        locals_dict : dict
            This should be ``locals()``. ``locals()`` contains *all* the
            variables defined in the local scope. Passed to
            ``self._store_args``.
        *skip : list of strings
            Passed on to ``self._store_args``. Refer to its documentation.

        Examples
        --------

        .. code:: python

            # when subclassing HCIPostProcAlgo, make sure you call super()
            # with locals()! This means:

            class MySuperAlgo(HCIPostProcAlgo):
                def __init__(self, algo_param_1=42, cool=True):
                    super(MySuperAlgo, self).__init__(locals())
                
                @calculates("frame")
                def run(self, dataset=None):
                    self.frame = 2 * self.algo_param_1

        """

        dataset = locals_dict.get("dataset", None)
        if not isinstance(dataset, (HCIDataset, type(None))):
            raise ValueError('`dataset` must be a HCIDataset object or None')

        self._store_args(locals_dict, *skip)

        verbose = locals_dict.get("verbose", True)
        if verbose:
            self._print_parameters()

    def _print_parameters(self):
        """ Printing out the parameters of the algorithm.
        """
        dicpar = self.get_params()
        for key in dicpar.keys():
            print("{}: {}".format(key, dicpar[key]))

    def _store_args(self, locals_dict, *skip):
        # TODO: this could be integrated with sklearn's BaseEstimator methods
        for k in locals_dict:
            if k == "self" or k in skip:
                continue
            setattr(self, k, locals_dict[k])

    def _get_dataset(self, dataset=None, verbose=True):
        """
        Handle a dataset passed to ``run()``.

        It is possible to specify a dataset using the constructor, or using the
        ``run()`` function. This helper function checks that there is a dataset
        to work with.

        Parameters
        ----------
        dataset : HCIDataset or None, optional
        verbose : bool, optional
            If ``True``, a message is printed out when a previous dataset was
            overwritten.

        Returns
        -------
        dataset : HCIDataset

        """
        if dataset is None:
            dataset = self.dataset
            if self.dataset is None:
                raise ValueError("no dataset specified!")
        else:
            self.dataset = dataset # needed for snr map generation
            if verbose:
                #print("self.dataset overwritten with the one you provided.")
                # -> debug
                pass

        return dataset

    def make_snr_map(self, method='fast', mode='sss', nproc=1, verbose=True):
        """
        Calculate a SNR map from ``self.frame_final``.

        Parameters
        ----------
        method : {'xpx', 'fast'}, str optional
            Method for the S/N map creation. The `xpx` method uses the per-pixel
            procedure of `vip_hci.metrics.snrmap`, while the `fast` method uses
            the approximation in `vip_hci.metrics.snrmap_fast`.
        mode : {'sss', 'peakstddev'}, optional
            [method=xpx] 'sss' uses the approach with the small sample
            statistics penalty and 'peakstddev' uses the
            peak(aperture)/std(annulus) version.
        nproc : int, optional
            Number of processes. Defaults to single-process (serial) processing.
        verbose : bool, optional
            Show more output.

        Notes
        -----
        This is needed for "classic" algorithms that produce a final residual
        image in their ``.run()`` method. To obtain a "detection map", which can
        be used for counting true/false positives, a SNR map has to be created.
        For other algorithms (like ANDROMEDA) which directly create a SNR or a
        probability map, this method should be overwritten and thus disabled.

        """
        if not hasattr(self, "frame_final"):
            raise RuntimeError("`.frame_final` attribute not found. Call"
                               "`.run()` first.")

        if method == 'fast':
            self.snr_map = snrmap_fast(self.frame_final, self.dataset.fwhm,
                                       nproc=nproc, verbose=verbose)
        elif method == 'xpx':
            self.snr_map = snrmap(self.frame_final, self.dataset.fwhm,
                                  plot=False, mode=mode, source_mask=None,
                                  nproc=nproc, save_plot=None, plot_title=None,
                                  verbose=verbose)
        else:
            raise ValueError('`method` not recognized')

        self.detection_map = self.snr_map

    def save(self, filename):
        """
        Pickle the algo object and save it to disk.

        Note that this also saves the associated ``self.dataset``, in a
        non-optimal way.
        """
        pickle.dump(self, open(filename, "wb"))

    def run(self, dataset=None, nproc=1, verbose=True):
        """
        Run the algorithm. Should at least set `` self.frame_final``.

        Notes
        -----
        This is the required signature of the ``run`` call. Child classes can
        add their own keyword arguments if needed.
        """

        raise NotImplementedError


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
    delta_sep : float or tuple of floats, optional
        The threshold separation in terms of the mean FWHM (for ADI+mSDI data).
        If a tuple of two values is provided, they are used as the lower and
        upper intervals for the threshold (grows as a function of the
        separation).
    nframes : even int, optional
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
    verbose : bool, optional
        Show more output.

    Notes
    -----
    TODO:
    - output cube and frames as HCIDataset and HCIFrame objects?

    """
    def __init__(self, dataset=None, mode='fullfr', radius_int=0, asize=1,
                 delta_rot=1, delta_sep=(0.2, 1), nframes=4, imlib='opencv',
                 interpolation='lanczos4', collapse='median',
                 verbose=True):
        super(HCIMedianSub, self).__init__(locals())

    def run(self, dataset=None, nproc=1, verbose=True):
        """ Running the HCI median subtraction algorithm for model PSF
        subtraction.

        Parameters
        ----------
        dataset : HCIDataset object
            An HCIDataset object to be processed.
        full_output: bool, optional
            Whether to return the final median combined image only or with other
            intermediate arrays.
        nproc : int, optional
            Number of processes for parallel computing. Defaults to single-core
            processing.
        verbose : bool, optional
            If True prints to stdout intermediate info.

        """

        dataset = self._get_dataset(dataset, verbose)

        if self.mode == 'annular' and dataset.fwhm is None:
            raise ValueError('`fwhm` has not been set')

        res = median_sub(dataset.cube, dataset.angles, dataset.wavelengths,
                         dataset.fwhm, self.radius_int, self.asize,
                         self.delta_rot, self.delta_sep, self.mode,
                         self.nframes, self.imlib, self.interpolation,
                         self.collapse, nproc, full_output=True,
                         verbose=verbose)

        self.cube_residuals, self.cube_residuals_der, self.frame_final = res



class HCIPca(HCIPostProcAlgo):
    """ HCI PCA algorithm.

    Parameters
    ----------
    dataset : HCIDataset object, optional
        An HCIDataset object to be processed. Can also be passed to ``.run()``.
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
    def __init__(self, dataset=None, ncomp=1, ncomp2=1, svd_mode='lapack', scaling=None,
                 adimsdi='double', mask_central_px=None, source_xy=None,
                 delta_rot=1, imlib='opencv', interpolation='lanczos4',
                 collapse='median', check_mem=True, crop_ifs=True, verbose=True):
        
        super(HCIPca, self).__init__(locals())

        # TODO: order/names of parameters are not consistent with ``pca`` core function

    def run(self, dataset=None, nproc=1, verbose=True, debug=False):
        """
        Run the HCI PCA algorithm for model PSF subtraction.

        Notes
        -----
        creates/sets the ``self.frame_final`` attribute, and depending on the
        input parameters:

            3D case:
                cube_reconstructed
                cube_residuals
                cube_residuals_der
            3D case, source_xy is None:
                cube_residuals
                pcs
            4D case, adimsdi="double":
                cube_residuals_per_channel
                cube_residuals_per_channel_der
            4D case, adimsdi="single":
                cube_residuals
                cube_residuals_resc

        Parameters
        ----------
        dataset : HCIDataset, optional
            Dataset to process. If not provided, ``self.dataset`` is used (as
            set when initializing this object).
        nproc : int, optional
            (not used) Note that ``HCIPca`` always works in single-processing
            mode.
        verbose : bool, optional
            Show more output.
        debug : bool, optional
            Whether to print debug information or not.

        """
        dataset = self._get_dataset(dataset, verbose)

        if self.source_xy is not None and dataset.fwhm is None:
            raise ValueError('`fwhm` has not been set')

        res = pca(dataset.cube, dataset.angles, dataset.cuberef,
                  dataset.wavelengths, self.ncomp, self.ncomp2, self.svd_mode,
                  self.scaling, self.adimsdi, self.mask_central_px,
                  self.source_xy, self.delta_rot, dataset.fwhm, self.imlib,
                  self.interpolation, self.collapse, self.check_mem,
                  self.crop_ifs, nproc, full_output=True, verbose=verbose,
                  debug=debug)

        if dataset.cube.ndim == 3:
            if self.source_xy is not None:
                cuberecon, cuberes, cuberesder, frame = res
                self.cube_reconstructed = cuberecon
                self.cube_residuals = cuberes
                self.cube_residuals_der = cuberesder
                self.frame_final = frame
            else:
                pcs, cuberecon, cuberes, cuberesder, frame = res
                self.pcs = pcs
                self.cube_reconstructed = cuberecon
                self.cube_residuals = cuberes
                self.cube_residuals_der = cuberesder
                self.frame_final = frame
        elif dataset.cube.ndim == 4:
            if self.adimsdi == 'double':
                cubereschan, cubereschander, frame = res
                self.cube_residuals_per_channel = cubereschan
                self.cube_residuals_per_channel_der = cubereschander
                self.frame_final = frame
            elif self.adimsdi == 'single':
                cuberes, cuberesresc, frame = res
                self.cube_residuals = cuberes
                self.cube_residuals_resc = cuberesresc
                self.frame_final = frame





