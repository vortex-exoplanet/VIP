#! /usr/bin/env python

"""
Module with the HCI<post-processing algorithms> classes.
"""

from __future__ import division, print_function

__author__ = 'Carlos Alberto Gomez Gonzalez, Ralf Farkas'
__all__ = ['HCIMedianSub',
           'HCIPca',
           'HCILoci']

from sklearn.base import BaseEstimator
from .hci_dataset import HCIDataset
from .medsub import median_sub
from .metrics import snrmap_fast, snrmap
from .pca import pca
import pickle
import numpy as np
from .leastsq import xloci
from .conf.utils_conf import algo_calculates as calculates


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
            if self.dataset is not None and verbose:
                print("a new dataset was provided to run(), all previous "
                      "results were cleared.")
            self.dataset = dataset
            self._reset_results()

        return dataset

    def _get_calculations(self):
        """
        Get a list of all attributes which are *calculated*.

        This iterates over all the elements in an object and finds the functions
        which were decorated with ``@calculates`` (which are identified by the
        function attribute ``_calculates``). It then stores the calculated
        attributes, together with the corresponding method, and returns it.

        Returns
        -------
        calculations : dict
            Dictionary mapping a single "calculated attribute" to the method
            which calculates it.

        """
        calculations = {}
        for e in dir(self):
            try:
                for k in getattr(getattr(self, e), "_calculates"):
                    calculations[k] = e
            except AttributeError:
                pass

        return calculations

    def _reset_results(self):
        """
        Remove all calculated results from the object.
        
        By design, the HCIPostPRocAlgo's can be initialized without a dataset,
        so the dataset can be provided to the ``run`` method. This makes it
        possible to run the same algorithm on multiple datasets. In order not to
        keep results from an older ``run`` call when working on a new dataset,
        the stored results are reset using this function every time the ``run``
        method is called.
        """
        for attr in self._get_calculations():
            try:
                delattr(self, attr)
            except AttributeError:
                pass  # attribute/result was not calculated yet. Skip.

    def __getattr__(self, a):
        """
        ``__getattr__`` is only called when an attribute does *not* exist.
        
        Catching this event allows us to output proper error messages when an
        attribute was not calculated yet.
        """
        calculations = self._get_calculations()
        if a in calculations:
            raise AttributeError("The '{}' was not calculated yet. Call '{}' "
                                 "first.".format(a, calculations[a]))
        else:
            # this raises a regular AttributeError:
            return self.__getattribute__(a)

    def _show_attribute_help(self, function_name):
        """
        Print information about the attributes a method calculated.

        This is called *automatically* when a method is decorated with
        ``@calculates``.

        Parameters
        ----------
        function_name : string
            The name of the method.

        """
        calculations = self._get_calculations()

        print("These attributes were just calculated:")
        for a, f in calculations.items():
            if hasattr(self, a) and function_name == f:
                print("\t{}".format(a))
        
        not_calculated_yet = [(a, f) for a, f in calculations.items()
                              if (f not in self._called_calculators
                                  and not hasattr(self, a))]
        if len(not_calculated_yet) > 0:
            print("The following attributes can be calculated now:")
            for a, f in not_calculated_yet:
                print("\t{}\twith .{}()".format(a, f))

    @calculates("snr_map", "detection_map")
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

    @calculates("frame_final")
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

    @calculates("cube_residuals", "cube_residuals_der", "frame_final")
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

    @calculates("frame_final",
                "cube_reconstructed", "cube_residuals", "cube_residuals_der",
                "pcs",
                "cube_residuals_per_channel", "cube_residuals_per_channel_der",
                "cube_residuals_resc")
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


class HCILoci(HCIPostProcAlgo):
    """
    HCI LOCI algorithm.
    """

    def __init__(self, dataset=None, scale_list=None, metric="manhattan",
                 dist_threshold=90, delta_rot=0.5, delta_sep=(0.1, 1),
                 radius_int=0, asize=4, n_segments=4, solver="lstsq", tol=1e-3,
                 optim_scale_fact=1, adimsdi="skipadi", imlib="opencv",
                 interpolation="lanczos4", collapse="median", verbose=True):
        super(HCILoci, self).__init__(locals())

    @calculates("frame_final", "cube_res", "cube_der")
    def run(self, dataset=None, nproc=1, verbose=True):
        """
        Run the HCI LOCI algorithm for model PSF subtraction.

        """

        dataset = self._get_dataset(dataset, verbose)

        res = xloci(dataset.cube, dataset.angles, self.scale_list, dataset.fwhm,
                    self.metric, self.dist_threshold, self.delta_rot,
                    self.delta_sep, self.radius_int, self.asize,
                    self.n_segments, nproc, self.solver, self.tol,
                    self.optim_scale_fact, self.adimsdi, self.imlib,
                    self.interpolation, self.collapse, verbose,
                    full_output=True)
        
        self.cube_res, self.cube_der, self.frame_final = res

