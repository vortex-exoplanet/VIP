"""
Implementation of the PACO algorithm for VIP, based on [FLA18]_.

Variable naming is based on the notation of [FLA18]_, see table 1 in the paper 
for a description.

Last updated 2022-05-09 by Evert Nasedkin (nasedkinevert@gmail.com).

.. [FLA18]
   | Flasseur et al. 2018
   | **Exoplanet detection in angular differential imaging by statistical 
     learning of the nonstationary patch covariances. The PACO algorithm**
   | *Astronomy & Astrophysics, Volume 618, p. 138*
   | `https://ui.adsabs.harvard.edu/abs/2018A%26A...618A.138F/abstract
     <https://ui.adsabs.harvard.edu/abs/2018A%26A...618A.138F/abstract>`_
     
"""

import sys
#import os
from abc import abstractmethod
# Required so numpy parallelization doesn't conflict with multiprocessing
# os.environ["MKL_NUM_THREADS"] = "1"
# os.environ["NUMEXPR_NUM_THREADS"] = "1"
# os.environ["OMP_NUM_THREADS"] = "1"

#from multiprocessing import Pool
from typing import Tuple, Union, Optional, Callable
import numpy as np
from scipy import ndimage
from scipy.ndimage import filters

from ..config.utils_conf import pool_map, iterable
from ..preproc.rescaling import frame_px_resampling, cube_px_resampling, frame_shift
from ..var.coords import cart_to_pol, pol_to_cart
from ..metrics.detection import detection
from ..fm import normalize_psf
__author__ = "Evert Nasedkin"
__all__ = ['FastPACO',
           'FullPACO']


class PACO:
    """
    This class implements the bulk of the PACO algorithm as described in 
    [FLA18]_. In general, the idea is to take in an ADI stack of images and 
    statistically determine if there is a signal above the background in each 
    'patch' of the image. This is done by tracing the ark of the hypothesized 
    planet through the stack, and comparing this set of patches to a set 
    consisting of background only. This is done for each pixel (or sub-pixel) 
    location in the image. The output is a signal-to-noise and/or a flux map 
    over the field of view. The user can choose to use FullPACO or FastPACO, 
    which are described by algorithms 1 and 2 of [FLA18]_. FastPACO has been 
    parallelized, and is the recommended usage.

    This output can then be used to compute an unbiased estimate of the flux
    of point sources detected in the image above some user-supplied detection
    threshold.

    Parameters
    ----------
    cube : numpy.ndarray
        3D science frames taken in pupil tracking/ADI mode.
        Dimensions should be (time, x, y), and units should be detector units 
        (ie output of SPHERE or GPI reduction pipelines). The data should be 
        centered, and have pre-processing already applied (e.g. bad pixel 
        correction).
    angles : numpy.ndarray
        List of parallactic angles for each frame in degrees. Length of this 
        array should be the same as the time axis of the science cube. 
    psf : numpy.ndarray
        Unsaturated PSF image. If a cube is provided, the median of the cube 
        will be used.
    dit_psf : float, optional
        Integration time of the unsaturated PSF in seconds. The PSF is 
        normalised to dit_science/dit_psf/nd_transmission.
    dit_science : float, optional
        Integration time of the science frames in seconds. The PSF is normalised
        to dit_science/dit_psf/nd_transmission.
    nd_transmission : float, optional
        Transmission of an ND filter used to aquire the unsaturated PSF. The PSF 
        is normalised to dit_science/dit_psf/nd_transmission.
    fwhm : float, optional
        FWHM of PSF in arcseconds. Default values give 4px radius.
    pixscale : float, optional
        Detector pixel scale in arcseconds per pixel.  Default values give 4px 
        radius.
    rescaling_factor : float, optional
        Scaling for sub/super pixel resolution for PACO. Will rescale both the 
        science cube and the PSF.
    verbose : bool, optional
        Sets level of printed outputs.
    """

    def __init__(self,
                 cube: np.ndarray,
                 angles: np.ndarray,
                 psf: np.ndarray,
                 dit_psf: Optional[float] = 1.0,
                 dit_science: Optional[float] = 1.0,
                 nd_transmission: Optional[float] = 1.0,
                 fwhm: Optional[float] = 4.0,
                 pixscale: Optional[float] = 1.0,
                 rescaling_factor: Optional[float] = 1.0,
                 verbose: Optional[bool] = False) -> None:

        # Science image setup
        try:
            self.cube = cube
        except BaseException:
            raise ValueError("You must provide a 3D cube of science data!")

        self.num_frames = self.cube.shape[0]
        self.width = self.cube.shape[2]
        self.height = self.cube.shape[1]

        # Parallactic angles
        try:
            self.angles = angles
        except BaseException:
            raise ValueError("You must provide an array of parallactic angles!")

        # Pixel scaling
        self.pixscale = pixscale
        self.rescaling_factor = rescaling_factor

        # PSF setup
        self.fwhm = int(fwhm/pixscale)
        try:
            # How do we want to deal with stacks of psfs? Median? Just take the first one?
            # Ideally if nPSFs = nImages, use each for each. Need to update!
            if len(psf.shape) > 2:
                psf = np.nanmedian(psf, axis=0)
            self.psf = psf * dit_science/dit_psf/nd_transmission
            self.dit_science = dit_science
            self.dit_psf = dit_psf

            mask = create_boolean_circular_mask(self.cube[0].shape,
                                                radius=self.fwhm)
            self.patch_area_pixels = self.cube[0][mask].ravel().shape[0]
        except BaseException:
            raise ValueError("You must provide an unsaturated PSF image!")

        self.patch_width = 2*int(self.fwhm) + 3
        self.verbose = verbose

        # These are what we're calculating
        self.snr = None
        self.flux = None
        self.std = None

        # Diagnostics
        if self.verbose:
            print("---------------------- ")
            print("Summary of PACO setup: \n")
            print(f"Image Cube shape = {self.cube.shape}")
            print(f"PIXSCALE = {self.pixscale:06}")
            print("PSF |  Area  |  Rad   |  Width | ")
            print(f"    |   {self.patch_area_pixels:01}   |"
                  + f"  {self.fwhm:02}    |  {self.psf.shape[0]:03}   | ")
            print(f"Patch width: {self.patch_width}")
            print("---------------------- \n")
            sys.stdout.flush()

    @abstractmethod
    def PACOCalc(self,
                 phi0s : np.ndarray,
                 use_subpixel_psf_astrometry : Optional[bool] = True,
                 cpu : Optional[int] = 1) -> None:
        """
        This function is algorithm dependant, and sets up the actual
        calculation process.

        Parameters
        ----------
        phi0s : numpy.ndarray
            Array of pixel coordinates to try to search for the planet signal. 
            Typically a grid created using numpy.meshgrid.
        use_subpixel_psf_astrometry : bool
            If true, the PSF model for each patch is shifted to the correct
            location as predicted by the starting location and the parallactic
            angles, before being resampled for the patch. If false, the PSF
            model is simply located at the center of each patch. Significantly
            improves performance if set to False, but the SNR is reduced.
        cpu : int, optional
            Number of cpus to use for parallelization.

        Returns
        -------
        a : numpy.ndarray
            a_l from Equation 15 of Flasseur+ 2018
        b : numpy.ndarray
            b_l from Equation 16 of Flasseur+ 2018
        """

    def run(self,
            cpu : Optional[int] = 1,
            imlib : Optional[str] = 'vip-fft',
            interpolation : Optional[str]= 'lanczos4',
            keep_center : Optional[bool] = True,
            use_subpixel_psf_astrometry : Optional[bool] = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run method of the PACO class. This function wraps up the PACO
        calculation steps, and returns the snr and flux estimates as
        outputs. The image library arguments are used if a rescaling of the
        data is desired: upsampling the images will result in a better SNR
        detection, but will be slower.

        Parameters
        ----------
        cpu : int, optional
            Number of processers to use
        imlib : str, optional
            See the documentation of the ``vip_hci.preproc.frame_px_resampling``
            function.
        interpolation : str, optional
            See the documentation of the ``vip_hci.preproc.frame_px_resampling``
            function.
        keep_center: bool, opt
            If input dimensions are even and the star centered (i.e. on
            dim//2, dim//2), whether to keep the star centered after scaling, i.e.
            on (new_dim//2, new_dim//2). For a non-centered input cube, better to
            leave it to False.
        use_subpixel_psf_astrometry : bool
            If true, the PSF model for each patch is shifted to the correct
            location as predicted by the starting location and the parallactic
            angles, before being resampled for the patch. If false, the PSF
            model is simply located at the center of each patch. Significantly
            improves performance if set to False, but the SNR is reduced.

        Returns
        -------
        snr : numpy.ndarray
            2D map of the signal-to-noise estimate as computed by PACO.
            This is b/sqrt(a), as in eqn 24 of Flasseur 2018.
        flux : numpy.ndarray
            2D map of the flux estimate as computed by PACO
            This is b/a, as in eqn 21 of Flasseur 2018.
        """

        if self.rescaling_factor != 1:
            self.rescale_cube_and_psf(imlib=imlib,
                                      interpolation=interpolation,
                                      keep_center=keep_center,
                                      verbose=self.verbose)
            if self.verbose:
                print("---------------------- ")
                print(f"Using {cpu} processor(s).")
                print(f"Rescaled Image Cube shape: {self.cube.shape}")
                print("Rescaled PSF:")
                print("PSF |  Area  |  Rad   |  Width | ")
                print(f"    |   {self.patch_area_pixels:01}   |"
                      + f"  {self.fwhm:02}    |  {self.psf.shape[0]:03}   | ")
                print("---------------------- \n")

        # Setup pixel coordinates
        x, y = np.meshgrid(np.arange(0, self.height),
                           np.arange(0, self.width))
        phi0s = np.column_stack((x.flatten(), y.flatten()))
        # Compute a,b
        a, b = self.PACOCalc(np.array(phi0s), cpu=cpu)

        # Reshape into a 2D image, with the same dimensions as the input images
        a = np.reshape(a, (self.height, self.width))
        b = np.reshape(b, (self.height, self.width))
        # Output arrays
        snr = b/np.sqrt(a)
        flux = b/a
        self.snr = snr
        self.flux = flux
        self.std = 1/np.sqrt(a)
        return snr, flux

    """
    Utility Functions
    """
    # Set the image stack to be processed

    def set_cube(self, cube: np.ndarray) -> None:
        """
        Provide a 3D image array to process. This updates
        the science cube, and the associated dimensions.

        Parameters
        ----------
        cube : numpy.ndarray
            3D science frames taken in pupil tracking/ADI mode.
            Dimensions should be (time, x, y), and units should be detector
            units (ie output of SPHERE or GPI reduction pipelines). The data
            should be centered, and have pre-processing already applied (e.g.
            bad pixel correction).
        """
        self.cube = np.array(cube)
        self.num_frames = self.cube.shape[0]
        self.width = self.cube.shape[2]
        self.height = self.cube.shape[1]

    # Set the template PSF
    def set_psf(self, psf: np.ndarray) -> None:
        """
        Read in the PSF template

        Parameters
        ----------
        psf: numpy.ndarray
            An unsaturated psf to use as the template.
        """
        self.psf = psf

    # Set parallactic angles
    def set_angles(self, angles: np.ndarray) -> None:
        """
        Set the rotation angle for each frame

        Parameters
        ----------
        angles: numpy.ndarray
            A list of the parallactic angles for each frame of the science data.
        """

        self.angles = angles

    def get_patch(self,
                  px: Tuple[int, int],
                  width: Optional[int] = None,
                  mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Gets patch at given pixel px with size k for the current img sequenc

        Parameters
        ----------
        px : Tuple[int, int]
            Pixel coordinates for center of patch
        width : int
            width of a square patch to be masked

        Returns
        -------
        patch : numpy.ndarray
            A PACO "patch". This is a column through the time dimension of the
            unrotated frames, used to build the background statistics at the
            location of a given pixel.
        """
        if width is None:
            width = self.patch_width
        if mask is None:
            mask = create_boolean_circular_mask(self.cube[0].shape,
                                                radius=self.fwhm,
                                                center=px)
        k = int(width/2)
        if width % 2 != 0:
            k2 = k+1
        else:
            k2 = k
        nx, ny = np.shape(self.cube[0])[:2]
        if px[0]+k2 > nx or px[0]-k < 0 or px[1]+k2 > ny or px[1]-k < 0:
            return np.ones((self.num_frames, self.patch_area_pixels))*np.nan
        patch = self.cube[np.broadcast_to(mask, self.cube.shape)]\
            .reshape(self.num_frames, self.patch_area_pixels)
        return patch

    def set_scale(self, scale: float) -> None:
        """
        Set subpixel scaling factor

        Parameters
        ----------
        scale : float
            Scaling factor. Greater than one will result in an upsampled image,
            less than one will result in a downsampled image.
        """

        self.rescaling_factor = scale

    def rescale_cube_and_psf(self,
                             imlib: Optional[str] = 'vip-fft',
                             interpolation: Optional[str] = 'lanczos4',
                             keep_center: Optional[bool] = True) -> None:
        """
        Rescale each image in the stack by the class level scaling factor
        set during initialization or with set_scale. A scale factor of greater
        than one will upsample the image, a factor of less than one will downsample
        the image. This function wraps the VIP scaling function, and uses the same
        arguments to choose libraries and interpolation methods.

        Parameters
        ----------
        imlib : str, optional
            See the documentation of the ``vip_hci.preproc.frame_px_resampling``
            function.
        interpolation : str, optional
            See the documentation of the ``vip_hci.preproc.frame_px_resampling``
            function.
        keep_center: bool, opt
            If input dimensions are even and the star centered (i.e. on
            dim//2, dim//2), whether to keep the star centered after scaling, i.e.
            on (new_dim//2, new_dim//2). For a non-centered input cube, better to
            leave it to False.
        """

        if self.rescaling_factor == 1:
            if self.verbose:
                print("Scale is 1, no scaling applied.")
            return

        # Resample the science cube
        cube_px_resampling(self.cube,
                           self.rescaling_factor,
                           imlib=imlib,
                           interpolation=interpolation,
                           keep_center=keep_center,
                           verbose=False)

        self.pixscale = self.pixscale/self.rescaling_factor
        self.fwhm = int(self.fwhm*self.rescaling_factor)

        # Resample the PSF
        if self.psf is not None:
            self.psf = frame_px_resampling(self.psf,
                                           self.rescaling_factor,
                                           imlib=imlib,
                                           interpolation=interpolation,
                                           keep_center=keep_center,
                                           verbose=False)
        mask = create_boolean_circular_mask(self.psf.shape, self.fwhm)
        self.patch_area_pixels = self.psf[mask].shape[0]
        self.patch_width = 2*int(self.fwhm) + 3

    """
    Math Functions
    """

    def psf_model_function(self, mean: float, model: Callable,
                           params: dict) -> np.ndarray:
        """
        This function is deprecated in favour of directly supplying
        a PSF. In principle, an analytic model (ie a gaussian or moffat PSF)
        can be used in place of a measured unsaturated PSF.

        Parameters
        ----------
        mean : float
            If using the psfTemplateModel function, the mean
        model : dnc
            numpy statistical model (need to import numpy module for this)
        **kwargs: dict
            additional arguments for model

        Returns
        -------
        self.psf : numpy.ndarray
            Returns the PSF template used by PACO
        """

        if self.psf:
            return self.psf
        if model is None:
            print("Please input either a 2D PSF or a model function.")
            sys.exit(1)
        else:
            if model.__name__ == "psfTemplateModel":
                try:
                    self.psf = model(mean, params)
                    return self.psf
                except ValueError:
                    print("Fix template size")
            self.psf = model(mean, params)
            return self.psf

    def al(self,
           hfl: Union[list, np.ndarray],
           Cfl_inv: Union[list, np.ndarray],
           method: Optional[str] = "") -> np.ndarray:
        """
        a_l
        The sum of a_l is the inverse of the variance of the background at the given pixel.
        Einsum can get slow with large tensors, and may not actually be faster.
        If einsum is used, arguments must be numpy arrays, otherwise lists.

        Parameters
        ----------
        hfl : list
            This is a list of flattened psf templates
        Cfl_inv : list
            This is a list of inverse covariance matrices
        method: string
            Can be empty or "einsum". This determines the method
            used to do the matrix operations. "einsum" is slower for large arrays.

        Returns
        -------
        a : numpy.ndarray
            a_l from equation 15 of Flasseur 2018.
        """
        if method == "einsum":
            d1 = np.einsum('ijk,gj', Cfl_inv, hfl)
            return np.einsum('ml,ml', hfl, np.diagonal(d1).T)

        a = np.sum(np.array([np.dot(hfl[i], np.dot(Cfl_inv[i], hfl[i]).T)
                             for i in range(len(hfl))]), axis=0)
        return a

    def bl(self, hfl: Union[list, np.ndarray],
           Cfl_inv: Union[list, np.ndarray],
           r_fl: Union[list, np.ndarray],
           m_fl: Union[list, np.ndarray],
           method: Optional[str] = "") -> np.ndarray:
        """
        b_l
        The sum of b_l is the flux estimate at the given pixel.
        Einsum can get slow with large tensors, and may not actually be faster.
        If einsum is used, arguments must be numpy arrays, otherwise lists.

        Parameters
        ----------
        hfl : numpy.ndarray
            This is an array of flattened psf templates.
        Cfl_inv : numpy.ndarray
            This is an array of inverse covariance matrices.
        r_fl : numpy.ndarray
            This is an array of flux measurements following the predicted path.
        m_fl : numpy.ndarray
            This is an array of mean background statistics for each location in the path.
        method: string
            Can be empty or "einsum". This determines the method
            used to do the matrix operations. "einsum" is slower for large arrays.

        Returns
        -------
        b : numpy.ndarray
            b_l from equation 16 of Flasseur 2018.

        """
        if method == "einsum":
            d1 = np.einsum('ijk,gj', Cfl_inv, r_fl-m_fl)
            return np.einsum('ml,ml', hfl, np.diagonal(d1).T)

        b = np.sum(np.array([np.dot(np.dot(Cfl_inv[i], hfl[i]).T, (r_fl[i]-m_fl[i]))
                             for i in range(len(hfl))]), axis=0)
        return b

    """
    FluxPACO
    """

    def flux_estimate(self,
                      phi0s: np.ndarray,
                      eps: Optional[float] = 0.1,
                      initial_est: Optional[list] = [0.0]) -> list:
        """
        Unbiased estimate of the flux of a source located at p0
        The estimate of the flux is given by ahat * h, where h is the PSF template.
        This implements algorithm 3 from Flasseur+ 2018.
        TODO: Further testing to ensure that the extracted contrast is actually unbiased.
        Don't trust this estimate without checking!

        Parameters
        ----------
        phi0s : numpy.ndarray
            List of locations of sources to compute unbiased flux estimate in pixel units.
            Origin is at the bottom left. Should be a list of (x,y) tuples, or a 2D numpy
            array.
        eps : float
            Precision requirement for iteration (0,1)
        initial_est : float
            Initial estimate of the flux at p0 in contrast units

        Returns
        -------
        ests : list
            List of a-hat values for each detected source in the SNR map. This is the unbiased
            estimate of the flux at that location. Practically, this is similar to negative PSF
            injection. If the PSF is correctly normalized, this should be in contrast units.
        stds : list
            List of the estimated standard deviation on the flux estimates in contrast units.
        norm : float
            np.nanmax(psf) - Scaling factor for flux estimate and standard deviations.
        """
        print("Computing unbiased flux estimate...")

        if self.verbose:
            print("Initial guesses:")
            print("Positions: ", phi0s)
            print("Contrasts: ", initial_est)

        dim = self.width/2
        # Create arrays needed for storage
        # Store for each image pixel, for each temporal frame an image
        # for patches: for each time, we need to store a column of patches
        normalised_psf,norm,fwhm = normalize_psf(self.psf,
                                            fwhm='fit',
                                            size=None,
                                            threshold=None,
                                            mask_core=None,
                                            model='airy',
                                            imlib='vip-fft',
                                            interpolation='lanczos4',
                                            force_odd=False,
                                            full_output=True,
                                            verbose=self.verbose,
                                            debug=False)

        psf_mask = create_boolean_circular_mask(normalised_psf.shape, radius=self.fwhm)
        hoff = np.zeros((self.num_frames,self.num_frames, self.patch_area_pixels)) # The off axis PSF at each point
        # Create arrays needed for storage
        # Store for each image pixel, for each temporal frame an image
        # for patches: for each time, we need to store a column of patches
        # 2d selection of pixels around a given point
        x, y = np.meshgrid(np.arange(-dim, dim), np.arange(-dim, dim))

        ests = []
        stds = []
        for i, p0 in enumerate(phi0s):
            p0 = (p0[1],p0[0])
            angles_px = np.array(get_rotated_pixel_coords(x, y, p0, self.angles))
            hon = []
            for l, ang in enumerate(angles_px):
                # Get the column of patches at this point
                offax = frame_shift(normalised_psf,
                                    ang[1]-int(ang[1]),
                                    ang[0]-int(ang[0]),
                                    imlib='vip-fft',
                                    interpolation='lanczos4',
                                    border_mode='reflect')[psf_mask]
                hoff[l,l] = offax
                hon.append(offax)

            Cinv, m, patches = self.compute_statistics(np.array(angles_px).astype(int))
            # Get Angles
            # Ensure within image bounds
            # Extract relevant patches and statistics
            Cinlst = []
            mlst = []
            patch = []
            for l, ang in enumerate(angles_px):
                Cinlst.append(Cinv[int(ang[0]), int(ang[1])])
                mlst.append(m[int(ang[0]), int(ang[1])])
                patch.append(patches[int(ang[0]), int(ang[1]),l])
            a = self.al(hon,Cinlst)
            b = self.bl(hon, Cinlst,patch, mlst)
            print(b/a)
            # Fill patches and signal template


            # Unbiased flux estimation
            ahat = initial_est[i]
            aprev = 1e10 # Arbitrary large value so that the loop will run
            while np.abs(ahat - aprev) > np.abs(ahat * eps):
                a = 0.0
                b = 0.0
                # the mean of a temporal column of patches at each pixel
                m = np.zeros((self.num_frames, self.patch_area_pixels))
                # the inverse covariance matrix at each point
                Cinv = np.zeros((self.num_frames, self.patch_area_pixels, self.patch_area_pixels))

                # Patches here are columns in time
                for l,ang in enumerate(angles_px):
                    apatch = self.get_patch(ang.astype(int))
                    m[l], Cinv[l] = self.iterate_flux_calc(ahat, apatch, hoff[l])
                # Patches here are where the planet is expected to be
                a = self.al(hon, Cinv)
                b = self.bl(hon, Cinv, patch, m)
                aprev = ahat
                ahat = b/a
                if self.verbose:
                    print(f"Flux estimate: {ahat/norm}")
            ests.append(np.abs(ahat/norm))
            stds.append(1/np.sqrt(a)/norm)
        print("Extracted contrasts")
        print("-------------------")
        for i in range(len(phi0s)):
            print(
                f"x: {phi0s[i][0]}, y: {phi0s[i][1]}, flux: {ests[i]}±{stds[i]}")
        return ests, stds, norm

    def iterate_flux_calc(self, est: float, patch: np.ndarray,
                          model: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the iterative estimates for the mean and inverse covariance.

        Parameters
        ----------
        est : float
            Current estimate for the magnitude of the flux
        patch : numpy.ndarray
            Column of patches about p0
        model : numpy.ndarray
            Template for PSF

        Returns
        -------
        m : numpy.ndarray
            Mean of background patches.
        Cinv : numpy.ndarray
            List of inverse covariance matrices between the patches.
        """

        if patch is None:
            return None, None

        unbiased = np.array([apatch - est*model[l] for l,apatch in enumerate(patch)])
        m,Cinv = compute_statistics_at_pixel(unbiased)
        return m, Cinv

    def subpixel_threshold_detect(self, snr_map: np.ndarray,
                                  threshold: float,
                                  mode: Optional[str] = 'lpeaks',
                                  bkg_sigma: Optional[float] = 5.0,
                                  matched_filter: Optional[bool] = False,
                                  mask: Optional[bool] = True,
                                  full_output: Optional[bool] = False,
                                  cpu: Optional[int] = 1) -> np.ndarray:
        """ Wraps VIP.metrics.detection.detection, see that function for further documentation.
        Note that the output convention here is different - this function returns xx,yy.

        Finds blobs in a 2d array. The algorithm is designed for automatically
        finding planets in post-processed high contrast final frames. Blob can be
        defined as a region of an image in which some properties are constant or
        vary within a prescribed range of values. See ``Notes`` below to read about
        the algorithm details.
        Parameters
        ----------
        snr_map : numpy ndarray, 2d
            Input frame.
        threshold : float
            S/N threshold for deciding whether the blob is a detection or not. Used
            to threshold the S/N map when ``mode`` is set to 'snrmap' or 'snrmapf'.
        mode : {'lpeaks', 'log', 'dog', 'snrmap', 'snrmapf'}, optional
            Sets with algorithm to use. Each algorithm yields different results. See
            notes for the details of each method.
        bkg_sigma : int or float, optional
            The number standard deviations above the clipped median for setting the
            background level. Used when ``mode`` is either 'lpeaks', 'dog' or 'log'.
        matched_filter : bool, optional
            Whether to correlate with the psf of not. Used when ``mode`` is either
            'lpeaks', 'dog' or 'log'.
        mask : bool, optional
            If True the central region (circular aperture of 2*FWHM radius) of the
            image will be masked out.
        full_output : bool, optional
            Whether to output just the coordinates of blobs that fulfill the SNR
            constraint or a table with all the blobs and the peak pixels and SNR.
        cpu : None or int, optional
            The number of processes for running the ``snrmap`` function.
        verbose : bool, optional
            Whether to print to stdout information about found blobs.
        Returns
        -------
        peaks : np.ndarray
            xx,yy values of the centers of local maxima above the provided threshold
        """
        peaks = detection(snr_map,
                          fwhm=self.fwhm,
                          psf=self.psf/np.nanmax(self.psf),
                          mode=mode,
                          bkg_sigma=bkg_sigma,
                          matched_filter=matched_filter,
                          mask=mask,
                          snr_thresh=threshold,
                          nproc=cpu,
                          plot=False,
                          debug=False,
                          full_output=full_output,
                          verbose=self.verbose)
        return peaks.T

    def pixel_threshold_detection(
            self, snr_map: np.ndarray, threshold: float) -> np.ndarray:
        """
        Returns a list of the pixel coordinates of center of signals above a given threshold

        Parameters
        ----------
        snr_map : numpy.ndarray
            SNR map, b/sqrt(a) as computed by run()
        threshold: float
            Threshold for detection in sigma

        Returns
        -------
        locs : numpy.array
            Array of (x,y) pixel location estimates for the location of point sources
            above the provided threshold.
        """

        data_max = filters.maximum_filter(snr_map, size=self.fwhm)
        maxima = (snr_map == data_max)
        diff = (data_max > threshold)
        maxima[diff == 0] = 0

        labeled, _ = ndimage.label(maxima)
        slices = ndimage.find_objects(labeled)
        x, y = [], []
        for dy, dx in slices:
            x_center = (dx.start + dx.stop - 1)/2
            x.append(x_center)
            y_center = (dy.start + dy.stop - 1)/2
            y.append(y_center)
        return np.array(list(zip(x, y)))

    def compute_statistics(self, phi0s : np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        This function computes the mean and inverse covariance matrix for
        each patch in the image stack in Serial. Used by FastPACO and flux
        estimation.

        Parameters
        ----------
        phi0s : numpy.ndarray
            Array of pixel locations to estimate companion position

        Returns
        -------
        Cinv : numpy.ndarray
            Inverse covariance matrix between the the mean of each of the patches.
            The patches are a column through the time axis of the unrotated
            science images. Together with the mean this provides an empirical
            estimate of the background statistics. An inv covariance matrix is provided
            for each pixel location in ph0s.
        m : numpy.ndarray
            Mean of each of the background patches along the time axis, for each pixel location
            in phi0s.
        patch : numpy.ndarray
            The background column for each test pixel location in phi0s.
        """
        if self.verbose:
            print("Precomputing Statistics...")

        # Store for each image pixel, for each temporal frame an image
        # for patches: for each time, we need to store a column of patches
        patch = np.zeros((self.width, self.height, self.num_frames, self.patch_area_pixels))

        # the mean of a temporal column of patches centered at each pixel
        m = np.zeros((self.height, self.width, self.patch_area_pixels))
        # the inverse covariance matrix at each point
        Cinv = np.zeros((self.height, self.width, self.patch_area_pixels, self.patch_area_pixels))

        # *** SERIAL ***
        # Loop over all pixels
        # i is the same as theta_k in the PACO paper
        for p0 in phi0s:
            apatch = self.get_patch(p0)
            # For some black magic reason this needs to be inverted here.
            m[p0[1]][p0[0]], Cinv[p0[1]][p0[0]] = compute_statistics_at_pixel(apatch)
            patch[p0[1]][p0[0]] = apatch
        return Cinv, m, patch

"""
**************************************************
*                                                *
*                  Fast PACO                     *
*                                                *
**************************************************
"""


class FastPACO(PACO):
    """
    This class implements Algorithm 2 from Flasseur+ 2018.
    """

    def PACOCalc(self,
                 phi0s : np.ndarray,
                 use_subpixel_psf_astrometry : Optional[bool] = True,
                 cpu : Optional[int] = 1) -> None:
        """
        PACOCalc

        This function iterates of a list of test points (phi0) and a list
        of angles between frames to produce 'a' and b', which can be used to
        generate a signal to noise map where SNR = b/sqrt(a) at each pixel.

        Parameters
        ----------
        phi0s : numpy.ndarray
            Array of (x,y) pixel locations to estimate companion position
        use_subpixel_psf_astrometry : bool
            If true, the PSF model for each patch is shifted to the correct
            location as predicted by the starting location and the parallactic
            angles, before being resampled for the patch. If false, the PSF
            model is simply located at the center of each patch. Significantly
            improves performance if set to False, but the SNR is reduced.
        cpu : int
            Number of cores to use for parallel processing

        Returns
        -------
        a : numpy.ndarray
            a_l from Equation 15 of Flasseur+ 2018
        b : numpy.ndarray
            b_l from Equation 16 of Flasseur+ 2018
        """
        npx = len(phi0s)  # Number of pixels in an image
        dim = self.width/2

        a = np.zeros(npx)  # Setup output arrays
        b = np.zeros(npx)
        phi0s = np.array([phi0s[:,1],phi0s[:,0]]).T

        if cpu == 1:
            Cinv, m, patches = self.compute_statistics(phi0s)
        else:
            Cinv, m, patches = self.compute_statistics_parallel(phi0s, cpu=cpu)
        normalised_psf = normalize_psf(self.psf,
                                       fwhm='fit',
                                       size=None,
                                       threshold=None,
                                       mask_core=None,
                                       model='airy',
                                       imlib='vip-fft',
                                       interpolation='lanczos4',
                                       force_odd=False,
                                       full_output=False,
                                       verbose=self.verbose,
                                       debug=False)
        psf_mask = create_boolean_circular_mask(normalised_psf.shape, radius=self.fwhm)

        # Create arrays needed for storage
        # Store for each image pixel, for each temporal frame an image
        # for patches: for each time, we need to store a column of patches

        # Currently forcing integer grid, but meshgrid takes floats as
        # arguments...
        x, y = np.meshgrid(np.arange(-dim, dim), np.arange(-dim, dim))
        if self.verbose:
            print("Running Fast PACO...")

        # Loop over all pixels
        # i is the same as theta_k in the PACO paper
        for i, p0 in enumerate(phi0s):
            # Get Angles
            angles_px = get_rotated_pixel_coords(x, y, p0, self.angles)
            # Ensure within image bounds
            if(int(np.max(angles_px.flatten())) >= self.width or
               int(np.min(angles_px.flatten())) < 0):
                a[i] = np.nan
                b[i] = np.nan
                continue

            # Extract relevant patches and statistics
            Cinlst = []
            mlst = []
            hlst = []
            patch = []
            for l, ang in enumerate(angles_px):
                Cinlst.append(Cinv[int(ang[0]), int(ang[1])])
                mlst.append(m[int(ang[0]), int(ang[1])])
                if use_subpixel_psf_astrometry:
                    offax = frame_shift(normalised_psf,
                                        ang[1]-int(ang[1]),
                                        ang[0]-int(ang[0]),
                                        imlib='vip-fft',
                                        interpolation='lanczos4',
                                        border_mode='reflect')[psf_mask]
                else:
                    offax = normalised_psf[psf_mask]
                hlst.append(offax)
                patch.append(patches[int(ang[0]), int(ang[1]), l])

            # Calculate a and b, matrices
            a[i] = self.al(hlst, Cinlst)
            b[i] = self.bl(hlst, Cinlst, patch, mlst)
        if self.verbose:
            print("Done")
        return a, b

    def compute_statistics_parallel(self,
                                    phi0s: np.ndarray,
                                    cpu: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        This function computes the mean and inverse covariance matrix for
        each patch in the image stack in parallel.

        Parameters
        ----------
        phi0s : int numpy.ndarray
            Array of pixel locations to estimate companion position
        cpu : int
            Number of processors to use

        Returns
        -------
        Cinv : numpy.ndarray
            Inverse covariance matrix between the the mean of each of the patches.
            The patches are a column through the time axis of the unrotated
            science images. Together with the mean this provides an empirical
            estimate of the background statistics. An inv covariance matrix is provided
            for each pixel location in ph0s.
        m : numpy.ndarray
            Mean of each of the background patches along the time axis, for each pixel location
            in phi0s.
        patch : numpy.ndarray
            The background column for each test pixel location in phi0s.

        """

        if self.verbose:
            print("Precomputing Statistics using %d Processes..."%cpu)
        # the mean of a temporal column of patches at each pixel
        m = np.zeros((self.height*self.width*self.patch_area_pixels))
        # the inverse covariance matrix at each point
        Cinv = np.zeros((self.height*
                         self.width*
                         self.patch_area_pixels*
                         self.patch_area_pixels))

        # *** Parallel Processing ***
        p_data = pool_map(cpu, self.get_patch, iterable(phi0s))
        patches = [p for p in p_data]
        data = pool_map(cpu, compute_statistics_at_pixel, iterable(patches))
        # p_pool = Pool(processes=cpu)
        #p_data = p_pool.map(self.get_patch, phi0s, chunksize=int(npx/cpu))
        # p_pool.close()
        # p_pool.join()
        # patches = [p for p in p_data]
        # p = Pool(processes=cpu)
        # data = p.map(compute_statistics_at_pixel, patches, chunksize=int(npx/cpu))
        # p.close()
        # p.join()
        ms, cs = [], []
        for d in data:
            if d[0] is None or d[1] is None:
                ms.append(np.full(self.patch_area_pixels, np.nan))
                cs.append(np.full((self.patch_area_pixels,
                                   self.patch_area_pixels), np.nan))
            else:
                ms.append(d[0])
                cs.append(d[1])
        ms = np.array(ms)
        cs = np.array(cs)
        patches = np.array(patches)

        # Reshape outputs
        patches = patches.reshape((self.width,
                                   self.height,
                                   self.num_frames,
                                   self.patch_area_pixels))
        m = ms.reshape((self.height,
                        self.width,
                        self.patch_area_pixels))
        Cinv = cs.reshape((self.height,
                           self.width,
                           self.patch_area_pixels,
                           self.patch_area_pixels))
        patches = np.swapaxes(patches,0,1)
        m = np.swapaxes(m,0,1)
        Cinv = np.swapaxes(Cinv,0,1)

        return Cinv, m, patches


"""
**************************************************
*                                                *
*                  Full PACO                     *
*                                                *
**************************************************
"""


class FullPACO(PACO):
    """
    Implementation of Algorithm 1 from Flasseur+ 2018
    """

    def PACOCalc(self,
                 phi0s : np.ndarray,
                 use_subpixel_psf_astrometry : Optional[bool] = True,
                 cpu : Optional[int] = 1) -> None:
        """
        PACOCalc

        This function iterates of a list of test points (phi0) and a list
        of angles between frames to produce 'a' and b', which can be used to
        generate a signal to noise map where SNR = b/sqrt(a) at each pixel.

        Parameters
        ----------
        phi0s : numpy.ndarray
            Array of (x,y) pixel locations to estimate companion position
        use_subpixel_psf_astrometry : bool
            If true, the PSF model for each patch is shifted to the correct
            location as predicted by the starting location and the parallactic
            angles, before being resampled for the patch. If false, the PSF
            model is simply located at the center of each patch. Significantly
            improves performance if set to False, but the SNR is reduced.
        cpu : int
            Number of cores to use for parallel processing. TODO: Not yet implemented.

        Returns
        -------
        a : numpy.ndarray
            a_l from Equation 15 of Flasseur+ 2018
        b : numpy.ndarray
            b_l from Equation 16 of Flasseur+ 2018
        """

        npx = len(phi0s)  # Number of pixels in an image
        dim = self.width/2

        a = np.zeros(npx)  # Setup output arrays
        b = np.zeros(npx)
        normalised_psf = normalize_psf(self.psf,
                                       fwhm='fit',
                                       size=None,
                                       threshold=None,
                                       mask_core=None,
                                       model='airy',
                                       imlib='vip-fft',
                                       interpolation='lanczos4',
                                       force_odd=False,
                                       full_output=False,
                                       verbose=self.verbose,
                                       debug=False)
        psf_mask = create_boolean_circular_mask(normalised_psf.shape, radius=self.fwhm)

        if self.verbose:
            print("Running Full PACO...")

        # Set up coordinates so 0 is at the center of the image
        x, y = np.meshgrid(np.arange(-dim, dim), np.arange(-dim, dim))

        if cpu > 1:
            print("Multiprocessing for full PACO is not yet implemented!")

        # Store intermediate results
        patch = np.zeros((self.width, self.height, self.num_frames, self.patch_area_pixels))
        # the mean of a temporal column of patches centered at each pixel
        m = np.zeros((self.height, self.width, self.patch_area_pixels))
        # the inverse covariance matrix at each point
        Cinv = np.zeros((self.height, self.width, self.patch_area_pixels, self.patch_area_pixels))

        # Loop over all pixels
        # i is the same as theta_k in the PACO paper
        for i, p0 in enumerate(phi0s):
            # Get list of pixels for each rotation angle
            angles_px = get_rotated_pixel_coords(x, y, (p0[1],p0[0]), self.angles)

            # Ensure within image bounds
            if(int(np.max(angles_px.flatten())) >= self.width or
               int(np.min(angles_px.flatten())) < 0):
                a[i] = np.nan
                b[i] = np.nan
                continue

            # Iterate over each temporal frame/each angle
            # Same as iterating over phi_l
            current_patch = []
            mlst = []
            h = []
            clst = []
            for l, ang in enumerate(angles_px):
                # Get the column of patches at this point
                if np.max(patch[int(ang[0]),int(ang[1])]) == 0:
                    apatch = self.get_patch((int(ang[1]),int(ang[0])))
                    patch[int(ang[0]),int(ang[1])] = apatch
                    m[int(ang[0]),int(ang[1])], Cinv[int(ang[0]),int(ang[1])] = compute_statistics_at_pixel(apatch)
                else:
                    apatch = patch[int(ang[0]),int(ang[1])]
                if apatch is None:
                    continue
                mlst.append(m[int(ang[0]),int(ang[1])])
                clst.append(Cinv[int(ang[0]),int(ang[1])])

                current_patch.append(apatch)
                if use_subpixel_psf_astrometry:
                    offax = frame_shift(normalised_psf,
                                        ang[1]-int(ang[1]),
                                        ang[0]-int(ang[0]),
                                        imlib='vip-fft',
                                        interpolation='lanczos4',
                                        border_mode='reflect')[psf_mask]
                else:
                    offax = normalised_psf[psf_mask]

                h.append(offax)
            current_patch = np.array(current_patch)
            patches = np.array([current_patch[l,l] for l in range(len(angles_px))])
            h = np.array(h)
            mlst = np.array(mlst)
            clst = np.array(clst)
            # Calculate a and b, matrices
            a[i] = self.al(h, clst)
            b[i] = self.bl(h, clst, patches, mlst)
        if self.verbose:
            print("Done")
        return a, b


"""
Math functions for computing patch covariance
"""


def compute_statistics_at_pixel(
        patch: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the mean and inverse covariance within a patch
    Reimplemented in PACO class, can probably be deleted

    Parameters
    ----------
    patch : numpy.ndarray
        Array of circular (flattened) patches centered on the same physical
        pixel vertically throughout the image stack
    """

    if patch is None:
        return None, None
    T = patch.shape[0]
    #size = patch.shape[1]

    # Calculate the mean of the column
    m = np.mean(patch, axis=0)
    # Calculate the covariance matrix
    S = sample_covariance(patch, m, T)
    rho = shrinkage_factor(S, T)
    F = diagsample_covariance(S)
    C = covariance(rho, S, F)
    Cinv = np.linalg.inv(C)
    return m, Cinv


def covariance(rho: np.ndarray, S: np.ndarray, F: np.ndarray) -> np.ndarray:
    """
    Ĉ: Shrinkage covariance matrix

    Parameters
    ----------
    rho : float
        Shrinkage factor weight
    S : numpy.ndarray
        Sample covariance matrix
    F : numpy.ndarray
        Diagonal of sample covariance matrix

    Returns
    -------
    m : numpy.ndarray
        Mean of each of the background patches along the time axis.
    Cinv : numpy.ndarray
        Inverse covariance matrix between the the mean of each of the patches.
        The patches are a column through the time axis of the unrotated
        science images. Together with the mean this provides an empirical
        estimate of the background statistics.
    """

    C = (1.0-rho)*S + rho*F
    return C


def sample_covariance(r: np.ndarray, m: np.ndarray,
                      T: np.ndarray) -> np.ndarray:
    """
    Ŝ: Sample covariance matrix

    Parameters
    ----------
    r : numpy.ndarray
        Observed intensity at position θk and time tl
    m : numpy.ndarray
        Mean of all background patches at position θk
    T : int
        Number of temporal frames

    Returns
    -------
    S : numpy.ndarray
        Sample covariance
    """

    #S = (1.0/T)*np.sum([np.outer((p-m).ravel(),(p-m).ravel().T) for p in r], axis=0)
    S = (1.0/T)*np.sum([np.cov(np.stack((p, m)),
                               rowvar=False, bias=False) for p in r], axis=0)
    return S


def diagsample_covariance(S: np.ndarray) -> np.ndarray:
    """
    F: Diagonal elements of the sample covariance matrix

    Parameters
    ----------
    S : arr
        Sample covariance matrix

    Returns
    -------
    F : numpy.ndarray
        Diagonal elements of the sample covariance matrix
    """

    return np.diag(np.diag(S))


def shrinkage_factor(S: np.ndarray, T: np.ndarray) -> float:
    """
    ρ: Shrinkage factor to regularize covariant matrix

    Parameters
    ----------
    S : numpy.ndarray
        Sample covariance matrix
    T : int
        Number of temporal frames

    Returns
    -------
    ρ : float
        Shrinkage factor to regularize covariant matrix
    """

    top = (np.trace(np.dot(S, S)) + np.trace(S)**2 -
           2.0*np.sum(S**2.0))
    bot = ((T+1.0)*(np.trace(np.dot(S, S)) -
                    np.sum(np.diag(S)**2.0)))
    p = top/bot
    return max(min(p, 1.0), 0.0)


def get_rotated_pixel_coords(x: np.ndarray,
                             y: np.ndarray,
                             p0: Tuple[int, int],
                             angles: np.ndarray,
                             astro_convention: Optional[bool] = False) -> np.ndarray:
    """
    For a given pixel, find the new pixel location after a rotation for each angle in angles

    Parameters
    ----------
    x : numpy.ndarrayr
        Grid of x components of pixel coordinates
    y : numpy.ndarrayr
        Grid of y components of pixel coordinates
    p0 : (int,int)
        Initial pixel location
    angles : numpy.ndarrayr
        List of angles for which to compute the new pixel location
    Returns
    -------
    nx : numpy.ndarray
        New array of x pixels coordinates following the rotation
    ny : numpy.ndarray
        New array of y pixels coordinates following the rotation

    """
    # Current pixel
    phi0 = np.array([x[int(p0[0]), int(p0[1])], y[int(p0[0]), int(p0[1])]])

    # Convert to polar coordinates
    rad, theta = cart_to_pol(
        phi0[0], phi0[1], astro_convention=astro_convention)

    # Rotate by parallactic angles
    angles_rad = -1*angles + theta

    # Rotate the polar coordinates by each frame angle
    angles_pol = np.array([rad*np.ones_like(angles_rad), angles_rad])

    # Find the new pixel coordinates after rotation
    nx, ny = pol_to_cart(
        angles_pol[0], angles_pol[1], astro_convention=astro_convention)

    # Shift to center coordinates (central pixel is 0)
    # TODO - use vip cx, cy arguments rather than shifting after?
    nx += +int(x.shape[0]/2)
    ny += +int(x.shape[0]/2)
    return np.array([nx, ny]).T


def create_boolean_circular_mask(shape: np.ndarray,
                                 radius: Optional[int] = 4,
                                 center: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """
    Returns a 2D boolean mask given some radius and location

    Parameters
    ----------
    shape : numpy.ndarray
        Shape of a 2D numpy array
    radius : int
        Radius of the mask in pixels
    center : (int,int)
        Pixel coordinates denoting the center of the mask,
        None defaults to center of shape

    Returns
    -------
    mask : numpy.ndarray
        A boolean mask of the the same shape as the science
        input data (provided by shape argument). The mask is 0
        outside of a circular region located at center, with a specified radius.
    """

    w = shape[0]
    h = shape[1]
    if center is None:
        center = [int(w/2), int(h/2)]
    if radius is None:
        radius = min(center[0], center[1], w-center[0], h-center[1])
    X, Y = np.ogrid[:w, :h]
    dist2 = (X - center[0])**2 + (Y-center[1])**2
    mask = dist2 <= radius**2
    return mask
