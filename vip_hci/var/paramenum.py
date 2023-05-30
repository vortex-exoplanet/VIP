"""Module containing enums for parameters of HCI algorithms."""
from enum import auto

from strenum import KebabCaseStrEnum as KebEnum
from strenum import LowercaseStrEnum as LowEnum


class SvdMode(LowEnum):
    """
    Define the various modes to use with SVD in PCA as constant strings.

    Modes
    -----
    * ``LAPACK``: uses the LAPACK linear algebra library through Numpy
    and it is the most conventional way of computing the SVD
    (deterministic result computed on CPU).

    * ``ARPACK``: uses the ARPACK Fortran libraries accessible through
    Scipy (computation on CPU).

    * ``EIGEN``: computes the singular vectors through the
    eigendecomposition of the covariance M.M' (computation on CPU).

    * ``RANDSVD``: uses the randomized_svd algorithm implemented in
    Sklearn (computation on CPU), proposed in [HAL09]_.

    * ``CUPY``: uses the Cupy library for GPU computation of the SVD as in
    the LAPACK version. `

    * ``EIGENCUPY``: offers the same method as with the ``eigen`` option
    but on GPU (through Cupy).

    * ``RANDCUPY``: is an adaptation of the randomized_svd algorithm,
    where all the computations are done on a GPU (through Cupy). `

    * ``PYTORCH``: uses the Pytorch library for GPU computation of the SVD.

    * ``EIGENPYTORCH``: offers the same method as with the ``eigen``
    option but on GPU (through Pytorch).

    * ``RANDPYTORCH``: is an adaptation of the randomized_svd algorithm,
    where all the linear algebra computations are done on a GPU
    (through Pytorch).

    """

    LAPACK = auto()
    ARPACK = auto()
    EIGEN = auto()
    RANDSVD = auto()
    CUPY = auto()
    EIGENCUPY = auto()
    RANDCUPY = auto()
    PYTORCH = auto()
    EIGENPYTORCH = auto()
    RANDPYTORCH = auto()


class Scaling(LowEnum):
    """
    Define modes for the pixel-wise scaling.

    Modes
    -----
    * ``TEMPMEAN``: temporal px-wise mean is subtracted.

    * ``SPATMEAN``: spatial mean is subtracted.

    * ``TEMPSTANDARD``: temporal mean centering plus scaling pixel values
    to unit variance. HIGHLY RECOMMENDED FOR ASDI AND RDI CASES!

    * ``SPATSTANDARD``: spatial mean centering plus scaling pixel values
    to unit variance.
    """

    TEMPMEAN = "temp-mean"
    SPATMEAN = "spat-mean"
    TEMPSTANDARD = "temp-standard"
    SPATSTANDARD = "spat-standard"


class Adimsdi(LowEnum):
    """
    Define modes for processing ADI+mSDI cubes through PCA.

    Modes
    -----
    * ``SINGLE``: the multi-spectral frames are rescaled wrt the largest
    wavelength to align the speckles and all the frames (n_channels *
    n_adiframes) are processed with a single PCA low-rank approximation.

    * ``DOUBLE``: a first stage is run on the rescaled spectral frames, and
    a second PCA frame is run on the residuals in an ADI fashion.
    """

    DOUBLE = auto()
    SINGLE = auto()


# TODO: document all modes
class Imlib(LowEnum):
    """
    Define modes for image transformations to be used.

    Modes
    -----
    * ̀``OPENCV``: uses OpenCV. Faster than Skimage or scipy.ndimage.

    * ̀``SKIMAGE``: uses Skimage.

    * ``NDIMAGE``: uses scipy.ndimage.

    * ̀``VIPFFT``: uses VIP FFT based rotation method.
    """

    OPENCV = auto()
    SKIMAGE = auto()
    NDIMAGE = auto()
    VIPFFT = "vip-fft"


# TODO: document all modes
class Interpolation(LowEnum):
    """
    Define modes for interpolation.

    Modes
    -----
    * ̀``NEARNEIG``

    * ̀``BILINEAR``

    * ̀``BIQUADRATIC`` : Default for Skimage (only).

    * ̀``BICUBIC``

    * ̀``BIQUARTIC`` : Skimage only.

    * ̀``BIQUINTIC`` : slowest and most accurate. Skimage only.

    * ̀``LANCZOS4`` : slowest and most accurate. Default for OpenCV.
    """

    NEARNEIG = auto()
    BILINEAR = auto()
    BIQUADRATIC = auto()
    BICUBIC = auto()
    BIQUARTIC = auto()
    BIQUINTIC = auto()
    LANCZOS4 = auto()


# TODO: document all modes
class Collapse(LowEnum):
    """
    Define modes for spectral/temporal residuals frames combining.

    Modes
    -----
    * ̀``MEDIAN``

    * ̀``MEAN``

    * ̀``SUM``

    * ̀``TRIMMEAN``
    """

    MEDIAN = auto()
    MEAN = auto()
    SUM = auto()
    TRIMMEAN = auto()


class ReturnList(LowEnum):
    """List of all possible modes of classic PCA."""

    ADIMSDI_DOUBLE = auto()
    ADIMSDI_SINGLE_NO_GRID = auto()
    ADIMSDI_SINGLE_GRID_NO_SOURCE = auto()
    ADIMSDI_SINGLE_GRID_SOURCE = auto()
    ADI_FULLFRAME_GRID = auto()
    ADI_FULLFRAME_STANDARD = auto()
    ADI_INCREMENTAL_BATCH = auto()
    PCA_GRID_SN = auto()
    PCA_ROT_THRESH = auto()


# TODO: document all metrics
class Metric(LowEnum):
    """Define all metrics possible for various post-processing functions."""

    CITYBLOCK = auto()
    COSINE = auto()
    EUCLIDEAN = auto()
    L1 = auto()
    L2 = auto()
    MANHATTAN = auto()
    CORRELATION = auto()
