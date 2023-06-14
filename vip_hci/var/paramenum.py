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
    * ``SINGLE``: PCA only. The multi-spectral frames are rescaled wrt the largest
    wavelength to align the speckles and all the frames (n_channels *
    n_adiframes) are processed with a single PCA low-rank approximation.

    * ``DOUBLE``: PCA and LOCI. A first stage is run on the rescaled spectral frames,
    and a second frame is run on the residuals in an ADI fashion.

    * ``SKIPADI``: LOCI only. The multi-spectral frames are rescaled wrt the largest
    wavelength to align the speckles and the least-squares model is
    subtracted on each spectral cube separately.
    """

    DOUBLE = auto()
    SINGLE = auto()
    SKIPADI = auto()


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


# TODO: document modes
class LowRankMode(LowEnum):
    """Define the values for the low rank mode for LLSG."""

    SVD = auto()
    BRP = auto()


# TODO: document modes
class AutoRankMode(LowEnum):
    """Define the values for the auto rank mode for LLSG."""

    NOISE = auto()
    CEVR = auto()


# TODO: document modes
class ThreshMode(LowEnum):
    """Define the values for thresholding modes for LLSG."""

    SOFT = auto()
    HARD = auto()


class Solver(LowEnum):
    """
    Define the solver for the least squares problem in LLSG.

    Modes
    -----
    * ``LSTSQ`` : uses the standard scipy least squares solver.

    * ``NNLS`` : uses the scipy non-negative least-squares solver.
    """

    LSTSQ = auto()
    NNLS = auto()


class Runmode(LowEnum):
    """
    Define the mode for the PostProc PCA object.

    Modes
    -----
    * ``CLASSIC`` : base PCA function, with multiple usages depending on the
    parameters given.

    * ``ANNULAR`` : annular PCA function.

    * ``GRID`` : grid PCA function (can be used implicitely from "classic").

    * ``ANNULUS`` : annulus PCA function.
    """

    CLASSIC = auto()
    ANNULAR = auto()
    GRID = auto()
    ANNULUS = auto()


class HandleNeg(LowEnum):
    """
    Define modes for handling negative values in NMF full-frame.

    Modes
    -----
    * ``SUBTR_MIN`` : subtract the minimum value in the arrays.

    * ``MASK`` : mask negative values.

    * ``NULL`` : set negative values to zero.
    """

    SUBTR_MIN = auto()
    MASK = auto()
    NULL = auto()


class Initsvd(LowEnum):
    """
    Define modes for initializing SVD for NMF full-frame.

    Modes
    -----
    * ``NNDSVD``: non-negative double SVD recommended for sparseness.

    * ``NNDSVDA`` : NNDSVD where zeros are filled with the average of cube;
        recommended when sparsity is not desired.

    * ``RANDOM`` : random initial non-negative matrix.
    """

    NNDSVD = auto()
    NNDSVDA = auto()
    RANDOM = auto()


# TODO: document modes
class OptMethod(LowEnum):
    """
    Defines the method of balancing for the flux difference for ANDROMEDA.

    Modes
    -----

    * ``NO``

    * ``TOTAL``

    * ``LSQ``

    * ``ROBUST``
    """

    NO = auto()
    TOTAL = auto()
    LSQ = auto()
    ROBUST = auto()


class VarEstim(LowEnum):
    """
    Define modes to use for the residual noise variance estimation in FMMF.

    Modes
    -----

    * ``FR``: consider the pixels in the selected annulus with a width equal
    to asize but separately for every frame.

    * ``FM``: consider the pixels in the selected annulus with a width
    equal to asize but separately for every frame. Apply a mask one FWHM
    on the selected pixel and its surrounding.

    * ``TE``:rely on the method developped in PACO to estimate the
    residual noise variance (take the pixels in a region of one FWHM
    arround the selected pixel, considering every frame in the
    derotated cube of residuals except for the selected frame).
    """

    FR = auto()
    FM = auto()
    TE = auto()
