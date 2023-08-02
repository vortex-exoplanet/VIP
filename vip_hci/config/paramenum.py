"""Module containing enums for parameters of HCI algorithms and literal constants."""
from enum import Enum

ALGO_KEY = "algo_params"
ALL_FITS = -2


class SvdMode(str, Enum):
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

    LAPACK = "lapack"
    ARPACK = "arpack"
    EIGEN = "eigen"
    RANDSVD = "randsvd"
    CUPY = "cupy"
    EIGENCUPY = "eigencupy"
    RANDCUPY = "randcupy"
    PYTORCH = "pytorch"
    EIGENPYTORCH = "eigenpytorch"
    RANDPYTORCH = "randpytorch"


class Scaling(str, Enum):
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


class Adimsdi(str, Enum):
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

    DOUBLE = "double"
    SINGLE = "single"
    SKIPADI = "skipadi"


# TODO: document all modes
class Imlib(str, Enum):
    """
    Define modes for image transformations to be used.

    Modes
    -----
    * ̀``OPENCV``: uses OpenCV. Faster than Skimage or scipy.ndimage.

    * ̀``SKIMAGE``: uses Skimage.

    * ``NDIMAGE``: uses scipy.ndimage.

    * ̀``VIPFFT``: uses VIP FFT based rotation method.
    """

    OPENCV = "opencv"
    SKIMAGE = "skimage"
    NDIMAGE = "ndimage"
    VIPFFT = "vip-fft"


# TODO: document all modes
class Interpolation(str, Enum):
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

    NEARNEIG = "nearneig"
    BILINEAR = "bilinear"
    BIQUADRATIC = "biquadratic"
    BICUBIC = "bicubic"
    BIQUARTIC = "biquartic"
    BIQUINTIC = "biquintic"
    LANCZOS4 = "lanczos4"


# TODO: document all modes
class Collapse(str, Enum):
    """
    Define modes for spectral/temporal residuals frames combining.

    Modes
    -----
    * ̀``MEDIAN``

    * ̀``MEAN``

    * ̀``SUM``

    * ̀``TRIMMEAN``
    """

    MEDIAN = "median"
    MEAN = "mean"
    SUM = "sum"
    TRIMMEAN = "trimmean"


class ReturnList(str, Enum):
    """List of all possible modes of classic PCA."""

    ADIMSDI_DOUBLE = "adimsdi_double"
    ADIMSDI_SINGLE_NO_GRID = "adimsdi_single_no_grid"
    ADIMSDI_SINGLE_GRID_NO_SOURCE = "adimsdi_single_grid_no_source"
    ADIMSDI_SINGLE_GRID_SOURCE = "adimsdi_single_grid_source"
    ADI_FULLFRAME_GRID = "adi_fullframe_grid"
    ADI_FULLFRAME_STANDARD = "adi_fullframe_standard"
    ADI_INCREMENTAL_BATCH = "adi_incremental_batch"
    PCA_GRID_SN = "pca_grid_sn"
    PCA_ROT_THRESH = "pca_rot_thresh"


# TODO: document all metrics
class Metric(str, Enum):
    """Define all metrics possible for various post-processing functions."""

    CITYBLOCK = "cityblock"
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    L1 = "l1"
    L2 = "l2"
    MANHATTAN = "manhattan"
    CORRELATION = "correlation"


# TODO: document modes
class LowRankMode(str, Enum):
    """Define the values for the low rank mode for LLSG."""

    SVD = "svd"
    BRP = "brp"


# TODO: document modes
class AutoRankMode(str, Enum):
    """Define the values for the auto rank mode for LLSG."""

    NOISE = "noise"
    CEVR = "cevr"


# TODO: document modes
class ThreshMode(str, Enum):
    """Define the values for thresholding modes for LLSG."""

    SOFT = "soft"
    HARD = "hard"


class Solver(str, Enum):
    """
    Define the solver for the least squares problem in LLSG.

    Modes
    -----
    * ``LSTSQ`` : uses the standard scipy least squares solver.

    * ``NNLS`` : uses the scipy non-negative least-squares solver.
    """

    LSTSQ = "lstsq"
    NNLS = "nnls"


class Runmode(str, Enum):
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

    CLASSIC = "classic"
    ANNULAR = "annular"
    GRID = "grid"
    ANNULUS = "annulus"


class HandleNeg(str, Enum):
    """
    Define modes for handling negative values in NMF full-frame.

    Modes
    -----
    * ``SUBTR_MIN`` : subtract the minimum value in the arrays.

    * ``MASK`` : mask negative values.

    * ``NULL`` : set negative values to zero.
    """

    SUBTR_MIN = "subtr_min"
    MASK = "mask"
    NULL = "null"


class Initsvd(str, Enum):
    """
    Define modes for initializing SVD for NMF full-frame.

    Modes
    -----
    * ``NNDSVD``: non-negative double SVD recommended for sparseness.

    * ``NNDSVDA`` : NNDSVD where zeros are filled with the average of cube;
        recommended when sparsity is not desired.

    * ``RANDOM`` : random initial non-negative matrix.
    """

    NNDSVD = "nndsvd"
    NNDSVDA = "nndsvda"
    RANDOM = "random"


# TODO: document modes
class OptMethod(str, Enum):
    """
    Defines the method of balancing for the flux difference for ANDROMEDA.

    Modes
    -----

    * ``NO``

    * ``TOTAL``

    * ``LSQ``

    * ``ROBUST``
    """

    NO = "no"
    TOTAL = "total"
    LSQ = "lsq"
    ROBUST = "robust"


class VarEstim(str, Enum):
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

    FR = "fr"
    FM = "fm"
    TE = "te"
