"""
The subpackage ``preproc`` contains some useful cosmetics and pre-processing
functionalities:

- resizing frames and cubes : upscaling/pixel binning,
- shifting frames,
- rotating frames and cubes,
- cropping frames and cubes,
- removing bad pixels from frames and cubes,
- correcting nan values from frames and cubes,
- detecting bad frames in cubes, using:
    - pixel statistics in annulus or circular aperture,
    - ellipticity of a point like source,
    - frames correlation,
- temporal subsampling of cubes (mean, median, trimmed mean),
- registration (re-centering) of frames, using:
    - centroid fitting a 2d gaussian or moffat,
    - DFT upsampling or fourier cross-correlation (Guizar et al. 2008),
    - radon transform for broadband frames (Pueyo et al. 2014),
    - using satellite/waffle spots (fitting plus intersection).
- sky subtraction (PCA method).

Astronomical calibration functionalities like flat-fielding and dark-sky
subtraction, in spite of their simplicity were not included in VIP because of 
the heterogeneity of the datasets coming from different observatories (each 
having different data storage and headers). You can perform this in python in
procedures of a few lines or using dedicated instrument pipelines such as
esorex (ESO instruments)."""


from .badframes import *
from .badpixremoval import *
from .cosmetics import *
from .derotation import *
from .parangles import *
from .recentering import *
from .rescaling import *
from .skysubtraction import *
from .subsampling import *
