"""
Subpackage ``var`` has helping functions such as:

- image filtering,
- shapes extraction (annulus, squares subimages, circular apertures),
- 2d fitting (Gaussian, Moffat),
- wavelet transform definition,
- Q/U to Qphi/Uphi image conversion (for polarimetric data).
"""

from .coords import *
from .shapes import *
from .fit_2d import *
from .iuwt import *
from .filters import *