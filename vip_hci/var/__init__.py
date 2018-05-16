"""
Subpackage ``var`` has helping functions such as:
    - image filtering,
    - shapes extraction (annulus, squares subimages, circular apertures),
    - plotting,
    - 2d fitting (Gaussian, Moffat).
"""
from __future__ import absolute_import

from .filters import *
from .fit_2d import *
from .shapes import *
from .plotting import *
