"""
Subpackage ``specfit`` has helping functions for the analysis of (low-res) 
spectra, including:

- fitting of input spectra to models and templates;
- mcmc sampling of model parameter space;
- best fit search within a template library;
- utility functions for the spectral fit.
"""

from .chi import *
from .model_resampling import *
from .mcmc_sampling_spec import *
from .template_lib import *
from .utils_spec import *