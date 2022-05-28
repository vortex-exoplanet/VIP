"""
Subpackage ``itpca`` contains the iterative PCA based algorithms for model PSF
subtraction, in an iterative fashion to minim
This is inspired by Pairet et al. (2019), and also described in Christiaens et
al. (2021b).
The different flavors of PCA available in the subpackage 'pca' have been
implemented in an iterative fashion.

Note: the routines of this subpackage were not added to the 'pca' subpackage to
avoid circular imports. This is because the iterative pca algorithms call
routines from the 'metrics' subpackage, which in turn requires the svd routines
defined in the 'pca' subpackage.
"""

from .opt_itpca import *
from .pca_fullfr import *
from .pca_local import *
from .utils_itpca import *
