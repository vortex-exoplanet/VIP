"""
In subpackage ``pca`` are the PCA based algorithms (Soumer et al. 2012 and Amara
et al. 2012) for cubes reference-PSF/background subtraction. PCA comes in
different flavors and with speed modifications:

- *Full-frame PCA*, using the whole cube as the PCA reference library in the
case of ADI or SDI (ISF cube), or a sequence of reference frames (reference
star) in the case of RDI. For ADI a big data matrix NxP, where N is the number
of frames and P the number of pixels in a frame is created. Then PCA is done
through eigen-decomposition of the covariance matrix (~$DD^T$) or the SVD of
the centered data matrix. SVD can be calculated using different libraries
including a very fast one (default option): randomized SVD (Halko et al. 2009).

- *Annular PCA*, and *subannular PCA* (quadrants of annulus) perform a local PCA
taking into account a parallactic angle rejection for allowing FOV rotation and
avoid planet self-subtraction. These local PCA algorithms process many (number
of patches times number of frames) smaller matrices increasing the computation
time.

- *parallel subannular PCA* which uses multiprocessing and allows to distribute
the computations amongst the available cores in your machine.
"""
from __future__ import absolute_import

from .pca_fullfr import *
from .pca_local import *
from .svd import *
from .utils_pca import *
