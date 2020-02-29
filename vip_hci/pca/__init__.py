"""
Subpackage ``pca`` contains the PCA based algorithms for model PSF subtraction.
The ADI version is inspired by Soumer et al. 2012 and Amara et al. 2012.
Different flavors of PCA (with improvements and speed tricks) have been
implemented:

- *Full-frame PCA*, using the whole cube as the PCA reference library in the
  case of ADI or ADI+mSDI (ISF cube), or a sequence of reference frames
  (reference star) in the case of RDI. For ADI a big data matrix NxP, where N
  is the number of frames and P the number of pixels in a frame is created. Then
  PCA is done through eigen-decomposition of the covariance matrix (~$DD^T$) or
  the SVD of the data matrix. SVD can be calculated using different libraries
  including the fast randomized SVD (Halko et al. 2009).

- *Full-frame incremental PCA* for big (larger than available memory) cubes.

- *Annular PCA* performs a local PCA (annulus-wise or in segments of annulus)
  taking into account a parallactic angle rejection for allowing FOV rotation
  and avoid planet self-subtraction. These local PCA algorithms process many
  (number of patches times number of frames) smaller matrices increasing the
  computation time. This implementation uses the Python multiprocessing
  capabilities.
"""

from .pca_fullfr import *
from .pca_local import *
from .svd import *
from .utils_pca import *
