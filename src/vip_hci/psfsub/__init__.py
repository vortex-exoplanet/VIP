"""
Subpackage ``psfsub`` contains a large number of stellar PSF modelling +
subtraction algorithms. The following methods have been implemented:
- *roll subtraction* [SCH98]_
- *median ADI/SDI* [MAR06]_ / [SPA02]_ / [THA07]_
- *frame differencing*
- a simplified version of *LOCI* [LAF07]_
- different flavours of *PCA* [AMA12]_ / [SOU12]_
working in full-frame, incremental and annular mode, including improvements,
speed tricks, compatibility with ADI/RDI/SDI datasets and datasets too large to
fit in memory.
- full-frame and annular versions of *NMF* [LEE99]_ / [GOM17]_.
"""
from .framediff import *
from .llsg import *
from .loci import *
from .medsub import *
from .nmf_fullfr import *
from .nmf_local import *
from .pca_fullfr import *
from .pca_local import *
from .rollsub import *
from .svd import *
from .utils_pca import *
