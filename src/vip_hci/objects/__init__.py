"""
Subpackage ``objects`` containing a large number object-oriented modules.

They are implementing PSF subtraction and inverse problem approach algorithms found in
``vip_hci.psfsub`` and ``vip_hci.invprob``. The modules included are the following :
- *median subtraction*.
- *frame differencing*.
- *LOCI*.
- *PCA*.
- full-frame and annular versions of *NMF*.
- *LLSG*.
- *ANDROMEDA*.
- *FMMF*.
"""
import sys

from .dataset import *
from .postproc import *
from .ppandromeda import *
from .ppfmmf import *
from .ppframediff import *
from .ppllsg import *
from .pploci import *
from .ppmediansub import *
from .ppnmf import *

if sys.version_info >= (3, 10):
    from .pppca import *
