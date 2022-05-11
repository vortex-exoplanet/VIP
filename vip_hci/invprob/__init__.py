"""
Subpackage ``invprob`` aims to contain post-processing algorithms based on an
inverse problem approach, such as ANDROMEDA (Mugnier et al. 2009; Cantalloube
et al. 2015), Foward Model Matched Filter (Ruffio et al. 2019; Dahlqvist et
al. 2021) or PACO (Flasseur et al 2018).
"""

from .andromeda import *
from .fmmf import *
from .paco import *
