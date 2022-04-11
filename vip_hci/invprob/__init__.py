"""
Subpackage ``invprob`` aims to contain post-processing algorithms based on an 
inverse problem approach, such as ANDROMEDA (Mugnier et al. 2009; Cantalloube 
et al. 2015) or Foward Model Matched Filter (Ruffio et al. 2019; Dahlqvist et
 al. 2021)
"""

from .andromeda import *
from .fmmf import *