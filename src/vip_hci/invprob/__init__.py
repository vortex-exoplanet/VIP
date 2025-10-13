"""
Subpackage ``invprob`` aims to contain post-processing algorithms based on an
inverse problem approach, such as ANDROMEDA [MUG09]_ / [CAN15]_, Forward Model
Matched Filter [RUF17]_ / [DAH21a]_ or PACO [FLA18]_.
"""

from .andromeda import *
from .fmmf import *
from .paco import *
