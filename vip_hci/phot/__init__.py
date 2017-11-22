"""
Subpackage ``phot`` includes functionalities such as:
    - signal-to-noise (S/R) estimation,
    - S/R map generation,
    - detection of point like sources (for pipelines),
    - fake companions injection,
    - algorithms throughput,
    - contrast curve generation.
"""
from __future__ import absolute_import

from .contrcurve import *
from .detection import *
from .fakecomp import *
from .frame_analysis import *
from .snr import *
