"""
Subpackage ``phot`` includes:
    - signal-to-noise (S/N) estimation,
    - S/N map generation,
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
