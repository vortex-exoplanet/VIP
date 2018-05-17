"""
Subpackage ``metrics`` includes:
    - signal-to-noise (S/N) estimation,
    - S/N map generation,
    - detection of point like sources (for pipelines),
    - fake companions injection,
    - algorithms throughput,
    - contrast curve generation,
    - receiver operating characteristic (ROC) curves.
"""
from __future__ import absolute_import

from .contrcurve import *
from .detection import *
from .fakecomp import *
from .frame_analysis import *
from .roc import *
from .snr import *
