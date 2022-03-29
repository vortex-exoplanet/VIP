"""
Subpackage ``metrics`` includes:
    - signal-to-noise (S/N) estimation,
    - S/N map generation,
    - STIM map generation,
    - detection of point like sources (for pipelines),
    - fake companions injection,
    - fake disks generation and injection,
    - algorithms throughput estimation,
    - contrast curve generation,
    - receiver operating characteristic (ROC) curves generation.
"""
from .contrcurve import *
from .detection import *
from .roc import *
from .snr_source import *
from .stim import *
