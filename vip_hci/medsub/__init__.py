"""
Subpackage ``medsub`` has the ADI cube basic processing (Marois et al. 2006): median
frame subtraction, and annular mode where ``n`` closest frames taking into account
a PA threshold are median collapsed and subtracted.
"""
from __future__ import absolute_import

from .medsub_source import *