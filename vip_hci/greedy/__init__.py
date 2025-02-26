"""
Subpackage ``greedy`` contains iterative implementations of stellar PSF
modelling + subtraction algorithms. The following methods have been implemented:
- iterative roll subtraction [HEA00]_
- iterative PCA in full frame [PAI18]_ / [PAI21]_
- iterative NMF in full frame (cite latest VIP paper if used).
"""
from .inmf_fullfr import *
from .ipca_fullfr import *
from .irollsub import *
