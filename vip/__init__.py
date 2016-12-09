from __future__ import (absolute_import)

from . import calib
from . import conf
from . import fits
from . import llsg
from . import madi
from . import negfc
from . import nmf
from . import pca
from . import phot
from . import stats
from . import var

__version__ = "0.6.0"

print("---------------------------------------------------")
print("         oooooo     oooo ooooo ooooooooo.          ")
print("          `888.     .8'  `888' `888   `Y88.        ")
print("           `888.   .8'    888   888   .d88'        ")
print("            `888. .8'     888   888ooo88P'         ")
print("             `888.8'      888   888                ")
print("              `888'       888   888                ")
print("               `8'       o888o o888o               ")
print("---------------------------------------------------")
print("     Vortex Image Processing pipeline v"+__version__)
print("---------------------------------------------------")
print("Please cite Gomez Gonzalez et al. 2016 (submitted) ")
print("whenever you publish data reduced with VIP. Thanks.")
