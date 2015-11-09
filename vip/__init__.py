import os

from . import calib
from . import conf
from . import fits
from . import madi
from . import negfc
from . import pca
from . import phot
from . import stats
from . import var

PACKAGE_PATH = os.path.abspath(os.path.join(__file__, os.pardir))[:-4]

with open(os.path.join(PACKAGE_PATH, 'VERSION')) as version_file:
    __version__ = version_file.read().strip()

print "---------------------------------------------------"
print "         oooooo     oooo ooooo ooooooooo.          "
print "          `888.     .8'  `888' `888   `Y88.        "
print "           `888.   .8'    888   888   .d88'        "
print "            `888. .8'     888   888ooo88P'         "
print "             `888.8'      888   888                "
print "              `888'       888   888                "
print "               `8'       o888o o888o               "
print "---------------------------------------------------"  
print "     Vortex Image Processing pipeline v"+__version__  
print "---------------------------------------------------" 
print "Please cite Gomez Gonzalez et al. 2016 (in prep.)  "
print "whenever you publish data reduced with VIP. Thanks."
    



