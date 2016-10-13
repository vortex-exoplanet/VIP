"""
Ensemble of methods required to determine the position and flux of a planet
in the framework of VIP. The main method consists in the negative fake 
companion method coupled with MCMC (based on emcee). 

"""

from simplex_fmerit import *
from mcmc_sampling import *
from nested_sampling import *
from simplex_optim import *
from utils_negfc import *