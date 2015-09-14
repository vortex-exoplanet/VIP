"""
Ensemble of methods required to determine the position and flux of a planet
in the framework of VIP. The main method consists in the negative fake 
companion method coupled with MCMC (based on emcee). 
 
Authors: O Wertz, O. Absil, C. Gomez
 
"""

from func_merit import *
from mcmc_opt import *
from simplex_opt import *
from subt_planet import *
from utils import *