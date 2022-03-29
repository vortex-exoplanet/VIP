r"""
Subpackge ``fm`` contains an ensemble of algorithms for forward modeling, 
including scattered light disk model creation, fake planet and fake disk 
injection, and planet position+flux estimation through the negative fake 
companion algorithm (NEGFC). NEGFC can work with either a simplex (Nelder-Mead) 
minimizer or coupled with Monte Carlo methods for posterior sampling. The 
latter allows the estimation of uncertainties for the photometry and position 
of the companion. Two possible ways of sampling the posteriors are implemented: 
using ``emcee`` and its Affine Invariant MCMC or ``nestle`` with either a 
single or multiple ellipsoid nested sampling procedure.

The main idea of the NegFC is to inject negative fake companions (candidates)
with varying position and flux in the original cube and minimize a figure of 
merit corresponding to the residuals in an aperture at the initial estimate for 
the location of the companion, in the post-processed image.

"""

from .scattered_light_disk import *
from .fakecomp import *
from .fakedisk import *
from .negfc_fmerit import *
from .negfc_mcmc import *
from .negfc_nested import *
from .negfc_simplex import *
from .negfc_speckle_noise import *
from .utils_mcmc import *
from .utils_negfc import *