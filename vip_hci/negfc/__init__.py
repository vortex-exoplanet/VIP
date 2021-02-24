r"""
Subpackge ``negfc`` contains an ensemble of algorithms for planet position and
flux estimation. It works with as an optimization procedure using a Simplex
approach or coupled with Monte Carlo methods for posterior sampling. Thanks to
this we can get the posterior distributions of the positions and fluxes allowing
us to define proper error-bars for the photometry and position of the companion.
2 possible ways of sampling the posteriors are possible: using ``emcee`` and its
Affine Invariant MCMC or ``nestle`` with a single ellipsoid nested sampling
procedure (much faster).

The main idea of the NegFC is to inject negative fake companions (candidates)
with varying position and flux in order to minimize (in the case of the Simplex)
a function of merit. This function of merit is defined as:

$chi^2 = sum(\|I_j\|)$,

where $j \in {1,...,N}$ and $N$ the total number of pixels contained in a
circular aperture (4xfwhm) centered on the injection position. This $chi^2$ is
measured on the PCA-processed frame or cube of residuals.

"""


from .simplex_fmerit import *
from .mcmc_sampling import *
from .nested_sampling import *
from .simplex_optim import *
from .speckle_noise import *
from .utils_mcmc import *
from .utils_negfc import *