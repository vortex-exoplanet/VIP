vip.negfc package
=================

Subpackge ``negfc`` contains an ensemble of algorithms for planet position and
flux estimation. The main method is based on the negative fake companion (NegFC)
technique coupled with Markov chain Monte Carlo (MCMC).

The main idea of the NegFC is to inject negative fake companions (candidates)
with varying position and flux in order to minimize a function of merit. This
function of merit is defined as:

$chi^2 = sum(|I_j|)$,

where $j \in {1,...,N}$ and $N$ the total number of pixels contained in a
circular aperture (4xfwhm) centered on the injection position. This $chi^2$ is
measured on the PCA-processed frame. Thanks to the use of ``emcee`` and its
Affine Invariant MCMC we can get the posterior distributions of the positions
and fluxes allowing us to define proper error-bars for the photometry and
position of the companion.

Submodules
----------

vip.negfc.func_merit module
---------------------------

.. automodule:: vip.negfc.func_merit
    :members:
    :undoc-members:
    :show-inheritance:

vip.negfc.mcmc_opt module
-------------------------

.. automodule:: vip.negfc.mcmc_opt
    :members:
    :undoc-members:
    :show-inheritance:

vip.negfc.simplex_opt module
----------------------------

.. automodule:: vip.negfc.simplex_opt
    :members:
    :undoc-members:
    :show-inheritance:

vip.negfc.subt_planet module
----------------------------

.. automodule:: vip.negfc.subt_planet
    :members:
    :undoc-members:
    :show-inheritance:

vip.negfc.utils module
----------------------

.. automodule:: vip.negfc.utils
    :members:
    :undoc-members:
    :show-inheritance:


.. automodule:: vip.negfc
    :members:
    :undoc-members:
    :show-inheritance:
