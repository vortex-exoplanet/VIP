Structure
---------
On the links below you can find the subpackages structure and access the
docstrings (internal documentation) of each one of ``VIP``'s functions.

``VIP`` implements basic image processing functionalities such as image
registration, rotation, shift, rescaling, pixel temporal and spatial
subsampling. On top of that several pre-processing functions are available
such as recentering and bad frame removal.

For Angular differential imaging (ADI) data several point spread function
subtraction techniques are available: pairwise frame differencing, median
subtraction, least-squares combination, NMF and PCA based algorithms. PCA
methods are implemented in different flavours. Also PCA can process RDI and
multiple channel SDI (IFS) data. ``VIP`` contains an implementation of the Local
Low-rank plus Sparse plus Gaussian-noise decomposition (LLSG, Gomez Gonzalez et
al. 2016).

Functions for signal-to-noise ratio (S/N) estimation and S/N map generation are
included, as well as injection of fake companions in 3D and 4D cubes. Flux and
position of point-like sources are estimated using the Negative Fake Companion
technique. ``VIP`` also implements algorithm throughput and contrast-curve
generation routines.