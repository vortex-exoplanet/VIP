.. This file should at least contain the root `toctree` directive.

.. Welcome to ``VIP``'s documentation
.. ==============================

.. image:: _static/logo.jpg
   :align: center
   :width: 400px

.. include:: ../../README.rst


.. toctree::
   :maxdepth: 1
   :caption: Tutorials

   tutorials/01_quickstart.ipynb
   tutorials/02_preproc.ipynb
   tutorials/03_psfsub.ipynb
   tutorials/04_metrics.ipynb
   tutorials/05_fm_planets.ipynb
   tutorials/06_fm_disk.ipynb
   tutorials/07_imlib_and_interpolation.ipynb
   tutorials/08_datasets_as_objects.ipynb

Frequently asked questions
--------------------------
Check out this questions if you find problems when installing or running ``VIP``
for the first time.

.. toctree::
   :maxdepth: 2

   faq


Package structure
-----------------
On the links below you can find the subpackages structure and access the
docstrings (internal documentation) of each one of ``VIP``'s functions.

``VIP`` implements basic image processing functionalities such as image
registration, rotation, shift, rescaling, pixel temporal and spatial
subsampling. On top of that several pre-processing functions are available
such as re-centering and bad frame removal.

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

.. toctree::
   :maxdepth: 3

   vip_hci


API
---

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

