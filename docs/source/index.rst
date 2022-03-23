.. This file should at least contain the root `toctree` directive.

.. Welcome to ``VIP``'s documentation
.. ==================================

.. image:: _static/logo.jpg
   :align: center
   :width: 400px

.. toctree::
   :maxdepth: 3
   :caption: Getting started

Introduction
------------

``VIP`` is a python package for angular, reference star and spectral
differential imaging for exoplanet and disk high-contrast imaging. ``VIP`` is
compatible with Python 3.7, 3.8 and 3.9 (Python 2 compatibility dropped with ``VIP`` 0.9.9).

.. image:: https://github.com/carlgogo/carlgogo.github.io/blob/master/assets/images/vip.png?raw=true
    :alt: Mosaic of S/N maps

The goal of ``VIP`` is to integrate open-source, efficient, easy-to-use and
well-documented implementations of high-contrast image processing algorithms to
the interested scientific community. The main repository of ``VIP`` resides on
`GitHub <https://github.com/vortex-exoplanet/VIP>`_, the standard for scientific
open source code distribution, using Git as a version control system.

``VIP`` started as the effort of `Carlos Alberto Gomez Gonzalez <https://carlgogo.github.io/>`_,
a former PhD student of the `VORTEX team <http://www.vortex.ulg.ac.be/>`_
(ULiege, Belgium). ``VIP``'s development has been led by Dr. Gomez with contributions
made by collaborators from several teams (take a look at the 
`contributors tab <https://github.com/vortex-exoplanet/VIP/graphs/contributors>`_ on
``VIP``'s GitHub repository). It is now maintained by Dr. Valentin Christiaens.
Most of ``VIP``'s functionalities are mature but
it doesn't mean it's free from bugs. The code is continuously evolving and
therefore feedback/contributions are greatly appreciated. If you want to report
a bug or suggest a functionality please create an issue on GitHub. Pull
requests are very welcomed!

.. include:: trimmed_readme.rst

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   tutorials/01_quickstart.ipynb
   tutorials/02_preproc.ipynb
   tutorials/03_psfsub.ipynb
   tutorials/04_metrics.ipynb
   tutorials/05_fm_planets.ipynb
   tutorials/06_fm_disk.ipynb
   tutorials/07_imlib_and_interpolation.ipynb
   tutorials/08_datasets_as_objects.ipynb

.. toctree::
   :maxdepth: 3
   :caption: About

Frequently asked questions
--------------------------
Check out the FAQ if you encounter problems when installing or running ``VIP``
for the first time.

.. include:: faq.rst

.. include:: about.rst

.. toctree::
   :maxdepth: 3
   :caption: Package content

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

.. toctree::
   :maxdepth: 3

   vip_hci


API
---

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

