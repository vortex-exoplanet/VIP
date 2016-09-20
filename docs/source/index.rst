.. This file should at least contain the root `toctree` directive.

.. Welcome to ``VIP``'s documentation
.. ==============================

Vortex Image Processing package
================================

.. image::  _static/logo.jpg
   :align:  center
   :scale:  20 %
   :target: #

.. commented
   include:: ../../README.rst


Attribution
------------

Please cite Gomez Gonzalez et al. 2016 (submitted) whenever you publish data
reduced with ``VIP``. Astrophysics Source Code Library reference [ascl:1603.003]


Introduction
-------------

``VIP`` is an open source package/pipeline for angular, reference star and spectral
differential imaging for exoplanet/disk detection through high-contrast imaging.

``VIP`` is being developed by the `VORTEX team <http://www.vortex.ulg.ac.be/>`_ and
collaborators. You can check on github the direct contributions to the code (tab
contributors). Most of ``VIP``'s functionalities are mature but it doesn't mean it's
free from bugs. The code will continue evolving and therefore all the feedback
and contributions will be greatly appreciated. If you want to report a bug,
suggest or add a functionality please create an issue or send a pull request on
the github repository.

The goal of ``VIP`` is to incorporate robust, efficient, easy-to-use, well-documented
and open source implementations of high-contrast image processing algorithms to
the interested scientific community. The main repository of ``VIP`` resides on Github,
the standard for scientific open source code distribution, using Git as a
version control system. Github is a repository hosting service with an amazing
web front-end for source-code management and collaboration. It provides features
such as access control, bug tracking, feature requests, task management, and
wikis for every project.

``VIP`` is being developed in Python 2.7, a modern open source high-level programming
language, able to express a large amount of functionality per line of code.
Python has a vast ecosystem of scientific open source libraries/packages (e.g.
numpy, scipy, astropy, scikit-learn, scikit-image) and many well-known libraries
have python bindings as well (e.g. opencv). On top of that exist a great tool,
the Jupyter (né IPython) notebook. A notebook file is simple a JSON document,
containing text, source code, rich media output, and metadata. It allows to
combine data analysis and visualization into an easily sharable format.


Jupyter notebook tutorial
-------------------------
``VIP`` tutorial (Jupyter notebook) was removed from the main ``VIP`` repository. It
will be updated and released in a separate repository.


Quick Setup Guide
------------------

Get the code from github by downloading it as a zip file or by cloning the
repository (make sure your system has git installed):

.. code-block:: bash

  $ git clone https://github.com/vortex-exoplanet/VIP.git

Install opencv. If you use Anaconda run:

.. code-block:: bash

  $ conda install opencv

From the root of the ``VIP`` package:

.. code-block:: bash

  $ python setup.py develop

Start Python (IPython or Jupyter notebook) and check if the setup worked fine by
importing ``VIP``:

.. code-block:: python

  import vip



Installation and dependencies
------------------------------

You can find a more detailed description of ``VIP``'s dependencies here.

.. toctree::
   :maxdepth: 2

   install

Frequently asked questions
---------------------------

Check out this questions if you find problems when installing or running ``VIP`` for
the first time.

.. toctree::
   :maxdepth: 2

   faq


Package structure
------------------

On the links below you can find the subpackages structure and access the
docstrings (internal documentation) of each one of ``VIP``'s functions.

``VIP`` implements basic image processing functionalities e.g. image registration,
image rotation, pixel temporal and spatial subsampling. On top of that several
pre-processing functions are available such as re-centering and bad frame
detection.

For Angular differential imaging (ADI) data several point spread gunction
subtraction techniques are available, from a simple median subtraction to
different low-rank approximations. Principal Component Analysis (PCA) methods
are implemented in different flavours. Also PCA can process RDI and multiple
channel SDI (IFS) data. ``VIP`` contains a first version of the Local Low-rank plus
Sparse plus Gaussian-noise decomposition (LLSG, Gomez Gonzalez et al. 2016).

Functions for Signal-to-noise ratio (S/R) estimation, S/R map generation are
included.

Flux and position of point-like sources are estimated using the Negative Fake
Companion technique. ``VIP`` also implements algorithm throughput and contrast-curve
generation routines.

.. toctree::
   :maxdepth: 3

   vip


API
----

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

