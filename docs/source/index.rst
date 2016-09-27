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
the interested scientific community. The main repository of ``VIP`` resides on
Github, the standard for scientific open source code distribution, using Git as a
version control system. Github is a repository hosting service with an amazing
web front-end for source-code management and collaboration. It provides features
such as access control, bug tracking, feature requests, task management, and
wikis for every project.

``VIP`` is being developed in Python 2.7, a modern open source high-level programming
language, able to express a large amount of functionality per line of code.
Python has a vast ecosystem of scientific open source libraries/packages (e.g.
``numpy``, ``scipy``, ``astropy``, ``scikit-learn``, ``scikit-image``) and many
well-known libraries have python bindings as well (e.g. ``opencv``). On top of
that exist a great tool, the ``Jupyter`` (né ``IPython``) notebook. A notebook
file is simple a JSON document, containing text, source code, rich media output,
and metadata. It allows to combine data analysis and visualization into an
easily sharable format.


Jupyter notebook tutorial
-------------------------
``VIP`` tutorial (Jupyter notebook) is available in `this repository
<https://github.com/carlgogo/vip-tutorial>`_ and can be visualized online `here
<http://nbviewer.jupyter.org/github/carlgogo/vip-tutorial/blob/master/Tutorial1_VIP_adi_pre-postproc_fluxpos_ccs.ipynb>`_.


Quick Setup Guide
------------------

Run:

.. code-block:: bash

   $ pip install git+https://github.com/vortex-exoplanet/VIP.git


Mailing list
------------
You can subscribe to our `mailing <http://lists.astro.caltech.edu:88/mailman/listinfo/vip>`_
list if you want to be informed of the latest developments of the ``VIP`` package
(new versions and/or updates).


Installation and dependencies
------------------------------

Here you can find a more detailed description of ``VIP``'s dependencies, how to
clone the repository and install ``Opencv`` (optional).

.. toctree::
   :maxdepth: 2

   install

Frequently asked questions
---------------------------

Check out this questions if you find problems when installing or running ``VIP``
for the first time.

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

For Angular differential imaging (ADI) data several point spread function
subtraction techniques are available, from a simple median subtraction to
different low-rank approximations. Principal Component Analysis (PCA) methods
are implemented in different flavours. Also PCA can process RDI and multiple
channel SDI (IFS) data. ``VIP`` contains a first version of the Local Low-rank plus
Sparse plus Gaussian-noise decomposition (LLSG, Gomez Gonzalez et al. 2016).

Functions for Signal-to-noise ratio (S/N) estimation and S/N map generation are
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

