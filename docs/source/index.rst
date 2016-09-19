.. This file should at least contain the root `toctree` directive.

.. Welcome to VIP's documentation
.. ==============================

.. VIP - Vortex Image Processing package
.. =====================================

.. image::  _static/logo.jpg
   :align:  center
   :scale:  20 %
   :target: #

.. commented
   include:: ../../README.rst


Attribution
------------

Please cite Gomez Gonzalez et al. 2016 (submitted) whenever you publish data
reduced with VIP. Astrophysics Source Code Library reference [ascl:1603.003]


Introduction
-------------

VIP is a package/pipeline for angular, reference star and spectral
differential imaging for exoplanet/disk detection through high-contrast imaging.
VIP is being developed in Python 2.7.

VIP is being developed within the VORTEX team @ University of Liege (Belgium).
It's in alpha version meaning that the code will change drastically before the
first release version. If you want to report a bug, suggest or add a
functionality please create an issue on the github repository.


Jupyter notebook tutorial
-------------------------
VIP tutorial (Jupyter notebook) was removed from the main VIP repository. It
will be updated and released in a separate repository.


Quick Setup Guide
------------------

Install opencv. If you use Anaconda run:

.. code-block:: bash

  $ conda install opencv

From the root of the VIP package:

.. code-block:: bash

  $ python setup.py develop


Installation and dependencies
------------------------------

You can find a more detailed description of VIP's dependencies here.

.. toctree::
   :maxdepth: 2

   install

Frequently asked questions
---------------------------

Check out this questions if you find problems when installing or running VIP for
the first time.

.. toctree::
   :maxdepth: 2

   faq

API documentation
------------------

In this pages you can find the subpackages structure and access the docstrings
or internal documentation of each one of VIP's functions.

.. toctree::
   :maxdepth: 3

   vip

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

