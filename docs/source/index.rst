.. This file should at least contain the root `toctree` directive.

.. Welcome to ``VIP``'s documentation
.. ==================================

.. image:: _static/logo.jpg
   :align: center
   :width: 400px

What is VIP?
------------
``VIP`` stands for Vortex Image Processing. 
It is a python package for high-contrast imaging of exoplanets and circumstellar disks. 
``VIP`` is compatible with Python 3.7, 3.8 and 3.9 (Python 2 compatibility dropped with ``VIP`` 0.9.9).

The goal of ``VIP`` is to integrate open-source, efficient, easy-to-use and
well-documented implementations of high-contrast image processing algorithms to
the interested scientific community. The main repository of ``VIP`` resides on
`GitHub <https://github.com/vortex-exoplanet/VIP>`_, the standard for scientific
open source code distribution, using Git as a version control system.

Most of ``VIP``'s functionalities are mature but
it does not mean it is free from bugs. The code is continuously evolving and
therefore feedback/contributions are greatly appreciated. Please refer to `these instructions <https://vip.readthedocs.io/en/latest/Contact.html>`_ if you want to report
a bug, ask a question, suggest a new functionality or contribute to the code (the latter is particularly welcome)!

.. image:: https://github.com/carlgogo/carlgogo.github.io/blob/master/assets/images/vip.png?raw=true
    :alt: Mosaic of S/N maps

.. toctree::
   :maxdepth: 2
   :caption: Getting started
   :hidden:

   TLDR
   Installation-and-dependencies
   Image-conventions

.. toctree::
   :maxdepth: 3
   :caption: Tutorials
   :hidden:

   tutos
   tutorials/01_quickstart.ipynb
   tutorials/02_preproc.ipynb
   tutorials/03_psfsub.ipynb
   tutorials/04_metrics.ipynb
   tutorials/05_fm_planets.ipynb
   tutorials/06_fm_disk.ipynb
   tutorials/07_ifs_psfsub_fm_planets.ipynb
   tutorials/08_imlib_and_interpolation.ipynb
   tutorials/09_datasets_as_objects.ipynb

.. toctree::
   :maxdepth: 2
   :caption: About
   :hidden:

   Contact
   Attribution
   faq

.. toctree::
   :maxdepth: 2
   :caption: Package content
   :hidden:

   vip_hci
   gen_index

