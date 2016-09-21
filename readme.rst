VIP - Vortex Image Processing package
=====================================

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
``VIP`` tutorial (Jupyter notebook) is available in `this repository
<https://github.com/carlgogo/vip-tutorial>`_ and can be visualized online `here
<http://nbviewer.jupyter.org/github/carlgogo/vip-tutorial/blob/master/Tutorial1_VIP_adi_pre-postproc_fluxpos_ccs.ipynb>`_.


Documentation
-------------
Documentation can be found in http://vip.readthedocs.io/.


Quick Setup Guide
------------------
Install opencv. If you use Anaconda run:

.. code-block:: bash
  
  $ conda install opencv

From the root of the VIP package:

.. code-block:: bash

  $ python setup.py develop   

