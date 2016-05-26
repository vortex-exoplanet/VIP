**************************************
Vortex Image Processing pipeline (VIP)
**************************************

  Please cite Gomez Gonzalez et al. 2016 (in prep.) whenever you publish data 
  reduced with VIP.

VIP is registered on the Astrophysics Source Code Library [ascl:1603.003].
  
VIP is a package/pipeline for angular, reference star and spectral 
differential imaging for exoplanet detection through high-contrast imaging. 
VIP is being developed in Python 2.7.

VIP is being developed within the VORTEX team @ University of Liege (Belgium).
It's in alpha version meaning that the code will change drastically before the
first release version. If you want to report a bug, suggest a feature or add a 
feature please contact the main developer at cgomez [at] ulg.ac.be or through 
github.


Documentation
=============
VIP tutorials (Jupyter notebook) was removed from the main VIP repository. It 
will be updated and released in a separate repository. Documentation in HTML and
PDF will be produced when VIP paper is published.


Quick Setup Guide
=================
Install opencv. If you use Anaconda run:

.. code-block:: bash
  
  $ conda install opencv

From the root of the VIP package:

.. code-block:: bash

  $ python setup.py develop   


Installation and dependencies
=============================
Having a python distribution installed, such as Canopy or Anaconda, will allow 
easy and robust package management and avoid messing up with your system's default 
python. An alternative is to use package managers like apt-get for Ubuntu or 
Homebrew/MacPorts/Fink for OSX. I personally recommend using Anaconda which you
can grab it from here: https://www.continuum.io/downloads. 

A setup.py file (Setuptools Python package) is included in the root folder of 
VIP. It takes care of installing most of the requirements for you. VIP depends on 
existing packages from the Python ecosystem, such as numpy, scipy, matplotlib, 
pandas, astropy, scikit-learn, scikit-image, photutils, emcee, etc.

VIP ships a stripped-down version of RO.DS9 (Russell Owen) for convenient 
xpaset/xpaget based interaction with DS9. VIP contains a class vipDS9 that works
on top of RO.DS9 containing several useful methods for DS9 control such as 
displaying arrays, manipulating regions, controlling the display options, etc. 
VipDS9 functionality will only be available if you have DS9 and XPA installed 
on your system PATH. 

OpenCV (Open source Computer Vision) is used for fast image processing operations. 
Opencv and its python bindings are usually difficult to install (can't be processed 
from the setup.py requirements) but fortunately with just one command we'll get it 
ready.

Make sure to install OpenCV before running setup.py and make sure you have a C 
compiler installed in your system, like g++. With conda (anaconda) just type:

.. code-block:: bash

  $ conda install opencv

For installing VIP in your system run setup.py:

.. code-block:: bash

  $ python setup.py install

The code is in continuous development and will be changing often. It's preferred 
to 'install' with the develop flag:

.. code-block:: bash

  $ python setup.py develop

In any case wait a couple of minutes until all the requirements are satisfied.


FAQ
===
- The VIP setup doesn't finish the job, it seems to be stuck. What do I do?
  We have experienced a few times that the setup script hangs while installing
  photutils. The reason why it crashes when compiling its own modules is unknown. 
  We recommend to kill the process (Ctrl + C) and restart it again by re-running 
  the setup command. A workaround is to install photutils before executing VIP
  setup:
  
.. code-block:: bash

  $ conda install --channel https://conda.anaconda.org/astropy photutils
  
- Why the setup fails complaining about the lack of a Fortran compiler?
  Fortran compilers are apparently needed for compiling Scipy from source. Make
  sure there is a Fortran compiler in your system. A workaround is to install
  Scipy through conda before running the setup script:
  
.. code-block:: bash

  $ conda install scipy   
   
- Why do I get and error related to importing cv2 package when importing VIP?
  cv2 is the name of opencv bindings for python. This library is needed for
  fast image transformations. You have to install by following the 
  aforementioned instructions.

- Why do I get a warning related to DS9/XPA when importing VIP?
  Please make sure you have DS9 and XPA in your system path. Try installing it
  with your system package management tool. 

- Why Python crashes when using some of the parallel functions, e.g. 
  *pca_adi_annular_quad* and *run_mcmc_astrometry*?
  These functions require running SVD on several processes and this can be 
  problematic depending on the linear algebra libraries on your machine. We've
  encountered this problem on OSX systems that use the ACCELERATE library for 
  linear algebra calculations (default in every OSX system). For this library
  the multiprocessing is broken. A workaround is to compile Python against other 
  linear algebra library (e.g. OPENBLAS). An quick-n-easy fix is to install the
  latest ANACONDA (2.5 or later) distribution which ships MKL library and 
  effectively replaces ACCELERATE on OSX systems. On linux with the default 
  LAPACK/BLAS libraries VIP successfully distributes the SVDs among all 
  the existing cores. 

- Why do I get, in linux, a matplotlib related error when importing VIP? 
  (Matplotlib backend_wx and backend_wxagg require wxPython >=2.8)
  If you use Canopy python distro then this is caused by the combination
  linux/Canopy. Nothing to do with the VIP pipeline. You may need to change the
  matplotlib backend. Find your matplotlibrc configuration file and change the 
  backend from WXAgg to Qt4Agg. More info here:
  http://matplotlib.org/faq/usage_faq.html#what-is-a-backend

- Why do I get, in OSX, the RuntimeError shown below?
  (Python is not installed as a framework. The Mac OS X backend will not be able 
  to function correctly if Python is not installed as a framework. See the 
  Python documentation for more information on installing Python as a framework 
  on Mac OS X. Please either reinstall Python as a framework, or try one of the 
  other backends.)
  Again, this is a matplotlib-backend issue (not VIP related). Read the link in
  the previous question. It can be solved setting the backend to WXAgg or TkAgg.

- I get an error: ValueError: "unknown locale: UTF-8" when importing VIP. 
  It's not a VIP related problem. The problem must be solved if you add these 
  lines in your ~/.bash_profile:
  
.. code-block:: bash

  export LC_ALL=en_US.UTF-8
  export LANG=en_US.UTF-8   
   
   






