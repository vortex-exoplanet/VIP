
FAQ
----
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