
FAQ
----
First things first. Please make sure you have the latest version of ``VIP``.
Please go and check the repository now. Alternatively, you can run:

.. code-block:: bash

  $ pip install --upgrade vip_hci


* On linux, why do I get a matplotlib related error when importing ``VIP``?
*ImportError: Matplotlib qt-based backends require an external PyQt4, PyQt5,
or PySide package to be installed, but it was not found.*
You may need to change the matplotlib backend. Find your matplotlibrc
configuration file and change the backend from WXAgg to TkAgg (or the appropriate
backend for your sytem). More info here:
http://matplotlib.org/faq/usage_faq.html#what-is-a-backend. On linux the
matplotlibrc file is located in:

.. code-block:: bash

  $HOME/.config/matplotlib/matplotlibrc


* Why do I get, in OSX, the following RuntimeError?
*Python is not installed as a framework. The Mac OS X backend will not be able
to function correctly if Python is not installed as a framework. See the
Python documentation for more information on installing Python as a framework
on Mac OS X. Please either reinstall Python as a framework, or try one of the
other backends.*
Again, this is a matplotlib-backend related issue. Read the link in the previous
answer. It can be solved setting the backend to WXAgg or TkAgg. Optionally, you
can change your matplotlib backend **before** importing ``VIP``:

.. code-block:: python

  import matplotlib as mpl
  mpl.use('TkAgg')
  import vip


* The ``VIP`` setup.py script doesn't finish the job, it seems to be stuck. What
to do?
This is very unlikely to happen with the latest versions of pip, setuptools
and ``VIP``'s setup script. If you encounter this situation just kill the process
(Ctrl + C) and start it again by re-running the setup command. A workaround
is to install the problematic dependency before executing ``VIP`` setup:

.. code-block:: bash

  $ pip install <problematic_dependency>


* Why the setup fails complaining about the lack of a Fortran compiler?
Fortran compilers are apparently needed for compiling ``Scipy`` from source. Make
sure there is a Fortran compiler in your system. A workaround is to install
``Scipy`` through conda before running the setup script:

.. code-block:: bash

  $ conda install scipy


* Why do I get and error related to importing ``cv2`` package when importing ``VIP``?
``cv2`` is the name of ``Opencv`` bindings for python. This library is needed for
fast image transformations, it is the default library used although it is optional.
You can install it with conda:

.. code-block:: bash

  $ conda install opencv


* Why do I get a warning related to ``DS9/XPA`` when importing ``VIP``?
Please make sure you have ``DS9`` and ``XPA`` in your system path. Try installing
them using your system's package management tool.


* Why Python crashes when using some of the parallel functions, e.g.
``pca_adi_annular`` and ``run_mcmc_astrometry``?
These functions require running SVD on several processes and this can be
problematic depending on the linear algebra libraries on your machine. We've
encountered this problem on OSX systems that use the ACCELERATE library for
linear algebra calculations (default in every OSX system). For this library
the multiprocessing is broken. A workaround is to compile Python against other
linear algebra library (e.g. OPENBLAS). An quick-n-easy fix is to install the
latest ANACONDA (2.5 or later) distribution which ships MKL library and
effectively replaces ACCELERATE on OSX systems. On linux with the default
LAPACK/BLAS libraries ``VIP`` successfully distributes the SVDs among all
the existing cores. With ``conda`` you can run:

.. code-block:: bash

  $ conda install mkl


* I get an error: ValueError: "unknown locale: UTF-8" when importing ``VIP``.
It's not ``VIP``'s fault. The problem must be solved if you add these lines in
your ~/.bash_profile:

.. code-block:: bash

  export LC_ALL=en_US.UTF-8
  export LANG=en_US.UTF-8