Installation and dependencies
-----------------------------
The benefits of using a Python package manager (distribution), such as
(ana)conda or Canopy, are many. Mainly, it brings easy and robust package
management and avoids messing up with your system's default python. An
alternative is to use package managers like apt-get for Ubuntu or
Homebrew/MacPorts/Fink for macOS. We recommend using 
`Miniconda <https://conda.io/miniconda>`_.

``VIP`` depends on existing packages from the Python ecosystem, such as
``numpy``, ``scipy``, ``matplotlib``, ``pandas``, ``astropy``, ``scikit-learn``,
``scikit-image``, ``photutils`` and others. There are different ways of
installing ``VIP`` suitable for different scenarios.


Using pip
^^^^^^^^^
The easiest way to install ``VIP`` is through the Python Package Index, aka
`PyPI <https://pypi.org/>`_, with the ``pip`` package manager. Simply run:

.. code-block:: bash

  $ pip install vip_hci

With ``pip`` you can easily uninstall, upgrade or install a specific version of
``VIP``. For upgrading the package run:

.. code-block:: bash

  $ pip install --upgrade vip_hci

Alternatively, you can use ``pip install`` and point to the GitHub repo:

.. code-block:: bash

  $ pip install git+https://github.com/vortex-exoplanet/VIP.git

Using the setup.py file
^^^^^^^^^^^^^^^^^^^^^^^
You can download ``VIP`` from its GitHub repository as a zip file. A ``setup.py``
file (setuptools) is included in the root folder of ``VIP``. Enter the package's
root folder and run:

.. code-block:: bash

  $ python setup.py install


Using Git
^^^^^^^^^
If you plan to contribute or experiment with the code you need to make a 
fork of the repository (click on the fork button in the top right corner) and 
clone it:

.. code-block:: bash

  $ git clone https://github.com/<replace-by-your-username>/VIP.git

If you do not create a fork, you can still benefit from the ``git`` syncing
functionalities by cloning the repository (but will not be able to contribute):

.. code-block:: bash

  $ git clone https://github.com/vortex-exoplanet/VIP.git

Before installing the package, it is highly recommended to create a dedicated
conda environment to not mess up with the package versions in your base 
environment. This can be done easily with (replace vipenv by the name you want
for your environment):

.. code-block:: bash

  $ conda create -n vipenv python=3.9 ipython

Note: installing ipython while creating the environment with the above line will
avoid a commonly reported issue which stems from trying to import VIP from 
within a base python2.7 ipython console.

To install VIP, simply cd into the VIP directory and run the setup file 
in 'develop' mode:

.. code-block:: bash

  $ cd VIP
  $ python setup.py develop

If cloned from your fork, make sure to link your VIP directory to the upstream 
source, to be able to easily update your local copy when a new version comes 
out or a bug is fixed:

.. code-block:: bash

  $ git add remote upstream https://github.com/vortex-exoplanet/VIP.git

If you plan to develop VIP or use it intensively, it is highly recommended to 
also install the optional dependencies listed below.


Optional dependencies
^^^^^^^^^^^^^^^^^^^^^
The following dependencies are not automatically installed upon installation of ``VIP`` but may significantly improve your experience:

- ``VIP`` contains a class ``vip_hci.vip_ds9`` that enables, through ``pyds9``, the interaction with a DS9 window (displaying numpy arrays, controlling the display options, etc). To enable this feature, ``pyds9`` must be installed from the latest development version: ``pip install git+git://github.com/ericmandel/pyds9.git#egg=pyds9``
- Also, you can install the Intel Math Kernel Library (``mkl``) optimizations (provided that you have a recent version of ``conda``) or ``openblas`` libraries. Either of them can be installed with ``conda install``. 
- ``VIP`` offers the possibility of computing SVDs on GPU by using ``CuPy`` (starting from version 0.8.0) or ``PyTorch`` (from version 0.9.2). These remain as optional requirements, to be installed by the user, as well as a proper CUDA environment (and a decent GPU card).
- Finally, bad pixel correction routines can be optimised with ``Numba``, which  converts some Python code, particularly ``NumPy``, into fast machine code. A factor up to ~50x times speed improvement can be obtained on large images compared to NumPy. Numba can be installed with ``conda install numba``.


Loading VIP
^^^^^^^^^^^
Finally, start Python (or IPython or a Jupyter notebook if you prefer) and check
that you are able to import ``VIP``:

.. code-block:: python

  import vip_hci as vip

If everything went fine with the installation, you will see a welcome message.
Now you can start finding exoplanets!


