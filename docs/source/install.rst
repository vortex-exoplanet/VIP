Installation and dependencies
=============================

The benefits of a python package manager or distribution, such as Anaconda or
Canoy, are multiple. Mainly it brings easy and robust package management and
avoids messing up with your system's default python. An alternative is to use
package managers like apt-get for Ubuntu or
Homebrew/MacPorts/Fink for OSX. I personally recommend using Miniconda which you
can find here: https://conda.io/miniconda.html.

``VIP`` depends on existing packages from the Python ecosystem, such as
``numpy``, ``scipy``, ``matplotlib``, ``pandas``, ``astropy``, ``scikit-learn``,
``scikit-image``, ``photutils`` and ``emcee``. There are different ways of
installing ``VIP`` suitable for different scenarios.


Using PIP
---------
The easiest way is to install it
from the Python Package Index, aka Pypi, with the package manager ``pip``:

.. code-block:: bash

  $ pip install vip_hci

With ``pip`` you can easily uninstall ``VIP`` or install specific version of it.
Alternatively, you could run ``pip install`` and point to the Github repository:

.. code-block:: bash

  $ pip install git+https://github.com/vortex-exoplanet/VIP.git

Using the setup.py file
-----------------------
You can download ``VIP`` from its Github repository as a zip file. A setup.py
file (Setuptools Python package) is included in the root folder of
``VIP``. Enter the root folder and run:

.. code-block:: bash

  $ python setup.py install

Using GIT
---------
If you want to use the ``git`` functionalities, you need to clone the repository
(make sure your system has ``git`` installed):

.. code-block:: bash

  $ git clone https://github.com/vortex-exoplanet/VIP.git

Then you need to install the package by following the previous steps, using the
setup.py file.

Other dependencies
------------------
``Opencv`` (Open source Computer Vision) provides fast c++ image processing
operations and is used by ``VIP`` for basic image transformations (starting from
version 0.5.2 of ``VIP`` the dependency on ``opencv`` is optional). If you don't
have/want the ``opencv`` python bindings, ``VIP`` will use the much slower
``ndimage/scikit-image`` libraries transparently. Installing ``opencv`` library
is nowadays and easy process that is done automatically with the ``VIP``
installation. Alternatively, you could use ``conda``:

.. code-block:: bash

  $ conda install opencv

``VIP`` ships a stripped-down version of ``RO.DS9`` (by Russell Owen) for convenient
``xpaset/xpaget`` based interaction with ``DS9``. ``VIP`` contains a class
``vipDS9`` that works on top of ``RO.DS9`` containing several useful methods for
``DS9`` control such as displaying arrays, manipulating regions, controlling the
display options, etc. ``VipDS9`` functionality will only be available if you have
``DS9`` and ``XPA`` installed on your system PATH.

Also, optionally you can install the Intel Math Kernel Library (MKL)
optimizations provided that you have Anaconda(>v2.5) and ``conda`` on your
system. This is recommended along with ``Opencv`` for maximum speed on ``VIP``
computations. Run:

.. code-block:: bash

  $ conda install mkl

Starting from version 0.8.0 ``VIP`` offers the possibility of computing SVDs
on GPU by using ``cupy``. This remains an optional requirement, to be installed
by the user, as it requires having and actual GPU card and a proper CUDA
environment.

Loading VIP
===========
Start Python (or IPython or a Jupyter notebook if you prefer) and check that you
are able to import ``VIP``:

.. code-block:: python

  import vip_hci as vip

If everything went fine with the installation, you will see a welcome message.
Now you can start finding exoplanets!