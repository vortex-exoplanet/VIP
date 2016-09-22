Installation and dependencies
------------------------------
The benefits of a python package manager or distribution, such as Canopy or
Anaconda, are multiple. Mainly it brings easy and robust package management and
avoids messing up with your system's default python. An alternative is to use
package managers like apt-get for Ubuntu or
Homebrew/MacPorts/Fink for OSX. I personally recommend using Anaconda which you
can find here: https://www.continuum.io/downloads.

A setup.py file (Setuptools Python package) is included in the root folder of
VIP. It takes care of installing most of the requirements for you. ``VIP`` depends
on existing packages from the Python ecosystem, such as ``numpy``, ``scipy``,
``matplotlib``, ``pandas``, ``astropy``, ``scikit-learn``, ``scikit-image``,
``photutils``, ``emcee``, etc.

``VIP`` ships a stripped-down version of ``RO.DS9`` (by Russell Owen) for convenient
``xpaset/xpaget`` based interaction with ``DS9``. ``VIP`` contains a class
``vipDS9`` that works on top of ``RO.DS9`` containing several useful methods for
``DS9`` control such as displaying arrays, manipulating regions, controlling the
display options, etc. ``VipDS9`` functionality will only be available if you have
``DS9`` and ``XPA`` installed on your system PATH.

First, get the code from github by downloading it as a zip file or by cloning the
repository (make sure your system has git installed):

.. code-block:: bash

  $ git clone https://github.com/vortex-exoplanet/VIP.git

``Opencv`` (Open source Computer Vision) provides fast c++ image processing
operations and is used by ``VIP`` for basic image transformations. Starting from
version 0.5.2 of ``VIP`` the dependency on ``opencv`` is optional. If you don't
have/want the ``opencv`` python bindings, ``VIP`` will use the much slower
``ndimage/scikit-image`` libraries transparently. Installing ``opencv`` library
is tricky, but having Anaconda python distribution (``conda`` package manager)
the process takes just a command (this is an optional step):

.. code-block:: bash

  $ conda install opencv

For installing ``VIP`` in your system run setup.py:

.. code-block:: bash

  $ python setup.py install

Wait a minute until all the requirements are satisfied. Finally start Python
(or IPython/Jupyter notebook if you prefer) and check that you are able to
import ``VIP``:

.. code-block:: python

  import vip

Hopefully you will see the welcome message of ``VIP``.