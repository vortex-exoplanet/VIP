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

The project started as the effort of `Carlos Alberto Gomez Gonzalez <https://carlgogo.github.io/>`_,
a former PhD student of the `VORTEX team <http://www.vortex.ulg.ac.be/>`_
(ULiege, Belgium). ``VIP``'s development has first been led by Dr. Gomez with contributions
made by collaborators from several teams (take a look at the 
`contributors tab <https://github.com/vortex-exoplanet/VIP/graphs/contributors>`_ on
``VIP``'s GitHub repository). It is now maintained and developed by Dr. Valentin Christiaens.
Most of ``VIP``'s functionalities are mature but
it doesn't mean it's free from bugs. The code is continuously evolving and
therefore feedback/contributions are greatly appreciated. If you want to report
a bug or suggest a functionality please create an issue on GitHub. Pull
requests are very welcomed!

.. image:: https://github.com/carlgogo/carlgogo.github.io/blob/master/assets/images/vip.png?raw=true
    :alt: Mosaic of S/N maps

.. toctree::
   :maxdepth: 2
   :caption: Getting started
   :hidden:

   trimmed_readme_1
   trimmed_readme_2
   trimmed_readme_3

.. toctree::
   :maxdepth: 3
   :caption: Tutorials
   :hidden:

   tuto_link
   tutorials/01_quickstart.ipynb
   tutorials/02_preproc.ipynb
   tutorials/03_psfsub.ipynb
   tutorials/04_metrics.ipynb
   tutorials/05_fm_planets.ipynb
   tutorials/06_fm_disk.ipynb
   tutorials/07_imlib_and_interpolation.ipynb
   tutorials/08_datasets_as_objects.ipynb

.. toctree::
   :maxdepth: 2
   :caption: About
   :hidden:

   trimmed_readme_5
   trimmed_readme_4
   faq

.. toctree::
   :maxdepth: 2
   :caption: Package content
   :hidden:

   vip_hci
   gen_index

