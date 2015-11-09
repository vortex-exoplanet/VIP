#!/usr/bin/env python
 
import os
try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup
 

readme = open('readme.rst').read()
doclink = """
Documentation
-------------
TO-DO. Meanwhile please rely on the docstrings and jupyter tutorial. 
"""
 
PACKAGE_PATH = os.path.abspath(os.path.join(__file__, os.pardir))

with open(os.path.join(PACKAGE_PATH, 'VERSION')) as version_file:
    __version__ = version_file.read().strip()
 
setup(
    name='vip',
    version=__version__,
    description='Package for astronomical high-contrast image processing and exoplanet detection.',
    long_description=readme + '\n\n' + doclink + '\n\n',
    author='Carlos Gomez',
    author_email='cgomez@ulg.ac.be',
    url='https://github.com/vortex-exoplanet/VIP',                    
    packages=find_packages(),
    include_package_data=True,
    install_requires=['cython',
                      'numpy >= 1.8',
                      'scipy >= 0.16',
                      'ipython >= 3.2',
                      'jupyter',
                      'astropy >= 1.0.2',
                      'emcee >= 2.1',
                      'corner',
                      'pandas >= 0.16',
                      'matplotlib >= 1.4.3',
                      'scikit-learn >= 0.16',
                      'scikit-image >= 0.11',
                      'pytest',
                      'photutils >= 0.1',
                      'image_registration',
                      'FITS_tools',
                      'pywavelets',
                      'pyprind'],
    zip_safe=False,
    classifiers=['Development Status :: 4 - Beta',
                 'Intended Audience :: Science/Research',
                 'Intended Audience :: Developers',
                 'License :: OSI Approved :: MIT License',
                 'Operating System :: MacOS :: MacOS X',
                 'Operating System :: POSIX :: Linux',
                 'Natural Language :: English',
                 'Programming Language :: Python :: 2.7',
                 'Topic :: Scientific/Engineering :: Astronomy'
                 ] 
)