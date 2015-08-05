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
To be uploaded. Meanwhile please rely on the docstrings. 
"""
 
PACKAGE_PATH = os.path.abspath(os.path.join(__file__, os.pardir))
 
setup(
    name='vip-pipeline',
    version='0.0a1',
    description='VIP package for astronomical high-contrast image processing and exoplanet detection.',
    long_description=readme + '\n\n' + doclink + '\n\n',
    author='Carlos Gomez',
    author_email='cgomez@ulg.ac.be',
    url='https://github.com/carlgogo/vip_exoplanets.git',                    
    packages=find_packages(),
    include_package_data=True,
    install_requires=['cython',
                      'numpy >= 1.8.1',
                      'scipy >= 0.14.0',
                      'ipython',
                      'astropy >= 1.0.2',
                      'emcee',
                      'triangle_plot',
                      'pandas',
                      'matplotlib',
                      'scikit-learn',
                      'scikit-image',
                      'pytest',
                      'photutils',
                      'image_registration',
                      'FITS_tools',
                      'pywavelets',
                      'pyprind'
                      ],
    license=' not defined ',
    zip_safe=False,
    keywords='VIP',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        "Intended Audience :: Science/Research",
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Natural Language :: English',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
    ] 
)