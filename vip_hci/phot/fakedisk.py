#! /usr/bin/env python

"""
Module with fake companion injection functions.
"""

from __future__ import division, print_function

__author__ = 'Carlos Alberto Gomez Gonzalez'
__all__ = ['collapse_psf_cube',
           'normalize_psf',
           'cube_inject_companions',
           'frame_inject_companion']

import numpy as np
import photutils
from ..preproc import cube_crop_frames, frame_shift, frame_crop
from ..var import (frame_center, fit_2dgaussian, fit_2dairydisk, fit_2dmoffat,
                   get_circle)


# 



