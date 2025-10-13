"""
Subpackage ``stats`` contains functionalities such as:

- extracting statistics (mean, median, std dev, sum) in regions of a frame
  or cube,
- median absolute deviation,
- sigma filtering of pixels in frames,
- distance (correlation) between the frames in a cube,
- distance (correlation) between a cube and a reference frame.

"""
from .clip_sigma import *
from .distances import *
from .im_stats import *
from .utils_stats import *
