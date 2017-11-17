"""
Subpackage ``fits`` includes fits handling functions:
    - fits opening
    - fits info
    - fits writing
    - appending extensions to a fit file
    - ADI cube opening (cube with PA attached as HDU extension)
    - vipDS9 class for interaction with DS9. It contains functionalities such as
    displaying fits files, in memory numpy arrays, saving a DS9 frame, changing
    various visualization options, manipulating regions, and passing XPAset and
    XPAget commands to DS9.
"""

from fits import *
from vipds9_source import *
