Conventions
-----------

Image Center
^^^^^^^^^^^^
By default, VIP routines are compatible with either even- or odd-dimension input frames. For VIP routines that require the star to be centered in the input images (e.g. post-processing routines involving (de)rotation or scaling), the code will assume that it is placed on (zero-based indexing):

- size/2-0.5 for odd-size input images;
- size/2 for even-size input images;

i.e. exactly on a pixel in either cases. The VIP recentering routines will place the star centroid at one of these locations accordingly.

Position angles
^^^^^^^^^^^^^^^
In VIP, all angles are measured counter-clockwise from the positive x axis (i.e. trigonometric angles), following the convention of most packages VIP leverages upon. This includes the position angles returned by algorithms in the forward modelling subpackage of VIP used to characterize directly imaged exoplanets (e.g. negative fake companion routines). This convention is different to the typical astronomical convention which measures angles east from north (for the conversion, simply subtract 90deg to angles returned by VIP).

4D IFS+ADI cubes
^^^^^^^^^^^^^^^^
For all routines compatible with 4D IFS+ADI cubes, the convention throughout VIP is to consider the zeroth axis to correspond to the spectral dimension, and the first axis to be the temporal (ADI) dimension.
