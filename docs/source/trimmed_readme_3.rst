Image conventions
-----------------

By default, VIP routines are compatible with either even- or odd-dimension input frames. For VIP routines that require the star to be centered in the input images (e.g. post-processing routines involving (de)rotation or scaling), the code will assume that it is placed on (zero-based indexing):

- size/2-0.5 for odd-size input images; 
- size/2 for even-size input images;

i.e. exactly on a pixel in either cases. The VIP recentering routines will place the star centroid at one of these locations accordingly.


