# encoding=UTF-8

"""Geometry utilities module.

This module provides utilities for the geometric transformation of data, as well
as a class for the handling of sets of segments, Segments.
"""

import numpy as np

class SegmentsException(Exception):
    pass

class Segments(object):
    """Class to represent a list of segments.

    The segments are represented by 4 floating point coordinates, the x and y
    coordinates of each endpoint, of the form [x1, y1, x2, y2].

    The class provides methods to compute the lengths of all the segments, their
    orientations (oriented (x1,y1)=>(x2,y2), or non-oriented), a bounding box of
    the segments, and allows filtering them using masks just like numpy.

    Internally, the list is a numpy array of shape (N, 4), where N is the number
    of segments. The lengths and orientations are cached, so that multiple calls
    to the helper functions do not trigger a recomputation every time.
    """

    def __init__(self, lines):
        """Initialize a Segments instance using a list of lines.

        Args:
            lines: a numpy array, of shape (N, 1, 4) or (N,4)
        """

        tot = lines.shape[0]
        if lines.shape == (tot, 1, 4):
            self.lines = np.reshape(lines, (tot, 4))
        elif lines.shape == (tot, 4):
            self.lines = lines
        elif tot == 0:
            self.lines = np.empty((0,4), dtype=np.float64)
        else:
            raise SegmentsException("lines should be a numpy.ndarray of shape (_, 1, 4) or (_, 4)")

        self.size = self.lines.shape[0]
        self.angles = None
        self.angles_unorient = None
        self.lengths = None
        self.bounds = None

    def get_lengths(self):
        """Returns an array of segment lengths."""
        if self.lengths is None:
            self.lengths = np.empty((self.lines.shape[0]), dtype=np.float64)
            for i in xrange(0, self.lines.shape[0]):
                self.lengths[i] = np.sqrt((self.lines[i][0]-self.lines[i][2])**2.0 + (self.lines[i][1]-self.lines[i][3])**2.0)
        return self.lengths

    def get_angles(self):
        """Returns an array of oriented segment angles.

        The returned angles are in [0, 2pi[.
        """
        if self.angles is None:
            linesf64 = np.asarray(self.lines, dtype=np.float64)
            a_shift = np.arctan2(linesf64[:,3]-linesf64[:,1], linesf64[:,2]-linesf64[:,0])
            # a_shift is in [-π, π], shift it to [0, 2π]
            self.angles = (a_shift + 2.*np.pi) % (2.*np.pi)
        return self.angles

    def get_angles_unoriented(self):
        """Returns an array of unoriented segment angles.

        The returned angles are in [0,pi[.
        """
        if self.angles_unorient is None:
            linesf64 = np.asarray(self.lines, dtype=np.float64)
            a = np.arctan2(linesf64[:,3]-linesf64[:,1], linesf64[:,2]-linesf64[:,0])
            # a is in [-π, π], add π to the negative values, and make sure
            # no angle is at π.
            a[a<0] += np.pi
            a[a>=np.pi] -= np.pi
            self.angles_unorient = a
        return self.angles_unorient

    def get_bounds(self):
        """Get a bounding box for the segments: (xmin,ymin,xmax,ymax).

        The segments are contained in [xmin;xmax[ x [ymin;ymax[.
        """
        if self.bounds is None:
            mins = self.lines.min(axis=0)
            maxs = self.lines.max(axis=0)
            xmin = min(mins[0], mins[2])
            xmax = max(maxs[0], maxs[2])
            ymin = min(mins[1], mins[3])
            ymax = max(maxs[1], maxs[3])
            self.bounds = (xmin,ymin,xmax+1,ymax+1)
        return self.bounds

    def filter_mask(self, mask):
        """Filters the segments using the provided mask."""
        filtered = Segments(self.lines[mask])
        if self.lengths is not None:
            filtered.lengths = self.lengths[mask]
        if self.angles is not None:
            filtered.angles = self.angles[mask]
        if self.angles_unorient is not None:
            filtered.angles_unorient = self.angles_unorient[mask]
        return filtered

    def filter_length(self, l):
        """Returns the segments with length at least l."""
        mask = self.get_filter_length_mask(l)
        return self.filter_mask(mask)

    def get_filter_length_mask(self, l):
        """Returns the mask to filter segments with length less than l."""
        lengths = self.get_lengths()
        return lengths >= l

    def count(self):
        return self.lines.shape[0]

    def __iter__(self):
        """Allow iteration over the data in this Segments instance."""
        for i in xrange(0,self.lines.shape[0]):
            yield self.lines[i]

    def __getitem__(self, i):
        """Allow direct access to the data by indexes on this instance."""
        return self.lines[i]


