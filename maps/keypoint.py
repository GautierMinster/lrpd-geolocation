# encoding=UTF-8

# stdlib
import math

# 3p
import numpy as np


class Keypoint(object):
    """Map image keypoint, with coordinates, scale, and orientation.

    Attrs:
        p: a numpy 2-vector defining of the form [x,y]
        s: the scale (usually, in pixels of the original-size map)
        o: the orientation, in radians
    """

    def __init__(self, s, o=0., x=None, y=None, pos=None, cast_int=False):
        """Initializes a Keypoint from its defining parameters.

        Args:
            s: the scale of the keypoint (usually in pixels)
            o: the orientation of the keypoint (in radians)
            x, y: the spatial coordinates of the keypoint. If not provided, pos
                must be defined
            pos: a numpy 2-vector of the form [x,y], the coordinates. If not
                provided, x and y must be defined
            cast_int: if True, the coordinates and scale will be cast to ints
        """
        if pos is None:
            assert x is not None and y is not None, \
                "x and y must be given if pos isn't."
            if cast_int:
                self.p = np.asarray([x, y], dtype=np.int_)
            else:
                self.p = np.asarray([x, y], dtype=np.float_)
        else:
            self.p = pos

        self.s = s
        self.o = o

        if cast_int:
            self.s = int(self.s)

    def __repr__(self):
        deg = self.o * 180. / math.pi
        return "<Keypoint (x,y,s,o)=({},{},{},{:.0f})>"\
            .format(self.p[0], self.p[1], self.s, deg)

