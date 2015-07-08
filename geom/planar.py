# encoding=UTF-8

"""Utilities for planar geometry.

In particular, the clipping of a bunch of segments by a rectangle is
implemented.
"""

import numpy as np

def intersect(l1, l2):
    """Intersect 2 lines."""
    assert l1.shape==(3,)
    assert l2.shape==(3,)

    return np.cross(l1, l2)

def fastintersect(l1, l2):
    u1, v1, w1 = l1
    u2, v2, w2 = l2
    return v1*w2-w1*v2, w1*u2-u1*w2, u1*v2-u2*v1

def join(p1, p2):
    """Join 2 points to form a line."""
    assert p1.shape==(3,)
    assert p2.shape==(3,)

    return np.cross(p1, p2)

def to_eucl(v):
    assert v.shape==(3,)
    assert v[2] != 0.

    return v[:2] / v[2]

def to_homo(v):
    assert v.shape==(2,)

    return np.array([v[0], v[1], 1.], dtype=np.float64)

def seg2line(s):
    assert s.shape==(4,)
    pt1,pt2 = np.empty(3,dtype=np.float64), np.empty(3,dtype=np.float64)
    pt1[0], pt1[1], pt1[2] = s[0], s[1], 1.
    pt2[0], pt2[1], pt2[2] = s[2], s[3], 1.
    return join(pt1, pt2)

def fastseg2line(s):
    # Solve directly for u, v, w in the equation: u*x + v*y + w = 0
    # (u,v,w) is the homogenous coordinate vector of the line
    # Using Cramer's rule, and our ability to choose the scale, we set w equal
    # to the opposite of the determinant of the 2 endpoints, and we obtain:
    # u = s[3]-s[1], v = s[0]-s[2], w = - det(s[:2], s[2:4])
    ax, ay, bx, by = s
    return by-ay, ax-bx, -(ax*by-bx*ay)

def intersect_segments(s1, s2):
    """See if 2 segments intersect.

    Returns:
        None if no intersection, an Euclidean point otherwise.
    """
    l1,l2 = fastseg2line(s1),fastseg2line(s2)

    # Find the intersection
    pt_h = fastintersect(l1, l2)

    # Is it at or near infinity?
    if abs(pt_h[2]) < 1.0e-7:
        return None

    pt = (pt_h[0]/pt_h[2], pt_h[1]/pt_h[2])

    # Is it inside the bounds of both segments?
    # vector for the whole segment
    v1 = [s1[2]-s1[0], s1[3]-s1[1]]
    # vector for the origin -> pt portion
    t1 = [pt[0]-s1[0], pt[1]-s1[1]]
    if v1[0]*t1[0]+v1[1]*t1[1] < 0. or t1[0]**2.+t1[1]**2. > v1[0]**2.+v1[1]**2.:
        return None

    v2 = [s2[2]-s2[0], s2[3]-s2[1]]
    t2 = [pt[0]-s2[0], pt[1]-s2[1]]
    if v2[0]*t2[0]+v2[1]*t2[1] < 0. or t2[0]**2.+t2[1]**2. > v2[0]**2.+v2[1]**2.:
        return None

    return pt

def intersect_segments_with_rect(bounds, segs):
    """Intersect a set of segments with a rectangle.

    Args:
        bounds: a (x_min,y_min,x_max,y_max) 4-uplet
        segs: numpy array of segments (ie (x1,y1,x2,y2) )

    Returns:
        A new numpy array of segments, all inside the rectangle.
    """
    # Use square brackets, in case we're actually given list or something
    x_min, y_min, x_max, y_max = bounds[0], bounds[1], bounds[2], bounds[3]

    # Create the left, right, top and bottom segments
    sides = [
        np.array([x_min, y_min, x_min, y_max], dtype=np.float64), # left segment
        np.array([x_max, y_min, x_max, y_max], dtype=np.float64), # right segment
        np.array([x_min, y_max, x_max, y_max], dtype=np.float64), # top segment
        np.array([x_min, y_min, x_max, y_min], dtype=np.float64)  # bottom segment
    ]

    truncated_segs = []

    for seg in segs:
        # Points inside the box
        in_pts = []

        # Add endpoints that are inside the box to the list
        if seg[0]>=x_min and seg[0]<=x_max and seg[1]>=y_min and seg[1]<=y_max:
            in_pts.append(seg[:2])
        if seg[2]>=x_min and seg[2]<=x_max and seg[3]>=y_min and seg[3]<=y_max:
            in_pts.append(seg[2:4])

        if len(in_pts) < 2:
            for side in sides:
                pt = intersect_segments(seg, side)
                if pt is not None:
                    in_pts.append(pt)
                    if len(in_pts) == 2:
                        break

        if len(in_pts) == 2:
            truncated_segs.append(np.array(in_pts).flatten())

    return np.array(truncated_segs, dtype=np.float_)

