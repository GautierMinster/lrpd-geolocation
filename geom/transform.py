# encoding=UTF-8

"""Various geometric transformation utilities."""

from math import cos, sin, tan

import cv2
import numpy as np

import geom

class UndefinedAxis(Exception):
    def __init__(self, op):
        self.op = op
    def __str__(self):
        return "{} axis must be one of 'x', 'y' or 'z'.".format(self.op)

def axis_rotation(dims, axis, angle, degs=False):
    """Returns a rotation matrix along a single axis.

    Args:
        dims: number of dimensions in the space, 2 or 3
        axis: axis, 'x', 'y', or 'z'
        angle: rotation angle, in radians by default, or degrees if degs is True
        degs: is the rotation angle in degrees? Default is radians

    Returns:
        A dims+1 square matrix representing the rotation by angle around axis.
    """
    assert dims in [2,3], "Can only handle 2D or 3D rotations."

    a = angle
    if degs:
        a *= np.pi / 180.

    if abs(a) < 1e-6:
        return np.eye(dims+1, dims+1, dtype=np.float_)

    if axis == 'x':
        r = np.array([
            [1,       0,       0, 0],
            [0,  cos(a), -sin(a), 0],
            [0,  sin(a),  cos(a), 0],
            [0,       0,       0, 1]], dtype=np.float_)
    elif axis == 'y':
        r = np.array([
            [ cos(a), 0, sin(a), 0],
            [      0, 1,      0, 0],
            [-sin(a), 0, cos(a), 0],
            [      0, 0,      0, 1]], dtype=np.float_)
    elif axis == 'z':
        r = np.array([
            [cos(a), -sin(a), 0, 0],
            [sin(a),  cos(a), 0, 0],
            [     0,       0, 1, 0],
            [     0,       0, 0, 1]], dtype=np.float_)
    else:
        raise UndefinedAxis("Rotation")

    if dims == 2:
        return r[:3,:3]
    else:
        return r

def rotation(angles, degs=False):
    """Returns a rotation matrix.

    Args:
        angles: rotation angles. Number of angles dictates the rotation space.
        degs: are the angles in degrees? Default is radians

    Returns:
        A len(angles)+1 dimensional square matrix representing the rotation.
    """
    dims = len(angles)
    axes = ['x', 'y', 'z']

    R = None
    for a,axis in zip(angles,axes):
        if R is None:
            R = axis_rotation(dims, axis, a, degs)
        else:
            R = R.dot(axis_rotation(dims, axis, a, degs))

    return R

def axis_translation(dims, axis, t):
    """Returns a translation matrix along an axis.

    Args:
        dims: number of dimensions in the space, 2 or 3
        axis: axis along which to translate, 'x', 'y' or 'z'
        t: amount to translate

    Returns:
        A dims+1 square matrix representing the translation.
    """
    assert dims in [2,3], "Can only handle 2D or 3D translations."
    axes = {'x':0, 'y':1, 'z':2}

    T = np.eye(dims+1, dims+1, dtype=np.float_)
    if axis in axes:
        T[axes[axis], dims] = t
        return T
    else:
        raise UndefinedAxis("Translation")

def compose(*args):
    """Composes a bunch of transforms.

    Args:
        args: transforms to be composed, in this order

    Returns:
        The matrix multiplication of all the transforms.
    """
    assert len(args) > 0, "Must provide at least one transform."
    M = args[0]
    for i in xrange(1,len(args)):
        M = M.dot(args[i])

    return M

def scaling_perspective_transform(height, width, target_size, angles, degs=False):
    """Computes a perspective transform, shifted and scaled to a desired size.

    Args:
        height: height of the bounding box of the data
        width: width of the bounding box of the data
        target_size: maximum size for the width and height of the resulting box
        angles: rotation angles to apply
        degs: are the angles in degrees? Default is False, they are radians.

    Returns:
        A 2-tuple (M, (h,w)), where:
            - M: a 4x4 matrix representing the perspective transform
            - h,w: dimensions of the transformed result
    """
    h, w = float(height), float(width)

    # 1. Compute the initial transform
    ##################################

    # First compute the projection from image coordinates to 3D space
    # Image center (w/2,h/2) is moved to the origin
    P_to_3d = np.array([
        [1, 0, -w/2.],
        [0, 1, -h/2.],
        [0, 0,    0.],
        [0, 0,    1.]], dtype=np.float_)

    # Rotation in 3D space
    R = rotation(angles, degs=True)

    # Translation along z to put the rotated plane in front of the virtual
    # camera
    # Set the virtual camera focal lergth to that of a 100° FOV
    fov_deg = 100.
    f = w / (2. * tan(fov_deg/2. * np.pi / 180.))
    # Set the translation to 1.5 times the focal length
    t = 1.5 * f
    # Compute the translation matrix
    T = axis_translation(3, 'z', t)

    # Backproject from 3D space into 2D using a virtual camera
    # Again, the origin is moved to the center of the obtained image coords
    P_to_2d = np.array([
        [f, 0, w/2., 0],
        [0, f, h/2., 0],
        [0, 0,   1., 0]], dtype=np.float_)

    # Combine all of these in a transform
    M_orig = compose(P_to_2d, T, R, P_to_3d)

    # 2. Compute the image of the corners and their bounding box
    ############################################################

    # Assume the region of interest is (0,0) => (w,h)
    corners = np.array([
            [0, 0, 1], # top left
            [w, 0, 1], # top right
            [0, h, 1], # bottom left
            [w, h, 1]  # bottom right
        ], dtype=np.float_).T # transpose to obtain corners as column vectors

    # Warp the corners
    corners_warped = M_orig.dot(corners)
    # Get Euclidean coordinates instead of homogenous ones
    corners_warped /= corners_warped[2]

    x_min = corners_warped[0,:].min()
    x_max = corners_warped[0,:].max()
    y_min = corners_warped[1,:].min()
    y_max = corners_warped[1,:].max()

    # Compute the center and dimensions of the region of interest
    center_roi = (x_max+x_min)/2., (y_max+y_min)/2.
    h_roi, w_roi = y_max-y_min, x_max-x_min

    # 3. Modify the transform to shift the image center to center_roi
    #################################################################

    P_to_2d[0,2] = w_roi/2. - (center_roi[0] - w/2.)
    P_to_2d[1,2] = h_roi/2. - (center_roi[1] - h/2.)
    M_shifted = compose(P_to_2d, T, R, P_to_3d)

    # 4. Scale the coordinates to obtain the required size
    ######################################################

    # Compute the required scale change (keep aspect ratio)
    scale = float(target_size) / max(w_roi, h_roi)

    # Apply the scale change
    S = np.array([
        [scale,     0, 0],
        [    0, scale, 0],
        [    0,     0, 1]], dtype=np.float_)

    M = compose(S, M_shifted)

    return M, (int(h_roi*scale), int(w_roi*scale))

def transform_segs_perspective(M, segs):
    """Apply a perspective transform to a bunch of segments.

    Args:
        M: a 3x3 perspective transform
        segs: a geom.Segments object

    Return:
        A geom.Segments object containing the transformed segments.
    """
    assert M.shape == (3,3)
    segs = segs.lines

    starts = np.concatenate( (segs[:, :2], np.ones((segs.shape[0],1), dtype=np.float_)), axis=1).T
    ends   = np.concatenate( (segs[:,2:4], np.ones((segs.shape[0],1), dtype=np.float_)), axis=1).T

    M_starts  = M.dot(starts)
    M_starts /= M_starts[2]

    M_ends    = M.dot(ends)
    M_ends   /= M_ends[2]

    result = np.concatenate((M_starts[:2].T, M_ends[:2].T), axis=1)

    return geom.Segments(result)

def transform_segs_affine(M, segs):
    """Apply an affine transform to a bunch of segments.

    Args:
        M: a 2x3 affine transform
        segs: a geom.Segments object

    Returns:
        A geom.Segments object containing the transformed segments.
    """
    assert M.shape == (2,3)

    # Simply add a dummy row to the matrix, and apply the perspective transform
    Mp = np.concatenate([M, np.array([[0.,0.,1.]], dtype=np.float_)], axis=0)

    return transform_segs_perspective(Mp, segs)

