# encoding=UTF-8

"""Utilities pertaining to images."""

import logging
import math

import cv2
import numpy as np

log = logging.getLogger(__name__)

def load_images(filenames, load_color=False, convert_to_color=False):
    """Load a list of images given their filenames.

    Args:
        filenames: list of file names
        load_color: if True, load the files as color images
        convert_to_color: if True, convert the image matrices to color
            after loading. Useful when combined with load_color=False,
            to obtain a grayscale image in a color image matrix
    """

    imgs = []
    for filename in filenames:
        read_flag = cv2.IMREAD_GRAYSCALE
        if load_color:
            read_flag = cv2.IMREAD_COLOR

        imgs.append(cv2.imread(filename, read_flag))

        if imgs[-1] is None:
            log.critical("Could not load image file {}".format(filename))
            raise Exception()

        if convert_to_color and not load_color:
            imgs[-1] = cv2.cvtColor(imgs[-1], cv2.COLOR_GRAY2BGR)

    return imgs

def line_pixels(l):
    """Generator that returns the coordinates of pixels on a line.

    The line coordinates are rounded to integers, and there is no guarantee
    about the order in which the pixels will be yielded.

    Args:
        l: an array-like 4-vector of coordinates

    Yields:
        (x, y) couples, the pixels on the line
    """
    [x0, y0, x1, y1] = map(int, map(round, l))
    dx = np.float64(x1-x0)
    dy = np.float64(y1-y0)

    dy_dominant = abs(dy) > abs(dx)

    # If the variation is mostly on y, we'll just swap x and y
    if dy_dominant:
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        dx, dy = dy, dx

    # If the line goes from right to left, switch it to left to right
    if dx < 0.:
        x0, x1 = x1, x0
        y0, y1 = y1, y0
        dx = -dx
        dy = -dy
    elif dx == 0.:
        # In this case the line is a single point
        if dy_dominant:
            yield (y0, x0)
        else:
            yield (x0, y0)
        return

    err = 0.
    derr = abs(dy/dx)
    ystep = 1 if y0 < y1 else -1

    y = y0
    for x in xrange(x0, x1+1):
        if dy_dominant:
            yield (y, x)
        else:
            yield (x, y)
        err += derr
        changed = False
        while err >= .5:
            if changed:
                if dy_dominant:
                    yield (y, x)
                else:
                    yield (x, y)
            y += ystep
            err -= 1.
            changed = True

def draw_segments(segs, img=None, width=1, color=(0,0,255)):
    """Draw segments on an image.

    Args:
        segs: a geom.Segments object
        img: if provided, the image where to draw. If not, an appropriately
            large image will be created
        width: width of the lines drawn
        color: the color to use to draw the segments

    Returns:
        The image with segments drawn onto it (redundant if img was provided.
    """
    xmin,ymin,xmax,ymax = segs.get_bounds()

    if img is None:
        img = np.zeros([int(math.ceil(ymax)),int(math.ceil(xmax)),3],
            dtype=np.uint8)
    assert len(img.shape)==3 and img.shape[2] == 3, \
        "Need a color image! (3-channel uint8)"

    for s in segs:
        x1,y1,x2,y2 = map(int, map(round, s))
        cv2.line(img, (x1,y1), (x2,y2), color, width)

    return img

def draw_svg_segments(segs, outputfile, bbox=None, width=1, bgcolor=(0,0,0),
        segcolor=(255,255,255)):
    """Create an SVG image of segments.

    Args:
        segs: a geom.Segments object
        outputfile: the name of the file to write the SVG image to
        bbox: if provided, a bounding box defining the displayed area, of the
            form (x1, y1, x2, y2), where (x1,y1) and (x2,y2) are opposite
            corners of the box
        width: the stroke width for the segments
        bgcolor: the background color
        segcolor: the segment color
    """
    if bbox is None:
        bbox = segs.get_bounds()
    svgwidth = int(round(
        max(bbox[0], bbox[2]) - min(bbox[0], bbox[2])
    ))
    svgheight = int(round(
        max(bbox[1], bbox[3]) - min(bbox[1], bbox[3])
    ))

    with open(outputfile, 'wb') as f:
        f.write('<svg height="{}" width="{}">'.format(svgheight, svgwidth))
        f.write('<rect width="{}" height="{}" style="fill:rgb({},{},{})" />'\
            .format(svgwidth, svgheight, *bgcolor)
        )

        f.write('<g stroke-linecap="round" stroke-width="{}" stroke="rgb({},{},{})">'\
            .format(width, *segcolor)
        )
        for s in segs:
            x1,y1,x2,y2 = s
            f.write('<line x1="{:.1f}" y1="{:.1f}" x2="{:.1f}" y2="{:.1f}" />'\
                .format(x1,y1,x2,y2)
            )
        f.write('</g>')
        f.write('</svg>')

def rotate_image(img, a, degs=True):
    """Performs a 2D plane rotation of an image.

    The rotated image will contain the entire source image: this means the
    transform being applied will actually be a rotation and a translation, so
    that everything is visible.

    Args:
        img: the image to be rotated
        a: the (signed) rotation angle
        degs: if True, the angle is in degrees

    Returns:
        A (transform, rotated_image) pair, where:
          - transform: a 2x3 matrix, representing the rotation + translation
            that was used (the input and output vectors are in homogeneous
            coordinates, but no modification of the scale actually happens)
          - rotated_image: the image obtained when applying transform to img
    """
    if degs:
        a = a * math.pi / 180.

    h, w = img.shape[:2]
    # Define the coordinates of the corners (as columns)
    corners = np.array([[0, w, 0, w],
                        [0, 0, h, h]], dtype=np.float_)
    # We'll first define the rotation, see where the image corners are warped,
    # and adjust the translation and image size in consequence
    r = np.array([[math.cos(a), -math.sin(a)],
                  [math.sin(a),  math.cos(a)]], dtype=np.float_)
    # Apply the rotation to the corners
    rc = r.dot(corners)
    # Compute the new bounding box
    mins = rc.min(axis=1)
    maxs = rc.max(axis=1)
    # bbox = [xmin, ymin, xmax, ymax]
    bbox = np.asarray(
        np.around([mins[0], mins[1], maxs[0], maxs[1]]),
        dtype=np.int_
    )

    # Size of the new image
    hrot, wrot = bbox[3]-bbox[1], bbox[2]-bbox[0]
    # Translation to apply
    t = np.array([[-bbox[0], -bbox[1]]], dtype=np.float_)

    # Final transform
    m = np.concatenate([r, t.T], axis=1)

    out = cv2.warpAffine(img, m, (wrot, hrot))

    return m, out

