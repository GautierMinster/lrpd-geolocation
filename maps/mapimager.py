# encoding=UTF-8

# stdlib
import logging
from math import cos,sin,tan

# 3p
import cv2
import numpy as np

# project
import geom.transform
import maps.mapprocessor
import geom


log = logging.getLogger(__name__)


class MapImager(object):
    """Creates images of an OpenStreetMap map in different sizes and viewpoints.

    Given a MapProcessor (ie pre-processed OpenStreetMap data), provides
    utilities to generate images of it in various sizes and under given
    viewpoints.

    Can filter the imaged map segments on their lengths.
    """

    def __init__(self, mapproc, road_length_disp_thres=0.01, road_length_description_thres=0.04):
        """Initializes the imager.

        Args:
            mapproc: the MapProcessor used to obtain the map data
            road_length_disp_thres: threshold on the minimum length of roads
                we'll display, in [0,1] (% of the maximum image size)
            road_length_description_thres: threshold on the length of segments
                we'll use for matching, in [0,1] (% of the maximum image size)
        """

        self.mapproc = mapproc

        self.roads = self.mapproc.roads

        self.rlen_disp_thres = road_length_disp_thres
        self.rlen_desc_thres = road_length_description_thres

    def create_image(self, max_size, line_width=2):
        """Creates an image and a set of segments to describe from roads.

        The image is generated from all road segments larger than a
        threshold (given on object creation).
        The segments to describe are all road segments larger than the
        other threshold given at object creation.

        Args:
            max_size: the maximum dimension, in pixel, of the image (original scale will
                be preserved, so all that is known is that each dimension will be less
                than max_size)
            line_width: the width in pixels of the lines drawn

        Returns:
            A couple, containing:
                - the segments to describe, as a geom.Segments object
                - the image, with segments drawn onto it
        """

        w, h = float(max_size), float(max_size)
        x_max = self.mapproc.scaling['x_max']
        y_max = self.mapproc.scaling['y_max']
        w *= x_max
        h *= y_max

        # Return to integer values
        w, h = int(w), int(h)

        # Find out which segments we'll display
        disp_mask = self._get_displayed_segs()

        # Compute the scaled segments
        scaled_segs = self._get_scaled_segs(max_size)

        # Draw the image
        img = self._draw_segments(w, h, scaled_segs.filter_mask(disp_mask),
                line_width)

        # Filter the segments to get the ones we'll describe
        final_mask = self._get_described_segs()

        return scaled_segs.filter_mask(final_mask), img

    def create_perspective_image(self, max_size, angles, line_width=2):
        """Creates a perspective image of the map.

        Args:
            max_size: the maximal dimension of the output image
            angles: a 3 element list of floats containing the rotation angles
                (in degrees) to apply to the image plane
            line_width: the width of the stroke used to draw the segments

        Returns:
            (roads, image), where:
            - roads is the list of roads in the image (same indexing that is
                returned by create_image()
            - image is the perspective image of the map
        """

        h, w = self.mapproc.scaling['y_max'], self.mapproc.scaling['x_max']

        M, (hnew,wnew) = geom.transform.scaling_perspective_transform(h, w,
                max_size, angles, degs=True)

        # Get the segments to display and describe
        disp_mask = self._get_displayed_segs()
        desc_mask = self._get_described_segs()

        # Transform the segments
        roads_warped = geom.transform.transform_segs_perspective(M, self.roads)

        # Create the final image
        output = self._draw_segments(wnew, hnew,
            roads_warped.filter_mask(disp_mask), line_width)

        return roads_warped.filter_mask(desc_mask), output

    def get_metric_distance(self, max_size, pt1, pt2=None):
        """Returns the metric distance between two points on an image of a map.

        The distance in expressed in the units of the original map data (eg in
        meters, probably).

        If no second point (pt2) is given, pt1 is assumed to be either a
        vector, in which case its metric length is returned, or a scalar, in
        which case it is assumed to be a distance in pixels to be converted.

        WARNING: obviously, this is only intended for non-perspective images.

        Args:
            max_size: the maximum dimension of the image (ie the size parameter
                that was given when the image was created in the first place)
            pt1: if pt2 is not None, a (floating point) pixel position. Else,
                either a pixel position too, or a simple distance in pixel-unit
            pt2: endpoint of the segment to compute the length of

        Returns:
            A float, the distance in world units between pt1 and pt2, or the
            length of pt1, or the conversion of pt1 pixels in world units.
        """
        if pt2 is not None:
            pt1 = map(float, pt1)
            pt2 = map(float, pt2)
            d = ((pt2[0]-pt1[0])**2. + (pt2[1]-pt1[1])**2.)**.5
        elif type(pt1) in [int, float]:
            d = float(pt1)
        else:
            pt1 = map(float, pt1)
            d = (pt1[0]**2. + pt1[1]**2.)**.5

        # Distance in world units
        d_wu = d / (float(max_size) * self.mapproc.scaling['world'])

        return d_wu

    def get_pixel_distance(self, max_size, distance_wu):
        """Returns the pixel distance corresponding to a world distance.

        The distance_wu is expressed in the units of the original map, ie most
        likely meters.

        WARNING: obviously, this is only intended for pixel distances in
        non-perspective images.

        Args:
            max_size: the maximum dimension of the image (ie the size parameter
                that is given to create a corresponding image)
            distance_wu: distance in world units to be converted

        Returns:
            A float, the distance in pixel units.
        """
        d_px = float(distance_wu) \
            * self.mapproc.scaling['world'] * float(max_size)

        return d_px

    def _draw_segments(self, width, height, segs, line_width):
        """Creates an image with segments drawn onto it."""

        img = np.zeros((height, width), dtype=np.uint8)

        for s in segs:
            pt1 = (int(s[0]), int(s[1]))
            pt2 = (int(s[2]), int(s[3]))
            img = cv2.line(img, pt1, pt2, 255, line_width)

        return img

    def _get_scaled_segs(self, max_size):
        """Returns the scaled segments."""

        # We scaled uniformly x and y, so simply multiply by max_size
        segs = geom.Segments(
            self.roads.lines * np.array([float(max_size)] * 4, dtype=np.float64)
        )
        return segs

    def _get_displayed_segs(self, segs=None):
        """Returns the mask of displayed segs."""

        if segs is None:
            segs = self.roads

        if self.rlen_disp_thres == 0.:
            return np.full(segs.count(), True, dtype=np.bool_)
        else:
            return segs.get_filter_length_mask(self.rlen_disp_thres)

    def _get_described_segs(self, segs=None):
        """Returns the mask of described segs."""

        if segs is None:
            segs = self.roads

        if self.rlen_desc_thres == 0.:
            return np.full(segs.count(), True, dtype=np.bool_)
        else:
            return segs.get_filter_length_mask(self.rlen_desc_thres)

