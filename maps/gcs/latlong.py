# encoding=UTF-8

# stdlib
import logging
import math
import re

# 3p
import numpy as np

# project
from maps.gcs.gcs import GCS
import geom


log = logging.getLogger(__name__)


class GCS_LatLong(GCS):
    """Latitude-Longitude Geographic Coordinate System.

    Latitude-Longitude GCS:
        The latitude-longitude GCS allows to position a point on the Earth. We
        assume that the described region is small enough that it can be
        considered planar, and that the latitude can be identified with the
        y-axis and the longitude with the x-axis of a standard 2D plane.

        Since no altitude value is specified, when converting latlong
        coordinates to a metric frame, we need to be given a rough estimate of
        the altitude. This, along with some constant values defining the radius
        of the Earth, will allow us to measure distances in meters.
    """

    # Constants for the Earth, in meters
    EQUATORIAL_RADIUS = 6378137.0
    POLAR_RADIUS      = 6356752.3

    # Regexps to parse the latitude-longitude segments data
    number_pat = r'''
            (?P<{}>
                -? # Optional minus sign
                \d+
                \.? # Optional decimal separator
                (?:\d+)? # Optional additional digits
            )
    '''
    pat = re.compile(
        ''.join([
            r'\A \s* (?:LINESTRING)? \s*', # Allow a leading LINESTRING (WKT files)
            r'\( ',
                number_pat.format('long1'),
                r'\s+',
                number_pat.format('lat1'),
                r', \s+',
                number_pat.format('long2'),
                r'\s+',
                number_pat.format('lat2'),
            r'\)'
        ]),
        re.VERBOSE
    )

    def __init__(self, data_iterator):
        log.debug('Loading segments in Latitude-Longitude format')
        coords = []
        for line in data_iterator:
            m = self.pat.match(line)
            if not m:
                continue
            d = m.groupdict()
            coords.append((
                float(d['lat1']),
                float(d['long1']),
                float(d['lat2']),
                float(d['long2'])
            ))

        self.segs_raw = np.asarray(coords, dtype=np.float_)

    def condition(self, params={}):
        """Condition the raw data to a local coordinate frame.

        Given the altitude of the described region, given in the parameters,
        we transform the coordinates into a standard metric frame, scale them
        so that the truncate_rect fits [0,1]x[0,1], and then make sure the axes
        are correctly oriented.

        Args:
            params: additional information for the operation, a dict containing
                - truncate_rect: (optional) a list of 4 floats, the first 2 are
                  the coordinates (latitude, longitude) of the top left corner,
                  the last 2 of the bottom right corner. If not specified, the
                  bounding box of the segments will be used instead.
                - altitude: the altitude of the described region, in meters

        Returns:
            See the documentation of GCS.condition().
        """
        box_latlong = params.get('truncate_rect')

        if box_latlong is None:
            segs_min = self.segs_raw.min(axis=0)
            segs_max = self.segs_raw.max(axis=0)
            lat_min = min(segs_min[0], segs_min[2])
            lat_max = max(segs_max[0], segs_max[2])
            long_min = min(segs_min[1], segs_min[3])
            long_max = max(segs_max[1], segs_max[3])
            box_latlong = np.array(
                [lat_max, long_min, lat_min, long_max],
                dtype=np.float_
            )
            truncate = False
        else:
            box_latlong = np.asarray(box_latlong, dtype=np.float_)
            lat_min = min(box[0], box[2])
            lat_max = max(box[0], box[2])
            long_min = min(box[1], box[3])
            long_max = max(box[1], box[3])
            truncate = True

        # Shift (well actually rotate) the coordinates so that the South-West
        # corner of the box is at (0,0)
        translation = np.array([lat_min, long_min, lat_min, long_min],
            dtype=np.float_)
        segs = self.segs_raw - translation

        # Get an approximate radius at this location
        r = self._earth_center_dist((lat_min+lat_max)/2.) + params['altitude']

        # Now convert the lat-long into meters by simply multiplying by the
        # radius (planar region hypothesis)
        segs *= r
        box = (box_latlong - translation) * r

        # Compute the scaling values
        dx = (long_max-long_min)*r
        dy = (lat_max-lat_min)*r
        world_scaling = 1./max(dx, dy)
        x_max = dx * world_scaling
        y_max = dy * world_scaling
        scaling = {
            'world': world_scaling,
            'x_max': x_max,
            'y_max': y_max
        }

        # Latitude corresponds to the y-axis, and longitude to x-axis, so use
        # indexing to swap them
        indexing = np.array([1, 0, 3, 2])

        # Scale the segments and fix the indexing
        segs_scaled = segs[:,indexing] * world_scaling

        # Now correct the y-axis orientation, we want it to be the distance
        # from the North side of the truncate_rect instead of the South side
        # Flip the orientation
        rev_y_axis = np.array([1., -1., 1., -1.], dtype=np.float_)
        # Shift the segments
        shift_y_axis = np.array([0., y_max, 0., y_max], dtype=np.float_)
        segs_cond = segs_scaled * rev_y_axis + shift_y_axis

        return geom.Segments(segs_cond), translation, scaling, truncate

    def _earth_center_dist(self, lat):
        """Approximate distance from sea-level to the Earth's center.

        Uses an ellipsoid shape model for the Earth.

        Args:
            lat: the latitude, in degrees
        """
        a = self.EQUATORIAL_RADIUS
        b = self.POLAR_RADIUS
        phi = math.pi * lat / 180.
        top = (a**2. * math.cos(phi))**2.
        top += (b**2. * math.sin(phi))**2.
        bot = (a * math.cos(phi))**2.
        bot += (b * math.sin(phi))**2.
        r = math.sqrt(top/bot)
        return r

