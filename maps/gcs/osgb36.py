# encoding=UTF-8

# stdlib
import logging
import re

# 3p
import numpy as np

# project
from maps.gcs.gcs import GCS
import geom


log = logging.getLogger(__name__)


class GCS_OSGB36(GCS):
    """OSGB36 Geographic Coordinate System.

    OSGB36 GCS:
        NOTE: this is a simple explanation, if unsure, look it up on the
        internet.

        The OSGB36 (Ordnance Survey Great Britain 1936) splits Great Britain
        into a grid, with the origin of the coordinate frame at the South-west
        corner.
        The coordinates are named Eastings and Northings, ie the distance to
        East and to the North of the point considered, from the South-West
        corner of the grid cell. The distance is expressed in meters.
        In our case, we use one of the allowed representations, which simply
        expresses the Eastings and Northings as a the distance in meters from
        the origin of the entire grid, instead of the distance from the cell
        origin.
    """

   # Regexps to parse OSGB-36 segments data
    number_pat = r'''
            (?P<{}>
                \d+
                \.? # Optional decimal separator
                (?:\d+)? # Optional additional digits
            )
    '''
    pat = re.compile(
        ''.join([
            r'\A \s* (?:LINESTRING)? \s*', # Allow a leading LINESTRING (WKT files)
            r'\( ',
                number_pat.format('eastings1'),
                r'\s+',
                number_pat.format('northings1'),
                r', \s+',
                number_pat.format('eastings2'),
                r'\s+',
                number_pat.format('northings2'),
            r'\)'
        ]),
        re.VERBOSE
    )

    def __init__(self, data_iterator):
        log.debug('Loading segments in OSGB36 format')
        coords = []
        for line in data_iterator:
            m = self.pat.match(line)
            if not m:
                continue
            d = m.groupdict()
            coords.append((
                float(d['eastings1']),
                float(d['northings1']),
                float(d['eastings2']),
                float(d['northings2'])
            ))

        self.segs_raw = np.asarray(coords, dtype=np.float_)

    def condition(self, params={}):
        """Condition the raw data to a local coordinate frame.

        Since the OSGB36 data is already in meters, all we need to do is
        shift the segments, scale them, and make sure the axes are correctly
        oriented.

        Args:
            params: additional information for the operation, a dict containing
                (optionally):
                - truncate_rect: a list of 4 floats, the first 2 are the
                  coordinates (eastings, northings) of the top left corner, the
                  last 2 of the bottom right corner. If not specified, the
                  bounding box of the segments will be used instead.

        Returns:
            See the documentation of GCS.condition().
        """
        box = params.get('truncate_rect')

        if box is None:
            segs_min = self.segs_raw.min(axis=0)
            segs_max = self.segs_raw.max(axis=0)
            e_min = min(segs_min[0], segs_min[2])
            e_max = max(segs_max[0], segs_max[2])
            n_min = min(segs_min[1], segs_min[3])
            n_max = max(segs_max[1], segs_max[3])
            box = np.array([e_min, n_max, e_max, n_min], dtype=np.float_)
            truncate = False
        else:
            box = np.asarray(box, dtype=np.float_)
            e_min = min(box[0], box[2])
            e_max = max(box[0], box[2])
            n_min = min(box[1], box[3])
            n_max = max(box[1], box[3])
            truncate = True

        # Shift the origin, the South-West corner should become (0,0)
        translation = np.array([e_min, n_min, e_min, n_min])

        # Compute the scaling values
        de = float(e_max - e_min)
        dn = float(n_max - n_min)
        world_scaling = 1./max(de, dn)
        x_max = de * world_scaling
        y_max = dn * world_scaling
        scaling = {
            'world': world_scaling,
            'x_max': x_max,
            'y_max': y_max
        }

        # Compute the vector we'll use to scale each segment
        scaling_vec = np.full(4, world_scaling, dtype=np.float_)

        # Apply the translation and scaling
        segs_scaled = (self.segs_raw - translation) * scaling_vec

        # Now correct the y-axis orientation, we want it to be the distance
        # from the North side of the truncate_rect, not the distance from
        # South side.
        # Flip the orientation
        rev_y_axis = np.array([1., -1., 1., -1.], dtype=np.float_)
        # Shift the segments
        shift_y_axis = np.array([0., y_max, 0., y_max], dtype=np.float_)
        segs_cond = segs_scaled * rev_y_axis + shift_y_axis

        return geom.Segments(segs_cond), translation, scaling, truncate

