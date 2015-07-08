# encoding=UTF-8

# stdlib
import itertools
import json
import logging

# 3p
import numpy as np

# project
import datamanager
import geom.planar as geom2d
import maps.gcs
import geom


log = logging.getLogger(__name__)


class MapProcessor(object):
    """Pre-process raw data from a GIS.

    Given a configuration file, which specifies parameters such as the file
    containing the actual segment data, the Geographic Coordinate System (GCS)
    it uses, coordinates of a rectangle to clip the input data, it loads the
    segments, normalizes them (ie scales and translates them so they fit in
    [0,1]x[0,1], while keeping the aspect ratio).

    For details on the supported GCS, see maps.gcs.

    Configuration file:
        The configuration is a JSON document, of the form:
        {
            "file": "/path/to/the/actual/segment/data",
            "gcs": "GCS identifier, eg: osgb36",
            "gcs_params": {dictionary of parameters for the GCS, see maps.gcs}
        }
    """

    def __init__(self, filename, _skip_init=False):
        """Parse an segment data configuration file.

        Args:
            filename: file where the segment configuration is
            _skip_init: (internal use) do not process the segment data after
                parsing the configuration, it will be restored externally.
        """

        self.conffile = filename
        conf = {}
        with open(filename, 'rb') as conffile:
            conf = json.load(conffile)

        if not _skip_init:
            GCS = maps.gcs.gcs_from_format(conf['gcs'])

            with open(conf['file'], 'rb') as f:
                gcs = GCS(f)
                params = conf.get('gcs_params', {})
                self.roads, self.translation, self.scaling, truncate = \
                    gcs.condition(params=params)

                if truncate:
                    self._truncate_segs()

    def save_to_manager(self, path):
        """Save the internal state of this object to external storage."""
        dm = datamanager.DataManager(path)
        meta = {
            'conf': self.conffile,
        }
        data = {
            'roads': self.roads.lines,
            'translation': self.translation,
            'scaling': self.scaling
        }

        dm.set(metadata=meta, data=data)

    @classmethod
    def load_from_manager(cls, path):
        """Create an instance of this class by loading its data from files."""
        dm = datamanager.DataManager(path)
        meta, data = dm.get()

        inst = cls(meta['conf'], _skip_init=True)

        inst.roads = geom.Segments(data['roads'])
        inst.translation = data['translation']
        inst.scaling = data['scaling']

        return inst

    def _truncate_segs(self):
        log.debug('Truncating segments to selection rectangle.')

        # Since the scaling is uniform, we need to determine the scaled coordinates
        # of the truncation rectangle
        x_max = self.scaling['x_max']
        y_max = self.scaling['y_max']

        total_segs = self.roads.count()

        self.roads = geom.Segments(
            geom2d.intersect_segments_with_rect(
                (0.,0.,x_max,y_max), self.roads.lines
            )
        )

        final_segs = self.roads.count()

        log.debug('{} segments out of {} are in bounds.'\
            .format(final_segs, total_segs))

