# encoding=UTF-8

# stdlib
import logging

# 3p
import numpy as np

# project
import maps.discretemap


log = logging.getLogger(__name__)


class MetricMap(maps.discretemap.DiscreteMap):
    """Represents a DiscreteMap, extended with metric scale information.

    Attrs: (added to DiscreteMap)
        scale: the ratio of world units over pixel units
    """

    def __init__(self, region, roads, distance_pixels, distance_world, name=None):
        """Initializes the map data and scale.

        Args:
            region, roads, name: see DiscreteMap
            distance_pixels: the distance in pixels that corresponds to
                distance_world
            distance_world: the distance in world units (eg meters) that
                corresponds to distance_pixels
        """
        super(MetricMap, self).__init__(region, roads, name=name)
        self.scale = float(distance_world) / float(distance_pixels)

    def save_to_manager(self, path, no_save=False):
        """Saves the object to external storage."""
        dm, meta, data = super(MetricMap, self).save_to_manager(
            path, no_save=True
        )
        meta['discrete_map_type'] = 'MetricMap'
        data['scale'] = self.scale
        if no_save:
            return dm, meta, data
        else:
            dm.set(metadata=meta, data=data)

    @classmethod
    def load_from_manager(cls, path, _return_dm=False):
        """Create an instance of this class by loading its data externally."""
        dm, inst, meta, data = super(MetricMap, cls).load_from_manager(path,
            _return_dm=True)

        if not _return_dm and meta['discrete_map_type'] != 'MetricMap':
            log.warn("Loading data of type {}, not 'MetricMap'.".format(
                meta['discrete_map_type']
            ))

        inst.__class__ = MetricMap
        inst.scale = data['scale']

        if _return_dm:
            return dm, inst, meta, data
        else:
            return inst

    def get_metric_distance(self, pt1, pt2=None):
        """Returns the metric distance corresponding to a pixel distance.

        If pt2 is None, then pt1 can be a vector or a scalar. Otherwise, both
        pt1 and pt2 are vectors.
        """
        if pt2 is not None:
            pt1 = map(float, pt1)
            pt2 = map(float, pt2)
            d = ((pt2[0]-pt1[0])**2. + (pt2[1]-pt1[1])**2.)**.5
        elif type(pt1) in [int, float, np.float64, np.float32]:
            d = float(pt1)
        else:
            pt1 = map(float, pt1)
            d = (pt1[0]**2. + pt1[1]**2.)**.5

        # Distance in world units
        d_wu = d * self.scale
        return d_wu

    def get_pixel_distance(self, pt1, pt2=None):
        """Returns the pixel distance corresponding to a metric distance.

        If pt2 is None, then pt1 can be a vector or a scalar. Otherwise, both
        pt1 and pt2 are vectors.
        """
        if pt2 is not None:
            pt1 = map(float, pt1)
            pt2 = map(float, pt2)
            d = ((pt2[0]-pt1[0])**2. + (pt2[1]-pt1[1])**2.)**.5
        elif type(pt1) in [int, float, np.float64, np.float32]:
            d = float(pt1)
        else:
            pt1 = map(float, pt1)
            d = (pt1[0]**2. + pt1[1]**2.)**.5

        # Distance in pixels
        d_px = d / self.scale
        return d_px

