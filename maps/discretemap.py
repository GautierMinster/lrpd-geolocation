# encoding=UTF-8

# stdlib
import itertools
import logging
import math

# 3p
import numpy as np

# project
import datamanager
import maps.labeller
import geom
import util.image


log = logging.getLogger(__name__)


class DiscreteMap(object):
    """A finite precision raster map (ie an image) and segment containers.

    This is a container for a raster map, represented by an image of multisets.

    Attrs:
        roads: the road segments, as a geom.Segments object
        img: the image of labels (multisets)
        labeller: the object used to assign and store the multisets. The
            elements of the multisets are segment indices (indices in the
            roads attribute). An instance of maps.labeller.MultiSetLabeller()
        name: an optional name for the map (convenient when saving the
            DiscreteMap with the DataManager)
    """

    def __init__(self, region, roads, name=None, _skip_init=False):
        """Creates the discrete map.

        The roads are drawn onto an image of the same dimensions as region.
        This means high-precision floating point roads will be drawn onto
        pixels.

        Args:
            region: a (xmin, ymin, xmax, ymax) 4-uplet, the region considered.
                This will determine the size of the image generated.
            roads: a geom.Segments object containing the roads
            name: a user-friendly name to identify the map
            _skip_init: internal parameter used when the internal state of the
                object is going to be set manually
        """
        self.name = name

        if _skip_init:
            return

        # Create the image, draw each segment on it while keeping track for
        # each pixel of which segments appeared there
        self.img = np.full((region[3]-region[1], region[2]-region[0]), 0,
            dtype=np.uint32)

        # Create a labeller for the pixels (the classes are the segments)
        self.labeller = maps.labeller.MultiSetLabeller()

        self.roads = geom.Segments(roads.lines - \
            np.array(
                [region[0], region[1], region[0], region[1]],
                dtype=np.float_
            )
        )

        for idx, s in enumerate(self.roads):
            # Trace the line
            s_int = map(int, map(round, s))
            for x, y in util.image.line_pixels(s_int):
                if x<0 or x>=self.img.shape[1] or y<0 or y>=self.img.shape[0]:
                    continue
                cur_label = self.img[y,x]
                # If cur_label is 0, this means there was nothing there
                # We add a single element, the current segment index, to
                # whatever was at at pixel. We use the segment index as a uid,
                # to always obtain the same new_label for a given (cur_label,
                # idx) pair
                new_label = self.labeller.add([(idx, 1)], l=cur_label, uid=idx)
                self.img[y,x] = new_label
        self.labeller.freeze()

    @classmethod
    def from_downsampling(cls, dmap_src, repeat=1):
        """Downsample by a power of 2 a DiscreteMap.

        The new segment labelling will be computed by merging squares of pixels
        (ie the labels of the source DiscreteMap).

        Args:
            dmap_src: the DiscreteMap to downsample
            repeat: number of times to downsample by a factor 2 (>=0)

        Returns:
            A new DiscreteMap, roughly 2**(-repeat) the size of dmap_src.
        """
        assert repeat>=0, "Can't downsample a negative amount of times"
        assert type(repeat)==int, \
            "Can only downsample an integer amount of times"

        if repeat == 0:
            return dmap_src

        # Scaling factor (int and float)
        f = 2.**repeat
        fi = 2**repeat

        # Downscale the roads, we'll need them for the final dmap
        roads_dst = geom.Segments(dmap_src.roads.lines/f)
        # Manually restore the angles, they haven't changed and we don't want to
        # lose precision
        roads_dst.angles_unorient = dmap_src.roads.get_angles_unoriented()

        img_src = dmap_src.img
        size_src = img_src.shape
        labeller_src = dmap_src.labeller

        # Create the new image
        size_dst = int(math.ceil(size_src[0]/f)), int(math.ceil(size_src[1]/f))
        img_dst = np.full(size_dst, 0, dtype=np.uint32)

        labeller_dst = maps.labeller.MultiSetLabeller()

        # Dictionary used to recognize identical batches of labels, and give
        # them a consistent UID
        uids = {}
        uid_count = 0

        # Iterate over each pixel of the destination image, so that we can
        # process the pixels of the source in square batches of
        # (2**repeat)x(2**repeat) pixels
        for y, x in itertools.product(xrange(size_dst[0]), xrange(size_dst[1])):
            # Get the corresponding source image region
            region = img_src[y*fi:(y+1)*fi, x*fi:(x+1)*fi]

            # List of labels found in the region, sorted so they can be hashed
            # and recognized. The sorting should be lightning fast, since we'll
            # only have a couple of elements
            labels = np.sort(region[region.nonzero()])
            # List of generators returned by the source labeller
            its = [labeller_src.get(l)[1] for l in labels]

            if len(its) == 0:
                continue

            # Compute the uid key, and see if it already exists
            key = tuple(labels)
            uid = None
            try:
                uid = uids[key]
            except KeyError:
                uids[key] = uid_count
                uid = uid_count
                uid_count += 1

            # Now create a single iterator from the iterators of each nonzero
            # pixel we processed
            elements = itertools.chain.from_iterable(its)

            # Feed this whole batch to the new labeller
            # Since we do this only once per pixel of the destination image, we
            # know the current label in the destination is 0
            new_label = labeller_dst.add(elements, uid=uid)
            img_dst[y,x] = new_label

        labeller_dst.freeze()

        dmap_dst = DiscreteMap(None, None, _skip_init=True)
        dmap_dst.img = img_dst
        dmap_dst.labeller = labeller_dst
        dmap_dst.roads = roads_dst

        return dmap_dst

    def save_to_manager(self, path, no_save=False):
        """Saves the object to external storage."""
        dm = datamanager.DataManager(path)
        meta = {
            'discrete_map_type': 'DiscreteMap',
            'name': self.name
        }
        data = {
            'img': self.img,
            'roads': self.roads.lines,
            'labeller': self.labeller.to_dict()
        }
        if no_save:
            return dm, meta, data
        else:
            dm.set(metadata=meta, data=data)

    @classmethod
    def load_from_manager(cls, path, _return_dm=False):
        """Create an instance of this class by loading its data externally."""
        dm = datamanager.DataManager(path)
        meta, data = dm.get()

        if not _return_dm and meta['discrete_map_type'] != 'DiscreteMap':
            log.warn("Loading data of type {}, not 'DiscreteMap'.".format(
                meta['discrete_map_type']
            ))

        roads = geom.Segments(data['roads'])

        inst = DiscreteMap(None, None, name=meta.get('name', None), _skip_init=True)
        inst.img = data['img']
        inst.roads = roads
        inst.labeller = maps.labeller.MultiSetLabeller.from_dict(data['labeller'])

        if _return_dm:
            return dm, inst, meta, data
        else:
            return inst

