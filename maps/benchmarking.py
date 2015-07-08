# encoding=UTF-8

"""Utilities for benchmarking."""

# stdlib
import random

# 3p
import numpy as np

# project
import maps.discretemap
import geom


class MapFuzzer(object):
    """Apply forms of noise (coordinate noise, add/remove segments) to a map.

    This class is intended at providing the means to benchmark the robustness to
    noise, extra/missing segments of the map matching process.

    An instance of this class is given as input the normal map, and can then be
    asked to produce noisy versions of it, using different parameters.

    Noise forms:
        jitter: the endpoints of the road segments are changed by adding noise
            following an uncorrelated bivariate normal distribution, of
            mean 0, and standard deviation specified by the user. This is
            intended to model the imprecisions of endpoint detection.

        inaccuracy: the segments (i.e. both endpoints simultaneously) are moved
            a bit, following an uncorrelated bivariate normal distribution, of
            mean 0, and standard deviation specified by the user. This is
            intended to model the imprecisions in the printed map.

        extra segments: extra segments are added to the map. The first endpoint
            is chosen amongst the image pixels randomly (uniformly). The length
            is chosen next, using a normal distribution with specified mean and
            standard deviation. The second endpoint is chosen randomly on the
            circle of radius the chosen length, centered on the chosen first
            endpoint. The number of added segments is chosen as a proportion of
            the total number of segments in the map.

        missing segments: some segments are removed from the map. A proportion
            of the existing segments is chosen to be removed, and the actual
            ones that will be removed are chosen randomly.
    """

    def __init__(self, dmap):
        """Initializes the fuzzer.

        Args:
            dmap: the original DiscreteMap we'll apply transformations on
        """
        self.dmap = dmap

    def get_fuzzed(self, jitter=0., inaccuracy=0., missing=0., extra=0.,
            extra_length=40., extra_std=20.):
        """Create a fuzzed map given some parameters.

        See the class documentation for the detailed explanation of each kind of
        noise.

        Args:
            jitter: the point jitter applied to road segment endpoints. A float,
                the standard deviation in pixels of the jitter distribution.
            inaccuracy: the inaccurate position of roads on the map. A float,
                the standard deviation in pixels of the displacement.
            missing: proportion of removed segments, in [0,1]. 0 means remove no
                segments, 1 means remove all of them.
            extra: proportion of added segments, >= 0. A value of 1.5 means add
                1.5 times the total number of segments in the original map.
            extra_length: the mean length of the added segments
            extra_std: the standard deviation of the distribution of lengths of
                the added segments

        Returns:
            A noisy DiscreteMap, fuzzed using the provided parameters.
        """
        segs = self.dmap.roads

        new_segs = np.copy(segs.lines)

        # Remove segments before applying jitter
        if missing > 0.:
            remove_count = min(segs.count(), int(round(missing * segs.count())))
            idx_to_remove = random.sample(xrange(segs.count()), remove_count)
            keep_mask = np.full(segs.count(), True, dtype=np.bool_)
            keep_mask[idx_to_remove] = False

            new_segs = new_segs[keep_mask]

        # Add jitter
        if jitter > 0.:
            # Compute the noise for each coordinate of each segment
            n = np.random.normal(
                loc=0., scale=jitter, size=new_segs.shape
            )
            new_segs += n

        # Move segments around
        if inaccuracy > 0.:
            # Compute the displacement for each segment
            d = np.random.normal(
                loc=0., scale=inaccuracy, size=(new_segs.shape[0],2)
            )
            # The displacement should be the same for both endpoints
            d = np.hstack((d,d))
            new_segs += d

        # Add segments
        if extra > 0.:
            assert extra_length>0, "Added segment length must be positive."
            assert extra_std>=0, "Added segment length std must be positive."

            # How many segments to add?
            add_count = int(round(extra * segs.count()))

            # Compute the first endpoints of all the segments
            starts_x = np.random.uniform(
                low=0., high=self.dmap.img.shape[1], size=add_count
            )
            starts_y = np.random.uniform(
                low=0., high=self.dmap.img.shape[0], size=add_count
            )

            # Compute the length of each segment
            if extra_std > 0.:
                lengths = np.random.normal(
                    loc=extra_length, scale=extra_std, size=add_count
                )
            else: # std == 0
                lengths = np.full(add_count, extra_length, dtype=np.float_)

            # Determine the last endpoint by taking a random angle
            angles = np.random.uniform(
                low=0., high=2.*np.pi, size=add_count
            )

            # Compute the second endpoint by going in the direction of angles,
            # by lengths.
            ends_x = starts_x + lengths * np.cos(angles)
            ends_y = starts_y + lengths * np.sin(angles)

            # Form the segments array
            added_segs = np.vstack(
                (starts_x, starts_y, ends_x, ends_y)
            ).T

            # Add them to the existing ones
            new_segs = np.vstack((new_segs, added_segs))

        new_segs = geom.Segments(new_segs)

        return maps.discretemap.DiscreteMap(
            [0,0,self.dmap.img.shape[1],self.dmap.img.shape[0]], new_segs
        )

