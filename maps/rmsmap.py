# encoding=UTF-8

# stdlib
import logging
import math

# 3p
import numpy as np

# project
import datamanager
import maps.density
import maps.discretemap
import maps.keypoint
import maps.lrpd
import util.grid
import util.progressmeter


log = logging.getLogger(__name__)


class RMSMap(object):
    """Rotational Multi-Scale Map.

    Used to represent a map at multiple scales, and compute rotation invariant
    descriptors. The multiple scales of the map (the octaves) are represented
    by downsampled DiscreteMaps.

    Scale:
        The original segments are considered to have a scale of 1. When the
        description region radius is multiplied by f, the description scale is
        too.
        For simplicity, the only scales we'll consider are powers of 2:
        starting from the original image, we'll successively downsample by a
        factor 2.

    Description region:
        At each scale s, several descriptor region sizes are used, to describe
        more accurately the intermediate scales.
        For each scale, we'll use radii r in [rmin; rmax[, where rmax = 2*rmin
        (doubling of the scale), with k values. To have a nicely increasing
        radius, the step is 2^(1/k).
        For e.g., r in [15, 19, 24], rmin=15, k=3.

    Starting/Ending scale and region size:
        We have no a priori knowledge of what minimum scale to use to describe
        the map. We therefore rely on a heuristic to determine it: we use the
        discrete map density (maps.density module) to obtain an estimate of a
        good starting value.
        For images of size 500-1500px, a density of 5% seems pretty reasonable.
        We thus compute the size corresponding roughly to a 5% density, which
        allows us to determine the starting scale and descriptor size.

        Similarly, we use another threshold for the maximum density. 15-20%
        seems decent.

    Class attrs:
        LRPD_CONFIG: default parameters of the LRPD descriptor, which can be
            overridden at initialization. Contains:
            - bins: number of bins for LRPD
            - std: standard deviation of LRPD
            - N: the size of the histogram grid to use
            - min_segs: the minimum number of segments for a region to be
                significant

    Attrs:
        dmap_origin: the source DiscreteMap
        dmaps: list of DiscreteMaps at different scales

        scales: list of scales of the dmaps
        smin_idx: index of the first scale to use (i means scale 2**i)
        smax_idx: index of the last (inclusive) scale to use

        valid_radii: list of possibly used radii for the descriptors
        radii: list of intra-scale radii to use for the descriptor. Ie, for a
            scale scales[i], radii[i] is a list of radii used at this scale.
            Note that given the method used to determine the minimum and
            maximum description scales, all the (radii[i])_i may not be the same
            length.
        rmin_idx: in the first scale, smallest descriptor radius to use (as an
            index of valid_radii)
        rmax_idx: in the last scale, largest descriptor radius to use (as an
            index of valid_radii)

        lrpd_config: parameters for the LRPD descriptor
        lrpds: list of LRPD instances, corresponding to the radii in valid_radii
        grids: list of lists of Grid objects, of the same shape as radii.
            Contains the descriptor grid used for each scale, for each radius
        descs: numpy array of shape (N, desc_size), where N is the total number
            of descriptors computed over all scales and radii, and desc_size
            is the size of the descriptor vectors
        descs_revidx: list, mapping row indices of the descs array to a 3-tuple:
            (i_s, i_r, grid_idx), where i_s and i_r are the corresponding scale
            and radius indices of the descriptor (ie radii[i_s][i_r]), and
            grid_idx is the linear index of the descriptor in the grid at this
            (scale, radius)
        angles: numpy vector of the same size as that of the descriptors, which
            stores the reference angle that was determined along with each
            descriptor
    """

    LRPD_CONFIG = {
        'bins': 9,
        'cap': 0.15,
        'N': 4,
        'min_segs': 10
    }

    GRID_OFFSET_FACTOR = 0.7

    REF_ANGLE = None

    DEBUG = False

    @classmethod
    def from_density(cls, dmap, density_min=0.05, density_max=0.2,
            radius_desc_min=15, radius_desc_k=3,
            lrpd_config=None):
        """Create a RMSMap by autotuning the scales using road density.

        Instead of specifying explicitly the smallest and biggest description
        radii, this computes their values by using the road density measure,
        and thresholds on minimum and maximum densities.

        Args:
            dmap: the DiscreteMap to process, of scale 1.0
            density_min: minimum density (in [0,1]) to use as a starting scale
            density_max: maximum density (in [0,1]) to use as a last scale
            radius_desc_min: minimal descriptor radius to be used at each scale
            radius_desc_k: number of intermediate descriptor radii in
                [radius_min; 2*radius_min[ (including min, excluding max)
            lrpd_config: (optional) override class default LRPD descriptor
                parameters (see maps.lrpd module for details on them)
        """
        # Estimate the minimum and maximum radii
        # Factor 0.5 is because the optimal size is a diameter, not a radius
        r_g_min = 0.5 * maps.density.find_optimal_size(
            dmap, target=density_min, smin=radius_desc_min
        )
        r_g_max = 0.5 * maps.density.find_optimal_size(
            dmap, target=density_max, smin=2.*radius_desc_min
        )

        return cls(
            dmap, r_g_min, r_g_max,
            radius_desc_min=radius_desc_min, radius_desc_k=radius_desc_k,
            lrpd_config=lrpd_config
        )

    @classmethod
    def from_metric_scale(cls, mmap,
            metric_radius_min=50., metric_radius_max=500.,
            radius_desc_min=15, radius_desc_k=3,
            lrpd_config=None):
        """Create a RMSMap by defining the scales using metric data.

        Instead of specifying pixel radii as minimum and maximum description
        sizes, this simply uses the metric information of a MetricMap to
        determine the sizes from metric minimum and maximum values.

        Args:
            mmap: a MetricMap to process
            metric_radius_min: minimum metric description radius
            metric_radius_max: maximum metric description radius
            radius_desc_min: minimal descriptor radius to be used at each scale
            radius_desc_k: number of intermediate descriptor radii in
                [radius_min; 2*radius_min[ (including min, excluding max)
            lrpd_config: (optional) override class default LRPD descriptor
                parameters (see maps.lrpd module for details on them)
        """
        r_g_min = int(mmap.get_pixel_distance(metric_radius_min))
        r_g_max = int(mmap.get_pixel_distance(metric_radius_max))

        # Make sure both r_g_min and r_g_max are larger than radius_desc_min
        r_g_min = max(r_g_min, radius_desc_min)
        r_g_max = max(r_g_max, radius_desc_min)

        return cls(
            mmap, r_g_min, r_g_max,
            radius_desc_min=radius_desc_min, radius_desc_k=radius_desc_k,
            lrpd_config=lrpd_config
        )

    def __init__(self, dmap,
            radius_global_min, radius_global_max,
            radius_desc_min=15, radius_desc_k=3,
            lrpd_config=None, _skip_init=False):
        """Create the rotational multi-scale discrete map representation.

        Args:
            dmap: the initial, scale 1.0 discrete map
            radius_desc_min: minimal descriptor radius to be used at each scale
            radius_desc_k: number of intermediate descriptor radii in
                [radius_min; 2*radius_min[ (including min, excluding max)
            lrpd_config: (optional) override class default LRPD descriptor
                parameters (see maps.lrpd module for details on them)
            _skip_init: (internal use) if True, do not perform computations to
                initialize object state, it will be restored manually
        """
        # Merge the default and overridden configuration for the descriptor
        if lrpd_config is None:
            lrpd_config = {}

        self.lrpd_config = {}
        for k, v in self.LRPD_CONFIG.iteritems():
            self.lrpd_config[k] = lrpd_config.get(k, v)

        if not _skip_init:
            self._compute(
                dmap,
                radius_global_min, radius_global_max,
                radius_desc_min, radius_desc_k
            )

    def _compute(self, dmap,
            radius_global_min, radius_global_max,
            radius_desc_min=15, radius_desc_k=3):
        """Performs the multi-scale representation computations."""
        self.dmap_origin = dmap

        self.radius_min = int(radius_desc_min)
        self.radius_k = radius_desc_k
        self.radius_global_min = radius_global_min
        self.radius_global_max = radius_global_max

        self._compute_scale_parameters()

        self._create_descriptor_objects()

        self._create_dmap_pyramid()

        self._compute_descriptors()

    def _compute_scale_parameters(self):
        """Computes the scale-representation parameters (scales, radii, etc.)"""
        self.valid_radii = np.asarray(
            np.around(
                self.radius_min * np.logspace(
                    start=0, stop=self.radius_k,
                    num=self.radius_k, endpoint=False,
                    base=2.**(1./self.radius_k)
                )
            ),
            dtype=np.int_
        )

        scale_min_idx = int(math.floor(
            math.log(float(self.radius_global_min)/float(self.radius_min), 2.)
        ))
        r_start = 0
        for i, r in enumerate(self.valid_radii):
            if r * 2**scale_min_idx > self.radius_global_min:
                # If the first radius of this scale is too much, we messed
                # up in the computation of the scale_min_idx
                assert i>0, "Bug in the minimum scale computation!"
                r_start = i-1
        self.smin_idx = scale_min_idx
        self.rmin_idx = r_start

        scale_max_idx = int(math.floor(
            math.log(float(self.radius_global_max)/float(self.radius_min), 2.)
        ))
        # If the last radius of this scale is still not enough, we should have
        # taken the ceil() of the above, not the floor().
        if self.valid_radii[-1] * 2**scale_max_idx < self.radius_global_max:
            scale_max_idx += 1
        r_end = None
        for i, r in enumerate(self.valid_radii):
            if r * 2**scale_max_idx >= self.radius_global_max:
                r_end = i
        # If all the radii at this scale are less than radius_global_max, we messed
        # up in the computation of scale_max_idx, we should have gone a scale
        # higher
        assert r_end is not None, "Bug in the maximum scale computation!"
        self.smax_idx = scale_max_idx
        self.rmax_idx = r_end

    def _create_descriptor_objects(self):
        """Creates the different-sized descriptors we'll use."""
        self.lrpds = [None] * self.valid_radii.size
        for i, r in enumerate(self.valid_radii):
            self.lrpds[i] = maps.lrpd.LRPD(r, **self.lrpd_config)

    def _create_dmap_pyramid(self):
        """Create the downsampling dmap pyramid and the descriptor grids."""
        # Init member variables
        self.dmaps = []
        self.scales = [2**s for s in xrange(self.smin_idx, self.smax_idx+1)]
        self.radii = [[] for s in self.scales]
        self.grids = [[] for s in self.scales]

        # Compute the discrete maps and the grids
        curr_dmap = self.dmap_origin
        curr_s_idx = 0
        for i, s_idx in enumerate(xrange(self.smin_idx, self.smax_idx+1)):
            curr_dmap = maps.discretemap.DiscreteMap.from_downsampling(
                curr_dmap, repeat=s_idx-curr_s_idx
            )
            curr_s_idx = s_idx
            self.dmaps.append(curr_dmap)

            r_start = 0
            if s_idx == self.smin_idx:
                r_start = self.rmin_idx

            r_end = self.valid_radii.size-1
            if s_idx == self.smax_idx:
                r_end = self.rmax_idx

            # For each radius at this scale, compute the grid
            for r_idx in xrange(r_start, r_end+1):
                r = self.valid_radii[r_idx]
                self.radii[i].append(r_idx)
                # Each grid starts at an offset equal to the descriptor radius
                # multiplied by the GRID_OFFSET_FACTOR. This factor is used
                # because since the descriptors use a central Gaussian
                # weighting, so it is unnecessary to start at an offset of 1
                # entire radius.
                # We shouldn't choose the factor too low however, we don't want
                # to assume that the absence of data is some sort of
                # information.
                offset = int(r*self.GRID_OFFSET_FACTOR)
                self.grids[i].append(
                    util.grid.Grid.by_spacing(
                        (
                            0,
                            0,
                            curr_dmap.img.shape[1]-offset,
                            curr_dmap.img.shape[0]-offset
                        ),
                        spacing=r, start_offset=offset
                    )
                )

    def _compute_descriptors(self):
        """Computes all the descriptors, for each scale and each grid cell."""
        # Compute the problem size, in number of descriptors to compute
        # Note that this is an upper bound on the number of descriptors. There
        # is no telling how many will be judged not significant
        total_descs = 0
        for i_s in xrange(len(self.grids)):
            for i_r in xrange(len(self.grids[i_s])):
                total_descs += self.grids[i_s][i_r].size()

        # Initialize the progress meter
        if self.DEBUG:
            pm = util.progressmeter.ProgressMeter(total_descs, dot_every=0)

        # Finally, compute the descriptors and the reverse mapping for the
        #indices
        self.descs = np.empty([total_descs, self.lrpds[0].size()],
            dtype=np.float_
        )
        self.angles = np.empty(total_descs, dtype=np.float_)
        self.descs_revidx = [None] * total_descs
        descs_pos = 0

        if self.DEBUG:
            pm.start()
        for i_s in xrange(len(self.scales)):
            for i_r, r_idx in enumerate(self.radii[i_s]):
                r = self.valid_radii[r_idx]

                for r, y, c, x in self.grids[i_s][i_r].iter_all(cast_int=True):
                    # Compute the reference angle and the descriptor vector
                    pair = self.lrpds[r_idx].describe(
                        self.dmaps[i_s], (x,y), ref_angle=self.REF_ANGLE
                    )
                    if pair is not None:
                        self.angles[descs_pos] = pair[0]
                        self.descs[descs_pos, :] = pair[1]
                        self.descs_revidx[descs_pos] = (
                            i_s, i_r, self.grids[i_s][i_r].to_lin(r, c)
                        )
                        descs_pos += 1
                    if self.DEBUG:
                        pm.incr()

        if self.DEBUG:
            pm.end()

        # Truncate the angles array
        self.angles = self.angles[:descs_pos]
        # Truncate the descs array
        self.descs = self.descs[:descs_pos, :]
        # Truncate the reverse index
        self.descs_revidx = self.descs_revidx[:descs_pos]

    def get_desc_scale(self, desc_idx, cast_int=False):
        """Returns the scale at which a given descriptor was computed.

        Args:
            desc_idx: row index of the descriptor in the descs array
            cast_int: if True, the result scale will be cast to an int

        Returns:
            A float representing the scale of the descriptor: the pixel size of
            the equivalent descriptor radius in the original image. I.e for a
            description radius r (eg 15px) at scale s (eg 4.0), the global
            scale is simply r*s.
        """
        i_s, i_r, _ = self.descs_revidx[desc_idx]
        s = self.valid_radii[self.radii[i_s][i_r]] * self.scales[i_s]
        if cast_int:
            s = int(round(s))
        return s

    def get_desc_location(self, desc_idx, cast_int=False):
        """Returns the location of the given descriptor at the original scale.

        Args:
            desc_idx: row index of the descriptor in the descs array
            cast_int: if True, the result coordinates will be cast to ints

        Returns:
            The (x,y) location of the descriptor in the original DiscreteMap,
            ie if the descriptor was computed at another scale, its position
            in the original image will be computed.
        """
        i_s, i_r, lin_idx = self.descs_revidx[desc_idx]
        x_s, y_s = self.grids[i_s][i_r].from_lin(lin_idx, spatial=True)
        scale = self.scales[i_s]
        x, y = x_s * scale, y_s * scale
        if cast_int:
            x, y = int(round(x)), int(round(y))
        return x, y

    def get_desc_angle(self, desc_idx):
        """Returns the reference angle of a computed descriptor.

        Args:
            desc_idx: row index of the descriptor in the descs array

        Returns:
            A float representing the angle (in radians) that was used as a
            reference for the computation of the descriptor.
        """
        return self.angles[desc_idx]

    def get_desc_keypoint(self, desc_idx, cast_int=False):
        """Returns a descriptor keypoint as a Keypoint object.

        Args:
            desc_idx: row index of the descriptor in the descs array
            cast_int: if True, the result coordinates and scale will be cast to
                ints

        Returns:
            A Keypoint instance representing the descriptor's keypoint, in
            the original DiscreteMap.
        """
        a = self.angles[desc_idx]
        i_s, i_r, lin_idx = self.descs_revidx[desc_idx]
        x_s, y_s = self.grids[i_s][i_r].from_lin(lin_idx, spatial=True)
        s = self.scales[i_s]
        kpscale = self.valid_radii[self.radii[i_s][i_r]] * s
        x, y = x_s * s, y_s * s

        return maps.keypoint.Keypoint(kpscale, o=a, x=x, y=y,
            cast_int=cast_int)

    def save_to_manager(self, path, _no_save=False):
        """Saves the object to external storage."""
        dm = datamanager.DataManager(path)
        meta = {
            'dmap_origin_path': path+'.dmap_origin',
            'dmaps_paths': [
                path+'.dmap{}'.format(i) for i in xrange(len(self.dmaps))
            ],
            'grids_paths': [
                [
                    path+'.grid{}_{}'.format(i, j)
                    for j in xrange(len(self.grids[i]))
                ] for i in xrange(len(self.grids))
            ],
            'lrpd_config': self.lrpd_config
        }
        data = {
            'scales': self.scales,
            'smin_idx': self.smin_idx,
            'smax_idx': self.smax_idx,
            'valid_radii': self.valid_radii,
            'radii': self.radii,
            'rmin_idx': self.rmin_idx,
            'rmax_idx': self.rmax_idx,
            'descs': self.descs,
            'descs_revidx': self.descs_revidx,
            'angles': self.angles
        }

        # Save the original dmap
        self.dmap_origin.save_to_manager(meta['dmap_origin_path'])
        # Save the scaled dmaps
        for i, dmap_path in enumerate(meta['dmaps_paths']):
            self.dmaps[i].save_to_manager(dmap_path)
        # Save the grids
        for i in xrange(len(self.grids)):
            for j in xrange(len(self.grids[i])):
                self.grids[i][j].save_to_manager(
                    meta['grids_paths'][i][j]
                )

        if _no_save:
            return dm, meta, data
        else:
            dm.set(metadata=meta, data=data)

    @classmethod
    def load_from_manager(cls, path, _return_dm=False):
        """Loads an object from external storage."""
        dm = datamanager.DataManager(path)
        meta, data = dm.get()
        inst = RMSMap(None, None, None, _skip_init=True)

        # Restore original dmap
        inst.dmap_origin = maps.discretemap.DiscreteMap.load_from_manager(
            meta['dmap_origin_path']
        )
        # Restore scaled dmaps
        inst.dmaps = []
        for dmap_path in meta['dmaps_paths']:
            inst.dmaps.append(maps.discretemap.DiscreteMap.load_from_manager(
                dmap_path
            ))
        # Restore grids
        inst.grids = []
        for i, grids in enumerate(meta['grids_paths']):
            inst.grids.append([])
            for j, grid_path in enumerate(grids):
                inst.grids[-1].append(util.grid.Grid.load_from_manager(
                    grid_path
                ))
        # Restore lrpd_config
        inst.lrpd_config = meta['lrpd_config']
        # Restore all data
        inst.scales = data['scales']
        inst.smin_idx = data['smin_idx']
        inst.smax_idx = data['smax_idx']
        inst.valid_radii = data['valid_radii']
        inst.radii = data['radii']
        inst.rmin_idx = data['rmin_idx']
        inst.rmax_idx = data['rmax_idx']
        inst.descs = data['descs']
        inst.descs_revidx = data['descs_revidx']
        inst.angles = data['angles']

        if _return_dm:
            return dm, inst, meta, data
        else:
            return inst

