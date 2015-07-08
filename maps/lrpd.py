# encoding=UTF-8

# stdlib
import itertools
import math
import logging

# 3p
import numpy as np
import scipy.stats


log = logging.getLogger(__name__)


class LRPD(object):
    """Local Road Pattern Descriptor.

    Descriptor that computes a NxN set of histograms in the descriptor region
    (following the concept of the SIFT descriptor).

    As in SIFT, each pixel in the descriptor region contributes to 4 histograms
    (ignoring boundary pixels, which may contribute to 2 for sides and 1 for
    corners). Let D be the distance between two histogram region centers (ie
    D = radius / N). For a given pixel, the weight to give a histogram at
    distance (dx,dy) is w = (1-dx/D)*(1-dy/D). This way, when a pixel is in the
    middle of two histograms, d=D/2, and the weight along this dimension is 0.5.
    (and thus for a perfectly centered pixel, each one of the 4 histograms
    would obtain a weight of 1/2 * 1/2 = 1/4.

    Here is how we go about making the descriptor rotation invariant:
      - before computing the array of histograms, a single global histogram is
        computed. The maximum peak of this global histogram is used to determine
        the prevailing road orientation in the described area. All of the NxN
        histograms are then computed rotated by this orientation. This means
        that not only are the bins cyclically shifted, but also that the
        histogram grid takes the rotation into account.
      - one caveat of computing a rotationally invariant histogram descriptor of
        unoriented angles (ie in [0,pi[), is that should the described region be
        rotated by pi, the global histogram is left unchanged. We therefore
        can't tell if the reference angle should be the global histogram's
        maximum, or if it should be the maximum plus pi. The decision is pretty
        much crucial, since the order of the histogram grid cells depends on it.
        Possible solutions to resolve this ambiguity include:
        - returning two descriptors, one for each of the two possible reference
          angles. This would double the number of descriptors, which might not
          be too good.
        - making the array of histograms invariant to rotations by pi, by
          considering only one half of the descriptor region (ie, folding the
          descriptor). This sacrifices some discriminative power of the
          descriptor in exchange for solving the ambiguity by making it
          pi-rotation invariant.
        - devising a way to choose, after setting the new reference angle to a
          value in [0, pi[, which half, upper or lower, should be the top one.

        We'll use this last option: after the histogram array computation, once
        the only thing left to do is decide to reverse or not the descriptor,
        we'll sum the 0th bins in each half of the histogram array (upper and
        lower), and see which one has the highest value. If it's the upper
        half, great. If it's the lower, we'll just rotate the array (only the
        grid, the bins in each histogram won't change) by pi (which just means
        performing some grid cell swapping).
    """

    def __init__(self, radius, N=4, bins=9, cap=1., min_segs=10):
        """Initializes the parameters of the descriptor.

        Args:
            radius: radius of the descriptor region to consider
            N: the descriptor region will be split in an NxN grid of histograms.
                N has to be even.
            bins: number of bins of the histograms (subdivision of [0,pi[)
            cap: cap on the histogram bin values; a value outside [0,1[ means
                disable it.
            min_segs: the minimum number of contributing segments for a
                descriptor to be considered meaningful. Setting to any value
                lower or equal to 1 disables the check
        """
        self.r = int(radius)
        self.K = bins
        self.cap = cap

        # Histogram grid size
        assert N%2==0, "Can only handle even grid sizes."
        self.N = N

        # Minimum number of contributing segments for meaningfulness
        self.min_segs = min_segs

        # To obtain a smooth descriptor, each pixel should contribute to the 4
        # histograms whose centers surround it
        # Distance between two cell centers:
        self.D = 2*self.r // self.N
        reg_size = 2*self.r
        D_f = float(self.D)

        # Since we know the described region size, we can precompute the
        # weights for the spatial interpolation, as well as which grid
        # cells to contribute to
        # Note that a fake line and a fake row will be added when computing the
        # actual grid of histograms, so that we never have to worry about bounds
        # when computing indices

        # Get the precise coordinates (in histogram grid units)
        coord_grid = np.arange(reg_size, dtype=np.float_)/D_f - 0.5
        # Get the coordinates of the bottom right histogram's region center
        coord_cell_br = np.ceil(coord_grid)
        # Compute the weight assigned to the bottom right histogram
        self.w_grid = 1. - (coord_cell_br - coord_grid)
        # Compute the index of the bottom right histogram in the grid
        self.cell_br_idx = np.asarray(coord_cell_br, dtype=np.int_)

        self._compute_central_gaussian(self.r+1, float(self.r))

        # TODO: this is convenient to see the effect of the choice of the final
        #       orientation, ref_angle / ref_angle+pi:
        #       we can observe the distribution of ratios, to see how pertinent
        #       a criterion seems to be.
        #       We should investigate more methods of choosing.
        self.chosen_half = {'keep': 0, 'switch': 0, 'discard': 0, 'ratio': []}

    def size(self):
        """Returns the size of the descriptor.

        Returns:
            N^2 * bins
        """
        return self.N * self.N * self.K

    def describe(self, dmap, pt, dg=None, ref_angle=None):
        """Describe the point at pt.

        Args:
            dmap: the DiscreteMap instance describing the pixel at point pt
            pt: point (x, y) to be described
            dg: if provided, must be a numpy K-vector (same number of bins as
                this descriptor). The global orientation histogram will be
                written to it
            ref_angle: if provided, will be used as the reference angle for the
                descriptor. In radians, in [-pi; pi[ (actually, anything works,
                the point is, it's an angle in the full circle, not just [0,pi[)

        Returns:
            If the described region is not meaningful (ie less than min_segs
            contributing segments), returns None.

            Otherwise, returns a (ref, desc) pair, where:
              - ref: the reference angle that was determined and used, in
                radians
              - desc: a numpy vector of np.float_, of size size(), the
                rotated descriptor
        """
        # Allocate the final descriptor
        d = np.zeros(self.size(), dtype=np.float_)
        # Allocate the global histogram we'll use to determine the main
        # orientation
        if dg is None:
            dg = np.zeros(self.K, dtype=np.float_)
        else:
            assert dg.shape == (self.K,), "Provided global histogram array must be coherent with descriptor."
            dg.fill(0.)
        # Get quick accessors
        hists = self._descriptor_vector_to_grid(d, dummy_ends=True)

        # Bounds of the descriptor region (not considering the actual image
        # bounds for now)
        xmin_reg, xmax_reg = pt[0]-self.r, pt[0]+self.r
        ymin_reg, ymax_reg = pt[1]-self.r, pt[1]+self.r

        # Intersection of the description region and the image bounds
        xmin = max(0, xmin_reg)
        xmax = min(dmap.img.shape[1], xmax_reg)
        ymin = max(0, ymin_reg)
        ymax = min(dmap.img.shape[0], ymax_reg)

        # Entire descriptor region
        region = dmap.img[ymin:ymax, xmin:xmax]

        # Histogram bin size (radians)
        bin_size = np.pi / np.float_(self.K)
        gbin_size = np.pi / np.float_(dg.size)

        # Get the segment orientations
        orients = dmap.roads.get_angles_unoriented()

        # Store whether the reference angle was overridden
        ref_angle_overridden = (ref_angle is not None)

        nonzero = np.transpose(region.nonzero())

        # If the whole region is zero, don't bother, return None
        if nonzero.shape[0] == 0:
            return None

        contributors = set()
        significant = False
        if self.min_segs <= 1:
            significant = True

        # First, create the global histogram, which will allow us to compute a
        # rotated descriptor
        # Even when the reference angle is specified, we still perform this step
        # since it allows us to count the contributing segments
        for y, x in nonzero:
            # Segments at this pixel
            total, segs_iterator = dmap.labeller.get(region[y,x])
            # Weight of the global Gaussian
            w_g = self.g_window[abs(x+xmin-pt[0]), abs(y+ymin-pt[1])] \
                  / float(total)

            for seg, count in segs_iterator:
                # Add the contribution of this segment
                if not significant:
                    contributors.add(seg)

                if ref_angle_overridden:
                    continue

                # Update the global histogram
                gbin_pos = orients[seg] / gbin_size
                lgbin_idx = int(gbin_pos) % dg.size
                ugbin_idx = (lgbin_idx+1) % dg.size
                w_go_u = gbin_pos % 1
                dg[lgbin_idx] += count * w_g * (1.-w_go_u)
                dg[ugbin_idx] += count * w_g * w_go_u

            if not significant and len(contributors) > self.min_segs:
                significant = True

        # If there's not enough contributors, abandon the computation
        if not significant:
            return None

        # Now determine the reference angle if needed
        if not ref_angle_overridden:
            ref_angle = self._determine_ref_angle(dg)
            if ref_angle is None:
                return None

        # Compute the rotation matrix:
        #   We want a rotation by an angle -ref_angle
        # The coordinates we have are of the form [[y x],[y x], [y x], ...], ie
        # each pixel is a row [y x]. So we need to tweak our rotation matrix to
        # be coherent with that:
        # Instead of R * [x, y].T, we switch x and y: R * [0 1] * [y, x].T
        #                                                 [1 0]
        # and then transpose: [y x] * [0 1] * R.T
        #                             [1 0]
        # So that the matrix M we want is M = [-sin(-ref), cos(-ref)]
        #                                     [ cos(-ref), sin(-ref)]
        # Which in turns simplifies to:
        #   M = [sin(ref),  cos(ref)]
        #       [cos(ref), -sin(ref)]
        # We'll obtain a matrix, where each row is [y_rot, x_rot]
        #
        # Note that we need a translation before the rotation, since the
        # coordinates in nonzero are not centered around the center of the
        # descriptor. We need to add xmin+pt[0] to the x coordinate of nonzero,
        # and a corresponding value to y.
        M = np.array([[math.sin(ref_angle),  math.cos(ref_angle)],
                      [math.cos(ref_angle), -math.sin(ref_angle)]],
                     dtype=np.float_)
        T = np.array([ymin-pt[1], xmin-pt[0]], dtype=np.float_)
        rotpixels = np.asarray(
            np.around((nonzero+T).dot(M)),
            dtype=np.int_
        )

        for (y, x), (yrot, xrot) in zip(nonzero, rotpixels):
            # Skip points that land outside the region
            if xrot<-self.r or xrot>=self.r or yrot<-self.r or yrot>=self.r:
                continue

            # Segments at this pixel
            total, segs_iterator = dmap.labeller.get(region[y,x])
            # Distance from the top left of the theoretical descriptor region
            x_reg, y_reg = xrot+self.r, yrot+self.r
            # Weight of the global Gaussian
            w_g = self.g_window[abs(x+xmin-pt[0]), abs(y+ymin-pt[1])] \
                  / float(total)

            # Contribute to the 4 surrounding histograms without worrying
            # about bounds: we added a fake line and a fake row
            w_x, w_y = self.w_grid[x_reg], self.w_grid[y_reg]
            w_x_rev, w_y_rev = 1.-w_x, 1.-w_y
            c, r = self.cell_br_idx[x_reg], self.cell_br_idx[y_reg]

            for seg, count in segs_iterator:
                # Precise floating point bin position
                bin_pos = ((orients[seg]-ref_angle) % math.pi) / bin_size
                # Compute the lower bin
                lbin_idx = int(bin_pos) % self.K
                # And the upper one
                ubin_idx = (lbin_idx+1) % self.K
                # The orientation weight for the upper bin (ie the fractional
                # part of bin_pos)
                w_o_u = bin_pos % 1
                w_o_l = 1. - w_o_u
                hists[r][c][lbin_idx]     += count * w_g * w_x     * w_y     * w_o_l
                hists[r][c][ubin_idx]     += count * w_g * w_x     * w_y     * w_o_u
                hists[r][c-1][lbin_idx]   += count * w_g * w_x_rev * w_y     * w_o_l
                hists[r][c-1][ubin_idx]   += count * w_g * w_x_rev * w_y     * w_o_u
                hists[r-1][c][lbin_idx]   += count * w_g * w_x     * w_y_rev * w_o_l
                hists[r-1][c][ubin_idx]   += count * w_g * w_x     * w_y_rev * w_o_u
                hists[r-1][c-1][lbin_idx] += count * w_g * w_x_rev * w_y_rev * w_o_l
                hists[r-1][c-1][ubin_idx] += count * w_g * w_x_rev * w_y_rev * w_o_u

        if ref_angle_overridden:
            d_final = d
        else:
            d_final = self._choose_main_half(d)
            if d_final is None:
                return None

        return ref_angle, self._normalize_and_cap(d_final)

    def _determine_ref_angle(self, dg):
        """Determines the reference angle from a global histogram.

        This simply takes the maximum bin, and performs a 2nd-order polynomial
        interpolation to determine the precise maximum angle.

        Args:
            dg: the global histogram to process. The angle range is assumed to
                be [0, pi[

        Returns:
            A float, an angle in [0, pi[, which corresponds to the maximum of
            the global histogram, if it could be computed.
            If something goes wrong in the interpolation process, None is
            returned.
        """
        # Determine the main orientation by 2nd order polynomial interpolation
        # around the maximum bin
        # Center bin
        bc = np.argmax(dg)
        # Lower bin (negative values are okay)
        bl = bc-1
        # Upper bin
        bu = (bc+1) % dg.size
        # Compute the position of the interpolated maximum, assuming
        # [bl, bc, bu] == [-1, 0, 1]
        div = dg[bu] - 2.*dg[bc] + dg[bl]
        if div == 0.:
            return None
        bopt = (dg[bl]-dg[bu]) / (2.*div)
        # Now shift the result to obtain a floating point bin index, and make
        # sure it's in [0, self.K[
        bopt = (bopt + bc) % dg.size

        # Now compute the reference orientation angle
        #   No need to normalize it, it's already in [0; pi[
        ref_angle = bopt * np.pi / np.float_(dg.size)

        return ref_angle

    def _choose_main_half(self, d):
        """Chooses which half of the descriptor should be the main one.

        The choice of a reference angle and the subsequent rotation of the
        descriptor have not completely resolved the ambiguity on the descriptor
        orientiation, since the reference angle is in [0, pi[:
        we still have to decide to rotate once more, by pi, or to keep the
        descriptor as is.

        This choice is done by choosing the half where there is the largest
        contribution on the 0th bin (ie the main orientation).

        Args:
            d: the descriptor to process

        Returns:
            A descriptor where the upper or lower half was chosen, and put on
            the upper half.
            If no choice could be performed (ie the two sums are equal), None is
            returned.
        """
        ptr = 0
        upper = 0.
        lower = 0.
        # Iterate over the lower half
        for i in xrange(self.N/2 * self.N):
            # Add the 0-th bin to the sum
            lower += d[ptr]
            ptr += self.K
        # Iterate over the upper half
        for i in xrange(self.N/2 * self.N):
            upper += d[ptr]
            ptr += self.K

        if upper == lower:
            self.chosen_half['discard'] += 1
            return None
        elif upper > lower:
            self.chosen_half['keep'] += 1
            if upper != 0.:
                self.chosen_half['ratio'].append(lower / upper)
            return d
        self.chosen_half['switch'] += 1
        if lower != 0.:
            self.chosen_half['ratio'].append(upper / lower)

        # lower > upper, we have to rotate by pi
        drot = np.empty_like(d)
        ptr = 0
        ptr_rev = drot.size
        # We'll just iterate over the histograms normally for d, and in reverse
        # for drot, and set the histograms this way
        for i in xrange(self.N**2):
            drot[ptr_rev-self.K:ptr_rev] = d[ptr:ptr+self.K]
            ptr += self.K
            ptr_rev -= self.K

        return drot

    def _descriptor_vector_to_grid(self, d, dummy_ends=False):
        """Creates a 2D list to access the various histograms in a grid.

        Accessing specific cells in a histogram grid is painful when all we have
        is a linear vector. This creates easy accessors for the data.

        When computing the descriptor, it's convenient to have a dummy row and a
        dummy column, so that the interpolation code doesn't have to worry about
        bounds. The dummy_ends specifies whether this should be done.

        Args:
            d: the linear vector containing the grid histograms
            dummy_ends: if True, one more row and column will be added to the
                2D array that is returned

        Returns:
            A list of lists, of size self.N x self.N (or self.N+1 x self.N+1) if
            dummy ends are requested, where h[r][c] is the histogram at row r,
            column c.
        """
        # Get quick accessors for the different histograms of d
        # Add a fake row and a fake column, which will be ignored in the final
        # descriptor, and just make our lives easier in the computation of the
        # 2D interpolation (when adding the contribution of a pixel)
        size = self.N+1 if dummy_ends else self.N
        grid = [[None for x in xrange(size)] for y in xrange(size)]
        dummy = np.empty(self.K, dtype=np.float_) if dummy_ends else None

        curr = 0
        for y in xrange(self.N):
            for x in xrange(self.N):
                grid[y][x] = d[curr:curr+self.K]
                curr += self.K
            if dummy_ends:
                grid[y][self.N] = dummy
        if dummy_ends:
            for x in xrange(self.N+1):
                grid[self.N][x] = dummy

        return grid

    def _compute_central_gaussian(self, length, std):
        """Computes the Gaussian coefficients for the central Gaussian."""
        g = np.arange(start=0, stop=length, dtype=np.float_)
        g = scipy.stats.norm.pdf(g, 0, std)
        self.g_window = np.empty((length, length), dtype=np.float_)
        for i, j in itertools.product(xrange(length), repeat=2):
            self.g_window[i, j] = g[i] * g[j]
        # Normalize
        norm = (4.*np.sum(self.g_window[1:,:]**2.)+self.g_window[0,0]**2.)**.5
        self.g_window = self.g_window / norm

    def _normalize_and_cap(self, vec):
        """Normalize and cap a vector."""
        # Normalize
        norm = np.linalg.norm(vec)
        d_normed = vec
        if norm > 0.:
            d_normed /= norm

        # Apply the cap if needed
        if self.cap > 0. and self.cap < 1.:
            too_high = d_normed > self.cap
            d_normed[too_high] = self.cap
            norm = np.linalg.norm(d_normed)
            if norm > 0.:
                d_normed /= norm

        return d_normed

