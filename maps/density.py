# encoding=UTF-8

"""Routines pertaining to map density estimation."""

# stdlib
import itertools
import math

# 3p
import numpy as np

def density(dmap, s):
    """Computes the density of a discrete map at the given size.

    The density of a discrete map is defined as the maximum density of cells in
    a grid subdivision of the discrete map, grid with cell spacing equal to s.

    The density of a cell is simply the number of non-zero pixels (ie pixels
    through which roads pass), divided by the total number of non-zero pixels
    in the image.

    NOTE: this definition of density is not suitable for arbitrarily large
    maps: for example, it is possible to keep the same spatial resolution while
    increasing the spatial size of the region shown. Doing this will increase
    the total number of roads arbitrarily, while the density of each cell won't
    vary much. Consequently, an order of magnitude of sizes of input discrete
    maps should be defined when using thresholds on density.
    For example, we could assume thresholds tailored to images of size ~1000px.

    NOTE: contrary to natural expectations, the density is not a monotonously
    increasing function of the size s: the division in cells is arbitrary, and
    may not match the patterns in the image. This density is only intended as
    a fast indicator, not a precise computation.
    This becomes increasingly prevalent as the size grows.

    Args:
        dmap: a DiscreteMap instance to process
        s: size of each cell (square, s x s)

    Returns:
        The density of the map at this size/scale, a float in [0,1].
    """
    s_i = int(s)
    s_f = float(s)
    rows = int(math.ceil(dmap.img.shape[0]/s_f))
    cols = int(math.ceil(dmap.img.shape[1]/s_f))

    max_count = 0
    total = 0
    for i, j in itertools.product(xrange(rows), xrange(cols)):
        region = dmap.img[i*s_i:(i+1)*s_i, j*s_i:(j+1)*s_i]
        count = np.count_nonzero(region)
        total += count
        if count > max_count:
            max_count = count

    return float(max_count) / float(total)

def find_optimal_size(dmap, target=0.05, smin=20, factor=1.5):
    """Find a region size which density close to a target.

    For the details of the density, its computation and caveats, see the doc
    for the density() routine.

    This functions produces an estimate of the region size yielding a density
    of target. (see density() for why this is only an estimate)

    Sizes are tested starting from smin, and then by multiplying the size by
    factor at every step, until the density exceeds target. The final estimate
    is a linear interpolation of the last two sizes tested.

    If the density at size smin is higher than target, smin is returned.

    Args:
        dmap: the DiscreteMap to process
        target: the density to reach (in [0,1])
        smin: the minimum region size to test
        factor: geometrical increase factor for the region size

    Returns:
        An integer region size which yields, more or less, a density of target.
    """
    assert target>=0. and target<=1., "Density is in [0,1]"
    assert factor>1., "The size increase factor must be > 1."

    s_low = smin
    s_high = smin
    d_low = density(dmap, s_low)
    d_high = d_low

    while d_high < target:
        s_low = s_high
        d_low = d_high
        s_high *= factor
        d_high = density(dmap, s_high)

    if s_low == s_high:
        return int(s_low)

    s_target = s_low + (target-d_low)*(s_high-s_low)/(d_high-d_low)

    return int(s_target)

