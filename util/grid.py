# encoding=UTF-8

"""Class to create and iterate over 2D grids."""

# 3p
import numpy as np

# project
import datamanager

class Grid(object):
    """Create and interact with a 2D grid."""

    @classmethod
    def by_spacing(cls, bbox, spacing, start_offset=0):
        """Create a Grid instance defined by the spacing between points.

        Args:
            bbox: bounding box of the region where the grid will be, of the
                form (xmin,ymin,xmax,ymax)
            spacing: size of each cell, or, equivalently, spacing between grid
                cell centers
        """
        def get_range(start, end):
            return np.arange(
                start=start, stop=end, step=spacing, dtype=np.float_
            )
        xcoords = get_range(bbox[0]+start_offset,bbox[2])
        ycoords = get_range(bbox[1]+start_offset,bbox[3])

        return Grid(xcoords, ycoords)

    def __init__(self, xcoords, ycoords):
        """Creates a grid using the given x and y sets of coordinates.

        Args:
            xcoords: list of x-coordinates of the grid
            ycoords: list of y-coordinates of the grid
        """
        self.xs = np.asarray(xcoords, dtype=np.float_)
        self.ys = np.asarray(ycoords, dtype=np.float_)

    def size(self):
        """Number of grid cells/points."""
        return self.xs.size * self.ys.size

    def __getitem__(self, rc):
        """Returns the x,y coordinates of the grid point at (r,c)."""
        return self.xs[rc[1]], self.ys[rc[0]]

    def to_lin(self, r, c):
        """Returns the linear index (index in a flattened grid) of (r,c).

        Elements are scanned line by line (y axis, rows), and within each line
        from left to right (ascending x axis, columns).

        Returns:
            r * xsize + c, where xsize is the number of columns.
        """
        return r * self.xs.size + c

    def from_lin(self, idx, spatial=False):
        """Returns the coordinates corresponding to a linear index.

        See to_lin() for the ordering.

        Args:
            idx: the linear index
            spatial: if False, a row-column (r,c) couple will be returned. If
                True, an (x,y) coordinates couple will be returned.

        Returns:
            r, c = idx // xsize, idx % xsize
                or
            x, y = grid[r,c]
        """
        r, c = idx // self.xs.size, idx % self.xs.size
        if spatial:
            return self.xs[c], self.ys[r]
        else:
            return r, c

    def iter_cells(self):
        """Generator that iterates over each cell.

        Yields:
            (r, c) row-column couples, going row by row, and within each row
            from leftmost column to rightmost.
        """
        for r in xrange(self.ys.size):
            for c in xrange(self.xs.size):
                yield r, c

    def iter_all(self, cast_int=False):
        """Generator that iterates over each cell.

        Args:
            cast_int: if True, the x and y spatial coordinates will be rounded
                and cast to integers.

        Yields:
            (r, y, c, x), where (r,c) are the grid cell coordinates, and (x,y)
            are the corresponding spatial coordinates.
        """

        cast = lambda arg: arg
        if cast_int:
            cast = lambda arg: int(round(arg))

        for r in xrange(self.ys.size):
            for c in xrange(self.xs.size):
                yield r, cast(self.ys[r]), c, cast(self.xs[c])

    def iter_coords(self, cast_int=False):
        """Generator that iterates over the coordinates of each cell.

        Args:
            cast_int: if True, the x and y spatial coordinates will be rounded
                and cast to integers.

        Yields:
            (x, y), the corresponding spatial coordinates of each cell/point.
        """
        for _, y, _, x in self.iter_all(cast_int=cast_int):
            yield x, y

    def save_to_manager(self, path):
        """Saves the object to external storage."""
        dm = datamanager.DataManager(path)
        data = {
            'xs': self.xs,
            'ys': self.ys
        }
        dm.set(data=data)

    @classmethod
    def load_from_manager(self, path):
        """Creates an instance of this class from a stored object."""
        dm = datamanager.DataManager(path)
        meta, data = dm.get()
        return Grid(data['xs'], data['ys'])

