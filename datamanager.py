# encoding=UTF-8

# stdlib
import json
import logging
import os
import os.path

# 3p
import numpy as np


log = logging.getLogger(__name__)


class NonExistentData(Exception):
    pass

class DataManager(object):
    """Save and load data from files.

    This is a helper class which aims at making it easy to save and load data
    from files. Data is identified by a path, of the form "path.to.some.item",
    which translates to a real filesystem directory "path/to/some/", and some
    metadata and data files, typically 'item_meta.json' and 'item_data.json'.
    The directory is relative to a data directory, DataManager.DATADIR.

    The data is stored in JSON format, but numpy arrays are automatically
    detected and stored separately (by default, 'item_data.npz').

    This class can either be used directly (eg for simple data, or manual
    inspection), or be instrumented by other classes, which must implement the
    serialization and de-serialization logic themselves (this class only encodes
    and decodes JSON).

    NOTE: be careful when using Numpy, it has its own types, which tend to
    confuse the JSON encoder. As an example, storing a dictionary like
    {'key': numpy.int32(2)} will fail, because the builtin python int is 64bits,
    and the JSON module doesn't know what to do with a numpy.int32.
    """

    DATADIR = 'data/storage'
    META_SUFFIX = '_meta.json'
    DATA_SUFFIX = '_data.json'
    NUMPY_SUFFIX = '_data.npz'
    NUMPY_KEY = '_numpy_data'

    @classmethod
    def get_data_dir(cls):
        return cls.DATADIR

    @classmethod
    def set_data_dir(cls, newdir):
        cls.DATADIR = newdir
        cls._init_data_dir()

    def _init_data_dir(self):
        fullpath = self.get_full_path()
        if not os.path.exists(fullpath):
            log.info(
                'Data dir "{}" does not exist, creating…'.format(fullpath)
            )
            os.makedirs(fullpath)

    def __init__(self, path):
        """Initialize the manager.

        Args:
            path: the data path, a string of the form "alpha.beta.gamma.delta",
                which is used to identify the data. It will be stored in
                DATADIR/alpha/beta/gamma/delta_{meta.json,data.json,data.npz}.
        """
        self.path = path
        self.metadata = None
        self.data = None

    def get_full_path(self):
        """Returns the full path to the data directory."""
        fulldir = os.path.join(
            self.DATADIR,
            os.path.split(self.path.replace('.', os.sep))[0]
        )
        return fulldir

    def get_basename(self):
        """Returns the prefix of the data filenames."""
        return os.path.split(self.path.replace('.', os.sep))[1]

    def get_metadata_path(self):
        """Returns the path to the metadata."""
        return os.path.join(self.get_full_path(),
            self.get_basename()+self.META_SUFFIX)

    def get_data_path(self):
        """Returns the path to the data."""
        return os.path.join(self.get_full_path(),
            self.get_basename()+self.DATA_SUFFIX)

    def exists(self):
        """Is there any stored data/metadata?"""
        return self.data_exists() or self.metadata_exists()

    def data_exists(self):
        """Is there any stored data?"""
        datapath = self.get_data_path()
        return os.path.exists(datapath)

    def metadata_exists(self):
        """Is there any stored metadata?"""
        metapath = self.get_metadata_path()
        return os.path.exists(metapath)

    def get(self):
        """Get the metadata and the data.

        Returns:
            A (metadata, data) pair.
        """
        if not self.exists():
            raise NonExistentData(
                'There is no data to get at "{}"'.format(self.path)
            )

        if self.data is None:
            if self.metadata_exists():
                with open(self.get_metadata_path(), 'rb') as f:
                    self.metadata = json.load(f)
            if self.data_exists():
                with open(self.get_data_path(), 'rb') as f:
                    self.data = json.load(f)

                # The data we just loaded might be incomplete (numpy arrays are
                # stored separately for example)
                if self.NUMPY_KEY in self.data:
                    numpy_data = np.load(os.path.join(
                        self.get_full_path(), self.data[self.NUMPY_KEY]
                    ))
                    for key in numpy_data:
                        self.data[key] = numpy_data[key]

        return self.metadata, self.data

    def set(self, metadata=None, data=None):
        """Add data to storage.

        No metadata/data merging is ever performed. This is the responsibility
        of the user.

        If metadata is not None, the current metadata will be overwritten (if
        it exists).
        If data is not None, the current data will be overwritten (if it
        exists).

        Args:
            metadata: some meta information about the data
            data: the actual data, a dictionary containing parts of the data as
                values and their names as keys. All the values should be
                storable as JSON, except the numpy arrays which will be
                detected and stored separately
        """
        if metadata is not None:
            self.metadata = metadata
            self._write_data(metadata=self.metadata)

        if data is not None:
            self.data = data
            numpy_data = {}
            for key in data.iterkeys():
                if type(data[key]) == np.ndarray or type(data[key]) == np.array:
                    numpy_data[key] = data[key]
                    data[key] = self.NUMPY_KEY

            if numpy_data:
                if self.NUMPY_KEY not in data:
                    data[self.NUMPY_KEY] = self.get_basename()+self.NUMPY_SUFFIX

            self._write_data(data=data, numpy_data=numpy_data)

    def _write_data(self, metadata=None, data=None, numpy_data=None):
        self._init_data_dir()

        if metadata is not None:
            with open(self.get_metadata_path(), 'wb') as f:
                json.dump(metadata, f)
        if data is not None:
            with open(self.get_data_path(), 'wb') as f:
                json.dump(data, f)

            if self.NUMPY_KEY in data:
                numpy_data_file = os.path.join(
                    self.get_full_path(), data[self.NUMPY_KEY]
                )
                with open(numpy_data_file, 'wb') as f:
                    np.savez_compressed(f, **numpy_data)

