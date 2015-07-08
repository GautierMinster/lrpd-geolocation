# encoding=UTF-8

"""Utilities to load and save data."""

import logging

import cv2
import numpy as np
import sys

from geom import Segments, SegmentsException

log = logging.getLogger(__name__)

def save_segments(filename, segs):
    array = segs
    if type(segs) == np.ndarray:
        pass
    elif type(segs) == Segments:
        array = segs.lines
    else:
        raise Exception("Unknow segment data type: {0}".format(type(segments)))

    # Open the file ourselves, we don't want to append a .npy ext
    with open(filename, 'wb') as f:
        np.save(f, array)

def load_segments(filename):
    try:
        return Segments(np.load(filename))
    except IOError:
        log.critical('Could not load segments data from {0}'.format(filename))
        raise
    except SegmentsException as e:
        log.critical('Invalid segments data: {0}'.format(e))
        raise

def load_segment_files(filenames):
    lines = []
    for filename in filenames:
        lines.append(load_segments(filename))
    return lines

def save_numpy_array(filename, array):
    with open(filename, 'wb') as f:
        np.save(f, array)

def load_numpy_array(filename):
    try:
        return np.load(filename)
    except IOError:
        log.critical("Could not load numpy array from {}".format(filename))
        raise Exception()

def load_numpy_arrays(filenames):
    data = []
    for filename in filenames:
        data.append(load_numpy_array(filename))
    return data

