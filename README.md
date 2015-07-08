# lrpd-geolocation

This repository contains code for the algorithm described in "Printed Map Geolocation Using Line Segments and Textureless SIFT-like Feature Matching".

It implements an algorithm for the registration of printed maps in a Geographic Information System such as OpenStreetMap. The registration is scale and rotation invariant, and aims at being a scalable first step in an Augmented Maps pipeline, preceding real-time tracking of perspective views of a map.

## Requirements

Here's what's needed to get started:
* Python 2
* NumPy (tested with version 1.8.2)
* SciPy (tested with version 0.14.1)
* OpenCV (tested with version 3.0.0-dev, but uses only basic functionality available in OpenCV 2), compiled with Python bindings

## Code overview

The code is organized in the following way:
* `datamanager.py`: code to make loading and saving data a breeze.
    It is used for example by `maps.rmsmap.RMSMap`, a class which does a fair bit of computations, to save and restore its state.
* `geom/`: module containing geometry-related utilities
    * `__init__.py`: contains the `geom.Segments` class, a container for line segments used throughout the code
    * `planar.py`: utilities for planar geometry
    * `transform.py`: utilities for computing general transformations
* `maps/`: code dealing with maps
    * `benchmarking.py`: contains a MapFuzzer class, which has a pleasant interface for adding different forms of noise to a `DiscreteMap`
    * `density.py`: utilities to estimate segment densities and corresponding region sizes
    * `discretemap.py`: contains `DiscreteMap`, the basic raster map container (multi-set image of segments)
    * `gcs/`: Geographic Coordinate System, contains modules used by `maps.mapprocessor` to load OpenStreetMap data
        * `gcs.py`: common interface for the various GCS
        * `__init__.py`: utility to find the adequate GCS parsing module
        * `latlong.py`: GCS class to parse latitude-longitude segment data
        * `osgb36.py`: GCS class to parse segment data in OSGB36 coordinates
    * `keypoint.py`: basic `Keypoint` class for storing position, scale and orientation of keypoints
    * `labeller.py`: contains `MultiSetLabeller`, a class for the creation and use of multi-sets, as used by `DiscreteMap`
    * `lrpd.py`: class `LRPD` to compute local road pattern descriptors
    * `mapimager.py`: once OpenStreetMap data is pre-processed (normalized), this class makes creating arbitrary size raster images of the data easy
    * `mapprocessor.py`: class to pre-process OpenStreetMap data, by parsing raw segment data and normalizing it
    * `metricmap.py`: superclass of `DiscreteMap`, which stores metric information. Useful when the metric is known, or when groundtruth scale for a raster map is known
    * `rmsmap.py`: class `RMSMap` (Rotational Multi-Scale Map) to represent a raster map (`DiscreteMap`) at multiple scales and describe it
    * `rmsmatcher.py`: class `RMSMatcher` to match test `RMSMap`s against an index of reference `RMSMap`s
* `results`: module to keep track of experiment results.
    When trying out different values for parameters, or running benchmarks, this module provides an interface to save and load (parameters, result) pairs (using the DataManager). New data can be transparently appended, and stored data can be retrieved and filtered (on both parameters and results).
* `tool-map-distances.py`: standalone script to interactively display distances on an OpenStreetMap map in pixels or meters
* `tool-road-selector.py`: standalone script to select roads in a map image using the mouse, and outputting a set of segments suitable for use in the registration.
* `util/`: various utility functions
    * `data.py`: utilities to (mostly) load and save `geom.Segments`
    * `grid.py`: `Grid` class to create and iterate over 2D grids
    * `image.py`: utilities pertaining to images and the drawing of segments
    * `progressmeter.py`: helper class to display a customizable progress bar on the console

