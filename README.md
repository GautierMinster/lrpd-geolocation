# lrpd-geolocation

This repository contains code for the algorithm described in "Printed Map Geolocation Using Line Segments and Textureless SIFT-like Feature Matching".

It implements an algorithm for the registration of printed maps in a Geographic Information System such as OpenStreetMap. The registration is scale and rotation invariant, and aims at being a scalable first step in an Augmented Maps pipeline, preceding real-time tracking of perspective views of a map.

## Requirements

Here's what's needed to get started:
* Python 2
* NumPy (tested with version 1.8.2)
* SciPy (tested with version 0.14.1)
* OpenCV (tested with version 3.0.0-dev, but uses only basic functionality available in OpenCV 2), compiled with Python bindings

