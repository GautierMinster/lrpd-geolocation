# encoding=UTF-8

"""Geographic Coordinate System

Module containing tools to parse and import data in various coordinate systems,
such as OSGB36 and Latitude-Longitude.
"""

# project
from maps.gcs.osgb36 import GCS_OSGB36
from maps.gcs.latlong import GCS_LatLong


class InvalidGCSFormat(Exception):
    pass

def gcs_from_format(gcs_format):
    """Returns the GCS subclass corresponding to a format."""
    f = gcs_format.lower()

    if f == 'osgb36':
        return GCS_OSGB36
    elif f == 'latlong':
        return GCS_LatLong
    else:
        raise InvalidGCSFormat("Unknown GCS format: {}".format(gcs_format))

