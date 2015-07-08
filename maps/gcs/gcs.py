# encoding=UTF-8

class GCS(object):
    """Common interface for the various GCS available.

    Attrs:
        segs_raw: raw segments that were parsed, in the original coordinate
            system
    """

    def __init__(self, data_iterator):
        """Initialize the GCS by parsing the raw data.

        Should set the segs_raw attribute.

        Args:
            data_iterator: an iterable of strings, where each line contains a
                segment, of the form "(a b, c d)"
        """
        raise NotImplemented("Subclasses should be doing the dirtywork.")

    def condition(self, params={}):
        """Condition the raw data to a local coordinate frame.

        If params contains a truncate_rect key, it defines the rectangle which
        must be made to fit in [0,1]x[0,1] (the fit may not be perfect, the
        aspect ratio is kept).
        If truncate_rect isn't defined, the rectangle used is the bounding box
        of all the segments.

        The resulting segments are in a coordinate frame oriented in a standard
        image processing way, ie x-axis horizontal towards the right, and
        y-axis vertical towards the bottom.

        If a truncate_rect was provided, the segments returned may not be in
        the [0,1]x[0,1] square, and therefore a rectangle in the new local
        frame is also returned, which gives the boundaries where truncation
        should be performed.

        Args:
            params: a dictionary, containing extra information that may be
                required by each specific GCS (eg truncate_rect, altitude at
                the described location for a latitude-longitude GCS)

        Returns:
            A 4-tuple: (segs, translation, scaling, truncate), where:
            - segs: a geom.Segments instance, containing the segments in the
              local frame described above
            - translation: the metric translation applied to the data
            - scaling: a dictionary containing:
              - world: the ratio of local frame distance over metric distance
                (ie distance (m) = distance (local frame) / scaling world)
              - x_max: the extent of the truncation rectangle on the x-axis:
                the width of the rectangle corresponds to [0,x_max] in the local
                frame
              - y_max: similarly, the height of the rectangle corresponds to
                [0,y_max] in the local frame
            - truncate: a boolean indicating whether the resulting segments
              should be truncated (to [0,x_max]x[0,y_max]) or not
        """
        raise NotImplemented("Subclasses should be doing the dirtywork.")

