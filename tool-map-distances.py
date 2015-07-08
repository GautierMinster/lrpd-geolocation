# encoding=UTF-8

"""Distance visualization tool for reference OSM maps.

This tool loads some OpenStreetMap map (either directly using the configuration
file of its segments, or using the DataManager), and shows an interactive
window to visualize pixel metric distances on it.
"""

import argparse
import logging
import sys

import cv2
import numpy as np

import maps.mapimager
import maps.mapprocessor
import util.data


log = logging.getLogger(__name__)


class MapDistances(object):
    def __init__(self, mi, size=1000, road_width=1, line_width=2):
        self.mi = mi
        self.size = size
        self.line_width = line_width

        _, self.img = mi.create_image(self.size, line_width=road_width)

        self.dist_pts = []
        self.radius_pt = None

    def run(self):
        cv2.namedWindow('map', cv2.WINDOW_AUTOSIZE)
        cv2.createTrackbar('radius', 'map', min(100, self.size), self.size,
            self._onradiuschange)
        cv2.setMouseCallback('map', self._onmouse)

        self.redraw()
        k = cv2.waitKey(0) & 0xff
        while k != ord('q'):
            if k == ord('n') or k == ord('p'):
                pos = cv2.getTrackbarPos('radius', 'map')
                if k == ord('n'):
                    pos += 1
                else:
                    pos -= 1
                cv2.setTrackbarPos('radius', 'map', pos)
            k = cv2.waitKey(0) & 0xff

    def _onradiuschange(self, *args, **kargs):
        r = cv2.getTrackbarPos('radius', 'map')
        log.info('Metric radius = {}'.format(
            self.mi.get_metric_distance(self.size, r)
        ))
        self.redraw()

    def _onmouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONUP:
            if not self.dist_pts:
                self.dist_pts.append((x, y))
            else:
                if len(self.dist_pts) == 2:
                    self.dist_pts = []
                else:
                    self.dist_pts.append((x, y))
                    log.info('Metric distance = {}'.format(
                        self.mi.get_metric_distance(self.size, *self.dist_pts)
                    ))
            self.redraw()
        elif event == cv2.EVENT_RBUTTONUP:
            if not self.radius_pt:
                self.radius_pt = x, y
            else:
                self.radius_pt = None
            self.redraw()

    def _draw_cross(self, img, pt):
        w = self.line_width
        x,y = pt
        cv2.line(img, (x-w,y-w), (x+w,y+w), (0,255,255), w)
        cv2.line(img, (x-w,y+w), (x+w,y-w), (0,255,255), w)

    def redraw(self):
        if len(self.img.shape) == 3 and self.img.shape[2] == 3:
            i = self.img.copy()
        else:
            i = cv2.cvtColor(self.img, cv2.COLOR_GRAY2BGR)

        w = self.line_width

        # Draw the distance measure line first
        if self.dist_pts:
            if len(self.dist_pts) == 2:
                cv2.line(i, self.dist_pts[0], self.dist_pts[1], (0,0,255), w)

            for x, y in self.dist_pts:
                self._draw_cross(i, (x, y))

        # Now draw a circle with selected radius
        if self.radius_pt:
            r = cv2.getTrackbarPos('radius', 'map')
            cv2.circle(i, self.radius_pt, r, (0,255,0), w)
            self._draw_cross(i, self.radius_pt)

        cv2.imshow('map', i)

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    parser = argparse.ArgumentParser()
    parser.add_argument('map', help='Map to show.')
    parser.add_argument('-p', '--path', action='store_true',
        help='Use a DataManager path instead of a file path?')
    parser.add_argument('-s', '--size', type=int, default=1000,
        help='Size of the generated image.')
    parser.add_argument('--roadwidth', type=int, default=1,
        help='Width of the roads drawn.')
    parser.add_argument('-l', '--linewidth', type=int, default=2,
        help='Width of the line stroke')

    args = parser.parse_args()

    if args.path:
        mp = maps.mapprocessor.MapProcessor.load_from_manager(args.map)
    else:
        mp = maps.mapprocessor.MapProcessor(args.map)

    mi = maps.mapimager.MapImager(mp, 0., 0.)
    mapdists = MapDistances(mi, args.size, args.roadwidth, args.linewidth)
    mapdists.run()

