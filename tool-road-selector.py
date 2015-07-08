# encoding=UTF-8

"""Manually select segments on a map.

This tool is a helper to select segments manually on a test map. Left click
places a point, right click removes the last point placed.

See the commandline help for information on arguments.

Note that it can also be given an existing set of segments to work on.
"""

import argparse
import logging
import sys

import cv2
import numpy as np

import util.data


log = logging.getLogger(__name__)


class RoadSelector(object):
    def __init__(self, image_file, output_file, segs=None, line_width=2):
        self.name = image_file
        self.outname = output_file
        self.line_width = line_width

        self.img = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
        if self.img is None:
            log.critical('Could not load image {}'.format(image_file))
            sys.exit(1)

        self.cur_pt = None
        if segs is None:
            self.segments = []
        else:
            s = np.asarray(segs.lines, dtype=np.int_)
            self.segments = [tuple([(l[0],l[1]),(l[2],l[3])]) for l in s]

    def run(self):
        cv2.namedWindow(self.name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.name, self._onmouse)

        self.redraw()
        k = cv2.waitKey(0) & 0xff
        while k != ord('q'):
            if k == ord('s'):
                self.save()
            k = cv2.waitKey(0) & 0xff

    def _onmouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONUP:
            if not self.cur_pt:
                self.cur_pt = x,y
            else:
                self.segments.append((self.cur_pt, (x, y)))
                log.debug('New segment: {} => {}'.format(self.cur_pt,(x,y)))
                self.cur_pt = None
            self.redraw()
        elif event == cv2.EVENT_RBUTTONUP:
            if self.cur_pt:
                self.cur_pt = None
            else:
                if len(self.segments)>0:
                    pt1,pt2 = self.segments.pop()
                    self.cur_pt = pt1
            self.redraw()

    def _draw_segments(self):
        # If we're given a color image, just copy it
        if len(self.img.shape) == 3 and self.img.shape[2] == 3:
            img_copy = self.img.copy()
        else:
            img_copy = cv2.cvtColor(self.img, cv2.COLOR_GRAY2BGR)

        for pt1,pt2 in self.segments:
            cv2.line(img_copy, pt1, pt2, (0, 0, 255), self.line_width)

        if self.cur_pt:
            w = self.line_width
            x,y = self.cur_pt
            cv2.line(img_copy, (x-w,y-w), (x+w,y+w), (0,255,255), w)
            cv2.line(img_copy, (x-w,y+w), (x+w,y-w), (0,255,255), w)

        return img_copy

    def redraw(self):
        i = self._draw_segments()
        cv2.imshow(self.name, i)

    def save(self):
        if len(self.segments) == 0:
            log.warn('No segments to save.')
            return
        s = np.asarray(self.segments, dtype=np.float_).reshape((-1,4))
        util.data.save_segments(self.outname, s)
        log.debug('Saved image to {}'.format(self.outname))
        i = self._draw_segments()
        cv2.putText(i, "saved", (0,self.img.shape[0]), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0))
        cv2.imshow(self.name, i)

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    parser = argparse.ArgumentParser()
    parser.add_argument('image', help='Image to process.')
    parser.add_argument('output', help='Output segments file.')
    parser.add_argument('--segments', default=None,
        help='An existing segments file to extend.')
    parser.add_argument('-l', '--linewidth', type=int, default=2,
        help='Width of the line stroke')

    args = parser.parse_args()
    if args.segments:
        log.info('Extending segments file {}.'.format(args.segments))
        segs = util.data.load_segments(args.segments)
    else:
        segs = None
    rs = RoadSelector(args.image, args.output, segs=segs,
        line_width=args.linewidth)
    rs.run()

