# encoding=UTF-8

"""Helper class to display progress to the user."""

import sys

class ProgressMeter(object):
    """Class to output progress during computations."""

    OUTPUT_DOT = 0
    OUTPUT_NUM = 1

    def __init__(self, size, dot_every=1., num_every=10.):
        """Initializes the progress meter.

        Args:
            size: the total size of the problem
            dot_every: output a dot every how many %. Set to 0 to disable dots
            num_every: output a progress value every how many %. Set to 0 to
                disable
        """
        assert size>0
        self.state = 0
        self.size = size
        self.dots = dot_every>0
        self.dstep = float(dot_every)
        self.nums = num_every>0
        self.nstep = float(num_every)

    def start(self):
        """Start the meter."""
        self.state = 0
        if self.nums:
            sys.stdout.write('0%')
            if not self.dots:
                sys.stdout.write(' ')
            sys.stdout.flush()

    def end(self):
        """End the meter, writing a final linefeed."""
        sys.stdout.write('\n')
        sys.stdout.flush()

    def incr(self):
        """Increment the counter."""
        # Progress percentages
        cur_p = 100.*float(self.state)/float(self.size)
        new_p = 100.*float(self.state+1)/float(self.size)
        self.state += 1

        if self.dots:
            next_dot = cur_p - (cur_p%self.dstep) + self.dstep
        if self.nums:
            next_num = cur_p - (cur_p%self.nstep) + self.nstep

        written = False

        while (self.dots and next_dot<=new_p) \
                or (self.nums and next_num<=new_p):
            wrote_dot = False
            if self.dots and next_dot <= new_p:
                if not (self.nums and next_num < next_dot):
                    sys.stdout.write('.')
                    written = True
                    wrote_dot = True
                    next_dot += self.dstep
            if (not wrote_dot) and self.nums and next_num <= new_p:
                fmt = '{:.1f}%'
                if self.nstep % 1.0 == 0.:
                    fmt = '{:.0f}%'
                if (not self.dots) and next_num != 100.:
                    fmt += ' '
                sys.stdout.write(fmt.format(next_num))
                written = True
                next_num += self.nstep

        if written:
            sys.stdout.flush()

