#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2015 Christopher C. Strelioff <chris.strelioff@gmail.com>
#
# Distributed under terms of the MIT license.

"""
plots.py

Methods for plotting.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from future.builtins import (ascii, bytes, chr, dict, filter, hex,  # noqa
                             input, int, map, next, oct, open, pow, range,
                             round, str, super, zip)

import numpy as np
import matplotlib.pyplot as plt
try:  # noqa
    plt.style.use('ggplot')
except:
    pass


def plot_parameter_pdf(ax, dist, dist_mean, x_param, fill=None, x_fill=None,
                       confidence=0.95, x_label=None, y_label=None,
                       color='r'):
    """Plot the probability density using the passed matplotlib axis.
    """
    try:
        line_format = color + '-'
        marker_format = color + 'o'
    except:
        pass

    if color == 'r':
        fill_color = 'red'
    else:
        fill_color = 'blue'

    # plot pdf
    ax.plot(x_param, dist.pdf(x_param), line_format)

    # plot mean
    ax.stem([dist_mean], [dist.pdf(dist_mean)],
            linefmt=line_format, markerfmt=marker_format,
            basefmt='w-')

    # set y upper-bound
    ax.set_ylim(0., 1.1*np.max(dist.pdf(x_param)))

    # fill
    if (fill is not None) and (x_fill is not None):
        ax.fill_between(x_fill, 0, dist.pdf(x_fill),
                        color=fill_color, alpha=0.2)

    # labels
    if y_label:
        ax.set_ylabel(y_label)

    if x_label:
        ax.set_xlabel(x_label)
