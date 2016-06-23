#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2015 Christopher C. Strelioff <chris.strelioff@gmail.com>
#
# Distributed under terms of the MIT license.

"""
utilities.py

Tools that are useful for application to many distributions.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from future.builtins import (ascii, bytes, chr, dict, filter, hex,  # noqa
                             input, int, map, next, oct, open, pow, range,
                             round, str, super, zip)

from scipy.optimize import fmin


def central_credible_region(dist, confidence=0.95):
    """Find the central credible region (CCR) for the passed
    distrbution at the specified level. This means there are
    equally-sized tails on each sise of the region.

    Arguments:
    ----------
    dist: frozen instance of `scipy.stats` distribution.
    confidence: probability associated with region, default 0.95.

    Returns:
    --------
    ccr: list with lower- and upper-bounds of region.
    """
    alpha = 1.0 - confidence

    return dist.ppf([alpha/2, 1.0 - alpha/2])


def high_density_credible_region(dist, confidence=0.95):
    """Find the high-density credible region (HDCR) for the passed
    distrbution at the specified level.

    Arguments:
    ----------
    dist: frozen instance of `scipy.stats` distribution.
    confidence: probability associated with region, default 0.95.

    Returns:
    --------
    hdcr: list with lower- and upper-bounds of region.

    Acknowledgements:
    -----------------
    Original version of this code to:

    http://stackoverflow.com/a/25777507

    Inspired by Kruschke's `Doing Bayesian Data Analysis`.
    """
    def region_width(lower_bound):
        return dist.ppf(lower_bound + confidence) - dist.ppf(lower_bound)

    # find minimum region
    hdcr_lower_bound = fmin(region_width, 1.0 - confidence,
                            ftol=1.e-8, disp=False)[0]

    return dist.ppf([hdcr_lower_bound, hdcr_lower_bound + confidence])
