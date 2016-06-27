#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2015 Christopher C. Strelioff <chris.strelioff@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Tests for the utilities.py
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from future.builtins import (ascii, bytes, chr, dict, filter, hex,  # noqa
                             input, int, map, next, oct, open, pow, range,
                             round, str, super, zip)

import pytest

from scipy.stats import beta

from conjugate import central_credible_region
from conjugate import high_density_credible_region


def test_hdcr_binomial_01():
    """
    * utils: test_hdcr_binomial_01 -- test HDCR for binomial inference 01
    """
    hdcr = list(high_density_credible_region(beta(5, 5)))
    pre_comp = [0.212008506778868, 0.787991493221132]

    assert hdcr == pre_comp

def test_hdcr_binomial_02():
    """
    * utils: test_hdcr_binomial_02 -- test HDCR for binomial inference 02
    """
    hdcr = list(high_density_credible_region(beta(1, 10)))
    pre_comp = [0.0, 0.25886555089305224]

    assert hdcr == pre_comp

def test_ccr_binomial_01():
    """
    * utils: test_ccr_binomial_01 -- test CCR for binomial inference 01
    """
    ccr = list(central_credible_region(beta(5, 5)))
    pre_comp = [0.212008506778868, 0.787991493221132]

    assert ccr, pre_comp

def test_ccr_binomial_02():
    """
    * utils: test_ccr_binomial_02 -- test CCR for binomial inference 02
    """
    ccr = list(central_credible_region(beta(1, 10)))
    pre_comp = [0.0025285785444617869, 0.30849710781876077]

    assert ccr == pre_comp
