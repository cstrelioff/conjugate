#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2015 Christopher C. Strelioff <chris.strelioff@gmail.com>
#
# Distributed under terms of the MIT license.

"""__init__.py

The conjugate package.

"""
from .abstract import PosteriorBase  # noqa

from .binomial import BinomialBeta
from .multinomial import MultinomialDirichlet

from .exceptions import ConjugateException  # noqa
from .exceptions import ConjugateDataException  # noqa
from .exceptions import ConjugateParameterException  # noqa

from .plots import plot_parameter_pdf  # noqa

from .utilities import central_credible_region  # noqa
from .utilities import high_density_credible_region  # noqa

__all__ = ['BinomialBeta',
           'MultinomialDirichlet']
