#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2015 Christopher C. Strelioff <chris.strelioff@gmail.com>
#
# Distributed under terms of the MIT license.

"""exceptions.py

Exceptions for the conjugate package.
"""


class ConjugateException(Exception):
    """Root conjugate exception."""
    pass


class ConjugateDataException(ConjugateException):
    """Invalid data(um)."""
    pass


class ConjugateParameterException(ConjugateException):
    """Invalid parameter(s)."""
    pass
