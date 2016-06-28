#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2015 Christopher C. Strelioff <chris.strelioff@gmail.com>
#
# Distributed under terms of the MIT license.

"""abstract.py

A location for all abstract classes in the package.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from future.builtins import (ascii, bytes, chr, dict, filter, hex,  # noqa
                             input, int, map, next, oct, open, pow, range,
                             round, str, super, zip)
import six
import abc


class PosteriorBase(six.with_metaclass(abc.ABCMeta, object)):
    """Abstract class for all posteriors."""

    @abc.abstractmethod
    def __contains__(self, parameter):
        pass # pragma: no cover

    @abc.abstractmethod
    def __iter__(self):
        pass # pragma: no cover

    @property
    @abc.abstractmethod
    def distribution(cls):
        """Return a string with distribution name."""
        return cls._distribution

    @property
    @abc.abstractmethod
    def distribution_parameter_names(self):
        """Return a list of parameter names for the distribution."""
        return self._distribution_parameter_names

    @property
    @abc.abstractmethod
    def distribution_parameter_support(self):
        """Return a dictionary with supoort for distribution parameter(s)."""
        return self._distribution_parameter_support

    @property
    @abc.abstractmethod
    def prior(cls):
        """Return a string with name of prior distribution."""
        return cls._prior

    @property
    @abc.abstractmethod
    def prior_hyperparameter_names(self):
        """Return a list of hyperparameter name(s) for the prior."""
        return sorted(self._prior_hyperparameters.keys())

    @property
    @abc.abstractmethod
    def prior_hyperparameters(self):
        """Dictionary containing the prior hyperparameters."""
        return self._prior_hyperparameters

    @prior_hyperparameters.setter
    @abc.abstractmethod
    def prior_hyperparameters(self, new_setting):
        """Set the values of the prior hyperparameters."""
        pass # pragma: no cover

    @property
    @abc.abstractmethod
    def data(self):
        """Dictionary containing (observed) data."""
        return self._data

    @data.setter
    @abc.abstractmethod
    def data(self, new_data):
        """Set new data, over-writing old data."""
        pass # pragma: no cover

    @abc.abstractmethod
    def add_data(self, data):
        """Add data, keeping old data, with validation and processing."""
        pass # pragma: no cover

    @abc.abstractmethod
    def prior_mean(self, parameter):
        """Return prior mean for passed parameter."""
        pass # pragma: no cover

    @abc.abstractmethod
    def posterior_mean(self, parameter):
        """Return posterior mean for passed parameter."""
        pass # pragma: no cover

    @abc.abstractmethod
    def posterior_central_credible_region(self, parameter, confidence):
        """Return central credible region of posterior for passed parameter."""
        pass # pragma: no cover

    @abc.abstractmethod
    def posterior_high_density_credible_region(self, parameter, confidence):
        """Return high-density credible region of posterior for passed
        parameter.
        """
        pass # pragma: no cover

    @abc.abstractmethod
    def plot_parameter_prior(self, parameter, **kwargs):
        """Plot the prior pdf for the passed parameter.
        """
        pass # pragma: no cover

    @abc.abstractmethod
    def plot_parameter_posterior(self, parameter, **kwargs):
        """Plot the posterior pdf for the passed parameter.
        """
        pass # pragma: no cover

    @abc.abstractmethod
    def plot_parameter_summary(self, parameter, **kwargs):
        """Plot the prior and posterior pdfs for the passed parameter.
        """
        pass # pragma: no cover

    @abc.abstractmethod
    def plot_summary(self, parameters, **kwargs):
        """Plot the posterior pdfs for all parameters in the passed list.
        """
        pass # pragma: no cover
