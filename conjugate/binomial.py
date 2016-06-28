#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2015 Christopher C. Strelioff <chris.strelioff@gmail.com>
#
# Distributed under terms of the MIT license.

"""binomial.py

Code for inference of parameters of the Binomial distribution.
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
from scipy.stats import beta

from .abstract import PosteriorBase

from .exceptions import ConjugateDataException
from .exceptions import ConjugateParameterException

from .plots import plot_parameter_pdf

from .utilities import central_credible_region
from .utilities import high_density_credible_region


class BinomialBeta(PosteriorBase):
    """Infer Binomial parameter :math:`p` given data :math:`D=k`, where
    :math:`k` is the *number of successes* in :math:`n` *attempts*.
    """

    _distribution = 'Distribution: Binomial'
    _prior = 'Prior: Beta'

    def __init__(self):
        """Initialize an instance of the BinomialPosterior class."""
        self._distribution_parameter_names = ['p']
        self._distribution_parameter_support = {'p': (0.0, 1.0)}
        self._prior_hyperparameters = {'alpha': 1, 'beta': 1}
        self._data = {'n': 0, 'k': 0}

    def __contains__(self, parameter):
        return parameter in self._distribution_parameter_names

    def __iter__(self):
        return iter(self._distribution_parameter_names)

    def __str__(self):
        prior = self.prior_hyperparameters
        tmp = ('bp = BinomialBeta()\n'
               'bp.data = {}\n'
               'bp.prior_hyperparameters = {}'.format(self.data, prior))

        return tmp

    def _posterior_scipy(self):
        """Return the scipy posterior. For Binomial inference this is the same
        as the marginal because there is a single model parameter.
        """
        a = self._prior_hyperparameters['alpha']
        b = self._prior_hyperparameters['beta']
        k = self._data['k']
        n = self._data['n']

        return beta(a+k, b+n-k)

    def _posterior_marginal_scipy(self, parameter):
        """Return the scipy (marginal) posterior for passed parameter."""
        return self._posterior_scipy()

    def _prior_scipy(self):
        """Return the scipy prior. For Binomial inference this the same as the
        marginal because there is a single model parameter."""
        a = self._prior_hyperparameters['alpha']
        b = self._prior_hyperparameters['beta']

        return beta(a, b)

    def _prior_marginal_scipy(self, parameter):
        """Return the scipy (marginal) prior for passed parameter."""
        return self._prior_scipy()

    def _plot_prior_pdf(self, parameter, ax, **kwargs):
        """Plot parameter prior pdf using passed matplotlib ax."""
        y_label = kwargs.pop('y_label', 'Prior pdf')
        x_label = kwargs.pop('x_label', parameter)

        x_min = self.distribution_parameter_support[parameter][0]
        x_max = self.distribution_parameter_support[parameter][1]
        dx = (x_max - x_min)/100
        x_vals = np.arange(dx, x_max, dx)

        prior = self._prior_marginal_scipy(parameter)
        prior_mean = self.prior_mean(parameter)
        plot_parameter_pdf(ax, prior, prior_mean, x_vals, fill=None,
                           x_fill=None, confidence=0.95,
                           y_label=y_label, x_label=x_label)

    def _plot_posterior_pdf(self, parameter, ax, **kwargs):
        """Plot parameter posterior pdf using passed matplotlib ax."""
        y_label = kwargs.pop('y_label', 'Posterior pdf')
        x_label = kwargs.pop('x_label', parameter)

        x_min = self.distribution_parameter_support[parameter][0]
        x_max = self.distribution_parameter_support[parameter][1]
        dx = (x_max - x_min)/100
        x_vals = np.arange(dx, x_max, dx)

        posterior = self._posterior_marginal_scipy(parameter)
        posterior_mean = self.posterior_mean(parameter)

        n = self.data['n']
        if n > 0:
            fill_type = 'hdcr'
            hdcr = self.posterior_high_density_credible_region
            low_p, high_p = hdcr(parameter)
        else:
            fill_type = 'ccr'
            low_p, high_p = self.posterior_central_credible_region(parameter)

        x_fill = np.arange(low_p, high_p, 0.01)

        plot_parameter_pdf(ax, posterior, posterior_mean, x_vals,
                           fill=fill_type, x_fill=x_fill, confidence=0.95,
                           y_label=y_label, x_label=x_label, color='b')

    @property
    def distribution(self):
        return super().distribution

    @property
    def distribution_parameter_names(self):
        return super().distribution_parameter_names

    @property
    def distribution_parameter_support(self):
        return super().distribution_parameter_support

    @property
    def prior(self):
        return super().prior

    @property
    def prior_hyperparameter_names(self):
        return super().prior_hyperparameter_names

    @property
    def prior_hyperparameters(self):
        return super().prior_hyperparameters

    @prior_hyperparameters.setter
    def prior_hyperparameters(self, new_setting):
        if not isinstance(new_setting, dict):
            raise ConjugateParameterException('Parameters must be passed '
                                              'as a dictionary!')

        if sorted(new_setting.keys()) != ['alpha', 'beta']:
            raise ConjugateParameterException('Keys of parameter dictionary '
                                              'must be: [alpha, beta]!')

        for val in new_setting.values():
            if float(val) <= 0.:
                raise ConjugateParameterException('Parameters alpha and beta '
                                                  'must be greater than '
                                                  'zero!')

        self._prior_hyperparameters = new_setting

    @property
    def data(self):
        return super().data

    @data.setter
    def data(self, new_data):
        # clear current data
        self._data = {'n': 0, 'k': 0}
        self.add_data(new_data)

    def add_data(self, data):
        """Add data, passed as as a dict with keys :math:`n` and :math:`k`."""
        if isinstance(data, list):
            raise ConjugateDataException('Data must be passed as n,k '
                                         'dictionary!')
        elif isinstance(data, dict):
            for key in data:
                if key in self._data:
                    self._data[key] += data[key]
                else:
                    raise ConjugateDataException('Key: {} in passed data not '
                                                 'valid!'.format(key))
        else:
            raise ConjugateDataException('Passed data is not a dictionary!')

        n = self.data['n']
        k = self.data['k']
        if k > n:
            raise ConjugateDataException('Data has k > n -- invalid!')

    def prior_mean(self, parameter):
        """Return the prior mean for the specified parameter."""
        if parameter not in self:
            raise ConjugateParameterException('Parameter not recognized!')
        else:
            a = self.prior_hyperparameters['alpha']
            b = self.prior_hyperparameters['beta']

            return a/(a+b)

    def prior_sample(self):
        """Return a sample of all parameters from the Beta prior."""
        pass

    def prior_sample_parameter(self, parameter):
        """Return a sample of the passed parameter from the Beta prior."""
        pass

    def posterior_mean(self, parameter):
        """Return the posterior mean for the specified parameter."""
        if parameter not in self:
            raise ConjugateParameterException('Parameter not recognized!')
        else:
            a = self.prior_hyperparameters['alpha']
            b = self.prior_hyperparameters['beta']
            n = self.data['n']
            k = self.data['k']

            return (a+k)/(a+b+n)

    def posterior_sample(self):
        """Return a sample of all parameters from the Beta posterior."""
        pass

    def posterior_sample_parameter(self, parameter):
        """Return a sample of the passed parameter from the Beta posterior."""
        pass

    def posterior_central_credible_region(self, parameter, confidence=0.95):
        """Return central credible region of posterior for passed parameter."""
        if parameter not in self:
            raise ConjugateParameterException('Parameter not recognized!')
        else:
            ccr = central_credible_region(self._posterior_marginal_scipy(parameter),
                                          confidence=confidence)

            return list(ccr)

    def posterior_high_density_credible_region(self, parameter,
                                               confidence=0.95):
        """Return high-density credible region of the posterior for passed
        parameter.
        """
        if parameter not in self:
            raise ConjugateParameterException('Parameter not recognized!')
        else:
            p = parameter
            hdcr = high_density_credible_region(self._posterior_marginal_scipy(p),
                                                confidence=confidence)

            return list(hdcr)

    def plot_parameter_prior(self, parameter, **kwargs):
        """Plot the prior pdf."""
        width = kwargs.pop('width', 8)
        height = kwargs.pop('height', 3)
        y_label = kwargs.pop('y_label', 'Prior pdf')
        x_label = kwargs.pop('x_label', 'p: Probability of success')

        fig, ax = plt.subplots(1, 1, figsize=(width, height))
        self._plot_prior_pdf(parameter, ax, y_label=y_label,
                             x_label=x_label)

        return fig, ax

    def plot_parameter_posterior(self, parameter, **kwargs):
        """Plot the posterior pdf."""
        width = kwargs.pop('width', 8)
        height = kwargs.pop('height', 3)
        y_label = kwargs.pop('y_label', 'Posterior pdf')
        x_label = kwargs.pop('x_label', 'p: Probability of success')

        fig, ax = plt.subplots(1, 1, figsize=(width, height))
        self._plot_posterior_pdf(parameter, ax, y_label=y_label,
                                 x_label=x_label)

        return fig, ax

    def plot_parameter_summary(self, parameter, **kwargs):
        """Plot prior and posterior summary."""
        width = kwargs.pop('width', 8)
        height = kwargs.pop('height', 6)
        prior_ylabel = kwargs.pop('prior_ylabel', 'Prior pdf')
        posterior_ylabel = kwargs.pop('posterior_ylabel', 'Posterior pdf')
        x_label = kwargs.pop('x_label', 'p: Probability of success')

        fig, ax = plt.subplots(2, 1, figsize=(width, height), sharex=True)
        self._plot_prior_pdf(parameter, ax[0], ylabel=prior_ylabel,
                             x_label=None)
        self._plot_posterior_pdf(parameter, ax[1], y_label=posterior_ylabel,
                                 x_label=x_label)

        return fig, ax

    def plot_summary(self, **kwargs):
        """Plot posterior pdfs for all parameters."""
        self.plot_parameter_posterior('p', **kwargs)
