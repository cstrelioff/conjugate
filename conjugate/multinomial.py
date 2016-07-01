#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2015 Christopher C. Strelioff <chris.strelioff@gmail.com>
#
# Distributed under terms of the MIT license.

"""multinomial.py

Code for inference of parameters of the Multinomial distribution using the
conjugate Dirichlet distribution prior.
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
import matplotlib.gridspec as gridspec
from scipy.stats import beta as _scipy_beta
from numpy.random import multinomial as _np_multinomial

from .abstract import PosteriorBase

from .exceptions import ConjugateDataException
from .exceptions import ConjugateParameterException

from .plots import plot_parameter_pdf

from .utilities import central_credible_region
from .utilities import high_density_credible_region


class MultinomialDirichlet(PosteriorBase):
    """Infer Multinomial parameters :math:`p_i` given data :math:`D=\{n_i\}`,
    where :math:`n_i` is the number of observations of type :math:`i` in the
    available data.
    """

    _distribution = 'Distribution: Multinomial'
    _prior = 'Prior: Dirichlet'

    def __init__(self, alphabet):
        """Initialize an instance of the MultinomialPosterior class.

        Arguments:
        ---------
        alphabet: the types of observations; ideally, a list of strings.
        """
        self.alphabet = [str(i) for i in alphabet]
        self._distribution_parameter_names = \
            [str('p_{}'.format(i)) for i in self.alphabet]

        self._distribution_parameter_support = \
            {str(p): (0.0, 1.0) for p in self._distribution_parameter_names}

        self._prior_hyperparameters = \
            {str('a_{}'.format(i)): 1 for i in self.alphabet}

        self._data = {i: 0 for i in self.alphabet}

    def __contains__(self, parameter):
        return parameter in self._distribution_parameter_names

    def __iter__(self):
        return iter(self._distribution_parameter_names)

    def __str__(self):
        prior = self.prior_hyperparameters
        tmp = ('mp = MultinomialDirichlet({})\n'
               'mp.data = {}\n'
               'mp.prior_hyperparameters = {}'.format(self.alphabet,
                                                      self.data,
                                                      prior))

        return tmp

    def _posterior_marginal_scipy(self, parameter):
        """Return the scipy (marginal) posterior for passed parameter."""
        letter = parameter.strip().split('_')[1]
        A = sum(self._prior_hyperparameters.values())
        ai = self.prior_hyperparameters['a_{}'.format(letter)]
        N = sum(self._data.values())
        ni = self._data[letter]

        return _scipy_beta(ai+ni, A-ai+N-ni)

    def _prior_marginal_scipy(self, parameter):
        """Return the scipy (marginal) prior for passed parameter."""
        letter = parameter.strip().split('_')[1]
        A = sum(self._prior_hyperparameters.values())
        ai = self.prior_hyperparameters['a_{}'.format(letter)]

        return _scipy_beta(ai, A-ai)

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

        N = sum(self.data.values())
        if N > 0:
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
            msg = 'Hyperparameters must passed as a dictionary!'
            raise ConjugateParameterException(msg)

        for ai in new_setting:
            if ai in self._prior_hyperparameters:
                if new_setting[ai] > 0.:
                    self._prior_hyperparameters[ai] = new_setting[ai]
                else:
                    msg = 'Hyperparameters must be greater than zero!'
                    raise ConjugateParameterException(msg)
            else:
                msg = 'Invalid hyperparameter: {}!'.format(ai)
                raise ConjugateParameterException(msg)

    @property
    def data(self):
        return super().data

    @data.setter
    def data(self, new_data):
        # clear current data
        for i in self.alphabet:
            self._data[i] = 0

        self.add_data(new_data)

    def add_data(self, data):
        """Add data, passed as list of 0's and 1's--- or as a dict with keys
        :math:`n` and :math:`k`."""
        if isinstance(data, dict):
            for i in data:
                if i in self.alphabet:
                    if data[i] >= 0:
                        self._data[i] += data[i]
                    else:
                        raise ConjugateDataException('Passed neagtive data!')
                else:
                    raise ConjugateDataException('Passed data has key not '
                                                 'found in alphabet: '
                                                 '{}!'.format(i))
        else:
            raise ConjugateDataException('Passed data is not a dict!')

    def prior_mean(self, parameter):
        """Return the prior mean for the specified parameter."""
        if parameter not in self:
            raise ConjugateParameterException('Parameter not recognized!')
        else:
            i = parameter.split('_')[1]
            ai = self.prior_hyperparameters['a_{}'.format(i)]
            A = sum(self.prior_hyperparameters.values())

            return ai/A

    def prior_sample(self):
        """Return a sample of all parameters from the Dirichlet prior."""
        pass

    def prior_sample_parameter(self, parameter):
        """Return a sample of the passed parameter from the (marginal) Beta
        prior.
        """
        pass

    def posterior_mean(self, parameter):
        """Return the posterior mean for the specified parameter."""
        if parameter not in self:
            raise ConjugateParameterException('Parameter not recognized!')
        else:
            i = parameter.split('_')[1]
            ai = self.prior_hyperparameters['a_{}'.format(i)]
            A = sum(self.prior_hyperparameters.values())
            ni = self.data[i]
            N = sum(self.data.values())

            return (ai+ni)/(A+N)

    def posterior_sample(self):
        """Return a sample of all parameters from the Dirichlet posterior."""
        pass

    def posterior_sample_parameter(self, parameter):
        """Return a sample of the passed parameter from the (marginal) Beta
        posterior.
        """
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
        x_label = kwargs.pop('x_label', parameter)

        fig, ax = plt.subplots(1, 1, figsize=(width, height))
        self._plot_prior_pdf(parameter, ax, y_label=y_label,
                             x_label=x_label)

        return fig, ax

    def plot_parameter_posterior(self, parameter, **kwargs):
        """Plot the posterior pdf."""
        width = kwargs.pop('width', 8)
        height = kwargs.pop('height', 3)
        y_label = kwargs.pop('y_label', 'Posterior pdf')
        x_label = kwargs.pop('x_label', parameter)

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
        x_label = kwargs.pop('x_label', parameter)

        fig, ax = plt.subplots(2, 1, figsize=(width, height), sharex=True)
        self._plot_prior_pdf(parameter, ax[0], ylabel=prior_ylabel,
                             x_label=None)
        self._plot_posterior_pdf(parameter, ax[1], y_label=posterior_ylabel,
                                 x_label=x_label)

        return fig, ax

    def plot_summary(self, **kwargs):
        """Plot posterior pdfs for all parameters."""
        ncols = 2
        nparams = len(self.distribution_parameter_names)
        d, r = divmod(nparams, 2)
        if r > 0:
            nrows = d + 1
        else:
            nrows = d

        gs = gridspec.GridSpec(nrows, ncols)
        fig = plt.figure(num=1, figsize=(16, 3*nrows))
        ax = []
        for n, parameter in enumerate(self):
            r, c = divmod(n, 2)
            ax.append(fig.add_subplot(gs[r, c]))

            if c == 0 and (n <= nparams-1):
                self._plot_posterior_pdf(parameter, ax[-1],
                                         y_label='Posterior pdf',
                                         x_label=parameter)
            elif n <= nparams-1:
                self._plot_posterior_pdf(parameter, ax[-1],
                                         y_label=None,
                                         x_label=parameter)

        fig.tight_layout()

        return fig, ax
