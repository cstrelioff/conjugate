#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2015 Christopher C. Strelioff <chris.strelioff@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Tests for the BinomialBeta class.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from future.builtins import (ascii, bytes, chr, dict, filter, hex,  # noqa
                             input, int, map, next, oct, open, pow, range,
                             round, str, super, zip)

import pytest

from conjugate import PosteriorBase
from conjugate import BinomialBeta
from conjugate import ConjugateDataException
from conjugate import ConjugateParameterException


@pytest.fixture
def binomp():
    return BinomialBeta()


def test_instantiate(binomp):
    """
    * binomial: test_instantiate -- instantiate with no arguments.
    """
    assert isinstance(binomp, PosteriorBase)


def test_contains(binomp):
    """
    * binomial: test_contains -- test the __contains__ method.
    """
    assert 'p' in binomp


def test_contains_invalid(binomp):
    """
    * binomial: test_contains_invalid -- test the __contains__ method w
    bad parameter.
    """
    assert 'a' not in binomp


def test_iter(binomp):
    """
    * binomial: test_iter -- test the __iter__ method.
    """
    params = []
    for item in binomp:
        params.append(item)

    assert params == ['p']


def test_str(binomp):
    """
    * binomial: test_str -- test the __str__ method.
    """
    data = {'n': 10, 'k': 2}
    binomp.data = data
    prior = binomp.prior_hyperparameters

    tmp = ('bp = BinomialBeta()\n'
           'bp.data = {}\n'
           'bp.prior_hyperparameters = {}'.format(data, prior))

    assert str(binomp) == tmp


def test_distribution(binomp):
    """
    * binomial: test_distribution -- test the distribution string property.
    """
    assert binomp.distribution == 'Distribution: Binomial'


def test_distribution_parameter_names(binomp):
    """
    * binomial: test_distribution_parameter_names -- test the distribution
    parameter names property.
    """
    assert binomp.distribution_parameter_names == ['p']


def test_distribution_parameter_support(binomp):
    """
    * binomial: test_distribution_parameter_support -- test the
    distribution parameter support property.
    """
    assert binomp.distribution_parameter_support == {'p': (0.0, 1.0)}


def test_prior(binomp):
    """
    * binomial: test_prior -- test the prior string property.
    """
    assert binomp.prior == 'Prior: Beta'


def test_prior_hyperparameter_names(binomp):
    """
    * binomial: test_prior_hyperparameter_names -- test the prior
    hyperparameter names property.
    """
    assert binomp.prior_hyperparameter_names == ['alpha', 'beta']


def test_prior_hyperparameters_valid(binomp):
    """
    * binomial: test_prior_hyperparameters_valid -- set prior
    hyperparameter values.
    """
    new_setting = {'alpha': 5, 'beta': 10}
    binomp.prior_hyperparameters = new_setting

    assert binomp.prior_hyperparameters == new_setting


def test_prior_hyperparameters_invalid01(binomp):
    """
    * binomial: test_prior_hyperparameters_invalid01 -- try to set
    invalid prior hyperparameter values.
    """
    # incorrect parameter names -- keys in dict
    new_prior_setting = {'alpha0': 5, 'alpha1': 10}

    with pytest.raises(ConjugateParameterException):
        binomp.prior_hyperparameters = new_prior_setting


def test_prior_hyperparameters_invalid02(binomp):
    """
    * binomial: test_prior_hyperparameters_invalid02 -- try to set
    invalid prior hyperparameter values.
    """
    # parameters values <= 0
    new_prior_setting = {'alpha': 5, 'beta': -1}

    with pytest.raises(ConjugateParameterException):
        binomp.prior_hyperparameters = new_prior_setting


def test_prior_hyperparameters_invalid03(binomp):
    """
    * binomial: test_prior_hyperparameters_invalid03 -- try to set
    invalid prior hyperparameter values.
    """
    # pass list instead of dict
    new_prior_setting = [5, 10]

    with pytest.raises(ConjugateParameterException):
        binomp.prior_hyperparameters = new_prior_setting

def test_prior_mean_valid(binomp):
    """
    * binomial: test_prior_mean_valid -- get prior mean for valid
    parameter.
    """
    # almost equal
    assert round(binomp.prior_mean('p') - 1/2) == 0


def test_prior_mean_invalid(binomp):
    """
    * binomial: test_prior_mean_invalid -- (try to) get prior mean for
    invalid parameter.
    """
    with pytest.raises(ConjugateParameterException):
        binomp.prior_mean('a')


def test_posterior_mean_valid(binomp):
    """
    * binomial: test_posterior_mean_valid -- get posterior mean for valid
    parameter.
    """
    binomp.add_data({'n': 5, 'k': 2})

    # almost equal
    assert round(binomp.posterior_mean('p') - (1+2)/(2+5)) == 0


def test_posterior_mean_invalid(binomp):
    """
    * binomial: test_posterior_mean_invalid -- (try to) get posterior mean
    for invalid parameter.
    """
    binomp.add_data({'n': 5, 'k': 2})

    with pytest.raises(ConjugateParameterException):
        binomp.posterior_mean('a')

def test_posterior_hdcr_valid(binomp):
    """
    * binomial: test_posterior_hdcr_valid -- get hdcr for posterior.
    """
    binomp.add_data({'n': 5, 'k': 2})

    hdcr = binomp.posterior_high_density_credible_region('p')
    hdcr_true = [0.10482705290430133, 0.76128981963103848]
    
    assert all(round(x-y) == 0 for x, y in zip(hdcr, hdcr_true))


def test_posterior_hdcr_invalid(binomp):
    """
    * binomial: test_posterior_hdcr_invalid -- (try to) get hdcr for
    invalid parameter.
    """
    binomp.add_data({'n': 5, 'k': 2})

    with pytest.raises(ConjugateParameterException):
        binomp.posterior_high_density_credible_region('nonsense')


def test_posterior_ccr_valid(binomp):
    """
    * binomial: test_posterior_ccr_valid -- get ccr for posterior.
    """
    binomp.add_data({'n': 5, 'k': 2})

    ccr = binomp.posterior_central_credible_region('p')
    ccr_true = [0.11811724875702526, 0.77722190449648787]

    assert all(round(x-y) == 0 for x, y in zip(ccr, ccr_true))


def test_posterior_ccr_invalid(binomp):
    """
    * binomial: test_posterior_ccr_invalid -- (try to) get ccr for
    invalid parameter.
    """
    binomp.add_data({'n': 5, 'k': 2})

    with pytest.raises(ConjugateParameterException):
        binomp.posterior_central_credible_region('nonsense')


def test_data_setter_valid01(binomp):
    """
    * binomial: test_data_setter_valid02 -- test use of data setter w
    valid data.
    """
    binomp.data = {'n': 45, 'k': 40}

    assert binomp.data == {'n': 45, 'k': 40}


def test_data_setter_invalid01(binomp):
    """
    * binomial: test_data_setter_invalid01 -- test use of data setter w
    invalid data-- a list..
    """
    with pytest.raises(ConjugateDataException):
        binomp.data = [0, 0, 1, 1, 1]


def test_data_setter_invalid02(binomp):
    """
    * binomial: test_data_setter_invalid02 -- test use of data setter w
    invalid data.
    """
    # k <= n
    with pytest.raises(ConjugateDataException):
        binomp.data = {'n': 2, 'k': 5}


def test_data_setter_invalid03(binomp):
    """
    * binomial: test_data_setter_invalid03 -- test use of data setter w
    invalid data.
    """
    # data must be dict with relevant keys
    with pytest.raises(ConjugateDataException):
        binomp.data = (5, 4)


def test_add_data_valid_dict(binomp):
    """
    * binomial: test_add_data_valid_dict -- test add_data() method w valid
    data dict.
    """
    binomp.data = {'n': 5, 'k': 1}
    binomp.add_data({'n': 10, 'k': 2})

    assert binomp._data == {'n': 15, 'k': 3}


def test_add_data_invalid_dict(binomp):
    """
    * binomial: test_add_data_invalid_dict -- test add_data() method with
    an valid data dict (invalid key).
    """
    with pytest.raises(ConjugateDataException):
        binomp.add_data({'m': 10, 'k': 2})
