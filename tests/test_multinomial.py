#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2015 Christopher C. Strelioff <chris.strelioff@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Tests for the MultinomialDirichlet class.
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
from conjugate import MultinomialDirichlet
from conjugate import ConjugateDataException
from conjugate import ConjugateParameterException


@pytest.fixture
def setup():
    alphabet = [str('a'), str('b'), str('c'), str('d')]
    multinomp = MultinomialDirichlet(alphabet)
    return {'alphabet': alphabet, 'multinomp': multinomp}


def test_instantiate(setup):
    """
    * multinomial: test_instantiate -- instantiate and test is instance
    of PosteriorBase ABC.
    """
    multinomp = setup['multinomp']

    assert isinstance(multinomp, PosteriorBase)


def test_contains(setup):
    """
    * mulitnomial: test_contains -- make sure p in self works...
    """
    multinomp = setup['multinomp']

    assert 'p_a' in multinomp


def test_contains_invalid(setup):
    """
    * mulitnomial: test_contains_invalid -- try invalid parameter...
    """
    multinomp = setup['multinomp']

    assert 'p_nonsense' not in multinomp


def test_iter(setup):
    """
    * multinomial: test_iter -- test the __iter__ method...
    """
    multinomp = setup['multinomp']

    params = []
    for item in multinomp:
        params.append(item)

    assert params == ['p_a', 'p_b', 'p_c', 'p_d']


def test_str(setup):
    """
    * multinomial: test_str -- does __str__ method work?, yup...
    """
    alphabet = setup['alphabet']
    multinomp = setup['multinomp']

    data = {str('a'): 10, str('b'): 0, str('c'): 5, str('d'): 2}
    multinomp.data = data
    prior = multinomp.prior_hyperparameters
    tmp = ('mp = MultinomialDirichlet({})\n'
           'mp.data = {}\n'
           'mp.prior_hyperparameters = {}'.format(alphabet,
                                                  data,
                                                  prior))
    assert tmp == str(multinomp)


def test_data_setter_valid01(setup):
    """
    * multinomial: test_data_setter_valid01 -- use the data setter with
    valid, simple input
    """
    multinomp = setup['multinomp']

    data = {'a': 10, 'b': 0, 'c': 5, 'd': 2}
    multinomp.data = data

    assert data == multinomp.data


def test_data_setter_valid02(setup):
    """
    * multinomial: test_data_setter_valid02 -- use the data setter with
    valid input; leave off one key-value pair
    """
    multinomp = setup['multinomp']

    data = {'a': 10, 'b': 0, 'd': 2}
    multinomp.data = data

    cnta = (multinomp.data['a'] == 10)
    cntb = (multinomp.data['b'] == 0)
    cntc = (multinomp.data['c'] == 0)  # this should b okay!
    cntd = (multinomp.data['d'] == 2)

    assert (cnta and cntb and cntc and cntd)


def test_data_setter_invalid01(setup):
    """
    * multinomial: test_data_setter_invalid01 -- use the data setter with
    invalid input; bad key.
    """
    multinomp = setup['multinomp']

    data = {'a': 10, 'b': 0, 'c': 5, 'g': 2}

    with pytest.raises(ConjugateDataException):
        multinomp.data = data


def test_data_setter_invalid02(setup):
    """
    * multinomial: test_data_setter_invalid02 -- use the data setter with
    invalid input; negative value.
    """
    multinomp = setup['multinomp']

    data = {'a': 10, 'b': 0, 'c': -2}

    with pytest.raises(ConjugateDataException):
        multinomp.data = data


def test_data_setter_invalid03(setup):
    """
    * multinomial: test_data_setter_invalid03 -- use the data setter with
    invalid input; data as list.
    """
    multinomp = setup['multinomp']

    data = [10, 0, 2]

    with pytest.raises(ConjugateDataException):
        multinomp.data = data


def test_distribution(setup):
    """
    * multinomial: test_distribution -- returns str with dist name...
    """
    multinomp = setup['multinomp']

    assert multinomp.distribution == 'Distribution: Multinomial'


def test_distribution_parameter_names_valid(setup):
    """
    * multinomial: test_distribution_parameter_names_valid -- basic test
    of returned list...
    """
    multinomp = setup['multinomp']

    assert multinomp.distribution_parameter_names == \
        ['p_a', 'p_b', 'p_c', 'p_d']


def test_distribution_parameter_names_invalid(setup):
    """
    * multinomial: test_distribution_parameter_names_invalid -- try to
    set the parameter names; can't, they're read-only...
    """
    multinomp = setup['multinomp']

    with pytest.raises(AttributeError):
        multinomp.distribution_parameter_names = ['p_one', 'p_two']


def test_distribution_parameter_support(setup):
    """
    * multinomial: test_distribution_parameter_support -- make sure all
    supports are (0.0, 1.0)...
    """
    multinomp = setup['multinomp']

    for p in multinomp:
        assert multinomp.distribution_parameter_support[p] == (0.0, 1.0)


def test_prior(setup):
    """
    * multinomial: test_prior -- returns str with prior name...
    """
    multinomp = setup['multinomp']

    assert multinomp.prior == 'Prior: Dirichlet'


def test_prior_hyperparameter_names(setup):
    """
    * multinomial: test_prior_hyperparameter_names -- returns list with
    prior hyperparameter name...
    """
    multinomp = setup['multinomp']

    assert multinomp.prior_hyperparameter_names == ['a_a', 'a_b', 'a_c', 'a_d']


def test_prior_hyperparameter_names_assign(setup):
    """
    * multinomial: test_prior_hyperparameter_names_assign -- try to assign
    prior hyperparameter names -- not allowed...
    """
    multinomp = setup['multinomp']

    with pytest.raises(AttributeError):
        multinomp.prior_hyperparameter_names = ['a_fred', 'a_ted']


def test_prior_hyperparameters(setup):
    """
    * multinomial: test_prior_hyperparameters -- returns dict with
    prior hyperparameter names and values...
    """
    alphabet = setup['alphabet']
    multinomp = setup['multinomp']

    hp = {'a_{}'.format(l): 1 for l in alphabet}

    assert multinomp.prior_hyperparameters == hp


def test_prior_hyperparameters_assign_valid01(setup):
    """
    * multinomial: test_prior_hyperparameters_assign_valid01 -- assign
    prior hyperparameter with valid dict...
    """
    alphabet = setup['alphabet']
    multinomp = setup['multinomp']

    hp = {'a_{}'.format(l): 5 for l in alphabet}
    multinomp.prior_hyperparameters = hp

    assert multinomp.prior_hyperparameters == hp


def test_prior_hyperparameters_assign_valid02(setup):
    """
    * multinomial: test_prior_hyperparameters_assign_valid02 -- assign
    prior hyperparameter using subset of hyperparameters...
    """
    alphabet = setup['alphabet']
    multinomp = setup['multinomp']

    hp_full = {'a_{}'.format(l): 1 for l in alphabet}
    hp_full['a_a'] = 10
    hp_full['a_c'] = 2
    # assign using only a_a and a_c; others default values
    multinomp.prior_hyperparameters = {'a_a': 10, 'a_c': 2}

    assert multinomp.prior_hyperparameters == hp_full


def test_prior_hyperparameters_assign_invalid01(setup):
    """
    * multinomial: test_prior_hyperparameters_assign_invalid01 -- assign
    prior hyperparameter with invalid dict...
    """
    alphabet = setup['alphabet']
    multinomp = setup['multinomp']

    # incorrect hyperparameter names
    hp = {'param_{}'.format(l): 5 for l in alphabet}

    with pytest.raises(ConjugateParameterException):
        multinomp.prior_hyperparameters = hp


def test_prior_hyperparameters_assign_invalid02(setup):
    """
    * multinomial: test_prior_hyperparameters_assign_invalid02 -- assign
    prior hyperparameter with invalid dict...
    """
    multinomp = setup['multinomp']

    with pytest.raises(ConjugateParameterException):
        # negative value
        multinomp.prior_hyperparameters = {'a_a': 10, 'a_b': -3}


def test_prior_hyperparameters_assign_invalid03(setup):
    """
    * multinomial: test_prior_hyperparameters_assign_invalid03 -- assign
    prior hyperparameter with list-- not allowed...
    """
    multinomp = setup['multinomp']

    with pytest.raises(ConjugateParameterException):
        # try to use list -- doesn't make sense
        multinomp.prior_hyperparameters = [10, 2, 5, 8]


def test_prior_mean_invalid01(setup):
    """
    * multinomial: test_prior_mean_invalid01 -- try to access prior mean
    for parameter that doesn't make sense...
    """
    multinomp = setup['multinomp']

    with pytest.raises(ConjugateParameterException):
        multinomp.prior_mean('p_nonsense')


def test_prior_mean_valid01(setup):
    """
    * multinomial: test_prior_mean_valid01 -- simple test of prior mean...
    """
    multinomp = setup['multinomp']

    for p in multinomp:
        assert multinomp.prior_mean(p) == 1/4


def test_prior_mean_valid02(setup):
    """
    * multinomial: test_prior_mean_valid02 -- assign prior hyperparameters
    and do simple test of prior mean...
    """
    multinomp = setup['multinomp']

    multinomp.prior_hyperparameters = {'a_a': 3}
    for p in multinomp:
        if p == 'p_a':
            assert multinomp.prior_mean(p) == 3/6
        else:
            assert multinomp.prior_mean(p) == 1/6
