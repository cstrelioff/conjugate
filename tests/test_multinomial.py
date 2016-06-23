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

import unittest

from conjugate import PosteriorBase
from conjugate import MultinomialDirichlet
from conjugate import ConjugateDataException
from conjugate import ConjugateParameterException


class MultinomialDirichletTest(unittest.TestCase):
    """Test the MultinomialDirichlet class."""

    def setUp(self):
        self.alphabet = [str('a'), str('b'), str('c'), str('d')]
        self.multinomp = MultinomialDirichlet(self.alphabet)

    def tearDown(self):
        del self.alphabet
        del self.multinomp

    def test_instantiate(self):
        """
        * multinomial: test_instantiate -- instantiate and test is instance
        of PosteriorBase ABC.
        """
        self.assertIsInstance(self.multinomp, PosteriorBase)

    def test_contains(self):
        """
        * mulitnomial: test_contains -- make sure p in self works...
        """
        self.assertTrue('p_a' in self.multinomp)

    def test_contains_invalid(self):
        """
        * mulitnomial: test_contains_invalid -- try invalid parameter...
        """
        self.assertFalse('p_nonsense' in self.multinomp)

    def test_iter(self):
        """
        * multinomial: test_iter -- test the __iter__ method...
        """
        params = []
        for item in self.multinomp:
            params.append(item)

        self.assertListEqual(params, ['p_a', 'p_b', 'p_c', 'p_d'])

    def test_str(self):
        """
        * multinomial: test_str -- does __str__ method work?, yup...
        """
        data = {str('a'): 10, str('b'): 0, str('c'): 5, str('d'): 2}
        self.multinomp.data = data
        prior = self.multinomp.prior_hyperparameters
        tmp = ('mp = MultinomialDirichlet({})\n'
               'mp.data = {}\n'
               'mp.prior_hyperparameters = {}'.format(self.alphabet,
                                                      data,
                                                      prior))
        self.assertTrue(tmp == str(self.multinomp))

    def test_data_setter_valid01(self):
        """
        * multinomial: test_data_setter_valid01 -- use the data setter with
        valid, simple input
        """
        data = {'a': 10, 'b': 0, 'c': 5, 'd': 2}
        self.multinomp.data = data

        self.assertDictEqual(data, self.multinomp.data)

    def test_data_setter_valid02(self):
        """
        * multinomial: test_data_setter_valid02 -- use the data setter with
        valid input; leave off one key-value pair
        """
        data = {'a': 10, 'b': 0, 'd': 2}
        self.multinomp.data = data

        cnta = (self.multinomp.data['a'] == 10)
        cntb = (self.multinomp.data['b'] == 0)
        cntc = (self.multinomp.data['c'] == 0)  # this should b okay!
        cntd = (self.multinomp.data['d'] == 2)
        self.assertTrue(cnta and cntb and cntc and cntd)

    def test_data_setter_invalid01(self):
        """
        * multinomial: test_data_setter_invalid01 -- use the data setter with
        invalid input; bad key.
        """
        data = {'a': 10, 'b': 0, 'c': 5, 'g': 2}

        with self.assertRaises(ConjugateDataException):
            self.multinomp.data = data

    def test_data_setter_invalid02(self):
        """
        * multinomial: test_data_setter_invalid02 -- use the data setter with
        invalid input; negative value.
        """
        data = {'a': 10, 'b': 0, 'c': -2}

        with self.assertRaises(ConjugateDataException):
            self.multinomp.data = data

    def test_data_setter_invalid03(self):
        """
        * multinomial: test_data_setter_invalid03 -- use the data setter with
        invalid input; data as list.
        """
        data = [10, 0, 2]

        with self.assertRaises(ConjugateDataException):
            self.multinomp.data = data

    def test_distribution(self):
        """
        * multinomial: test_distribution -- returns str with dist name...
        """
        self.assertTrue(self.multinomp.distribution ==
                        'Distribution: Multinomial')

    def test_distribution_parameter_names_valid(self):
        """
        * multinomial: test_distribution_parameter_names_valid -- basic test
        of returned list...
        """
        self.assertListEqual(self.multinomp.distribution_parameter_names,
                             ['p_a', 'p_b', 'p_c', 'p_d'])

    def test_distribution_parameter_names_invalid(self):
        """
        * multinomial: test_distribution_parameter_names_invalid -- try to
        set the parameter names; can't, they're read-only...
        """
        with self.assertRaises(AttributeError):
            self.multinomp.distribution_parameter_names = ['p_one', 'p_two']

    def test_distribution_parameter_supoort(self):
        """
        * multinomial: test_distribution_parameter_support -- make sure all
        supports are (0.0, 1.0)...
        """
        sups = []
        for p in self.multinomp:
            sups.append(self.multinomp.distribution_parameter_support[p] ==
                        (0.0, 1.0))

        self.assertTrue(sups.count(True) == 4)

    def test_prior(self):
        """
        * multinomial: test_prior -- returns str with prior name...
        """
        self.assertTrue(self.multinomp.prior ==
                        'Prior: Dirichlet')

    def test_prior_hyperparameter_names(self):
        """
        * multinomial: test_prior_hyperparameter_names -- returns list with
        prior hyperparameter name...
        """
        self.assertListEqual(self.multinomp.prior_hyperparameter_names,
                             ['a_a', 'a_b', 'a_c', 'a_d'])

    def test_prior_hyperparameter_names_assign(self):
        """
        * multinomial: test_prior_hyperparameter_names_assign -- try to assign
        prior hyperparameter names -- not allowed...
        """
        with self.assertRaises(AttributeError):
            self.multinomp.prior_hyperparameter_names = ['a_fred', 'a_ted']

    def test_prior_hyperparameters(self):
        """
        * multinomial: test_prior_hyperparameters -- returns dict with
        prior hyperparameter names and values...
        """
        hp = {'a_{}'.format(l): 1 for l in self.alphabet}

        self.assertDictEqual(self.multinomp.prior_hyperparameters, hp)

    def test_prior_hyperparameters_assign_valid01(self):
        """
        * multinomial: test_prior_hyperparameters_assign_valid01 -- assign
        prior hyperparameter with valid dict...
        """
        hp = {'a_{}'.format(l): 5 for l in self.alphabet}
        self.multinomp.prior_hyperparameters = hp

        self.assertDictEqual(self.multinomp.prior_hyperparameters, hp)

    def test_prior_hyperparameters_assign_valid02(self):
        """
        * multinomial: test_prior_hyperparameters_assign_valid02 -- assign
        prior hyperparameter using subset of hyperparameters...
        """
        hp = {'a_a': 10, 'a_c': 2}
        self.multinomp.prior_hyperparameters = hp

    def test_prior_hyperparameters_assign_invalid01(self):
        """
        * multinomial: test_prior_hyperparameters_assign_invalid01 -- assign
        prior hyperparameter with invalid dict...
        """
        # incorrect hyperparameter names
        hp = {'param_{}'.format(l): 5 for l in self.alphabet}

        with self.assertRaises(ConjugateParameterException):
            self.multinomp.prior_hyperparameters = hp

    def test_prior_hyperparameters_assign_invalid02(self):
        """
        * multinomial: test_prior_hyperparameters_assign_invalid02 -- assign
        prior hyperparameter with invalid dict...
        """
        with self.assertRaises(ConjugateParameterException):
            # negative value
            self.multinomp.prior_hyperparameters = {'a_a': 10, 'a_b': -3}

    def test_prior_hyperparameters_assign_invalid03(self):
        """
        * multinomial: test_prior_hyperparameters_assign_invalid03 -- assign
        prior hyperparameter with list-- not allowed...
        """
        with self.assertRaises(ConjugateParameterException):
            # try to use list -- doesn'tmake sense
            self.multinomp.prior_hyperparameters = [10, 2, 5, 8]

    def test_prior_mean_invalid01(self):
        """
        * multinomial: test_prior_mean_invalid01 -- try to access prior mean
        for parameter that doesn't make sense...
        """
        with self.assertRaises(ConjugateParameterException):
            self.multinomp.prior_mean('p_nonsense')

    def test_prior_mean_valid01(self):
        """
        * multinomial: test_prior_mean_valid01 -- simple test of prior mean...
        """
        pa = (self.multinomp.prior_mean('p_a') == 1/4)
        pb = (self.multinomp.prior_mean('p_b') == 1/4)
        pc = (self.multinomp.prior_mean('p_c') == 1/4)
        pd = (self.multinomp.prior_mean('p_d') == 1/4)

        self.assertTrue(pa and pb and pc and pd)

    def test_prior_mean_valid02(self):
        """
        * multinomial: test_prior_mean_valid02 -- assign prior hyperparameters
        and do simple test of prior mean...
        """
        self.multinomp.prior_hyperparameters = {'a_a': 3}

        pa = (self.multinomp.prior_mean('p_a') == 3/6)
        pb = (self.multinomp.prior_mean('p_b') == 1/6)
        pc = (self.multinomp.prior_mean('p_c') == 1/6)
        pd = (self.multinomp.prior_mean('p_d') == 1/6)

        self.assertTrue(pa and pb and pc and pd)
