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

import unittest

from conjugate import PosteriorBase
from conjugate import BinomialBeta
from conjugate import ConjugateDataException
from conjugate import ConjugateParameterException


class BinomialBetaTest(unittest.TestCase):
    """Test the BinomialBeta class."""

    def setUp(self):
        self.binomp = BinomialBeta()

    def tearDown(self):
        del self.binomp

    def test_instantiate(self):
        """
        * binomial: test_instantiate -- instantiate with no arguments.
        """
        self.assertIsInstance(self.binomp, PosteriorBase)

    def test_contains(self):
        """
        * binomial: test_contains -- test the __contains__ method.
        """
        self.assertTrue('p' in self.binomp)

    def test_contains_invalid(self):
        """
        * binomial: test_contains_invalid -- test the __contains__ method w
        bad parameter.
        """
        self.assertFalse('a' in self.binomp)

    def test_iter(self):
        """
        * binomial: test_iter -- test the __iter__ method.
        """
        params = []
        for item in self.binomp:
            params.append(item)

        self.assertTrue(params == ['p'])

    def test_str(self):
        """
        * binomial: test_str -- test the __str__ method.
        """
        data = {'n': 10, 'k': 2}
        self.binomp.data = data
        prior = self.binomp.prior_hyperparameters

        tmp = ('bp = BinomialBeta()\n'
               'bp.data = {}\n'
               'bp.prior_hyperparameters = {}'.format(data, prior))

        self.assertTrue(str(self.binomp) == tmp)

    def test_distribution(self):
        """
        * binomial: test_distribution -- test the distribution string property.
        """
        self.assertTrue(self.binomp.distribution == 'Distribution: Binomial')

    def test_distribution_parameter_names(self):
        """
        * binomial: test_distribution_parameter_names -- test the distribution
        parameter names property.
        """
        self.assertTrue(self.binomp.distribution_parameter_names == ['p'])

    def test_distribution_parameter_support(self):
        """
        * binomial: test_distribution_parameter_support -- test the
        distribution parameter support property.
        """
        self.assertTrue(self.binomp.distribution_parameter_support ==
                        {'p': (0.0, 1.0)})

    def test_prior(self):
        """
        * binomial: test_prior -- test the prior string property.
        """
        self.assertTrue(self.binomp.prior == 'Prior: Beta')

    def test_prior_hyperparameter_names(self):
        """
        * binomial: test_prior_hyperparameter_names -- test the prior
        hyperparameter names property.
        """
        self.assertTrue(self.binomp.prior_hyperparameter_names ==
                        ['alpha', 'beta'])

    def test_prior_hyperparameters_valid(self):
        """
        * binomial: test_prior_hyperparameters_valid -- set prior
        hyperparameter values.
        """
        new_setting = {'alpha': 5, 'beta': 10}
        self.binomp.prior_hyperparameters = new_setting

        self.assertTrue(self.binomp.prior_hyperparameters == new_setting)

    def test_prior_hyperparameters_invalid01(self):
        """
        * binomial: test_prior_hyperparameters_invalid01 -- try to set
        invalid prior hyperparameter values.
        """
        # incorrect parameter names -- keys in dict
        new_prior_setting = {'alpha0': 5, 'alpha1': 10}

        with self.assertRaises(ConjugateParameterException):
            self.binomp.prior_hyperparameters = new_prior_setting

    def test_prior_hyperparameters_invalid02(self):
        """
        * binomial: test_prior_hyperparameters_invalid02 -- try to set
        invalid prior hyperparameter values.
        """
        # parameters values <= 0
        new_prior_setting = {'alpha': 5, 'beta': -1}

        with self.assertRaises(ConjugateParameterException):
            self.binomp.prior_hyperparameters = new_prior_setting

    def test_prior_hyperparameters_invalid03(self):
        """
        * binomial: test_prior_hyperparameters_invalid03 -- try to set
        invalid prior hyperparameter values.
        """
        # pass list instead of dict
        new_prior_setting = [5, 10]

        with self.assertRaises(ConjugateParameterException):
            self.binomp.prior_hyperparameters = new_prior_setting

    def test_prior_mean_valid(self):
        """
        * binomial: test_prior_mean_valid -- get prior mean for valid
        parameter.
        """
        self.assertEqual(self.binomp.prior_mean('p'), 1/2)

    def test_prior_mean_invalid(self):
        """
        * binomial: test_prior_mean_invalid -- (try to) get prior mean for
        invalid parameter.
        """
        with self.assertRaises(ConjugateParameterException):
            self.binomp.prior_mean('a')

    def test_posterior_mean_valid(self):
        """
        * binomial: test_posterior_mean_valid -- get posterior mean for valid
        parameter.
        """
        self.binomp.add_data({'n': 5, 'k': 2})

        self.assertEqual(self.binomp.posterior_mean('p'), (1+2)/(2+5))

    def test_posterior_mean_invalid(self):
        """
        * binomial: test_posterior_mean_invalid -- (try to) get posterior mean
        for invalid parameter.
        """
        self.binomp.add_data({'n': 5, 'k': 2})

        with self.assertRaises(ConjugateParameterException):
            self.binomp.posterior_mean('a')

    def test_posterior_hdcr_valid(self):
        """
        * binomial: test_posterior_hdcr_valid -- get hdcr for posterior.
        """
        self.binomp.add_data({'n': 5, 'k': 2})

        hdcr = self.binomp.posterior_high_density_credible_region('p')
        self.assertEqual(hdcr, [0.10482705290430133, 0.76128981963103848])

    def test_posterior_hdcr_invalid(self):
        """
        * binomial: test_posterior_hdcr_invalid -- (try to) get hdcr for
        invalid parameter.
        """
        self.binomp.add_data({'n': 5, 'k': 2})

        with self.assertRaises(ConjugateParameterException):
            self.binomp.posterior_high_density_credible_region('nonsense')

    def test_posterior_ccr_valid(self):
        """
        * binomial: test_posterior_ccr_valid -- get ccr for posterior.
        """
        self.binomp.add_data({'n': 5, 'k': 2})

        ccr = self.binomp.posterior_central_credible_region('p')
        self.assertEqual(ccr, [0.11811724875702526, 0.77722190449648787])

    def test_posterior_ccr_invalid(self):
        """
        * binomial: test_posterior_ccr_invalid -- (try to) get ccr for
        invalid parameter.
        """
        self.binomp.add_data({'n': 5, 'k': 2})

        with self.assertRaises(ConjugateParameterException):
            self.binomp.posterior_central_credible_region('nonsense')

    def test_data_setter_valid01(self):
        """
        * binomial: test_data_setter_valid02 -- test use of data setter w
        valid data.
        """
        self.binomp.data = {'n': 45, 'k': 40}

        self.assertDictEqual(self.binomp.data, {'n': 45, 'k': 40})

    def test_data_setter_invalid01(self):
        """
        * binomial: test_data_setter_invalid01 -- test use of data setter w
        invalid data-- a list..
        """
        with self.assertRaises(ConjugateDataException):
            self.binomp.data = [0, 0, 1, 1, 1]

    def test_data_setter_invalid02(self):
        """
        * binomial: test_data_setter_invalid02 -- test use of data setter w
        invalid data.
        """
        with self.assertRaises(ConjugateDataException):
            self.binomp.data = {'n': 2, 'k': 5}

    def test_data_setter_invalid03(self):
        """
        * binomial: test_data_setter_invalid03 -- test use of data setter w
        invalid data.
        """
        with self.assertRaises(ConjugateDataException):
            self.binomp.data = (5, 4)

    def test_add_data_valid_dict(self):
        """
        * binomial: test_add_data_valid_dict -- test add_data() method w valid
        data dict.
        """
        self.binomp.add_data({'n': 10, 'k': 2})

        self.assertDictEqual(self.binomp._data, {'n': 10, 'k': 2})

    def test_add_data_invalid_dict(self):
        """
        * binomial: test_add_data_invalid_dict -- test add_data() method with
        an valid data dict (invalid key).
        """
        with self.assertRaises(ConjugateDataException):
            self.binomp.add_data({'m': 10, 'k': 2})
