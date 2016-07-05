BinomialBeta
============

The :class:`BinomialBeta` class is used to infer Binomial parameter :math:`p`-- the
probability of success-- given a data set with :math:`k` *successes* in
:math:`n` *attempts*.

Quick start
-----------

A simple inference session using :class:`BinomialBeta` would go something like
this if you are using :code:`ipython` and Python 3:

.. sourcecode:: ipython

  In [1]: import conjugate

  In [2]: bp = conjugate.BinomialBeta()

  In [3]: bp.add_data({'n': 10, 'k':9})

  In [4]: print(bp)
  bp = BinomialBeta()
  bp.data = {'n': 10, 'k': 9}
  bp.prior_hyperparameters = {'alpha': 1, 'beta': 1}

  In [5]: print("prior mean    : {prior:.2f}\n"
     ...:       "posterior mean: {post:.2f}".format(prior=bp.prior_mean('p'),
     ...:                                           post=bp.posterior_mean('p')))
  prior mean    : 0.50
  posterior mean: 0.83

Of course, the normal :code:`python` terminal would work just as well!

If you are using Python 2.7 it is good practice to use :code:`__future__`
imports to enable Python 3 style code.  So, a session with :code:`ipython` using
Python 2.7 would look like this:

.. sourcecode:: ipython

  In [1]: from __future__ import print_function

  In [2]: from __future__ import division

  In [3]: from __future__ import unicode_literals

  In [4]: import conjugate

  In [5]: bp = conjugate.BinomialBeta()

  In [6]: bp.add_data({'n': 10, 'k':9})

  In [7]: print(bp)
  bp = BinomialBeta()
  bp.data = {'n': 10, 'k': 9}
  bp.prior_hyperparameters = {'alpha': 1, 'beta': 1}

  In [8]: print("prior mean    : {prior:.2f}\n"
     ...:       "posterior mean: {post:.2f}".format(prior=bp.prior_mean('p'),
     ...:                                           post=bp.posterior_mean('p')))
  prior mean    : 0.50
  posterior mean: 0.83

Of course, the results are the same!

Mathematical details
--------------------

The Binomial probability mass function (pmf), as well as the form of the
likelihood function, is given by:

.. math::

   \Pr(D=k \vert \theta=p; n) = {n \choose k} p^{k} (1-p)^{n-k}

The *conjugate prior* for the Binomial pmf is the Beta probability density
function (pdf) with parameters :math:`(\alpha, \beta)`:

.. math::

   \Pr(\theta=p) = \frac{\Gamma(\alpha + \beta)}{
                         \Gamma(\alpha) \Gamma(\beta)}
                     p^{\alpha - 1} (1 - p)^{\beta - 1}

The resulting posterior is also a Beta pdf, now with parameters
:math:`(\alpha + k, \beta + n - k)`:

.. math::

   \Pr(\theta=p \vert D=k; n) =
       \frac{\Gamma(\alpha + \beta + n)}{
                      \Gamma(\alpha + k) \Gamma(\beta + n - k)}
                      p^{\alpha + k - 1} (1 - p)^{\beta + n - k - 1}

