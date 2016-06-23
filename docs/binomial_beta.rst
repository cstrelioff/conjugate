Binomial with Beta Prior
========================

Infer Binomial parameter :math:`p` given data :math:`D=k`, where
:math:`k` is the *number of successes* in :math:`n` *attempts*.

Usage
-----

**To do.**

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

