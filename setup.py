#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2015 Christopher C. Strelioff <chris.strelioff@gmail.com>
#
# Distributed under terms of the MIT license.

"""setup.py

Setup for conjugate project.
"""
from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the relevant file
with open(path.join(here, 'DESCRIPTION.rst'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='conjugate',
    version='0.1',
    description='A Python package for Bayesian inference with conjugate priors',
    long_description=long_description,
    url='https://github.com/cstrelioff/conjugate',
    author='Christopher C. Strelioff',
    author_email='chris.strelioff@gmail.com',
    license='MIT',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.4',
    ],
    keywords='statistics inference Bayesian',
    packages=['conjugate'],
    install_requires=[
        'future>=0.15.2',
        'numpy>=1.9.3',
        'scipy>=0.16.0',
        'matplotlib>=1.4.3'
    ],
    # install using
    # $ pip install -e .[doc,test]
    # -- or --
    # $ pip install -e .[doc]
    # $ pip install -e .[test]
    extras_require={
        'doc': ['sphinx', 'sphinx-bootstrap-theme'],
        'test': ['nose', 'coverage', 'flake8']
    }
)

