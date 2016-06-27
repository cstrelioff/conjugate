install:
	pip install -e .

install-doc:
	pip install -e .[doc]

install-test:
	pip install -e .[test]

test:
	python setup.py nosetests

test-binomial:
	py.test --cov-report term-missing --cov=conjugate tests/test_binomial.py

test-multinomial:
	py.test --cov-report term-missing --cov=conjugate tests/test_multinomial.py
