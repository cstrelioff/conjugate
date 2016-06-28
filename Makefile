install:
	pip install -e .

install-doc:
	pip install -e .[doc]

install-test:
	pip install -e .[test]

test:
	py.test --cov-report term-missing --cov=conjugate tests/

test-binomial:
	py.test -v tests/test_binomial.py

test-multinomial:
	py.test -v tests/test_multinomial.py

test-utilities:
	py.test -v tests/test_utilities.py
