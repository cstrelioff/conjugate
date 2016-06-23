install:
	pip install -e .

install-doc:
	pip install -e .[doc]

install-test:
	pip install -e .[test]

test:
	python setup.py nosetests
