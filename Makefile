.PHONY: help pypi pypi-test docs coverage test clean

help:
	@echo "pypi - submit to PyPI server"
	@echo "pypi-test - submit to TestPyPI server"
	@echo "docs - generate Sphinx documentation"
	@echo "coverage - check code coverage"
	@echo "test - run unit tests"
	@echo "clean - remove artifacts"

pypi:
	python setup.py sdist bdist_wheel
	twine upload dist/*

pypi-test:
	python setup.py sdist bdist_wheel
	twine upload --repository-url https://test.pypi.org/legacy/ dist/*

docs:
	rm -rf docs/api
	sphinx-apidoc -o docs/source vip_hci
	cd docs/source/
	$(MAKE) -C docs clean
	$(MAKE) -C docs html

coverage:
	coverage run --source=vip_hci -m pytest
	coverage report -m

test:
	pytest --cov=vip_hci/ --cov-report=xml

pep8-format:
	autopep8 --in-place --aggressive vip_hci/*.py
	autopep8 --in-place --aggressive vip_hci/*/*.py
	autopep8 --in-place --aggressive tests/*.py

clean:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/
	rm -rf vip_hci.egg-info/
	rm -rf .pytest_cache/
	rm -f .coverage
