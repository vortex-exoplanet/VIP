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
	rm -f docs/source/vip_hci.config.rst
	rm -f docs/source/vip_hci.fits.rst
	rm -f docs/source/vip_hci.fm.rst
	rm -f docs/source/vip_hci.hci_dataset.rst
	rm -f docs/source/vip_hci.hci_postproc.rst
	rm -f docs/source/vip_hci.invprob.rst
	rm -f docs/source/vip_hci.metrics.rst
	rm -f docs/source/vip_hci.preproc.rst
	rm -f docs/source/vip_hci.psfsub.rst
	rm -f docs/source/vip_hci.rst
	rm -f docs/source/vip_hci.stats.rst
	rm -f docs/source/vip_hci.var.rst
	rm -f docs/source/vip_hci.vip_ds9.rst
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
