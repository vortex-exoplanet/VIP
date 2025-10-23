.PHONY: help pypi pypi-test docs coverage test clean

help:
	@echo "pypi - submit to PyPI server"
	@echo "pypi-test - submit to TestPyPI server"
	@echo "docs - generate Sphinx documentation"
	@echo "coverage - check code coverage"
	@echo "test - run unit tests"
	@echo "clean - remove artifacts"

pypi:
	rm dist/*
	rm -rf build/*
	rm -r vip_hci.egg-info
	pip install --upgrade pip setuptools wheel build twine
	python -m build
	twine upload dist/*

pypi-test:
	rm dist/*
	rm -rf build/*
	rm -r vip_hci.egg-info
	pip install --upgrade pip setuptools wheel build twine
	python -m build
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
	sphinx-apidoc -o docs/source src/vip_hci
	cd docs/source/
	$(MAKE) -C docs clean
	$(MAKE) -C docs html

coverage:
	coverage run --source=src -m pytest
	coverage report -m

test:
	pre-commit clean
	pre-commit install --hook-type pre-merge-commit
	pre-commit install --hook-type pre-push
	pre-commit install --hook-type post-rewrite
	pre-commit install-hooks
	pre-commit install
	pre-commit run --files src/**/*.py
	coverate run -m pytest
	coverage xml
	rm confi_hist.pdf
	rm confi_hist_gaussfit.pdf
	rm confidence.txt
	rm corner_plot.pdf
	rm walk_plot.pdf
	rm -rf results/

pep8-format:
	autopep8 --in-place --aggressive src/**/*.py
	autopep8 --in-place --aggressive tests/*.py

clean:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/
	rm -rf vip_hci.egg-info/
	rm -rf .pytest_cache/
	rm -f .coverage
