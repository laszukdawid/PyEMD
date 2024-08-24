LINT_TARGET_DIRS := PyEMD doc example

init:
	python -m venv .venv
	.venv/bin/pip install -r requirements.txt
	.venv/bin/pip install -e .[dev]
	@echo "Run 'source .venv/bin/activate' to activate the virtual environment"

test:
	python -m PyEMD.tests.test_all

clean:
	find PyEMD -name __pycache__ -execdir rm -r {} +

.PHONY: doc
doc:
	cd doc && make html

format:
	python -m black $(LINT_TARGET_DIRS)
	python -m isort PyEMD

lint-check:
	python -m isort --check PyEMD
	python -m black --check $(LINT_TARGET_DIRS)

perf/build:
	docker build -t pyemd-perf -f perf_test/Dockerfile .