LINT_TARGET_DIRS := PyEMD doc example
PYTHON := .venv/bin/python

init:
	python -m venv .venv
	.venv/bin/pip install -r requirements.txt
	.venv/bin/pip install -e .[dev]
	@echo "Run 'source .venv/bin/activate' to activate the virtual environment"

test:
	$(PYTHON) -m PyEMD.tests.test_all

clean:
	find PyEMD -name __pycache__ -execdir rm -r {} +

.PHONY: doc
doc:
	cd doc && make html

format:
	$(PYTHON) -m black $(LINT_TARGET_DIRS)
	$(PYTHON) -m isort PyEMD

lint-check:
	$(PYTHON) -m isort --check PyEMD
	$(PYTHON) -m black --check $(LINT_TARGET_DIRS)

# Performance tests
# Results saved to perf_test/results/<timestamp>/

.PHONY: perf perf-quick perf-scaling perf-splines perf-extrema perf-eemd perf-ceemdan perf-complexity perf-sifting

perf:
	@echo "Running full performance test suite..."
	$(PYTHON) perf_test/perf_test_comprehensive.py

perf-quick:
	@echo "Running quick performance test suite..."
	$(PYTHON) perf_test/perf_test_comprehensive.py --quick

perf-scaling:
	@echo "Running EMD scaling test..."
	$(PYTHON) perf_test/perf_test_comprehensive.py --test scaling

perf-splines:
	@echo "Running spline comparison test..."
	$(PYTHON) perf_test/perf_test_comprehensive.py --test splines

perf-extrema:
	@echo "Running extrema detection test..."
	$(PYTHON) perf_test/perf_test_comprehensive.py --test extrema

perf-eemd:
	@echo "Running EEMD parallel scaling test..."
	$(PYTHON) perf_test/perf_test_comprehensive.py --test eemd

perf-ceemdan:
	@echo "Running CEEMDAN performance test..."
	$(PYTHON) perf_test/perf_test_comprehensive.py --test ceemdan

perf-complexity:
	@echo "Running signal complexity test..."
	$(PYTHON) perf_test/perf_test_comprehensive.py --test complexity

perf-sifting:
	@echo "Running sifting parameters test..."
	$(PYTHON) perf_test/perf_test_comprehensive.py --test sifting

perf-compare:
	@if [ $$(ls -d perf_test/results/*/ 2>/dev/null | wc -l) -lt 2 ]; then \
		echo "Error: Need at least 2 result directories to compare"; \
		exit 1; \
	fi
	@BASELINE=$$(ls -dt perf_test/results/*/ | sed -n '2p' | sed 's/\/$$//'); \
	COMPARISON=$$(ls -dt perf_test/results/*/ | sed -n '1p' | sed 's/\/$$//'); \
	echo "Comparing:"; \
	echo "  Baseline:   $$BASELINE"; \
	echo "  Comparison: $$COMPARISON"; \
	echo ""; \
	$(PYTHON) perf_test/compare_results.py "$$BASELINE" "$$COMPARISON"

perf-list:
	@echo "Available performance test targets:"
	@echo "  make perf            - Full test suite"
	@echo "  make perf-quick      - Quick test suite (smaller parameters)"
	@echo "  make perf-scaling    - EMD scaling with signal length"
	@echo "  make perf-splines    - Spline method comparison"
	@echo "  make perf-extrema    - Extrema detection comparison"
	@echo "  make perf-eemd       - EEMD parallel scaling"
	@echo "  make perf-ceemdan    - CEEMDAN performance"
	@echo "  make perf-complexity - Signal complexity impact"
	@echo "  make perf-sifting    - Sifting parameters impact"
	@echo ""
	@echo "Comparison:"
	@echo "  make perf-compare    - Compare two most recent results"
	@echo "  $(PYTHON) perf_test/compare_results.py <baseline> <comparison>"
	@echo ""
	@echo "Results saved to: perf_test/results/<timestamp>_<test>/"

perf/build:
	docker build -t pyemd-perf -f perf_test/Dockerfile .