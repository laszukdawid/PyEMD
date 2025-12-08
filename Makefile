LINT_TARGET_DIRS := PyEMD doc example

.PHONY: init sync test clean doc format lint-check freeze nox nox-lint
.PHONY: perf perf-quick perf-scaling perf-splines perf-extrema perf-eemd perf-ceemdan perf-complexity perf-sifting perf-compare perf-list perf/build

# Development setup
init:
	uv sync --all-extras
	@echo "Development environment ready. Run 'source .venv/bin/activate' to activate."

sync:
	uv sync --all-extras

# Testing
test:
	uv run python -m PyEMD.tests.test_all

test-pytest:
	uv run pytest PyEMD/tests/

# Multi-version testing with nox
nox:
	uv run nox -s tests

nox-lint:
	uv run nox -s lint

nox-all:
	uv run nox

# Code quality
format:
	uv run black $(LINT_TARGET_DIRS)
	uv run isort PyEMD

lint-check:
	uv run isort --check PyEMD
	uv run black --check $(LINT_TARGET_DIRS)

# Documentation
doc:
	cd doc && make html

# Cleanup
clean:
	find PyEMD -name __pycache__ -execdir rm -r {} +
	rm -rf .venv

# Export requirements for pip users
freeze:
	uv export --no-hashes --no-dev --no-emit-project -o requirements.txt
	uv export --no-hashes --only-dev --no-emit-project -o requirements-dev.txt
	@echo "Exported requirements.txt and requirements-dev.txt"

# Performance tests
# Results saved to perf_test/results/<timestamp>/

perf:
	@echo "Running full performance test suite..."
	uv run python perf_test/perf_test_comprehensive.py

perf-quick:
	@echo "Running quick performance test suite..."
	uv run python perf_test/perf_test_comprehensive.py --quick

perf-scaling:
	@echo "Running EMD scaling test..."
	uv run python perf_test/perf_test_comprehensive.py --test scaling

perf-splines:
	@echo "Running spline comparison test..."
	uv run python perf_test/perf_test_comprehensive.py --test splines

perf-extrema:
	@echo "Running extrema detection test..."
	uv run python perf_test/perf_test_comprehensive.py --test extrema

perf-eemd:
	@echo "Running EEMD parallel scaling test..."
	uv run python perf_test/perf_test_comprehensive.py --test eemd

perf-ceemdan:
	@echo "Running CEEMDAN performance test..."
	uv run python perf_test/perf_test_comprehensive.py --test ceemdan

perf-complexity:
	@echo "Running signal complexity test..."
	uv run python perf_test/perf_test_comprehensive.py --test complexity

perf-sifting:
	@echo "Running sifting parameters test..."
	uv run python perf_test/perf_test_comprehensive.py --test sifting

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
	uv run python perf_test/compare_results.py "$$BASELINE" "$$COMPARISON"

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
	@echo "  uv run python perf_test/compare_results.py <baseline> <comparison>"
	@echo ""
	@echo "Results saved to: perf_test/results/<timestamp>_<test>/"

perf/build:
	docker build -t pyemd-perf -f perf_test/Dockerfile .
