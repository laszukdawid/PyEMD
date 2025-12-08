import nox

# Use uv for faster dependency installation and Python version management
nox.options.default_venv_backend = "uv"

# Test against all currently supported Python versions (3.10+)
PYTHON_VERSIONS = ["3.9", "3.10", "3.11", "3.12", "3.13"]


@nox.session(python=PYTHON_VERSIONS)
def tests(session: nox.Session) -> None:
    """Run the test suite across Python versions."""
    session.install(".[test]")
    session.run("python", "-m", "PyEMD.tests.test_all")


@nox.session(python=PYTHON_VERSIONS)
def tests_pytest(session: nox.Session) -> None:
    """Run the test suite with pytest across Python versions."""
    session.install(".[test]")
    session.run("pytest", "PyEMD/tests/", "-v")


@nox.session(python=PYTHON_VERSIONS[-1])  # Latest Python only
def lint(session: nox.Session) -> None:
    """Run linting checks."""
    session.install(".[dev]")
    session.run("isort", "--check", "PyEMD")
    session.run("black", "--check", "PyEMD", "doc", "example")


@nox.session(python=PYTHON_VERSIONS[-1])  # Latest Python only
def typecheck(session: nox.Session) -> None:
    """Run type checking (if mypy is added later)."""
    session.install(".", "mypy")
    session.run("mypy", "PyEMD", "--ignore-missing-imports")
