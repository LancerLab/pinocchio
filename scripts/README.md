# Development Scripts

This directory contains utility scripts for development and maintenance of the Pinocchio project.

## `run_checks.sh` / `run_checks.py`

These scripts run the same checks that are performed by GitHub Actions CI pipeline, allowing you to catch issues before pushing your code.

### Features

- Linting with flake8
- Code formatting check with black
- Import sorting check with isort
- Type checking with mypy
- Unit tests with pytest and coverage

### Usage

```bash
# Run all checks
./scripts/run_checks.sh

# Auto-fix formatting issues (black and isort)
./scripts/run_checks.sh --fix

# Only run tests with coverage
./scripts/run_checks.sh --tests-only

# Only run linting checks (flake8, black, isort, mypy)
./scripts/run_checks.sh --lint-only
```

You can also run the Python script directly:

```bash
poetry run python scripts/run_checks.py [--fix] [--tests-only] [--lint-only]
```

### Exit Codes

- `0`: All checks passed
- `1`: One or more checks failed

## `check_chinese_chars.py`

Utility to check for Chinese characters in files that should only contain English.

## `setup_dev.sh`

Setup script for development environment.
