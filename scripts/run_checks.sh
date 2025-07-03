#!/usr/bin/env bash
# Run local checks that mirror GitHub Actions CI pipeline
# Usage: ./scripts/run_checks.sh [--fix] [--tests-only] [--lint-only]

set -e

# Change to the project root directory
cd "$(dirname "$0")/.."

# Check if poetry is installed
if ! command -v poetry &> /dev/null; then
    echo "Poetry is not installed. Please install it first:"
    echo "curl -sSL https://install.python-poetry.org | python3 -"
    exit 1
fi

# Check if the virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "No virtual environment is active. Activating poetry environment..."
    if [ -f .venv/bin/activate ]; then
        source .venv/bin/activate
    else
        echo "Poetry virtual environment not found. Installing dependencies..."
        poetry install
    fi
fi

# Run the Python script with the same arguments
poetry run python scripts/run_checks.py "$@"
