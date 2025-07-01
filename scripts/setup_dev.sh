#!/bin/bash

# Script for setting up development environment and installing pre-commit hooks

# Exit on error
set -e

echo "=== Setting up Pinocchio development environment ==="

# Check if Poetry is installed
if ! command -v poetry &> /dev/null; then
    echo "Poetry not found, installing..."
    curl -sSL https://install.python-poetry.org | python3 -
else
    echo "Poetry is already installed"
fi

# Install project dependencies
echo "Installing project dependencies..."
poetry install

# Install pre-commit hooks
echo "Installing pre-commit hooks..."
poetry run pre-commit install

# Run pre-commit hooks initialization
echo "Running pre-commit hooks initialization..."
poetry run pre-commit run --all-files || true

echo "=== Development environment setup complete ==="
echo "You can run tests with:"
echo "  poetry run pytest"
echo "You can format code with:"
echo "  poetry run black ."
echo "  poetry run isort ."
echo ""
echo "Pre-commit hooks will automatically run on git commit to check and format code"
