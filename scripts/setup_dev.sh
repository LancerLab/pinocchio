#!/bin/bash

# Setup development environment for Pinocchio project
# This script ensures all pre-commit dependencies are properly installed

set -e

echo "Setting up development environment..."

# Install poetry dependencies
echo "Installing poetry dependencies..."
poetry install

# Install pre-commit hooks
echo "Installing pre-commit hooks..."
poetry run pre-commit install

# Install additional dependencies that pre-commit might need
echo "Installing additional pre-commit dependencies..."
poetry run pip install tomli click pyflakes

# Clean and reinstall pre-commit hooks
echo "Cleaning and reinstalling pre-commit hooks..."
poetry run pre-commit clean
poetry run pre-commit install

echo "Development environment setup complete!"
echo "You can now run: git commit -m 'your message'"
