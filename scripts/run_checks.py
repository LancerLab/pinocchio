#!/usr/bin/env python3
"""
Local check script that runs the same checks as GitHub Actions.

This script runs:
1. Linting with flake8
2. Code formatting check with black
3. Import sorting check with isort
4. Type checking with mypy
5. Unit tests with pytest and coverage

Usage:
    poetry run python scripts/run_checks.py [--fix] [--tests-only] [--lint-only]

Options:
    --fix           Auto-fix formatting issues with black and isort
    --tests-only    Only run the tests with coverage
    --lint-only     Only run the linting checks (flake8, black, isort, mypy)
"""

import argparse
import os
import subprocess
import sys
from typing import List, Tuple


def run_command(
    cmd: List[str], description: str, check: bool = True
) -> Tuple[int, str, str]:
    """Run a command and return the exit code, stdout, and stderr."""
    print(f"\n\033[1;34m=== {description} ===\033[0m")
    print(f"Running: {' '.join(cmd)}")

    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    stdout, stderr = process.communicate()

    if process.returncode != 0 and check:
        print(f"\033[1;31mFAILED: {description}\033[0m")
        print(f"Exit code: {process.returncode}")
        if stdout:
            print(f"Standard output:\n{stdout}")
        if stderr:
            print(f"Standard error:\n{stderr}")
    elif process.returncode == 0:
        print(f"\033[1;32mPASSED: {description}\033[0m")

    return process.returncode, stdout, stderr


def run_flake8() -> int:
    """Run flake8 linting checks."""
    # First run: stop the build if there are Python syntax errors or undefined names
    cmd1 = [
        "poetry",
        "run",
        "flake8",
        "pinocchio/",
        "tests/",
        "--count",
        "--select=E9,F63,F7,F82",
        "--show-source",
        "--statistics",
    ]
    code1, _, _ = run_command(cmd1, "Linting with flake8 (critical errors)")

    # Second run: check for other issues but don't fail
    cmd2 = [
        "poetry",
        "run",
        "flake8",
        "pinocchio/",
        "tests/",
        "--count",
        "--exit-zero",
        "--max-complexity=10",
        "--max-line-length=88",
        "--statistics",
    ]
    code2, _, _ = run_command(cmd2, "Linting with flake8 (warnings)", check=False)

    return code1  # Only return the first code as that's what would fail CI


def run_black(fix: bool = False) -> int:
    """Run black formatting check."""
    cmd = ["poetry", "run", "black"]
    if not fix:
        cmd.append("--check")
    cmd.append(".")

    description = "Formatting with black" + (" (fixing)" if fix else " (checking)")
    code, _, _ = run_command(cmd, description)
    return code


def run_isort(fix: bool = False) -> int:
    """Run isort import sorting check."""
    cmd = ["poetry", "run", "isort"]
    if not fix:
        cmd.append("--check")
    cmd.append(".")

    description = "Import sorting with isort" + (" (fixing)" if fix else " (checking)")
    code, _, _ = run_command(cmd, description)
    return code


def run_mypy() -> int:
    """Run mypy type checking."""
    cmd = ["poetry", "run", "mypy", "pinocchio"]
    code, _, _ = run_command(cmd, "Type checking with mypy")
    return code


def run_pytest(with_coverage: bool = True) -> int:
    """Run pytest with optional coverage."""
    cmd = ["poetry", "run", "pytest", "tests/"]
    if with_coverage:
        cmd.extend(["--cov=pinocchio", "--cov-report=xml", "--cov-report=term"])

    description = "Running tests" + (" with coverage" if with_coverage else "")
    code, _, _ = run_command(cmd, description)
    return code


def main() -> int:
    """Run all checks and return an exit code."""
    parser = argparse.ArgumentParser(description="Run local checks for the project.")
    parser.add_argument("--fix", action="store_true", help="Auto-fix formatting issues")
    parser.add_argument("--tests-only", action="store_true", help="Only run tests")
    parser.add_argument(
        "--lint-only", action="store_true", help="Only run linting checks"
    )

    args = parser.parse_args()

    exit_codes = []

    # Make sure we're in the project root directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(os.path.dirname(script_dir))

    if not args.tests_only:
        # Linting and formatting checks
        exit_codes.append(run_flake8())
        exit_codes.append(run_black(fix=args.fix))
        exit_codes.append(run_isort(fix=args.fix))
        exit_codes.append(run_mypy())

    if not args.lint_only:
        # Tests and coverage
        exit_codes.append(run_pytest(with_coverage=True))

    # Summary
    failed = sum(1 for code in exit_codes if code != 0)
    total = len(exit_codes)

    if failed == 0:
        print(f"\n\033[1;32mAll {total} checks passed!\033[0m")
        return 0
    else:
        print(f"\n\033[1;31m{failed} of {total} checks failed!\033[0m")
        return 1


if __name__ == "__main__":
    sys.exit(main())
