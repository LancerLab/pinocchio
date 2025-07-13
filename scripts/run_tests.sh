#!/bin/bash

# Unified test runner for Pinocchio project
# Supports multiple testing modes: fast, dev, local, real-llm, coverage, etc.

set -e

echo "🧪 Pinocchio Test Runner"
echo "========================"

# Set default environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export FAST_TEST=1
export ENABLE_REAL_LLM_TESTS=0

# Parse command line arguments
MODE="dev"  # Default mode
REAL_LLM=false
MOCK_ONLY=false
COVERAGE=false
VERBOSE=false
FAIL_FAST=false
HELP=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --fast)
            MODE="fast"
            shift
            ;;
        --dev)
            MODE="dev"
            shift
            ;;
        --local)
            MODE="local"
            export ENABLE_REAL_LLM_TESTS=1
            shift
            ;;
        --real-llm)
            REAL_LLM=true
            export ENABLE_REAL_LLM_TESTS=1
            shift
            ;;
        --mock-only)
            MOCK_ONLY=true
            shift
            ;;
        --verbose|-v)
            VERBOSE=true
            shift
            ;;
        --coverage)
            COVERAGE=true
            shift
            ;;
        --fail-fast)
            FAIL_FAST=true
            shift
            ;;
        --help|-h)
            HELP=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Show help if requested
if [ "$HELP" = true ]; then
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Testing Modes:"
    echo "  --fast          Fast tests only (skip slow tests, CI-friendly)"
    echo "  --dev           Development tests (default, skip slow and real-llm)"
    echo "  --local         Local LLM integration tests (check LLM service)"
    echo "  --real-llm      Include real LLM tests"
    echo "  --mock-only     Mock tests only"
    echo ""
    echo "Options:"
    echo "  --verbose       Verbose output"
    echo "  --coverage      Run with coverage report"
    echo "  --fail-fast     Stop on first failure"
    echo "  --help          Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                    # Development tests (default)"
    echo "  $0 --fast            # Fast tests only"
    echo "  $0 --local           # Local LLM integration"
    echo "  $0 --coverage        # With coverage report"
    echo "  $0 --real-llm --verbose  # Real LLM with verbose output"
    echo "  $0 --mock-only --fail-fast  # Mock tests, fail fast"
    exit 0
fi

# Display current configuration
echo "📊 Test Configuration:"
echo "   Mode: $MODE"
echo "   Real LLM tests: $([ "$REAL_LLM" = true ] && echo "✅ Enabled" || echo "❌ Disabled")"
echo "   Mock only: $([ "$MOCK_ONLY" = true ] && echo "✅ Enabled" || echo "❌ Disabled")"
echo "   Verbose mode: $([ "$VERBOSE" = true ] && echo "✅ Enabled" || echo "❌ Disabled")"
echo "   Coverage: $([ "$COVERAGE" = true ] && echo "✅ Enabled" || echo "❌ Disabled")"
echo "   Fail fast: $([ "$FAIL_FAST" = true ] && echo "✅ Enabled" || echo "❌ Disabled")"
echo ""

# Check if we're in the right directory
if [ ! -f "pinocchio.json" ]; then
    echo "❌ Error: pinocchio.json not found. Please run from project root."
    exit 1
fi

# LLM service check for local mode
if [ "$MODE" = "local" ]; then
    echo "🔍 Checking LLM service availability..."
    python3 -c "
import requests
import json
from pinocchio.config.config_manager import ConfigManager

try:
    config_manager = ConfigManager()
    llm_config = config_manager.config.llm
    if llm_config and llm_config.base_url:
        print(f'LLM Config: {llm_config.base_url}')

        # Test basic connectivity
        response = requests.get(llm_config.base_url, timeout=5)
        print(f'✅ LLM service is reachable (status: {response.status_code})')
    else:
        print('⚠️  No LLM configuration found')
        print('   Real LLM tests will be skipped.')
except Exception as e:
    print(f'⚠️  LLM service not available: {e}')
    print('   Real LLM tests will be skipped.')
"
    echo ""
fi

# Build pytest command
PYTEST_CMD="python -m pytest tests/"

# Add markers based on mode and options
if [ "$MOCK_ONLY" = true ]; then
    PYTEST_CMD="$PYTEST_CMD -m 'not local_only and not real_llm'"
elif [ "$REAL_LLM" = true ]; then
    PYTEST_CMD="$PYTEST_CMD -m 'not slow'"
elif [ "$MODE" = "fast" ]; then
    PYTEST_CMD="$PYTEST_CMD -m 'not slow'"
elif [ "$MODE" = "dev" ]; then
    PYTEST_CMD="$PYTEST_CMD -m 'not slow and not real_llm'"
elif [ "$MODE" = "local" ]; then
    PYTEST_CMD="$PYTEST_CMD -m 'not slow'"
fi

# Add common options
PYTEST_CMD="$PYTEST_CMD --tb=short --durations=10"

if [ "$VERBOSE" = true ]; then
    PYTEST_CMD="$PYTEST_CMD -v"
fi

if [ "$COVERAGE" = true ]; then
    PYTEST_CMD="$PYTEST_CMD --cov=pinocchio --cov-report=term-missing --cov-report=html"
fi

if [ "$FAIL_FAST" = true ]; then
    PYTEST_CMD="$PYTEST_CMD --maxfail=1"
fi

# Run tests
echo "🧪 Running tests..."
echo "Command: $PYTEST_CMD"
echo ""

eval $PYTEST_CMD

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Tests completed successfully!"

    if [ "$COVERAGE" = true ]; then
        echo "📊 Coverage report generated in htmlcov/index.html"
    fi

    echo ""
    echo "💡 Tips:"
    echo "   - Use --fast for CI/quick regression"
    echo "   - Use --dev for development (default)"
    echo "   - Use --local for LLM integration testing"
    echo "   - Use --coverage for code coverage analysis"
    echo "   - Use --fail-fast for quick feedback during development"
    echo "   - Use --verbose for detailed output"
else
    echo ""
    echo "❌ Tests failed!"
    echo ""
    echo "🔧 Troubleshooting tips:"
    echo "   - Check if all dependencies are installed"
    echo "   - Verify test data files are present"
    echo "   - Run with --verbose for more details"
    echo "   - Check for import errors in test files"
    echo "   - Use --fail-fast to stop on first failure"
    exit 1
fi
