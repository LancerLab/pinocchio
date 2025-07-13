#!/bin/bash

# Local test runner for real LLM tests
# This script enables real LLM tests for local development

set -e

echo "🔧 Setting up local test environment..."

# Check if we're in the right directory
if [ ! -f "pinocchio.json" ]; then
    echo "❌ Error: pinocchio.json not found. Please run from project root."
    exit 1
fi

# Enable real LLM tests
export ENABLE_REAL_LLM_TESTS=1
export PYTHONPATH=.

echo "✅ Environment variables set:"
echo "   ENABLE_REAL_LLM_TESTS=$ENABLE_REAL_LLM_TESTS"
echo "   PYTHONPATH=$PYTHONPATH"

# Check if LLM service is available
echo "🔍 Checking LLM service availability..."
python3 -c "
import requests
import json
from pinocchio.config import ConfigManager

try:
    config = ConfigManager()
    llm_config = config.get_llm_config()
    print(f'LLM Config: {llm_config.base_url}')

    # Test basic connectivity
    response = requests.get(llm_config.base_url, timeout=5)
    print(f'✅ LLM service is reachable (status: {response.status_code})')
except Exception as e:
    print(f'⚠️  LLM service not available: {e}')
    print('   Real LLM tests will be skipped.')
"

echo ""
echo "🧪 Running tests..."

# Run all tests including real LLM tests
if [ "$1" = "--real-llm-only" ]; then
    echo "Running only real LLM tests..."
    pytest tests/ -v -m "local_only or real_llm" --tb=short
elif [ "$1" = "--mock-only" ]; then
    echo "Running only mock tests..."
    pytest tests/ -v -m "not local_only and not real_llm" --tb=short
else
    echo "Running all tests..."
    pytest tests/ -v --tb=short
fi

echo ""
echo "✅ Test run completed!"
echo ""
echo "💡 Tips:"
echo "   - Use --real-llm-only to run only real LLM tests"
echo "   - Use --mock-only to run only mock tests"
echo "   - Set ENABLE_REAL_LLM_TESTS=0 to disable real LLM tests"
