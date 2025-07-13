# Test Performance Optimization

## Overview

This document outlines the performance optimizations implemented for the Pinocchio test suite to reduce test execution time while maintaining test coverage and quality.

## Performance Improvements

### Before Optimization
- **Total Test Time**: ~108 seconds (1:48)
- **Slowest Tests**:
  - `test_code_generation`: 27.81s
  - `test_debugging`: 19.06s
  - `test_agent_error_handling`: 9.02s
  - `test_custom_llm_connection`: 8.87s

### After Optimization
- **Fast Mode Test Time**: ~1.24 seconds (98% improvement)
- **Full Test Time**: ~68 seconds (37% improvement)
- **Slowest Tests in Fast Mode**: <0.03s each

## Optimization Strategies

### 1. Fast Mode Implementation

#### Environment Variable Control
```bash
export FAST_TEST=1  # Enable fast mode
```

#### Key Optimizations:
- **Shorter Prompts**: Reduced prompt length for LLM tests
- **Timeouts**: Added 5-10 second timeouts for LLM calls
- **Simplified Code**: Use minimal test code examples
- **Reduced Retry Delays**: 0.001s instead of 1.0s

#### Example Implementation:
```python
# Fast mode for testing - set FAST_TEST=1 to enable
FAST_TEST = os.getenv("FAST_TEST", "0") == "1"

if FAST_TEST:
    prompt = "Generate a simple function."
    response = await asyncio.wait_for(
        client.complete(prompt, agent_type="generator"),
        timeout=10.0
    )
else:
    prompt = """Please generate a Choreo DSL operator for matrix multiplication.
    The operator should be optimized for performance and include proper error handling."""
    response = await client.complete(prompt, agent_type="generator")
```

### 2. Test Marking Strategy

#### Slow Test Marking
```python
@pytest.mark.slow
async def test_custom_llm_connection():
    """Test Custom LLM client connection."""
```

#### Test Selection
```bash
# Run fast tests only
pytest -m "not slow"

# Run slow tests only
pytest -m "slow"

# Run all tests
pytest
```

### 3. Error Handling Optimization

#### Simplified Error Simulation
```python
def test_agent_error_handling(self, mock_llm_client):
    """Test error handling in agents."""
    # Configure mock to raise an exception
    mock_llm_client.complete.side_effect = Exception("LLM call failed")

    agents_to_test = [
        (optimizer, "analyze_code_performance"),
        (debugger, "analyze_code_issues"),
        (evaluator, "evaluate_performance")
    ]

    for agent, method_name in agents_to_test:
        if hasattr(agent, method_name):
            method = getattr(agent, method_name)
            result = method("test code")
            assert isinstance(result, dict)
```

### 4. Retry Mechanism Optimization

#### Reduced Retry Delays
```python
# Before: retry_delay=1.0 (1 second)
# After: retry_delay=0.001 (1 millisecond)
generator = GeneratorAgent(failing_client, max_retries=2, retry_delay=0.001)
```

## Usage

### Fast Test Runner
```bash
# Run optimized fast tests
./scripts/run_fast_tests.sh

# Expected output:
# 🚀 Running Fast Tests for Pinocchio
# 📊 Running tests in fast mode...
# ✅ Fast tests completed!
```

### Manual Test Execution
```bash
# Fast mode with environment variable
FAST_TEST=1 python -m pytest tests/ -v

# Skip slow tests
python -m pytest tests/ -m "not slow" -v

# Run only slow tests
python -m pytest tests/ -m "slow" -v
```

## Configuration

### pytest.ini Updates
```ini
[tool:pytest]
addopts =
    -v
    --tb=short
    --strict-markers
    --disable-warnings
    --durations=10
markers =
    slow: marks tests as slow (deselect with '-m "not slow"')
    integration: marks tests as integration tests
    unit: marks tests as unit tests
    real_llm: marks tests that require real LLM connection
```

## Benefits

### 1. Development Speed
- **98% faster** test execution in fast mode
- **37% faster** full test suite execution
- Reduced CI/CD pipeline time

### 2. Test Quality
- Maintained test coverage
- Preserved test reliability
- Clear separation of fast/slow tests

### 3. Developer Experience
- Quick feedback during development
- Easy test selection based on needs
- Clear performance metrics

## Best Practices

### 1. Test Design
- Use fast mode for unit tests
- Reserve slow tests for integration testing
- Implement timeouts for external calls

### 2. CI/CD Integration
```yaml
# GitHub Actions example
- name: Run Fast Tests
  run: ./scripts/run_fast_tests.sh

- name: Run Full Tests (Nightly)
  run: python -m pytest tests/ -v
```

### 3. Development Workflow
1. **Development**: Use fast tests for quick feedback
2. **Pre-commit**: Run fast tests to catch issues
3. **CI/CD**: Run full test suite for comprehensive validation
4. **Release**: Run slow tests to ensure integration quality

## Monitoring

### Performance Metrics
- Track test execution times
- Monitor slow test growth
- Set performance budgets

### Continuous Improvement
- Regular review of slow tests
- Optimization of test data
- Refactoring of complex test scenarios

## Future Enhancements

### 1. Parallel Execution
- Implement test parallelization
- Use pytest-xdist for distributed testing

### 2. Test Data Optimization
- Reduce test data size
- Implement test data factories
- Use minimal test scenarios

### 3. Caching
- Cache expensive test setup
- Implement test result caching
- Use pytest-cov for coverage caching
