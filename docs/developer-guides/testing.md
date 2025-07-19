# Testing Guide

## Overview

Pinocchio uses a comprehensive testing strategy with different test categories to ensure reliability and maintainability:

- **Unit Tests**: Fast, isolated tests using mocks (safe for CI)
- **Integration Tests**: Tests that verify component interactions
- **Real LLM Tests**: Tests that require actual LLM service (local only)

## Test Categories

### 1. Unit Tests (CI Safe)
- **Location**: All test files in `tests/unittests/`
- **Characteristics**: Use mocks, no external dependencies
- **CI Status**: ✅ Always run in CI
- **Command**: `pytest tests/ -m "not local_only and not real_llm"`

### 2. Real LLM Tests (Local Only)
- **Location**: Tests marked with `@pytest.mark.local_only`
- **Characteristics**: Require actual LLM service, test real functionality
- **CI Status**: ❌ Never run in CI
- **Command**: `ENABLE_REAL_LLM_TESTS=1 pytest tests/ -m "local_only"`

### 3. Integration Tests
- **Location**: Tests marked with `@pytest.mark.integration`
- **Characteristics**: Test component interactions
- **CI Status**: ✅ Run in CI (if no external dependencies)
- **Command**: `pytest tests/ -m "integration"`

## Running Tests

### Quick Start (Local Development)
```bash
# Run all tests (including real LLM tests if service available)
./scripts/run_local_tests.sh

# Run only mock tests (CI safe)
./scripts/run_local_tests.sh --mock-only

# Run only real LLM tests
./scripts/run_local_tests.sh --real-llm-only
```

### Manual Test Execution
```bash
# Run all tests except real LLM tests
pytest tests/ -v -m "not local_only and not real_llm"

# Run only real LLM tests
ENABLE_REAL_LLM_TESTS=1 pytest tests/ -v -m "local_only"

# Run specific test file
pytest tests/unittests/agents/test_agent_specific_config.py -v

# Run specific test class
pytest tests/unittests/agents/test_agent_specific_config.py::TestAgentSpecificConfig -v

# Run specific test method
pytest tests/unittests/agents/test_agent_specific_config.py::TestAgentSpecificConfig::test_config_manager_get_agent_llm_config -v
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ENABLE_REAL_LLM_TESTS` | `0` | Enable real LLM tests |
| `PYTHONPATH` | `.` | Python path for imports |

## Test Configuration

### pytest.ini
```ini
[tool:pytest]
markers =
    local_only: marks tests as local-only (not for CI)
    real_llm: marks tests that require real LLM service
    slow: marks tests as slow running
    integration: marks tests as integration tests
    unit: marks tests as unit tests

addopts =
    -v
    --strict-markers
    --disable-warnings
    -m "not local_only and not real_llm"
```

### CI Configuration
The GitHub Actions workflow automatically:
- Sets `ENABLE_REAL_LLM_TESTS=0`
- Excludes `local_only` and `real_llm` tests
- Runs tests on multiple Python versions
- Generates coverage reports

## Writing Tests

### Unit Tests (Recommended for CI)
```python
def test_agent_configuration():
    """Test agent configuration with mocks."""
    with patch('pinocchio.config.ConfigManager') as mock_config:
        # Mock configuration
        mock_config.get_agent_llm_config.return_value = mock_llm_config

        # Test agent initialization
        agent = GeneratorAgent()
        assert isinstance(agent.llm_client, CustomLLMClient)
```

### Real LLM Tests (Local Only)
```python
@pytest.mark.local_only
class TestRealLLM:
    @pytest.mark.skipif(
        not os.getenv("ENABLE_REAL_LLM_TESTS"),
        reason="Real LLM tests disabled"
    )
    def test_real_llm_connection(self):
        """Test real LLM connectivity."""
        agent = GeneratorAgent()
        # Test with real LLM service
        # Handle service unavailability gracefully
```

### Test Markers
```python
@pytest.mark.unit          # Unit tests
@pytest.mark.integration   # Integration tests
@pytest.mark.local_only    # Local only tests
@pytest.mark.real_llm      # Real LLM tests
@pytest.mark.slow          # Slow running tests
```

## Test Coverage

### Coverage Reports
```bash
# Generate coverage report
pytest tests/ --cov=pinocchio --cov-report=html

# View coverage report
open htmlcov/index.html
```

### Coverage Targets
- **Unit Tests**: >90% coverage
- **Integration Tests**: >80% coverage
- **Real LLM Tests**: Basic connectivity and functionality

## Troubleshooting

### Common Issues

1. **Real LLM Tests Skipped**
   ```bash
   # Enable real LLM tests
   export ENABLE_REAL_LLM_TESTS=1
   pytest tests/ -m "local_only"
   ```

2. **Import Errors**
   ```bash
   # Set Python path
   export PYTHONPATH=.
   pytest tests/
   ```

3. **LLM Service Unavailable**
   ```bash
   # Check service status
   curl http://10.0.16.46:8001/health

   # Run mock tests only
   pytest tests/ -m "not local_only"
   ```

### Debug Mode
```bash
# Run with debug output
pytest tests/ -v -s --tb=long

# Run specific test with debug
pytest tests/unittests/agents/test_agent_specific_config.py::TestAgentSpecificConfig::test_config_manager_get_agent_llm_config -v -s
```

## Best Practices

1. **Test Isolation**: Each test should be independent
2. **Mock External Dependencies**: Use mocks for CI-safe tests
3. **Real Service Testing**: Test real LLM functionality locally
4. **Error Handling**: Test both success and failure scenarios
5. **Performance**: Keep unit tests fast (<1s each)
6. **Documentation**: Document complex test scenarios

## CI/CD Integration

### GitHub Actions
- Runs on push to main/develop branches
- Runs on pull requests
- Excludes real LLM tests automatically
- Generates coverage reports
- Runs linting checks

### Local Development
- Use `./scripts/run_local_tests.sh` for convenience
- Enable real LLM tests for comprehensive testing
- Run mock tests for quick feedback
