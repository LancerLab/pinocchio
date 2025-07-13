# Scripts Directory

This directory contains utility scripts for the Pinocchio project.

## LLM Health & Performance Testing

### Why Separate Scripts?

Real LLM connection tests are **not** included in regular unit tests or CI/CD because:

- **External Dependencies**: Require real API keys and network connectivity
- **Cost**: Each test consumes API calls and may incur charges
- **Instability**: Network delays, service outages can cause flaky tests
- **Speed**: Real API calls are much slower than mock tests

### Available Scripts

#### `health_check.py` - Comprehensive Health & Performance Check

Comprehensive health check with configuration analysis, performance metrics (latency, throughput), and detailed recommendations.

```bash
# Full health and performance check
python scripts/health_check.py

# Save results to JSON file
python scripts/health_check.py --output health_report.json

# Verbose output with configuration analysis
python scripts/health_check.py --verbose

# Use custom config file
python scripts/health_check.py --config custom_config.json
```

**Features:**
- **Configuration Analysis**: Validates all service configurations
- **Performance Metrics**:
  - Latency (response time in milliseconds)
  - Throughput (characters per second)
  - Response length analysis
- **Multiple Test Runs**: Runs 3 tests per service for accurate metrics
- **Detailed Diagnostics**: Comprehensive error analysis and recommendations
- **JSON Output**: Machine-readable results for automation
- **Service Deduplication**: Avoids testing identical services
- **Clear Reporting**: Human-readable reports with performance summaries

#### `run_tests.sh` - Unified Test Runner

Unified test runner supporting multiple testing modes for different scenarios.

```bash
# Development tests (default, skip slow and real-llm)
bash scripts/run_tests.sh

# Fast tests only (CI-friendly)
bash scripts/run_tests.sh --fast

# Local LLM integration tests
bash scripts/run_tests.sh --local

# Real LLM tests with verbose output
bash scripts/run_tests.sh --real-llm --verbose

# Mock tests only with fail-fast
bash scripts/run_tests.sh --mock-only --fail-fast

# Coverage analysis
bash scripts/run_tests.sh --coverage
```

**Testing Modes:**
- **`--dev`** (default): Development tests - skip slow and real-llm tests
- **`--fast`**: Fast tests only - skip slow tests, CI-friendly
- **`--local`**: Local LLM integration - check LLM service availability
- **`--real-llm`**: Include real LLM tests
- **`--mock-only`**: Mock tests only

**Options:**
- **`--verbose`**: Verbose output
- **`--coverage`**: Run with coverage report
- **`--fail-fast`**: Stop on first failure
- **`--help`**: Show help message

### Performance Metrics Explained

The health check script measures and analyzes:

1. **Latency**: Average response time across multiple test runs
   - Good: < 2 seconds
   - Acceptable: 2-5 seconds
   - Poor: > 5 seconds

2. **Throughput**: Characters generated per second
   - Good: > 50 chars/sec
   - Acceptable: 10-50 chars/sec
   - Poor: < 10 chars/sec

3. **Response Quality**: Length and consistency of responses

### Configuration Support

The scripts automatically read from `pinocchio.json` and support:

- **Main LLM**: `llm` section
- **Agent-specific LLMs**: `llm_generator`, `llm_optimizer`, `llm_debugger`, `llm_evaluator`
- **Local services**: `provider: "custom"` with `base_url`
- **API services**: `provider: "openai"`/`"anthropic"`/`"google"` with `api_key`

### Example Configuration

```json
{
  "llm": {
    "provider": "custom",
    "base_url": "http://10.0.16.46:8001",
    "model_name": "Qwen/Qwen3-32B",
    "timeout": 120
  },
  "llm_generator": {
    "provider": "openai",
    "model_name": "gpt-4",
    "api_key": "your-api-key"
  }
}
```

### When to Use

**Use `health_check.py` when:**
- Quick validation of all configured services
- Performance analysis and optimization
- Detailed diagnostics and troubleshooting
- Configuration validation
- Generating reports for team analysis
- Automation integration
- Pre-deployment verification

**Use `run_tests.sh` when:**
- **`--dev`**: Daily development, debugging, feature development
- **`--fast`**: CI/CD pipelines, quick regression testing
- **`--local`**: Local LLM integration testing, end-to-end validation
- **`--real-llm`**: Testing with actual LLM services
- **`--mock-only`**: Unit testing, isolated component testing
- **`--coverage`**: Code coverage analysis, quality assurance

### Example Usage

```bash
# 1. Quick health check
python scripts/health_check.py

# 2. Development testing (default)
bash scripts/run_tests.sh

# 3. Fast regression for CI
bash scripts/run_tests.sh --fast

# 4. Local LLM integration
bash scripts/run_tests.sh --local

# 5. Coverage analysis
bash scripts/run_tests.sh --coverage

# 6. Real LLM testing
bash scripts/run_tests.sh --real-llm --verbose

# 7. Save health check results
python scripts/health_check.py --output results.json
```

### Sample Output

```
================================================================================
LLM Health & Performance Check Report
================================================================================
Configuration file: pinocchio.json
Check time: 2025-07-14 13:37:39

✅ Main LLM
   Source: llm
   Type: local
   Endpoint: http://10.0.16.46:8001
   Provider: custom
   Model: Qwen/Qwen3-32B
   Timeout: 120s
   Max Retries: 3
   Latency: 3397.0ms
   Throughput: 57.4 chars/sec
   Response Length: 195.0 chars
   Tests Run: 3/3

================================================================================
SUMMARY
================================================================================
Total services: 1
✅ Successful: 1
⚠️  Skipped: 0
❌ Failed: 0
Total issues found: 0
Total recommendations: 0

Performance Summary:
  Average Latency: 3397.0ms
  Average Throughput: 57.4 chars/sec

🎉 All LLM services are healthy!
```

### Exit Codes

- `0`: All tests passed
- `1`: All tests failed
- `2`: Partial success (some working, some failed)

### Troubleshooting

**Common Issues:**

1. **Missing API Key**
   ```
   ⚠️  Main LLM
      Error: API key not provided for openai
      Action: Add API key to configuration
   ```
   Solution: Add `api_key` to the service configuration

2. **Invalid base_url**
   ```
   ❌ Main LLM
      Error: Invalid base_url format
   ```
   Solution: Ensure base_url starts with `http://` or `https://`

3. **High Latency**
   ```
   ✅ Main LLM
      Latency: 8500.0ms
      Issues: High latency (> 10 seconds)
      Recommendations: Consider optimizing network or model configuration
   ```
   Solution: Check network connectivity, model size, or server performance

4. **Low Throughput**
   ```
   ✅ Main LLM
      Throughput: 5.2 chars/sec
      Issues: Low throughput (< 10 chars/sec)
      Recommendations: Consider using a faster model or optimizing prompts
   ```
   Solution: Use a faster model or optimize prompt design

5. **Connection Timeout**
   ```
   ❌ Main LLM
      Error: Connection timeout
   ```
   Solution: Check network connectivity and increase timeout

6. **Empty Response**
   ```
   ❌ Main LLM
      Error: All performance tests failed
   ```
   Solution: Check if the LLM service is responding correctly

### Security Notes

- Never commit API keys to version control
- Use environment variables for sensitive data
- Consider using `.env` files for local development
- Rotate API keys regularly

### Integration with Development

You can integrate these scripts into your development workflow:

```bash
# Pre-deployment check
python scripts/health_check.py --output pre_deploy_check.json

# Quick validation during development
bash scripts/run_tests.sh --dev

# Automated health monitoring
python scripts/health_check.py --output daily_check.json

# Performance benchmarking
python scripts/health_check.py --output performance_benchmark.json

# CI/CD pipeline
bash scripts/run_tests.sh --fast

# Local integration testing
bash scripts/run_tests.sh --local
```

### Performance Optimization Tips

1. **For High Latency:**
   - Use smaller/faster models
   - Optimize network configuration
   - Consider local deployment
   - Increase server resources

2. **For Low Throughput:**
   - Use more efficient models
   - Optimize prompt design
   - Reduce response length requirements
   - Use streaming responses

3. **For Configuration Issues:**
   - Validate all required fields
   - Check API key format and permissions
   - Verify endpoint URLs
   - Test network connectivity
