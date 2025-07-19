# Development & Debugging Guide

This comprehensive guide covers development workflows, debugging tools, testing strategies, and operational procedures for the Pinocchio system.

## Table of Contents

- [Development Environment Setup](#development-environment-setup)
- [Project Structure](#project-structure)
- [Configuration System](#configuration-system)
- [Debugging Tools & Scripts](#debugging-tools--scripts)
- [Testing Framework](#testing-framework)
- [Logging & Monitoring](#logging--monitoring)
- [Performance Optimization](#performance-optimization)
- [Deployment Strategies](#deployment-strategies)
- [Troubleshooting Guide](#troubleshooting-guide)

## Development Environment Setup

### Prerequisites

```bash
# System requirements
Python 3.8+
CUDA Toolkit 11.0+ (optional, for full tool functionality)
Git
```

### Installation Steps

```bash
# 1. Clone repository
git clone <repository-url>
cd pinocchio

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate   # Windows

# 3. Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # Development dependencies

# 4. Install pre-commit hooks
pre-commit install

# 5. Setup configuration
cp pinocchio.json.example pinocchio.json
# Edit pinocchio.json with your settings
```

### Development Dependencies

```bash
# Core development tools
pytest>=7.0.0
pytest-asyncio>=0.20.0
pytest-cov>=4.0.0
black>=22.0.0
flake8>=5.0.0
mypy>=0.991
pre-commit>=2.20.0

# Testing utilities
aioresponses>=0.7.4
responses>=0.22.0
factory-boy>=3.2.1

# Documentation tools
mkdocs>=1.4.0
mkdocs-material>=8.5.0
```

### IDE Configuration

#### VS Code Settings (.vscode/settings.json)

```json
{
    "python.defaultInterpreterPath": "./venv/bin/python",
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": false,
    "python.linting.flake8Enabled": true,
    "python.formatting.provider": "black",
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": [
        "tests/"
    ],
    "files.exclude": {
        "**/__pycache__": true,
        "**/*.pyc": true,
        ".pytest_cache": true,
        ".coverage": true
    }
}
```

#### PyCharm Configuration

1. Set interpreter to `./venv/bin/python`
2. Configure pytest as default test runner
3. Enable Black formatter
4. Configure flake8 as linter
5. Set project structure with `pinocchio` as source root

## Project Structure

```
pinocchio/
├── pinocchio/                  # Main package
│   ├── __init__.py
│   ├── __main__.py            # CLI entry point
│   ├── coordinator.py         # Central orchestrator
│   ├── session_logger.py      # Session logging
│   │
│   ├── agents/                # AI agents
│   │   ├── __init__.py
│   │   ├── base.py           # Base agent class
│   │   ├── generator.py      # Code generation agent
│   │   ├── optimizer.py      # Optimization agent
│   │   ├── debugger.py       # Debugging agent
│   │   └── evaluator.py      # Evaluation agent
│   │
│   ├── cli/                   # Command line interface
│   │   ├── __init__.py
│   │   └── main.py
│   │
│   ├── config/                # Configuration management
│   │   ├── __init__.py
│   │   ├── config_manager.py
│   │   └── models.py         # Pydantic config models
│   │
│   ├── data_models/           # Data structures
│   │   ├── __init__.py
│   │   ├── agent.py          # Agent-related models
│   │   ├── session.py        # Session models
│   │   └── task.py           # Task models
│   │
│   ├── errors/                # Custom exceptions
│   │   ├── __init__.py
│   │   └── exceptions.py
│   │
│   ├── knowledge/             # Knowledge management
│   │   ├── __init__.py
│   │   └── manager.py
│   │
│   ├── llm/                   # LLM integrations
│   │   ├── __init__.py
│   │   ├── base.py           # Base LLM client
│   │   └── custom_llm_client.py
│   │
│   ├── memory/                # Memory management
│   │   ├── __init__.py
│   │   └── manager.py
│   │
│   ├── plugins/               # Plugin system
│   │   ├── __init__.py
│   │   ├── base.py           # Plugin base classes
│   │   ├── prompt_plugins.py
│   │   ├── workflow_plugins.py
│   │   └── agent_plugins.py
│   │
│   ├── prompt/                # Prompt management
│   │   ├── __init__.py
│   │   └── manager.py
│   │
│   ├── session/               # Session management
│   │   ├── __init__.py
│   │   └── manager.py
│   │
│   ├── task_planning/         # Task planning system
│   │   ├── __init__.py
│   │   ├── planner.py
│   │   ├── executor.py
│   │   └── models.py
│   │
│   ├── tools/                 # MCP tools integration
│   │   ├── __init__.py
│   │   ├── base.py           # Tool base classes
│   │   ├── cuda_debug_tools.py
│   │   └── cuda_eval_tools.py
│   │
│   ├── utils/                 # Utility functions
│   │   ├── __init__.py
│   │   ├── json_parser.py
│   │   ├── temp_utils.py
│   │   └── verbose_logger.py
│   │
│   └── workflows/             # Workflow management
│       ├── __init__.py
│       └── manager.py
│
├── tests/                     # Test suite
│   ├── __init__.py
│   ├── conftest.py           # Pytest configuration
│   ├── unit/                 # Unit tests
│   ├── integration/          # Integration tests
│   └── fixtures/             # Test fixtures
│
├── scripts/                   # Development scripts
│   ├── test_llm_connection.py
│   ├── health_check.py
│   └── run_fast_tests.sh
│
├── docs/                      # Documentation
│   ├── architecture.md
│   ├── customization.md
│   ├── development.md
│   └── ...
│
├── examples/                  # Example implementations
│   ├── basic_usage.py
│   ├── custom_agent.py
│   └── ...
│
├── pinocchio.json.example     # Example configuration
├── pinocchio.json            # Main configuration
├── requirements.txt          # Production dependencies
├── requirements-dev.txt      # Development dependencies
├── setup.py                  # Package setup
├── pyproject.toml           # Modern Python project config
└── README.md                # Project overview
```

## Configuration System

### Configuration Files

#### Main Configuration (pinocchio.json)

```json
{
  "llm": {
    "provider": "custom",
    "base_url": "http://localhost:8001",
    "model_name": "Qwen/Qwen3-32B",
    "timeout": 120,
    "max_retries": 3,
    "api_key": null
  },
  "agents": {
    "generator": {"enabled": true, "max_retries": 3},
    "debugger": {"enabled": true, "max_retries": 3},
    "optimizer": {"enabled": true, "max_retries": 3},
    "evaluator": {"enabled": true, "max_retries": 3}
  },
  "plugins": {
    "enabled": true,
    "plugins_directory": "./plugins",
    "active_plugins": {
      "prompt": "cuda_prompt_plugin",
      "workflow": "json_workflow_plugin"
    }
  },
  "tools": {
    "enabled": true,
    "debug_tools": {
      "cuda_compile": {"enabled": true, "timeout": 60},
      "cuda_memcheck": {"enabled": true, "timeout": 120},
      "cuda_syntax_check": {"enabled": true, "timeout": 30}
    },
    "eval_tools": {
      "cuda_profile": {"enabled": true, "timeout": 180},
      "cuda_occupancy": {"enabled": true, "timeout": 30},
      "cuda_performance_analyze": {"enabled": true, "timeout": 60}
    }
  },
  "storage": {
    "sessions_path": "./sessions",
    "memories_path": "./memories",
    "knowledge_path": "./knowledge"
  },
  "verbose": {
    "enabled": true,
    "level": "maximum",
    "show_agent_instructions": true,
    "show_execution_times": true
  },
  "logging": {
    "level": "DEBUG",
    "console_output": true,
    "file_output": true
  }
}
```

#### Environment-specific Configurations

```bash
# Development environment
export PINOCCHIO_ENV=development
export LLM_PROVIDER=openai
export LLM_API_KEY=your-api-key
export DEBUG_LEVEL=DEBUG
export STORAGE_PATH=./dev_storage

# Testing environment
export PINOCCHIO_ENV=testing
export LLM_PROVIDER=mock
export DEBUG_LEVEL=WARNING
export FAST_TEST=1

# Production environment
export PINOCCHIO_ENV=production
export LLM_PROVIDER=production
export DEBUG_LEVEL=ERROR
export STORAGE_PATH=/opt/pinocchio/storage
```

### Configuration Loading

```python
from pinocchio.config import ConfigManager

# Load configuration
config = ConfigManager("pinocchio.json")

# Access configuration values
llm_config = config.get("llm", {})
agent_config = config.get_agent_llm_config("generator")
verbose_enabled = config.get("verbose.enabled", True)

# Environment-specific overrides
if os.getenv("PINOCCHIO_ENV") == "development":
    config.override("verbose.enabled", True)
    config.override("tools.enabled", True)
```

## Debugging Tools & Scripts

### Core Debugging Scripts

#### 1. LLM Connection Test (`scripts/test_llm_connection.py`)

```python
#!/usr/bin/env python3
"""
Test LLM connection and validate API configuration.
"""

import asyncio
import sys
import os
from pinocchio.config import ConfigManager
from pinocchio.llm.custom_llm_client import CustomLLMClient

async def test_llm_connection():
    """Test LLM connection with current configuration."""
    try:
        # Load configuration
        config = ConfigManager()
        llm_config = config.get("llm", {})

        print(f"Testing LLM connection:")
        print(f"  Provider: {llm_config.get('provider', 'unknown')}")
        print(f"  Base URL: {llm_config.get('base_url', 'unknown')}")
        print(f"  Model: {llm_config.get('model_name', 'unknown')}")

        # Create LLM client
        client = CustomLLMClient(llm_config)

        # Test simple completion
        test_prompt = "Hello, this is a test message. Please respond with 'Connection successful'."
        response = await client.complete(test_prompt)

        if response.get("success", False):
            print("✅ LLM connection successful!")
            print(f"Response: {response.get('output', {}).get('explanation', 'No explanation')}")
            return True
        else:
            print("❌ LLM connection failed!")
            print(f"Error: {response.get('error', 'Unknown error')}")
            return False

    except Exception as e:
        print(f"❌ LLM connection test failed: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_llm_connection())
    sys.exit(0 if success else 1)
```

Usage:
```bash
# Test with current configuration
python scripts/test_llm_connection.py

# Test with specific provider
LLM_PROVIDER=openai python scripts/test_llm_connection.py
```

#### 2. System Health Check (`scripts/health_check.py`)

```python
#!/usr/bin/env python3
"""
Comprehensive system health check.
"""

import asyncio
import json
import sys
from pathlib import Path
from pinocchio.config import ConfigManager
from pinocchio.tools import ToolManager, CudaDebugTools, CudaEvalTools

class HealthChecker:
    """System health checker."""

    def __init__(self):
        self.results = {
            "config": {"status": "unknown", "details": {}},
            "llm": {"status": "unknown", "details": {}},
            "tools": {"status": "unknown", "details": {}},
            "storage": {"status": "unknown", "details": {}},
            "dependencies": {"status": "unknown", "details": {}}
        }

    async def run_all_checks(self):
        """Run all health checks."""
        print("🔍 Running Pinocchio System Health Check...")
        print("=" * 50)

        await self._check_configuration()
        await self._check_llm_connection()
        await self._check_tools()
        await self._check_storage()
        await self._check_dependencies()

        self._print_summary()

    async def _check_configuration(self):
        """Check configuration validity."""
        print("\n📋 Checking Configuration...")
        try:
            config = ConfigManager()

            # Check required sections
            required_sections = ["llm", "agents", "storage"]
            missing_sections = []

            for section in required_sections:
                if not config.get(section):
                    missing_sections.append(section)

            if missing_sections:
                self.results["config"]["status"] = "warning"
                self.results["config"]["details"]["missing_sections"] = missing_sections
                print(f"⚠️  Missing configuration sections: {missing_sections}")
            else:
                self.results["config"]["status"] = "ok"
                print("✅ Configuration is valid")

        except Exception as e:
            self.results["config"]["status"] = "error"
            self.results["config"]["details"]["error"] = str(e)
            print(f"❌ Configuration error: {e}")

    async def _check_llm_connection(self):
        """Check LLM connection."""
        print("\n🤖 Checking LLM Connection...")
        try:
            from scripts.test_llm_connection import test_llm_connection
            success = await test_llm_connection()

            if success:
                self.results["llm"]["status"] = "ok"
                print("✅ LLM connection working")
            else:
                self.results["llm"]["status"] = "error"
                print("❌ LLM connection failed")

        except Exception as e:
            self.results["llm"]["status"] = "error"
            self.results["llm"]["details"]["error"] = str(e)
            print(f"❌ LLM connection error: {e}")

    async def _check_tools(self):
        """Check MCP tools availability."""
        print("\n🛠️  Checking MCP Tools...")
        try:
            tool_manager = ToolManager()
            CudaDebugTools.register_tools(tool_manager)
            CudaEvalTools.register_tools(tool_manager)

            tools = tool_manager.list_tools()
            working_tools = []
            broken_tools = []

            for tool_name in tools:
                try:
                    # Test with minimal input
                    result = tool_manager.execute_tool(
                        tool_name,
                        cuda_code="__global__ void test() {}"
                    )

                    if result.status.value in ["success", "error"]:  # Both are valid for test
                        working_tools.append(tool_name)
                    else:
                        broken_tools.append(tool_name)

                except Exception as e:
                    broken_tools.append(f"{tool_name}: {str(e)}")

            self.results["tools"]["details"]["working"] = working_tools
            self.results["tools"]["details"]["broken"] = broken_tools

            if not broken_tools:
                self.results["tools"]["status"] = "ok"
                print(f"✅ All {len(working_tools)} tools working")
            else:
                self.results["tools"]["status"] = "warning"
                print(f"⚠️  {len(working_tools)} working, {len(broken_tools)} issues")

        except Exception as e:
            self.results["tools"]["status"] = "error"
            self.results["tools"]["details"]["error"] = str(e)
            print(f"❌ Tools check error: {e}")

    async def _check_storage(self):
        """Check storage directories."""
        print("\n💾 Checking Storage...")
        try:
            config = ConfigManager()
            storage_paths = {
                "sessions": config.get("storage.sessions_path", "./sessions"),
                "memories": config.get("storage.memories_path", "./memories"),
                "knowledge": config.get("storage.knowledge_path", "./knowledge")
            }

            accessible_paths = {}
            inaccessible_paths = {}

            for name, path in storage_paths.items():
                try:
                    Path(path).mkdir(parents=True, exist_ok=True)
                    if Path(path).is_dir() and os.access(path, os.R_OK | os.W_OK):
                        accessible_paths[name] = path
                    else:
                        inaccessible_paths[name] = f"{path} (permission denied)"
                except Exception as e:
                    inaccessible_paths[name] = f"{path} ({str(e)})"

            self.results["storage"]["details"]["accessible"] = accessible_paths
            self.results["storage"]["details"]["inaccessible"] = inaccessible_paths

            if not inaccessible_paths:
                self.results["storage"]["status"] = "ok"
                print("✅ All storage paths accessible")
            else:
                self.results["storage"]["status"] = "error"
                print(f"❌ Storage issues: {list(inaccessible_paths.keys())}")

        except Exception as e:
            self.results["storage"]["status"] = "error"
            self.results["storage"]["details"]["error"] = str(e)
            print(f"❌ Storage check error: {e}")

    async def _check_dependencies(self):
        """Check system dependencies."""
        print("\n📦 Checking Dependencies...")
        try:
            dependencies = {
                "nvcc": self._check_command("nvcc --version"),
                "cuda-memcheck": self._check_command("cuda-memcheck --version"),
                "nvprof": self._check_command("nvprof --version"),
                "python": self._check_command("python --version"),
                "pip": self._check_command("pip --version")
            }

            available = {k: v for k, v in dependencies.items() if v["available"]}
            missing = {k: v for k, v in dependencies.items() if not v["available"]}

            self.results["dependencies"]["details"]["available"] = available
            self.results["dependencies"]["details"]["missing"] = missing

            # CUDA tools are optional
            required_missing = {k: v for k, v in missing.items() if k in ["python", "pip"]}

            if not required_missing:
                self.results["dependencies"]["status"] = "ok" if not missing else "warning"
                status = "✅" if not missing else "⚠️ "
                print(f"{status} Dependencies: {len(available)} available, {len(missing)} missing")
            else:
                self.results["dependencies"]["status"] = "error"
                print(f"❌ Missing required dependencies: {list(required_missing.keys())}")

        except Exception as e:
            self.results["dependencies"]["status"] = "error"
            self.results["dependencies"]["details"]["error"] = str(e)
            print(f"❌ Dependencies check error: {e}")

    def _check_command(self, command: str) -> Dict[str, Any]:
        """Check if command is available."""
        import subprocess
        try:
            result = subprocess.run(
                command.split(),
                capture_output=True,
                text=True,
                timeout=10
            )
            return {
                "available": True,
                "version": result.stdout.strip() or result.stderr.strip(),
                "exit_code": result.returncode
            }
        except Exception as e:
            return {
                "available": False,
                "error": str(e)
            }

    def _print_summary(self):
        """Print health check summary."""
        print("\n" + "=" * 50)
        print("📊 Health Check Summary")
        print("=" * 50)

        for component, result in self.results.items():
            status_icon = {
                "ok": "✅",
                "warning": "⚠️ ",
                "error": "❌",
                "unknown": "❓"
            }.get(result["status"], "❓")

            print(f"{status_icon} {component.upper()}: {result['status']}")

        # Overall status
        statuses = [r["status"] for r in self.results.values()]
        if "error" in statuses:
            overall = "❌ SYSTEM HAS ERRORS"
        elif "warning" in statuses:
            overall = "⚠️  SYSTEM HAS WARNINGS"
        else:
            overall = "✅ SYSTEM HEALTHY"

        print(f"\n🏥 Overall Status: {overall}")

        # Save detailed results
        with open("health_check_results.json", "w") as f:
            json.dump(self.results, f, indent=2)
        print("\n📄 Detailed results saved to: health_check_results.json")

async def main():
    """Main health check function."""
    checker = HealthChecker()
    await checker.run_all_checks()

if __name__ == "__main__":
    asyncio.run(main())
```

Usage:
```bash
# Full health check
python scripts/health_check.py

# Quick check (skip slow tests)
QUICK_CHECK=1 python scripts/health_check.py
```

#### 3. Fast Test Runner (`scripts/run_fast_tests.sh`)

```bash
#!/bin/bash
"""
Fast test runner for development workflow.
"""

set -e

echo "🚀 Running Fast Pinocchio Tests"
echo "================================"

# Set environment for fast testing
export FAST_TEST=1
export PYTEST_CURRENT_TEST=""

# Run tests with optimizations
python -m pytest tests/ \
    -v \
    --tb=short \
    --disable-warnings \
    -x \
    --maxfail=5 \
    --durations=10 \
    -m "not slow and not real_llm" \
    --cov=pinocchio \
    --cov-report=term-missing \
    --cov-report=html:htmlcov \
    --cov-fail-under=80

echo ""
echo "✅ Fast tests completed!"
echo "📊 Coverage report: htmlcov/index.html"
```

Usage:
```bash
# Make executable and run
chmod +x scripts/run_fast_tests.sh
./scripts/run_fast_tests.sh
```

### Debugging Utilities

#### 1. Session Debugger

```python
# scripts/debug_session.py
"""
Debug session data and analyze execution flow.
"""

import json
from pathlib import Path
from datetime import datetime

class SessionDebugger:
    """Debug and analyze session data."""

    def analyze_session(self, session_file: str):
        """Analyze session execution."""
        with open(session_file, 'r') as f:
            session_data = json.load(f)

        print(f"📊 Session Analysis: {session_data.get('session_id', 'unknown')}")
        print(f"Created: {session_data.get('created_at', 'unknown')}")
        print(f"Status: {session_data.get('status', 'unknown')}")

        # Analyze agent interactions
        self._analyze_agent_interactions(session_data)

        # Analyze performance
        self._analyze_performance(session_data)

        # Analyze errors
        self._analyze_errors(session_data)

    def _analyze_agent_interactions(self, session_data):
        """Analyze agent interaction patterns."""
        interactions = session_data.get('agent_interactions', {})

        print("\n🤖 Agent Interactions:")
        for agent, data in interactions.items():
            print(f"  {agent}: {data.get('call_count', 0)} calls")
            print(f"    Success rate: {data.get('success_rate', 0):.1%}")
            print(f"    Avg time: {data.get('avg_time_ms', 0):.1f}ms")

    def _analyze_performance(self, session_data):
        """Analyze performance metrics."""
        metrics = session_data.get('performance_metrics', {})

        print("\n⚡ Performance Metrics:")
        print(f"  Total time: {metrics.get('total_time_ms', 0):.1f}ms")
        print(f"  LLM calls: {metrics.get('llm_calls', 0)}")
        print(f"  Tool executions: {metrics.get('tool_executions', 0)}")

    def _analyze_errors(self, session_data):
        """Analyze errors and failures."""
        errors = session_data.get('errors', [])

        if errors:
            print(f"\n❌ Errors Found ({len(errors)}):")
            for error in errors:
                print(f"  {error.get('timestamp', 'unknown')}: {error.get('message', 'unknown')}")
        else:
            print("\n✅ No errors found")
```

#### 2. Configuration Validator

```python
# scripts/validate_config.py
"""
Validate configuration files.
"""

from pinocchio.config import ConfigManager
from pinocchio.config.models import PinocchioConfig

def validate_configuration(config_path: str = "pinocchio.json"):
    """Validate configuration file."""
    try:
        # Load and validate with Pydantic
        config = ConfigManager(config_path)
        validated_config = PinocchioConfig(**config.config)

        print("✅ Configuration is valid!")

        # Additional custom validation
        warnings = []

        # Check LLM configuration
        llm_config = validated_config.llm
        if not llm_config.api_key and llm_config.provider in ["openai", "anthropic"]:
            warnings.append("LLM API key not set for external provider")

        # Check storage paths
        for path_name, path in [
            ("sessions_path", validated_config.storage.sessions_path),
            ("memories_path", validated_config.storage.memories_path),
            ("knowledge_path", validated_config.storage.knowledge_path)
        ]:
            if not Path(path).exists():
                warnings.append(f"Storage path does not exist: {path_name} = {path}")

        if warnings:
            print("\n⚠️  Warnings:")
            for warning in warnings:
                print(f"  • {warning}")

        return True

    except Exception as e:
        print(f"❌ Configuration validation failed: {e}")
        return False

if __name__ == "__main__":
    import sys
    config_file = sys.argv[1] if len(sys.argv) > 1 else "pinocchio.json"
    validate_configuration(config_file)
```

## Testing Framework

### Test Structure

```
tests/
├── conftest.py              # Pytest configuration and fixtures
├── unit/                    # Unit tests
│   ├── test_agents.py
│   ├── test_config.py
│   ├── test_tools.py
│   └── ...
├── integration/             # Integration tests
│   ├── test_coordinator.py
│   ├── test_workflows.py
│   └── ...
└── fixtures/                # Test data and fixtures
    ├── config_samples/
    ├── cuda_code_samples/
    └── mock_responses/
```

### Pytest Configuration (conftest.py)

```python
import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock
from pinocchio.config import ConfigManager

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def temp_dir():
    """Create temporary directory for tests."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path)

@pytest.fixture
def mock_config():
    """Mock configuration for testing."""
    return {
        "llm": {
            "provider": "mock",
            "model_name": "test-model",
            "timeout": 30
        },
        "agents": {
            "generator": {"enabled": True, "max_retries": 1},
            "debugger": {"enabled": True, "max_retries": 1}
        },
        "storage": {
            "sessions_path": "./test_sessions",
            "memories_path": "./test_memories",
            "knowledge_path": "./test_knowledge"
        },
        "verbose": {"enabled": False}
    }

@pytest.fixture
def mock_llm_client():
    """Mock LLM client for testing."""
    client = Mock()
    client.complete = Mock(return_value={
        "agent_type": "test",
        "success": True,
        "output": {"result": "test output"},
        "explanation": "Test explanation",
        "confidence": 0.95
    })
    return client

@pytest.fixture
def sample_cuda_code():
    """Sample CUDA code for testing."""
    return """
__global__ void vectorAdd(float *a, float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}
"""
```

### Test Categories and Markers

```python
# pytest.ini
[tool:pytest]
testpaths = tests
addopts =
    -ra
    --strict-markers
    --strict-config
    --cov=pinocchio
    --cov-branch
    --cov-report=term-missing:skip-covered
    --cov-report=html:htmlcov
    --cov-report=xml

markers =
    slow: marks tests as slow (deselect with '-m "not slow"')
    real_llm: marks tests that require real LLM connection
    cuda_tools: marks tests that require CUDA toolkit
    integration: marks tests as integration tests
    unit: marks tests as unit tests
```

### Example Test Cases

#### Unit Test Example

```python
# tests/unit/test_tools.py
import pytest
from pinocchio.tools import ToolManager, CudaDebugTools

class TestCudaDebugTools:
    """Test CUDA debugging tools."""

    def test_tool_registration(self):
        """Test tool registration."""
        tool_manager = ToolManager()
        CudaDebugTools.register_tools(tool_manager)

        tools = tool_manager.list_tools()
        expected_tools = ["cuda_compile", "cuda_memcheck", "cuda_syntax_check"]

        for tool in expected_tools:
            assert tool in tools

    def test_syntax_checker(self, sample_cuda_code):
        """Test CUDA syntax checker."""
        tool_manager = ToolManager()
        CudaDebugTools.register_tools(tool_manager)

        # Test with valid code
        result = tool_manager.execute_tool(
            "cuda_syntax_check",
            cuda_code=sample_cuda_code,
            strict=True
        )

        assert result.status.value == "success"
        assert "No issues found" in result.output

    @pytest.mark.cuda_tools
    def test_compiler_tool(self, sample_cuda_code):
        """Test CUDA compiler tool (requires nvcc)."""
        tool_manager = ToolManager()
        CudaDebugTools.register_tools(tool_manager)

        result = tool_manager.execute_tool(
            "cuda_compile",
            cuda_code=sample_cuda_code,
            arch="compute_75"
        )

        # Should either succeed or fail gracefully
        assert result.status.value in ["success", "error", "not_found"]
```

#### Integration Test Example

```python
# tests/integration/test_coordinator.py
import pytest
from pinocchio.coordinator import Coordinator

class TestCoordinator:
    """Test coordinator integration."""

    @pytest.mark.asyncio
    async def test_basic_request_processing(self, mock_config, temp_dir):
        """Test basic request processing."""
        # Setup coordinator with mock config
        coordinator = Coordinator(config_path=None)
        coordinator.config_manager.config = mock_config

        # Process test request
        request = {
            "task": "Generate a simple CUDA kernel",
            "requirements": {
                "language": "cuda",
                "optimization_level": "basic"
            }
        }

        result = await coordinator.process_request(request)

        assert result["success"] is True
        assert "session_id" in result
        assert "results" in result

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_multi_agent_workflow(self, mock_config):
        """Test multi-agent workflow execution."""
        coordinator = Coordinator(config_path=None)
        coordinator.config_manager.config = mock_config

        request = {
            "task": "Generate, debug, and optimize CUDA code",
            "workflow": "complete_pipeline"
        }

        result = await coordinator.process_request(request)

        # Verify all agents were involved
        assert "generator" in result.get("agent_participation", {})
        assert "debugger" in result.get("agent_participation", {})
        assert "optimizer" in result.get("agent_participation", {})
```

### Running Tests

```bash
# Run all tests
python -m pytest tests/

# Run specific test categories
python -m pytest tests/unit/                    # Unit tests only
python -m pytest tests/integration/             # Integration tests only
python -m pytest -m "not slow"                  # Skip slow tests
python -m pytest -m "not real_llm"             # Skip real LLM tests
python -m pytest -m "cuda_tools"               # Only CUDA tool tests

# Run with coverage
python -m pytest --cov=pinocchio --cov-report=html

# Run specific test file
python -m pytest tests/unit/test_tools.py -v

# Run specific test
python -m pytest tests/unit/test_tools.py::TestCudaDebugTools::test_syntax_checker -v

# Run tests in parallel
python -m pytest -n auto                       # Requires pytest-xdist
```

## Logging & Monitoring

### Logging Configuration

```python
# pinocchio/utils/logging_config.py
import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler

def setup_logging(config: Dict[str, Any]):
    """Setup comprehensive logging configuration."""
    log_level = getattr(logging, config.get("level", "INFO").upper())

    # Create logs directory
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)

    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    simple_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )

    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Console handler
    if config.get("console_output", True):
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(simple_formatter)
        root_logger.addHandler(console_handler)

    # File handler
    if config.get("file_output", True):
        file_handler = RotatingFileHandler(
            logs_dir / "pinocchio.log",
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(detailed_formatter)
        root_logger.addHandler(file_handler)

    # Component-specific loggers
    setup_component_loggers(config)

def setup_component_loggers(config: Dict[str, Any]):
    """Setup component-specific loggers."""
    component_configs = {
        'pinocchio.agents': {'level': 'INFO', 'file': 'agents.log'},
        'pinocchio.tools': {'level': 'DEBUG', 'file': 'tools.log'},
        'pinocchio.llm': {'level': 'INFO', 'file': 'llm.log'},
        'pinocchio.session': {'level': 'DEBUG', 'file': 'sessions.log'}
    }

    for component, comp_config in component_configs.items():
        logger = logging.getLogger(component)
        logger.setLevel(getattr(logging, comp_config['level']))

        # Component-specific file handler
        handler = RotatingFileHandler(
            Path("logs") / comp_config['file'],
            maxBytes=5 * 1024 * 1024,  # 5MB
            backupCount=3
        )
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
```

### Performance Monitoring

```python
# pinocchio/utils/performance_monitor.py
import time
import psutil
import threading
from collections import defaultdict
from typing import Dict, Any

class PerformanceMonitor:
    """Monitor system and application performance."""

    def __init__(self):
        self.metrics = defaultdict(list)
        self.start_time = time.time()
        self._monitoring = False
        self._monitor_thread = None

    def start_monitoring(self):
        """Start performance monitoring."""
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop)
        self._monitor_thread.daemon = True
        self._monitor_thread.start()

    def stop_monitoring(self):
        """Stop performance monitoring."""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join()

    def _monitor_loop(self):
        """Main monitoring loop."""
        while self._monitoring:
            timestamp = time.time() - self.start_time

            # System metrics
            self.metrics['cpu_percent'].append({
                'timestamp': timestamp,
                'value': psutil.cpu_percent()
            })

            self.metrics['memory_percent'].append({
                'timestamp': timestamp,
                'value': psutil.virtual_memory().percent
            })

            self.metrics['disk_io'].append({
                'timestamp': timestamp,
                'value': psutil.disk_io_counters()._asdict()
            })

            time.sleep(1)  # Sample every second

    def record_agent_metrics(self, agent_type: str, execution_time: float,
                           success: bool, llm_calls: int):
        """Record agent-specific metrics."""
        timestamp = time.time() - self.start_time

        self.metrics[f'agent_{agent_type}'].append({
            'timestamp': timestamp,
            'execution_time': execution_time,
            'success': success,
            'llm_calls': llm_calls
        })

    def record_tool_metrics(self, tool_name: str, execution_time: float,
                          status: str):
        """Record tool execution metrics."""
        timestamp = time.time() - self.start_time

        self.metrics[f'tool_{tool_name}'].append({
            'timestamp': timestamp,
            'execution_time': execution_time,
            'status': status
        })

    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        summary = {
            'monitoring_duration': time.time() - self.start_time,
            'system_metrics': {},
            'agent_metrics': {},
            'tool_metrics': {}
        }

        # System metrics summary
        if 'cpu_percent' in self.metrics:
            cpu_values = [m['value'] for m in self.metrics['cpu_percent']]
            summary['system_metrics']['cpu'] = {
                'avg': sum(cpu_values) / len(cpu_values),
                'max': max(cpu_values),
                'min': min(cpu_values)
            }

        # Agent metrics summary
        for key, values in self.metrics.items():
            if key.startswith('agent_'):
                agent_name = key[6:]  # Remove 'agent_' prefix
                execution_times = [v['execution_time'] for v in values]
                success_count = sum(1 for v in values if v['success'])

                summary['agent_metrics'][agent_name] = {
                    'call_count': len(values),
                    'success_rate': success_count / len(values) if values else 0,
                    'avg_execution_time': sum(execution_times) / len(execution_times) if execution_times else 0,
                    'total_llm_calls': sum(v['llm_calls'] for v in values)
                }

        return summary
```

## Performance Optimization

### Profiling Code

```python
# scripts/profile_performance.py
import cProfile
import pstats
import asyncio
from pinocchio.coordinator import Coordinator

async def profile_request_processing():
    """Profile request processing performance."""
    coordinator = Coordinator()

    # Sample request
    request = {
        "task": "Generate CUDA matrix multiplication kernel",
        "requirements": {
            "optimization_level": "high",
            "target_arch": "compute_75"
        }
    }

    # Profile the execution
    profiler = cProfile.Profile()
    profiler.enable()

    result = await coordinator.process_request(request)

    profiler.disable()

    # Save profiling results
    profiler.dump_stats('profile_results.prof')

    # Print top time consumers
    stats = pstats.Stats('profile_results.prof')
    stats.sort_stats('tottime')
    stats.print_stats(20)

    return result

if __name__ == "__main__":
    asyncio.run(profile_request_processing())
```

### Memory Usage Analysis

```python
# scripts/analyze_memory.py
import tracemalloc
import asyncio
from pinocchio.coordinator import Coordinator

async def analyze_memory_usage():
    """Analyze memory usage during execution."""
    # Start tracing
    tracemalloc.start()

    coordinator = Coordinator()

    # Take snapshot before
    snapshot1 = tracemalloc.take_snapshot()

    # Execute request
    request = {
        "task": "Generate multiple CUDA kernels",
        "count": 5
    }

    result = await coordinator.process_request(request)

    # Take snapshot after
    snapshot2 = tracemalloc.take_snapshot()

    # Analyze differences
    top_stats = snapshot2.compare_to(snapshot1, 'lineno')

    print("Top 10 memory differences:")
    for stat in top_stats[:10]:
        print(stat)

    # Current memory usage
    current, peak = tracemalloc.get_traced_memory()
    print(f"Current memory usage: {current / 1024 / 1024:.1f} MB")
    print(f"Peak memory usage: {peak / 1024 / 1024:.1f} MB")

    tracemalloc.stop()

if __name__ == "__main__":
    asyncio.run(analyze_memory_usage())
```

## Deployment Strategies

### Docker Configuration

```dockerfile
# Dockerfile
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install CUDA toolkit (optional)
# RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
# RUN mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
# ... (CUDA installation steps)

WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt requirements-prod.txt ./
RUN pip install --no-cache-dir -r requirements-prod.txt

# Copy application code
COPY pinocchio/ ./pinocchio/
COPY pinocchio.json.example ./pinocchio.json

# Create non-root user
RUN groupadd -r pinocchio && useradd -r -g pinocchio pinocchio
RUN chown -R pinocchio:pinocchio /app
USER pinocchio

# Create directories for data
RUN mkdir -p sessions memories knowledge logs

EXPOSE 8000

CMD ["python", "-m", "pinocchio", "--host", "0.0.0.0", "--port", "8000"]
```

### Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  pinocchio:
    build: .
    container_name: pinocchio-app
    environment:
      - PINOCCHIO_ENV=production
      - LLM_PROVIDER=${LLM_PROVIDER}
      - LLM_API_KEY=${LLM_API_KEY}
    volumes:
      - ./data/sessions:/app/sessions
      - ./data/memories:/app/memories
      - ./data/knowledge:/app/knowledge
      - ./data/logs:/app/logs
      - ./pinocchio.prod.json:/app/pinocchio.json
    ports:
      - "8000:8000"
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "python", "scripts/health_check.py"]
      interval: 30s
      timeout: 10s
      retries: 3

  # Optional: Redis for caching
  redis:
    image: redis:7-alpine
    container_name: pinocchio-redis
    volumes:
      - redis_data:/data
    restart: unless-stopped

  # Optional: Monitoring
  prometheus:
    image: prom/prometheus
    container_name: pinocchio-prometheus
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    ports:
      - "9090:9090"
    restart: unless-stopped

volumes:
  redis_data:
  prometheus_data:
```

### Kubernetes Deployment

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: pinocchio
  labels:
    app: pinocchio
spec:
  replicas: 3
  selector:
    matchLabels:
      app: pinocchio
  template:
    metadata:
      labels:
        app: pinocchio
    spec:
      containers:
      - name: pinocchio
        image: pinocchio:latest
        ports:
        - containerPort: 8000
        env:
        - name: PINOCCHIO_ENV
          value: "production"
        - name: LLM_API_KEY
          valueFrom:
            secretKeyRef:
              name: pinocchio-secrets
              key: llm-api-key
        volumeMounts:
        - name: config
          mountPath: /app/pinocchio.json
          subPath: pinocchio.json
        - name: data
          mountPath: /app/data
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
      volumes:
      - name: config
        configMap:
          name: pinocchio-config
      - name: data
        persistentVolumeClaim:
          claimName: pinocchio-data
```

## Troubleshooting Guide

### Common Issues and Solutions

#### 1. LLM Connection Issues

**Problem**: Cannot connect to LLM provider
```
ERROR: Cannot connect to host localhost:8001 ssl:default [Connect call failed ('127.0.0.1', 8001)]
```

**Solutions**:
```bash
# Check LLM service status
curl http://localhost:8001/v1/models

# Test with different provider
LLM_PROVIDER=openai python scripts/test_llm_connection.py

# Check configuration
python scripts/validate_config.py

# Check network connectivity
ping <llm-server-host>
telnet <llm-server-host> <port>
```

#### 2. CUDA Tools Not Found

**Problem**: CUDA tools (nvcc, cuda-memcheck) not available
```
ERROR: Command not found: nvcc
```

**Solutions**:
```bash
# Install CUDA toolkit
# Ubuntu/Debian:
sudo apt-get install nvidia-cuda-toolkit

# Check CUDA installation
nvcc --version
which nvcc

# Update PATH if needed
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Disable CUDA tools for development
# In pinocchio.json:
{
  "tools": {
    "enabled": false
  }
}
```

#### 3. Configuration Validation Errors

**Problem**: Configuration file validation fails
```
ERROR: 3 validation errors for PinocchioConfig
```

**Solutions**:
```bash
# Validate configuration
python scripts/validate_config.py

# Use example configuration as base
cp pinocchio.json.example pinocchio.json

# Check for missing required fields
# Compare with example configuration

# Use environment variables for overrides
export LLM_PROVIDER=openai
export LLM_API_KEY=your-key
```

#### 4. Memory/Storage Issues

**Problem**: Storage path not accessible
```
ERROR: Permission denied: ./sessions
```

**Solutions**:
```bash
# Create storage directories
mkdir -p sessions memories knowledge logs

# Fix permissions
chmod 755 sessions memories knowledge logs

# Use temporary directory for testing
export STORAGE_PATH=/tmp/pinocchio_test

# Check disk space
df -h .
```

#### 5. Plugin Loading Failures

**Problem**: Plugin initialization fails
```
ERROR: Failed to load plugin: custom_workflow
```

**Solutions**:
```bash
# Check plugin directory
ls -la ./plugins/

# Validate plugin configuration
python -c "import json; print(json.dumps(json.load(open('pinocchio.json'))['plugins'], indent=2))"

# Disable problematic plugins
# In pinocchio.json:
{
  "plugins": {
    "enabled": false
  }
}

# Check plugin dependencies
pip list | grep plugin-name
```

### Debug Mode Activation

```bash
# Enable maximum verbosity
export PINOCCHIO_DEBUG=1
export DEBUG_LEVEL=DEBUG

# Run with debug flags
python -m pinocchio --debug --verbose

# Enable tool debugging
export TOOL_DEBUG=1

# Enable session debugging
export SESSION_DEBUG=1
```

### Log Analysis

```bash
# View recent logs
tail -f logs/pinocchio.log

# Search for specific errors
grep -i error logs/pinocchio.log
grep -i "cuda" logs/tools.log

# Analyze session logs
python scripts/debug_session.py sessions/latest_session.json

# Check system logs
journalctl -u pinocchio -f  # If running as service
```

### Performance Troubleshooting

```bash
# Profile performance
python scripts/profile_performance.py

# Monitor memory usage
python scripts/analyze_memory.py

# Check system resources
htop
iotop
nvidia-smi  # For GPU usage
```

### Emergency Recovery

```bash
# Reset to clean state
rm -rf sessions/* memories/* knowledge/*

# Use minimal configuration
cp configs/minimal.json pinocchio.json

# Run basic health check
python scripts/health_check.py

# Test individual components
python -c "from pinocchio.config import ConfigManager; print('Config OK')"
python -c "from pinocchio.agents.generator import GeneratorAgent; print('Agents OK')"
python -c "from pinocchio.tools import ToolManager; print('Tools OK')"
```

---

This development guide provides comprehensive coverage of the development workflow, debugging tools, and operational procedures for the Pinocchio system, enabling efficient development and troubleshooting.
