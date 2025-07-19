# Pinocchio Developer Guide

Welcome to the Pinocchio development guide! This document provides comprehensive information for developers who want to contribute to, extend, or understand the Pinocchio multi-agent system.

## Table of Contents

- [Development Setup](#development-setup)
- [Architecture Overview](#architecture-overview)
- [Contributing Guidelines](#contributing-guidelines)
- [Testing Procedures](#testing-procedures)
- [Deployment Instructions](#deployment-instructions)
- [Development Workflow](#development-workflow)
- [Code Standards](#code-standards)

## Development Setup

### Prerequisites

- Python 3.9 or higher
- Poetry (for dependency management)
- Git
- CUDA toolkit (optional, for GPU acceleration)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/LancerLab/pinocchio.git
   cd pinocchio
   ```

2. **Install dependencies:**
   ```bash
   poetry install
   ```

3. **Activate the virtual environment:**
   ```bash
   poetry shell
   ```

4. **Install pre-commit hooks:**
   ```bash
   pre-commit install
   ```

5. **Verify installation:**
   ```bash
   python -c "from pinocchio.config import ConfigManager; print('✅ Installation successful')"
   ```

### Configuration

1. **Create configuration file:**
   ```bash
   cp configs/debug.json pinocchio.json
   ```

2. **Update LLM configuration:**
   Edit `pinocchio.json` to configure your LLM provider:
   ```json
   {
     "llms": [
       {
         "id": "main",
         "provider": "custom",
         "base_url": "http://localhost:8001",
         "model_name": "your-model",
         "priority": 1
       }
     ]
   }
   ```

## Architecture Overview

Pinocchio is a multi-agent system designed for high-performance code generation and optimization. The system consists of several key components:

### Core Components

1. **Agents** (`pinocchio/agents/`)
   - **Generator Agent**: Creates initial code implementations
   - **Debugger Agent**: Analyzes and fixes code issues
   - **Evaluator Agent**: Assesses code performance and quality
   - **Optimizer Agent**: Improves code performance

2. **Coordinator** (`pinocchio/session/coordinator.py`)
   - Orchestrates multi-agent workflows
   - Manages task execution and agent communication
   - Handles session lifecycle

3. **Task Planning** (`pinocchio/task_planning/`)
   - Creates execution plans for user requests
   - Manages task dependencies and sequencing
   - Supports both workflow and adaptive planning strategies

4. **Configuration Management** (`pinocchio/config/`)
   - Centralized configuration system
   - LLM provider management
   - Agent-specific settings

5. **Memory System** (`pinocchio/memory/`)
   - Stores agent interactions and results
   - Provides context for future operations
   - Supports session-based organization

6. **Knowledge Management** (`pinocchio/knowledge/`)
   - Maintains domain-specific knowledge bases
   - Supports CUDA programming expertise
   - Enables knowledge-enhanced prompts

### Data Flow

```
User Request → Coordinator → Task Planner → Agent Execution → Results
     ↑                                           ↓
     └── Session Management ← Memory Storage ←──┘
```

## Contributing Guidelines

### Code Style

- Follow PEP 8 Python style guidelines
- Use Black for code formatting
- Use isort for import sorting
- All code, comments, and documentation must be in English
- No Chinese characters in source code (except in .md/.mdc files)

### Commit Messages

Follow conventional commit format:
```
type(scope): description

[optional body]

[optional footer]
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

### Pull Request Process

1. Create a feature branch from `main`
2. Make your changes with appropriate tests
3. Ensure all tests pass
4. Update documentation if needed
5. Submit a pull request with clear description

### Code Review

- All changes require code review
- Maintain test coverage above 90%
- Ensure backward compatibility
- Document breaking changes

## Testing Procedures

### Running Tests

1. **Full test suite:**
   ```bash
   python -m pytest tests/
   ```

2. **Unit tests only:**
   ```bash
   python -m pytest tests/unittests/
   ```

3. **Integration tests:**
   ```bash
   python -m pytest tests/integration/
   ```

4. **With coverage:**
   ```bash
   python -m pytest tests/ --cov=pinocchio --cov-report=html
   ```

### Test Categories

- **Unit Tests**: Test individual components in isolation
- **Integration Tests**: Test component interactions
- **End-to-End Tests**: Test complete workflows
- **Performance Tests**: Validate system performance

### Writing Tests

- Use pytest framework
- Follow AAA pattern (Arrange, Act, Assert)
- Mock external dependencies
- Test both success and failure scenarios
- Include performance benchmarks for critical paths

## Deployment Instructions

### Local Development

```bash
# Start the CLI interface
python -m pinocchio.cli.main

# Run with specific configuration
PINOCCHIO_CONFIG_FILE=./custom-config.json python -m pinocchio.cli.main
```

### Production Deployment

1. **Environment Setup:**
   ```bash
   # Set production configuration
   export PINOCCHIO_CONFIG_FILE=/path/to/production-config.json
   export PINOCCHIO_LOG_LEVEL=INFO
   ```

2. **Service Configuration:**
   - Configure LLM endpoints
   - Set up persistent storage
   - Configure logging and monitoring

3. **Health Checks:**
   ```bash
   python scripts/health_check.py
   ```

### Docker Deployment

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY . .
RUN pip install poetry && poetry install --no-dev

CMD ["python", "-m", "pinocchio.cli.main"]
```

## Development Workflow

### Feature Development

1. **Planning Phase:**
   - Create GitHub issue
   - Design API changes
   - Plan testing strategy

2. **Implementation Phase:**
   - Create feature branch
   - Implement core functionality
   - Add comprehensive tests
   - Update documentation

3. **Review Phase:**
   - Submit pull request
   - Address review feedback
   - Ensure CI passes

4. **Release Phase:**
   - Merge to main
   - Update changelog
   - Tag release if needed

### Debugging

1. **Enable verbose logging:**
   ```json
   {
     "verbose": {
       "mode": "development",
       "level": "maximum"
     }
   }
   ```

2. **Use debugging tools:**
   ```bash
   # Run with debugger
   python -m pdb -m pinocchio.cli.main
   
   # Profile performance
   python -m cProfile -o profile.stats -m pinocchio.cli.main
   ```

## Code Standards

### Documentation

- All public APIs must have docstrings
- Use Google-style docstrings
- Include type hints for all functions
- Maintain up-to-date README files

### Error Handling

- Use specific exception types
- Provide meaningful error messages
- Log errors with appropriate levels
- Implement graceful degradation

### Performance

- Profile critical code paths
- Use async/await for I/O operations
- Implement caching where appropriate
- Monitor memory usage

### Security

- Validate all inputs
- Use secure defaults
- Protect sensitive configuration
- Regular dependency updates

## Additional Resources

- [Architecture Documentation](docs/developer-guides/architecture.md)
- [Testing Guide](docs/developer-guides/testing.md)
- [API Reference](docs/api-reference/)
- [User Guides](docs/user-guides/)

## Getting Help

- Create GitHub issues for bugs
- Use discussions for questions
- Check existing documentation
- Review code examples in tests

---

For more detailed information, see the documentation in the `docs/` directory.
