# Pinocchio

Pinocchio is a multi-agent system for automatically writing, debugging, and optimizing Choreo compute kernel DSL operators. The system is implemented in Python, supporting end-to-end Prompt engineering to chain Agents, with message passing based on a lightweight message queue, and data and interactions using JSON format standards.

## Features

- Multi-agent collaboration: Generator, Debugger, Optimizer, Evaluator
- Standard JSON message format
- Lightweight in-memory message queue
- Complete interaction logging
- Knowledge base and Prompt template version management
- Support for OpenAI and Anthropic LLM interfaces

## Installation

This project uses Poetry for dependency management. Make sure you have Poetry installed, then run:

```bash
poetry install
```

## Usage

1. Configure environment variables:
```bash
export OPENAI_API_KEY=your_key_here
export ANTHROPIC_API_KEY=your_key_here
```

2. Run tests:
```bash
poetry run pytest
```

3. Use CLI:
```bash
poetry run invoke --list  # View available commands
```

## Project Structure

```
pinocchio/
├── agents/           # Multi-agent Agent modules
├── workflows/        # Task scheduling and collaboration workflows
├── knowledge/        # Choreo-related knowledge and Prompt templates
├── memory/           # Interaction records and summary log management
├── prompt/           # Prompt template management and formatting
├── llm/              # LLM call encapsulation
├── config/           # Configuration management module
├── session/          # Session management module
├── tests/            # Test code
└── pyproject.toml    # Poetry project configuration
```

## Development

1. Setup development environment:
```bash
# Run the setup script to install dependencies and pre-commit hooks
bash scripts/setup_dev.sh
```

2. Run code formatting:
```bash
poetry run black .
poetry run isort .
```

3. Run type checking:
```bash
poetry run mypy .
```

4. Run linting:
```bash
poetry run flake8 pinocchio/ tests/
```

5. Pre-commit hooks:
```bash
# Install pre-commit hooks (already done by setup_dev.sh)
poetry run pre-commit install

# Run pre-commit hooks on all files
poetry run pre-commit run --all-files

# Run a specific hook
poetry run pre-commit run black --all-files
```

## Unit Testing

The project uses pytest as the testing framework. Here are some common test commands:

### Run all tests

```bash
# Use PYTHONPATH to ensure modules can be imported correctly
PYTHONPATH=. pytest tests/
```

### Run tests with coverage reports

```bash
# Generate console coverage report
PYTHONPATH=. pytest tests/ --cov=pinocchio

# Generate detailed HTML coverage report
PYTHONPATH=. pytest tests/ --cov=pinocchio --cov-report=html
```

The HTML report will be generated in the `htmlcov` directory. You can open `htmlcov/index.html` in a browser to view the detailed report.

### Run specific tests

```bash
# Run a specific test file
PYTHONPATH=. pytest tests/test_config.py

# Run a specific test function
PYTHONPATH=. pytest tests/test_config.py::test_settings_load_from_dict

# Run tests with verbose output
PYTHONPATH=. pytest tests/test_config.py -v
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details. 