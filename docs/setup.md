# System Setup & Installation Guide

This guide provides step-by-step instructions for setting up and installing the Pinocchio multi-agent CUDA programming system.

## Table of Contents

- [System Requirements](#system-requirements)
- [Installation Methods](#installation-methods)
- [Configuration Setup](#configuration-setup)
- [Verification & Testing](#verification--testing)
- [Docker Installation](#docker-installation)
- [Development Setup](#development-setup)
- [Troubleshooting](#troubleshooting)

## System Requirements

### Minimum Requirements

- **Operating System**: Linux (Ubuntu 18.04+, CentOS 7+), macOS 10.15+, Windows 10/11
- **Python**: 3.8 or higher
- **Memory**: 4GB RAM (8GB recommended)
- **Storage**: 2GB free disk space
- **Network**: Internet connection for LLM API access

### Recommended Requirements

- **Operating System**: Linux (Ubuntu 20.04+)
- **Python**: 3.9 or 3.10
- **Memory**: 16GB RAM
- **Storage**: 10GB free disk space (for sessions, models, etc.)
- **GPU**: NVIDIA GPU with CUDA support (for full tool functionality)
- **CUDA Toolkit**: 11.0 or higher (optional, for debugging/evaluation tools)

### Optional Dependencies

- **NVIDIA CUDA Toolkit**: For CUDA compilation and debugging tools
  - `nvcc` (CUDA compiler)
  - `cuda-memcheck` (Memory checker)
  - `nvprof` or `nsight` (Performance profiler)
- **Docker**: For containerized deployment
- **Git**: For version control and development

## Installation Methods

### Method 1: Standard Installation (Recommended)

#### Step 1: Clone Repository

```bash
# Clone the repository
git clone <repository-url>
cd pinocchio

# Or download and extract release archive
wget https://github.com/your-org/pinocchio/archive/v1.0.0.tar.gz
tar -xzf v1.0.0.tar.gz
cd pinocchio-1.0.0
```

#### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/macOS:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

#### Step 3: Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install production dependencies
pip install -r requirements.txt

# Optional: Install development dependencies
pip install -r requirements-dev.txt
```

#### Step 4: Install Pinocchio Package

```bash
# Install in development mode (recommended for customization)
pip install -e .

# Or install as package
pip install .
```

### Method 2: pip Installation (When Available)

```bash
# Install from PyPI (when published)
pip install pinocchio-cuda

# Or install from GitHub directly
pip install git+https://github.com/your-org/pinocchio.git
```

### Method 3: Conda Installation

```bash
# Create conda environment
conda create -n pinocchio python=3.9
conda activate pinocchio

# Install dependencies
conda install -c conda-forge asyncio aiohttp pydantic rich

# Install pinocchio
pip install -e .
```

## Configuration Setup

### Step 1: Create Configuration File

```bash
# Copy example configuration
cp pinocchio.json.example pinocchio.json

# Or create minimal configuration
cat > pinocchio.json << 'EOF'
{
  "llm": {
    "provider": "openai",
    "api_key": "your-api-key-here",
    "model_name": "gpt-3.5-turbo"
  },
  "agents": {
    "generator": {"enabled": true},
    "debugger": {"enabled": true},
    "optimizer": {"enabled": true},
    "evaluator": {"enabled": true}
  }
}
EOF
```

### Step 2: Configure LLM Provider

Choose one of the following LLM provider configurations:

#### OpenAI Configuration

```json
{
  "llm": {
    "provider": "openai",
    "api_key": "sk-your-openai-api-key",
    "model_name": "gpt-4",
    "timeout": 120,
    "max_retries": 3
  }
}
```

#### Custom LLM Server Configuration

```json
{
  "llm": {
    "provider": "custom",
    "base_url": "http://your-llm-server:8001",
    "model_name": "your-model-name",
    "timeout": 180,
    "max_retries": 3,
    "headers": {
      "Authorization": "Bearer your-token"
    }
  }
}
```

#### Anthropic Configuration

```json
{
  "llm": {
    "provider": "anthropic",
    "api_key": "your-anthropic-api-key",
    "model_name": "claude-3-opus-20240229",
    "timeout": 120,
    "max_retries": 3
  }
}
```

### Step 3: Set Environment Variables

```bash
# Set API key (alternative to config file)
export LLM_API_KEY="your-api-key-here"

# Set environment
export PINOCCHIO_ENV="development"

# Optional: Set custom paths
export STORAGE_PATH="./data"
export PLUGINS_DIRECTORY="./custom_plugins"
```

### Step 4: Create Storage Directories

```bash
# Create default storage directories
mkdir -p sessions memories knowledge logs

# Set appropriate permissions
chmod 755 sessions memories knowledge logs

# Optional: Create custom storage structure
mkdir -p data/{sessions,memories,knowledge,logs,backups}
```

## Verification & Testing

### Step 1: Basic Installation Test

```bash
# Test Python import
python -c "import pinocchio; print('✅ Pinocchio imported successfully')"

# Test CLI access
python -m pinocchio --version

# Test configuration loading
python -c "from pinocchio.config import ConfigManager; print('✅ Configuration loaded')"
```

### Step 2: LLM Connection Test

```bash
# Test LLM connection
python scripts/test_llm_connection.py

# Expected output:
# Testing LLM connection:
#   Provider: openai
#   Model: gpt-3.5-turbo
# ✅ LLM connection successful!
```

### Step 3: System Health Check

```bash
# Run comprehensive health check
python scripts/health_check.py

# Expected output:
# 🔍 Running Pinocchio System Health Check...
# ✅ Configuration is valid
# ✅ LLM connection working
# ⚠️  Some CUDA tools not available (optional)
# ✅ All storage paths accessible
# ✅ Dependencies available
#
# 🏥 Overall Status: ✅ SYSTEM HEALTHY
```

### Step 4: Run Test Suite

```bash
# Run fast tests (excludes slow/external dependencies)
./scripts/run_fast_tests.sh

# Or run all tests
python -m pytest tests/ -v

# Run specific test categories
python -m pytest tests/unit/ -v                    # Unit tests
python -m pytest tests/integration/ -v             # Integration tests
python -m pytest -m "not real_llm" -v             # Skip real LLM tests
```

### Step 5: Basic Functionality Test

```bash
# Test basic request processing
python -c "
import asyncio
from pinocchio.coordinator import Coordinator

async def test():
    coordinator = Coordinator()
    result = await coordinator.process_request({
        'task': 'Generate a simple CUDA vector addition kernel',
        'requirements': {'language': 'cuda'}
    })
    print('✅ Basic functionality working')
    print(f'Session ID: {result.get(\"session_id\", \"unknown\")}')

asyncio.run(test())
"
```

## Docker Installation

### Step 1: Using Pre-built Image

```bash
# Pull pre-built image (when available)
docker pull pinocchio/cuda-programming:latest

# Run container
docker run -d \
    --name pinocchio \
    -p 8000:8000 \
    -e LLM_API_KEY="your-api-key" \
    -v $(pwd)/data:/app/data \
    pinocchio/cuda-programming:latest
```

### Step 2: Build from Source

```bash
# Build Docker image
docker build -t pinocchio:local .

# Run with custom configuration
docker run -d \
    --name pinocchio \
    -p 8000:8000 \
    -v $(pwd)/pinocchio.json:/app/pinocchio.json \
    -v $(pwd)/data:/app/data \
    -e PINOCCHIO_ENV=production \
    pinocchio:local
```

### Step 3: Docker Compose Setup

```bash
# Copy docker-compose configuration
cp docker-compose.yml.example docker-compose.yml

# Edit environment variables
cat > .env << 'EOF'
LLM_PROVIDER=openai
LLM_API_KEY=your-api-key-here
PINOCCHIO_ENV=production
EOF

# Start services
docker-compose up -d

# Check status
docker-compose ps
docker-compose logs pinocchio
```

### Step 4: CUDA-Enabled Docker (Optional)

For full CUDA tool functionality:

```bash
# Use NVIDIA runtime
docker run --gpus all \
    --name pinocchio-cuda \
    -p 8000:8000 \
    -e LLM_API_KEY="your-api-key" \
    -v $(pwd)/data:/app/data \
    pinocchio/cuda-programming:cuda-latest
```

## Development Setup

### Step 1: Clone for Development

```bash
# Clone with development branches
git clone -b develop <repository-url>
cd pinocchio

# Setup pre-commit hooks
pip install pre-commit
pre-commit install
```

### Step 2: Install Development Dependencies

```bash
# Create development environment
python -m venv venv-dev
source venv-dev/bin/activate

# Install all dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt
pip install -e .

# Install additional development tools
pip install black flake8 mypy pytest-xdist
```

### Step 3: IDE Configuration

#### VS Code Setup

```bash
# Create VS Code workspace
mkdir .vscode
cat > .vscode/settings.json << 'EOF'
{
    "python.defaultInterpreterPath": "./venv-dev/bin/python",
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.formatting.provider": "black",
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": ["tests/"]
}
EOF
```

#### PyCharm Setup

1. Open project in PyCharm
2. Configure interpreter: `venv-dev/bin/python`
3. Set pytest as test runner
4. Configure Black as code formatter
5. Enable flake8 for linting

### Step 4: Development Configuration

```bash
# Create development configuration
cp pinocchio.json.example pinocchio.dev.json

# Edit for development
cat > pinocchio.dev.json << 'EOF'
{
  "llm": {
    "provider": "openai",
    "api_key": "your-dev-api-key",
    "model_name": "gpt-3.5-turbo",
    "timeout": 60
  },
  "agents": {
    "generator": {"enabled": true, "max_retries": 2},
    "debugger": {"enabled": true, "max_retries": 2},
    "optimizer": {"enabled": false},
    "evaluator": {"enabled": false}
  },
  "tools": {"enabled": false},
  "verbose": {"enabled": true, "level": "maximum"},
  "logging": {"level": "DEBUG", "console_output": true}
}
EOF

# Use development config
export PINOCCHIO_CONFIG=pinocchio.dev.json
```

## Troubleshooting

### Common Installation Issues

#### Issue: Python Version Incompatibility

```bash
# Check Python version
python --version

# If using Python < 3.8, install newer version
# Ubuntu/Debian:
sudo apt update
sudo apt install python3.9 python3.9-venv python3.9-pip

# Create environment with specific Python version
python3.9 -m venv venv
```

#### Issue: pip Install Failures

```bash
# Update pip and setuptools
pip install --upgrade pip setuptools wheel

# Install with verbose output to see errors
pip install -v -r requirements.txt

# For specific package issues:
pip install --no-cache-dir package-name

# Clear pip cache
pip cache purge
```

#### Issue: Permission Errors

```bash
# Fix directory permissions
sudo chown -R $USER:$USER pinocchio/
chmod -R 755 pinocchio/

# Use user install if needed
pip install --user -r requirements.txt

# Create directories with proper permissions
mkdir -p ~/.local/share/pinocchio/{sessions,memories,knowledge}
```

#### Issue: Network/Firewall Problems

```bash
# Test internet connectivity
curl -I https://api.openai.com/v1/models

# Configure proxy if needed
export HTTP_PROXY=http://proxy.company.com:8080
export HTTPS_PROXY=http://proxy.company.com:8080

# Install with proxy
pip install --proxy http://proxy.company.com:8080 -r requirements.txt
```

### Configuration Issues

#### Issue: Invalid Configuration File

```bash
# Validate JSON syntax
python -m json.tool pinocchio.json

# Use configuration validator
python scripts/validate_config.py

# Start with minimal configuration
cat > pinocchio.json << 'EOF'
{
  "llm": {
    "provider": "openai",
    "api_key": "your-api-key"
  }
}
EOF
```

#### Issue: LLM Connection Problems

```bash
# Test API key
curl -H "Authorization: Bearer your-api-key" \
     https://api.openai.com/v1/models

# Test custom LLM server
curl http://your-llm-server:8001/v1/models

# Use alternative provider
export LLM_PROVIDER=anthropic
export LLM_API_KEY=your-anthropic-key
```

#### Issue: CUDA Tools Not Found

```bash
# Check CUDA installation
nvcc --version
which nvcc

# Install CUDA toolkit (Ubuntu)
sudo apt update
sudo apt install nvidia-cuda-toolkit

# Add CUDA to PATH
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Disable CUDA tools for now
export TOOLS_ENABLED=false
```

### Runtime Issues

#### Issue: High Memory Usage

```bash
# Monitor memory usage
python scripts/analyze_memory.py

# Reduce session size limits
# In pinocchio.json:
{
  "session": {
    "max_session_size_mb": 50,
    "compression": {"enabled": true}
  }
}

# Clear old sessions
rm -rf sessions/*
```

#### Issue: Slow Performance

```bash
# Profile performance
python scripts/profile_performance.py

# Reduce timeout values
# In pinocchio.json:
{
  "llm": {"timeout": 60},
  "agents": {
    "generator": {"timeout": 120}
  }
}

# Disable verbose logging
export VERBOSE_ENABLED=false
export DEBUG_LEVEL=WARNING
```

### Getting Help

#### Debug Information Collection

```bash
# Collect system information
python -c "
import sys, platform, pip
print(f'Python: {sys.version}')
print(f'Platform: {platform.platform()}')
print(f'Pip packages:')
import pkg_resources
for pkg in pkg_resources.working_set:
    if 'pinocchio' in pkg.key or pkg.key in ['aiohttp', 'pydantic', 'rich']:
        print(f'  {pkg.key}: {pkg.version}')
"

# Generate health report
python scripts/health_check.py > health_report.txt

# Check recent logs
tail -50 logs/pinocchio.log
```

#### Support Channels

1. **Documentation**: Check docs/ directory for detailed guides
2. **GitHub Issues**: Report bugs and request features
3. **Health Check**: Run `python scripts/health_check.py` for diagnosis
4. **Configuration Validation**: Use `python scripts/validate_config.py`

#### Creating Minimal Reproduction

```bash
# Create minimal test case
cat > test_minimal.py << 'EOF'
import asyncio
from pinocchio.coordinator import Coordinator

async def minimal_test():
    coordinator = Coordinator()
    result = await coordinator.process_request({
        "task": "test request"
    })
    print(f"Result: {result}")

if __name__ == "__main__":
    asyncio.run(minimal_test())
EOF

python test_minimal.py
```

---

This setup guide provides comprehensive instructions for installing and configuring the Pinocchio system across different environments and use cases.
