# Configuration Guide

This guide provides comprehensive documentation for configuring the Pinocchio multi-agent CUDA programming system.

## Table of Contents

- [Configuration Overview](#configuration-overview)
- [Main Configuration File](#main-configuration-file)
- [Configuration Sections](#configuration-sections)
- [Environment Variables](#environment-variables)
- [Configuration Examples](#configuration-examples)
- [Validation and Troubleshooting](#validation-and-troubleshooting)

## Configuration Overview

Pinocchio uses a JSON-based configuration system with support for environment variable overrides. The main configuration file is `pinocchio.json`, located in the project root.

### Configuration Hierarchy

```
1. Default values (in code)
2. Main configuration file (pinocchio.json)
3. Environment-specific overrides
4. Environment variables
5. Command-line arguments (if applicable)
```

### Configuration Loading Process

```python
# Configuration is loaded in this order:
1. Load pinocchio.json
2. Apply environment-specific settings based on PINOCCHIO_ENV
3. Override with environment variables
4. Validate configuration with Pydantic models
5. Initialize components with validated configuration
```

## Main Configuration File

### Basic Structure

```json
{
  "llm": { ... },
  "agents": { ... },
  "plugins": { ... },
  "tools": { ... },
  "workflow": { ... },
  "session": { ... },
  "storage": { ... },
  "verbose": { ... },
  "logging": { ... }
}
```

### Complete Example Configuration

```json
{
  "llm": {
    "provider": "custom",
    "base_url": "http://10.0.16.46:8001",
    "model_name": "Qwen/Qwen3-32B",
    "timeout": 120,
    "max_retries": 3,
    "api_key": null,
    "headers": null,
    "priority": 10,
    "label": null
  },
  "llm_generator": null,
  "llm_optimizer": null,
  "llm_debugger": null,
  "llm_evaluator": null,
  "agents": {
    "generator": {
      "enabled": true,
      "max_retries": 3,
      "timeout": null
    },
    "debugger": {
      "enabled": true,
      "max_retries": 3,
      "timeout": null
    },
    "optimizer": {
      "enabled": true,
      "max_retries": 3,
      "timeout": null
    },
    "evaluator": {
      "enabled": true,
      "max_retries": 3,
      "timeout": null
    }
  },
  "plugins": {
    "enabled": true,
    "plugins_directory": "./plugins",
    "active_plugins": {
      "prompt": "cuda_prompt_plugin",
      "workflow": "json_workflow_plugin",
      "agent": "custom_agent_plugin"
    },
    "plugin_configs": {
      "cuda_prompt_plugin": {
        "expertise_level": "expert",
        "target_domain": "CUDA"
      },
      "json_workflow_plugin": {
        "workflows": {
          "cuda_development": {
            "name": "CUDA Development Workflow",
            "description": "Complete CUDA development pipeline",
            "tasks": [
              {
                "id": "generate_cuda_code",
                "agent_type": "generator",
                "description": "Generate optimized CUDA implementation",
                "requirements": {
                  "language": "cuda",
                  "optimization_level": "high",
                  "include_host_code": true
                },
                "optimization_goals": ["performance", "memory_efficiency"],
                "priority": "critical"
              }
            ]
          }
        }
      }
    }
  },
  "workflow": {
    "use_plugin": true,
    "fallback_to_task_planning": true,
    "default_workflow": "cuda_development",
    "task_planning_as_backup": true
  },
  "session": {
    "auto_save": true,
    "save_interval_seconds": 300,
    "max_session_size_mb": 100
  },
  "storage": {
    "sessions_path": "./sessions",
    "memories_path": "./memories",
    "knowledge_path": "./knowledge"
  },
  "tools": {
    "enabled": true,
    "debug_tools": {
      "cuda_compile": {
        "enabled": true,
        "timeout": 60,
        "default_arch": "compute_75"
      },
      "cuda_memcheck": {
        "enabled": true,
        "timeout": 120,
        "default_check_type": "memcheck"
      },
      "cuda_syntax_check": {
        "enabled": true,
        "timeout": 30,
        "strict_mode": true
      }
    },
    "eval_tools": {
      "cuda_profile": {
        "enabled": true,
        "timeout": 180,
        "default_profiler": "nvprof",
        "default_metrics": ["gld_efficiency", "gst_efficiency", "sm_efficiency", "achieved_occupancy"]
      },
      "cuda_occupancy": {
        "enabled": true,
        "timeout": 30,
        "default_block_size": 256
      },
      "cuda_performance_analyze": {
        "enabled": true,
        "timeout": 60,
        "default_analysis_type": "comprehensive"
      }
    }
  },
  "debug_repair": {
    "max_repair_attempts": 3
  },
  "optimization": {
    "max_optimisation_rounds": 3,
    "optimizer_enabled": true
  },
  "verbose": {
    "enabled": true,
    "level": "maximum",
    "show_agent_instructions": true,
    "show_execution_times": true,
    "show_task_details": true,
    "show_progress_updates": true
  },
  "logging": {
    "level": "DEBUG",
    "console_output": true,
    "file_output": true
  }
}
```

## Configuration Sections

### LLM Configuration

Controls language model provider settings and connection parameters.

```json
{
  "llm": {
    "provider": "custom|openai|anthropic|cohere",
    "base_url": "http://localhost:8001",
    "model_name": "Qwen/Qwen3-32B",
    "timeout": 120,
    "max_retries": 3,
    "api_key": "your-api-key-here",
    "headers": {
      "Custom-Header": "value"
    },
    "priority": 10,
    "label": "primary-llm"
  }
}
```

**Configuration Options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `provider` | string | "custom" | LLM provider type |
| `base_url` | string | null | API base URL for custom providers |
| `model_name` | string | "default" | Model identifier |
| `timeout` | integer | 120 | Request timeout in seconds |
| `max_retries` | integer | 3 | Maximum retry attempts |
| `api_key` | string | null | API authentication key |
| `headers` | object | null | Custom HTTP headers |
| `priority` | integer | 10 | Provider priority (for fallback) |
| `label` | string | null | Human-readable provider label |

**Agent-Specific LLM Configuration:**

You can override LLM settings for specific agents:

```json
{
  "llm": { /* default LLM config */ },
  "llm_generator": {
    "provider": "openai",
    "model_name": "gpt-4",
    "timeout": 180
  },
  "llm_debugger": {
    "provider": "anthropic",
    "model_name": "claude-3-opus"
  }
}
```

### Agent Configuration

Controls individual agent behavior and capabilities.

```json
{
  "agents": {
    "generator": {
      "enabled": true,
      "max_retries": 3,
      "timeout": 300,
      "custom_settings": {
        "creativity_level": "high",
        "code_style": "optimized"
      }
    },
    "debugger": {
      "enabled": true,
      "max_retries": 5,
      "timeout": 180,
      "tool_integration": true
    },
    "optimizer": {
      "enabled": true,
      "max_retries": 3,
      "timeout": 240,
      "optimization_strategies": ["memory", "performance", "occupancy"]
    },
    "evaluator": {
      "enabled": true,
      "max_retries": 2,
      "timeout": 300,
      "evaluation_depth": "comprehensive"
    }
  }
}
```

**Agent Configuration Options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `enabled` | boolean | true | Enable/disable agent |
| `max_retries` | integer | 3 | Maximum retry attempts for failed operations |
| `timeout` | integer | null | Agent-specific timeout (seconds) |
| `custom_settings` | object | {} | Agent-specific custom configuration |

### Plugin Configuration

Controls the plugin system and individual plugin settings.

```json
{
  "plugins": {
    "enabled": true,
    "plugins_directory": "./plugins",
    "auto_discover": true,
    "active_plugins": {
      "prompt": "cuda_prompt_plugin",
      "workflow": "json_workflow_plugin",
      "agent": "custom_agent_plugin",
      "tool": "external_tool_plugin"
    },
    "plugin_configs": {
      "cuda_prompt_plugin": {
        "expertise_level": "expert",
        "target_domain": "CUDA",
        "include_examples": true,
        "context_length": 2000
      },
      "json_workflow_plugin": {
        "validation_strict": true,
        "workflow_timeout": 3600,
        "workflows": {
          "custom_workflow": {
            "name": "Custom Development Workflow",
            "tasks": [...]
          }
        }
      }
    }
  }
}
```

**Plugin Configuration Options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `enabled` | boolean | true | Enable/disable plugin system |
| `plugins_directory` | string | "./plugins" | Plugin discovery directory |
| `auto_discover` | boolean | true | Automatically discover plugins |
| `active_plugins` | object | {} | Map of plugin types to active plugins |
| `plugin_configs` | object | {} | Individual plugin configurations |

### Tools Configuration

Controls MCP (Model Context Protocol) tools integration.

```json
{
  "tools": {
    "enabled": true,
    "timeout_default": 60,
    "max_concurrent": 5,
    "debug_tools": {
      "cuda_compile": {
        "enabled": true,
        "timeout": 60,
        "default_arch": "compute_75",
        "include_paths": ["/usr/local/cuda/include"],
        "compiler_flags": ["-O3", "-lineinfo"]
      },
      "cuda_memcheck": {
        "enabled": true,
        "timeout": 120,
        "default_check_type": "memcheck",
        "additional_checks": ["racecheck", "synccheck"]
      },
      "cuda_syntax_check": {
        "enabled": true,
        "timeout": 30,
        "strict_mode": true,
        "check_performance": true
      }
    },
    "eval_tools": {
      "cuda_profile": {
        "enabled": true,
        "timeout": 180,
        "default_profiler": "nvprof",
        "default_metrics": [
          "gld_efficiency",
          "gst_efficiency",
          "sm_efficiency",
          "achieved_occupancy"
        ],
        "profile_iterations": 100
      },
      "cuda_occupancy": {
        "enabled": true,
        "timeout": 30,
        "default_block_size": 256,
        "architectures": ["compute_75", "compute_80", "compute_86"]
      },
      "cuda_performance_analyze": {
        "enabled": true,
        "timeout": 60,
        "default_analysis_type": "comprehensive",
        "analysis_categories": ["memory", "compute", "general"]
      }
    }
  }
}
```

### Workflow Configuration

Controls workflow execution and task planning.

```json
{
  "workflow": {
    "use_plugin": true,
    "fallback_to_task_planning": true,
    "default_workflow": "cuda_development",
    "task_planning_as_backup": true,
    "execution_settings": {
      "max_parallel_tasks": 3,
      "task_timeout": 600,
      "retry_failed_tasks": true,
      "dependency_checking": true
    }
  }
}
```

**Workflow Configuration Options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `use_plugin` | boolean | true | Use plugin-based workflows |
| `fallback_to_task_planning` | boolean | true | Fallback to automatic task planning |
| `default_workflow` | string | null | Default workflow to use |
| `task_planning_as_backup` | boolean | true | Use task planning as backup |
| `execution_settings` | object | {} | Workflow execution parameters |

### Session Configuration

Controls session management and persistence.

```json
{
  "session": {
    "auto_save": true,
    "save_interval_seconds": 300,
    "max_session_size_mb": 100,
    "cleanup_old_sessions": true,
    "session_retention_days": 30,
    "compression": {
      "enabled": true,
      "algorithm": "gzip",
      "level": 6
    }
  }
}
```

### Storage Configuration

Controls data storage locations and settings.

```json
{
  "storage": {
    "sessions_path": "./sessions",
    "memories_path": "./memories",
    "knowledge_path": "./knowledge",
    "logs_path": "./logs",
    "temp_path": "./temp",
    "backup_settings": {
      "enabled": true,
      "backup_path": "./backups",
      "backup_interval_hours": 24,
      "max_backups": 7
    }
  }
}
```

### Verbose and Logging Configuration

Controls system verbosity and logging behavior.

```json
{
  "verbose": {
    "enabled": true,
    "level": "maximum",
    "show_agent_instructions": true,
    "show_execution_times": true,
    "show_task_details": true,
    "show_progress_updates": true,
    "show_llm_interactions": false,
    "show_tool_executions": true
  },
  "logging": {
    "level": "DEBUG",
    "console_output": true,
    "file_output": true,
    "log_file": "pinocchio.log",
    "max_file_size_mb": 50,
    "backup_count": 5,
    "format": "detailed",
    "include_timestamps": true
  }
}
```

**Verbosity Levels:**

- `minimal`: Essential information only
- `normal`: Standard operational information
- `detailed`: Detailed execution information
- `maximum`: All available information including debug details

**Logging Levels:**

- `DEBUG`: Detailed debugging information
- `INFO`: General information messages
- `WARNING`: Warning messages
- `ERROR`: Error messages only
- `CRITICAL`: Critical errors only

## Environment Variables

### Core Environment Variables

```bash
# Environment Configuration
export PINOCCHIO_ENV=development|testing|production

# LLM Configuration Overrides
export LLM_PROVIDER=openai
export LLM_API_KEY=your-api-key-here
export LLM_BASE_URL=http://custom-llm-server:8001
export LLM_MODEL_NAME=gpt-4
export LLM_TIMEOUT=180

# System Configuration
export DEBUG_LEVEL=DEBUG
export STORAGE_PATH=/custom/storage/path
export PLUGINS_DIRECTORY=/custom/plugins/path

# Feature Toggles
export TOOLS_ENABLED=true
export PLUGINS_ENABLED=false
export VERBOSE_ENABLED=true

# Testing Configuration
export FAST_TEST=1
export MOCK_LLM=true
export SKIP_CUDA_TOOLS=true
```

### Environment-Specific Configurations

#### Development Environment

```bash
export PINOCCHIO_ENV=development
export DEBUG_LEVEL=DEBUG
export VERBOSE_ENABLED=true
export TOOLS_ENABLED=true
export PLUGINS_ENABLED=true
export AUTO_SAVE_SESSIONS=false
```

#### Testing Environment

```bash
export PINOCCHIO_ENV=testing
export DEBUG_LEVEL=WARNING
export VERBOSE_ENABLED=false
export TOOLS_ENABLED=false
export PLUGINS_ENABLED=false
export MOCK_LLM=true
export FAST_TEST=1
```

#### Production Environment

```bash
export PINOCCHIO_ENV=production
export DEBUG_LEVEL=ERROR
export VERBOSE_ENABLED=false
export TOOLS_ENABLED=true
export PLUGINS_ENABLED=true
export SESSION_COMPRESSION=true
export BACKUP_ENABLED=true
```

## Configuration Examples

### Development Configuration

```json
{
  "llm": {
    "provider": "openai",
    "model_name": "gpt-3.5-turbo",
    "timeout": 60,
    "max_retries": 2
  },
  "agents": {
    "generator": {"enabled": true, "max_retries": 2},
    "debugger": {"enabled": true, "max_retries": 3},
    "optimizer": {"enabled": false},
    "evaluator": {"enabled": false}
  },
  "tools": {
    "enabled": false
  },
  "verbose": {
    "enabled": true,
    "level": "maximum"
  },
  "logging": {
    "level": "DEBUG",
    "console_output": true,
    "file_output": false
  }
}
```

### Production Configuration

```json
{
  "llm": {
    "provider": "custom",
    "base_url": "http://production-llm-server:8001",
    "model_name": "production-model",
    "timeout": 180,
    "max_retries": 5
  },
  "agents": {
    "generator": {"enabled": true, "max_retries": 3},
    "debugger": {"enabled": true, "max_retries": 3},
    "optimizer": {"enabled": true, "max_retries": 3},
    "evaluator": {"enabled": true, "max_retries": 3}
  },
  "tools": {
    "enabled": true,
    "debug_tools": {
      "cuda_compile": {"enabled": true},
      "cuda_memcheck": {"enabled": false},
      "cuda_syntax_check": {"enabled": true}
    }
  },
  "session": {
    "auto_save": true,
    "save_interval_seconds": 60,
    "compression": {"enabled": true}
  },
  "verbose": {
    "enabled": false
  },
  "logging": {
    "level": "WARNING",
    "console_output": false,
    "file_output": true,
    "max_file_size_mb": 100
  }
}
```

### Testing Configuration

```json
{
  "llm": {
    "provider": "mock",
    "timeout": 10,
    "max_retries": 1
  },
  "agents": {
    "generator": {"enabled": true, "max_retries": 1, "timeout": 30},
    "debugger": {"enabled": true, "max_retries": 1, "timeout": 30},
    "optimizer": {"enabled": false},
    "evaluator": {"enabled": false}
  },
  "tools": {
    "enabled": false
  },
  "plugins": {
    "enabled": false
  },
  "session": {
    "auto_save": false
  },
  "storage": {
    "sessions_path": "./test_sessions",
    "memories_path": "./test_memories",
    "knowledge_path": "./test_knowledge"
  },
  "verbose": {
    "enabled": false
  },
  "logging": {
    "level": "ERROR",
    "console_output": false,
    "file_output": false
  }
}
```

### Minimal Configuration

```json
{
  "llm": {
    "provider": "openai",
    "api_key": "your-api-key"
  },
  "agents": {
    "generator": {"enabled": true}
  }
}
```

### High-Performance Configuration

```json
{
  "llm": {
    "provider": "custom",
    "base_url": "http://gpu-llm-cluster:8001",
    "model_name": "high-performance-model",
    "timeout": 300,
    "max_retries": 5
  },
  "agents": {
    "generator": {"enabled": true, "max_retries": 5, "timeout": 600},
    "debugger": {"enabled": true, "max_retries": 3, "timeout": 300},
    "optimizer": {"enabled": true, "max_retries": 5, "timeout": 900},
    "evaluator": {"enabled": true, "max_retries": 3, "timeout": 600}
  },
  "tools": {
    "enabled": true,
    "max_concurrent": 10,
    "debug_tools": {
      "cuda_compile": {"enabled": true, "timeout": 120},
      "cuda_memcheck": {"enabled": true, "timeout": 300},
      "cuda_syntax_check": {"enabled": true, "timeout": 60}
    },
    "eval_tools": {
      "cuda_profile": {"enabled": true, "timeout": 600},
      "cuda_occupancy": {"enabled": true, "timeout": 60},
      "cuda_performance_analyze": {"enabled": true, "timeout": 180}
    }
  },
  "workflow": {
    "execution_settings": {
      "max_parallel_tasks": 5,
      "task_timeout": 1800
    }
  },
  "session": {
    "auto_save": true,
    "save_interval_seconds": 120,
    "max_session_size_mb": 500,
    "compression": {"enabled": true, "level": 9}
  }
}
```

## Validation and Troubleshooting

### Configuration Validation

#### Using the Validation Script

```bash
# Validate current configuration
python scripts/validate_config.py

# Validate specific configuration file
python scripts/validate_config.py config/production.json

# Validate with environment variables
PINOCCHIO_ENV=production python scripts/validate_config.py
```

#### Manual Validation

```python
from pinocchio.config import ConfigManager
from pinocchio.config.models import PinocchioConfig

try:
    # Load configuration
    config_manager = ConfigManager("pinocchio.json")

    # Validate with Pydantic
    validated_config = PinocchioConfig(**config_manager.config)

    print("✅ Configuration is valid!")

except Exception as e:
    print(f"❌ Configuration validation failed: {e}")
```

### Common Configuration Issues

#### Issue: Missing Required Fields

```json
// ❌ Invalid - missing required LLM provider
{
  "llm": {
    "model_name": "gpt-4"
  }
}

// ✅ Valid
{
  "llm": {
    "provider": "openai",
    "model_name": "gpt-4"
  }
}
```

#### Issue: Invalid Data Types

```json
// ❌ Invalid - timeout should be integer
{
  "llm": {
    "timeout": "120"
  }
}

// ✅ Valid
{
  "llm": {
    "timeout": 120
  }
}
```

#### Issue: Unknown Configuration Keys

```json
// ❌ Invalid - unknown key "unknown_setting"
{
  "llm": {
    "provider": "openai",
    "unknown_setting": "value"
  }
}

// ✅ Valid - use custom_settings for unknown keys
{
  "llm": {
    "provider": "openai",
    "headers": {
      "custom_header": "value"
    }
  }
}
```

### Configuration Debugging

#### Debug Configuration Loading

```python
import os
import json
from pinocchio.config import ConfigManager

# Enable debug mode
os.environ['CONFIG_DEBUG'] = '1'

# Load configuration with debugging
config_manager = ConfigManager("pinocchio.json")

# Print loaded configuration
print("Loaded configuration:")
print(json.dumps(config_manager.config, indent=2))

# Print effective configuration (with overrides)
print("Effective configuration:")
print(json.dumps(config_manager.get_effective_config(), indent=2))
```

#### Check Environment Variable Overrides

```bash
# List all Pinocchio-related environment variables
env | grep -i pinocchio
env | grep -i llm
env | grep -i debug

# Check specific variables
echo "LLM_PROVIDER: $LLM_PROVIDER"
echo "DEBUG_LEVEL: $DEBUG_LEVEL"
echo "PINOCCHIO_ENV: $PINOCCHIO_ENV"
```

### Configuration Schema Reference

The complete configuration schema is defined using Pydantic models in `pinocchio/config/models.py`. You can generate a JSON schema for validation:

```python
from pinocchio.config.models import PinocchioConfig
import json

# Generate JSON schema
schema = PinocchioConfig.schema()
print(json.dumps(schema, indent=2))
```

This configuration guide provides comprehensive documentation for all configuration options and usage patterns in the Pinocchio system, enabling users to customize the system for their specific needs and environments.
