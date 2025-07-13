# Agent-Specific LLM Configuration

## Overview

Pinocchio now supports agent-specific LLM configurations, allowing each agent (Generator, Optimizer, Debugger, Evaluator) to use different LLM models and configurations. This feature enables:

- **Model Specialization**: Different agents can use models optimized for their specific tasks
- **Resource Optimization**: Use smaller models for simpler tasks, larger models for complex ones
- **Flexibility**: Mix and match different LLM providers and configurations
- **Fallback Support**: Agents automatically fallback to global LLM config if agent-specific config is not provided

## Configuration Structure

### Global LLM Configuration
```json
{
  "llm": {
    "provider": "custom",
    "base_url": "http://10.0.16.46:8001",
    "model_name": "Qwen/Qwen3-32B",
    "timeout": 120,
    "max_retries": 3
  }
}
```

### Agent-Specific LLM Configuration
```json
{
  "llm": {
    "provider": "custom",
    "base_url": "http://10.0.16.46:8001",
    "model_name": "Qwen/Qwen3-32B",
    "timeout": 120,
    "max_retries": 3
  },
  "llm_generator": {
    "provider": "custom",
    "base_url": "http://10.0.16.46:8001",
    "model_name": "Qwen/Qwen3-32B",
    "timeout": 120,
    "max_retries": 3
  },
  "llm_optimizer": {
    "provider": "custom",
    "base_url": "http://10.0.16.46:8001",
    "model_name": "Qwen/Qwen3-14B",
    "timeout": 120,
    "max_retries": 3
  },
  "llm_debugger": {
    "provider": "custom",
    "base_url": "http://10.0.16.46:8001",
    "model_name": "Qwen/Qwen3-7B",
    "timeout": 120,
    "max_retries": 3
  },
  "llm_evaluator": {
    "provider": "custom",
    "base_url": "http://10.0.16.46:8001",
    "model_name": "Qwen/Qwen3-32B",
    "timeout": 120,
    "max_retries": 3
  }
}
```

## Configuration Priority

1. **Agent-Specific Config**: If `llm_generator`, `llm_optimizer`, etc. are defined, agents use their specific config
2. **Global Config Fallback**: If agent-specific config is not provided, agents use the global `llm` config
3. **Default Config**: If no config is provided, system uses default configuration

## Supported Agent Types

- `llm_generator`: Generator agent configuration
- `llm_optimizer`: Optimizer agent configuration
- `llm_debugger`: Debugger agent configuration
- `llm_evaluator`: Evaluator agent configuration

## Usage Examples

### Example 1: Different Models for Different Agents
```json
{
  "llm_generator": {
    "provider": "custom",
    "base_url": "http://10.0.16.46:8001",
    "model_name": "Qwen/Qwen3-32B"
  },
  "llm_optimizer": {
    "provider": "custom",
    "base_url": "http://10.0.16.46:8001",
    "model_name": "Qwen/Qwen3-14B"
  },
  "llm_debugger": {
    "provider": "custom",
    "base_url": "http://10.0.16.46:8001",
    "model_name": "Qwen/Qwen3-7B"
  }
}
```

### Example 2: Different Endpoints for Different Agents
```json
{
  "llm_generator": {
    "provider": "custom",
    "base_url": "http://10.0.16.46:8001",
    "model_name": "Qwen/Qwen3-32B"
  },
  "llm_optimizer": {
    "provider": "custom",
    "base_url": "http://10.0.16.46:8002",
    "model_name": "Qwen/Qwen3-14B"
  }
}
```

### Example 3: Mixed Provider Configuration
```json
{
  "llm_generator": {
    "provider": "custom",
    "base_url": "http://10.0.16.46:8001",
    "model_name": "Qwen/Qwen3-32B"
  },
  "llm_optimizer": {
    "provider": "openai",
    "model_name": "gpt-4",
    "api_key": "your-openai-key"
  }
}
```

## Verbose Output

When verbose mode is enabled, the CLI will show which LLM configuration each agent is using:

```
[LLM VERBOSE] Selected LLM: Qwen/Qwen3-32B (http://10.0.16.46:8001)
[LLM VERBOSE] Selected LLM: Qwen/Qwen3-14B (http://10.0.16.46:8001)
[LLM VERBOSE] Selected LLM: Qwen/Qwen3-7B (http://10.0.16.46:8001)
[LLM VERBOSE] Selected LLM: Qwen/Qwen3-32B (http://10.0.16.46:8001)
```

## Implementation Details

### Agent Initialization
Each agent automatically creates its own LLM client instance during initialization:

```python
class GeneratorAgent(AgentWithRetry):
    def __init__(self, llm_client: Any = None, max_retries: int = 3):
        if llm_client is None:
            config_manager = ConfigManager()
            agent_llm_config = config_manager.get_agent_llm_config("generator")
            verbose = config_manager.get("verbose.enabled", True)
            llm_client = CustomLLMClient(agent_llm_config, verbose=verbose)
        super().__init__("generator", llm_client, max_retries)
```

### Configuration Manager
The `ConfigManager.get_agent_llm_config()` method handles the priority logic:

```python
def get_agent_llm_config(self, agent_type: str) -> LLMConfigEntry:
    # Try to get agent-specific config
    agent_llm_key = f"llm_{agent_type}"
    agent_config = getattr(self.config, agent_llm_key, None)

    if agent_config is not None:
        return agent_config

    # Fallback to global config
    return self.get_llm_config()
```

## Testing

Unit tests are available in `tests/unittests/agents/test_agent_specific_config.py` to verify:

- Agent-specific config priority
- Global config fallback
- Configuration validation
- Agent initialization with different configs

## Migration Guide

### From Global-Only Configuration
If you currently use only global LLM configuration, no changes are needed. The system will continue to work as before.

### To Agent-Specific Configuration
1. Add agent-specific configurations to your `pinocchio.json`
2. Test with verbose mode enabled to verify each agent uses the correct config
3. Monitor performance and adjust configurations as needed

## Best Practices

1. **Start Simple**: Begin with global configuration, then add agent-specific configs as needed
2. **Monitor Performance**: Use verbose mode to track which models are being used
3. **Resource Planning**: Consider computational requirements when choosing different models
4. **Testing**: Always test configurations in development before production deployment
5. **Documentation**: Document your configuration choices and reasoning
