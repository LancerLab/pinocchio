# Pinocchio Verbose Configuration System

## Overview

Pinocchio now supports a comprehensive verbose logging system with multiple configuration modes, making it easy for developers to switch between development, production, and debug environments.

## Configuration Modes

### 1. Development Mode (`configs/development.json`)
**Best for:** Active development and debugging
- ✅ Full verbose logging enabled
- ✅ Performance tracking
- ✅ Session tracking
- ✅ Export on exit
- ✅ Detailed agent communications
- ✅ LLM request/response logging

### 2. Production Mode (`configs/production.json`)
**Best for:** End users and clean experience
- ❌ Verbose logging disabled
- ❌ Performance tracking disabled
- ❌ Session tracking disabled
- ✅ Basic progress updates only
- ✅ Clean user experience

### 3. Debug Mode (`configs/debug.json`)
**Best for:** Troubleshooting and maximum detail
- ✅ Maximum verbose logging
- ✅ Raw prompt/response logging
- ✅ Internal state logging
- ✅ Memory operations logging
- ✅ Configuration change logging
- ✅ Maximum recursion depth (10)

## Quick Start

### Using the Mode Switch Script

```bash
# List available modes
python scripts/switch_mode.py --list

# Show current mode
python scripts/switch_mode.py --current

# Switch to development mode
python scripts/switch_mode.py development

# Switch to production mode
python scripts/switch_mode.py production

# Switch to debug mode
python scripts/switch_mode.py debug
```

### Using CLI with Custom Config

```bash
# Use development config
python -m pinocchio.cli.main --config configs/development.json

# Use production config
python -m pinocchio.cli.main --config configs/production.json

# Use debug config
python -m pinocchio.cli.main --config configs/debug.json
```

## Configuration Structure

The verbose configuration is defined in the `verbose` section of your config file:

```json
{
  "verbose": {
    "enabled": true,
    "mode": "development",
    "level": "detailed",
    "log_file": "./logs/verbose.log",
    "max_depth": 5,
    "enable_colors": true,
    "show_agent_instructions": true,
    "show_execution_times": true,
    "show_task_details": true,
    "show_progress_updates": true,
    "performance_tracking": true,
    "session_tracking": true,
    "export_on_exit": false,
    "log_llm_requests": true,
    "log_llm_responses": true,
    "log_agent_communications": true,
    "log_coordinator_activities": true,
    "log_performance_metrics": true,
    "log_error_details": true,
    "log_session_summary": true
  }
}
```

## Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `enabled` | boolean | false | Enable/disable verbose logging |
| `mode` | string | "production" | Mode: "development", "production", "debug" |
| `level` | string | "minimal" | Level: "minimal", "detailed", "maximum" |
| `log_file` | string | "./logs/verbose.log" | Path to verbose log file |
| `max_depth` | integer | 5 | Maximum recursion depth for nested data |
| `enable_colors` | boolean | true | Enable colored output |
| `show_agent_instructions` | boolean | false | Show detailed agent instructions |
| `show_execution_times` | boolean | false | Show execution time information |
| `show_task_details` | boolean | false | Show detailed task information |
| `show_progress_updates` | boolean | true | Show progress updates |
| `performance_tracking` | boolean | false | Track performance metrics |
| `session_tracking` | boolean | false | Track session information |
| `export_on_exit` | boolean | false | Export logs on exit |
| `log_llm_requests` | boolean | false | Log LLM request details |
| `log_llm_responses` | boolean | false | Log LLM response details |
| `log_agent_communications` | boolean | false | Log agent communication |
| `log_coordinator_activities` | boolean | false | Log coordinator activities |
| `log_performance_metrics` | boolean | false | Log performance metrics |
| `log_error_details` | boolean | false | Log detailed error information |
| `log_session_summary` | boolean | false | Log session summaries |

## CLI Commands

When verbose logging is enabled, you can use these commands in the CLI:

- `/verbose` - Toggle verbose logging mode
- `/performance` - Show performance metrics
- `/logs` - Export verbose logs
- `/session` - Show session summary

## Environment Variables

You can also set the configuration file path using environment variables:

```bash
export PINOCCHIO_CONFIG_FILE=configs/development.json
python -m pinocchio.cli.main
```

## Log Files

Verbose logs are saved to:
- `./logs/verbose.log` - Main verbose log file
- `./logs/verbose_dev.log` - Development mode logs
- `./logs/verbose_prod.log` - Production mode logs
- `./logs/verbose_debug.log` - Debug mode logs

## Performance Impact

- **Development Mode**: Moderate performance impact, detailed logging
- **Production Mode**: Minimal performance impact, clean output
- **Debug Mode**: Higher performance impact, maximum detail

## Best Practices

1. **Development**: Use development mode for active development
2. **Testing**: Use debug mode for troubleshooting issues
3. **Production**: Use production mode for end users
4. **Custom Configs**: Create your own config files for specific needs

## Creating Custom Configurations

You can create custom configuration files by copying one of the existing modes and modifying the settings:

```bash
cp configs/development.json configs/my_custom.json
# Edit my_custom.json with your preferred settings
python -m pinocchio.cli.main --config configs/my_custom.json
```

## Troubleshooting

### Log File Issues
- Ensure the `logs` directory exists
- Check file permissions
- Verify disk space

### Configuration Issues
- Validate JSON syntax
- Check file paths
- Ensure all required fields are present

### Performance Issues
- Reduce `max_depth` for large data structures
- Disable unnecessary logging options
- Use production mode for better performance
