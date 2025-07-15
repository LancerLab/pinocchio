# Pinocchio CLI Mode Feature Implementation

## Overview

The mode feature has been successfully implemented to control fallback behavior in the Pinocchio CLI system. This feature allows users to choose between development and production modes, which control how the system handles errors and failures.

## Implementation Details

### 1. Configuration Model Updates

**File**: `pinocchio/config/models.py`

- Added `CLIConfig` model with `mode` field
- Integrated `CLIConfig` into `PinocchioConfig`
- Default mode is "production"

```python
class CLIConfig(BaseModel):
    """CLI configuration."""
    mode: str = Field(
        default="production",
        description="CLI mode (development/production). Development mode raises errors, production mode allows fallback"
    )
```

### 2. CLI Argument Parsing

**File**: `pinocchio/cli/main.py`

- Added `--mode` argument with choices: `["development", "production"]`
- Updated help text to reflect mode behavior
- Mode parameter takes precedence over config file setting

### 3. Coordinator Integration

**File**: `pinocchio/coordinator.py`

- Modified `Coordinator.__init__()` to accept `mode` parameter
- Passes mode to `TaskPlanner`
- Stores mode for fallback behavior control

### 4. TaskPlanner Mode Control

**File**: `pinocchio/task_planning/task_planner.py`

- Modified `TaskPlanner.__init__()` to accept `mode` parameter
- Updated `_analyze_request()` method to control fallback behavior:
  - **Development mode**: Raises errors on LLM failures
  - **Production mode**: Allows fallback on LLM failures

## Mode Behavior

### Development Mode (`--mode development`)

- **LLM Client Missing**: Raises `RuntimeError`
- **LLM Analysis Failure**: Raises the original exception
- **Task Planning Failure**: Propagates errors to user
- **Use Case**: Development and debugging

### Production Mode (`--mode production`)

- **LLM Client Missing**: Uses fallback analysis
- **LLM Analysis Failure**: Uses fallback analysis
- **Task Planning Failure**: Creates minimal fallback plan
- **Use Case**: End-user deployment

## Usage Examples

### Basic Usage

```bash
# Development mode - raises errors on failures
pinocchio --mode development

# Production mode - allows fallback on failures
pinocchio --mode production
```

### Combined with Other Options

```bash
# Development mode with workflow strategy
pinocchio --mode development --strategy workflow

# Production mode with planning strategy
pinocchio --mode production --strategy planning

# Development mode with dry-run for testing
pinocchio --mode development --dry-run
```

### Configuration File

You can also set the mode in your `pinocchio.json` configuration file:

```json
{
  "cli": {
    "mode": "development"
  },
  "llm": {
    "provider": "custom",
    "base_url": "http://localhost:8001",
    "model_name": "default"
  }
}
```

## Testing Results

All integration tests passed successfully:

- ✅ **Development mode**: raises errors on LLM failures
- ✅ **Production mode**: allows fallback on LLM failures
- ✅ **Mode properly passed through system**: from CLI to TaskPlanner
- ✅ **Configuration loading**: works with both CLI args and config file
- ✅ **Coordinator integration**: properly handles mode parameter

## Error Handling

### Development Mode Errors

When in development mode, the system will raise specific errors:

1. **No LLM Client**: `RuntimeError("No LLM client available for task planning in development mode")`
2. **LLM Analysis Failure**: Original exception from LLM client
3. **Task Planning Failure**: Propagated through Coordinator

### Production Mode Fallbacks

When in production mode, the system will:

1. **No LLM Client**: Use fallback analysis with basic requirements extraction
2. **LLM Analysis Failure**: Use fallback analysis
3. **Task Planning Failure**: Create minimal fallback plan with single generator task

## Verbose Logging

The system provides detailed verbose logging for both modes:

- **Development mode**: Full error details and stack traces
- **Production mode**: Fallback decisions and minimal error messages

## Benefits

1. **Development Efficiency**: Developers can quickly identify and fix issues
2. **Production Reliability**: End users get graceful degradation
3. **Flexible Configuration**: Mode can be set via CLI or config file
4. **Clear Error Messages**: Different error handling for different use cases

## Future Enhancements

Potential improvements for the mode feature:

1. **Debug Mode**: Add a third mode for maximum debugging information
2. **Custom Fallback Strategies**: Allow custom fallback behavior configuration
3. **Mode-Specific Logging**: Different logging levels for different modes
4. **Mode Validation**: Validate mode settings in configuration files
