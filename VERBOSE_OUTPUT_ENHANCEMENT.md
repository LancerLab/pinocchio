# Pinocchio Verbose Output Enhancement

## Overview

This document describes the enhanced verbose output system in Pinocchio's task planning and execution workflow. The improvements provide detailed visibility into agent participation, task instructions, and execution progress.

## Key Features

### 1. Detailed Agent Instructions

Each task now includes specific, detailed instructions for the assigned agent:

#### Generator Agent Instructions
```
Generate high-performance Choreo DSL operator code based on the user request.

Key Requirements:
- primary_goal: main objective
- secondary_goals: [goal1, goal2]

Focus on:
- Performance optimization (loop tiling, vectorization, memory coalescing)
- Memory efficiency and access patterns
- Correctness and safety with proper error checking
- Code readability and maintainability
- Following Choreo DSL syntax and conventions
```

#### Optimizer Agent Instructions
```
Analyze and optimize the generated Choreo DSL code for better performance.

Optimization Goals:
- performance
- memory_efficiency
- scalability

Optimization Focus:
- Identify performance bottlenecks
- Apply advanced optimization techniques
- Maintain code correctness
- Provide detailed optimization explanations
- Suggest hyperparameter tuning
```

#### Debugger Agent Instructions
```
Analyze the generated code for potential issues, errors, and improvements.

Debugging Focus:
- Syntax errors and compatibility issues
- Logic errors and edge cases
- Performance bottlenecks
- Memory access patterns
- Error handling and validation

Provide:
- Detailed analysis of issues found
- Specific fixes with explanations
- Improved code version
- Recommendations for robustness
```

#### Evaluator Agent Instructions
```
Evaluate the generated code for performance, correctness, and quality.

Evaluation Criteria:
- performance
- memory_efficiency

Evaluation Focus:
- Code quality and maintainability
- Performance characteristics
- Memory usage patterns
- Correctness and safety
- Optimization effectiveness
- Scalability considerations
```

### 2. Visual Indicators

#### Agent Type Emojis
- ⚡ **Generator**: Code generation
- 🚀 **Optimizer**: Performance optimization
- 🔧 **Debugger**: Error analysis and fixing
- 📊 **Evaluator**: Performance evaluation

#### Priority Indicators
- 🔴 **Critical**: Highest priority tasks
- 🟡 **High**: Important tasks
- 🟢 **Medium**: Standard tasks
- 🔵 **Low**: Optional tasks

### 3. Enhanced Task Plan Overview

The system now displays a comprehensive task plan overview:

```
📋 Task Plan Overview:
  1. ⚡ GENERATOR (🔴 critical)
     📝 Generate high-performance matrix multiplication kernel
     💡 Instruction: Generate high-performance Choreo DSL operator code...
     🔗 Dependencies: -
```

### 4. Real-time Execution Progress

During execution, users see detailed progress information:

```
🔄 Executing ⚡ GENERATOR (Task task_1)
   📋 Description: Generate high-performance matrix multiplication kernel
   💡 Detailed Instruction:
      Generate high-performance Choreo DSL operator code based on the user request.
      Key Requirements:
      Focus on:
      - Performance optimization (loop tiling, vectorization, memory coalescing)
      - Memory efficiency and access patterns
      - Correctness and safety with proper error checking
      - Code readability and maintainability
      - Following Choreo DSL syntax and conventions

✅ ⚡ GENERATOR completed successfully
   📊 Generated 37 lines of code
   ⏱️ Execution time: 26336ms
```

### 5. Agent Participation Summary

At the end of execution, a comprehensive summary is provided:

```
🤖 Agent Participation Summary:
   ⚡ GENERATOR: 1/1 (100.0% success)
   🚀 OPTIMIZER: 1/1 (100.0% success)
   🔧 DEBUGGER: 0/0 (0.0% success)
   📊 EVALUATOR: 0/0 (0.0% success)
```

## Implementation Details

### Task Planning Enhancements

#### TaskPlanner._build_*_instruction() Methods

Each agent type has a dedicated instruction builder:

```python
def _build_generator_instruction(self, context: TaskPlanningContext) -> str:
    """Build detailed instruction for generator agent."""
    instruction_parts = [
        "Generate high-performance Choreo DSL operator code based on the user request.",
        "",
        "Key Requirements:",
    ]
    # ... build comprehensive instruction
    return "\n".join(instruction_parts)
```

#### Task Data Model Updates

Tasks now include detailed instructions in their input_data:

```python
generator_task = Task(
    task_id=f"task_{task_counter}",
    agent_type=AgentType.GENERATOR,
    task_description=context.user_request,
    input_data={
        "user_request": context.user_request,
        "instruction": generator_instruction,
    },
)
```

### Task Execution Enhancements

#### Refactored TaskExecutor

The execute_plan method was refactored into smaller, focused methods:

- `_display_task_plan_overview()`: Shows task plan details
- `_execute_tasks()`: Manages task execution loop
- `_execute_single_task_with_verbose()`: Handles individual task execution
- `_finalize_plan()`: Provides final summary
- `_calculate_agent_stats()`: Computes participation statistics

#### Agent Request Preparation

Enhanced request preparation includes detailed instructions:

```python
def _prepare_agent_request(self, task: Task, previous_results: Dict[str, Any]) -> Dict[str, Any]:
    request = {
        "request_id": f"{task.task_id}_{task.created_at.timestamp()}",
        "task_description": task.task_description,
        "detailed_instruction": task.input_data.get("instruction", ""),
        # ... other fields
    }
    return request
```

### Agent Prompt Building Updates

All agents now support detailed instructions:

```python
def _build_generation_prompt(self, request: Dict[str, Any]) -> str:
    detailed_instruction = request.get("detailed_instruction", "")

    if detailed_instruction:
        prompt_parts.extend([
            "",
            "Detailed Instructions:",
            detailed_instruction
        ])
    else:
        # Fallback to basic requirements
        # ...
```

## Usage Examples

### Basic Code Generation

```python
# User request
user_request = "Generate a matrix multiplication kernel"

# Enhanced verbose output shows:
# - Task plan with detailed instructions
# - Real-time execution progress
# - Agent participation summary
```

### Complex Multi-Agent Workflow

```python
# User request
user_request = "Create an optimized convolution kernel with error handling and performance evaluation"

# System creates tasks for:
# 1. Generator: Create initial code
# 2. Optimizer: Optimize performance
# 3. Debugger: Check for errors
# 4. Evaluator: Assess performance
```

## Benefits

1. **Improved Visibility**: Users can see exactly what each agent is doing
2. **Better Debugging**: Detailed instructions help identify issues
3. **Performance Monitoring**: Execution times and success rates are tracked
4. **Visual Clarity**: Emojis and colors make output more readable
5. **Comprehensive Tracking**: Full workflow visibility from planning to completion

## Backward Compatibility

The system maintains backward compatibility:
- Agents can work with or without detailed instructions
- Fallback to basic requirements when detailed instructions are not available
- Existing APIs remain unchanged

## Future Enhancements

1. **Parallel Execution**: Support for concurrent task execution
2. **Interactive Mode**: Real-time user interaction during execution
3. **Custom Instructions**: User-defined instruction templates
4. **Performance Analytics**: Advanced metrics and analysis
5. **Visual Workflows**: Graphical representation of task dependencies
