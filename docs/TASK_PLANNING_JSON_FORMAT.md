# TaskPlanner JSON Format Requirements

## Overview

TaskPlanner uses LLM analysis to create `TaskPlanningContext` objects, which are then used to generate task plans. The LLM must return a specific JSON format that can be parsed into the required fields.

## JSON Format Requirements

### New Format (Recommended)

```json
{
    "agent_type": "planner",
    "success": true,
    "output": {
        "requirements": {
            "primary_goal": "main objective of the request",
            "secondary_goals": ["list", "of", "secondary", "goals"],
            "code_requirements": ["efficient_data_structures", "performance_optimization", "memory_optimization"]
        },
        "optimization_goals": ["performance", "memory_efficiency", "scalability"],
        "constraints": ["simplicity", "safety", "compatibility"],
        "user_preferences": {
            "complexity_level": "simple|moderate|advanced",
            "optimization_aggressiveness": "conservative|standard|aggressive"
        },
        "planning_strategy": "conservative|standard|aggressive"
    },
    "explanation": "Brief explanation of the analysis approach",
    "confidence": 0.95
}
```

### Old Format (Compatible)

```json
{
    "requirements": {
        "primary_goal": "main objective of the request",
        "secondary_goals": ["list", "of", "secondary", "goals"],
        "code_requirements": ["efficient_data_structures", "performance_optimization", "memory_optimization"]
    },
    "optimization_goals": ["performance", "memory_efficiency", "scalability"],
    "constraints": ["simplicity", "safety", "compatibility"],
    "user_preferences": {
        "complexity_level": "simple|moderate|advanced",
        "optimization_aggressiveness": "conservative|standard|aggressive"
    },
    "planning_strategy": "conservative|standard|aggressive"
}
```

## Field Descriptions

### requirements
- **primary_goal**: The main objective extracted from the user request
- **secondary_goals**: List of additional goals or requirements
- **code_requirements**: List of technical requirements (e.g., ["efficient_data_structures", "performance_optimization"])

### optimization_goals
List of optimization targets (e.g., ["performance", "memory_efficiency", "scalability"])

### constraints
List of limitations or constraints (e.g., ["simplicity", "safety", "compatibility"])

### user_preferences
- **complexity_level**: One of "simple", "moderate", or "advanced"
- **optimization_aggressiveness**: One of "conservative", "standard", or "aggressive"

### planning_strategy
One of "conservative", "standard", or "aggressive"

## Critical Requirements

1. **JSON Only**: The response must be ONLY valid JSON format
2. **No Extra Text**: Do NOT include any text before or after the JSON
3. **No Markdown**: Do NOT include markdown formatting, code blocks, or any other text
4. **Valid JSON**: The response must be parseable as valid JSON

## Implementation Details

### Prompt Engineering

The TaskPlanner uses a carefully crafted prompt that:
- Explicitly requires JSON-only responses
- Provides a complete JSON template
- Includes field descriptions and examples
- Emphasizes the critical requirements multiple times

### Parsing Logic

The `_parse_analysis_response` method:
1. Attempts to parse the response as JSON
2. Checks for both new format (with "output" field) and old format
3. Validates required fields are present
4. Falls back to basic analysis if parsing fails

### Fallback Behavior

If LLM analysis fails:
- **Development mode**: Raises an error
- **Production mode**: Uses fallback analysis with basic keyword extraction

## Example Usage

```python
# Create TaskPlanner
planner = TaskPlanner(llm_client=llm_client, mode="development")

# Create task plan
plan = await planner.create_task_plan("Generate a high-performance CUDA matrix multiplication kernel")

# The plan will contain tasks based on the analyzed requirements
for task in plan.tasks:
    print(f"{task.agent_type}: {task.task_description}")
```

## Testing

The JSON format has been thoroughly tested with:
- Mock LLM responses
- Real LLM response simulations
- Prompt format validation
- Parsing logic verification

All tests confirm that the current implementation correctly handles the expected JSON format.
