"""Generator agent for code generation."""

import logging
from typing import Any, Dict

from pinocchio.config import ConfigManager
from pinocchio.llm.custom_llm_client import CustomLLMClient

from ..data_models.agent import AgentResponse
from ..utils.json_parser import extract_code_from_response
from .base import AgentWithRetry

logger = logging.getLogger(__name__)


class GeneratorAgent(AgentWithRetry):
    """Agent responsible for generating Choreo DSL operator code."""

    def __init__(self, llm_client: Any = None, max_retries: int = 3):
        """
        Initialize Generator agent.

        Args:
            llm_client: LLM client instance (optional)
            max_retries: Maximum retry attempts for LLM calls
        """
        if llm_client is None:
            config_manager = ConfigManager()
            # Try agent-specific config first
            agent_llm_config = config_manager.get_agent_llm_config("generator")
            verbose = config_manager.get("verbose.enabled", True)
            llm_client = CustomLLMClient(agent_llm_config, verbose=verbose)
        super().__init__("generator", llm_client, max_retries)
        logger.info("GeneratorAgent initialized with its own LLM client")

    async def execute(self, request: Dict[str, Any]) -> AgentResponse:
        """
        Execute code generation task.

        Args:
            request: Generation request containing task description

        Returns:
            Agent response with generated code
        """
        request_id = request.get("request_id", "unknown")

        try:
            # Build prompt for code generation
            prompt = self._build_generation_prompt(request)

            # Call LLM with retry
            llm_response = await self._call_llm_with_retry(prompt)

            # Extract and process the response
            output = self._process_generation_response(llm_response, request)

            # Create successful response
            return self._create_response(
                request_id=request_id,
                success=True,
                output=output,
                processing_time_ms=int(self.get_average_processing_time()),
            )

        except Exception as e:
            return self._handle_error(request_id, e)

    def _build_generation_prompt(self, request: Dict[str, Any]) -> str:
        """
        Build specific prompt for code generation.

        Args:
            request: Generation request

        Returns:
            Formatted prompt string
        """
        task_description = request.get("task_description", "")
        context = request.get("context", {})
        requirements = request.get("requirements", {})
        optimization_goals = request.get("optimization_goals", [])
        detailed_instruction = request.get("detailed_instruction", "")

        prompt_parts = [
            "You are a Generator agent in the Pinocchio multi-agent system.",
            "Your primary task is to generate high-performance Choreo DSL operator code.",
            "",
            f"Task Description: {task_description}",
        ]

        # Add detailed instruction if available
        if detailed_instruction:
            prompt_parts.extend(["", "Detailed Instructions:", detailed_instruction])
        else:
            # Fallback to basic requirements
            if requirements:
                prompt_parts.extend(
                    ["", "Requirements:", self._format_requirements(requirements)]
                )

            if optimization_goals:
                prompt_parts.extend(
                    ["", "Optimization Goals:", "- " + "\n- ".join(optimization_goals)]
                )

        if context:
            prompt_parts.extend(["", "Context:", str(context)])

        prompt_parts.extend(
            [
                "",
                self._get_agent_instructions(),
                "",
                self._get_generation_output_format(),
            ]
        )

        return "\n".join(prompt_parts)

    def _get_agent_instructions(self) -> str:
        """Get Generator-specific instructions."""
        return """
Instructions for Code Generation:
1. Generate efficient Choreo DSL operator code that meets the specified requirements
2. Apply appropriate optimization techniques (loop tiling, vectorization, memory coalescing)
3. Include proper error checking and bounds validation
4. Use meaningful variable names and add comments where necessary
5. Consider memory access patterns for optimal performance
6. Ensure the code follows Choreo DSL syntax and conventions

Focus on:
- Performance optimization
- Memory efficiency
- Correctness and safety
- Code readability and maintainability
"""

    def _get_generation_output_format(self) -> str:
        """Get output format for generation response."""
        return """
Please provide your response in JSON format with the following structure:
{
    "agent_type": "generator",
    "success": true,
    "output": {
        "code": "// Generated Choreo DSL code here",
        "language": "choreo_dsl",
        "explanation": "Brief explanation of the generated code",
        "optimization_techniques": ["technique1", "technique2"],
        "hyperparameters": {
            "param1": "value1",
            "param2": "value2"
        },
        "performance_notes": "Notes about expected performance characteristics",
        "dependencies": ["dependency1", "dependency2"],
        "complexity": "O(n) time, O(1) space"
    },
    "error_message": null
}
"""

    def _format_requirements(self, requirements: Dict[str, Any]) -> str:
        """Format requirements for prompt."""
        formatted = []
        for key, value in requirements.items():
            formatted.append(f"- {key}: {value}")
        return "\n".join(formatted)

    def _process_generation_response(
        self, llm_response: Dict[str, Any], request: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process LLM response and extract generation results.

        Args:
            llm_response: Raw LLM response
            request: Original request

        Returns:
            Processed output dictionary
        """
        # Extract output from LLM response
        if llm_response.get("success") and "output" in llm_response:
            output = llm_response["output"]
        else:
            # Fallback: try to extract code from response content
            code = extract_code_from_response(llm_response)
            output = {
                "code": code or "// No code generated",
                "language": "choreo_dsl",
                "explanation": "Code generated using fallback extraction",
                "optimization_techniques": [],
                "hyperparameters": {},
                "performance_notes": "Performance characteristics not analyzed",
                "dependencies": [],
                "complexity": "Not analyzed",
            }

        # Ensure required fields are present
        output.setdefault("code", "// No code generated")
        output.setdefault("language", "choreo_dsl")
        output.setdefault("explanation", "No explanation provided")
        output.setdefault("optimization_techniques", [])
        output.setdefault("hyperparameters", {})
        output.setdefault("performance_notes", "")
        output.setdefault("dependencies", [])
        output.setdefault("complexity", "Not specified")

        # Add generation metadata
        output["generation_metadata"] = {
            "agent_type": "generator",
            "request_id": request.get("request_id", "unknown"),
            "task_description": request.get("task_description", ""),
            "generated_at": request.get("timestamp", ""),
            "call_count": self.call_count,
        }

        return output

    def generate_simple_code(self, task_description: str) -> Dict[str, Any]:
        """
        Generate simple code synchronously (for testing).

        Args:
            task_description: Description of the task

        Returns:
            Simple generated code structure
        """
        # Extract operation type from description
        operation = "generic_op"
        if (
            "matmul" in task_description.lower()
            or "matrix" in task_description.lower()
            or "矩阵" in task_description
        ):
            operation = "matrix_multiplication"
        elif "conv" in task_description.lower():
            operation = "convolution"
        elif "add" in task_description.lower() or "加法" in task_description:
            operation = "addition"

        # Generate simple template code with the operation name in the code
        code_template = f"""
// Generated Choreo DSL operator for {operation}
func {operation}_kernel(input: tensor, output: tensor) {{
    // Basic implementation for {operation}
    for i in range(input.shape[0]) {{
        for j in range(input.shape[1]) {{
            output[i][j] = compute_{operation}(input[i][j]);
        }}
    }}
}}
""".strip()

        return {
            "code": code_template,
            "language": "choreo_dsl",
            "explanation": f"Generated basic {operation} operator with simple loop structure",
            "optimization_techniques": ["basic_loops"],
            "hyperparameters": {"block_size": 32},
            "performance_notes": "Basic implementation, can be optimized further",
            "dependencies": [],
            "complexity": "O(n²) time, O(1) space",
        }
