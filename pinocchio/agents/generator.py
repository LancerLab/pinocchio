"""Generator agent for code generation."""

import logging
import time
from typing import Any, Dict

from pinocchio.config import ConfigManager
from pinocchio.llm.custom_llm_client import CustomLLMClient

from ..data_models.agent import AgentResponse
from ..utils.json_parser import extract_code_from_response, format_json_response
from ..utils.temp_utils import cleanup_temp_files, create_temp_file
from ..utils.verbose_logger import LogLevel, get_verbose_logger
from .base import AgentWithRetry

logger = logging.getLogger(__name__)


class GeneratorAgent(AgentWithRetry):
    """Agent responsible for generating Choreo DSL operator code."""

    def __init__(
        self, llm_client: Any = None, max_retries: int = 3, retry_delay: float = 1.0
    ):
        """
        Initialize Generator agent.

        Args:
            llm_client: LLM client instance (optional)
            max_retries: Maximum retry attempts for LLM calls
            retry_delay: Delay between retry attempts in seconds
        """
        if llm_client is None:
            config_manager = ConfigManager()
            # Try agent-specific config first
            agent_llm_config = config_manager.get_agent_llm_config("generator")
            verbose = config_manager.get("verbose.enabled", True)
            llm_client = CustomLLMClient(agent_llm_config, verbose=verbose)
        super().__init__("generator", llm_client, max_retries, retry_delay)
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

        # Log generator execution start with detailed input analysis
        self.verbose_logger.log_agent_activity(
            self.agent_type,
            "Generator execution started",
            data={
                "request_id": request_id,
                "request_keys": list(request.keys()),
                "request_size": len(str(request)),
                "task_description": request.get("task_description", ""),
                "has_context": "context" in request,
                "has_requirements": "requirements" in request,
                "has_optimization_goals": "optimization_goals" in request,
                "has_detailed_instruction": "detailed_instruction" in request,
                "context_size": len(str(request.get("context", {}))),
                "requirements_count": len(request.get("requirements", {})),
                "optimization_goals_count": len(request.get("optimization_goals", [])),
            },
            session_id=getattr(self, "session_id", None),
            step_id=request.get("step_id"),
        )

        try:
            # Build prompt for code generation with detailed logging
            self.verbose_logger.log_agent_activity(
                self.agent_type,
                "Building generation prompt",
                data={"request_id": request_id},
                session_id=getattr(self, "session_id", None),
                step_id=request.get("step_id"),
            )

            prompt = self._build_generation_prompt(request)

            # Log prompt details
            self.verbose_logger.log_agent_activity(
                self.agent_type,
                "Generation prompt built",
                data={
                    "request_id": request_id,
                    "prompt_length": len(prompt),
                    "prompt_preview": prompt[:300] + "..."
                    if len(prompt) > 300
                    else prompt,
                },
                session_id=getattr(self, "session_id", None),
                step_id=request.get("step_id"),
            )

            # Call LLM with retry and detailed logging
            self.verbose_logger.log_agent_activity(
                self.agent_type,
                "Calling LLM for code generation",
                data={
                    "request_id": request_id,
                    "prompt_length": len(prompt),
                    "max_retries": self.max_retries,
                    "retry_delay": self.retry_delay,
                },
                session_id=getattr(self, "session_id", None),
                step_id=request.get("step_id"),
            )

            llm_response = await self._call_llm_with_retry(prompt)

            # Log LLM response details
            self.verbose_logger.log_agent_activity(
                self.agent_type,
                "LLM response received for generation",
                data={
                    "request_id": request_id,
                    "llm_response_keys": list(llm_response.keys())
                    if isinstance(llm_response, dict)
                    else [],
                    "llm_response_type": type(llm_response).__name__,
                    "llm_response_size": len(str(llm_response)),
                    "llm_success": llm_response.get("success", False),
                },
                session_id=getattr(self, "session_id", None),
                step_id=request.get("step_id"),
            )

            # Extract and process the response with detailed logging
            self.verbose_logger.log_agent_activity(
                self.agent_type,
                "Processing generation response",
                data={"request_id": request_id},
                session_id=getattr(self, "session_id", None),
                step_id=request.get("step_id"),
            )

            output = self._process_generation_response(llm_response, request)

            # Log processed output details
            self.verbose_logger.log_agent_activity(
                self.agent_type,
                "Generation response processed",
                data={
                    "request_id": request_id,
                    "output_keys": list(output.keys())
                    if isinstance(output, dict)
                    else [],
                    "output_size": len(str(output)),
                    "has_code": "code" in output,
                    "has_explanation": "explanation" in output,
                    "has_optimization_techniques": "optimization_techniques" in output,
                    "code_length": len(output.get("code", "")),
                    "explanation_length": len(output.get("explanation", "")),
                    "optimization_techniques_count": len(
                        output.get("optimization_techniques", [])
                    ),
                },
                session_id=getattr(self, "session_id", None),
                step_id=request.get("step_id"),
            )

            # Create successful response with detailed logging
            self.verbose_logger.log_agent_activity(
                self.agent_type,
                "Creating generation response",
                data={
                    "request_id": request_id,
                    "output_keys": list(output.keys())
                    if isinstance(output, dict)
                    else [],
                    "processing_time_ms": int(self.get_average_processing_time()),
                },
                session_id=getattr(self, "session_id", None),
                step_id=request.get("step_id"),
            )

            response = self._create_response(
                request_id=request_id,
                success=True,
                output=output,
                processing_time_ms=int(self.get_average_processing_time()),
            )

            # Log generation completion
            self.verbose_logger.log_agent_activity(
                self.agent_type,
                "Generator execution completed",
                data={
                    "request_id": request_id,
                    "response_success": response.success,
                    "response_output_keys": list(response.output.keys())
                    if isinstance(response.output, dict)
                    else [],
                    "response_processing_time_ms": response.processing_time_ms,
                    "generated_code_length": len(response.output.get("code", "")),
                    "call_count": self.call_count,
                    "total_processing_time": self.total_processing_time,
                },
                session_id=getattr(self, "session_id", None),
                step_id=request.get("step_id"),
            )

            return response

        except Exception as e:
            # Log generation error with detailed error information
            self.verbose_logger.log(
                LogLevel.ERROR,
                f"agent:{self.agent_type}",
                "Generator execution failed",
                data={
                    "request_id": request_id,
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "error_details": getattr(e, "__dict__", {}),
                    "request_keys": list(request.keys()),
                    "call_count": self.call_count,
                },
                session_id=getattr(self, "session_id", None),
                step_id=request.get("step_id"),
            )

            return self._handle_error(request_id, e)

    def _build_generation_prompt(self, request: Dict[str, Any]) -> str:
        """
        Build specific prompt for code generation.

        Args:
            request: Generation request

        Returns:
            Formatted prompt string
        """
        # Log prompt building start with detailed input analysis
        self.verbose_logger.log_agent_activity(
            self.agent_type,
            "Building generation prompt from request",
            data={
                "request_keys": list(request.keys()),
                "request_size": len(str(request)),
                "has_task_description": "task_description" in request,
                "has_context": "context" in request,
                "has_requirements": "requirements" in request,
                "has_optimization_goals": "optimization_goals" in request,
                "has_detailed_instruction": "detailed_instruction" in request,
            },
            session_id=getattr(self, "session_id", None),
            step_id=request.get("step_id"),
        )

        task_description = request.get("task_description", "")
        context = request.get("context", {})
        requirements = request.get("requirements", {})
        optimization_goals = request.get("optimization_goals", [])
        detailed_instruction = request.get("detailed_instruction", "")

        # Log extracted components
        self.verbose_logger.log_agent_activity(
            self.agent_type,
            "Extracted generation components",
            data={
                "task_description_length": len(task_description),
                "context_size": len(str(context)),
                "requirements_count": len(requirements),
                "optimization_goals_count": len(optimization_goals),
                "detailed_instruction_length": len(detailed_instruction),
            },
            session_id=getattr(self, "session_id", None),
            step_id=request.get("step_id"),
        )

        prompt_parts = [
            "You are a Generator agent in the Pinocchio multi-agent system.",
            "Your primary task is to generate high-performance Choreo DSL operator code.",
            "",
            f"Task Description: {task_description}",
        ]

        # Add detailed instruction if available
        if detailed_instruction:
            prompt_parts.extend(["", "Detailed Instructions:", detailed_instruction])
            self.verbose_logger.log_agent_activity(
                self.agent_type,
                "Added detailed instructions to prompt",
                data={"detailed_instruction_length": len(detailed_instruction)},
                session_id=getattr(self, "session_id", None),
                step_id=request.get("step_id"),
            )
        else:
            # Fallback to basic requirements
            if requirements:
                formatted_requirements = self._format_requirements(requirements)
                prompt_parts.extend(["", "Requirements:", formatted_requirements])
                self.verbose_logger.log_agent_activity(
                    self.agent_type,
                    "Added requirements to prompt",
                    data={
                        "requirements_count": len(requirements),
                        "formatted_requirements_length": len(formatted_requirements),
                    },
                    session_id=getattr(self, "session_id", None),
                    step_id=request.get("step_id"),
                )

            if optimization_goals:
                prompt_parts.extend(
                    ["", "Optimization Goals:", "- " + "\n- ".join(optimization_goals)]
                )
                self.verbose_logger.log_agent_activity(
                    self.agent_type,
                    "Added optimization goals to prompt",
                    data={"optimization_goals_count": len(optimization_goals)},
                    session_id=getattr(self, "session_id", None),
                    step_id=request.get("step_id"),
                )

        if context:
            prompt_parts.extend(["", "Context:", str(context)])
            self.verbose_logger.log_agent_activity(
                self.agent_type,
                "Added context to prompt",
                data={"context_size": len(str(context))},
                session_id=getattr(self, "session_id", None),
                step_id=request.get("step_id"),
            )

        # Get agent instructions and output format
        instructions = self._get_agent_instructions()
        output_format = self._get_generation_output_format()

        # Log instructions and format details
        self.verbose_logger.log_agent_activity(
            self.agent_type,
            "Retrieved generation instructions and format",
            data={
                "instructions_length": len(instructions),
                "output_format_length": len(output_format),
            },
            session_id=getattr(self, "session_id", None),
            step_id=request.get("step_id"),
        )

        prompt_parts.extend(
            [
                "",
                instructions,
                "",
                output_format,
            ]
        )

        final_prompt = "\n".join(prompt_parts)

        # Log final prompt details
        self.verbose_logger.log_agent_activity(
            self.agent_type,
            "Generation prompt built successfully",
            data={
                "final_prompt_length": len(final_prompt),
                "prompt_parts_count": len(prompt_parts),
                "has_detailed_instruction": bool(detailed_instruction),
                "has_requirements": bool(requirements),
                "has_optimization_goals": bool(optimization_goals),
                "has_context": bool(context),
                "prompt_preview": final_prompt[:400] + "..."
                if len(final_prompt) > 400
                else final_prompt,
            },
            session_id=getattr(self, "session_id", None),
            step_id=request.get("step_id"),
        )

        return final_prompt

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
        # Log requirements formatting
        self.verbose_logger.log_agent_activity(
            self.agent_type,
            "Formatting requirements",
            data={
                "requirements_count": len(requirements),
                "requirements_keys": list(requirements.keys()),
            },
            session_id=getattr(self, "session_id", None),
        )

        formatted = []
        for key, value in requirements.items():
            formatted.append(f"- {key}: {value}")

        result = "\n".join(formatted)

        # Log formatting result
        self.verbose_logger.log_agent_activity(
            self.agent_type,
            "Requirements formatted",
            data={
                "formatted_length": len(result),
                "formatted_lines": len(formatted),
            },
            session_id=getattr(self, "session_id", None),
        )

        return result

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
        # Log response processing start
        self.verbose_logger.log_agent_activity(
            self.agent_type,
            "Processing generation response",
            data={
                "llm_response_keys": list(llm_response.keys())
                if isinstance(llm_response, dict)
                else [],
                "llm_response_type": type(llm_response).__name__,
                "llm_success": llm_response.get("success", False),
                "request_id": request.get("request_id", "unknown"),
            },
            session_id=getattr(self, "session_id", None),
            step_id=request.get("step_id"),
        )

        # Extract output from LLM response
        if llm_response.get("success") and "output" in llm_response:
            output = llm_response["output"]
            self.verbose_logger.log_agent_activity(
                self.agent_type,
                "Extracted output from successful LLM response",
                data={
                    "output_keys": list(output.keys())
                    if isinstance(output, dict)
                    else [],
                    "output_type": type(output).__name__,
                },
                session_id=getattr(self, "session_id", None),
                step_id=request.get("step_id"),
            )
        else:
            # Fallback: try to extract code from response content
            self.verbose_logger.log_agent_activity(
                self.agent_type,
                "LLM response not successful, using fallback extraction",
                data={
                    "llm_success": llm_response.get("success", False),
                    "has_output": "output" in llm_response,
                },
                session_id=getattr(self, "session_id", None),
                step_id=request.get("step_id"),
            )

            code = extract_code_from_response(llm_response)
            output = {
                "code": code or "// No code generated",
                "language": "choreo_dsl",
                "explanation": "Code generated using fallback extraction",
                "optimization_techniques": [],
                "hyperparameters": {},
                "performance_notes": "Performance characteristics not analyzed",
                "dependencies": [],
                "complexity": "Unknown",
            }

            self.verbose_logger.log_agent_activity(
                self.agent_type,
                "Fallback code extraction completed",
                data={
                    "extracted_code_length": len(code) if code else 0,
                    "fallback_output_keys": list(output.keys()),
                },
                session_id=getattr(self, "session_id", None),
                step_id=request.get("step_id"),
            )

        # Validate and enhance output
        if not isinstance(output, dict):
            output = {"code": str(output), "language": "choreo_dsl"}

        # Ensure required fields exist
        required_fields = ["code", "language", "explanation"]
        for field in required_fields:
            if field not in output:
                output[field] = "Not provided"
                self.verbose_logger.log_agent_activity(
                    self.agent_type,
                    f"Added missing required field: {field}",
                    data={"field": field, "default_value": "Not provided"},
                    session_id=getattr(self, "session_id", None),
                    step_id=request.get("step_id"),
                )

        # Log final processed output
        self.verbose_logger.log_agent_activity(
            self.agent_type,
            "Generation response processing completed",
            data={
                "output_keys": list(output.keys()),
                "output_size": len(str(output)),
                "has_code": "code" in output,
                "has_explanation": "explanation" in output,
                "has_optimization_techniques": "optimization_techniques" in output,
                "code_length": len(output.get("code", "")),
                "explanation_length": len(output.get("explanation", "")),
            },
            session_id=getattr(self, "session_id", None),
            step_id=request.get("step_id"),
        )

        return output

    def generate_simple_code(self, task_description: str) -> Dict[str, Any]:
        """
        Generate simple code for given task description.

        Args:
            task_description: Description of the code generation task

        Returns:
            Generated code and metadata
        """
        # Log simple code generation start
        self.verbose_logger.log_agent_activity(
            self.agent_type,
            "Simple code generation started",
            data={
                "task_description_length": len(task_description),
                "task_description_preview": task_description[:100] + "..."
                if len(task_description) > 100
                else task_description,
            },
            session_id=getattr(self, "session_id", None),
        )

        request = {
            "task_description": task_description,
            "request_id": f"simple_gen_{int(time.time())}",
        }

        # Create a simple response for synchronous use
        output = {
            "code": f"// Generated code for: {task_description}",
            "language": "choreo_dsl",
            "explanation": "Simple code generation result",
            "optimization_techniques": ["basic"],
            "hyperparameters": {},
            "performance_notes": "Basic performance characteristics",
            "dependencies": [],
            "complexity": "O(1)",
        }

        # Log simple code generation completion
        self.verbose_logger.log_agent_activity(
            self.agent_type,
            "Simple code generation completed",
            data={
                "output_keys": list(output.keys()),
                "code_length": len(output.get("code", "")),
                "explanation_length": len(output.get("explanation", "")),
            },
            session_id=getattr(self, "session_id", None),
        )

        return output
