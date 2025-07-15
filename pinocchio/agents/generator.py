"""Generator agent for code generation."""

import logging
import time
from typing import Any, Dict

from pinocchio.config import ConfigManager
from pinocchio.llm.custom_llm_client import CustomLLMClient
from pinocchio.utils.file_utils import get_operator_name_from_task, get_output_path

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
        # Ensure agent_type is always AgentType enum
        from ..data_models.task_planning import AgentType as PlanningAgentType

        if "agent_type" in request and isinstance(request["agent_type"], str):
            request["agent_type"] = PlanningAgentType[request["agent_type"].upper()]

        request_id = request.get("request_id", "unknown")

        # === Tracing: log request structure ===
        self.verbose_logger.log_agent_activity(
            self.agent_type,
            "[Tracing] Agent execute entry",
            data={
                "request_keys": list(request.keys()),
                "has_prompt": "prompt" in request,
                "task_description": request.get("task_description"),
                "context_keys": list(request.get("context", {}).keys()),
                "previous_results_keys": list(request.get("previous_results", {})),
            },
            session_id=getattr(self, "session_id", None),
            step_id=request.get("step_id"),
        )

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

            prompt = request.get("prompt")
            if prompt:
                self.verbose_logger.log_agent_activity(
                    self.agent_type,
                    "Using prompt from request['prompt'] (PromptManager)",
                    data={
                        "request_id": request_id,
                        "prompt_length": len(prompt),
                        "prompt_preview": prompt[:500] + "..."
                        if len(prompt) > 500
                        else prompt,
                    },
                    session_id=getattr(self, "session_id", None),
                    step_id=request.get("step_id"),
                )
            else:
                import os

                mode = os.environ.get("PINOCCHIO_MODE", "production")
                if mode == "development":
                    raise RuntimeError(
                        "[DEV] Missing prompt in request; PromptManager must provide prompt in development mode."
                    )
                prompt = self._build_generation_prompt(request)
                self.verbose_logger.log_agent_activity(
                    self.agent_type,
                    "Built prompt using _build_generation_prompt (agent fallback, production only)",
                    data={
                        "request_id": request_id,
                        "prompt_length": len(prompt),
                        "prompt_preview": prompt[:500] + "..."
                        if len(prompt) > 500
                        else prompt,
                    },
                    session_id=getattr(self, "session_id", None),
                    step_id=request.get("step_id"),
                )

            # Log prompt details
            self.verbose_logger.log_agent_activity(
                self.agent_type,
                "[Tracing] Final prompt hash and tail",
                data={
                    "prompt_hash": hash(prompt),
                    "prompt_head": prompt[:500],
                    "prompt_tail": prompt[-500:],
                },
                session_id=getattr(self, "session_id", None),
                step_id=request.get("step_id"),
            )

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

            # Log LLM response summary
            self.verbose_logger.log_agent_activity(
                self.agent_type,
                "[Tracing] LLM response summary",
                data={
                    "llm_response_type": type(llm_response).__name__,
                    "llm_response_keys": list(llm_response.keys())
                    if isinstance(llm_response, dict)
                    else [],
                    "llm_response_preview": str(llm_response)[:500],
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

            # Log processed output details WITH FULL CODE CONTENT
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
                    # FULL CODE CONTENT FOR LOG
                    "generated_code": output.get("code", ""),
                    "full_explanation": output.get("explanation", ""),
                    "optimization_techniques": output.get(
                        "optimization_techniques", []
                    ),
                    "hyperparameters": output.get("hyperparameters", {}),
                    "performance_notes": output.get("performance_notes", ""),
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
        cuda_context = self._get_default_cuda_context()
        return f"""
{cuda_context}

## Generator Agent Instructions for CUDA Code Generation:

### Primary Responsibilities:
1. Generate efficient, production-ready CUDA code based on user requirements
2. Apply appropriate optimization techniques for the target GPU architecture
3. Implement proper error checking and bounds validation
4. Use meaningful variable names and comprehensive documentation
5. Consider memory access patterns for optimal performance
6. Ensure code follows CUDA best practices and conventions

### Code Generation Focus Areas:
- **Performance Optimization**: Loop tiling, vectorization, memory coalescing
- **Memory Management**: Efficient use of global, shared, constant, and texture memory
- **Thread Organization**: Optimal block and grid dimensions, warp efficiency
- **Algorithm Design**: GPU-friendly data structures and computational patterns
- **Error Handling**: Comprehensive CUDA error checking and validation
- **Scalability**: Code that scales across different GPU architectures

### CUDA-Specific Requirements:
- Always include proper CUDA error checking (cudaGetLastError, etc.)
- Optimize for memory bandwidth and computational throughput
- Consider occupancy and resource utilization
- Implement bounds checking for array accesses
- Use appropriate CUDA memory types based on access patterns
- Include performance hints and optimization comments
- Provide kernel launch configuration recommendations

### Output Quality Standards:
- Code must be compilable with nvcc
- Include complete examples with host code when appropriate
- Provide clear explanations of optimization techniques used
- List assumptions and limitations
- Include performance expectations and bottleneck analysis
"""

    def _get_generation_output_format(self) -> str:
        """Get output format for generation response (简洁版)."""
        return (
            "Please provide your response in JSON format with the following structure:\n"
            "{\n"
            '    "agent_type": "generator",\n'
            '    "success": true,\n'
            '    "output": {\n'
            '        "code": "// CUDA or Python code",\n'
            '        "language": "cuda",\n'
            '        "kernel_type": "compute|memory|mixed",\n'
            '        "explanation": "Brief explanation of the code",\n'
            '        "optimization_techniques": ["memory_coalescing", "shared_memory"],\n'
            '        "hyperparameters": {\n'
            '            "block_size": "...",\n'
            '            "grid_size": "..."\n'
            "        }\n"
            "    },\n"
            '    "error_message": null\n'
            "}"
        )

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
            "code": self._generate_simple_cuda_code(task_description),
            "language": "cuda",
            "kernel_type": "compute",
            "explanation": f"Simple CUDA implementation for: {task_description}",
            "optimization_techniques": ["basic_parallelization"],
            "hyperparameters": {
                "block_size": "256",
                "grid_size": "calculated_from_problem_size",
                "shared_memory_size": "0",
                "registers_per_thread": "estimated",
            },
            "performance_notes": "Basic parallel implementation, may benefit from optimization",
            "dependencies": ["cuda_runtime"],
            "complexity": "O(n) parallel execution",
            "compilation_flags": ["-arch=sm_60", "-O2"],
            "memory_requirements": {
                "device_memory": "problem_dependent",
                "shared_memory": "0",
                "registers": "low",
            },
            "launch_configuration": {
                "recommended_block_size": [256, 1, 1],
                "grid_size_calculation": "(n + 255) / 256",
                "occupancy_analysis": "high_theoretical_occupancy",
            },
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

    def _generate_simple_cuda_code(self, task_description: str) -> str:
        """Generate a simple CUDA code template based on task description."""
        # Basic CUDA kernel template
        if "matrix" in task_description.lower() or "matmul" in task_description.lower():
            return """
#include <cuda_runtime.h>
#include <cublas_v2.h>

__global__ void matrixMultiply(float* A, float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Host function
cudaError_t launchMatrixMultiply(float* h_A, float* h_B, float* h_C, int N) {
    float *d_A, *d_B, *d_C;
    size_t size = N * N * sizeof(float);

    // Allocate device memory
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // Copy data to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 blockSize(16, 16);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (N + blockSize.y - 1) / blockSize.y);
    matrixMultiply<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);

    // Copy result back
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return cudaGetLastError();
}
"""
        elif "vector" in task_description.lower() or "add" in task_description.lower():
            return """
#include <cuda_runtime.h>

__global__ void vectorAdd(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// Host function
cudaError_t launchVectorAdd(float* h_a, float* h_b, float* h_c, int n) {
    float *d_a, *d_b, *d_c;
    size_t size = n * sizeof(float);

    // Allocate device memory
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    // Copy data to device
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // Launch kernel
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    vectorAdd<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);

    // Copy result back
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return cudaGetLastError();
}
"""
        else:
            # Generic parallel computation template
            return f"""
#include <cuda_runtime.h>

// CUDA kernel for: {task_description}
__global__ void computeKernel(float* input, float* output, int n) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {{
        // TODO: Implement specific computation for: {task_description}
        output[idx] = input[idx] * 2.0f;  // Placeholder computation
    }}
}}

// Host function
cudaError_t launchComputation(float* h_input, float* h_output, int n) {{
    float *d_input, *d_output;
    size_t size = n * sizeof(float);

    // Allocate device memory
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);

    // Copy data to device
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

    // Launch kernel
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    computeKernel<<<gridSize, blockSize>>>(d_input, d_output, n);

    // Check for errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {{
        return error;
    }}

    // Copy result back
    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);

    return cudaSuccess;
}}
"""
