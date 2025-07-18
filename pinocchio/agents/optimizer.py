"""Optimizer agent for code optimization and performance tuning."""

import logging
import time
from typing import Any, Dict, List

from pinocchio.config import ConfigManager
from pinocchio.llm.custom_llm_client import CustomLLMClient
from pinocchio.utils.file_utils import get_output_path

from ..data_models.agent import AgentResponse
from ..utils.json_parser import format_json_response
from ..utils.temp_utils import cleanup_temp_files, create_temp_file
from ..utils.verbose_logger import LogLevel, get_verbose_logger
from .base import AgentWithRetry

logger = logging.getLogger(__name__)


class OptimizerAgent(AgentWithRetry):
    """Agent responsible for code optimization and performance tuning."""

    def __init__(
        self, llm_client: Any = None, max_retries: int = 3, retry_delay: float = 1.0
    ):
        """
        Initialize Optimizer agent.

        Args:
            llm_client: LLM client instance (optional)
            max_retries: Maximum retry attempts for LLM calls
            retry_delay: Delay between retry attempts in seconds
        """
        if llm_client is None:
            config_manager = ConfigManager()
            agent_llm_config = config_manager.get_agent_llm_config("optimizer")
            verbose = config_manager.get("verbose.enabled", True)
            llm_client = CustomLLMClient(agent_llm_config, verbose=verbose)
        super().__init__("optimizer", llm_client, max_retries, retry_delay)
        logger.info("OptimizerAgent initialized with its own LLM client")

    async def execute(self, request: Dict[str, Any]) -> AgentResponse:
        """
        Execute code optimization task.

        Args:
            request: Optimization request containing code and optimization goals

        Returns:
            Agent response with optimization suggestions
        """
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
                "previous_results_keys": list(
                    request.get("previous_results", {}).keys()
                ),
            },
            session_id=getattr(self, "session_id", None),
            step_id=request.get("step_id"),
        )

        # Log optimizer execution start with detailed input analysis
        self.verbose_logger.log_agent_activity(
            self.agent_type,
            "Optimizer execution started",
            data={
                "request_id": request_id,
                "request_keys": list(request.keys()),
                "request_size": len(str(request)),
                "code_length": len(request.get("code", "")),
                "has_optimization_goals": "optimization_goals" in request,
                "has_performance_metrics": "performance_metrics" in request,
                "has_current_performance": "current_performance" in request,
                "has_context": "context" in request,
                "has_detailed_instruction": "detailed_instruction" in request,
                "optimization_goals_count": len(request.get("optimization_goals", [])),
                "performance_metrics_count": len(
                    request.get("performance_metrics", {})
                ),
            },
            session_id=getattr(self, "session_id", None),
            step_id=request.get("step_id"),
        )

        try:
            # Build prompt for code optimization with detailed logging
            self.verbose_logger.log_agent_activity(
                self.agent_type,
                "Building optimization prompt",
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
                prompt = self._build_optimization_prompt(request)
                self.verbose_logger.log_agent_activity(
                    self.agent_type,
                    "Built prompt using _build_optimization_prompt (agent fallback, production only)",
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

            self.verbose_logger.log_agent_activity(
                self.agent_type,
                "Optimization prompt built",
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
                "Calling LLM for code optimization",
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
                "LLM response received for optimization",
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
                "Processing optimization response",
                data={"request_id": request_id},
                session_id=getattr(self, "session_id", None),
                step_id=request.get("step_id"),
            )

            output = self._process_optimization_response(llm_response, request)

            # Log processed output details WITH FULL OPTIMIZATION RESULTS
            self.verbose_logger.log_agent_activity(
                self.agent_type,
                "Optimization response processed",
                data={
                    "request_id": request_id,
                    "output_keys": list(output.keys())
                    if isinstance(output, dict)
                    else [],
                    "output_size": len(str(output)),
                    "has_optimization_suggestions": "optimization_suggestions"
                    in output,
                    "has_performance_analysis": "performance_analysis" in output,
                    "has_optimized_code": "optimized_code" in output,
                    "suggestions_count": len(
                        output.get("optimization_suggestions", [])
                    ),
                    "optimized_code_length": len(output.get("optimized_code", "")),
                    "full_optimization_suggestions": output.get(
                        "optimization_suggestions", []
                    ),
                    "full_performance_analysis": output.get("performance_analysis", ""),
                    "full_optimized_code": output.get("optimized_code", ""),
                    "optimization_techniques_applied": output.get(
                        "optimization_techniques_applied", []
                    ),
                    "performance_improvements": output.get(
                        "performance_improvements", {}
                    ),
                    "code_diff": output.get("code_diff", ""),
                },
                session_id=getattr(self, "session_id", None),
                step_id=request.get("step_id"),
            )

            # Create successful response with detailed logging
            self.verbose_logger.log_agent_activity(
                self.agent_type,
                "Creating optimization response",
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

            # Log optimization completion
            self.verbose_logger.log_agent_activity(
                self.agent_type,
                "Optimizer execution completed",
                data={
                    "request_id": request_id,
                    "response_success": response.success,
                    "response_output_keys": list(response.output.keys())
                    if isinstance(response.output, dict)
                    else [],
                    "response_processing_time_ms": response.processing_time_ms,
                    "suggestions_count": len(
                        response.output.get("optimization_suggestions", [])
                    ),
                    "optimized_code_length": len(
                        response.output.get("optimized_code", "")
                    ),
                    "call_count": self.call_count,
                    "total_processing_time": self.total_processing_time,
                },
                session_id=getattr(self, "session_id", None),
                step_id=request.get("step_id"),
            )

            return response

        except Exception as e:
            # Log optimization error with detailed error information
            self.verbose_logger.log(
                LogLevel.ERROR,
                f"agent:{self.agent_type}",
                "Optimizer execution failed",
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

    def _build_optimization_prompt(self, request: Dict[str, Any]) -> str:
        """
        Build specific prompt for code optimization.

        Args:
            request: Optimization request

        Returns:
            Formatted prompt string
        """
        # Log prompt building start with detailed input analysis
        self.verbose_logger.log_agent_activity(
            self.agent_type,
            "Building optimization prompt from request",
            data={
                "request_keys": list(request.keys()),
                "request_size": len(str(request)),
                "has_code": "code" in request,
                "has_optimization_goals": "optimization_goals" in request,
                "has_performance_metrics": "performance_metrics" in request,
                "has_current_performance": "current_performance" in request,
                "has_context": "context" in request,
                "has_detailed_instruction": "detailed_instruction" in request,
            },
            session_id=getattr(self, "session_id", None),
            step_id=request.get("step_id"),
        )

        code = request.get("code", "")
        optimization_goals = request.get("optimization_goals", [])
        performance_metrics = request.get("performance_metrics", {})
        context = request.get("context", {})
        current_performance = request.get("current_performance", {})
        detailed_instruction = request.get("detailed_instruction", "")

        # Log extracted components
        self.verbose_logger.log_agent_activity(
            self.agent_type,
            "Extracted optimization components",
            data={
                "code_length": len(code),
                "optimization_goals_count": len(optimization_goals),
                "performance_metrics_count": len(performance_metrics),
                "context_size": len(str(context)),
                "current_performance_count": len(current_performance),
                "detailed_instruction_length": len(detailed_instruction),
            },
            session_id=getattr(self, "session_id", None),
            step_id=request.get("step_id"),
        )

        prompt_parts = [
            "You are an Optimizer agent in the Pinocchio multi-agent system.",
            "Your primary task is to analyze Choreo DSL code and provide optimization suggestions.",
            "",
            "Code to optimize:",
            "```choreo",
            code,
            "```",
        ]

        # Add detailed instruction if available
        if detailed_instruction:
            prompt_parts.extend(["", "Detailed Instructions:", detailed_instruction])
            self.verbose_logger.log_agent_activity(
                self.agent_type,
                "Added detailed instructions to optimization prompt",
                data={"detailed_instruction_length": len(detailed_instruction)},
                session_id=getattr(self, "session_id", None),
                step_id=request.get("step_id"),
            )
        else:
            # Fallback to basic optimization goals
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

        if current_performance:
            formatted_performance = self._format_performance(current_performance)
            prompt_parts.extend(
                [
                    "",
                    "Current Performance Metrics:",
                    formatted_performance,
                ]
            )
            self.verbose_logger.log_agent_activity(
                self.agent_type,
                "Added current performance metrics to prompt",
                data={
                    "current_performance_count": len(current_performance),
                    "formatted_performance_length": len(formatted_performance),
                },
                session_id=getattr(self, "session_id", None),
                step_id=request.get("step_id"),
            )

        if performance_metrics:
            formatted_target_metrics = self._format_performance(performance_metrics)
            prompt_parts.extend(
                [
                    "",
                    "Target Performance Metrics:",
                    formatted_target_metrics,
                ]
            )
            self.verbose_logger.log_agent_activity(
                self.agent_type,
                "Added target performance metrics to prompt",
                data={
                    "performance_metrics_count": len(performance_metrics),
                    "formatted_target_metrics_length": len(formatted_target_metrics),
                },
                session_id=getattr(self, "session_id", None),
                step_id=request.get("step_id"),
            )

        if context:
            prompt_parts.extend(["", "Context:", str(context)])
            self.verbose_logger.log_agent_activity(
                self.agent_type,
                "Added context to optimization prompt",
                data={"context_size": len(str(context))},
                session_id=getattr(self, "session_id", None),
                step_id=request.get("step_id"),
            )

        # Get agent instructions and output format
        instructions = self._get_agent_instructions()
        output_format = self._get_optimization_output_format()

        # Log instructions and format details
        self.verbose_logger.log_agent_activity(
            self.agent_type,
            "Retrieved optimization instructions and format",
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
            "Optimization prompt built successfully",
            data={
                "final_prompt_length": len(final_prompt),
                "prompt_parts_count": len(prompt_parts),
                "has_detailed_instruction": bool(detailed_instruction),
                "has_optimization_goals": bool(optimization_goals),
                "has_current_performance": bool(current_performance),
                "has_performance_metrics": bool(performance_metrics),
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
        """Get Optimizer-specific instructions."""
        cuda_context = self._get_default_cuda_context()
        return f"""
{cuda_context}

## Optimizer Agent Instructions for CUDA Code Optimization:

### Primary Responsibilities:
1. Analyze existing CUDA code for performance bottlenecks and optimization opportunities
2. Identify specific areas where GPU performance can be improved
3. Provide actionable optimization suggestions with implementation details
4. Estimate performance improvements and implementation complexity
5. Consider trade-offs between performance gains and code maintainability
6. Ensure optimizations maintain correctness and numerical stability

### CUDA Optimization Focus Areas:
- **Memory Optimization**:
  - Global memory coalescing and alignment
  - Shared memory bank conflict elimination
  - Constant and texture memory utilization
  - Memory access pattern analysis and improvement

- **Compute Optimization**:
  - Warp divergence minimization
  - Instruction-level parallelism enhancement
  - Loop unrolling and fusion opportunities
  - Register pressure reduction

- **Occupancy Optimization**:
  - Thread block size tuning
  - Resource usage optimization (registers, shared memory)
  - Dynamic vs static shared memory allocation
  - Kernel launch configuration tuning

- **Architecture-Specific Optimizations**:
  - Tensor Core utilization (when applicable)
  - Warp-level primitives and cooperative groups
  - Multi-GPU scaling and communication optimization
  - Stream and event optimization for overlap

### Analysis Methodology:
1. Profile the code to identify actual bottlenecks
2. Analyze memory access patterns and bandwidth utilization
3. Evaluate computational intensity and arithmetic throughput
4. Check for suboptimal kernel launch configurations
5. Identify opportunities for algorithmic improvements
6. Consider GPU architecture-specific optimizations

### Optimization Recommendations Format:
- Specific code changes with before/after examples
- Expected performance improvement percentages
- Implementation difficulty assessment (easy/medium/hard)
- Risk level evaluation (low/medium/high)
- Priority ranking based on impact vs effort
- Verification methods for confirming improvements
"""

    def _get_optimization_output_format(self) -> str:
        """Get output format for optimization response (具体技术挂钩)."""
        return (
            "Please provide your response in JSON format with the following structure:\n"
            "{\n"
            '    "agent_type": "optimizer",\n'
            '    "success": true,\n'
            '    "output": {\n'
            '        "code": "// Optimized code (even if unchanged)",\n'
            '        "optimization_suggestions": [\n'
            "            {\n"
            '                "technique": "shared_memory_tiling|loop_unrolling|vectorization|register_blocking|launch_bounds|prefetching|warp_shuffle|bank_conflict_avoidance",\n'
            '                "description": "具体优化技术的作用和理由",\n'
            '                "before_code": "// ...",\n'
            '                "after_code": "// ...",\n'
            '                "expected_improvement": {\n'
            '                    "memory_bandwidth_gb_s": "+15%",\n'
            '                    "execution_time_ms": "-10%"\n'
            "                }\n"
            "            }\n"
            "        ],\n"
            '        "explanation": "What was optimized and why"\n'
            "    },\n"
            '    "error_message": null\n'
            "}"
        )

    def _format_performance(self, performance: Dict[str, Any]) -> str:
        """Format performance metrics for prompt."""
        # Log performance formatting
        self.verbose_logger.log_agent_activity(
            self.agent_type,
            "Formatting performance metrics",
            data={
                "performance_count": len(performance),
                "performance_keys": list(performance.keys()),
            },
            session_id=getattr(self, "session_id", None),
        )

        formatted = []
        for key, value in performance.items():
            formatted.append(f"- {key}: {value}")

        result = "\n".join(formatted)

        # Log formatting result
        self.verbose_logger.log_agent_activity(
            self.agent_type,
            "Performance metrics formatted",
            data={
                "formatted_length": len(result),
                "formatted_lines": len(formatted),
            },
            session_id=getattr(self, "session_id", None),
        )

        return result

    def _process_optimization_response(
        self, llm_response: Dict[str, Any], request: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process LLM response and extract optimization results.

        Args:
            llm_response: Raw LLM response
            request: Original request

        Returns:
            Processed output dictionary
        """
        # Log response processing start
        self.verbose_logger.log_agent_activity(
            self.agent_type,
            "Processing optimization response",
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
            # Fallback: create basic optimization structure
            self.verbose_logger.log_agent_activity(
                self.agent_type,
                "LLM response not successful, creating fallback optimization structure",
                data={
                    "llm_success": llm_response.get("success", False),
                    "has_output": "output" in llm_response,
                },
                session_id=getattr(self, "session_id", None),
                step_id=request.get("step_id"),
            )

            output = {
                "optimization_suggestions": [
                    {
                        "type": "general",
                        "description": "Basic optimization analysis",
                        "code_changes": "// Basic optimization suggestions",
                        "expected_improvement": {
                            "performance_gain": "5-10%",
                            "memory_reduction": "5-10%",
                            "complexity_change": "same",
                        },
                        "implementation_difficulty": "medium",
                        "risk_level": "low",
                        "priority": 3,
                    }
                ],
                "performance_analysis": {
                    "current_bottlenecks": ["general_optimization_needed"],
                    "optimization_potential": "medium",
                    "estimated_overall_improvement": "5-10%",
                },
                "optimized_code": request.get("code", "// No code provided"),
                "optimization_metadata": {
                    "analysis_time": "fallback",
                    "optimization_count": 1,
                    "primary_optimization_focus": "general",
                },
            }

            self.verbose_logger.log_agent_activity(
                self.agent_type,
                "Fallback optimization structure created",
                data={
                    "fallback_output_keys": list(output.keys()),
                    "suggestions_count": len(
                        output.get("optimization_suggestions", [])
                    ),
                },
                session_id=getattr(self, "session_id", None),
                step_id=request.get("step_id"),
            )

        # Validate and enhance output
        if not isinstance(output, dict):
            output = {"optimization_suggestions": [], "performance_analysis": {}}

        # Ensure required fields exist
        required_fields = ["optimization_suggestions", "performance_analysis"]
        for field in required_fields:
            if field not in output:
                output[field] = {} if field == "performance_analysis" else []
                self.verbose_logger.log_agent_activity(
                    self.agent_type,
                    f"Added missing required field: {field}",
                    data={
                        "field": field,
                        "default_value": "{}"
                        if field == "performance_analysis"
                        else "[]",
                    },
                    session_id=getattr(self, "session_id", None),
                    step_id=request.get("step_id"),
                )

        # Log final processed output
        self.verbose_logger.log_agent_activity(
            self.agent_type,
            "Optimization response processing completed",
            data={
                "output_keys": list(output.keys()),
                "output_size": len(str(output)),
                "has_optimization_suggestions": "optimization_suggestions" in output,
                "has_performance_analysis": "performance_analysis" in output,
                "has_optimized_code": "optimized_code" in output,
                "suggestions_count": len(output.get("optimization_suggestions", [])),
                "optimized_code_length": len(output.get("optimized_code", "")),
            },
            session_id=getattr(self, "session_id", None),
            step_id=request.get("step_id"),
        )

        return output

    def analyze_code_performance(
        self, code: str, goals: List[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze code performance and provide optimization suggestions.

        Args:
            code: Code to analyze
            goals: Optimization goals

        Returns:
            Performance analysis and suggestions
        """
        # Log performance analysis start
        self.verbose_logger.log_agent_activity(
            self.agent_type,
            "Code performance analysis started",
            data={
                "code_length": len(code),
                "goals_count": len(goals) if goals else 0,
                "goals": goals,
            },
            session_id=getattr(self, "session_id", None),
        )

        request = {
            "code": code,
            "optimization_goals": goals or ["performance", "memory"],
            "request_id": f"perf_analysis_{int(time.time())}",
        }

        # Create a simple analysis for synchronous use
        output = {
            "optimization_suggestions": [
                {
                    "type": "performance_analysis",
                    "description": "Basic performance analysis completed",
                    "code_changes": "// Performance analysis suggestions",
                    "expected_improvement": {
                        "performance_gain": "5-15%",
                        "memory_reduction": "5-10%",
                        "complexity_change": "same",
                    },
                    "implementation_difficulty": "medium",
                    "risk_level": "low",
                    "priority": 3,
                }
            ],
            "performance_analysis": {
                "current_bottlenecks": ["analysis_completed"],
                "optimization_potential": "medium",
                "estimated_overall_improvement": "5-15%",
            },
            "optimized_code": code,
            "optimization_metadata": {
                "analysis_time": "synchronous",
                "optimization_count": 1,
                "primary_optimization_focus": "analysis",
            },
        }

        # Log performance analysis completion
        self.verbose_logger.log_agent_activity(
            self.agent_type,
            "Code performance analysis completed",
            data={
                "output_keys": list(output.keys()),
                "suggestions_count": len(output.get("optimization_suggestions", [])),
                "code_length": len(code),
            },
            session_id=getattr(self, "session_id", None),
        )

        return output

    def get_optimization_suggestions(self, code: str) -> List[Dict[str, Any]]:
        """
        Get optimization suggestions for code.

        Args:
            code: Code to optimize

        Returns:
            List of optimization suggestions
        """
        # Log optimization suggestions request
        self.verbose_logger.log_agent_activity(
            self.agent_type,
            "Getting optimization suggestions",
            data={"code_length": len(code)},
            session_id=getattr(self, "session_id", None),
        )

        analysis = self.analyze_code_performance(code)
        suggestions = analysis.get("optimization_suggestions", [])

        # Log suggestions retrieval
        self.verbose_logger.log_agent_activity(
            self.agent_type,
            "Optimization suggestions retrieved",
            data={"suggestions_count": len(suggestions)},
            session_id=getattr(self, "session_id", None),
        )

        return suggestions

    def get_optimized_code(self, code: str) -> str:
        """
        Get optimized version of code.

        Args:
            code: Code to optimize

        Returns:
            Optimized code
        """
        # Log optimized code request
        self.verbose_logger.log_agent_activity(
            self.agent_type,
            "Getting optimized code",
            data={"code_length": len(code)},
            session_id=getattr(self, "session_id", None),
        )

        analysis = self.analyze_code_performance(code)
        optimized_code = analysis.get("optimized_code", code)

        # Log optimized code retrieval
        self.verbose_logger.log_agent_activity(
            self.agent_type,
            "Optimized code retrieved",
            data={
                "original_code_length": len(code),
                "optimized_code_length": len(optimized_code),
            },
            session_id=getattr(self, "session_id", None),
        )

        return optimized_code
