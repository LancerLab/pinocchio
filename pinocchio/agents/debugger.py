"""Debugger agent for code analysis and error detection."""

import logging
import time
import traceback
from typing import Any, Dict, List

from pinocchio.config import ConfigManager
from pinocchio.llm.custom_llm_client import CustomLLMClient
from pinocchio.tools import CudaDebugTools, ToolManager
from pinocchio.utils.file_utils import get_output_path

from ..data_models.agent import AgentResponse
from ..utils.json_parser import format_json_response
from ..utils.temp_utils import cleanup_temp_files, create_temp_file
from ..utils.verbose_logger import LogLevel, get_verbose_logger
from .base import AgentWithRetry

logger = logging.getLogger(__name__)


class DebuggerAgent(AgentWithRetry):
    """Agent responsible for code analysis and error detection."""

    def __init__(
        self, llm_client: Any = None, max_retries: int = 3, retry_delay: float = 1.0
    ):
        """
        Initialize Debugger agent.

        Args:
            llm_client: LLM client instance (optional)
            max_retries: Maximum retry attempts for LLM calls
            retry_delay: Delay between retry attempts in seconds
        """
        if llm_client is None:
            config_manager = ConfigManager()
            agent_llm_config = config_manager.get_agent_llm_config("debugger")
            verbose = config_manager.get("verbose.enabled", True)
            llm_client = CustomLLMClient(agent_llm_config, verbose=verbose)
        super().__init__("debugger", llm_client, max_retries, retry_delay)

        # Initialize tool manager and register debugging tools
        self.tool_manager = ToolManager()
        CudaDebugTools.register_tools(self.tool_manager)

        logger.info(
            "DebuggerAgent initialized with its own LLM client and debugging tools"
        )

    def safe_len(self, value, field_name, context=None):
        """Safe to get length, None returns 0, other types use len, otherwise returns 0."""
        try:
            if value is None:
                return 0
            return len(value)
        except Exception:
            self.verbose_logger.log_agent_activity(
                self.agent_type,
                f"[Tracing] NoneType or invalid type encountered for len({field_name})",
                data={
                    "field": field_name,
                    "context": context,
                    "traceback": traceback.format_stack(),
                },
                session_id=getattr(self, "session_id", None),
            )
            return 0

    def _run_debugging_tools(
        self, cuda_code: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Run debugging tools on CUDA code.

        Args:
            cuda_code: CUDA source code to debug
            context: Additional context for debugging

        Returns:
            Dict containing tool results
        """
        tool_results = {}

        try:
            # Run syntax checker
            syntax_result = self.tool_manager.execute_tool(
                "cuda_syntax_check", cuda_code=cuda_code, strict=True
            )
            tool_results["syntax_check"] = {
                "status": syntax_result.status.value,
                "output": syntax_result.output,
                "metadata": syntax_result.metadata,
            }

            # Run compiler check
            compile_result = self.tool_manager.execute_tool(
                "cuda_compile",
                cuda_code=cuda_code,
                arch=context.get("target_arch", "compute_75"),
                verbose=True,
            )
            tool_results["compilation"] = {
                "status": compile_result.status.value,
                "output": compile_result.output,
                "metadata": compile_result.metadata,
            }

            # If compilation succeeds, run memory check
            if compile_result.status.value == "success":
                memcheck_result = self.tool_manager.execute_tool(
                    "cuda_memcheck",
                    cuda_code=cuda_code,
                    check_type="memcheck",
                    arch=context.get("target_arch", "compute_75"),
                )
                tool_results["memory_check"] = {
                    "status": memcheck_result.status.value,
                    "output": memcheck_result.output,
                    "metadata": memcheck_result.metadata,
                }

        except Exception as e:
            logger.error(f"Error running debugging tools: {e}")
            tool_results["error"] = str(e)

        return tool_results

    def _format_tool_results_for_prompt(self, tool_results: Dict[str, Any]) -> str:
        """
        Format tool results for inclusion in LLM prompt.

        Args:
            tool_results: Results from debugging tools

        Returns:
            Formatted string for prompt
        """
        formatted = []

        for tool_name, result in tool_results.items():
            if tool_name == "error":
                formatted.append(f"Tool Execution Error: {result}")
                continue

            status = result.get("status", "unknown")
            output = result.get("output", "")
            metadata = result.get("metadata", {})

            formatted.append(f"=== {tool_name.upper()} RESULTS ===")
            formatted.append(f"Status: {status}")

            if output:
                formatted.append(f"Output:\n{output}")

            if metadata and isinstance(metadata, dict):
                # Extract key information from metadata
                if "error_analysis" in metadata:
                    error_analysis = metadata["error_analysis"]
                    formatted.append(
                        f"Error Analysis: {error_analysis.get('error_count', 0)} errors, {error_analysis.get('warning_count', 0)} warnings"
                    )

                if "compilation_successful" in metadata:
                    formatted.append(
                        f"Compilation: {'SUCCESS' if metadata['compilation_successful'] else 'FAILED'}"
                    )

                if "memcheck_analysis" in metadata:
                    memcheck = metadata["memcheck_analysis"]
                    formatted.append(
                        f"Memory Issues: {memcheck.get('total_issues', 0)} found"
                    )

            formatted.append("")  # Empty line separator

        return "\n".join(formatted)

    async def execute(self, request: Dict[str, Any]) -> AgentResponse:
        """Execute the debugging agent with the given request and return the agent response."""
        # === safe_len tool function, ensure scope is available ===
        # Replace all len() calls with self.safe_len, e.g.:
        # self.safe_len(request.get("code"), "request.code", request)

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

        # Log debugger execution start with detailed input analysis
        self.verbose_logger.log_agent_activity(
            self.agent_type,
            "Debugger execution started",
            data={
                "request_id": request_id,
                "request_keys": list(request.keys()),
                "request_size": self.safe_len(request, "request", request),
                "code_length": self.safe_len(
                    request.get("code"), "request.code", request
                ),
                "has_error_message": "error_message" in request,
                "has_context": "context" in request,
                "has_detailed_instruction": "detailed_instruction" in request,
                "error_message_length": self.safe_len(
                    request.get("error_message"), "request.error_message", request
                ),
                "context_size": self.safe_len(
                    request.get("context"), "request.context", request
                ),
            },
            session_id=getattr(self, "session_id", None),
            step_id=request.get("step_id"),
        )

        try:
            # Build prompt for debugging with detailed logging
            self.verbose_logger.log_agent_activity(
                self.agent_type,
                "Building debugging prompt",
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
                        "prompt_length": self.safe_len(prompt, "prompt", request),
                        "prompt_preview": prompt[:500] + "..."
                        if self.safe_len(prompt, "prompt", request) > 500
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
                prompt = self._build_debugging_prompt(request)
                self.verbose_logger.log_agent_activity(
                    self.agent_type,
                    "Built prompt using _build_debugging_prompt (agent fallback, production only)",
                    data={
                        "request_id": request_id,
                        "prompt_length": self.safe_len(prompt, "prompt", request),
                        "prompt_preview": prompt[:500] + "..."
                        if self.safe_len(prompt, "prompt", request) > 500
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

            # Run debugging tools if CUDA code is present
            tool_results = {}
            if request.get("code") and "cuda" in request.get("language", "").lower():
                self.verbose_logger.log_agent_activity(
                    self.agent_type,
                    "Running CUDA debugging tools",
                    data={"request_id": request_id},
                    session_id=getattr(self, "session_id", None),
                    step_id=request.get("step_id"),
                )

                tool_results = self._run_debugging_tools(
                    request["code"], request.get("context", {})
                )

                # LOG FULL MCP TOOL RESULTS
                self.verbose_logger.log_agent_activity(
                    self.agent_type,
                    "MCP debugging tools completed",
                    data={
                        "request_id": request_id,
                        "tool_results_count": self.safe_len(
                            tool_results, "tool_results", request
                        ),
                        "tool_names": list(tool_results.keys()),
                        "full_tool_results": tool_results,  # COMPLETE TOOL OUTPUT
                        "tool_execution_summary": {
                            name: {
                                "status": result.get("status", "unknown"),
                                "output_length": self.safe_len(
                                    result.get("output", ""), "result.output", request
                                ),
                                "has_error": bool(result.get("error")),
                                "execution_time": result.get("execution_time", 0),
                            }
                            for name, result in tool_results.items()
                        },
                    },
                    session_id=getattr(self, "session_id", None),
                    step_id=request.get("step_id"),
                )

                # Enhance prompt with tool results
                if tool_results:
                    tool_summary = self._format_tool_results_for_prompt(tool_results)
                    prompt = f"{prompt}\n\nDEBUGGING TOOL RESULTS:\n{tool_summary}\n\nPlease analyze these tool results and provide comprehensive debugging guidance."

            # Log prompt details
            self.verbose_logger.log_agent_activity(
                self.agent_type,
                "Debugging prompt built",
                data={
                    "request_id": request_id,
                    "prompt_length": self.safe_len(prompt, "prompt", request),
                    "prompt_preview": prompt[:300] + "..."
                    if self.safe_len(prompt, "prompt", request) > 300
                    else prompt,
                },
                session_id=getattr(self, "session_id", None),
                step_id=request.get("step_id"),
            )

            # Call LLM with retry and detailed logging
            self.verbose_logger.log_agent_activity(
                self.agent_type,
                "Calling LLM for code debugging",
                data={
                    "request_id": request_id,
                    "prompt_length": self.safe_len(prompt, "prompt", request),
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
                "LLM response received for debugging",
                data={
                    "request_id": request_id,
                    "llm_response_keys": list(llm_response.keys())
                    if isinstance(llm_response, dict)
                    else [],
                    "llm_response_type": type(llm_response).__name__,
                    "llm_response_size": self.safe_len(
                        llm_response, "llm_response", request
                    ),
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
                "Processing debugging response",
                data={"request_id": request_id},
                session_id=getattr(self, "session_id", None),
                step_id=request.get("step_id"),
            )

            output = self._process_debugging_response(llm_response, request)

            # Log processed output details WITH FULL DEBUGGING RESULTS
            self.verbose_logger.log_agent_activity(
                self.agent_type,
                "Debugging response processed",
                data={
                    "request_id": request_id,
                    "output_keys": list(output.keys())
                    if isinstance(output, dict)
                    else [],
                    "output_size": self.safe_len(output, "output", request),
                    "has_issues_found": "issues_found" in output,
                    "has_code_analysis": "code_analysis" in output,
                    "has_debugged_code": "debugged_code" in output,
                    "issues_count": self.safe_len(
                        output.get("issues_found", []), "output.issues_found", request
                    ),
                    "debugged_code_length": self.safe_len(
                        output.get("debugged_code", ""), "output.debugged_code", request
                    ),
                    # FULL DEBUGGING RESULTS FOR LOG
                    "full_issues_found": output.get("issues_found", []),
                    "full_code_analysis": output.get("code_analysis", ""),
                    "full_debugged_code": output.get("debugged_code", ""),
                    "debugging_recommendations": output.get(
                        "debugging_recommendations", []
                    ),
                    "performance_issues": output.get("performance_issues", []),
                    "compilation_issues": output.get("compilation_issues", []),
                },
                session_id=getattr(self, "session_id", None),
                step_id=request.get("step_id"),
            )

            # Create successful response with detailed logging
            self.verbose_logger.log_agent_activity(
                self.agent_type,
                "Creating debugging response",
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

            # Log debugging completion
            self.verbose_logger.log_agent_activity(
                self.agent_type,
                "Debugger execution completed",
                data={
                    "request_id": request_id,
                    "response_success": response.success,
                    "response_output_keys": list(response.output.keys())
                    if isinstance(response.output, dict)
                    else [],
                    "response_processing_time_ms": response.processing_time_ms,
                    "issues_count": self.safe_len(
                        response.output.get("issues_found", []),
                        "response.output.issues_found",
                        request,
                    ),
                    "debugged_code_length": self.safe_len(
                        response.output.get("debugged_code", ""),
                        "response.output.debugged_code",
                        request,
                    ),
                    "call_count": self.call_count,
                    "total_processing_time": self.total_processing_time,
                },
                session_id=getattr(self, "session_id", None),
                step_id=request.get("step_id"),
            )

            return response

        except Exception as e:
            # Log debugging error with detailed error information
            self.verbose_logger.log(
                LogLevel.ERROR,
                f"agent:{self.agent_type}",
                "Debugger execution failed",
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

    def _build_debugging_prompt(self, request: Dict[str, Any]) -> str:
        """
        Build specific prompt for code debugging.

        Args:
            request: Debugging request

        Returns:
            Formatted prompt string
        """
        # Log prompt building start with detailed input analysis
        self.verbose_logger.log_agent_activity(
            self.agent_type,
            "Building debugging prompt from request",
            data={
                "request_keys": list(request.keys()),
                "request_size": self.safe_len(request, "request", request),
                "has_code": "code" in request,
                "has_error_message": "error_message" in request,
                "has_context": "context" in request,
                "has_detailed_instruction": "detailed_instruction" in request,
            },
            session_id=getattr(self, "session_id", None),
            step_id=request.get("step_id"),
        )

        code = request.get("code", "")
        error_message = request.get("error_message", "")
        context = request.get("context", {})
        detailed_instruction = request.get("detailed_instruction", "")

        # Log extracted components
        self.verbose_logger.log_agent_activity(
            self.agent_type,
            "Extracted debugging components",
            data={
                "code_length": self.safe_len(code, "code", request),
                "error_message_length": self.safe_len(
                    error_message, "error_message", request
                ),
                "context_size": self.safe_len(context, "context", request),
                "detailed_instruction_length": self.safe_len(
                    detailed_instruction, "detailed_instruction", request
                ),
            },
            session_id=getattr(self, "session_id", None),
            step_id=request.get("step_id"),
        )

        prompt_parts = [
            "You are a Debugger agent in the Pinocchio multi-agent system.",
            "Your primary task is to analyze Choreo DSL code for issues and provide fixes.",
            "",
            "Code to debug:",
            "```choreo",
            code,
            "```",
        ]

        if error_message:
            prompt_parts.extend(
                [
                    "",
                    "Error Message:",
                    error_message,
                ]
            )
            self.verbose_logger.log_agent_activity(
                self.agent_type,
                "Added error message to debugging prompt",
                data={
                    "error_message_length": self.safe_len(
                        error_message, "error_message", request
                    )
                },
                session_id=getattr(self, "session_id", None),
                step_id=request.get("step_id"),
            )

        # Add detailed instruction if available
        if detailed_instruction:
            prompt_parts.extend(["", "Detailed Instructions:", detailed_instruction])
            self.verbose_logger.log_agent_activity(
                self.agent_type,
                "Added detailed instructions to debugging prompt",
                data={
                    "detailed_instruction_length": self.safe_len(
                        detailed_instruction, "detailed_instruction", request
                    )
                },
                session_id=getattr(self, "session_id", None),
                step_id=request.get("step_id"),
            )

        if context:
            prompt_parts.extend(["", "Context:", str(context)])
            self.verbose_logger.log_agent_activity(
                self.agent_type,
                "Added context to debugging prompt",
                data={"context_size": self.safe_len(context, "context", request)},
                session_id=getattr(self, "session_id", None),
                step_id=request.get("step_id"),
            )

        # Get agent instructions and output format
        instructions = self._get_agent_instructions()
        output_format = self._get_debugging_output_format()

        # Log instructions and format details
        self.verbose_logger.log_agent_activity(
            self.agent_type,
            "Retrieved debugging instructions and format",
            data={
                "instructions_length": self.safe_len(
                    instructions, "instructions", request
                ),
                "output_format_length": self.safe_len(
                    output_format, "output_format", request
                ),
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
            "Debugging prompt built successfully",
            data={
                "final_prompt_length": self.safe_len(
                    final_prompt, "final_prompt", request
                ),
                "prompt_parts_count": len(prompt_parts),
                "has_error_message": bool(error_message),
                "has_detailed_instruction": bool(detailed_instruction),
                "has_context": bool(context),
                "prompt_preview": final_prompt[:400] + "..."
                if self.safe_len(final_prompt, "final_prompt", request) > 400
                else final_prompt,
            },
            session_id=getattr(self, "session_id", None),
            step_id=request.get("step_id"),
        )

        return final_prompt

    def _get_agent_instructions(self) -> str:
        """Get Debugger-specific instructions."""
        cuda_context = self._get_default_cuda_context()
        return f"""
{cuda_context}

## Debugger Agent Instructions for CUDA Code Analysis and Debugging:

### Primary Responsibilities:
1. Analyze CUDA code for compilation errors, runtime issues, and logical bugs
2. Identify common CUDA programming pitfalls and anti-patterns
3. Provide detailed error analysis with root cause identification
4. Suggest specific fixes and improvements for identified issues
5. Validate code correctness and GPU-specific considerations
6. Ensure proper CUDA error handling and resource management

### CUDA-Specific Debugging Areas:
- **Compilation Issues**:
  - NVCC compiler errors and warnings
  - Architecture compatibility problems
  - Header and library linkage issues
  - Template instantiation problems

- **Runtime Errors**:
  - CUDA API call failures and error codes
  - Memory access violations (out-of-bounds, misaligned)
  - Launch configuration errors (invalid grid/block dimensions)
  - Resource exhaustion (memory, registers, shared memory)

- **Memory Management Issues**:
  - Memory leaks and improper deallocation
  - Host-device memory transfer errors
  - Uncoalesced memory access patterns
  - Race conditions and synchronization problems

- **Performance Issues**:
  - Warp divergence and inefficient branching
  - Bank conflicts in shared memory
  - Low occupancy and resource underutilization
  - Inefficient kernel launch configurations

### Analysis Process:
1. **Static Code Analysis**: Review code structure and CUDA best practices
2. **Error Pattern Recognition**: Identify common CUDA error patterns
3. **Memory Safety Check**: Validate pointer usage and bounds checking
4. **Synchronization Review**: Check for proper CUDA synchronization
5. **Performance Assessment**: Identify obvious performance issues
6. **Resource Usage Analysis**: Check register and memory usage

### Error Reporting Format:
- Clear error categorization (compilation/runtime/logic/performance)
- Specific line numbers and code sections with issues
- Detailed explanation of the problem and its implications
- Step-by-step fix recommendations with code examples
- Alternative approaches when applicable
- Validation methods to confirm fixes work correctly

### Code Quality Checks:
- Proper CUDA error checking after API calls
- Appropriate use of __syncthreads() and memory fences
- Correct kernel launch parameter calculations
- Memory allocation/deallocation pairing
- Thread safety and race condition prevention
"""

    def _get_debugging_output_format(self) -> str:
        """Get output format for debugging response (简洁版)."""
        return (
            "Please provide your response in JSON format with the following structure:\n"
            "{\n"
            '    "agent_type": "debugger",\n'
            '    "success": true,\n'
            '    "output": {\n'
            '        "code": "// Debugged code (even if unchanged)",\n'
            '        "issues_found": [\n'
            "            {\n"
            '                "type": "syntax_error|logic_error|performance_issue",\n'
            '                "description": "..."\n'
            "            }\n"
            "        ],\n"
            '        "explanation": "What was fixed or checked"\n'
            "    },\n"
            '    "error_message": null\n'
            "}"
        )

    def _format_runtime_info(self, runtime_info: Dict[str, Any]) -> str:
        """Format runtime information for prompt."""
        # Log runtime info formatting
        self.verbose_logger.log_agent_activity(
            self.agent_type,
            "Formatting runtime information",
            data={
                "runtime_info_count": self.safe_len(runtime_info, "runtime_info", None),
                "runtime_info_keys": list(runtime_info.keys()),
            },
            session_id=getattr(self, "session_id", None),
        )

        formatted = []
        for key, value in runtime_info.items():
            formatted.append(f"- {key}: {value}")

        result = "\n".join(formatted)

        # Log formatting result
        self.verbose_logger.log_agent_activity(
            self.agent_type,
            "Runtime information formatted",
            data={
                "formatted_length": self.safe_len(result, "result", None),
                "formatted_lines": self.safe_len(formatted, "formatted", None),
            },
            session_id=getattr(self, "session_id", None),
        )

        return result

    def _process_debugging_response(
        self, llm_response: Dict[str, Any], request: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process LLM response and extract debugging results.

        Args:
            llm_response: Raw LLM response
            request: Original request

        Returns:
            Processed output dictionary
        """
        # Log response processing start
        self.verbose_logger.log_agent_activity(
            self.agent_type,
            "Processing debugging response",
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
            # Fallback: create basic debugging structure
            self.verbose_logger.log_agent_activity(
                self.agent_type,
                "LLM response not successful, creating fallback debugging structure",
                data={
                    "llm_success": llm_response.get("success", False),
                    "has_output": "output" in llm_response,
                },
                session_id=getattr(self, "session_id", None),
                step_id=request.get("step_id"),
            )

            output = {
                "issues_found": [
                    {
                        "type": "general",
                        "severity": "medium",
                        "line_number": "unknown",
                        "description": "Basic debugging analysis completed",
                        "root_cause": "Analysis not available",
                        "suggested_fix": "Consider detailed debugging",
                        "prevention_tips": "Use comprehensive testing",
                    }
                ],
                "code_analysis": {
                    "overall_health": "fair",
                    "critical_issues_count": 0,
                    "total_issues_count": 1,
                    "suggested_improvements": ["detailed_analysis_needed"],
                },
                "debugged_code": request.get("code", "// No code provided"),
                "debugging_metadata": {
                    "analysis_time": "fallback",
                    "issues_count": 1,
                    "primary_issue_type": "general",
                },
            }

            self.verbose_logger.log_agent_activity(
                self.agent_type,
                "Fallback debugging structure created",
                data={
                    "fallback_output_keys": list(output.keys()),
                    "issues_count": self.safe_len(
                        output.get("issues_found", []), "output.issues_found", request
                    ),
                    "debugged_code_length": self.safe_len(
                        output.get("debugged_code", ""), "output.debugged_code", request
                    ),
                },
                session_id=getattr(self, "session_id", None),
                step_id=request.get("step_id"),
            )

        # Validate and enhance output
        if not isinstance(output, dict):
            output = {"issues_found": [], "code_analysis": {}}

        # Ensure required fields exist
        required_fields = [
            "issues_found",
            "code_analysis",
            "debugged_code",
            "debugging_metadata",
        ]
        for field in required_fields:
            if field not in output:
                if field == "issues_found":
                    output[field] = []
                elif field == "debugging_metadata":
                    output[field] = {
                        "analysis_time": "unknown",
                        "issues_count": 0,
                        "primary_issue_type": "unknown",
                    }
                else:
                    output[field] = {}
                self.verbose_logger.log_agent_activity(
                    self.agent_type,
                    f"Added missing required field: {field}",
                    data={
                        "field": field,
                        "default_value": "[]" if field == "issues_found" else "{}",
                    },
                    session_id=getattr(self, "session_id", None),
                    step_id=request.get("step_id"),
                )

        # Log final processed output
        self.verbose_logger.log_agent_activity(
            self.agent_type,
            "Debugging response processing completed",
            data={
                "output_keys": list(output.keys()),
                "output_size": self.safe_len(output, "output", request),
                "has_issues_found": "issues_found" in output,
                "has_code_analysis": "code_analysis" in output,
                "has_debugged_code": "debugged_code" in output,
                "has_debugging_metadata": "debugging_metadata" in output,
                "issues_count": self.safe_len(
                    output.get("issues_found", []), "output.issues_found", request
                ),
                "debugged_code_length": self.safe_len(
                    output.get("debugged_code", ""), "output.debugged_code", request
                ),
            },
            session_id=getattr(self, "session_id", None),
            step_id=request.get("step_id"),
        )

        return output

    def analyze_code_issues(self, code: str, error_message: str = "") -> Dict[str, Any]:
        """
        Analyze code for issues and provide debugging information.

        Args:
            code: Code to analyze
            error_message: Optional error message

        Returns:
            Debugging analysis results
        """
        # Log code issues analysis start
        self.verbose_logger.log_agent_activity(
            self.agent_type,
            "Code issues analysis started",
            data={
                "code_length": self.safe_len(code, "code", None),
                "error_message_length": self.safe_len(
                    error_message, "error_message", None
                ),
                "has_error_message": bool(error_message),
            },
            session_id=getattr(self, "session_id", None),
        )

        request = {
            "code": code,
            "error_message": error_message,
            "request_id": f"issues_analysis_{int(time.time())}",
        }

        # Create a simple analysis for synchronous use
        output = {
            "issues_found": [
                {
                    "type": "analysis",
                    "severity": "medium",
                    "line_number": "unknown",
                    "description": "Basic code analysis completed",
                    "root_cause": "Analysis not available",
                    "suggested_fix": "Consider detailed analysis",
                    "prevention_tips": "Use comprehensive testing",
                }
            ],
            "code_analysis": {
                "overall_health": "fair",
                "critical_issues_count": 0,
                "total_issues_count": 1,
                "suggested_improvements": ["detailed_analysis_needed"],
            },
            "debugged_code": code,
            "debugging_metadata": {
                "analysis_time": "synchronous",
                "issues_count": 1,
                "primary_issue_type": "analysis",
            },
        }

        # Log code issues analysis completion
        self.verbose_logger.log_agent_activity(
            self.agent_type,
            "Code issues analysis completed",
            data={
                "output_keys": list(output.keys()),
                "issues_count": self.safe_len(
                    output.get("issues_found", []), "output.issues_found", request
                ),
                "code_length": self.safe_len(code, "code", request),
            },
            session_id=getattr(self, "session_id", None),
        )

        return output

    def get_code_issues(self, code: str) -> List[Dict[str, Any]]:
        """
        Get list of issues found in code.

        Args:
            code: Code to analyze

        Returns:
            List of issues found
        """
        # Log code issues request
        self.verbose_logger.log_agent_activity(
            self.agent_type,
            "Getting code issues",
            data={"code_length": self.safe_len(code, "code", None)},
            session_id=getattr(self, "session_id", None),
        )

        analysis = self.analyze_code_issues(code)
        issues = analysis.get("issues_found", [])

        # Log issues retrieval
        self.verbose_logger.log_agent_activity(
            self.agent_type,
            "Code issues retrieved",
            data={"issues_count": self.safe_len(issues, "issues", None)},
            session_id=getattr(self, "session_id", None),
        )

        return issues

    def get_debugged_code(self, code: str, error_message: str = "") -> str:
        """
        Get debugged version of code.

        Args:
            code: Code to debug
            error_message: Optional error message

        Returns:
            Debugged code
        """
        # Log debugged code request
        self.verbose_logger.log_agent_activity(
            self.agent_type,
            "Getting debugged code",
            data={
                "code_length": self.safe_len(code, "code", None),
                "error_message_length": self.safe_len(
                    error_message, "error_message", None
                ),
            },
            session_id=getattr(self, "session_id", None),
        )

        analysis = self.analyze_code_issues(code, error_message)
        debugged_code = analysis.get("debugged_code", code)

        # Log debugged code retrieval
        self.verbose_logger.log_agent_activity(
            self.agent_type,
            "Debugged code retrieved",
            data={
                "original_code_length": self.safe_len(code, "code", None),
                "debugged_code_length": self.safe_len(
                    debugged_code, "debugged_code", None
                ),
            },
            session_id=getattr(self, "session_id", None),
        )

        return debugged_code

    def get_code_health_score(self, code: str) -> Dict[str, Any]:
        """
        Get code health score and analysis.

        Args:
            code: Code to analyze

        Returns:
            Code health score and analysis
        """
        # Log code health score request
        self.verbose_logger.log_agent_activity(
            self.agent_type,
            "Getting code health score",
            data={"code_length": self.safe_len(code, "code", None)},
            session_id=getattr(self, "session_id", None),
        )

        analysis = self.analyze_code_issues(code)
        health_data = {
            "overall_health": analysis.get("code_analysis", {}).get(
                "overall_health", "unknown"
            ),
            "critical_issues_count": analysis.get("code_analysis", {}).get(
                "critical_issues_count", 0
            ),
            "total_issues_count": analysis.get("code_analysis", {}).get(
                "total_issues_count", 0
            ),
            "issues_found": analysis.get("issues_found", []),
        }

        # Log health score retrieval
        self.verbose_logger.log_agent_activity(
            self.agent_type,
            "Code health score retrieved",
            data={
                "overall_health": health_data.get("overall_health"),
                "critical_issues_count": health_data.get("critical_issues_count"),
                "total_issues_count": health_data.get("total_issues_count"),
            },
            session_id=getattr(self, "session_id", None),
        )

        return health_data
