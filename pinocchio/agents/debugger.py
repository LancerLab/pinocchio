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

        try:
            # Build prompt for code debugging
            prompt = self._build_debugging_prompt(request)

            # Call LLM with retry
            llm_response = await self._call_llm_with_retry(prompt)

            # Extract and process the response
            output = self._process_debugging_response(llm_response, request)

            # Create successful response
            return self._create_response(
                request_id=request_id,
                success=True,
                output=output,
                processing_time_ms=int(self.get_average_processing_time()),
            )

        except Exception as e:
            return self._handle_error(request_id, e)

    def _build_debugging_prompt(self, request: Dict[str, Any]) -> str:
        """
        Build specific prompt for code debugging.

        Args:
            request: Debugging request

        Returns:
            Formatted prompt string
        """
        code = request.get("code", "")
        error_message = request.get("error_message", "")
        context = request.get("context", {})
        detailed_instruction = request.get("detailed_instruction", "")

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

        # Add detailed instruction if available
        if detailed_instruction:
            prompt_parts.extend(["", "Detailed Instructions:", detailed_instruction])

        if context:
            prompt_parts.extend(["", "Context:", str(context)])

        prompt_parts.extend(
            [
                "",
                self._get_agent_instructions(),
                "",
                self._get_debugging_output_format(),
            ]
        )

        return "\n".join(prompt_parts)

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
        formatted = []
        for key, value in runtime_info.items():
            formatted.append(f"- {key}: {value}")
        return "\n".join(formatted)

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

        # Ensure required fields are present
        output.setdefault("issues_found", [])
        output.setdefault("code_analysis", {})
        output.setdefault("debugged_code", request.get("code", ""))
        output.setdefault("debugging_metadata", {})

        # Add debugging metadata
        output["debugging_metadata"].update(
            {
                "agent_type": "debugger",
                "request_id": request.get("request_id", "unknown"),
                "original_code_length": len(request.get("code", "")),
                "error_message": request.get("error_message", ""),
                "error_type": request.get("error_type", ""),
            }
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
        request = {
            "code": code,
            "error_message": error_message,
            "error_type": "runtime_error" if error_message else "static_analysis",
            "request_id": f"debug_{int(time.time())}",
            "timestamp": time.time(),
        }

        # This is a synchronous wrapper for the async execute method
        import asyncio

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        response = loop.run_until_complete(self.execute(request))
        return response.output if response.success else {}

    def get_code_issues(self, code: str) -> List[Dict[str, Any]]:
        """
        Get list of issues found in code.

        Args:
            code: Code to analyze

        Returns:
            List of issues found
        """
        analysis = self.analyze_code_issues(code)
        return analysis.get("issues_found", [])

    def get_debugged_code(self, code: str, error_message: str = "") -> str:
        """
        Get debugged version of code.

        Args:
            code: Code to debug
            error_message: Optional error message

        Returns:
            Debugged code
        """
        analysis = self.analyze_code_issues(code, error_message)
        return analysis.get("debugged_code", code)

    def get_code_health_score(self, code: str) -> Dict[str, Any]:
        """
        Get code health score and analysis.

        Args:
            code: Code to analyze

        Returns:
            Code health score and analysis
        """
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
