"""Debugger agent for code analysis and error detection."""

import logging
import time
from typing import Any, Dict, List

from pinocchio.config import ConfigManager
from pinocchio.llm.custom_llm_client import CustomLLMClient

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
        logger.info("DebuggerAgent initialized with its own LLM client")

    async def execute(self, request: Dict[str, Any]) -> AgentResponse:
        """
        Execute code debugging task.

        Args:
            request: Debugging request containing code and error information

        Returns:
            Agent response with debugging analysis
        """
        request_id = request.get("request_id", "unknown")

        # Log debugger execution start with detailed input analysis
        self.verbose_logger.log_agent_activity(
            self.agent_type,
            "Debugger execution started",
            data={
                "request_id": request_id,
                "request_keys": list(request.keys()),
                "request_size": len(str(request)),
                "code_length": len(request.get("code", "")),
                "has_error_message": "error_message" in request,
                "has_context": "context" in request,
                "has_detailed_instruction": "detailed_instruction" in request,
                "error_message_length": len(request.get("error_message", "")),
                "context_size": len(str(request.get("context", {}))),
            },
            session_id=getattr(self, "session_id", None),
            step_id=request.get("step_id"),
        )

        try:
            # Build prompt for code debugging with detailed logging
            self.verbose_logger.log_agent_activity(
                self.agent_type,
                "Building debugging prompt",
                data={"request_id": request_id},
                session_id=getattr(self, "session_id", None),
                step_id=request.get("step_id"),
            )

            prompt = self._build_debugging_prompt(request)

            # Log prompt details
            self.verbose_logger.log_agent_activity(
                self.agent_type,
                "Debugging prompt built",
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
                "Calling LLM for code debugging",
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
                "LLM response received for debugging",
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
                "Processing debugging response",
                data={"request_id": request_id},
                session_id=getattr(self, "session_id", None),
                step_id=request.get("step_id"),
            )

            output = self._process_debugging_response(llm_response, request)

            # Log processed output details
            self.verbose_logger.log_agent_activity(
                self.agent_type,
                "Debugging response processed",
                data={
                    "request_id": request_id,
                    "output_keys": list(output.keys())
                    if isinstance(output, dict)
                    else [],
                    "output_size": len(str(output)),
                    "has_issues_found": "issues_found" in output,
                    "has_code_analysis": "code_analysis" in output,
                    "has_debugged_code": "debugged_code" in output,
                    "issues_count": len(output.get("issues_found", [])),
                    "debugged_code_length": len(output.get("debugged_code", "")),
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
                    "issues_count": len(response.output.get("issues_found", [])),
                    "debugged_code_length": len(
                        response.output.get("debugged_code", "")
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
                "request_size": len(str(request)),
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
                "code_length": len(code),
                "error_message_length": len(error_message),
                "context_size": len(str(context)),
                "detailed_instruction_length": len(detailed_instruction),
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
                data={"error_message_length": len(error_message)},
                session_id=getattr(self, "session_id", None),
                step_id=request.get("step_id"),
            )

        # Add detailed instruction if available
        if detailed_instruction:
            prompt_parts.extend(["", "Detailed Instructions:", detailed_instruction])
            self.verbose_logger.log_agent_activity(
                self.agent_type,
                "Added detailed instructions to debugging prompt",
                data={"detailed_instruction_length": len(detailed_instruction)},
                session_id=getattr(self, "session_id", None),
                step_id=request.get("step_id"),
            )

        if context:
            prompt_parts.extend(["", "Context:", str(context)])
            self.verbose_logger.log_agent_activity(
                self.agent_type,
                "Added context to debugging prompt",
                data={"context_size": len(str(context))},
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
            "Debugging prompt built successfully",
            data={
                "final_prompt_length": len(final_prompt),
                "prompt_parts_count": len(prompt_parts),
                "has_error_message": bool(error_message),
                "has_detailed_instruction": bool(detailed_instruction),
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
        """Get Debugger-specific instructions."""
        return """
Instructions for Code Debugging:
1. Analyze the provided Choreo DSL code for potential errors and issues
2. Identify problems in:
   - Syntax errors and compilation issues
   - Logic errors and algorithmic problems
   - Memory access violations and bounds checking
   - Type mismatches and data flow issues
   - Performance bottlenecks that could cause runtime errors
   - Resource management issues
3. Provide specific, actionable debugging suggestions
4. Suggest fixes and improvements for each identified issue
5. Consider both static analysis and potential runtime issues
6. Ensure suggestions maintain code correctness and performance

Focus on:
- Syntax and compilation errors
- Logic and algorithmic errors
- Memory and resource management
- Type safety and data validation
- Performance issues that could cause failures
- Best practices and code quality
"""

    def _get_debugging_output_format(self) -> str:
        """Get output format for debugging response."""
        return """
Please provide your response in JSON format with the following structure:
{
    "agent_type": "debugger",
    "success": true,
    "output": {
        "issues_found": [
            {
                "type": "syntax_error|logic_error|memory_error|performance_issue|best_practice",
                "severity": "critical|high|medium|low",
                "line_number": "approximate_line_number",
                "description": "Detailed description of the issue",
                "root_cause": "Analysis of what caused the issue",
                "suggested_fix": "Specific code changes or fixes",
                "prevention_tips": "How to prevent similar issues"
            }
        ],
        "code_analysis": {
            "overall_health": "excellent|good|fair|poor",
            "critical_issues_count": "number",
            "total_issues_count": "number",
            "suggested_improvements": ["improvement1", "improvement2"]
        },
        "debugged_code": "Corrected version of the code",
        "debugging_metadata": {
            "analysis_time": "timestamp",
            "issues_count": "number_of_issues",
            "primary_issue_type": "syntax|logic|memory|performance"
        }
    },
    "error_message": null
}
"""

    def _format_runtime_info(self, runtime_info: Dict[str, Any]) -> str:
        """Format runtime information for prompt."""
        # Log runtime info formatting
        self.verbose_logger.log_agent_activity(
            self.agent_type,
            "Formatting runtime information",
            data={
                "runtime_info_count": len(runtime_info),
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
                "formatted_length": len(result),
                "formatted_lines": len(formatted),
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
                    "issues_count": len(output.get("issues_found", [])),
                    "debugged_code_length": len(output.get("debugged_code", "")),
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
                "output_size": len(str(output)),
                "has_issues_found": "issues_found" in output,
                "has_code_analysis": "code_analysis" in output,
                "has_debugged_code": "debugged_code" in output,
                "has_debugging_metadata": "debugging_metadata" in output,
                "issues_count": len(output.get("issues_found", [])),
                "debugged_code_length": len(output.get("debugged_code", "")),
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
                "code_length": len(code),
                "error_message_length": len(error_message),
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
                "issues_count": len(output.get("issues_found", [])),
                "code_length": len(code),
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
            data={"code_length": len(code)},
            session_id=getattr(self, "session_id", None),
        )

        analysis = self.analyze_code_issues(code)
        issues = analysis.get("issues_found", [])

        # Log issues retrieval
        self.verbose_logger.log_agent_activity(
            self.agent_type,
            "Code issues retrieved",
            data={"issues_count": len(issues)},
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
                "code_length": len(code),
                "error_message_length": len(error_message),
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
                "original_code_length": len(code),
                "debugged_code_length": len(debugged_code),
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
            data={"code_length": len(code)},
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
