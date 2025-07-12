"""Debugger agent for code analysis and error detection."""

import logging
import time
from typing import Any, Dict, List

from ..data_models.agent import AgentResponse
from .base import AgentWithRetry

logger = logging.getLogger(__name__)


class DebuggerAgent(AgentWithRetry):
    """Agent responsible for code analysis and error detection."""

    def __init__(self, llm_client: Any, max_retries: int = 3):
        """
        Initialize Debugger agent.

        Args:
            llm_client: LLM client instance
            max_retries: Maximum retry attempts for LLM calls
        """
        super().__init__("debugger", llm_client, max_retries)
        logger.info("DebuggerAgent initialized")

    async def execute(self, request: Dict[str, Any]) -> AgentResponse:
        """
        Execute code debugging task.

        Args:
            request: Debugging request containing code and error information

        Returns:
            Agent response with debugging analysis
        """
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
        # Extract output from LLM response
        if llm_response.get("success") and "output" in llm_response:
            output = llm_response["output"]
        else:
            # Fallback: create basic debugging structure
            output = {
                "issues_found": [
                    {
                        "type": "general_issue",
                        "severity": "medium",
                        "line_number": "unknown",
                        "description": "Basic code analysis performed",
                        "root_cause": "Analysis not available",
                        "suggested_fix": "// Debugging suggestions not available",
                        "prevention_tips": "Follow best practices and test thoroughly",
                    }
                ],
                "code_analysis": {
                    "overall_health": "unknown",
                    "critical_issues_count": 0,
                    "total_issues_count": 1,
                    "suggested_improvements": ["Perform thorough testing"],
                },
                "debugged_code": request.get("code", "// No debugged code available"),
                "debugging_metadata": {
                    "analysis_time": request.get("timestamp", ""),
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
        Analyze code for issues and provide debugging suggestions.

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
        Get list of issues found in the code.

        Args:
            code: Code to analyze

        Returns:
            List of identified issues
        """
        analysis = self.analyze_code_issues(code)
        return analysis.get("issues_found", [])

    def get_debugged_code(self, code: str, error_message: str = "") -> str:
        """
        Get debugged version of the code.

        Args:
            code: Original code
            error_message: Optional error message

        Returns:
            Debugged code
        """
        analysis = self.analyze_code_issues(code, error_message)
        return analysis.get("debugged_code", code)

    def get_code_health_score(self, code: str) -> Dict[str, Any]:
        """
        Get overall code health assessment.

        Args:
            code: Code to assess

        Returns:
            Code health analysis
        """
        analysis = self.analyze_code_issues(code)
        code_analysis = analysis.get("code_analysis", {})
        # Ensure required fields are present
        code_analysis.setdefault("overall_health", "unknown")
        code_analysis.setdefault("critical_issues_count", 0)
        code_analysis.setdefault("total_issues_count", 0)
        code_analysis.setdefault("suggested_improvements", [])
        return code_analysis
