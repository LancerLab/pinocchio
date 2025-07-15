"""Base agent interface and abstract classes."""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from ..data_models.agent import AgentResponse
from ..utils.json_parser import parse_structured_output, validate_agent_response
from ..utils.verbose_logger import LogLevel, get_verbose_logger

logger = logging.getLogger(__name__)


class Agent(ABC):
    """Base class for all agents in Pinocchio system."""

    def __init__(self, agent_type: str, llm_client: Any):
        """
        Initialize agent.

        Args:
            agent_type: Type identifier for this agent
            llm_client: LLM client instance for making API calls
        """
        self.agent_type = agent_type
        self.llm_client = llm_client
        self.call_count = 0
        self.total_processing_time = 0.0

        # Get verbose logger for detailed tracing
        self.verbose_logger = get_verbose_logger()

        # Log agent initialization
        self.verbose_logger.log_agent_activity(
            self.agent_type,
            "Agent initialized",
            data={
                "agent_type": agent_type,
                "llm_client_type": type(llm_client).__name__,
                "call_count": self.call_count,
                "total_processing_time": self.total_processing_time,
            },
        )

        logger.info("Agent initialized: %s", agent_type)

    async def execute(self, request: Dict[str, Any]) -> AgentResponse:
        """
        Execute agent with given request.

        Args:
            request: Agent request data

        Returns:
            Agent response
        """
        start_time = time.time()
        request_id = request.get("request_id", f"{self.agent_type}_{int(time.time())}")

        # Log verbose agent execution start with detailed input analysis
        self.verbose_logger.log_agent_activity(
            self.agent_type,
            "Execution started",
            data={
                "request_id": request_id,
                "request_keys": list(request.keys()),
                "request_size": len(str(request)),
                "task_description": request.get("task_description", ""),
                "has_code": "code" in request,
                "has_context": "context" in request,
                "has_requirements": "requirements" in request,
                "has_optimization_goals": "optimization_goals" in request,
                "has_performance_metrics": "performance_metrics" in request,
                "has_error_message": "error_message" in request,
            },
            session_id=getattr(self, "session_id", None),
            step_id=request.get("step_id"),
        )

        try:
            # Build prompt with detailed logging
            self.verbose_logger.log_agent_activity(
                self.agent_type,
                "Building prompt",
                data={"request_id": request_id},
                session_id=getattr(self, "session_id", None),
                step_id=request.get("step_id"),
            )

            prompt = self._build_prompt(request)

            # Log prompt details
            self.verbose_logger.log_agent_activity(
                self.agent_type,
                "Prompt built",
                data={
                    "request_id": request_id,
                    "prompt_length": len(prompt),
                    "prompt_preview": prompt[:200] + "..."
                    if len(prompt) > 200
                    else prompt,
                },
                session_id=getattr(self, "session_id", None),
                step_id=request.get("step_id"),
            )

            # Call LLM with detailed logging
            self.verbose_logger.log_agent_activity(
                self.agent_type,
                "Calling LLM",
                data={
                    "request_id": request_id,
                    "prompt_length": len(prompt),
                    "call_count_before": self.call_count,
                },
                session_id=getattr(self, "session_id", None),
                step_id=request.get("step_id"),
            )

            llm_response = await self._call_llm(prompt)

            # Extract response data with detailed logging
            self.verbose_logger.log_agent_activity(
                self.agent_type,
                "Processing LLM response",
                data={
                    "request_id": request_id,
                    "llm_response_keys": list(llm_response.keys())
                    if isinstance(llm_response, dict)
                    else [],
                    "llm_response_type": type(llm_response).__name__,
                },
                session_id=getattr(self, "session_id", None),
                step_id=request.get("step_id"),
            )

            success = llm_response.get("success", False)
            output = llm_response.get("output", {})
            error_message = llm_response.get("error_message")

            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000

            # Create response with detailed logging
            self.verbose_logger.log_agent_activity(
                self.agent_type,
                "Creating response",
                data={
                    "request_id": request_id,
                    "success": success,
                    "output_keys": list(output.keys())
                    if isinstance(output, dict)
                    else [],
                    "error_message": error_message,
                    "processing_time_ms": processing_time,
                },
                session_id=getattr(self, "session_id", None),
                step_id=request.get("step_id"),
            )

            response = self._create_response(
                request_id=request_id,
                success=success,
                output=output,
                error_message=error_message,
                processing_time_ms=processing_time,
            )

            # Log verbose agent execution completion with detailed output analysis
            self.verbose_logger.log_agent_activity(
                self.agent_type,
                "Execution completed",
                data={
                    "request_id": request_id,
                    "success": success,
                    "output_keys": list(output.keys())
                    if isinstance(output, dict)
                    else [],
                    "output_size": len(str(output)),
                    "processing_time_ms": processing_time,
                    "call_count_after": self.call_count,
                    "total_processing_time_after": self.total_processing_time,
                    "average_processing_time": self.get_average_processing_time(),
                },
                session_id=getattr(self, "session_id", None),
                step_id=request.get("step_id"),
                duration_ms=processing_time,
            )

            return response

        except Exception as e:
            processing_time = (time.time() - start_time) * 1000

            # Log verbose agent execution error with detailed error information
            self.verbose_logger.log(
                LogLevel.ERROR,
                f"agent:{self.agent_type}",
                "Agent execution failed",
                data={
                    "request_id": request_id,
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "error_details": getattr(e, "__dict__", {}),
                    "request_keys": list(request.keys()),
                    "processing_time_ms": processing_time,
                    "call_count": self.call_count,
                },
                session_id=getattr(self, "session_id", None),
                step_id=request.get("step_id"),
                duration_ms=processing_time,
            )

            return self._handle_error(request_id, e)

    async def _call_llm(self, prompt: str) -> Dict[str, Any]:
        """
        Call LLM with prompt and parse response.

        Args:
            prompt: Prompt string to send to LLM

        Returns:
            Parsed LLM response

        Raises:
            Exception: If LLM call fails
        """
        start_time = time.time()

        try:
            # Log verbose LLM call start with detailed request analysis
            self.verbose_logger.log_llm_activity(
                f"LLM call started for {self.agent_type}",
                request_data={
                    "agent_type": self.agent_type,
                    "prompt_length": len(prompt),
                    "prompt_preview": prompt[:100] + "..."
                    if len(prompt) > 100
                    else prompt,
                    "call_count_before": self.call_count,
                    "total_processing_time_before": self.total_processing_time,
                },
                session_id=getattr(self, "session_id", None),
            )

            # Call LLM
            self.verbose_logger.log_llm_activity(
                f"Making LLM API call for {self.agent_type}",
                request_data={
                    "agent_type": self.agent_type,
                    "llm_client_type": type(self.llm_client).__name__,
                },
                session_id=getattr(self, "session_id", None),
            )

            response = await self.llm_client.complete(
                prompt, agent_type=self.agent_type
            )

            # Log raw response details
            self.verbose_logger.log_llm_activity(
                f"Raw LLM response received for {self.agent_type}",
                response_data={
                    "response_type": type(response).__name__,
                    "response_length": len(str(response)),
                    "response_preview": str(response)[:200] + "..."
                    if len(str(response)) > 200
                    else str(response),
                },
                session_id=getattr(self, "session_id", None),
            )

            # Parse response with detailed logging
            self.verbose_logger.log_llm_activity(
                f"Parsing LLM response for {self.agent_type}",
                request_data={
                    "agent_type": self.agent_type,
                    "raw_response_length": len(str(response)),
                },
                session_id=getattr(self, "session_id", None),
            )

            parsed_response = parse_structured_output(response)

            # Log parsing results
            self.verbose_logger.log_llm_activity(
                f"LLM response parsed for {self.agent_type}",
                response_data={
                    "parsed_response_type": type(parsed_response).__name__,
                    "parsed_response_keys": list(parsed_response.keys())
                    if isinstance(parsed_response, dict)
                    else [],
                    "parsed_response_size": len(str(parsed_response)),
                },
                session_id=getattr(self, "session_id", None),
            )

            # Validate response structure with detailed logging
            self.verbose_logger.log_llm_activity(
                f"Validating LLM response for {self.agent_type}",
                request_data={
                    "agent_type": self.agent_type,
                    "parsed_response_keys": list(parsed_response.keys())
                    if isinstance(parsed_response, dict)
                    else [],
                },
                session_id=getattr(self, "session_id", None),
            )

            validation_result = validate_agent_response(parsed_response)

            self.verbose_logger.log_llm_activity(
                f"LLM response validation completed for {self.agent_type}",
                response_data={
                    "validation_passed": validation_result,
                    "parsed_response_keys": list(parsed_response.keys())
                    if isinstance(parsed_response, dict)
                    else [],
                },
                session_id=getattr(self, "session_id", None),
            )

            if not validation_result:
                logger.warning(
                    f"Invalid agent response structure for {self.agent_type}"
                )
                self.verbose_logger.log(
                    LogLevel.WARNING,
                    f"llm:{self.agent_type}",
                    "Invalid agent response structure",
                    data={
                        "agent_type": self.agent_type,
                        "parsed_response": parsed_response,
                    },
                    session_id=getattr(self, "session_id", None),
                )

            # Update statistics with detailed logging
            processing_time = (time.time() - start_time) * 1000  # Convert to ms
            old_call_count = self.call_count
            old_total_time = self.total_processing_time

            self.call_count += 1
            self.total_processing_time += processing_time

            self.verbose_logger.log_llm_activity(
                f"LLM call statistics updated for {self.agent_type}",
                response_data={
                    "processing_time_ms": processing_time,
                    "call_count_before": old_call_count,
                    "call_count_after": self.call_count,
                    "total_processing_time_before": old_total_time,
                    "total_processing_time_after": self.total_processing_time,
                    "average_processing_time": self.get_average_processing_time(),
                },
                session_id=getattr(self, "session_id", None),
                duration_ms=processing_time,
            )

            logger.debug(
                f"LLM call completed for {self.agent_type} in {processing_time:.2f}ms"
            )

            # Log verbose LLM call completion with comprehensive details
            self.verbose_logger.log_llm_activity(
                f"LLM call completed for {self.agent_type}",
                response_data={
                    "response_length": len(str(response)),
                    "parsed_response_keys": list(parsed_response.keys())
                    if isinstance(parsed_response, dict)
                    else [],
                    "validation_passed": validation_result,
                    "processing_time_ms": processing_time,
                    "call_count": self.call_count,
                    "total_processing_time": self.total_processing_time,
                    "average_processing_time": self.get_average_processing_time(),
                },
                session_id=getattr(self, "session_id", None),
                duration_ms=processing_time,
            )

            return parsed_response

        except Exception as e:
            processing_time = (time.time() - start_time) * 1000

            # Log verbose LLM call error with detailed error information
            self.verbose_logger.log(
                LogLevel.ERROR,
                "llm",
                f"LLM call failed for {self.agent_type}",
                data={
                    "agent_type": self.agent_type,
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "error_details": getattr(e, "__dict__", {}),
                    "prompt_length": len(prompt),
                    "processing_time_ms": processing_time,
                    "call_count": self.call_count,
                },
                session_id=getattr(self, "session_id", None),
                duration_ms=processing_time,
            )

            raise

    def _build_prompt(self, request: Dict[str, Any]) -> str:
        """
        Build prompt from request data.

        Args:
            request: Agent request data

        Returns:
            Formatted prompt string
        """
        # Log prompt building start
        self.verbose_logger.log_agent_activity(
            self.agent_type,
            "Building prompt from request",
            data={
                "request_keys": list(request.keys()),
                "request_size": len(str(request)),
                "has_task_description": "task_description" in request,
                "has_code": "code" in request,
                "has_context": "context" in request,
            },
            session_id=getattr(self, "session_id", None),
            step_id=request.get("step_id"),
        )

        # Get agent-specific instructions
        instructions = self._get_agent_instructions()

        # Log instructions details
        self.verbose_logger.log_agent_activity(
            self.agent_type,
            "Agent instructions retrieved",
            data={
                "instructions_length": len(instructions),
                "instructions_preview": instructions[:100] + "..."
                if len(instructions) > 100
                else instructions,
            },
            session_id=getattr(self, "session_id", None),
            step_id=request.get("step_id"),
        )

        # Get output format
        output_format = self._get_output_format()

        # Log output format details
        self.verbose_logger.log_agent_activity(
            self.agent_type,
            "Output format retrieved",
            data={
                "output_format_length": len(output_format),
                "output_format_preview": output_format[:100] + "..."
                if len(output_format) > 100
                else output_format,
            },
            session_id=getattr(self, "session_id", None),
            step_id=request.get("step_id"),
        )

        # Build prompt components
        task_description = request.get("task_description", "")
        code = request.get("code", "")
        context = request.get("context", {})

        prompt_parts = [
            f"You are a {self.agent_type} agent in the Pinocchio multi-agent system.",
            "",
            f"Task Description: {task_description}",
        ]

        if code:
            prompt_parts.extend(["", "Code:", "```choreo", code, "```"])

        if context:
            prompt_parts.extend(["", "Context:", str(context)])

        prompt_parts.extend(["", instructions, "", output_format])

        final_prompt = "\n".join(prompt_parts)

        # Log final prompt details
        self.verbose_logger.log_agent_activity(
            self.agent_type,
            "Prompt built successfully",
            data={
                "final_prompt_length": len(final_prompt),
                "prompt_parts_count": len(prompt_parts),
                "has_code": bool(code),
                "has_context": bool(context),
                "prompt_preview": final_prompt[:200] + "..."
                if len(final_prompt) > 200
                else final_prompt,
            },
            session_id=getattr(self, "session_id", None),
            step_id=request.get("step_id"),
        )

        return final_prompt

    @abstractmethod
    def _get_agent_instructions(self) -> str:
        """Get agent-specific instructions."""
        pass

    def _get_output_format(self) -> str:
        """Get output format for agent response."""
        return """
Please provide your response in JSON format with the following structure:
{
    "agent_type": "agent_type",
    "success": true,
    "output": {
        // Agent-specific output fields
    },
    "error_message": null
}
"""

    def _create_response(
        self,
        request_id: str,
        success: bool,
        output: Dict[str, Any],
        error_message: Optional[str] = None,
        processing_time_ms: Optional[int] = None,
    ) -> AgentResponse:
        """
        Create agent response object.

        Args:
            request_id: Request identifier
            success: Whether operation was successful
            output: Response output data
            error_message: Error message if any
            processing_time_ms: Processing time in milliseconds

        Returns:
            Agent response object
        """
        # Log response creation start
        self.verbose_logger.log_agent_activity(
            self.agent_type,
            "Creating agent response",
            data={
                "request_id": request_id,
                "success": success,
                "output_keys": list(output.keys()) if isinstance(output, dict) else [],
                "output_size": len(str(output)),
                "error_message": error_message,
                "processing_time_ms": processing_time_ms,
            },
            session_id=getattr(self, "session_id", None),
        )

        response = AgentResponse(
            request_id=request_id,
            agent_type=self.agent_type,
            success=success,
            output=output,
            error_message=error_message,
            processing_time_ms=processing_time_ms,
        )

        # Log response creation completion
        self.verbose_logger.log_agent_activity(
            self.agent_type,
            "Agent response created",
            data={
                "request_id": request_id,
                "response_type": type(response).__name__,
                "response_success": response.success,
                "response_output_keys": list(response.output.keys())
                if isinstance(response.output, dict)
                else [],
                "response_error_message": response.error_message,
                "response_processing_time_ms": response.processing_time_ms,
            },
            session_id=getattr(self, "session_id", None),
        )

        return response

    def _handle_error(self, request_id: str, error: Exception) -> AgentResponse:
        """
        Handle execution error and create error response.

        Args:
            request_id: Request identifier
            error: Exception that occurred

        Returns:
            Error response object
        """
        # Log error handling start
        self.verbose_logger.log(
            LogLevel.ERROR,
            f"agent:{self.agent_type}",
            "Handling execution error",
            data={
                "request_id": request_id,
                "error_type": type(error).__name__,
                "error_message": str(error),
                "error_details": getattr(error, "__dict__", {}),
            },
            session_id=getattr(self, "session_id", None),
        )

        error_response = AgentResponse(
            request_id=request_id,
            agent_type=self.agent_type,
            success=False,
            output={},
            error_message=str(error),
            processing_time_ms=None,
        )

        # Log error response creation
        self.verbose_logger.log(
            LogLevel.ERROR,
            f"agent:{self.agent_type}",
            "Error response created",
            data={
                "request_id": request_id,
                "error_response_type": type(error_response).__name__,
                "error_response_success": error_response.success,
                "error_response_error_message": error_response.error_message,
            },
            session_id=getattr(self, "session_id", None),
        )

        return error_response

    def get_stats(self) -> Dict[str, Any]:
        """
        Get agent statistics.

        Returns:
            Dictionary containing agent statistics
        """
        stats = {
            "agent_type": self.agent_type,
            "call_count": self.call_count,
            "total_processing_time": self.total_processing_time,
            "average_processing_time": self.get_average_processing_time(),
        }

        # Log stats retrieval
        self.verbose_logger.log_agent_activity(
            self.agent_type,
            "Statistics retrieved",
            data=stats,
            session_id=getattr(self, "session_id", None),
        )

        return stats

    def get_average_processing_time(self) -> float:
        """
        Get average processing time per call.

        Returns:
            Average processing time in milliseconds
        """
        if self.call_count == 0:
            return 0.0
        return self.total_processing_time / self.call_count

    def reset_stats(self) -> None:
        """Reset agent statistics."""
        old_stats = self.get_stats()

        self.call_count = 0
        self.total_processing_time = 0.0

        # Log stats reset
        self.verbose_logger.log_agent_activity(
            self.agent_type,
            "Statistics reset",
            data={
                "old_stats": old_stats,
                "new_call_count": self.call_count,
                "new_total_processing_time": self.total_processing_time,
            },
            session_id=getattr(self, "session_id", None),
        )


class AgentWithRetry(Agent):
    """Agent with retry capability for LLM calls."""

    def __init__(
        self,
        agent_type: str,
        llm_client: Any,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        """
        Initialize agent with retry capability.

        Args:
            agent_type: Type identifier for this agent
            llm_client: LLM client instance
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds
        """
        super().__init__(agent_type, llm_client)
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # Log retry configuration
        self.verbose_logger.log_agent_activity(
            self.agent_type,
            "Retry configuration set",
            data={
                "max_retries": max_retries,
                "retry_delay": retry_delay,
            },
            session_id=getattr(self, "session_id", None),
        )

    async def _call_llm_with_retry(self, prompt: str) -> Dict[str, Any]:
        """
        Call LLM with retry logic.

        Args:
            prompt: Prompt string to send to LLM

        Returns:
            Parsed LLM response

        Raises:
            Exception: If all retry attempts fail
        """
        last_exception = None

        # Log retry attempt start
        self.verbose_logger.log_agent_activity(
            self.agent_type,
            "Starting LLM call with retry",
            data={
                "max_retries": self.max_retries,
                "retry_delay": self.retry_delay,
                "prompt_length": len(prompt),
            },
            session_id=getattr(self, "session_id", None),
        )

        for attempt in range(self.max_retries + 1):
            try:
                # Log attempt details
                self.verbose_logger.log_agent_activity(
                    self.agent_type,
                    f"LLM call attempt {attempt + 1}/{self.max_retries + 1}",
                    data={
                        "attempt_number": attempt + 1,
                        "max_attempts": self.max_retries + 1,
                        "prompt_length": len(prompt),
                    },
                    session_id=getattr(self, "session_id", None),
                )

                return await self._call_llm(prompt)

            except Exception as e:
                last_exception = e

                # Log attempt failure
                self.verbose_logger.log(
                    LogLevel.WARNING,
                    f"agent:{self.agent_type}",
                    f"LLM call attempt {attempt + 1} failed",
                    data={
                        "attempt_number": attempt + 1,
                        "max_attempts": self.max_retries + 1,
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                        "retries_remaining": self.max_retries - attempt,
                    },
                    session_id=getattr(self, "session_id", None),
                )

                if attempt < self.max_retries:
                    # Log retry delay
                    self.verbose_logger.log_agent_activity(
                        self.agent_type,
                        f"Waiting before retry attempt {attempt + 2}",
                        data={
                            "retry_delay": self.retry_delay,
                            "next_attempt": attempt + 2,
                        },
                        session_id=getattr(self, "session_id", None),
                    )

                    await asyncio.sleep(self.retry_delay)

        # Log final failure
        self.verbose_logger.log(
            LogLevel.ERROR,
            f"agent:{self.agent_type}",
            "All LLM call retry attempts failed",
            data={
                "max_retries": self.max_retries,
                "total_attempts": self.max_retries + 1,
                "final_error_type": type(last_exception).__name__,
                "final_error_message": str(last_exception),
            },
            session_id=getattr(self, "session_id", None),
        )

        raise last_exception
