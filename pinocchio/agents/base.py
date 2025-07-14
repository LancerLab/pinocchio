"""Base agent interface and abstract classes."""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from ..data_models.agent import AgentResponse
from ..utils.json_parser import parse_structured_output, validate_agent_response

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

        logger.info(f"Agent initialized: {agent_type}")

    @abstractmethod
    async def execute(self, request: Dict[str, Any]) -> AgentResponse:
        """
        Execute agent with given request.

        Args:
            request: Agent request data

        Returns:
            Agent response
        """
        pass

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
            # Call LLM
            response = await self.llm_client.complete(
                prompt, agent_type=self.agent_type
            )

            # Parse response
            parsed_response = parse_structured_output(response)

            # Validate response structure
            if not validate_agent_response(parsed_response):
                logger.warning(
                    f"Invalid agent response structure for {self.agent_type}"
                )

            # Update statistics
            processing_time = (time.time() - start_time) * 1000  # Convert to ms
            self.call_count += 1
            self.total_processing_time += processing_time

            logger.debug(
                f"LLM call completed for {self.agent_type} in {processing_time:.2f}ms"
            )

            return parsed_response

        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            logger.error(f"LLM call failed for {self.agent_type}: {e}")
            raise Exception(f"LLM call failed: {str(e)}")

    def _build_prompt(self, request: Dict[str, Any]) -> str:
        """
        Build prompt string from request data.

        Args:
            request: Agent request data

        Returns:
            Formatted prompt string
        """
        # Extract basic information
        task_description = request.get("task_description", "")
        context = request.get("context", {})

        # Build basic prompt
        prompt_parts = [
            f"You are a {self.agent_type} agent in the Pinocchio multi-agent system.",
            f"Task: {task_description}",
        ]

        # Add context if available
        if context:
            prompt_parts.append(f"Context: {context}")

        # Add agent-specific instructions
        prompt_parts.append(self._get_agent_instructions())

        # Add output format requirements
        prompt_parts.append(self._get_output_format())

        return "\n\n".join(prompt_parts)

    @abstractmethod
    def _get_agent_instructions(self) -> str:
        """
        Get agent-specific instructions for the prompt.

        Returns:
            Agent-specific instruction string
        """
        pass

    def _get_output_format(self) -> str:
        """
        Get output format requirements.

        Returns:
            Output format instruction string
        """
        return f"""
Please provide your response in JSON format with the following structure:
{{
    "agent_type": "{self.agent_type}",
    "success": true/false,
    "output": {{
        // Agent-specific output fields
    }},
    "error_message": "error description if success is false"
}}
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
        Create standardized agent response.

        Args:
            request_id: ID of the request being responded to
            success: Whether execution was successful
            output: Agent output data
            error_message: Error message if execution failed
            processing_time_ms: Processing time in milliseconds

        Returns:
            Standardized agent response
        """
        return AgentResponse(
            agent_type=self.agent_type,
            success=success,
            output=output,
            error_message=error_message,
            request_id=request_id,
            processing_time_ms=processing_time_ms,
            metadata={
                "call_count": self.call_count,
                "average_processing_time": self.get_average_processing_time(),
            },
        )

    def _handle_error(self, request_id: str, error: Exception) -> AgentResponse:
        """
        Handle and format error responses.

        Args:
            request_id: ID of the request that failed
            error: Exception that occurred

        Returns:
            Error response
        """
        error_message = str(error)
        logger.error(f"Agent {self.agent_type} error: {error_message}")

        return self._create_response(
            request_id=request_id, success=False, output={}, error_message=error_message
        )

    def get_stats(self) -> Dict[str, Any]:
        """
        Get agent statistics.

        Returns:
            Agent statistics dictionary
        """
        return {
            "agent_type": self.agent_type,
            "call_count": self.call_count,
            "total_processing_time_ms": self.total_processing_time,
            "average_processing_time_ms": self.get_average_processing_time(),
        }

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
        self.call_count = 0
        self.total_processing_time = 0.0
        logger.debug(f"Stats reset for agent: {self.agent_type}")


class AgentWithRetry(Agent):
    """Agent base class with retry capability."""

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
        last_error = None

        for attempt in range(self.max_retries + 1):
            try:
                return await self._call_llm(prompt)
            except Exception as e:
                last_error = e

                if attempt < self.max_retries:
                    logger.warning(
                        f"LLM call attempt {attempt + 1} failed for {self.agent_type}: {e}"
                    )
                    logger.info(f"Retrying in {self.retry_delay} seconds...")
                    await asyncio.sleep(self.retry_delay)
                else:
                    logger.error(
                        f"All {self.max_retries + 1} attempts failed for {self.agent_type}"
                    )

        # If we get here, all attempts failed
        # Return structured failure response with detailed error and suggestion
        logger.error(
            f"LLM call failed after {self.max_retries + 1} attempts: {last_error}"
        )
        return {
            "agent_type": self.agent_type,
            "success": False,
            "output": {},
            "error_message": f"LLM call failed after {self.max_retries + 1} attempts: {last_error}",
            "suggestion": "Please check your input, network, or LLM configuration. If the problem persists, try simplifying your request or contact support.",
            "terminated": True,
        }
