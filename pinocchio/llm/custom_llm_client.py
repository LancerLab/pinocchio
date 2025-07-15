"""Custom LLM client for local network deployment."""

import json
import logging
from typing import Any, Dict, Optional

import aiohttp

from pinocchio.config.config_manager import ConfigManager
from pinocchio.utils.file_utils import get_output_path

from ..config.models import LLMConfigEntry
from ..utils.json_parser import format_json_response, safe_json_parse
from .base_client import BaseLLMClient

config = ConfigManager()
logs_root = config.get_logs_path()

logger = logging.getLogger(__name__)


class CustomLLMClient(BaseLLMClient):
    """Custom LLM client for local network deployment."""

    def __init__(
        self, config: LLMConfigEntry, verbose: bool = False, verbose_callback=None
    ):
        """
        Initialize Custom LLM client.

        Args:
            config: LLM configuration entry object
            verbose: Whether to print verbose output
            verbose_callback: Optional callback for verbose messages (e.g., CLI.add_llm_verbose_message)
        """
        super().__init__(verbose=verbose)
        self.config = config
        self.base_url = config.base_url.rstrip("/") if config.base_url else None
        self.model_name = config.model_name
        self.timeout = config.timeout
        self.max_retries = config.max_retries
        self.api_key = config.api_key
        self.headers = config.headers or {}
        self.session = None
        self.verbose_callback = verbose_callback

        if self.verbose:
            msg = f"[LLM VERBOSE] Selected LLM: provider=custom, model={self.model_name}, base_url={self.base_url}"
            if self.verbose_callback:
                self.verbose_callback(msg)
            else:
                print(msg)

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            self.session = aiohttp.ClientSession(timeout=timeout)
        return self.session

    async def _make_request(
        self,
        prompt: str,
        agent_type: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> Dict[str, Any]:
        """
        Make request to LLM API.

        Args:
            prompt: Input prompt
            agent_type: Optional agent type for context
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            API response as dictionary
        """
        session = await self._get_session()

        # Prepare the request payload
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": self._get_system_prompt(agent_type)},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }

        url = f"{self.base_url}/v1/chat/completions"

        # Prepare headers
        headers = self.headers.copy()
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        if self.verbose:
            msg = f"[LLM VERBOSE] Sending request to {url}"
            if self.verbose_callback:
                self.verbose_callback(msg)
            else:
                print(msg)
            msg = f"[LLM VERBOSE] Payload: {json.dumps(payload, indent=2)}"
            if self.verbose_callback:
                self.verbose_callback(msg)
            else:
                print(msg)

        async with session.post(url, json=payload, headers=headers) as response:
            if self.verbose:
                msg = f"[LLM VERBOSE] Response status: {response.status}"
                if self.verbose_callback:
                    self.verbose_callback(msg)
                else:
                    print(msg)
            if response.status != 200:
                error_text = await response.text()
                if self.verbose:
                    msg = f"[LLM VERBOSE] Error: {error_text}"
                    if self.verbose_callback:
                        self.verbose_callback(msg)
                    else:
                        print(msg)
                raise Exception(f"LLM API error {response.status}: {error_text}")

            result = await response.json()
            if self.verbose:
                msg = f"[LLM VERBOSE] Response: {json.dumps(result, indent=2)}"
                if self.verbose_callback:
                    self.verbose_callback(msg)
                else:
                    print(msg)
            return result

    def _get_system_prompt(self, agent_type: Optional[str] = None) -> str:
        """Get system prompt based on agent type."""
        base_prompt = """You are an expert AI assistant specialized in high-performance computing and code generation.
You excel at creating optimized algorithms, debugging code issues, and providing detailed technical explanations.

Please provide your responses in JSON format with the following structure:
{
    "agent_type": "generator|debugger|optimizer|evaluator",
    "success": true,
    "output": {
        // Agent-specific output fields
    },
    "explanation": "Detailed explanation of your approach",
    "confidence": 0.95
}"""

        if agent_type == "generator":
            base_prompt += """

For code generation tasks:
- Generate clean, efficient code
- Include detailed comments explaining the logic
- Consider performance optimizations
- Provide usage examples
- Output should include: code, language, explanation, optimization_techniques, hyperparameters"""

        elif agent_type == "debugger":
            base_prompt += """

For debugging tasks:
- Analyze code for potential issues
- Provide specific fixes with explanations
- Consider edge cases and error handling
- Output should include: fixed_code, issues_found, fixes_applied, confidence"""

        elif agent_type == "optimizer":
            base_prompt += """

For optimization tasks:
- Analyze performance bottlenecks
- Suggest specific optimizations
- Provide before/after comparisons
- Output should include: optimized_code, optimization_suggestions, expected_improvement, risk_assessment"""

        elif agent_type == "evaluator":
            base_prompt += """

For evaluation tasks:
- Assess code quality and correctness
- Provide performance metrics
- Identify potential improvements
- Output should include: evaluation_results, performance_metrics, test_results, overall_score, recommendations"""

        elif agent_type == "planner":
            base_prompt += """

For task planning and analysis:
- Analyze user requests for code generation and optimization requirements
- Extract primary and secondary goals from the request
- Identify optimization targets and constraints
- Determine appropriate planning strategy
- Output should include: requirements, optimization_goals, constraints, user_preferences, planning_strategy
- IMPORTANT: Always respond with valid JSON format only, no additional text"""

        return base_prompt

    async def complete(self, prompt: str, agent_type: Optional[str] = None) -> str:
        """
        Complete prompt with LLM response.

        Args:
            prompt: Input prompt
            agent_type: Optional agent type for context

        Returns:
            LLM response as JSON string
        """
        self.call_count += 1
        logger.debug(f"CustomLLMClient.complete called (count: {self.call_count})")

        try:
            response = await self._make_request(prompt, agent_type)

            # Extract the content from the response
            if "choices" in response and len(response["choices"]) > 0:
                content = response["choices"][0]["message"]["content"]

                # Try to parse as JSON, if not, wrap it
                parsed_content = safe_json_parse(content)
                if parsed_content is not None:
                    return format_json_response(parsed_content)
                else:
                    # If not valid JSON, create a structured response
                    return self._create_structured_response(content, agent_type)
            else:
                raise Exception("Invalid response format from LLM API")

        except Exception as e:
            logger.error(f"Error in CustomLLMClient.complete: {e}")
            # Return a fallback response
            return self._create_fallback_response(prompt, agent_type, str(e))

    async def complete_structured(
        self, prompt: str, agent_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Complete prompt and return structured response.

        Args:
            prompt: Input prompt
            agent_type: Optional agent type for context

        Returns:
            Structured response as dictionary
        """
        response_json = await self.complete(prompt, agent_type)
        try:
            return json.loads(response_json)
        except json.JSONDecodeError:
            logger.error(f"Failed to parse JSON response: {response_json}")
            return self._create_fallback_structured_response(prompt, agent_type)

    def _create_structured_response(
        self, content: str, agent_type: Optional[str] = None
    ) -> str:
        """Create structured response from raw content."""
        response = {
            "agent_type": agent_type or "generator",
            "success": True,
            "output": {"content": content, "raw_response": content},
            "explanation": "Response from custom LLM model",
            "confidence": 0.8,
        }
        return json.dumps(response, indent=2)

    def _create_fallback_response(
        self, prompt: str, agent_type: Optional[str] = None, error: str = ""
    ) -> str:
        """Create fallback response when API call fails."""
        response = {
            "agent_type": agent_type or "generator",
            "success": False,
            "output": {"error": error, "fallback": True},
            "explanation": f"Fallback response due to error: {error}",
            "confidence": 0.0,
        }
        return json.dumps(response, indent=2)

    def _create_fallback_structured_response(
        self, prompt: str, agent_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create fallback structured response."""
        return {
            "agent_type": agent_type or "generator",
            "success": False,
            "output": {"error": "Failed to parse response", "fallback": True},
            "explanation": "Fallback response due to parsing error",
            "confidence": 0.0,
        }

    async def close(self):
        """Close the client session."""
        if self.session and not self.session.closed:
            await self.session.close()

    def __del__(self):
        """Cleanup on deletion."""
        if self.session and not self.session.closed:
            import asyncio

            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(self.session.close())
            except RuntimeError:
                pass
