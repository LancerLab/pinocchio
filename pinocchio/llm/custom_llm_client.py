"""Custom LLM client for local network deployment."""

import json
import logging
from typing import Any, Dict, Optional

import aiohttp

from ..config.models import LLMConfig
from .base_client import BaseLLMClient

logger = logging.getLogger(__name__)


class CustomLLMClient(BaseLLMClient):
    """Custom LLM client for local network deployment."""

    def __init__(self, config: LLMConfig):
        """
        Initialize Custom LLM client.

        Args:
            config: LLM configuration object
        """
        super().__init__()
        self.config = config
        self.base_url = config.base_url.rstrip("/")
        self.model_name = config.model_name
        self.timeout = config.timeout
        self.max_retries = config.max_retries
        self.api_key = config.api_key
        self.headers = config.headers or {}
        self.session = None

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
        max_tokens: int = 2048,
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

        logger.debug(f"Making request to {url}")
        logger.debug(f"Payload: {json.dumps(payload, indent=2)}")

        async with session.post(url, json=payload, headers=headers) as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(f"LLM API error {response.status}: {error_text}")

            result = await response.json()
            logger.debug(f"Response: {json.dumps(result, indent=2)}")
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
                try:
                    json.loads(content)
                    return content
                except json.JSONDecodeError:
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
