"""Mock LLM client for testing and development."""

import asyncio
import json
import logging
import random
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class MockLLMClient:
    """Mock LLM client for testing and development."""

    def __init__(self, response_delay_ms: int = 100, failure_rate: float = 0.0):
        """
        Initialize mock LLM client.

        Args:
            response_delay_ms: Simulated response delay in milliseconds
            failure_rate: Probability of simulated failures (0.0 to 1.0)
        """
        self.response_delay_ms = response_delay_ms
        self.failure_rate = failure_rate
        self.call_count = 0

        # Predefined responses for different agent types
        self.response_templates = {
            "generator": {
                "code": """
// Generated Choreo DSL operator for {task}
func {function_name}(input: tensor, output: tensor) {{
    // Basic implementation
    for i in range(input.shape[0]) {{
        for j in range(input.shape[1]) {{
            output[i][j] = compute(input[i][j]);
        }}
    }}
}}
""".strip(),
                "language": "choreo_dsl",
                "explanation": "Generated a basic Choreo DSL operator with optimized memory access patterns.",
                "optimization_techniques": ["loop_tiling", "memory_coalescing"],
                "hyperparameters": {
                    "tile_size": 32,
                    "unroll_factor": 4,
                    "vectorization": True,
                },
            },
            "debugger": {
                "fixed_code": """
// Fixed Choreo DSL operator
func {function_name}(input: tensor, output: tensor) {{
    // Fixed: Added bounds checking
    for i in range(min(input.shape[0], output.shape[0])) {{
        for j in range(min(input.shape[1], output.shape[1])) {{
            output[i][j] = compute(input[i][j]);
        }}
    }}
}}
""".strip(),
                "issues_found": [
                    "Array bounds not checked",
                    "Potential buffer overflow",
                    "Missing null pointer checks",
                ],
                "fixes_applied": [
                    "Added bounds checking in loops",
                    "Added input validation",
                    "Fixed memory access patterns",
                ],
                "confidence": 0.85,
            },
            "optimizer": {
                "optimized_code": """
// Optimized Choreo DSL operator
func {function_name}_optimized(input: tensor, output: tensor) {{
    // Optimized with tiling and vectorization
    const tile_size = 32;
    for ti in range(0, input.shape[0], tile_size) {{
        for tj in range(0, input.shape[1], tile_size) {{
            for i in range(ti, min(ti + tile_size, input.shape[0])) {{
                #pragma vectorize
                for j in range(tj, min(tj + tile_size, input.shape[1])) {{
                    output[i][j] = compute(input[i][j]);
                }}
            }}
        }}
    }}
}}
""".strip(),
                "optimization_suggestions": [
                    "Apply loop tiling for better cache locality",
                    "Use vectorization for SIMD operations",
                    "Consider memory prefetching",
                    "Optimize data layout for coalesced access",
                ],
                "expected_improvement": {
                    "performance_gain": "2.5x",
                    "memory_efficiency": "40% improvement",
                    "cache_miss_reduction": "60%",
                },
                "risk_assessment": "Low risk - standard optimization techniques",
            },
            "evaluator": {
                "evaluation_results": {
                    "correctness": "PASS",
                    "performance": "GOOD",
                    "memory_usage": "OPTIMAL",
                    "code_quality": "HIGH",
                },
                "performance_metrics": {
                    "execution_time_ms": 45.2,
                    "memory_usage_mb": 128.5,
                    "throughput_ops_per_sec": 15600,
                    "cache_hit_rate": 0.94,
                },
                "test_results": [
                    {
                        "test_name": "basic_functionality",
                        "status": "PASS",
                        "score": 1.0,
                    },
                    {"test_name": "edge_cases", "status": "PASS", "score": 0.95},
                    {
                        "test_name": "performance_benchmark",
                        "status": "PASS",
                        "score": 0.88,
                    },
                    {"test_name": "memory_safety", "status": "PASS", "score": 1.0},
                ],
                "overall_score": 0.92,
                "recommendations": [
                    "Consider further optimization for large datasets",
                    "Add more comprehensive error handling",
                    "Implement parallel execution for multi-core systems",
                ],
            },
        }

    def _should_simulate_failure(self) -> bool:
        """Check if should simulate a failure based on failure rate."""
        return random.random() < self.failure_rate

    async def _simulate_delay(self) -> None:
        """Simulate network/processing delay."""
        if self.response_delay_ms > 0:
            await asyncio.sleep(self.response_delay_ms / 1000.0)

    def _extract_agent_type(self, prompt: str) -> str:
        """Extract agent type from prompt."""
        prompt_lower = prompt.lower()

        if any(
            word in prompt_lower
            for word in ["generate", "create", "write", "implement"]
        ):
            return "generator"
        elif any(word in prompt_lower for word in ["debug", "fix", "error", "bug"]):
            return "debugger"
        elif any(
            word in prompt_lower for word in ["optimize", "improve", "performance"]
        ):
            return "optimizer"
        elif any(
            word in prompt_lower for word in ["evaluate", "test", "assess", "analyze"]
        ):
            return "evaluator"
        else:
            return "generator"  # Default to generator

    def _extract_task_info(self, prompt: str) -> Dict[str, str]:
        """Extract task information from prompt."""
        task_info = {"task": "conv2d operation", "function_name": "conv2d_kernel"}

        # Simple keyword extraction
        if "conv" in prompt.lower():
            task_info["task"] = "convolution operation"
            task_info["function_name"] = "conv_kernel"
        elif "matmul" in prompt.lower() or "matrix" in prompt.lower():
            task_info["task"] = "matrix multiplication"
            task_info["function_name"] = "matmul_kernel"
        elif "add" in prompt.lower():
            task_info["task"] = "element-wise addition"
            task_info["function_name"] = "add_kernel"

        return task_info

    def _generate_response(self, prompt: str, agent_type: str) -> Dict[str, Any]:
        """Generate mock response based on agent type."""
        task_info = self._extract_task_info(prompt)

        if agent_type not in self.response_templates:
            agent_type = "generator"

        template = self.response_templates[agent_type].copy()

        # Format template with task info
        for key, value in template.items():
            if isinstance(value, str) and "{" in value:
                template[key] = value.format(**task_info)

        return {
            "agent_type": agent_type,
            "success": True,
            "output": template,
            "processing_time_ms": self.response_delay_ms,
            "metadata": {
                "mock_response": True,
                "call_count": self.call_count,
                "detected_agent_type": agent_type,
            },
        }

    async def complete(self, prompt: str, agent_type: Optional[str] = None) -> str:
        """
        Complete prompt with mock response.

        Args:
            prompt: Input prompt
            agent_type: Optional agent type override

        Returns:
            Mock LLM response as JSON string

        Raises:
            Exception: If simulated failure occurs
        """
        self.call_count += 1
        logger.debug(f"MockLLMClient.complete called (count: {self.call_count})")

        # Simulate processing delay
        await self._simulate_delay()

        # Simulate failures
        if self._should_simulate_failure():
            raise Exception(f"Simulated LLM API failure (call #{self.call_count})")

        # Determine agent type
        if not agent_type:
            agent_type = self._extract_agent_type(prompt)

        # Generate response
        response = self._generate_response(prompt, agent_type)

        # Return as JSON string
        return json.dumps(response, indent=2)

    async def complete_structured(
        self, prompt: str, agent_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Complete prompt and return structured response.

        Args:
            prompt: Input prompt
            agent_type: Optional agent type override

        Returns:
            Mock response as dictionary
        """
        response_json = await self.complete(prompt, agent_type)
        return json.loads(response_json)

    def set_custom_response(
        self, agent_type: str, response_template: Dict[str, Any]
    ) -> None:
        """
        Set custom response template for an agent type.

        Args:
            agent_type: Agent type to customize
            response_template: Custom response template
        """
        self.response_templates[agent_type] = response_template
        logger.debug(f"Custom response template set for {agent_type}")

    def reset_stats(self) -> None:
        """Reset call statistics."""
        self.call_count = 0
        logger.debug("MockLLMClient stats reset")

    def get_stats(self) -> Dict[str, Any]:
        """Get client statistics."""
        return {
            "call_count": self.call_count,
            "response_delay_ms": self.response_delay_ms,
            "failure_rate": self.failure_rate,
            "available_agent_types": list(self.response_templates.keys()),
        }
