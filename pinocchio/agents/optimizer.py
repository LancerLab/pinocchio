"""Optimizer agent for code optimization and performance tuning."""

import logging
from typing import Any, Dict, List

from pinocchio.config import ConfigManager
from pinocchio.llm.custom_llm_client import CustomLLMClient

from ..data_models.agent import AgentResponse
from .base import AgentWithRetry

logger = logging.getLogger(__name__)


class OptimizerAgent(AgentWithRetry):
    """Agent responsible for code optimization and performance tuning."""

    def __init__(self, llm_client: Any = None, max_retries: int = 3):
        """
        Initialize Optimizer agent.

        Args:
            llm_client: LLM client instance (optional)
            max_retries: Maximum retry attempts for LLM calls
        """
        if llm_client is None:
            config_manager = ConfigManager()
            agent_llm_config = config_manager.get_agent_llm_config("optimizer")
            verbose = config_manager.get("verbose.enabled", True)
            llm_client = CustomLLMClient(agent_llm_config, verbose=verbose)
        super().__init__("optimizer", llm_client, max_retries)
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

        try:
            # Build prompt for code optimization
            prompt = self._build_optimization_prompt(request)

            # Call LLM with retry
            llm_response = await self._call_llm_with_retry(prompt)

            # Extract and process the response
            output = self._process_optimization_response(llm_response, request)

            # Create successful response
            return self._create_response(
                request_id=request_id,
                success=True,
                output=output,
                processing_time_ms=int(self.get_average_processing_time()),
            )

        except Exception as e:
            return self._handle_error(request_id, e)

    def _build_optimization_prompt(self, request: Dict[str, Any]) -> str:
        """
        Build specific prompt for code optimization.

        Args:
            request: Optimization request

        Returns:
            Formatted prompt string
        """
        code = request.get("code", "")
        optimization_goals = request.get("optimization_goals", [])
        performance_metrics = request.get("performance_metrics", {})
        context = request.get("context", {})
        current_performance = request.get("current_performance", {})
        detailed_instruction = request.get("detailed_instruction", "")

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
        else:
            # Fallback to basic optimization goals
            if optimization_goals:
                prompt_parts.extend(
                    ["", "Optimization Goals:", "- " + "\n- ".join(optimization_goals)]
                )

        if current_performance:
            prompt_parts.extend(
                [
                    "",
                    "Current Performance Metrics:",
                    self._format_performance(current_performance),
                ]
            )

        if performance_metrics:
            prompt_parts.extend(
                [
                    "",
                    "Target Performance Metrics:",
                    self._format_performance(performance_metrics),
                ]
            )

        if context:
            prompt_parts.extend(["", "Context:", str(context)])

        prompt_parts.extend(
            [
                "",
                self._get_agent_instructions(),
                "",
                self._get_optimization_output_format(),
            ]
        )

        return "\n".join(prompt_parts)

    def _get_agent_instructions(self) -> str:
        """Get Optimizer-specific instructions."""
        return """
Instructions for Code Optimization:
1. Analyze the provided Choreo DSL code for performance bottlenecks
2. Identify optimization opportunities in:
   - Loop structures and iteration patterns
   - Memory access patterns and cache utilization
   - Data locality and spatial/temporal locality
   - Parallelization opportunities
   - Vectorization possibilities
   - Memory allocation and deallocation patterns
3. Provide specific, actionable optimization suggestions
4. Estimate performance improvements for each suggestion
5. Consider trade-offs between performance and code complexity
6. Ensure optimizations maintain correctness and numerical stability

Focus on:
- Loop optimization (tiling, unrolling, fusion)
- Memory optimization (coalescing, prefetching, alignment)
- Parallelization (threading, vectorization)
- Algorithmic improvements
- Compiler-friendly code patterns
"""

    def _get_optimization_output_format(self) -> str:
        """Get output format for optimization response."""
        return """
Please provide your response in JSON format with the following structure:
{
    "agent_type": "optimizer",
    "success": true,
    "output": {
        "optimization_suggestions": [
            {
                "type": "loop_optimization|memory_optimization|parallelization|algorithmic",
                "description": "Detailed description of the optimization",
                "code_changes": "Specific code changes or new code",
                "expected_improvement": {
                    "performance_gain": "estimated_percentage",
                    "memory_reduction": "estimated_percentage",
                    "complexity_change": "increase|decrease|same"
                },
                "implementation_difficulty": "easy|medium|hard",
                "risk_level": "low|medium|high",
                "priority": 1-5
            }
        ],
        "performance_analysis": {
            "current_bottlenecks": ["bottleneck1", "bottleneck2"],
            "optimization_potential": "high|medium|low",
            "estimated_overall_improvement": "percentage"
        },
        "optimized_code": "Complete optimized version of the code",
        "optimization_metadata": {
            "analysis_time": "timestamp",
            "optimization_count": "number_of_suggestions",
            "primary_optimization_focus": "loop|memory|parallel|algorithm"
        }
    },
    "error_message": null
}
"""

    def _format_performance(self, performance: Dict[str, Any]) -> str:
        """Format performance metrics for prompt."""
        formatted = []
        for key, value in performance.items():
            formatted.append(f"- {key}: {value}")
        return "\n".join(formatted)

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
        # Extract output from LLM response
        if llm_response.get("success") and "output" in llm_response:
            output = llm_response["output"]
        else:
            # Fallback: create basic optimization structure
            output = {
                "optimization_suggestions": [
                    {
                        "type": "general_optimization",
                        "description": "Basic optimization analysis",
                        "code_changes": "// Optimization suggestions not available",
                        "expected_improvement": {
                            "performance_gain": "unknown",
                            "memory_reduction": "unknown",
                            "complexity_change": "same",
                        },
                        "implementation_difficulty": "medium",
                        "risk_level": "low",
                        "priority": 3,
                    }
                ],
                "performance_analysis": {
                    "current_bottlenecks": ["Analysis not available"],
                    "optimization_potential": "unknown",
                    "estimated_overall_improvement": "unknown",
                },
                "optimized_code": request.get("code", "// No optimized code available"),
                "optimization_metadata": {
                    "analysis_time": request.get("timestamp", ""),
                    "optimization_count": 1,
                    "primary_optimization_focus": "general",
                },
            }

        # Ensure required fields are present
        output.setdefault("optimization_suggestions", [])
        output.setdefault("performance_analysis", {})
        output.setdefault("optimized_code", request.get("code", ""))
        output.setdefault("optimization_metadata", {})

        # Add optimization metadata
        output["optimization_metadata"].update(
            {
                "agent_type": "optimizer",
                "request_id": request.get("request_id", "unknown"),
                "original_code_length": len(request.get("code", "")),
                "optimization_goals": request.get("optimization_goals", []),
            }
        )

        return output

    def analyze_code_performance(
        self, code: str, goals: List[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze code performance and provide optimization suggestions.

        Args:
            code: Code to analyze
            goals: List of optimization goals

        Returns:
            Optimization analysis results
        """
        import time

        request = {
            "code": code,
            "optimization_goals": goals or ["performance", "memory_efficiency"],
            "request_id": f"analysis_{int(time.time())}",
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

    def get_optimization_suggestions(self, code: str) -> List[Dict[str, Any]]:
        """
        Get optimization suggestions for given code.

        Args:
            code: Code to optimize

        Returns:
            List of optimization suggestions
        """
        analysis = self.analyze_code_performance(code)
        return analysis.get("optimization_suggestions", [])

    def get_optimized_code(self, code: str) -> str:
        """
        Get optimized version of the code.

        Args:
            code: Original code

        Returns:
            Optimized code
        """
        analysis = self.analyze_code_performance(code)
        return analysis.get("optimized_code", code)
