"""Evaluator agent for performance analysis and optimization suggestions."""

import logging
import time
from typing import Any, Dict, List

from ..data_models.agent import AgentResponse
from .base import AgentWithRetry

logger = logging.getLogger(__name__)


class EvaluatorAgent(AgentWithRetry):
    """Agent responsible for performance analysis and evaluation."""

    def __init__(self, llm_client: Any, max_retries: int = 3):
        """
        Initialize Evaluator agent.

        Args:
            llm_client: LLM client instance
            max_retries: Maximum retry attempts for LLM calls
        """
        super().__init__("evaluator", llm_client, max_retries)
        logger.info("EvaluatorAgent initialized")

    async def execute(self, request: Dict[str, Any]) -> AgentResponse:
        """
        Execute performance evaluation task.

        Args:
            request: Evaluation request containing code and performance data

        Returns:
            Agent response with evaluation results
        """
        request_id = request.get("request_id", "unknown")

        try:
            # Build prompt for performance evaluation
            prompt = self._build_evaluation_prompt(request)

            # Call LLM with retry
            llm_response = await self._call_llm_with_retry(prompt)

            # Extract and process the response
            output = self._process_evaluation_response(llm_response, request)

            # Create successful response
            return self._create_response(
                request_id=request_id,
                success=True,
                output=output,
                processing_time_ms=int(self.get_average_processing_time()),
            )

        except Exception as e:
            return self._handle_error(request_id, e)

    def _build_evaluation_prompt(self, request: Dict[str, Any]) -> str:
        """
        Build specific prompt for performance evaluation.

        Args:
            request: Evaluation request

        Returns:
            Formatted prompt string
        """
        code = request.get("code", "")
        performance_metrics = request.get("performance_metrics", {})
        baseline_metrics = request.get("baseline_metrics", {})
        context = request.get("context", {})
        evaluation_criteria = request.get("evaluation_criteria", [])

        prompt_parts = [
            "You are an Evaluator agent in the Pinocchio multi-agent system.",
            "Your primary task is to analyze Choreo DSL code performance and provide evaluation reports.",
            "",
            "Code to evaluate:",
            "```choreo",
            code,
            "```",
        ]

        if performance_metrics:
            prompt_parts.extend(
                [
                    "",
                    "Performance Metrics:",
                    self._format_performance_metrics(performance_metrics),
                ]
            )

        if baseline_metrics:
            prompt_parts.extend(
                [
                    "",
                    "Baseline Metrics:",
                    self._format_performance_metrics(baseline_metrics),
                ]
            )

        if evaluation_criteria:
            prompt_parts.extend(
                ["", "Evaluation Criteria:", "- " + "\n- ".join(evaluation_criteria)]
            )

        if context:
            prompt_parts.extend(["", "Context:", str(context)])

        prompt_parts.extend(
            [
                "",
                self._get_agent_instructions(),
                "",
                self._get_evaluation_output_format(),
            ]
        )

        return "\n".join(prompt_parts)

    def _get_agent_instructions(self) -> str:
        """Get Evaluator-specific instructions."""
        return """
Instructions for Performance Evaluation:
1. Analyze the provided Choreo DSL code for performance characteristics
2. Evaluate performance in terms of:
   - Computational complexity (time and space)
   - Memory efficiency and access patterns
   - Cache utilization and data locality
   - Parallelization potential and scalability
   - Algorithmic efficiency and optimization opportunities
   - Resource utilization and bottlenecks
3. Compare against baseline performance if provided
4. Provide detailed performance analysis and recommendations
5. Identify areas for improvement and optimization
6. Consider both theoretical and practical performance aspects

Focus on:
- Time complexity analysis
- Space complexity analysis
- Memory access patterns
- Cache performance
- Parallelization opportunities
- Scalability assessment
- Bottleneck identification
- Optimization recommendations
"""

    def _get_evaluation_output_format(self) -> str:
        """Get output format for evaluation response."""
        return """
Please provide your response in JSON format with the following structure:
{
    "agent_type": "evaluator",
    "success": true,
    "output": {
        "performance_analysis": {
            "time_complexity": "O(n) analysis",
            "space_complexity": "O(n) analysis",
            "memory_efficiency": "high|medium|low",
            "cache_performance": "excellent|good|fair|poor",
            "parallelization_potential": "high|medium|low",
            "scalability": "excellent|good|fair|poor"
        },
        "performance_metrics": {
            "execution_time": "estimated_time",
            "memory_usage": "estimated_memory",
            "throughput": "operations_per_second",
            "efficiency_score": "percentage"
        },
        "bottlenecks": [
            {
                "type": "memory|computation|io|synchronization",
                "description": "Description of the bottleneck",
                "impact": "high|medium|low",
                "suggested_improvement": "How to address this bottleneck"
            }
        ],
        "optimization_recommendations": [
            {
                "type": "algorithmic|memory|parallelization|cache",
                "description": "Detailed recommendation",
                "expected_improvement": "percentage_or_description",
                "implementation_effort": "easy|medium|hard"
            }
        ],
        "comparison_with_baseline": {
            "performance_difference": "percentage",
            "improvement_areas": ["area1", "area2"],
            "regression_areas": ["area1", "area2"]
        },
        "evaluation_metadata": {
            "analysis_time": "timestamp",
            "evaluation_score": "overall_score",
            "confidence_level": "high|medium|low"
        }
    },
    "error_message": null
}
"""

    def _format_performance_metrics(self, metrics: Dict[str, Any]) -> str:
        """Format performance metrics for prompt."""
        formatted = []
        for key, value in metrics.items():
            formatted.append(f"- {key}: {value}")
        return "\n".join(formatted)

    def _process_evaluation_response(
        self, llm_response: Dict[str, Any], request: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process LLM response and extract evaluation results.

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
            # Fallback: create basic evaluation structure
            output = {
                "performance_analysis": {
                    "time_complexity": "O(n) - estimated",
                    "space_complexity": "O(n) - estimated",
                    "memory_efficiency": "medium",
                    "cache_performance": "unknown",
                    "parallelization_potential": "unknown",
                    "scalability": "unknown",
                },
                "performance_metrics": {
                    "execution_time": "unknown",
                    "memory_usage": "unknown",
                    "throughput": "unknown",
                    "efficiency_score": "unknown",
                },
                "bottlenecks": [
                    {
                        "type": "general",
                        "description": "Basic performance analysis performed",
                        "impact": "unknown",
                        "suggested_improvement": "Detailed analysis not available",
                    }
                ],
                "optimization_recommendations": [
                    {
                        "type": "general",
                        "description": "Perform detailed performance analysis",
                        "expected_improvement": "unknown",
                        "implementation_effort": "medium",
                    }
                ],
                "comparison_with_baseline": {
                    "performance_difference": "unknown",
                    "improvement_areas": ["Analysis not available"],
                    "regression_areas": [],
                },
                "evaluation_metadata": {
                    "analysis_time": request.get("timestamp", ""),
                    "evaluation_score": "unknown",
                    "confidence_level": "low",
                },
            }

        # Ensure required fields are present
        output.setdefault("performance_analysis", {})
        output.setdefault("performance_metrics", {})
        output.setdefault("bottlenecks", [])
        output.setdefault("optimization_recommendations", [])
        output.setdefault("comparison_with_baseline", {})
        output.setdefault("evaluation_metadata", {})

        # Add evaluation metadata
        output["evaluation_metadata"].update(
            {
                "agent_type": "evaluator",
                "request_id": request.get("request_id", "unknown"),
                "code_length": len(request.get("code", "")),
                "evaluation_criteria": request.get("evaluation_criteria", []),
            }
        )

        return output

    def evaluate_performance(
        self, code: str, performance_metrics: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Evaluate code performance and provide analysis.

        Args:
            code: Code to evaluate
            performance_metrics: Optional performance metrics

        Returns:
            Performance evaluation results
        """
        request = {
            "code": code,
            "performance_metrics": performance_metrics or {},
            "evaluation_criteria": ["performance", "memory_efficiency", "scalability"],
            "request_id": f"eval_{int(time.time())}",
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

    def get_performance_analysis(self, code: str) -> Dict[str, Any]:
        """
        Get detailed performance analysis of the code.

        Args:
            code: Code to analyze

        Returns:
            Performance analysis results
        """
        evaluation = self.evaluate_performance(code)
        return evaluation.get("performance_analysis", {})

    def get_optimization_recommendations(self, code: str) -> List[Dict[str, Any]]:
        """
        Get optimization recommendations for the code.

        Args:
            code: Code to analyze

        Returns:
            List of optimization recommendations
        """
        evaluation = self.evaluate_performance(code)
        return evaluation.get("optimization_recommendations", [])

    def get_performance_score(self, code: str) -> Dict[str, Any]:
        """
        Get overall performance score and metrics.

        Args:
            code: Code to evaluate

        Returns:
            Performance score and metrics
        """
        evaluation = self.evaluate_performance(code)
        return {
            "performance_metrics": evaluation.get("performance_metrics", {}),
            "evaluation_metadata": evaluation.get("evaluation_metadata", {}),
            "bottlenecks": evaluation.get("bottlenecks", []),
        }

    def compare_with_baseline(
        self, code: str, baseline_metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Compare code performance with baseline metrics.

        Args:
            code: Code to evaluate
            baseline_metrics: Baseline performance metrics

        Returns:
            Comparison results
        """
        request = {
            "code": code,
            "baseline_metrics": baseline_metrics,
            "evaluation_criteria": ["performance_comparison", "improvement_analysis"],
            "request_id": f"compare_{int(time.time())}",
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
        return (
            response.output.get("comparison_with_baseline", {})
            if response.success
            else {}
        )
