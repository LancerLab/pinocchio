"""Optimization technique models for Pinocchio multi-agent system."""
from typing import Any, Dict, List

from pydantic import BaseModel, Field


class OptimizationTechnique(BaseModel):
    """Optimization technique configuration and metadata."""

    name: str
    category: str
    description: str
    hyperparameters: Dict[str, Any]
    applicable_scenarios: List[str]
    expected_improvement: str
    implementation_guide: str
    examples: List[str] = Field(default_factory=list)


class OptimizationHistory(BaseModel):
    """Optimization history tracking for sessions."""

    session_id: str
    techniques_used: List[str] = Field(default_factory=list)
    hyperparameter_history: List[Dict[str, Any]] = Field(default_factory=list)
    performance_impact: Dict[str, float] = Field(default_factory=dict)
    iteration_count: int = 0

    def add_iteration(
        self,
        techniques: List[str],
        hyperparameters: Dict[str, Any],
        performance_impact: Dict[str, float],
    ) -> None:
        """Add a new optimization iteration to history."""
        self.techniques_used = techniques
        self.hyperparameter_history.append(hyperparameters)
        self.performance_impact = performance_impact
        self.iteration_count += 1

    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get optimization summary for the session."""
        return {
            "total_iterations": self.iteration_count,
            "techniques_used": self.techniques_used,
            "current_hyperparameters": self.hyperparameter_history[-1]
            if self.hyperparameter_history
            else {},
            "performance_impact": self.performance_impact,
        }
