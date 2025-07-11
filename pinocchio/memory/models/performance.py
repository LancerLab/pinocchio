"""Performance metrics models for Pinocchio multi-agent system."""
from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class PerformanceMetrics(BaseModel):
    """Performance metrics for code execution."""

    execution_time_ms: float
    memory_usage_mb: float
    cache_miss_rate: Optional[float] = None
    cpu_utilization: Optional[float] = None
    throughput: Optional[float] = None
    latency: Optional[float] = None
    power_consumption: Optional[float] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    session_id: str
    code_version_id: str
    agent_type: str


class PerformanceHistory(BaseModel):
    """Performance history tracking for sessions."""

    session_id: str
    metrics: List[PerformanceMetrics] = Field(default_factory=list)

    def add_metrics(self, metrics: PerformanceMetrics) -> None:
        """Add performance metrics to history."""
        self.metrics.append(metrics)

    def get_latest_metrics(self) -> Optional[PerformanceMetrics]:
        """Get the latest performance metrics."""
        if self.metrics:
            return self.metrics[-1]
        return None

    def get_performance_trend(self) -> Dict[str, List[float]]:
        """Get performance trend data."""
        if not self.metrics:
            return {}
        trend = {
            "execution_time": [m.execution_time_ms for m in self.metrics],
            "memory_usage": [m.memory_usage_mb for m in self.metrics],
        }
        if any(m.cache_miss_rate is not None for m in self.metrics):
            trend["cache_miss_rate"] = [m.cache_miss_rate or 0.0 for m in self.metrics]
        return trend

    def get_metrics_dict(self) -> Dict[str, List[float]]:
        """Return metrics as a dictionary of lists of floats."""
        return {
            "latency": [m.latency or 0.0 for m in self.metrics],
            "throughput": [m.throughput or 0.0 for m in self.metrics],
            "execution_time": [m.execution_time_ms or 0.0 for m in self.metrics],
            "memory_usage": [m.memory_usage_mb or 0.0 for m in self.metrics],
        }
