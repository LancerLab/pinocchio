"""Agent data models for Pinocchio."""

import uuid
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class AgentRequest(BaseModel):
    """Agent request data model."""

    agent_type: str = Field(
        description="Type of agent (generator, debugger, optimizer, evaluator)"
    )
    prompt: Dict[str, Any] = Field(description="Prompt data for the agent")
    context: Dict[str, Any] = Field(
        default_factory=dict, description="Additional context"
    )
    request_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()), description="Unique request ID"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="Request timestamp"
    )
    session_id: Optional[str] = Field(
        default=None, description="Session ID this request belongs to"
    )

    model_config = {"json_encoders": {datetime: lambda v: v.isoformat()}}


class AgentResponse(BaseModel):
    """Agent response data model."""

    agent_type: str = Field(description="Type of agent that generated this response")
    success: bool = Field(description="Whether the agent execution was successful")
    output: Dict[str, Any] = Field(
        default_factory=dict, description="Agent output data"
    )
    error_message: Optional[str] = Field(
        default=None, description="Error message if execution failed"
    )
    request_id: str = Field(
        description="ID of the request this response corresponds to"
    )
    response_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()), description="Unique response ID"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="Response timestamp"
    )
    processing_time_ms: Optional[int] = Field(
        default=None, description="Processing time in milliseconds"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )

    model_config = {"json_encoders": {datetime: lambda v: v.isoformat()}}


class GeneratorRequest(AgentRequest):
    """Specialized request for Generator agent."""

    agent_type: Literal["generator"] = Field(default="generator")
    task_description: str = Field(description="Description of the code generation task")
    requirements: Dict[str, Any] = Field(
        default_factory=dict, description="Specific requirements"
    )
    optimization_goals: List[str] = Field(
        default_factory=list, description="Performance optimization goals"
    )


class GeneratorResponse(AgentResponse):
    """Specialized response for Generator agent."""

    agent_type: Literal["generator"] = Field(default="generator")
    code: Optional[str] = Field(default=None, description="Generated code")
    language: Optional[str] = Field(default=None, description="Programming language")
    explanation: Optional[str] = Field(default=None, description="Code explanation")
    optimization_techniques: List[str] = Field(
        default_factory=list, description="Applied optimization techniques"
    )
    hyperparameters: Dict[str, Any] = Field(
        default_factory=dict, description="Configuration parameters"
    )


class DebuggerRequest(AgentRequest):
    """Specialized request for Debugger agent."""

    agent_type: Literal["debugger"] = Field(default="debugger")
    code: str = Field(description="Code to debug")
    error_info: Optional[Dict[str, Any]] = Field(
        default=None, description="Error information"
    )
    previous_attempts: List[Dict[str, Any]] = Field(
        default_factory=list, description="Previous debug attempts"
    )


class DebuggerResponse(AgentResponse):
    """Specialized response for Debugger agent."""

    agent_type: Literal["debugger"] = Field(default="debugger")
    fixed_code: Optional[str] = Field(default=None, description="Fixed code")
    issues_found: List[str] = Field(
        default_factory=list, description="Issues identified"
    )
    fixes_applied: List[str] = Field(default_factory=list, description="Fixes applied")
    confidence: Optional[float] = Field(
        default=None, description="Confidence in the fix (0-1)"
    )


class OptimizerRequest(AgentRequest):
    """Specialized request for Optimizer agent."""

    agent_type: Literal["optimizer"] = Field(default="optimizer")
    code: str = Field(description="Code to optimize")
    performance_metrics: Dict[str, Any] = Field(
        default_factory=dict, description="Current performance metrics"
    )
    optimization_targets: List[str] = Field(
        default_factory=list, description="Optimization targets"
    )


class OptimizerResponse(AgentResponse):
    """Specialized response for Optimizer agent."""

    agent_type: Literal["optimizer"] = Field(default="optimizer")
    optimized_code: Optional[str] = Field(default=None, description="Optimized code")
    optimization_suggestions: List[str] = Field(
        default_factory=list, description="Optimization suggestions"
    )
    expected_improvement: Dict[str, Any] = Field(
        default_factory=dict, description="Expected performance improvement"
    )
    risk_assessment: Optional[str] = Field(
        default=None, description="Risk assessment of optimizations"
    )


class EvaluatorRequest(AgentRequest):
    """Specialized request for Evaluator agent."""

    agent_type: Literal["evaluator"] = Field(default="evaluator")
    code: str = Field(description="Code to evaluate")
    test_cases: List[Dict[str, Any]] = Field(
        default_factory=list, description="Test cases"
    )
    evaluation_criteria: List[str] = Field(
        default_factory=list, description="Evaluation criteria"
    )


class EvaluatorResponse(AgentResponse):
    """Specialized response for Evaluator agent."""

    agent_type: Literal["evaluator"] = Field(default="evaluator")
    evaluation_results: Dict[str, Any] = Field(
        default_factory=dict, description="Evaluation results"
    )
    performance_metrics: Dict[str, Any] = Field(
        default_factory=dict, description="Performance metrics"
    )
    test_results: List[Dict[str, Any]] = Field(
        default_factory=list, description="Test execution results"
    )
    overall_score: Optional[float] = Field(
        default=None, description="Overall quality score (0-1)"
    )
    recommendations: List[str] = Field(
        default_factory=list, description="Improvement recommendations"
    )


class AgentInteraction(BaseModel):
    """Model for recording agent interactions."""

    interaction_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()), description="Unique interaction ID"
    )
    session_id: str = Field(description="Session this interaction belongs to")
    step_id: str = Field(description="Step ID in the workflow")
    agent_type: str = Field(description="Type of agent")
    request: AgentRequest = Field(description="Agent request")
    response: AgentResponse = Field(description="Agent response")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="Interaction timestamp"
    )
    duration_ms: Optional[int] = Field(
        default=None, description="Interaction duration in milliseconds"
    )

    model_config = {"json_encoders": {datetime: lambda v: v.isoformat()}}
