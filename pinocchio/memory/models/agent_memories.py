"""
Agent memory models for Pinocchio multi-agent system.

This module defines the models for tracking agent interactions and memories.
"""

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class BaseAgentMemory(BaseModel):
    """
    Base agent memory model.

    Base class for all agent-specific memory records, containing common fields
    like input/output data, processing time, and status.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str
    agent_type: str  # generator, debugger, evaluator
    version_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    input_data: Dict[str, Any]
    output_data: Dict[str, Any]
    processing_time_ms: int
    status: str = "success"  # success, error
    error_details: Optional[Dict[str, Any]] = None
    code_version_id: Optional[str] = None  # Associated code version ID
    parent_version_id: Optional[str] = None


class GeneratorMemory(BaseAgentMemory):
    """
    Generator agent memory.

    Records interactions with the code generator agent.
    """

    agent_type: str = "generator"
    generation_strategy: str  # The generation strategy used
    optimization_techniques: List[str] = Field(default_factory=list)
    hyperparameters: Dict[str, Any] = Field(default_factory=dict)
    kernel_type: str = ""
    language: str = ""
    comments: List[str] = Field(default_factory=list)
    knowledge_fragments: Dict[str, Any] = Field(default_factory=dict)


class DebuggerMemory(BaseAgentMemory):
    """
    Debugger agent memory.

    Records interactions with the code debugger agent.
    """

    agent_type: str = "debugger"
    compilation_status: str = "unknown"
    runtime_status: str = "unknown"
    performance_metrics: Dict[str, Any] = Field(default_factory=dict)
    modified_code: Optional[str] = None
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    execution_log: List[str] = Field(default_factory=list)
    preserved_optimization_techniques: List[str] = Field(default_factory=list)
    preserved_hyperparameters: Dict[str, Any] = Field(default_factory=dict)


class OptimizerMemory(BaseAgentMemory):
    """
    Optimizer agent memory.

    Records interactions with the code optimizer agent.
    """

    agent_type: str = "optimizer"
    optimization_patterns: List[Dict[str, Any]]  # Identified optimization patterns
    selected_pattern: Optional[Dict[str, Any]] = None  # Selected optimization pattern
    recommendation: str  # Optimization recommendation


class EvaluatorMemory(BaseAgentMemory):
    """
    Evaluator agent memory.

    Records interactions with the code evaluator agent.
    """

    agent_type: str = "evaluator"
    current_optimization_techniques: List[str] = Field(default_factory=list)
    current_hyperparameters: Dict[str, Any] = Field(default_factory=dict)
    optimization_suggestions: Dict[str, Any] = Field(default_factory=dict)
    performance_analysis: Dict[str, Any] = Field(default_factory=dict)
    next_iteration_prompt: str = ""
    bottlenecks: List[str] = Field(default_factory=list)
    target_performance: Dict[str, Any] = Field(default_factory=dict)


class AgentMemory(BaseAgentMemory):
    """
    Generic agent memory.

    Records interactions for any agent type (generator, debugger, evaluator, etc).
    This is a flexible memory model for integration and workflow use cases.
    """

    pass
