"""
Pinocchio unified data models package.

This package provides a unified interface for accessing data models
from different modules, ensuring backward compatibility.
"""

from ..knowledge.models import KnowledgeItem, KnowledgeMemory

# Import from respective modules
from ..memory.models import (
    BaseAgentMemory,
    BaseMemory,
    CodeMemory,
    CodeVersion,
    DebuggerMemory,
    EvaluatorMemory,
    GeneratorMemory,
    OptimizerMemory,
)
from ..prompt.models import PromptMemory, PromptTemplate
from ..session.models import SessionMetadata

__all__ = [
    # Memory models
    "BaseMemory",
    "CodeVersion",
    "CodeMemory",
    "BaseAgentMemory",
    "GeneratorMemory",
    "DebuggerMemory",
    "OptimizerMemory",
    "EvaluatorMemory",
    # Prompt models
    "PromptTemplate",
    "PromptMemory",
    # Knowledge models
    "KnowledgeItem",
    "KnowledgeMemory",
    # Session models
    "SessionMetadata",
]
