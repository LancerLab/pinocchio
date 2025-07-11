"""
Pinocchio prompt models package.

This package contains all the data models used by the prompt module.
"""

from .template import (
    AgentType,
    PromptMemory,
    PromptTemplate,
    PromptType,
    StructuredInput,
    StructuredOutput,
)

__all__ = [
    "PromptTemplate",
    "PromptMemory",
    "StructuredInput",
    "StructuredOutput",
    "AgentType",
    "PromptType",
]
