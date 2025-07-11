"""
Pinocchio prompt module.

This module provides functionality for managing, formatting, and versioning prompt templates.
"""

from .formatter import TemplateFormatter
from .manager import PromptManager
from .models import (
    AgentType,
    PromptMemory,
    PromptTemplate,
    PromptType,
    StructuredInput,
    StructuredOutput,
)
from .template_loader import (
    APITemplateLoader,
    CompositeTemplateLoader,
    DatabaseTemplateLoader,
    DirectoryTemplateLoader,
    FileTemplateLoader,
    TemplateLoader,
    YAMLTemplateLoader,
    create_loader_from_config,
)
from .version_control import BranchInfo, VersionControl, VersionInfo, VersionStatus

__all__ = [
    # Models
    "PromptTemplate",
    "PromptMemory",
    "StructuredInput",
    "StructuredOutput",
    "AgentType",
    "PromptType",
    # Manager
    "PromptManager",
    # Template Loaders
    "TemplateLoader",
    "FileTemplateLoader",
    "YAMLTemplateLoader",
    "DirectoryTemplateLoader",
    "DatabaseTemplateLoader",
    "APITemplateLoader",
    "CompositeTemplateLoader",
    "create_loader_from_config",
    # Version Control
    "VersionControl",
    "VersionInfo",
    "BranchInfo",
    "VersionStatus",
    # Formatter
    "TemplateFormatter",
]
