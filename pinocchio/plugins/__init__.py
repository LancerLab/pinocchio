"""
Plugin system for Pinocchio multi-agent system.

This module provides a plugin architecture for customizing various components
including prompts, agents, workflows, and other system modules.
"""

from .agent_plugins import AgentPluginBase, CustomAgentPlugin
from .base import Plugin, PluginManager, PluginType
from .prompt_plugins import CustomPromptPlugin, PromptPluginBase
from .workflow_plugins import CustomWorkflowPlugin, WorkflowPluginBase

__all__ = [
    "Plugin",
    "PluginManager",
    "PluginType",
    "PromptPluginBase",
    "CustomPromptPlugin",
    "AgentPluginBase",
    "CustomAgentPlugin",
    "WorkflowPluginBase",
    "CustomWorkflowPlugin",
]
