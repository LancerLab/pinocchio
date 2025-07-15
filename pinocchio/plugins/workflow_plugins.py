"""
Workflow plugins for Pinocchio.

This module provides plugin interfaces for customizing workflow behavior and task planning.
"""

import logging
from abc import abstractmethod
from typing import Any, Dict, List, Optional

from ..data_models.task_planning import AgentType, Task, TaskPlan, TaskPriority
from .base import Plugin, PluginType

logger = logging.getLogger(__name__)


class WorkflowPluginBase(Plugin):
    """Base class for workflow plugins."""

    def __init__(self, name: str, version: str = "1.0.0"):
        """Initialize the WorkflowPlugin class."""
        super().__init__(name, PluginType.WORKFLOW, version)
        self.workflow_configs: Dict[str, Dict[str, Any]] = {}

    @abstractmethod
    def create_workflow(self, user_request: str, config: Dict[str, Any]) -> TaskPlan:
        """Create a custom workflow based on configuration."""
        pass

    def execute(self, action: str, *args, **kwargs) -> Any:
        """Execute plugin action."""
        if action == "create_workflow":
            return self.create_workflow(
                kwargs.get("user_request"), kwargs.get("config", {})
            )
        else:
            raise ValueError(f"Unknown action: {action}")


class CustomWorkflowPlugin(WorkflowPluginBase):
    """Custom workflow plugin with JSON-defined workflows."""

    def __init__(self):
        """Initialize the CustomWorkflowPlugin class."""
        super().__init__("json_workflow_plugin")
        self.metadata = {
            "description": "JSON-configurable workflow templates",
            "supports_fallback": True,
            "custom_workflows": [],
        }

    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize plugin with configuration."""
        self.config = config
        self.workflow_configs = config.get("workflows", {})
        logger.info(
            f"JSON workflow plugin initialized with {len(self.workflow_configs)} workflows"
        )

    def create_workflow(self, user_request: str, config: Dict[str, Any]) -> TaskPlan:
        """Create workflow from JSON configuration."""
        workflow_name = config.get("workflow_name", "default")
        workflow_config = self.workflow_configs.get(workflow_name)

        if not workflow_config:
            raise ValueError(f"Workflow configuration not found: {workflow_name}")

        # Create tasks based on workflow configuration
        tasks = self._create_tasks_from_config(workflow_config, user_request)

        # Create task plan
        plan = TaskPlan(
            plan_id=f"json_workflow_{workflow_name}",
            user_request=user_request,
            tasks=tasks,
            context={"workflow_type": "json_defined", "workflow_name": workflow_name},
        )

        return plan

    def _create_tasks_from_config(
        self, workflow_config: Dict[str, Any], user_request: str
    ) -> List[Task]:
        """Create tasks from workflow configuration."""
        tasks = []
        task_definitions = workflow_config.get("tasks", [])

        for i, task_def in enumerate(task_definitions):
            task = Task(
                task_id=task_def.get("id", f"task_{i+1}"),
                agent_type=self._parse_agent_type(
                    task_def.get("agent_type", "generator")
                ),
                task_description=task_def.get("description", user_request),
                requirements=task_def.get("requirements", {}),
                optimization_goals=task_def.get("optimization_goals", []),
                priority=self._parse_priority(task_def.get("priority", "high")),
                dependencies=self._parse_dependencies(task_def.get("dependencies", [])),
                input_data=task_def.get("input_data", {}),
            )

            # Add user request to input data
            task.input_data["user_request"] = user_request

            tasks.append(task)

        return tasks

    def _parse_agent_type(self, agent_type_str: str) -> AgentType:
        """Parse agent type from string."""
        agent_type_map = {
            "generator": AgentType.GENERATOR,
            "optimizer": AgentType.OPTIMIZER,
            "debugger": AgentType.DEBUGGER,
            "evaluator": AgentType.EVALUATOR,
        }
        return agent_type_map.get(agent_type_str.lower(), AgentType.GENERATOR)

    def _parse_priority(self, priority_str: str) -> TaskPriority:
        """Parse task priority from string."""
        priority_map = {
            "low": TaskPriority.LOW,
            "medium": TaskPriority.MEDIUM,
            "high": TaskPriority.HIGH,
            "critical": TaskPriority.CRITICAL,
        }
        return priority_map.get(priority_str.lower(), TaskPriority.HIGH)

    def _parse_dependencies(self, dependencies: List[str]) -> List[Any]:
        """Parse task dependencies."""
        # For now, return empty list - will implement dependency parsing later
        return []

    def add_workflow_config(self, name: str, config: Dict[str, Any]) -> None:
        """Add a new workflow configuration."""
        self.workflow_configs[name] = config
        logger.info(f"Added workflow configuration: {name}")

    def get_available_workflows(self) -> List[str]:
        """Get list of available workflow names."""
        return list(self.workflow_configs.keys())
