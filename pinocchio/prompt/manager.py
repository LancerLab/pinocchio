"""
Pinocchio prompt manager.

This module provides a comprehensive prompt management system with support for
multi-agent templates, version control, performance tracking, and structured I/O.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .formatter import TemplateFormatter
from .models import (
    AgentType,
    PromptMemory,
    PromptTemplate,
    PromptType,
    StructuredInput,
    StructuredOutput,
)


class PromptManager:
    """
    Comprehensive prompt manager for Pinocchio.

    Handles template management, version control, performance tracking,
    and structured input/output processing.
    """

    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize the prompt manager.

        Args:
            storage_path: Path to store prompt templates and metadata
        """
        self.memory = PromptMemory()
        self.formatter = TemplateFormatter()
        self.storage_path = (
            Path(storage_path) if storage_path else Path("./prompt_storage")
        )
        self.storage_path.mkdir(exist_ok=True)

        # Load existing templates if available
        self._load_templates()

    def create_template(
        self,
        template_name: str,
        content: str,
        agent_type: AgentType = AgentType.GENERATOR,
        prompt_type: PromptType = PromptType.CODE_GENERATION,
        description: str = "",
        input_schema: Optional[StructuredInput] = None,
        output_schema: Optional[StructuredOutput] = None,
        tags: Optional[List[str]] = None,
        parent_version_id: Optional[str] = None,
    ) -> PromptTemplate:
        """
        Create a new prompt template.

        Args:
            template_name: Name of the template
            content: Template content with variables
            agent_type: Type of agent this template is for
            prompt_type: Type of prompt
            description: Template description
            input_schema: Structured input schema
            output_schema: Structured output schema
            tags: Template tags
            parent_version_id: Parent version ID for versioning

        Returns:
            Created prompt template
        """
        template = PromptTemplate.create_new_version(
            template_name=template_name,
            content=content,
            agent_type=agent_type,
            prompt_type=prompt_type,
            description=description,
            input_schema=input_schema,
            output_schema=output_schema,
            tags=tags,
            parent_version_id=parent_version_id,
        )

        self.memory.add_template(template)
        self._save_template(template)

        return template

    def get_template(
        self,
        template_name: str,
        version_id: Optional[str] = None,
        agent_type: Optional[AgentType] = None,
    ) -> Optional[PromptTemplate]:
        """
        Get a prompt template.

        Args:
            template_name: Name of the template
            version_id: Specific version ID (optional)
            agent_type: Agent type filter (optional)

        Returns:
            Prompt template or None if not found
        """
        if agent_type is not None:
            return self.memory.get_template_by_agent(agent_type, template_name)

        return self.memory.get_template(template_name, version_id)

    def format_template(
        self,
        template_name: str,
        variables: Dict[str, Any],
        version_id: Optional[str] = None,
        agent_type: Optional[AgentType] = None,
    ) -> Optional[str]:
        """
        Format a prompt template with variables.

        Args:
            template_name: Name of the template
            variables: Variables to substitute
            version_id: Specific version ID (optional)
            agent_type: Agent type filter (optional)

        Returns:
            Formatted template or None if not found
        """
        template = self.get_template(template_name, version_id, agent_type)
        if template is None:
            return None

        return self.formatter.format_template(template.content, variables)

    def format_structured_prompt(
        self,
        template_name: str,
        structured_input: StructuredInput,
        version_id: Optional[str] = None,
        agent_type: Optional[AgentType] = None,
    ) -> Optional[str]:
        """
        Format a prompt template with structured input.

        Args:
            template_name: Name of the template
            structured_input: Structured input data
            version_id: Specific version ID (optional)
            agent_type: Agent type filter (optional)

        Returns:
            Formatted template or None if not found
        """
        template = self.get_template(template_name, version_id, agent_type)
        if template is None:
            return None

        # Convert structured input to variables
        variables = structured_input.to_dict()

        return self.formatter.format_template(template.content, variables)

    def list_templates(self, agent_type: Optional[AgentType] = None) -> Dict[str, str]:
        """
        List available templates.

        Args:
            agent_type: Filter by agent type (optional)

        Returns:
            Dictionary of template names to version IDs
        """
        if agent_type is not None:
            return self.memory.list_templates_by_agent(agent_type)

        return self.memory.list_templates()

    def list_template_versions(self, template_name: str) -> Dict[str, PromptTemplate]:
        """
        List all versions of a template.

        Args:
            template_name: Name of the template

        Returns:
            Dictionary of version IDs to templates
        """
        return self.memory.list_template_versions(template_name)

    def set_current_version(self, template_name: str, version_id: str) -> bool:
        """
        Set the current version of a template.

        Args:
            template_name: Name of the template
            version_id: Version ID to set as current

        Returns:
            True if successful, False otherwise
        """
        success = self.memory.set_current_version(template_name, version_id)
        if success:
            self._save_memory_state()
        return success

    def remove_template(
        self, template_name: str, version_id: Optional[str] = None
    ) -> bool:
        """
        Remove a template or specific version.

        Args:
            template_name: Name of the template
            version_id: Specific version to remove (optional, removes all if None)

        Returns:
            True if successful, False otherwise
        """
        success = self.memory.remove_template(template_name, version_id)
        if success:
            self._save_memory_state()
            if version_id is None:
                # Remove all files for this template
                self._remove_template_files(template_name)
            else:
                # Remove specific version file
                self._remove_template_file(template_name, version_id)
        return success

    def search_templates(
        self, query: str, agent_type: Optional[AgentType] = None
    ) -> List[PromptTemplate]:
        """
        Search templates by content, description, or tags.

        Args:
            query: Search query
            agent_type: Filter by agent type (optional)

        Returns:
            List of matching templates
        """
        return self.memory.search_templates(query, agent_type)

    def update_template_stats(
        self,
        template_name: str,
        success: bool,
        response_time: float,
        version_id: Optional[str] = None,
    ) -> bool:
        """
        Update template usage statistics.

        Args:
            template_name: Name of the template
            success: Whether the template usage was successful
            response_time: Response time in seconds
            version_id: Specific version ID (optional)

        Returns:
            True if successful, False otherwise
        """
        template = self.get_template(template_name, version_id)
        if template is None:
            return False

        template.update_usage_stats(success, response_time)
        self._save_template(template)
        return True

    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get overall performance statistics.

        Returns:
            Performance statistics dictionary
        """
        return self.memory.get_performance_stats()

    def export_template(
        self, template_name: str, version_id: Optional[str] = None, format: str = "json"
    ) -> Optional[str]:
        """
        Export a template to a specific format.

        Args:
            template_name: Name of the template
            version_id: Specific version ID (optional)
            format: Export format ("json" or "yaml")

        Returns:
            Exported template string or None if not found
        """
        template = self.get_template(template_name, version_id)
        if template is None:
            return None

        if format.lower() == "json":
            return json.dumps(template.to_dict(), indent=2)
        elif format.lower() == "yaml":
            import yaml

            return yaml.dump(template.to_dict(), default_flow_style=False)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def import_template(
        self, template_data: Union[str, Dict[str, Any]], format: str = "json"
    ) -> PromptTemplate:
        """
        Import a template from a specific format.

        Args:
            template_data: Template data (string or dict)
            format: Import format ("json" or "yaml")

        Returns:
            Imported prompt template
        """
        if isinstance(template_data, str):
            if format.lower() == "json":
                data = json.loads(template_data)
            elif format.lower() == "yaml":
                import yaml

                data = yaml.safe_load(template_data)
            else:
                raise ValueError(f"Unsupported format: {format}")
        else:
            data = template_data

        template = PromptTemplate.from_dict(data)
        self.memory.add_template(template)
        self._save_template(template)

        return template

    def _save_template(self, template: PromptTemplate) -> None:
        """Save a template to storage."""
        template_file = (
            self.storage_path / f"{template.template_name}_{template.version_id}.json"
        )
        with open(template_file, "w") as f:
            json.dump(template.to_dict(), f, indent=2)

    def _load_templates(self) -> None:
        """Load templates from storage."""
        if not self.storage_path.exists():
            return

        for template_file in self.storage_path.glob("*.json"):
            try:
                with open(template_file, "r") as f:
                    data = json.load(f)
                template = PromptTemplate.from_dict(data)
                self.memory.add_template(template)
            except Exception as e:
                print(f"Error loading template from {template_file}: {e}")

    def _save_memory_state(self) -> None:
        """Save memory state to storage."""
        state_file = self.storage_path / "memory_state.json"
        state = {
            "current_versions": self.memory.current_versions,
            "agent_templates": {
                agent_type.value: templates
                for agent_type, templates in self.memory.agent_templates.items()
            },
            "total_usage": self.memory.total_usage,
            "success_rate": self.memory.success_rate,
        }
        with open(state_file, "w") as f:
            json.dump(state, f, indent=2)

    def _remove_template_file(self, template_name: str, version_id: str) -> None:
        """Remove a specific template file."""
        template_file = self.storage_path / f"{template_name}_{version_id}.json"
        if template_file.exists():
            template_file.unlink()

    def _remove_template_files(self, template_name: str) -> None:
        """Remove all files for a template."""
        for template_file in self.storage_path.glob(f"{template_name}_*.json"):
            template_file.unlink()
