"""
Pinocchio prompt template models.

This module defines the data models for prompt templates, including structured
input/output models, multi-agent support, and version control.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class AgentType(Enum):
    """Agent types for prompt templates."""

    GENERATOR = "generator"
    DEBUGGER = "debugger"
    EVALUATOR = "evaluator"
    OPTIMIZER = "optimizer"


class PromptType(Enum):
    """Prompt types for different use cases."""

    CODE_GENERATION = "code_generation"
    CODE_DEBUGGING = "code_debugging"
    CODE_EVALUATION = "code_evaluation"
    CODE_OPTIMIZATION = "code_optimization"
    KNOWLEDGE_EXTRACTION = "knowledge_extraction"
    PERFORMANCE_ANALYSIS = "performance_analysis"


@dataclass
class StructuredInput:
    """Structured input model for prompts."""

    code_snippet: Optional[str] = None
    requirements: Optional[Dict[str, Any]] = None
    context: Optional[Dict[str, Any]] = None
    constraints: Optional[List[str]] = None
    performance_metrics: Optional[Dict[str, float]] = None
    optimization_targets: Optional[List[str]] = None
    debug_info: Optional[Dict[str, Any]] = None
    evaluation_criteria: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "code_snippet": self.code_snippet,
            "requirements": self.requirements,
            "context": self.context,
            "constraints": self.constraints,
            "performance_metrics": self.performance_metrics,
            "optimization_targets": self.optimization_targets,
            "debug_info": self.debug_info,
            "evaluation_criteria": self.evaluation_criteria,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StructuredInput":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class StructuredOutput:
    """Structured output model for prompts."""

    generated_code: Optional[str] = None
    debug_suggestions: Optional[List[str]] = None
    evaluation_results: Optional[Dict[str, Any]] = None
    optimization_suggestions: Optional[List[str]] = None
    performance_improvements: Optional[Dict[str, float]] = None
    knowledge_fragments: Optional[List[Dict[str, Any]]] = None
    confidence_score: Optional[float] = None
    execution_time: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "generated_code": self.generated_code,
            "debug_suggestions": self.debug_suggestions,
            "evaluation_results": self.evaluation_results,
            "optimization_suggestions": self.optimization_suggestions,
            "performance_improvements": self.performance_improvements,
            "knowledge_fragments": self.knowledge_fragments,
            "confidence_score": self.confidence_score,
            "execution_time": self.execution_time,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StructuredOutput":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class PromptTemplate:
    """Prompt template model with structured input/output support."""

    template_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    template_name: str = ""
    version_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    agent_type: AgentType = AgentType.GENERATOR
    prompt_type: PromptType = PromptType.CODE_GENERATION

    # Template content
    content: str = ""
    description: str = ""

    # Structured models
    input_schema: Optional[StructuredInput] = None
    output_schema: Optional[StructuredOutput] = None

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    tags: List[str] = field(default_factory=list)
    priority: int = 1
    optimization_level: float = field(default=1.0)

    # Performance tracking
    usage_count: int = 0
    success_rate: float = 0.0
    average_response_time: float = 0.0

    # Version control
    parent_version_id: Optional[str] = None
    change_log: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Initialize default schemas if not provided."""
        if self.input_schema is None:
            self.input_schema = StructuredInput()
        if self.output_schema is None:
            self.output_schema = StructuredOutput()

    @classmethod
    def create_new_version(
        cls,
        template_name: str,
        content: str,
        agent_type: AgentType = AgentType.GENERATOR,
        prompt_type: PromptType = PromptType.CODE_GENERATION,
        description: str = "",
        parent_version_id: Optional[str] = None,
        input_schema: Optional[StructuredInput] = None,
        output_schema: Optional[StructuredOutput] = None,
        tags: Optional[List[str]] = None,
    ) -> "PromptTemplate":
        """Create a new version of a prompt template."""
        return cls(
            template_name=template_name,
            content=content,
            agent_type=agent_type,
            prompt_type=prompt_type,
            description=description,
            parent_version_id=parent_version_id,
            input_schema=input_schema,
            output_schema=output_schema,
            tags=tags or [],
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert template to dictionary."""
        from dataclasses import asdict

        result = asdict(self)
        result = self._convert_datetime_fields(result)
        result = self._convert_enum_fields(result)
        result = self._convert_schema_fields(result)
        result = self._ensure_optimization_level(result)
        return result

    def _convert_datetime_fields(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Convert datetime objects to ISO strings."""
        if "created_at" in result and isinstance(result["created_at"], datetime):
            result["created_at"] = result["created_at"].isoformat()
        if "updated_at" in result and isinstance(result["updated_at"], datetime):
            result["updated_at"] = result["updated_at"].isoformat()
        return result

    def _convert_enum_fields(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Convert enum values to strings."""
        if "agent_type" in result and hasattr(result["agent_type"], "value"):
            result["agent_type"] = result["agent_type"].value
        if "prompt_type" in result and hasattr(result["prompt_type"], "value"):
            result["prompt_type"] = result["prompt_type"].value
        return result

    def _convert_schema_fields(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Convert structured schemas."""
        if "input_schema" in result and result["input_schema"]:
            if hasattr(result["input_schema"], "to_dict"):
                result["input_schema"] = result["input_schema"].to_dict()
        if "output_schema" in result and result["output_schema"]:
            if hasattr(result["output_schema"], "to_dict"):
                result["output_schema"] = result["output_schema"].to_dict()
        return result

    def _ensure_optimization_level(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure optimization_level is float."""
        if "optimization_level" in result and not isinstance(
            result["optimization_level"], float
        ):
            try:
                result["optimization_level"] = float(result["optimization_level"])
            except Exception:
                result["optimization_level"] = 1.0
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PromptTemplate":
        """Create template from dictionary."""
        # Convert string enums back to enum objects
        if "agent_type" in data and isinstance(data["agent_type"], str):
            data["agent_type"] = AgentType(data["agent_type"])
        if "prompt_type" in data and isinstance(data["prompt_type"], str):
            data["prompt_type"] = PromptType(data["prompt_type"])

        # Convert datetime strings back to datetime objects
        if "created_at" in data and isinstance(data["created_at"], str):
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        if "updated_at" in data and isinstance(data["updated_at"], str):
            data["updated_at"] = datetime.fromisoformat(data["updated_at"])

        # Convert schemas
        if "input_schema" in data and data["input_schema"]:
            data["input_schema"] = StructuredInput.from_dict(data["input_schema"])
        if "output_schema" in data and data["output_schema"]:
            data["output_schema"] = StructuredOutput.from_dict(data["output_schema"])

        return cls(**data)

    def update_usage_stats(self, success: bool, response_time: float) -> None:
        """Update usage statistics."""
        self.usage_count += 1
        self.updated_at = datetime.now()

        # Update success rate
        if self.usage_count == 1:
            self.success_rate = 1.0 if success else 0.0
        else:
            current_successes = self.success_rate * (self.usage_count - 1)
            if success:
                current_successes += 1
            self.success_rate = current_successes / self.usage_count

        # Update average response time
        if self.usage_count == 1:
            self.average_response_time = response_time
        else:
            total_time = (
                self.average_response_time * (self.usage_count - 1) + response_time
            )
            self.average_response_time = total_time / self.usage_count

    def format(self, **kwargs: Any) -> str:
        """Format the template content with provided variables."""
        content = self.content

        # Simple variable substitution
        for key, value in kwargs.items():
            placeholder = f"{{{{{key}}}}}"
            if placeholder in content:
                content = content.replace(placeholder, str(value))

        return content


@dataclass
class PromptMemory:
    """Memory manager for prompt templates with multi-agent and multi-version support."""

    # Template storage: {template_name: {version_id: PromptTemplate}}
    templates: Dict[str, Dict[str, PromptTemplate]] = field(default_factory=dict)

    # Current version tracking: {template_name: version_id}
    current_versions: Dict[str, str] = field(default_factory=dict)

    # Agent-specific template mappings: {agent_type: {template_name: version_id}}
    agent_templates: Dict[AgentType, Dict[str, str]] = field(default_factory=dict)

    # Performance tracking
    total_usage: int = 0
    success_rate: float = 0.0

    def add_template(self, template: PromptTemplate) -> str:
        """Add a template to memory."""
        if template.template_name not in self.templates:
            self.templates[template.template_name] = {}

        self.templates[template.template_name][template.version_id] = template

        # Always set as current version (latest version wins)
        self.current_versions[template.template_name] = template.version_id

        # Update agent-specific mappings
        if template.agent_type not in self.agent_templates:
            self.agent_templates[template.agent_type] = {}
        self.agent_templates[template.agent_type][
            template.template_name
        ] = template.version_id

        return template.version_id

    def get_template(
        self, template_name: str, version_id: Optional[str] = None
    ) -> Optional[PromptTemplate]:
        """Get a template by name and optional version."""
        if template_name not in self.templates:
            return None

        if version_id is None:
            # Get current version
            if template_name not in self.current_versions:
                return None
            version_id = self.current_versions[template_name]

        return self.templates[template_name].get(version_id)

    def get_template_by_agent(
        self, agent_type: AgentType, template_name: str
    ) -> Optional[PromptTemplate]:
        """Get a template by agent type and name."""
        if agent_type not in self.agent_templates:
            return None

        version_id = self.agent_templates[agent_type].get(template_name)
        if version_id is None:
            return None

        return self.get_template(template_name, version_id)

    def list_templates(self) -> Dict[str, str]:
        """List all templates with their current versions."""
        return self.current_versions.copy()

    def list_templates_by_agent(self, agent_type: AgentType) -> Dict[str, str]:
        """List templates for a specific agent type."""
        return self.agent_templates.get(agent_type, {}).copy()

    def list_template_versions(self, template_name: str) -> Dict[str, PromptTemplate]:
        """List all versions of a template."""
        return self.templates.get(template_name, {}).copy()

    def set_current_version(self, template_name: str, version_id: str) -> bool:
        """Set the current version of a template."""
        if template_name not in self.templates:
            return False

        if version_id not in self.templates[template_name]:
            return False

        self.current_versions[template_name] = version_id
        return True

    def remove_template(
        self, template_name: str, version_id: Optional[str] = None
    ) -> bool:
        """Remove a template or specific version."""
        if template_name not in self.templates:
            return False

        if version_id is None:
            # Remove all versions
            del self.templates[template_name]
            if template_name in self.current_versions:
                del self.current_versions[template_name]

            # Remove from agent mappings
            for agent_type in self.agent_templates:
                if template_name in self.agent_templates[agent_type]:
                    del self.agent_templates[agent_type][template_name]
            return True

        # Remove specific version
        if version_id not in self.templates[template_name]:
            return False

        del self.templates[template_name][version_id]

        # Update current version if needed
        if (
            template_name in self.current_versions
            and self.current_versions[template_name] == version_id
        ):
            if self.templates[template_name]:
                # Set to the most recent version
                latest_version = max(self.templates[template_name].keys())
                self.current_versions[template_name] = latest_version
            else:
                del self.current_versions[template_name]

        return True

    def search_templates(
        self, query: str, agent_type: Optional[AgentType] = None
    ) -> List[PromptTemplate]:
        """Search templates by content, description, or tags."""
        results = []

        templates_to_search = []
        if agent_type is not None:
            # Search only templates for specific agent
            agent_templates = self.agent_templates.get(agent_type, {})
            for template_name, version_id in agent_templates.items():
                template = self.get_template(template_name, version_id)
                if template:
                    templates_to_search.append(template)
        else:
            # Search all templates
            for template_name in self.templates:
                for template in self.templates[template_name].values():
                    templates_to_search.append(template)

        query_lower = query.lower()
        for template in templates_to_search:
            if (
                query_lower in template.content.lower()
                or query_lower in template.description.lower()
                or any(query_lower in tag.lower() for tag in template.tags)
            ):
                results.append(template)

        return results

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get overall performance statistics."""
        total_usage = 0
        total_successes: float = 0.0
        total_response_time = 0.0
        template_count = 0

        for template_name in self.templates:
            for template in self.templates[template_name].values():
                total_usage += template.usage_count
                total_successes += float(template.success_rate * template.usage_count)
                total_response_time += (
                    template.average_response_time * template.usage_count
                )
                template_count += 1

        return {
            "total_templates": template_count,
            "total_usage": total_usage,
            "overall_success_rate": total_successes / total_usage
            if total_usage > 0
            else 0.0,
            "average_response_time": total_response_time / total_usage
            if total_usage > 0
            else 0.0,
        }

    def record_usage(self, template_name: str, success: bool = True) -> None:
        """Record usage of a template."""
        template = self.get_template(template_name)
        if template:
            template.update_usage_stats(success, 0.0)  # Default response time
            self.total_usage += 1

            # Update overall success rate
            if self.total_usage == 1:
                self.success_rate = 1.0 if success else 0.0
            else:
                current_successes = self.success_rate * (self.total_usage - 1)
                if success:
                    current_successes += 1
                self.success_rate = current_successes / self.total_usage
