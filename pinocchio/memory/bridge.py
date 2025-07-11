"""Memory bridge for Pinocchio."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from ..memory.models import CodeMemory
from ..prompt.models import PromptMemory

if TYPE_CHECKING:
    from ..memory.manager import MemoryManager


@dataclass
class MemoryPromptBridge:
    """Bridge between memory and prompt modules."""

    prompt_memory: PromptMemory
    code_memory: CodeMemory
    memory_manager: Optional["MemoryManager"] = None

    def get_code_context_for_prompt(
        self, session_id: str, version_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get code context for prompt generation."""
        if version_id:
            code_version = self.code_memory.get_version(version_id)
        else:
            code_version = self.code_memory.get_current_version()

        if not code_version:
            return {}

        return {
            "code": code_version.code,
            "language": getattr(code_version, "language", "unknown"),
            "kernel_type": getattr(code_version, "kernel_type", "unknown"),
            "optimization_level": getattr(code_version, "optimization_level", 1.0),
            "performance_metrics": getattr(code_version, "performance_metrics", {}),
        }

    def format_prompt_with_code(
        self,
        template_name: str,
        session_id: str,
        version_id: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        """Format prompt template with code context."""
        # Get code context
        code_context = self.get_code_context_for_prompt(session_id, version_id)

        # Get prompt template
        template = self.prompt_memory.get_template(template_name)
        if not template:
            raise ValueError(f"Template '{template_name}' not found")

        # Merge context
        context = {**code_context, **kwargs}

        # Format template
        return template.format(**context)

    def get_agent_history(
        self, session_id: str, agent_type: str
    ) -> List[Dict[str, Any]]:
        """Get agent interaction history for prompt context."""
        # This would typically query session history
        # For now, return empty list
        return []

    def update_prompt_usage(self, template_name: str, success: bool = True) -> None:
        """Update prompt usage statistics."""
        self.prompt_memory.record_usage(template_name, success)

    def create_debugger_prompt(self, version_id: str) -> Dict[str, Any]:
        """Create a debugger prompt for a specific code version."""
        code_version = self.code_memory.get_version(version_id)
        if not code_version:
            raise ValueError(f"Code version {version_id} not found")

        template = self.prompt_memory.get_template("debugger")
        if not template:
            raise ValueError("Debugger template not found")

        content = template.content.replace("{{code}}", code_version.code)

        return {
            "content": content,
            "template_name": "debugger",
            "version_id": version_id,
            "code": code_version.code,
        }

    def create_optimizer_prompt(self, version_id: str) -> Dict[str, Any]:
        """Create an optimizer prompt with agent history."""
        code_version = self.code_memory.get_version(version_id)
        if not code_version:
            raise ValueError(f"Code version {version_id} not found")

        template = self.prompt_memory.get_template("optimizer")
        if not template:
            raise ValueError("Optimizer template not found")

        # Get agent history
        agent_memories = []
        if self.memory_manager:
            # Query memories for debugger and generator agents separately
            debugger_memories = self.memory_manager.query_agent_memories(
                session_id=code_version.session_id,
                agent_type="debugger",
            )
            generator_memories = self.memory_manager.query_agent_memories(
                session_id=code_version.session_id,
                agent_type="generator",
            )
            agent_memories = debugger_memories + generator_memories

        # Extract previous issues
        previous_issues = []
        for memory in agent_memories:
            if hasattr(memory, "identified_issues"):
                previous_issues.extend(
                    [issue.get("issue", "") for issue in memory.identified_issues]
                )
            elif hasattr(memory, "errors"):
                previous_issues.extend(memory.errors)

        previous_issues_text = "\n".join([f"- {issue}" for issue in previous_issues])

        content = template.content.replace("{{code}}", code_version.code)
        content = content.replace("{{previous_issues}}", previous_issues_text)

        return {
            "content": content,
            "template_name": "optimizer",
            "version_id": version_id,
            "code": code_version.code,
            "previous_issues": previous_issues_text,
        }
