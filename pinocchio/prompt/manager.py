"""
Pinocchio prompt manager.

This module provides a comprehensive prompt management system with support for
multi-agent templates, version control, performance tracking, and structured I/O.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pinocchio.agents.debugger import DebuggerAgent
from pinocchio.agents.evaluator import EvaluatorAgent
from pinocchio.agents.generator import GeneratorAgent
from pinocchio.agents.optimizer import OptimizerAgent

from ..utils.file_utils import ensure_directory, safe_read_json, safe_write_json
from ..utils.temp_utils import cleanup_temp_files, create_temp_file
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
        self.storage_path = Path(storage_path) if storage_path else Path("./prompts")
        # Use utils function to ensure directory exists
        ensure_directory(self.storage_path)

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
        # Use utils function for safe JSON writing
        success = safe_write_json(template.to_dict(), template_file)
        if not success:
            raise RuntimeError(f"Failed to save template {template.template_name}")

    def _load_templates(self) -> None:
        """Load templates from storage."""
        if not self.storage_path.exists():
            return

        for template_file in self.storage_path.glob("*.json"):
            try:
                data = safe_read_json(template_file)
                if data is not None:
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
        # Use utils function for safe JSON writing
        success = safe_write_json(state, state_file)
        if not success:
            raise RuntimeError("Failed to save memory state")

    def integrate_memory_and_knowledge(
        self, memory_manager=None, knowledge_manager=None
    ):
        """
        Integrate memory and knowledge managers for enhanced prompt generation.

        Args:
            memory_manager: MemoryManager instance
            knowledge_manager: KnowledgeManager instance
        """
        self.memory_manager = memory_manager
        self.knowledge_manager = knowledge_manager

    def format_template_with_context(
        self,
        template_name: str,
        variables: Dict[str, Any],
        session_id: Optional[str] = None,
        agent_type: Optional[AgentType] = None,
        include_memory: bool = True,
        include_knowledge: bool = True,
        memory_keywords: Optional[List[str]] = None,
        knowledge_keywords: Optional[List[str]] = None,
        step_id: Optional[str] = None,
        response: Optional[str] = None,
    ) -> Optional[str]:
        """
        Format a prompt template with enhanced context from memory and knowledge.

        Args:
            template_name: Name of the template
            variables: Variables to substitute
            session_id: Session ID for memory context
            agent_type: Agent type filter
            include_memory: Whether to include memory context
            include_knowledge: Whether to include knowledge context
            memory_keywords: Keywords for memory search
            knowledge_keywords: Keywords for knowledge search

        Returns:
            Formatted template with enhanced context
        """
        # Get base template
        template = self.get_template(template_name, agent_type=agent_type)
        if template is None:
            return None

        # Prepare enhanced variables
        enhanced_variables = variables.copy()

        # Add memory context if available and requested
        if include_memory and self.memory_manager and session_id and memory_keywords:
            memory_context = self.memory_manager.query_memories_by_keywords(
                session_id=session_id, keywords=memory_keywords, limit=5
            )
            enhanced_variables["memory_context"] = memory_context
            enhanced_variables[
                "previous_interactions"
            ] = self._format_memory_for_prompt(memory_context)

        # Add knowledge context if available and requested
        if include_knowledge and self.knowledge_manager and knowledge_keywords:
            knowledge_context = self.knowledge_manager.query_by_keywords(
                keywords=knowledge_keywords, session_id=session_id, limit=3
            )
            enhanced_variables["knowledge_context"] = knowledge_context
            enhanced_variables[
                "relevant_knowledge"
            ] = self._format_knowledge_for_prompt(knowledge_context)

        # Format template with enhanced context
        prompt = self.formatter.format_template(template.input_schema.requirements, enhanced_variables)
        
        # 读取./prompt_storage/common_knowledge/cuda_example.cu文件的代码，将其作为cuda代码拼接在prompt后面
        cuda_example_path = Path("./prompt_storage/common_knowledge/cuda_example.cu")
        if cuda_example_path.exists():
            with open(cuda_example_path, "r", encoding="utf-8") as f:
                cuda_code = f.read()
            prompt += f"\n\n```cuda\n{cuda_code}\n```"
        
        # 读取./prompt_storage/common_knowledge/hardware_specification.json的文件，将其作为对GPU硬件规格的描述拼接在prompt后面
        hardware_spec_path = Path("./prompt_storage/common_knowledge/hardware_specification.json")
        if hardware_spec_path.exists():
            with open(hardware_spec_path, "r", encoding="utf-8") as f:
                hardware_spec = json.load(f)
            hardware_description = json.dumps(hardware_spec, indent=2)
            prompt += f"\n\n## GPU Hardware Specification:\n{hardware_description}"
        
        # New: Save prompt and response to storage_path/{session_id}/{agent_type}/{step_id}_prompt.txt
        if prompt and session_id and agent_type:
            try:
                agent_type_str = str(agent_type).lower()
                storage_dir = self.storage_path / session_id / agent_type_str
                storage_dir.mkdir(parents=True, exist_ok=True)
                fname = f"{step_id or 'unknown'}"
                prompt_fpath = storage_dir / f"{fname}_prompt.txt"
                with open(prompt_fpath, "w", encoding="utf-8") as f:
                    f.write(prompt)
                # Synchronously save response
                if response:
                    resp_fpath = storage_dir / f"{fname}_response.txt"
                    with open(resp_fpath, "w", encoding="utf-8") as f:
                        f.write(response)
            except Exception as e:
                import logging

                logging.warning(f"Failed to save prompt/response for {session_id}: {e}")
        return prompt

    def _format_memory_for_prompt(self, memory_context: List[Dict[str, Any]]) -> str:
        """Format memory context for inclusion in prompts."""
        if not memory_context:
            return "No relevant previous interactions found."

        formatted_parts = ["## Previous Interactions:"]

        for memory in memory_context:
            agent_type = memory.get("agent_type", "unknown")
            memory_type = memory.get("type", "interaction")
            timestamp = memory.get("timestamp", "unknown")

            formatted_parts.append(
                f"\n### {agent_type.title()} ({memory_type}) - {timestamp}"
            )

            if memory.get("input_summary"):
                formatted_parts.append(f"Input: {memory['input_summary']}")

            if memory.get("output_summary"):
                formatted_parts.append(f"Output: {memory['output_summary']}")

            # Add specific details based on memory type
            if memory_type == "generation" and memory.get("optimization_techniques"):
                formatted_parts.append(
                    f"Optimization techniques: {', '.join(memory['optimization_techniques'])}"
                )

            elif memory_type == "debugging" and memory.get("errors"):
                formatted_parts.append(
                    f"Errors found: {', '.join(memory['errors'][:2])}"
                )

            elif memory_type == "evaluation" and memory.get("bottlenecks"):
                formatted_parts.append(
                    f"Bottlenecks: {', '.join(memory['bottlenecks'][:2])}"
                )

        return "\n".join(formatted_parts)

    def _format_knowledge_for_prompt(
        self, knowledge_context: List[Dict[str, Any]]
    ) -> str:
        """Format knowledge context for inclusion in prompts."""
        if not knowledge_context:
            return "No relevant knowledge fragments found."

        formatted_parts = ["## Relevant Knowledge:"]

        for fragment in knowledge_context:
            title = fragment.get("title", "Unknown")
            content = fragment.get("content", "")
            category = fragment.get("category", "general")

            formatted_parts.append(f"\n### {title} ({category})")
            formatted_parts.append(content)

        return "\n".join(formatted_parts)

    AGENT_CLASS_MAP = {
        "generator": GeneratorAgent,
        "debugger": DebuggerAgent,
        "optimizer": OptimizerAgent,
        "evaluator": EvaluatorAgent,
    }

    def get_output_template_for_agent(self, agent_type):
        """Get the output template string for the specified agent type."""
        agent_type_str = (
            agent_type.value if hasattr(agent_type, "value") else str(agent_type)
        )
        agent_class = self.AGENT_CLASS_MAP.get(agent_type_str.lower())
        if not agent_class:
            return ""
        try:
            agent = agent_class(llm_client=None)
        except Exception:
            from pinocchio.agents.base import Agent

            agent = Agent(agent_type_str, llm_client=None)
        # Dynamically select method
        if agent_type_str.lower() == "generator" and hasattr(
            agent, "_get_generation_output_format"
        ):
            return agent._get_generation_output_format()
        elif agent_type_str.lower() == "debugger" and hasattr(
            agent, "_get_debugging_output_format"
        ):
            return agent._get_debugging_output_format()
        elif agent_type_str.lower() == "optimizer" and hasattr(
            agent, "_get_optimization_output_format"
        ):
            return agent._get_optimization_output_format()
        elif agent_type_str.lower() == "evaluator" and hasattr(
            agent, "_get_evaluation_output_format"
        ):
            return agent._get_evaluation_output_format()
        elif hasattr(agent, "_get_output_format"):
            return agent._get_output_format()
        else:
            return ""

    def create_context_aware_prompt(
        self,
        agent_type: Any,  # allow str or AgentType
        task_description: str,
        user_request: Optional[str] = None,
        session_id: Optional[str] = None,
        code: Optional[str] = None,
        keywords: Optional[List[str]] = None,
        previous_results: Optional[dict] = None,
        session_obj: Optional[Any] = None,
    ) -> str:
        """
        Create a context-aware prompt with memory and knowledge integration.
        """
        from ..data_models.task_planning import AgentType as PlanningAgentType

        if isinstance(agent_type, str):
            agent_type = PlanningAgentType[agent_type.upper()]

        template_map = {
            PlanningAgentType.GENERATOR: "cuda_generator",
            PlanningAgentType.OPTIMIZER: "cuda_optimizer",
            PlanningAgentType.DEBUGGER: "cuda_debugger",
            PlanningAgentType.EVALUATOR: "cuda_evaluator",
        }
        template_name = template_map.get(agent_type, "default")
        variables = {
            "task_description": task_description,
            "agent_type": agent_type.value,
        }
        # code fallback: always a string
        if code is None and session_obj is not None:
            code = session_obj.get_latest_code()
        if code is None:
            code = ""
        variables["code"] = code
        if keywords is None:
            keywords = self._extract_keywords(task_description, code)
        prompt_main = self.format_template_with_context(
            template_name=template_name,
            variables=variables,
            session_id=session_id,
            agent_type=agent_type,
            memory_keywords=keywords,
            knowledge_keywords=keywords,
        ) or self._create_fallback_prompt(agent_type, task_description, code)

        # Concatenate previous_results structured artifacts
        previous_block = ""
        if previous_results:
            previous_block += "\n\n## Previous Agent Results:"
            for task_id, result in previous_results.items():
                if not isinstance(result, dict):
                    continue
                agent_type_str = result.get("agent_type", "unknown")
                output = result.get("output", result)
                code_str = output.get("code", "") if isinstance(output, dict) else ""
                explanation = (
                    output.get("explanation", "") if isinstance(output, dict) else ""
                )
                optimization_techniques = (
                    output.get("optimization_techniques", [])
                    if isinstance(output, dict)
                    else []
                )
                previous_block += f"\n### From {task_id} ({agent_type_str}):"
                if code_str:
                    previous_block += f"\n```cuda\n{code_str}\n```"
                if explanation:
                    previous_block += f"\nExplanation: {explanation}"
                if optimization_techniques:
                    previous_block += f"\nOptimization Techniques: {', '.join(optimization_techniques)}"
        # Concatenate output template
        output_template = self.get_output_template_for_agent(agent_type)
        prompt = prompt_main.strip() + previous_block + "\n\n" + output_template.strip()
        return prompt

    def create_context_aware_request(
        self,
        agent_type: Any,  # allow str or AgentType
        task_description: str,
        session_id: Optional[str] = None,
        code: Optional[str] = None,
        keywords: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None,
        previous_results: Optional[Dict[str, Any]] = None,
        session_obj: Optional[Any] = None,
        step_id: Optional[str] = None,
        response: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Build a structured agent request with memory, knowledge, and context integration.
        Args:
            agent_type: Agent type (generator, debugger, optimizer, evaluator) (str or AgentType)
            task_description: Task description string
            session_id: Session ID for context
            code: Optional code string for context
            keywords: Optional keywords for memory/knowledge search
            context: Additional context dict (optional)
            previous_results: Previous task results (optional)
            session_obj: Optional session object for code retrieval
        Returns:
            Structured request dict for agent execution
        """
        # Ensure agent_type is AgentType enum
        from ..data_models.task_planning import AgentType as PlanningAgentType

        if isinstance(agent_type, str):
            agent_type = PlanningAgentType[agent_type.upper()]
        # Build context-aware prompt
        prompt = self.create_context_aware_prompt(
            agent_type=agent_type,
            task_description=task_description,
            session_id=session_id,
            code=code,
            keywords=keywords,
            previous_results=previous_results,
            session_obj=session_obj,
        )
        if code is None and session_obj is not None:
            code = session_obj.get_latest_code()
        request = {
            "agent_type": agent_type,  # keep as enum type
            "task_description": task_description,
            "session_id": session_id,
            "code": code,
            "keywords": keywords,
            "context": context or {},
            "previous_results": previous_results or {},
            "prompt": prompt,
        }
        return request

    def _extract_keywords(
        self, task_description: str, code: Optional[str] = None
    ) -> List[str]:
        """Extract relevant keywords from task description and code."""
        keywords = []

        # Extract from task description
        cuda_terms = [
            "cuda",
            "gpu",
            "kernel",
            "memory",
            "optimization",
            "performance",
            "parallel",
            "thread",
            "block",
            "grid",
            "shared",
            "global",
        ]

        task_lower = task_description.lower()
        for term in cuda_terms:
            if term in task_lower:
                keywords.append(term)

        # Extract from code if provided
        if code:
            code_lower = code.lower()
            code_terms = [
                "__global__",
                "__shared__",
                "__device__",
                "cudamalloc",
                "cudamemcpy",
                "blockdim",
                "griddim",
                "threadidx",
                "blockidx",
            ]

            for term in code_terms:
                if term in code_lower:
                    keywords.append(term.replace("__", ""))

        return list(set(keywords))  # Remove duplicates

    def _create_fallback_prompt(
        self, agent_type: AgentType, task_description: str, code: Optional[str] = None
    ) -> str:
        """Create a fallback prompt when template is not available."""
        base_prompt = (
            f"You are a {agent_type.value} agent specializing in CUDA programming.\n\n"
        )
        base_prompt += f"Task: {task_description}\n\n"

        if code:
            base_prompt += f"Code to analyze:\n```cuda\n{code}\n```\n\n"

        base_prompt += "Please provide your response in JSON format with appropriate fields for your agent type."

        return base_prompt

    def _remove_template_file(self, template_name: str, version_id: str) -> None:
        """Remove a specific template file."""
        template_file = self.storage_path / f"{template_name}_{version_id}.json"
        if template_file.exists():
            template_file.unlink()

    def _remove_template_files(self, template_name: str) -> None:
        """Remove all files for a template."""
        for template_file in self.storage_path.glob(f"{template_name}_*.json"):
            template_file.unlink()
