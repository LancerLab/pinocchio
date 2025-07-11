"""
Knowledge utilities for Pinocchio multi-agent system.

This module provides utility functions for knowledge fragment processing,
conversion, and analysis.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from .models.knowledge import KnowledgeCategory, KnowledgeContentType, KnowledgeFragment


class KnowledgeUtils:
    """
    Utility class for knowledge fragment processing and conversion.
    """

    @staticmethod
    def _parse_headers(line: str, result: Dict[str, Any]) -> None:
        """Parse markdown headers."""
        stripped = line.lstrip()
        if stripped.startswith("#"):
            level = len(stripped) - len(stripped.lstrip("#"))
            header_text = stripped.lstrip("#").strip()
            result["headers"].append({"level": level, "text": header_text})

    @staticmethod
    def _parse_code_blocks(
        line: str,
        current_code_block: List[str],
        in_code_block: bool,
        result: Dict[str, Any],
    ) -> tuple[List[str], bool]:
        """Parse markdown code blocks."""
        stripped = line.lstrip()
        if stripped.startswith("```"):
            if in_code_block:
                # End of code block
                if current_code_block:
                    result["code_blocks"].append("\n".join(current_code_block))
                current_code_block = []
                in_code_block = False
            else:
                # Start of code block
                in_code_block = True
        elif in_code_block:
            current_code_block.append(line)
        return current_code_block, in_code_block

    @staticmethod
    def _parse_lists(
        line: str, current_list: List[str], result: Dict[str, Any]
    ) -> List[str]:
        """Parse markdown lists."""
        stripped = line.lstrip()
        if stripped.startswith(("-", "*", "1.", "2.", "3.")):
            current_list.append(stripped)
        elif current_list and stripped:
            # Continue list
            current_list.append(stripped)
        elif current_list:
            # End of list
            result["lists"].append(current_list)
            current_list = []
        return current_list

    @staticmethod
    def parse_markdown_content(content: str) -> Dict[str, Any]:
        """
        Parse markdown content and extract structured information.

        Args:
            content: Markdown content string

        Returns:
            Dictionary with parsed markdown structure
        """
        result: Dict[str, Any] = {
            "headers": [],
            "code_blocks": [],
            "lists": [],
            "text_content": "",
        }

        lines = content.split("\n")
        current_code_block: List[str] = []
        in_code_block = False
        current_list: List[str] = []

        for line in lines:
            KnowledgeUtils._parse_headers(line, result)
            current_code_block, in_code_block = KnowledgeUtils._parse_code_blocks(
                line, current_code_block, in_code_block, result
            )
            current_list = KnowledgeUtils._parse_lists(line, current_list, result)

            # Regular text
            if not in_code_block and not line.lstrip().startswith(
                ("#", "```", "-", "*", "1.", "2.", "3.")
            ):
                result["text_content"] += line + "\n"

        # Add any remaining list
        if current_list:
            result["lists"].append(current_list)

        return result

    @staticmethod
    def extract_code_from_markdown(content: str) -> List[str]:
        """
        Extract code blocks from markdown content.

        Args:
            content: Markdown content string

        Returns:
            List of code block strings
        """
        code_blocks = []
        lines = content.split("\n")
        current_block: List[str] = []
        in_code_block = False

        for line in lines:
            if line.startswith("```"):
                if in_code_block:
                    # End of code block
                    if current_block:
                        code_blocks.append("\n".join(current_block))
                    current_block = []
                    in_code_block = False
                else:
                    # Start of code block
                    in_code_block = True
            elif in_code_block:
                current_block.append(line)

        return code_blocks

    @staticmethod
    def convert_to_structured_json(fragment: KnowledgeFragment) -> Dict[str, Any]:
        """
        Convert a knowledge fragment to structured JSON format.

        Args:
            fragment: The knowledge fragment

        Returns:
            Structured JSON representation
        """

        def get_category_value(cat: Any) -> Any:
            return cat.value if hasattr(cat, "value") else cat

        if fragment.content_type == KnowledgeContentType.JSON:
            return fragment.content if isinstance(fragment.content, dict) else {}

        elif fragment.content_type == KnowledgeContentType.MARKDOWN:
            parsed = KnowledgeUtils.parse_markdown_content(str(fragment.content))
            return {
                "title": fragment.title,
                "category": get_category_value(fragment.category),
                "version": fragment.version,
                "parsed_content": parsed,
                "metadata": fragment.metadata,
            }

        else:
            return {
                "title": fragment.title,
                "category": get_category_value(fragment.category),
                "version": fragment.version,
                "content": str(fragment.content),
                "metadata": fragment.metadata,
            }

    @staticmethod
    def merge_fragments(
        fragments: List[KnowledgeFragment],
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Merge multiple knowledge fragments by category.
        """
        merged: Dict[str, List[Dict[str, Any]]] = {
            "optimization_techniques": [],
            "domain_knowledge": [],
            "dsl_syntax": [],
            "other": [],
        }

        for fragment in fragments:
            cat = (
                fragment.category.value
                if hasattr(fragment.category, "value")
                else fragment.category
            )
            if cat == "optimization":
                merged["optimization_techniques"].append(
                    {
                        "title": fragment.title,
                        **(
                            fragment.content
                            if isinstance(fragment.content, dict)
                            else {}
                        ),
                    }
                )
            elif cat == "domain":
                merged["domain_knowledge"].append(
                    {
                        "title": fragment.title,
                        **(
                            fragment.content
                            if isinstance(fragment.content, dict)
                            else {}
                        ),
                    }
                )
            elif cat == "dsl":
                merged["dsl_syntax"].append(
                    {
                        "title": fragment.title,
                        **(
                            fragment.content
                            if isinstance(fragment.content, dict)
                            else {}
                        ),
                    }
                )
            else:
                merged["other"].append(
                    {
                        "title": fragment.title,
                        **(
                            fragment.content
                            if isinstance(fragment.content, dict)
                            else {}
                        ),
                    }
                )

        return merged

    @staticmethod
    def create_optimization_knowledge(
        technique: str,
        description: str,
        parameters: Dict[str, Any],
        examples: List[str],
        session_id: Optional[str] = None,
        agent_type: Optional[str] = None,
    ) -> KnowledgeFragment:
        """
        Create optimization knowledge fragment.

        Args:
            technique: Optimization technique name
            description: Technique description
            parameters: Technique parameters
            examples: Code examples
            session_id: Session ID
            agent_type: Agent type

        Returns:
            Knowledge fragment for optimization technique
        """
        content = {
            "technique": technique,
            "description": description,
            "parameters": parameters,
            "examples": examples,
            "metadata": {
                "type": "optimization_technique",
                "created_at": str(datetime.utcnow()),
            },
        }

        return KnowledgeFragment.create_fragment(
            session_id=session_id,
            agent_type=agent_type,
            category=KnowledgeCategory.OPTIMIZATION,
            title=f"Optimization Technique: {technique}",
            content=content,
            content_type=KnowledgeContentType.JSON,
        )

    @staticmethod
    def create_domain_knowledge(
        domain: str,
        concepts: List[str],
        patterns: List[str],
        session_id: Optional[str] = None,
        agent_type: Optional[str] = None,
    ) -> KnowledgeFragment:
        """
        Create domain knowledge fragment.

        Args:
            domain: Domain name
            concepts: Domain concepts
            patterns: Domain patterns
            session_id: Session ID
            agent_type: Agent type

        Returns:
            Knowledge fragment for domain knowledge
        """
        content = {
            "domain": domain,
            "concepts": concepts,
            "patterns": patterns,
            "metadata": {
                "type": "domain_knowledge",
                "created_at": str(datetime.utcnow()),
            },
        }

        return KnowledgeFragment.create_fragment(
            session_id=session_id,
            agent_type=agent_type,
            category=KnowledgeCategory.DOMAIN,
            title=f"Domain Knowledge: {domain}",
            content=content,
            content_type=KnowledgeContentType.JSON,
        )

    @staticmethod
    def create_dsl_knowledge(
        language: str,
        syntax: Dict[str, Any],
        templates: List[str],
        session_id: Optional[str] = None,
        agent_type: Optional[str] = None,
    ) -> KnowledgeFragment:
        """
        Create DSL knowledge fragment.

        Args:
            language: DSL language name
            syntax: Language syntax rules
            templates: Code templates
            session_id: Session ID
            agent_type: Agent type

        Returns:
            Knowledge fragment for DSL knowledge
        """
        content = {
            "language": language,
            "syntax": syntax,
            "templates": templates,
            "metadata": {"type": "dsl_knowledge", "created_at": str(datetime.utcnow())},
        }

        return KnowledgeFragment.create_fragment(
            session_id=session_id,
            agent_type=agent_type,
            category=KnowledgeCategory.DSL,
            title=f"DSL Knowledge: {language}",
            content=content,
            content_type=KnowledgeContentType.JSON,
        )

    @staticmethod
    def validate_fragment_content(fragment: KnowledgeFragment) -> bool:
        """
        Validate knowledge fragment content.

        Args:
            fragment: The knowledge fragment to validate

        Returns:
            True if content is valid
        """
        try:
            if fragment.content_type == KnowledgeContentType.JSON:
                if not isinstance(fragment.content, dict):
                    return False

                # Check required fields based on category
                if fragment.category == KnowledgeCategory.OPTIMIZATION:
                    required_fields = ["technique", "description"]
                    return all(field in fragment.content for field in required_fields)

                elif fragment.category == KnowledgeCategory.DOMAIN:
                    required_fields = ["domain", "concepts"]
                    return all(field in fragment.content for field in required_fields)

                elif fragment.category == KnowledgeCategory.DSL:
                    required_fields = ["language", "syntax"]
                    return all(field in fragment.content for field in required_fields)

                return True

            elif fragment.content_type == KnowledgeContentType.MARKDOWN:
                return (
                    isinstance(fragment.content, str)
                    and len(fragment.content.strip()) > 0
                )

            else:
                return fragment.content is not None

        except Exception:
            return False

    @staticmethod
    def extract_keywords(fragment: KnowledgeFragment) -> List[str]:
        """
        Extract keywords from a knowledge fragment.

        Args:
            fragment: The knowledge fragment

        Returns:
            List of extracted keywords
        """
        keywords = []

        # Extract from title
        keywords.extend(fragment.title.lower().split())

        # Extract from content
        if fragment.content_type == KnowledgeContentType.JSON:
            if isinstance(fragment.content, dict):
                for key, value in fragment.content.items():
                    if isinstance(value, str):
                        keywords.extend(value.lower().split())
                    elif isinstance(value, list):
                        for item in value:
                            if isinstance(item, str):
                                keywords.extend(item.lower().split())

        elif fragment.content_type == KnowledgeContentType.MARKDOWN:
            keywords.extend(str(fragment.content).lower().split())

        # Remove common words and duplicates
        common_words = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
        }
        keywords = [
            word for word in keywords if word not in common_words and len(word) > 2
        ]

        return list(set(keywords))
