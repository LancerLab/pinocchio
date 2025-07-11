"""
Prompt template formatter for Pinocchio.

This module provides functionality for formatting prompt templates.
"""

import json
import re
from typing import Any, Dict, List, Optional


class TemplateFormatter:
    """
    Template formatter for prompt templates.

    Handles variable substitution and template inheritance.
    """

    def __init__(self) -> None:
        """Initialize the template formatter."""
        # Regex for variable substitution: {{variable_name}}
        self.variable_pattern = re.compile(r"{{(.*?)}}")

    def format_string(self, template_str: str, variables: Dict[str, Any]) -> str:
        """
        Format a template string by substituting variables.

        Args:
            template_str: The template string with variables in {{variable}} format
            variables: Dictionary of variable names and values

        Returns:
            The formatted string with variables substituted
        """

        def replace_var(match: re.Match[str]) -> str:
            var_name = match.group(1).strip()
            if var_name in variables:
                value = variables[var_name]
                # Handle different types of values
                if isinstance(value, (dict, list)):
                    return json.dumps(value, indent=2)
                return str(value)
            # Keep the variable placeholder if not found
            return match.group(0)

        return self.variable_pattern.sub(replace_var, template_str)

    def format_template(
        self,
        template_content: str,
        variables: Dict[str, Any],
        templates: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Format a template with variable substitution and template inheritance.

        Args:
            template_content: The template content
            variables: Dictionary of variable names and values
            templates: Optional dictionary of template names to their content for inheritance

        Returns:
            The formatted template
        """
        # Process template inheritance if templates are provided
        if templates:
            # Simple include tag processing: {% include 'template_name' %}
            include_pattern = re.compile(r"{%\s*include\s+\'(.*?)\'\s*%}")

            def replace_include(match: re.Match[str]) -> str:
                template_name = match.group(1)
                if template_name in templates:
                    return templates[template_name]
                # Keep the include tag if template not found
                return match.group(0)

            template_content = include_pattern.sub(replace_include, template_content)

        # Process variable substitution
        return self.format_string(template_content, variables)


def format_prompt(template: str, variables: Dict[str, Any]) -> str:
    """Format a prompt template with variables."""
    return template.format(**variables)


def format_code_block(code: str) -> str:
    """Format a code block for display."""
    return f"```\n{code}\n```"


def format_table(headers: List[str], rows: List[List[Any]]) -> str:
    """Format a table for display."""
    table = "\t".join(headers) + "\n"
    for row in rows:
        table += "\t".join(str(cell) for cell in row) + "\n"
    return table
