"""
Pinocchio template loader.

This module provides functionality for loading prompt templates from various sources
including files, databases, APIs, and other storage systems.
"""

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import yaml

from .models import PromptTemplate


class TemplateLoader(ABC):
    """Abstract base class for template loaders."""

    @abstractmethod
    def load_template(self, identifier: str) -> Optional[PromptTemplate]:
        """Load a single template by identifier."""
        pass

    @abstractmethod
    def load_templates(self) -> Iterator[PromptTemplate]:
        """Load all available templates."""
        pass

    @abstractmethod
    def list_available(self) -> List[str]:
        """List available template identifiers."""
        pass


class FileTemplateLoader(TemplateLoader):
    """Load templates from file system."""

    def __init__(self, directory: str, file_pattern: str = "*.json"):
        """
        Initialize file template loader.

        Args:
            directory: Directory to search for templates
            file_pattern: File pattern to match (default: "*.json")
        """
        self.directory = Path(directory)
        self.file_pattern = file_pattern

    def load_template(self, identifier: str) -> Optional[PromptTemplate]:
        """Load a template from file."""
        file_path = self.directory / f"{identifier}.json"
        if not file_path.exists():
            return None

        try:
            with open(file_path, "r") as f:
                data = json.load(f)
            return PromptTemplate.from_dict(data)
        except Exception as e:
            print(f"Error loading template from {file_path}: {e}")
            return None

    def load_templates(self) -> Iterator[PromptTemplate]:
        """Load all templates from directory."""
        for file_path in self.directory.glob(self.file_pattern):
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)
                template = PromptTemplate.from_dict(data)
                yield template
            except Exception as e:
                print(f"Error loading template from {file_path}: {e}")

    def list_available(self) -> List[str]:
        """List available template files."""
        return [f.stem for f in self.directory.glob(self.file_pattern)]

    def _load_template_from_file(self, file_path: Path) -> Optional[PromptTemplate]:
        """Load template from file."""
        try:
            with open(file_path, "r") as f:
                template_data = json.load(f)
            return PromptTemplate.from_dict(template_data)
        except Exception as e:
            print(f"Error loading template from {file_path}: {e}")
            return None

    def _save_template_to_file(self, template: PromptTemplate, file_path: Path) -> bool:
        """Save template to file."""
        try:
            with open(file_path, "w") as f:
                json.dump(template.to_dict(), f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving template to {file_path}: {e}")
            return False


class YAMLTemplateLoader(TemplateLoader):
    """Load templates from YAML files."""

    def __init__(self, directory: str):
        """
        Initialize YAML template loader.

        Args:
            directory: Directory to search for YAML templates
        """
        self.directory = Path(directory)

    def load_template(self, identifier: str) -> Optional[PromptTemplate]:
        """Load a template from YAML file."""
        file_path = self.directory / f"{identifier}.yaml"
        if not file_path.exists():
            return None

        try:
            with open(file_path, "r") as f:
                data = yaml.safe_load(f)
            return PromptTemplate.from_dict(data)
        except Exception as e:
            print(f"Error loading template from {file_path}: {e}")
            return None

    def load_templates(self) -> Iterator[PromptTemplate]:
        """Load all templates from YAML files."""
        for file_path in self.directory.glob("*.yaml"):
            try:
                with open(file_path, "r") as f:
                    data = yaml.safe_load(f)
                template = PromptTemplate.from_dict(data)
                yield template
            except Exception as e:
                print(f"Error loading template from {file_path}: {e}")

    def list_available(self) -> List[str]:
        """List available YAML template files."""
        return [f.stem for f in self.directory.glob("*.yaml")]


class DirectoryTemplateLoader(TemplateLoader):
    """Load templates from a directory structure."""

    def __init__(self, base_directory: str):
        """
        Initialize directory template loader.

        Args:
            base_directory: Base directory containing template structure
        """
        self.base_directory = Path(base_directory)

    def load_template(self, identifier: str) -> Optional[PromptTemplate]:
        """Load a template from directory structure."""
        template_dir = self.base_directory / identifier
        if not template_dir.exists() or not template_dir.is_dir():
            return None

        # Look for template files
        template_file = template_dir / "template.json"
        if not template_file.exists():
            template_file = template_dir / "template.yaml"

        if not template_file.exists():
            return None

        try:
            if template_file.suffix == ".json":
                with open(template_file, "r") as f:
                    data = json.load(f)
            else:
                with open(template_file, "r") as f:
                    data = yaml.safe_load(f)

            return PromptTemplate.from_dict(data)
        except Exception as e:
            print(f"Error loading template from {template_file}: {e}")
            return None

    def load_templates(self) -> Iterator[PromptTemplate]:
        """Load all templates from directory structure."""
        for template_dir in self.base_directory.iterdir():
            if template_dir.is_dir():
                template = self.load_template(template_dir.name)
                if template:
                    yield template

    def list_available(self) -> List[str]:
        """List available template directories."""
        return [d.name for d in self.base_directory.iterdir() if d.is_dir()]


class DatabaseTemplateLoader(TemplateLoader):
    """Load templates from database."""

    def __init__(self, connection_string: str, table_name: str = "prompt_templates"):
        """
        Initialize database template loader.

        Args:
            connection_string: Database connection string
            table_name: Table name containing templates
        """
        self.connection_string = connection_string
        self.table_name = table_name
        self._connection = None

    def _get_connection(self) -> Any:
        """Get database connection."""
        if self._connection is None:
            # This is a placeholder - implement actual database connection
            # based on the connection string format
            raise NotImplementedError("Database connection not implemented")
        return self._connection

    def load_template(self, identifier: str) -> Optional[PromptTemplate]:
        """Load a template from database."""
        try:
            # conn = self._get_connection()
            # Placeholder for database query
            # query = f"SELECT * FROM {self.table_name} WHERE template_id = %s"
            # result = conn.execute(query, (identifier,))
            # data = result.fetchone()
            # return PromptTemplate.from_dict(data) if data else None
            raise NotImplementedError("Database loading not implemented")
        except Exception as e:
            print(f"Error loading template from database: {e}")
            return None

    def load_templates(self) -> Iterator[PromptTemplate]:
        """Load all templates from database."""
        try:
            # conn = self._get_connection()
            # Placeholder for database query
            # query = f"SELECT * FROM {self.table_name}"
            # results = conn.execute(query)
            # for row in results:
            #     yield PromptTemplate.from_dict(row)
            raise NotImplementedError("Database loading not implemented")
        except Exception as e:
            print(f"Error loading templates from database: {e}")
            return iter([])

    def list_available(self) -> List[str]:
        """List available template IDs in database."""
        try:
            # conn = self._get_connection()
            # Placeholder for database query
            # query = f"SELECT template_id FROM {self.table_name}"
            # results = conn.execute(query)
            # return [row['template_id'] for row in results]
            raise NotImplementedError("Database listing not implemented")
        except Exception as e:
            print(f"Error listing templates from database: {e}")
            return []


class APITemplateLoader(TemplateLoader):
    """Load templates from API endpoints."""

    def __init__(self, base_url: str, api_key: Optional[str] = None):
        """
        Initialize API template loader.

        Args:
            base_url: Base URL for API
            api_key: API key for authentication (optional)
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self._session: Optional[Any] = None

    def _get_session(self) -> Any:
        """Get HTTP session."""
        if self._session is None:
            import requests

            self._session = requests.Session()
            if self.api_key:
                self._session.headers.update(
                    {"Authorization": f"Bearer {self.api_key}"}
                )
        return self._session

    def load_template(self, identifier: str) -> Optional[PromptTemplate]:
        """Load a template from API."""
        try:
            session = self._get_session()
            url = f"{self.base_url}/templates/{identifier}"
            response = session.get(url)
            response.raise_for_status()

            data = response.json()
            return PromptTemplate.from_dict(data)
        except Exception as e:
            print(f"Error loading template from API: {e}")
            return None

    def load_templates(self) -> Iterator[PromptTemplate]:
        """Load all templates from API."""
        try:
            session = self._get_session()
            url = f"{self.base_url}/templates"
            response = session.get(url)
            response.raise_for_status()

            templates_data = response.json()
            for template_data in templates_data:
                yield PromptTemplate.from_dict(template_data)
        except Exception as e:
            print(f"Error loading templates from API: {e}")

    def list_available(self) -> List[str]:
        """List available template IDs from API."""
        try:
            session = self._get_session()
            url = f"{self.base_url}/templates"
            response = session.get(url)
            response.raise_for_status()

            templates_data = response.json()
            return [template["template_id"] for template in templates_data]
        except Exception as e:
            print(f"Error listing templates from API: {e}")
            return []


class CompositeTemplateLoader(TemplateLoader):
    """Composite loader that combines multiple loaders."""

    def __init__(self, loaders: List[TemplateLoader]):
        """
        Initialize composite template loader.

        Args:
            loaders: List of template loaders to combine
        """
        self.loaders = loaders

    def load_template(self, identifier: str) -> Optional[PromptTemplate]:
        """Load template from first loader that has it."""
        for loader in self.loaders:
            template = loader.load_template(identifier)
            if template:
                return template
        return None

    def load_templates(self) -> Iterator[PromptTemplate]:
        """Load templates from all loaders."""
        seen_templates = set()
        for loader in self.loaders:
            for template in loader.load_templates():
                if template.template_id not in seen_templates:
                    seen_templates.add(template.template_id)
                    yield template

    def list_available(self) -> List[str]:
        """List available templates from all loaders."""
        identifiers = set()
        for loader in self.loaders:
            identifiers.update(loader.list_available())
        return list(identifiers)


def create_loader_from_config(config: Dict[str, Any]) -> TemplateLoader:
    """
    Create a template loader from configuration.

    Args:
        config: Configuration dictionary

    Returns:
        Configured template loader
    """
    loader_type = config.get("type", "file")

    if loader_type == "file":
        return FileTemplateLoader(
            directory=config["directory"],
            file_pattern=config.get("file_pattern", "*.json"),
        )
    elif loader_type == "yaml":
        return YAMLTemplateLoader(directory=config["directory"])
    elif loader_type == "directory":
        return DirectoryTemplateLoader(base_directory=config["directory"])
    elif loader_type == "database":
        return DatabaseTemplateLoader(
            connection_string=config["connection_string"],
            table_name=config.get("table_name", "prompt_templates"),
        )
    elif loader_type == "api":
        return APITemplateLoader(
            base_url=config["base_url"], api_key=config.get("api_key")
        )
    elif loader_type == "composite":
        loaders = [
            create_loader_from_config(loader_config)
            for loader_config in config["loaders"]
        ]
        return CompositeTemplateLoader(loaders)
    else:
        raise ValueError(f"Unknown loader type: {loader_type}")
