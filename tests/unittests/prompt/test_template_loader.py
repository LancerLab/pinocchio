"""
Tests for the template loader.
"""

import json
import shutil
import tempfile
from pathlib import Path

import pytest
import yaml

from pinocchio.prompt import AgentType, PromptTemplate, PromptType
from pinocchio.prompt.template_loader import (
    CompositeTemplateLoader,
    DirectoryTemplateLoader,
    FileTemplateLoader,
    YAMLTemplateLoader,
    create_loader_from_config,
)
from tests.utils import (
    assert_session_valid,
    assert_task_valid,
    create_mock_llm_client,
    create_test_session,
    create_test_task,
)


class TestFileTemplateLoader:
    """Tests for FileTemplateLoader."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def sample_template(self):
        """Create a sample template for testing."""
        return PromptTemplate.create_new_version(
            template_name="test_template",
            content="Generate {{task}}",
            agent_type=AgentType.GENERATOR,
            prompt_type=PromptType.CODE_GENERATION,
            description="Test template",
        )

    def test_file_loader_creation(self, temp_dir):
        """Test FileTemplateLoader creation."""
        loader = FileTemplateLoader(temp_dir)
        assert loader.directory == Path(temp_dir)
        assert loader.file_pattern == "*.json"

    def test_file_loader_with_custom_pattern(self, temp_dir):
        """Test FileTemplateLoader with custom file pattern."""
        loader = FileTemplateLoader(temp_dir, "*.txt")
        assert loader.file_pattern == "*.txt"

    def test_save_and_load_template(self, temp_dir, sample_template):
        """Test saving and loading a template."""
        loader = FileTemplateLoader(temp_dir)

        # Save template to file
        template_file = Path(temp_dir) / f"{sample_template.template_name}.json"
        with open(template_file, "w") as f:
            json.dump(sample_template.to_dict(), f, indent=2)

        # Load template
        loaded_template = loader.load_template(sample_template.template_name)
        assert loaded_template is not None
        assert loaded_template.template_name == sample_template.template_name
        assert loaded_template.content == sample_template.content

    def test_load_nonexistent_template(self, temp_dir):
        """Test loading a nonexistent template."""
        loader = FileTemplateLoader(temp_dir)
        template = loader.load_template("nonexistent")
        assert template is None

    def test_load_all_templates(self, temp_dir, sample_template):
        """Test loading all templates from directory."""
        loader = FileTemplateLoader(temp_dir)

        # Save multiple templates
        templates = [
            sample_template,
            PromptTemplate.create_new_version("template2", "Content 2"),
            PromptTemplate.create_new_version("template3", "Content 3"),
        ]

        for template in templates:
            template_file = Path(temp_dir) / f"{template.template_name}.json"
            with open(template_file, "w") as f:
                json.dump(template.to_dict(), f, indent=2)

        # Load all templates
        loaded_templates = list(loader.load_templates())
        assert len(loaded_templates) == 3

        template_names = [t.template_name for t in loaded_templates]
        assert "test_template" in template_names
        assert "template2" in template_names
        assert "template3" in template_names

    def test_list_available_templates(self, temp_dir, sample_template):
        """Test listing available templates."""
        loader = FileTemplateLoader(temp_dir)

        # Save template
        template_file = Path(temp_dir) / f"{sample_template.template_name}.json"
        with open(template_file, "w") as f:
            json.dump(sample_template.to_dict(), f, indent=2)

        # List available templates
        available = loader.list_available()
        assert sample_template.template_name in available

    def test_load_invalid_json(self, temp_dir):
        """Test loading invalid JSON file."""
        loader = FileTemplateLoader(temp_dir)

        # Create invalid JSON file
        template_file = Path(temp_dir) / "invalid.json"
        with open(template_file, "w") as f:
            f.write("invalid json content")

        # Should handle error gracefully
        template = loader.load_template("invalid")
        assert template is None


class TestYAMLTemplateLoader:
    """Tests for YAMLTemplateLoader."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def sample_template(self):
        """Create a sample template for testing."""
        return PromptTemplate.create_new_version(
            template_name="test_template",
            content="Generate {{task}}",
            agent_type=AgentType.GENERATOR,
            prompt_type=PromptType.CODE_GENERATION,
            description="Test template",
        )

    def test_yaml_loader_creation(self, temp_dir):
        """Test YAMLTemplateLoader creation."""
        loader = YAMLTemplateLoader(temp_dir)
        assert loader.directory == Path(temp_dir)

    def test_save_and_load_yaml_template(self, temp_dir, sample_template):
        """Test saving and loading a YAML template."""
        loader = YAMLTemplateLoader(temp_dir)

        # Save template to YAML file
        template_file = Path(temp_dir) / f"{sample_template.template_name}.yaml"
        with open(template_file, "w") as f:
            yaml.dump(sample_template.to_dict(), f, default_flow_style=False)

        # Load template
        loaded_template = loader.load_template(sample_template.template_name)
        assert loaded_template is not None
        assert loaded_template.template_name == sample_template.template_name
        assert loaded_template.content == sample_template.content

    def test_load_all_yaml_templates(self, temp_dir, sample_template):
        """Test loading all YAML templates."""
        loader = YAMLTemplateLoader(temp_dir)

        # Save multiple templates
        templates = [
            sample_template,
            PromptTemplate.create_new_version("template2", "Content 2"),
            PromptTemplate.create_new_version("template3", "Content 3"),
        ]

        for template in templates:
            template_file = Path(temp_dir) / f"{template.template_name}.yaml"
            with open(template_file, "w") as f:
                yaml.dump(template.to_dict(), f, default_flow_style=False)

        # Load all templates
        loaded_templates = list(loader.load_templates())
        assert len(loaded_templates) == 3

    def test_list_available_yaml_templates(self, temp_dir, sample_template):
        """Test listing available YAML templates."""
        loader = YAMLTemplateLoader(temp_dir)

        # Save template
        template_file = Path(temp_dir) / f"{sample_template.template_name}.yaml"
        with open(template_file, "w") as f:
            yaml.dump(sample_template.to_dict(), f, default_flow_style=False)

        # List available templates
        available = loader.list_available()
        assert sample_template.template_name in available


class TestDirectoryTemplateLoader:
    """Tests for DirectoryTemplateLoader."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def sample_template(self):
        """Create a sample template for testing."""
        return PromptTemplate.create_new_version(
            template_name="test_template",
            content="Generate {{task}}",
            agent_type=AgentType.GENERATOR,
            prompt_type=PromptType.CODE_GENERATION,
            description="Test template",
        )

    def test_directory_loader_creation(self, temp_dir):
        """Test DirectoryTemplateLoader creation."""
        loader = DirectoryTemplateLoader(temp_dir)
        assert loader.base_directory == Path(temp_dir)

    def test_save_and_load_directory_template(self, temp_dir, sample_template):
        """Test saving and loading a template from directory structure."""
        loader = DirectoryTemplateLoader(temp_dir)

        # Create directory structure
        template_dir = Path(temp_dir) / sample_template.template_name
        template_dir.mkdir()

        # Save template as JSON
        template_file = template_dir / "template.json"
        with open(template_file, "w") as f:
            json.dump(sample_template.to_dict(), f, indent=2)

        # Load template
        loaded_template = loader.load_template(sample_template.template_name)
        assert loaded_template is not None
        assert loaded_template.template_name == sample_template.template_name
        assert loaded_template.content == sample_template.content

    def test_save_and_load_yaml_directory_template(self, temp_dir, sample_template):
        """Test saving and loading a YAML template from directory structure."""
        loader = DirectoryTemplateLoader(temp_dir)

        # Create directory structure
        template_dir = Path(temp_dir) / sample_template.template_name
        template_dir.mkdir()

        # Save template as YAML
        template_file = template_dir / "template.yaml"
        with open(template_file, "w") as f:
            yaml.dump(sample_template.to_dict(), f, default_flow_style=False)

        # Load template
        loaded_template = loader.load_template(sample_template.template_name)
        assert loaded_template is not None
        assert loaded_template.template_name == sample_template.template_name

    def test_load_all_directory_templates(self, temp_dir, sample_template):
        """Test loading all templates from directory structure."""
        loader = DirectoryTemplateLoader(temp_dir)

        # Create multiple template directories
        templates = [
            sample_template,
            PromptTemplate.create_new_version("template2", "Content 2"),
            PromptTemplate.create_new_version("template3", "Content 3"),
        ]

        for template in templates:
            template_dir = Path(temp_dir) / template.template_name
            template_dir.mkdir()

            template_file = template_dir / "template.json"
            with open(template_file, "w") as f:
                json.dump(template.to_dict(), f, indent=2)

        # Load all templates
        loaded_templates = list(loader.load_templates())
        assert len(loaded_templates) == 3

    def test_list_available_directory_templates(self, temp_dir, sample_template):
        """Test listing available templates from directory structure."""
        loader = DirectoryTemplateLoader(temp_dir)

        # Create template directory
        template_dir = Path(temp_dir) / sample_template.template_name
        template_dir.mkdir()

        template_file = template_dir / "template.json"
        with open(template_file, "w") as f:
            json.dump(sample_template.to_dict(), f, indent=2)

        # List available templates
        available = loader.list_available()
        assert sample_template.template_name in available

    def test_load_nonexistent_directory_template(self, temp_dir):
        """Test loading a nonexistent directory template."""
        loader = DirectoryTemplateLoader(temp_dir)
        template = loader.load_template("nonexistent")
        assert template is None

    def test_load_template_without_template_file(self, temp_dir):
        """Test loading from directory without template file."""
        loader = DirectoryTemplateLoader(temp_dir)

        # Create directory without template file
        template_dir = Path(temp_dir) / "empty_template"
        template_dir.mkdir()

        template = loader.load_template("empty_template")
        assert template is None


class TestCompositeTemplateLoader:
    """Tests for CompositeTemplateLoader."""

    @pytest.fixture
    def temp_dir1(self):
        """Create first temporary directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def temp_dir2(self):
        """Create second temporary directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    def test_composite_loader_creation(self, temp_dir1, temp_dir2):
        """Test CompositeTemplateLoader creation."""
        loader1 = FileTemplateLoader(temp_dir1)
        loader2 = YAMLTemplateLoader(temp_dir2)

        composite = CompositeTemplateLoader([loader1, loader2])
        assert len(composite.loaders) == 2

    def test_composite_loader_load_template(self, temp_dir1, temp_dir2):
        """Test loading template from composite loader."""
        # Create template in first loader
        template = PromptTemplate.create_new_version("test", "Content")
        template_file = Path(temp_dir1) / "test.json"
        with open(template_file, "w") as f:
            json.dump(template.to_dict(), f, indent=2)

        loader1 = FileTemplateLoader(temp_dir1)
        loader2 = YAMLTemplateLoader(temp_dir2)
        composite = CompositeTemplateLoader([loader1, loader2])

        # Load template
        loaded_template = composite.load_template("test")
        assert loaded_template is not None
        assert loaded_template.template_name == "test"

    def test_composite_loader_load_all_templates(self, temp_dir1, temp_dir2):
        """Test loading all templates from composite loader."""
        # Create templates in both loaders
        template1 = PromptTemplate.create_new_version("template1", "Content 1")
        template2 = PromptTemplate.create_new_version("template2", "Content 2")

        # Save to first loader
        template_file1 = Path(temp_dir1) / "template1.json"
        with open(template_file1, "w") as f:
            json.dump(template1.to_dict(), f, indent=2)

        # Save to second loader
        template_file2 = Path(temp_dir2) / "template2.yaml"
        with open(template_file2, "w") as f:
            yaml.dump(template2.to_dict(), f, default_flow_style=False)

        loader1 = FileTemplateLoader(temp_dir1)
        loader2 = YAMLTemplateLoader(temp_dir2)
        composite = CompositeTemplateLoader([loader1, loader2])

        # Load all templates
        loaded_templates = list(composite.load_templates())
        assert len(loaded_templates) == 2

        template_names = [t.template_name for t in loaded_templates]
        assert "template1" in template_names
        assert "template2" in template_names

    def test_composite_loader_list_available(self, temp_dir1, temp_dir2):
        """Test listing available templates from composite loader."""
        # Create templates in both loaders
        template1 = PromptTemplate.create_new_version("template1", "Content 1")
        template2 = PromptTemplate.create_new_version("template2", "Content 2")

        # Save to first loader
        template_file1 = Path(temp_dir1) / "template1.json"
        with open(template_file1, "w") as f:
            json.dump(template1.to_dict(), f, indent=2)

        # Save to second loader
        template_file2 = Path(temp_dir2) / "template2.yaml"
        with open(template_file2, "w") as f:
            yaml.dump(template2.to_dict(), f, default_flow_style=False)

        loader1 = FileTemplateLoader(temp_dir1)
        loader2 = YAMLTemplateLoader(temp_dir2)
        composite = CompositeTemplateLoader([loader1, loader2])

        # List available templates
        available = composite.list_available()
        assert "template1" in available
        assert "template2" in available

    def test_composite_loader_duplicate_handling(self, temp_dir1, temp_dir2):
        """Test handling of duplicate templates in composite loader."""
        # Create same template in both loaders
        template = PromptTemplate.create_new_version("duplicate", "Content")

        # Save to first loader
        template_file1 = Path(temp_dir1) / "duplicate.json"
        with open(template_file1, "w") as f:
            json.dump(template.to_dict(), f, indent=2)

        # Save to second loader
        template_file2 = Path(temp_dir2) / "duplicate.yaml"
        with open(template_file2, "w") as f:
            yaml.dump(template.to_dict(), f, default_flow_style=False)

        loader1 = FileTemplateLoader(temp_dir1)
        loader2 = YAMLTemplateLoader(temp_dir2)
        composite = CompositeTemplateLoader([loader1, loader2])

        # Should only return one instance (from first loader)
        loaded_templates = list(composite.load_templates())
        assert len(loaded_templates) == 1
        assert loaded_templates[0].template_name == "duplicate"


class TestCreateLoaderFromConfig:
    """Tests for create_loader_from_config function."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    def test_create_file_loader(self, temp_dir):
        """Test creating file loader from config."""
        config = {"type": "file", "directory": temp_dir, "file_pattern": "*.json"}

        loader = create_loader_from_config(config)
        assert isinstance(loader, FileTemplateLoader)
        assert loader.directory == Path(temp_dir)
        assert loader.file_pattern == "*.json"

    def test_create_yaml_loader(self, temp_dir):
        """Test creating YAML loader from config."""
        config = {"type": "yaml", "directory": temp_dir}

        loader = create_loader_from_config(config)
        assert isinstance(loader, YAMLTemplateLoader)
        assert loader.directory == Path(temp_dir)

    def test_create_directory_loader(self, temp_dir):
        """Test creating directory loader from config."""
        config = {"type": "directory", "directory": temp_dir}

        loader = create_loader_from_config(config)
        assert isinstance(loader, DirectoryTemplateLoader)
        assert loader.base_directory == Path(temp_dir)

    def test_create_composite_loader(self, temp_dir):
        """Test creating composite loader from config."""
        config = {
            "type": "composite",
            "loaders": [
                {"type": "file", "directory": temp_dir, "file_pattern": "*.json"},
                {"type": "yaml", "directory": temp_dir},
            ],
        }

        loader = create_loader_from_config(config)
        assert isinstance(loader, CompositeTemplateLoader)
        assert len(loader.loaders) == 2
        assert isinstance(loader.loaders[0], FileTemplateLoader)
        assert isinstance(loader.loaders[1], YAMLTemplateLoader)

    def test_create_loader_with_default_type(self, temp_dir):
        """Test creating loader with default type."""
        config = {"directory": temp_dir}

        loader = create_loader_from_config(config)
        assert isinstance(loader, FileTemplateLoader)

    def test_create_loader_with_unknown_type(self, temp_dir):
        """Test creating loader with unknown type."""
        config = {"type": "unknown", "directory": temp_dir}

        with pytest.raises(ValueError, match="Unknown loader type"):
            create_loader_from_config(config)
