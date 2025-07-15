"""
Base plugin architecture for Pinocchio.

This module defines the core plugin interfaces and management functionality.
"""

import importlib.util
import inspect
import logging
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union

logger = logging.getLogger(__name__)


class PluginType(Enum):
    """Plugin type enumeration."""

    PROMPT = "prompt"
    AGENT = "agent"
    WORKFLOW = "workflow"
    MEMORY = "memory"
    KNOWLEDGE = "knowledge"
    LLM = "llm"
    TASK_PLANNING = "task_planning"


class Plugin(ABC):
    """
    Base plugin interface.

    All plugins must inherit from this class and implement the required methods.
    """

    def __init__(self, name: str, plugin_type: PluginType, version: str = "1.0.0"):
        """Initialize plugin."""
        self.name = name
        self.plugin_type = plugin_type
        self.version = version
        self.enabled = True
        self.metadata = {}

    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize plugin with configuration."""
        pass

    @abstractmethod
    def execute(self, *args, **kwargs) -> Any:
        """Execute plugin functionality."""
        pass

    def cleanup(self) -> None:
        """Cleanup plugin resources."""
        pass

    def get_info(self) -> Dict[str, Any]:
        """Get plugin information."""
        return {
            "name": self.name,
            "type": self.plugin_type.value,
            "version": self.version,
            "enabled": self.enabled,
            "metadata": self.metadata,
        }


class PluginManager:
    """
    Plugin manager for loading and managing plugins.
    """

    def __init__(self, plugins_dir: Optional[str] = None):
        """Initialize plugin manager."""
        self.plugins_dir = Path(plugins_dir) if plugins_dir else Path("./plugins")
        self.plugins: Dict[str, Plugin] = {}
        self.plugin_configs: Dict[str, Dict[str, Any]] = {}

    def register_plugin(
        self, plugin: Plugin, config: Optional[Dict[str, Any]] = None
    ) -> None:
        """Register a plugin instance."""
        if config is None:
            config = {}

        self.plugins[plugin.name] = plugin
        self.plugin_configs[plugin.name] = config

        try:
            plugin.initialize(config)
            logger.info(
                f"Registered plugin: {plugin.name} ({plugin.plugin_type.value})"
            )
        except Exception as e:
            logger.error(f"Failed to initialize plugin {plugin.name}: {e}")
            self.plugins.pop(plugin.name, None)
            self.plugin_configs.pop(plugin.name, None)
            raise

    def load_plugin_from_file(
        self, file_path: Union[str, Path], config: Optional[Dict[str, Any]] = None
    ) -> None:
        """Load plugin from Python file."""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Plugin file not found: {file_path}")

        spec = importlib.util.spec_from_file_location(file_path.stem, file_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load plugin from {file_path}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Find plugin classes in the module
        plugin_classes = []
        for name, obj in inspect.getmembers(module):
            if inspect.isclass(obj) and issubclass(obj, Plugin) and obj != Plugin:
                plugin_classes.append(obj)

        if not plugin_classes:
            raise ValueError(f"No plugin classes found in {file_path}")

        # Register the first plugin class found
        plugin_class = plugin_classes[0]
        plugin_instance = plugin_class()
        self.register_plugin(plugin_instance, config)

    def load_plugins_from_directory(
        self, directory: Optional[Union[str, Path]] = None
    ) -> None:
        """Load all plugins from a directory."""
        if directory is None:
            directory = self.plugins_dir
        else:
            directory = Path(directory)

        if not directory.exists():
            logger.warning(f"Plugins directory does not exist: {directory}")
            return

        for plugin_file in directory.glob("*.py"):
            if plugin_file.name.startswith("_"):
                continue  # Skip private files

            try:
                self.load_plugin_from_file(plugin_file)
            except Exception as e:
                logger.error(f"Failed to load plugin from {plugin_file}: {e}")

    def get_plugin(self, name: str) -> Optional[Plugin]:
        """Get plugin by name."""
        return self.plugins.get(name)

    def get_plugins_by_type(self, plugin_type: PluginType) -> List[Plugin]:
        """Get all plugins of a specific type."""
        return [
            plugin
            for plugin in self.plugins.values()
            if plugin.plugin_type == plugin_type and plugin.enabled
        ]

    def execute_plugin(self, name: str, *args, **kwargs) -> Any:
        """Execute a plugin by name."""
        plugin = self.get_plugin(name)
        if plugin is None:
            raise ValueError(f"Plugin not found: {name}")

        if not plugin.enabled:
            raise ValueError(f"Plugin is disabled: {name}")

        return plugin.execute(*args, **kwargs)

    def enable_plugin(self, name: str) -> None:
        """Enable a plugin."""
        plugin = self.get_plugin(name)
        if plugin:
            plugin.enabled = True
            logger.info(f"Enabled plugin: {name}")

    def disable_plugin(self, name: str) -> None:
        """Disable a plugin."""
        plugin = self.get_plugin(name)
        if plugin:
            plugin.enabled = False
            logger.info(f"Disabled plugin: {name}")

    def unregister_plugin(self, name: str) -> None:
        """Unregister a plugin."""
        plugin = self.plugins.pop(name, None)
        if plugin:
            try:
                plugin.cleanup()
            except Exception as e:
                logger.error(f"Error during plugin cleanup {name}: {e}")

            self.plugin_configs.pop(name, None)
            logger.info(f"Unregistered plugin: {name}")

    def list_plugins(self) -> List[Dict[str, Any]]:
        """List all registered plugins."""
        return [plugin.get_info() for plugin in self.plugins.values()]

    def cleanup_all(self) -> None:
        """Cleanup all plugins."""
        for name in list(self.plugins.keys()):
            self.unregister_plugin(name)
