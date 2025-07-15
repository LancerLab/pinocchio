"""
Base classes and interfaces for MCP (Model Context Protocol) tools.
"""

import json
import logging
import os
import subprocess
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class ToolExecutionStatus(Enum):
    """Tool execution status enumeration."""

    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"
    NOT_FOUND = "not_found"


@dataclass
class ToolResult:
    """Result of tool execution."""

    status: ToolExecutionStatus
    output: str
    error: Optional[str] = None
    exit_code: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None


class MCPTool(ABC):
    """
    Base class for MCP-compatible tools.

    All tools should inherit from this class and implement the required methods.
    """

    def __init__(self, name: str, description: str, timeout: int = 30):
        """
        Initialize the MCP tool.

        Args:
            name: Tool name
            description: Tool description
            timeout: Execution timeout in seconds
        """
        self.name = name
        self.description = description
        self.timeout = timeout

    @abstractmethod
    def execute(self, **kwargs) -> ToolResult:
        """
        Execute the tool with given parameters.

        Args:
            **kwargs: Tool-specific parameters

        Returns:
            ToolResult: Execution result
        """
        pass

    @abstractmethod
    def get_schema(self) -> Dict[str, Any]:
        """
        Get the tool's parameter schema.

        Returns:
            Dict describing the tool's parameters and their types
        """
        pass

    def _run_command(
        self,
        command: List[str],
        input_data: Optional[str] = None,
        cwd: Optional[str] = None,
    ) -> ToolResult:
        """
        Run a shell command and return the result.

        Args:
            command: Command and arguments as list
            input_data: Optional input data to send to command
            cwd: Working directory for command execution

        Returns:
            ToolResult: Command execution result
        """
        try:
            logger.info(f"Executing MCP tool command: {' '.join(command)}")
            logger.debug(
                f"Command details - cwd: {cwd}, input_data_length: {len(input_data) if input_data else 0}"
            )

            process = subprocess.run(
                command,
                input=input_data,
                text=True,
                capture_output=True,
                timeout=self.timeout,
                cwd=cwd,
            )

            status = (
                ToolExecutionStatus.SUCCESS
                if process.returncode == 0
                else ToolExecutionStatus.ERROR
            )

            # Log tool execution results
            logger.info(
                f"MCP tool execution completed - status: {status.value}, exit_code: {process.returncode}"
            )
            if process.stdout:
                logger.debug(
                    f"Tool stdout: {process.stdout[:500]}{'...' if len(process.stdout) > 500 else ''}"
                )
            if process.stderr:
                logger.warning(
                    f"Tool stderr: {process.stderr[:500]}{'...' if len(process.stderr) > 500 else ''}"
                )

            result = ToolResult(
                status=status,
                output=process.stdout,
                error=process.stderr if process.stderr else None,
                exit_code=process.returncode,
                metadata={
                    "command": " ".join(command),
                    "cwd": cwd or os.getcwd(),
                    "execution_time": 0,  # Could add timing if needed
                    "output_length": len(process.stdout),
                    "error_length": len(process.stderr) if process.stderr else 0,
                },
            )

            logger.info(
                f"MCP tool result - status: {result.status.value}, output_length: {len(result.output)}"
            )
            return result

        except subprocess.TimeoutExpired:
            logger.error(f"MCP tool command timeout: {' '.join(command)}")
            return ToolResult(
                status=ToolExecutionStatus.TIMEOUT,
                output="",
                error=f"Command timed out after {self.timeout} seconds",
                metadata={"command": " ".join(command), "timeout": self.timeout},
            )
        except Exception as e:
            logger.error(f"MCP tool command failed: {' '.join(command)}, error: {e}")
            return ToolResult(
                status=ToolExecutionStatus.ERROR,
                output="",
                error=f"Command execution failed: {str(e)}",
                metadata={"command": " ".join(command), "exception": str(e)},
            )

    def _create_temp_file(self, content: str, suffix: str = ".cu") -> str:
        """
        Create a temporary file with given content.

        Args:
            content: File content
            suffix: File suffix

        Returns:
            str: Path to temporary file
        """
        with tempfile.NamedTemporaryFile(mode="w", suffix=suffix, delete=False) as f:
            f.write(content)
            return f.name

    def _cleanup_temp_file(self, filepath: str) -> None:
        """
        Clean up temporary file.

        Args:
            filepath: Path to file to delete
        """
        try:
            if os.path.exists(filepath):
                os.unlink(filepath)
                logger.debug(f"Cleaned up temporary file: {filepath}")
        except Exception as e:
            logger.warning(f"Failed to cleanup temp file {filepath}: {e}")


class ToolManager:
    """
    Manager for MCP tools.

    Handles tool registration, discovery, and execution.
    """

    def __init__(self):
        """Initialize the tool manager."""
        self.tools: Dict[str, MCPTool] = {}

    def register_tool(self, tool: MCPTool) -> None:
        """
        Register a tool.

        Args:
            tool: Tool to register
        """
        self.tools[tool.name] = tool
        logger.info(f"Registered tool: {tool.name}")

    def get_tool(self, name: str) -> Optional[MCPTool]:
        """
        Get a tool by name.

        Args:
            name: Tool name

        Returns:
            MCPTool or None if not found
        """
        return self.tools.get(name)

    def list_tools(self) -> List[str]:
        """
        List all registered tool names.

        Returns:
            List of tool names
        """
        return list(self.tools.keys())

    def get_tool_schemas(self) -> Dict[str, Dict[str, Any]]:
        """
        Get schemas for all registered tools.

        Returns:
            Dict mapping tool names to their schemas
        """
        return {name: tool.get_schema() for name, tool in self.tools.items()}

    def execute_tool(self, name: str, **kwargs) -> ToolResult:
        """
        Execute a tool by name.

        Args:
            name: Tool name
            **kwargs: Tool parameters

        Returns:
            ToolResult: Execution result
        """
        logger.info(f"Executing MCP tool: {name}")
        logger.debug(f"Tool parameters: {kwargs}")

        tool = self.get_tool(name)
        if not tool:
            logger.error(f"MCP tool not found: {name}")
            return ToolResult(
                status=ToolExecutionStatus.NOT_FOUND,
                output="",
                error=f"Tool not found: {name}",
                metadata={
                    "tool_name": name,
                    "available_tools": list(self.tools.keys()),
                },
            )

        try:
            logger.info(f"Starting execution of MCP tool: {name}")
            result = tool.execute(**kwargs)

            # Log detailed execution results
            logger.info(f"MCP tool '{name}' completed - status: {result.status.value}")
            logger.debug(f"Tool '{name}' output length: {len(result.output)}")
            logger.debug(f"Tool '{name}' metadata: {result.metadata}")

            if result.status == ToolExecutionStatus.SUCCESS:
                logger.info(
                    f"MCP tool '{name}' succeeded with {len(result.output)} characters output"
                )
            else:
                logger.warning(
                    f"MCP tool '{name}' failed - status: {result.status.value}, error: {result.error}"
                )

            return result

        except Exception as e:
            logger.error(f"MCP tool execution failed for {name}: {e}")
            return ToolResult(
                status=ToolExecutionStatus.ERROR,
                output="",
                error=f"Tool execution failed: {str(e)}",
                metadata={
                    "tool_name": name,
                    "exception": str(e),
                    "exception_type": type(e).__name__,
                },
            )
