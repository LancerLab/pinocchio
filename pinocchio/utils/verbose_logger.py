"""Unified verbose logging interface for Pinocchio multi-agent system."""

import json
import logging
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.tree import Tree


class LogLevel(Enum):
    """Log level enumeration."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class LogEntry:
    """Structured log entry."""

    timestamp: str
    level: LogLevel
    component: str
    message: str
    data: Optional[Dict[str, Any]] = None
    duration_ms: Optional[float] = None
    session_id: Optional[str] = None
    agent_id: Optional[str] = None
    step_id: Optional[str] = None


class VerboseLogger:
    """Unified verbose logging interface with recursive structured output."""

    def __init__(
        self,
        console: Optional[Console] = None,
        log_file: Optional[Path] = None,
        max_depth: int = 5,
        enable_colors: bool = True,
    ):
        """
        Initialize verbose logger.

        Args:
            console: Rich console instance
            log_file: Optional log file path
            max_depth: Maximum recursion depth for nested structures
            enable_colors: Whether to enable colored output
        """
        self.console = console or Console()
        self.log_file = log_file
        self.max_depth = max_depth
        self.enable_colors = enable_colors
        self.entries: List[LogEntry] = []
        self.session_stats: Dict[str, Any] = {}
        self.performance_metrics: Dict[str, List[float]] = {}

        # Initialize rich components
        self._setup_rich_components()

    def _setup_rich_components(self) -> None:
        """Setup rich console components."""
        self.table_style = "cyan"
        self.panel_style = "blue"
        self.tree_style = "green"
        self.error_style = "red"
        self.warning_style = "yellow"
        self.success_style = "green"

    def log(
        self,
        level: LogLevel,
        component: str,
        message: str,
        data: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        step_id: Optional[str] = None,
        duration_ms: Optional[float] = None,
    ) -> LogEntry:
        """
        Log a structured entry.

        Args:
            level: Log level
            component: Component name
            message: Log message
            data: Optional structured data
            session_id: Session identifier
            agent_id: Agent identifier
            step_id: Step identifier
            duration_ms: Duration in milliseconds

        Returns:
            Created log entry
        """
        entry = LogEntry(
            timestamp=datetime.utcnow().isoformat(),
            level=level,
            component=component,
            message=message,
            data=data,
            duration_ms=duration_ms,
            session_id=session_id,
            agent_id=agent_id,
            step_id=step_id,
        )

        self.entries.append(entry)
        self._display_entry(entry)
        self._save_to_file(entry)

        return entry

    def _display_entry(self, entry: LogEntry) -> None:
        """Display log entry using rich formatting."""
        if entry.level == LogLevel.ERROR:
            self._display_error(entry)
        elif entry.level == LogLevel.WARNING:
            self._display_warning(entry)
        elif entry.level == LogLevel.INFO:
            self._display_info(entry)
        else:
            self._display_debug(entry)

    def _display_error(self, entry: LogEntry) -> None:
        """Display error entry."""
        panel = Panel(
            f"[bold red]{entry.message}[/bold red]\n"
            f"[dim]Component: {entry.component}[/dim]\n"
            f"[dim]Time: {entry.timestamp}[/dim]",
            title="❌ ERROR",
            border_style="red",
            box=box.ROUNDED,
        )
        self.console.print(panel)

        if entry.data:
            self._display_structured_data(entry.data, "Error Details")

    def _display_warning(self, entry: LogEntry) -> None:
        """Display warning entry."""
        panel = Panel(
            f"[bold yellow]{entry.message}[/bold yellow]\n"
            f"[dim]Component: {entry.component}[/dim]\n"
            f"[dim]Time: {entry.timestamp}[/dim]",
            title="⚠️ WARNING",
            border_style="yellow",
            box=box.ROUNDED,
        )
        self.console.print(panel)

        if entry.data:
            self._display_structured_data(entry.data, "Warning Details")

    def _display_info(self, entry: LogEntry) -> None:
        """Display info entry."""
        content = f"[bold blue]{entry.message}[/bold blue]\n"

        if entry.duration_ms:
            content += f"[dim]Duration: {entry.duration_ms:.2f}ms[/dim]\n"

        if entry.session_id:
            content += f"[dim]Session: {entry.session_id}[/dim]\n"

        if entry.agent_id:
            content += f"[dim]Agent: {entry.agent_id}[/dim]\n"

        content += f"[dim]Time: {entry.timestamp}[/dim]"

        panel = Panel(
            content,
            title="ℹ️ INFO",
            border_style="blue",
            box=box.ROUNDED,
        )
        self.console.print(panel)

        if entry.data:
            self._display_structured_data(entry.data, "Details")

    def _display_debug(self, entry: LogEntry) -> None:
        """Display debug entry."""
        content = f"[cyan]{entry.message}[/cyan]\n"
        content += f"[dim]Component: {entry.component}[/dim]\n"
        content += f"[dim]Time: {entry.timestamp}[/dim]"

        panel = Panel(
            content,
            title="🔍 DEBUG",
            border_style="cyan",
            box=box.ROUNDED,
        )
        self.console.print(panel)

        if entry.data:
            self._display_structured_data(entry.data, "Debug Data")

    def _display_structured_data(self, data: Dict[str, Any], title: str) -> None:
        """Display structured data recursively."""
        if not data:
            return

        # Create tree structure for nested data
        tree = Tree(f"[bold]{title}[/bold]")
        self._build_data_tree(data, tree, depth=0)

        panel = Panel(
            tree,
            border_style="dim",
            box=box.SIMPLE,
        )
        self.console.print(panel)

    def _build_data_tree(self, data: Any, tree: Tree, depth: int) -> None:
        """Recursively build tree structure for data."""
        if depth >= self.max_depth:
            tree.add("[dim]... (max depth reached)[/dim]")
            return

        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, (dict, list)):
                    branch = tree.add(f"[bold]{key}[/bold]")
                    self._build_data_tree(value, branch, depth + 1)
                else:
                    tree.add(f"[bold]{key}[/bold]: {self._format_value(value)}")

        elif isinstance(data, list):
            for i, item in enumerate(data):
                if isinstance(item, (dict, list)):
                    branch = tree.add(f"[bold][{i}][/bold]")
                    self._build_data_tree(item, branch, depth + 1)
                else:
                    tree.add(f"[bold][{i}][/bold]: {self._format_value(item)}")

        else:
            tree.add(self._format_value(data))

    def _format_value(self, value: Any) -> str:
        """Format value for display."""
        if isinstance(value, str):
            return f'"{value}"'
        elif isinstance(value, (int, float)):
            return str(value)
        elif isinstance(value, bool):
            return "[green]True[/green]" if value else "[red]False[/red]"
        elif value is None:
            return "[dim]None[/dim]"
        else:
            return str(value)

    def _save_to_file(self, entry: LogEntry) -> None:
        """Save log entry to file if configured."""
        if not self.log_file:
            return

        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                json.dump(asdict(entry), f, ensure_ascii=False, default=str)
                f.write("\n")
        except Exception as e:
            # Don't log logging errors to avoid infinite recursion
            print(f"Failed to save log entry: {e}", file=sys.stderr)

    def log_performance(self, operation: str, duration_ms: float) -> None:
        """Log performance metric."""
        if operation not in self.performance_metrics:
            self.performance_metrics[operation] = []

        self.performance_metrics[operation].append(duration_ms)

        self.log(
            level=LogLevel.INFO,
            component="performance",
            message=f"Performance: {operation}",
            data={"duration_ms": duration_ms, "operation": operation},
            duration_ms=duration_ms,
        )

    def log_agent_activity(
        self,
        agent_id: str,
        action: str,
        data: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
        step_id: Optional[str] = None,
        duration_ms: Optional[float] = None,
    ) -> LogEntry:
        """Log agent activity."""
        return self.log(
            level=LogLevel.INFO,
            component=f"agent:{agent_id}",
            message=f"Agent {agent_id}: {action}",
            data=data,
            session_id=session_id,
            agent_id=agent_id,
            step_id=step_id,
            duration_ms=duration_ms,
        )

    def log_coordinator_activity(
        self,
        action: str,
        data: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
        duration_ms: Optional[float] = None,
    ) -> LogEntry:
        """Log coordinator activity."""
        return self.log(
            level=LogLevel.INFO,
            component="coordinator",
            message=f"Coordinator: {action}",
            data=data,
            session_id=session_id,
            duration_ms=duration_ms,
        )

    def log_llm_activity(
        self,
        operation: str,
        request_data: Optional[Dict[str, Any]] = None,
        response_data: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
        duration_ms: Optional[float] = None,
    ) -> LogEntry:
        """Log LLM activity."""
        data = {}
        if request_data:
            data["request"] = request_data
        if response_data:
            data["response"] = response_data

        return self.log(
            level=LogLevel.INFO,
            component="llm",
            message=f"LLM: {operation}",
            data=data,
            session_id=session_id,
            duration_ms=duration_ms,
        )

    def display_performance_summary(self) -> None:
        """Display performance summary."""
        if not self.performance_metrics:
            return

        table = Table(
            title="Performance Summary", show_header=True, header_style="bold magenta"
        )
        table.add_column("Operation", style="cyan")
        table.add_column("Count", style="green")
        table.add_column("Avg (ms)", style="yellow")
        table.add_column("Min (ms)", style="blue")
        table.add_column("Max (ms)", style="red")

        for operation, metrics in self.performance_metrics.items():
            if metrics:
                avg_ms = sum(metrics) / len(metrics)
                min_ms = min(metrics)
                max_ms = max(metrics)

                table.add_row(
                    operation,
                    str(len(metrics)),
                    f"{avg_ms:.2f}",
                    f"{min_ms:.2f}",
                    f"{max_ms:.2f}",
                )

        self.console.print(table)

    def display_session_summary(self, session_id: str) -> None:
        """Display session summary."""
        session_entries = [e for e in self.entries if e.session_id == session_id]

        if not session_entries:
            return

        # Group by component
        component_stats: Dict[str, int] = {}
        level_stats: Dict[str, int] = {}

        for entry in session_entries:
            component_stats[entry.component] = (
                component_stats.get(entry.component, 0) + 1
            )
            level_stats[entry.level.value] = level_stats.get(entry.level.value, 0) + 1

        # Create summary table
        table = Table(
            title=f"Session Summary: {session_id}",
            show_header=True,
            header_style="bold magenta",
        )
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Total Entries", str(len(session_entries)))
        table.add_row("Components", str(len(component_stats)))
        table.add_row("Levels", str(len(level_stats)))

        self.console.print(table)

        # Component breakdown
        if component_stats:
            comp_table = Table(
                title="Component Breakdown", show_header=True, header_style="bold blue"
            )
            comp_table.add_column("Component", style="cyan")
            comp_table.add_column("Count", style="green")

            for component, count in sorted(
                component_stats.items(), key=lambda x: x[1], reverse=True
            ):
                comp_table.add_row(component, str(count))

            self.console.print(comp_table)

    def get_entries_by_session(self, session_id: str) -> List[LogEntry]:
        """Get all entries for a specific session."""
        return [e for e in self.entries if e.session_id == session_id]

    def get_entries_by_component(self, component: str) -> List[LogEntry]:
        """Get all entries for a specific component."""
        return [e for e in self.entries if e.component == component]

    def clear_entries(self) -> None:
        """Clear all stored entries."""
        self.entries.clear()
        self.performance_metrics.clear()

    def export_entries(self, file_path: Path) -> None:
        """Export all entries to JSON file."""
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(
                    [asdict(entry) for entry in self.entries],
                    f,
                    ensure_ascii=False,
                    default=str,
                    indent=2,
                )
        except Exception as e:
            self.log(LogLevel.ERROR, "verbose_logger", f"Failed to export entries: {e}")


# Global verbose logger instance
_global_verbose_logger: Optional[VerboseLogger] = None


def get_verbose_logger() -> VerboseLogger:
    """Get global verbose logger instance."""
    global _global_verbose_logger
    if _global_verbose_logger is None:
        _global_verbose_logger = VerboseLogger()
    return _global_verbose_logger


def set_verbose_logger(logger: VerboseLogger) -> None:
    """Set global verbose logger instance."""
    global _global_verbose_logger
    _global_verbose_logger = logger


def log_verbose(
    level: LogLevel,
    component: str,
    message: str,
    data: Optional[Dict[str, Any]] = None,
    session_id: Optional[str] = None,
    agent_id: Optional[str] = None,
    step_id: Optional[str] = None,
    duration_ms: Optional[float] = None,
) -> LogEntry:
    """Convenience function to log using global verbose logger."""
    return get_verbose_logger().log(
        level=level,
        component=component,
        message=message,
        data=data,
        session_id=session_id,
        agent_id=agent_id,
        step_id=step_id,
        duration_ms=duration_ms,
    )
