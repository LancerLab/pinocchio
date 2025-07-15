#!/usr/bin/env python3
"""
Pinocchio CLI - Multi-Agent Collaboration System
"""

import argparse
import asyncio
import atexit
import logging
import signal
import sys
import time
from pathlib import Path
from typing import List, Optional

from rich.console import Console
from rich.panel import Panel

from pinocchio.config.config_manager import ConfigManager, get_config_value
from pinocchio.coordinator import Coordinator
from pinocchio.data_models.task_planning import AgentType, Task, TaskStatus
from pinocchio.session.models.session import Session
from pinocchio.utils.file_utils import get_output_path
from pinocchio.utils.verbose_logger import (
    LogLevel,
    VerboseLogger,
    get_verbose_logger,
    set_verbose_logger,
)

logger = logging.getLogger(__name__)

from pinocchio.utils.string_utils import extract_and_pretty_json_from_str


def print_logo_and_welcome(console):
    """Print the logo and welcome info only once, not in any panel."""
    # Print logo
    try:
        from pyfiglet import Figlet

        start_color = "#8B5CF6"  # Purple
        end_color = "#10B981"  # Green
        text = "PINOCCHIO"
        f = Figlet(font="block", width=90)
        ascii_art = f.renderText(text).splitlines()
        max_width = max(len(line) for line in ascii_art if line.strip())
        for line in ascii_art:
            if line.strip():
                filled_line = line.replace("_", "█").replace("|", "█")
                colored_line = ""
                for i, char in enumerate(filled_line):
                    if char == "█":
                        ratio = i / max_width
                        color = interpolate_color(start_color, end_color, ratio)
                        colored_line += f"[bold {color}]█[/]"
                    else:
                        colored_line += char
                console.print(colored_line)
    except ImportError:
        # Fallback to simple logo with gradient
        colors = [
            "#8B5CF6",
            "#7C3AED",
            "#6D28D9",
            "#4C1D95",
            "#047857",
            "#059669",
            "#10B981",
            "#34D399",
            "#6EE7B7",
        ]
        text = "PINOCCHIO"
        styled = ""
        for i, c in enumerate(text):
            color = colors[i % len(colors)]
            styled += f"[bold {color}]{c}[/]"
        console.print(styled.center(80))
    # Print welcome info
    console.print("🎭 Pinocchio CLI - Multi-Agent Collaboration System", style="bold")
    console.print("Type your request and press Enter to start...")
    console.print("Type '/help' for available commands")
    console.print("Type '/quit' to exit\n")


def hex_to_rgb(hex_color: str) -> tuple:
    """Convert hex color to RGB tuple."""
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))


def rgb_to_hex(rgb: tuple) -> str:
    """Convert RGB tuple to hex color."""
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


def interpolate_color(color1: str, color2: str, ratio: float) -> str:
    """Linearly interpolate between two hex colors."""
    rgb1 = hex_to_rgb(color1)
    rgb2 = hex_to_rgb(color2)
    interpolated = tuple(int(rgb1[i] + (rgb2[i] - rgb1[i]) * ratio) for i in range(3))
    return rgb_to_hex(interpolated)


def print_block_logo():
    """Print large block font PINOCCHIO logo with smooth linear gradient"""
    console = Console()

    try:
        from pyfiglet import Figlet

        start_color = "#8B5CF6"  # Purple
        end_color = "#10B981"  # Green
        text = "PINOCCHIO"
        f = Figlet(font="block", width=90)
        ascii_art = f.renderText(text).splitlines()

        # Find the maximum width of the logo
        max_width = max(len(line) for line in ascii_art if line.strip())

        # Print each line with smooth left-to-right color gradient
        for line in ascii_art:
            if line.strip():
                filled_line = line.replace("_", "█").replace("|", "█")
                colored_line = ""
                for i, char in enumerate(filled_line):
                    if char == "█":
                        ratio = i / max_width
                        color = interpolate_color(start_color, end_color, ratio)
                        colored_line += f"[bold {color}]█[/]"
                    else:
                        colored_line += char
                console.print(colored_line)

        console.print(
            "\n🎭 Pinocchio CLI - Multi-Agent Collaboration System", style="bold"
        )
        console.print("Type your request and press Enter to start...")
        console.print("Type '/help' for available commands")
        console.print("Type '/quit' to exit\n")

    except ImportError:
        console.print("[red]pyfiglet not installed. Falling back to simple logo.[/red]")
        print_simple_logo()


def print_simple_logo():
    """Print simple logo when pyfiglet is not available."""
    console = Console()
    colors = [
        "#8B5CF6",
        "#7C3AED",
        "#6D28D9",
        "#4C1D95",
        "#047857",
        "#059669",
        "#10B981",
        "#34D399",
        "#6EE7B7",
    ]
    text = "PINOCCHIO"
    styled = ""
    for i, c in enumerate(text):
        color = colors[i % len(colors)]
        styled += f"[bold {color}]{c}[/]"
    console.print(styled.center(80))
    console.print("\n🎭 Pinocchio CLI - Multi-Agent Collaboration System", style="bold")
    console.print("Type your request and press Enter to start...")
    console.print("Type '/help' for available commands")
    console.print("Type '/quit' to exit\n")


def print_logo():
    """Print the Pinocchio CLI logo with block style"""
    print_block_logo()


def print_help():
    """Print help information"""
    help_text = """
📋 Pinocchio CLI - Available Commands

Navigation & Control:
• /chat    - Enter multi-agent conversation mode
• /cmd     - Return to welcome page (from chat mode)
• /quit    - Exit Pinocchio CLI

History & Help:
• /history - View your previous commands and sessions
• /help    - Show this help message
• /clear   - Clear the screen and refresh display

Session Management:
• /status  - Show current session status
• /sessions- List all active sessions
• /export  - Export current session data

Development Tools:
• /debug   - Enable debug mode for verbose output
• /verbose - Toggle verbose logging
• /config  - Show current configuration
• /performance - Show performance metrics
• /logs    - Export verbose logs

Usage Examples:
• Type '/chat' to start coding with AI agents
• Type '/history' to see your previous commands
• Type '/help' anytime for this help message
• Type '/quit' to exit the application

For more information, visit the Pinocchio documentation.
    """
    print(help_text)


def handle_commands(
    command: str, cli_instance: Optional["PinocchioCLI"] = None
) -> bool:
    """Handle CLI commands. Returns True if should continue, False if should quit."""
    console = Console()

    if command == "/help":
        print_help()
        return True
    elif command == "/quit":
        console.print("[yellow]Exiting Pinocchio CLI...[/yellow]")
        return False
    elif command == "/clear":
        # Remove unbound print_welcome_screen call or replace with self/legacy layout call if needed
        console.print(
            "[yellow]Screen cleared (logo/tips not shown in this mode).[/yellow]"
        )
        return True
    elif command == "/status":
        # Implement status command
        console.print("[blue]System Status:[/blue]")
        console.print("  • CLI: Active")
        console.print("  • Agents: Ready")
        console.print("  • LLM: Connected")
        console.print("  • Memory: Available")
        console.print("  • Knowledge: Loaded")
        return True
    elif command == "/history":
        # Implement history command
        console.print("[blue]Command History:[/blue]")
        console.print("  • Recent commands will be displayed here")
        console.print("  • Session history available")
        console.print("  • Use /help for available commands")
        return True
    elif command == "/verbose":
        if cli_instance:
            cli_instance.toggle_verbose_mode()
        else:
            console.print("[yellow]Verbose mode toggle requires CLI instance[/yellow]")
        return True
    elif command == "/performance":
        if cli_instance:
            cli_instance.show_performance_metrics()
        else:
            console.print("[yellow]Performance metrics require CLI instance[/yellow]")
        return True
    elif command == "/logs":
        if cli_instance:
            cli_instance.export_verbose_logs()
        else:
            console.print("[yellow]Log export requires CLI instance[/yellow]")
        return True
    elif command == "/session":
        if cli_instance:
            cli_instance.show_session_summary()
        else:
            console.print("[yellow]Session summary requires CLI instance[/yellow]")
        return True
    else:
        console.print(f"[red]Unknown command: {command}[/red]")
        console.print("Type '/help' for available commands")
        return True


# Legacy implementation - kept for reference
class LegacyPinocchioLayout:
    """Legacy layout manager for Pinocchio CLI - kept for reference."""

    def __init__(self):
        """Initialize legacy layout."""
        self.console = Console()
        self.command_history = []
        # Chat interface data
        self.todolist_tasks: List[Task] = []
        self.conversation_messages: List[str] = []
        self.current_plan_id: Optional[str] = None
        self.coordinator: Optional[Coordinator] = None

        # Refresh control
        self.last_refresh_time = 0
        self.refresh_interval = 0.5  # Refresh every 0.5 seconds
        self.needs_refresh = False

        # Display limits
        self.max_conversation_lines = 15
        self.max_todolist_lines = 10
        self.max_panel_height = 20

    def set_coordinator(self, coordinator: Coordinator):
        """Set the coordinator for this layout."""
        self.coordinator = coordinator

    def add_command_to_history(self, command: str):
        """Add command to history."""
        self.command_history.append(command)

    def add_conversation_message(self, message: str):
        """Add message to conversation and mark for refresh."""
        self.conversation_messages.append(message)
        self.needs_refresh = True

    def clear_conversation(self):
        """Clear conversation messages."""
        self.conversation_messages.clear()
        self.needs_refresh = True

    def update_todolist(self, tasks: List[Task], plan_id: Optional[str] = None):
        """Update todolist tasks and mark for refresh."""
        self.todolist_tasks = tasks
        self.current_plan_id = plan_id
        self.needs_refresh = True

    def update_todolist_from_coordinator(self):
        """Update todolist from coordinator and mark for refresh."""
        if self.coordinator and hasattr(self.coordinator, "todolist_tasks"):
            self.todolist_tasks = self.coordinator.todolist_tasks
            self.current_plan_id = getattr(self.coordinator, "current_plan_id", None)
            self.needs_refresh = True

    def should_refresh(self) -> bool:
        """Check if it's time to refresh the display."""
        current_time = time.time()
        if (
            self.needs_refresh
            and (current_time - self.last_refresh_time) >= self.refresh_interval
        ):
            self.last_refresh_time = current_time
            self.needs_refresh = False
            return True
        return False

    def create_logo_panel(self) -> Panel:
        """Create the logo and welcome panel."""
        try:
            from pyfiglet import Figlet

            start_color = "#8B5CF6"  # Purple
            end_color = "#10B981"  # Green
            text = "PINOCCHIO"
            f = Figlet(font="block", width=90)
            ascii_art = f.renderText(text).splitlines()
            max_width = max(len(line) for line in ascii_art if line.strip())

            logo_lines = []
            for line in ascii_art:
                if line.strip():
                    filled_line = line.replace("_", "█").replace("|", "█")
                    # Create gradient effect
                    gradient_line = ""
                    for i, char in enumerate(filled_line):
                        ratio = i / max_width
                        color = interpolate_color(start_color, end_color, ratio)
                        gradient_line += f"[{color}]{char}[/{color}]"
                    logo_lines.append(gradient_line)
                else:
                    logo_lines.append("")

            content = "\n".join(logo_lines) + "\n"
            return Panel(
                content,
                title="🤖 Pinocchio CLI",
                border_style="purple",
                padding=(1, 2),
            )
        except ImportError:
            # Fallback to simple logo
            content = """
[bold purple]PINOCCHIO[/bold purple]
[dim]Multi-Agent Collaboration System[/dim]
            """
            return Panel(
                content,
                title="🤖 Pinocchio CLI",
                border_style="purple",
                padding=(1, 2),
            )

    def create_tips_panel(self) -> Panel:
        """Create the tips panel."""
        content = """
[bold green]Available Commands:[/bold green]
• /chat - Enter chat interface for multi-agent collaboration
• /history - Show command history
• /help - Show this help message
• /quit - Exit the application

[bold blue]Tips:[/bold blue]
• Use /chat to start collaborating with AI agents
• Commands are case-insensitive
• Type your questions or tasks in natural language
        """
        return Panel(
            content,
            title="💡 Tips & Commands",
            border_style="blue",
            padding=(1, 2),
        )

    def create_conversation_panel(self) -> Panel:
        """Create the conversation panel with dynamic height based on content and screen."""
        if not self.conversation_messages:
            content = "No messages yet"
            display_lines = 1
        else:
            # Get screen height and calculate available space
            try:
                import shutil

                screen_height = shutil.get_terminal_size().lines
            except Exception:
                screen_height = 24  # Fallback height

            # Calculate available height for panels (subtract logo panel height, borders, input area)
            # Logo panel is about 10 lines, input area is about 3 lines, so reserve about 13 lines
            available_height = max(
                screen_height - 13, 6
            )  # Reserve space for logo and input

            # Calculate how many lines we can show
            max_display_lines = max(available_height - 2, 1)  # Subtract 2 for borders

            # Get only the last N messages that fit
            recent_messages = self.conversation_messages[-max_display_lines:]
            content_lines = []
            content_lines.append("💬 Conversation")
            content_lines.append("")

            for message in recent_messages:
                # Wrap long messages to fit panel width
                wrapped_lines = self._wrap_text(message, width=60)
                content_lines.extend(wrapped_lines)

            content = "\n".join(content_lines)
            display_lines = len(content_lines)

        return Panel(
            content,
            title="💬 Conversation",
            border_style="green",
            padding=(0, 1),
            height=display_lines,
        )

    def create_todolist_panel(self) -> Panel:
        """Create the todolist panel with dynamic height based on content and screen."""
        if not self.todolist_tasks:
            content = "No active tasks"
            display_lines = 1
        else:
            # Get screen height and calculate available space
            try:
                import shutil

                screen_height = shutil.get_terminal_size().lines
            except Exception:
                screen_height = 24  # Fallback height

            # Calculate available height for panels (subtract logo panel height, borders, input area)
            # Logo panel is about 10 lines, input area is about 3 lines, so reserve about 13 lines
            available_height = max(
                screen_height - 13, 6
            )  # Reserve space for logo and input

            # Calculate how many lines we can show
            max_display_lines = max(available_height - 2, 1)  # Subtract 2 for borders

            # Limit the number of tasks displayed
            recent_tasks = self.todolist_tasks[-max_display_lines:]
            content_lines = []
            content_lines.append("📋 Task Plan")
            if self.current_plan_id:
                content_lines.append(f"Plan ID: {self.current_plan_id}")
            content_lines.append("")

            for i, task in enumerate(recent_tasks, 1):
                status_emoji = {
                    TaskStatus.PENDING: "⏳",
                    TaskStatus.RUNNING: "🔄",
                    TaskStatus.COMPLETED: "✅",
                    TaskStatus.FAILED: "❌",
                }.get(task.status, "❓")
                agent_emoji = {
                    AgentType.GENERATOR: "⚡",
                    AgentType.DEBUGGER: "🔧",
                    AgentType.OPTIMIZER: "🚀",
                    AgentType.EVALUATOR: "📊",
                }.get(task.agent_type, "🤖")

                deps = (
                    ", ".join([dep.task_id for dep in task.dependencies])
                    if task.dependencies
                    else "-"
                )

                # Wrap task description to fit panel width
                task_desc = task.task_description
                if len(task_desc) > 25:  # Shorter limit for todolist
                    task_desc = task_desc[:25] + "..."

                task_line = f"{i}. {status_emoji} {agent_emoji} {task_desc}"
                content_lines.append(task_line)
                content_lines.append(f"   Status: {task.status.value}")
                content_lines.append(f"   Dependencies: {deps}")
                content_lines.append("")

            content = "\n".join(content_lines)
            display_lines = len(content_lines)

        return Panel(
            content,
            title="📋 Task Plan",
            border_style="blue",
            padding=(0, 1),
            height=display_lines,
        )

    def print_welcome_screen(self):
        """Print the welcome screen without Live display."""
        # Print logo
        self.console.print(self.create_logo_panel())

        # Print tips
        self.console.print(self.create_tips_panel())

        # Print input prompt
        self.console.print("\n[bold green]Enter your command:[/bold green]")

    def print_chat_screen(self, show_logo: bool = True, show_panels: bool = True):
        """Print the chat screen without Live display."""
        if show_logo:
            # Print logo only on first display
            self.console.print(self.create_logo_panel())

        if show_panels:
            # Print conversation and todolist panels side by side without Layout
            from rich.columns import Columns

            # Create panels
            conversation_panel = self.create_conversation_panel()
            todolist_panel = self.create_todolist_panel()

            # Use Columns for side-by-side layout without extra spacing
            columns = Columns(
                [conversation_panel, todolist_panel], equal=False, expand=True
            )
            self.console.print(columns, end="")

        # Print input prompt immediately after panels
        self.console.print("\n[bold green]Enter your command:[/bold green]")

    def print_panels_only(self):
        """Print only the panels without logo and input prompt."""
        # Print conversation and todolist panels directly (left-right using columns)
        # Create a custom layout with specific widths
        from rich.layout import Layout

        # Create panels
        conversation_panel = self.create_conversation_panel()
        todolist_panel = self.create_todolist_panel()

        # Create a layout for better control
        layout_container = Layout()
        layout_container.split_row(
            Layout(conversation_panel, ratio=3, name="conversation"),
            Layout(todolist_panel, ratio=1, name="todolist"),
        )

        self.console.print(layout_container)

    def _wrap_text(self, text: str, width: int) -> List[str]:
        """Wrap text to fit within specified width."""
        if len(text) <= width:
            return [text]

        lines = []
        words = text.split()
        current_line = ""

        for word in words:
            if len(current_line + " " + word) <= width:
                current_line += (" " + word) if current_line else word
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word

        if current_line:
            lines.append(current_line)

        return lines


# New simplified design
class PinocchioCLI:
    """New simplified Pinocchio CLI design."""

    def __init__(self):
        """Initialize the new CLI."""
        self.console = Console()
        self.coordinator: Optional[Coordinator] = None
        self.messages: List[str] = []
        self.tasks: List[Task] = []
        self.last_task_plan = None
        # Task plan overview buffering
        self.task_plan_overview_buffer = []
        self.is_collecting_task_plan_overview = False

        # Task details buffering
        self.task_details_buffer = []
        self.is_collecting_task_details = False

        # Verbose mode state
        self.verbose_mode = False

        # Initialize verbose logger
        self.verbose_logger = VerboseLogger(
            console=self.console,
            log_file=Path("./logs/verbose.log") if self.verbose_mode else None,
            max_depth=5,
            enable_colors=True,
        )
        set_verbose_logger(self.verbose_logger)

    def set_coordinator(self, coordinator: Coordinator):
        """Set the coordinator."""
        self.coordinator = coordinator

    def print_logo(self):
        """Print the Pinocchio logo."""
        try:
            from pyfiglet import Figlet

            start_color = "#8B5CF6"  # Purple
            end_color = "#10B981"  # Green
            text = "PINOCCHIO"
            f = Figlet(font="block", width=90)
            ascii_art = f.renderText(text).splitlines()
            max_width = max(len(line) for line in ascii_art if line.strip())

            logo_lines = []
            for line in ascii_art:
                if line.strip():
                    filled_line = line.replace("_", "█").replace("|", "█")
                    # Create gradient effect
                    gradient_line = ""
                    for i, char in enumerate(filled_line):
                        ratio = i / max_width
                        color = interpolate_color(start_color, end_color, ratio)
                        gradient_line += f"[{color}]{char}[/{color}]"
                    logo_lines.append(gradient_line)
                else:
                    logo_lines.append("")

            content = "\n".join(logo_lines) + "\n"
            self.console.print(content)
        except ImportError:
            # Fallback to simple logo
            self.console.print("[bold purple]PINOCCHIO[/bold purple]")
            self.console.print("[dim]Multi-Agent Collaboration System[/dim]\n")

    def print_tips(self):
        """Print tips and help information."""
        tips = """
[bold green]Welcome to Pinocchio![/bold green]

[bold blue]How to use:[/bold blue]
• Type your questions or tasks in natural language
• Watch as multiple AI agents collaborate to solve your problems
• Each agent's action and message will be displayed in sequence
• Task completion status will be shown separately

[bold yellow]Example prompts:[/bold yellow]
• "Write a high-performance sorting algorithm"
• "Create a thread-safe queue implementation"
• "Optimize this code for better performance"
• "Debug this memory leak issue"

[bold red]Commands:[/bold red]
• /quit - Exit the application
• /help - Show this help message
        """
        self.console.print(tips)

    def print_input_border(self):
        """Print input area with border."""
        self.console.print("\n" + "─" * 80)
        self.console.print("[bold green]Enter your prompt:[/bold green]")
        self.console.print("─" * 80)

    def add_message(self, message: str, sender: str = "🎭 pinocchio"):
        """Add a message to the conversation."""
        import re

        overview_start_pat = re.compile(r".*📋 Task Plan Overview:")
        details_start_pat = re.compile(r".*📋 Task Details:")

        # If buffering task plan overview, check for END marker first
        if self.is_collecting_task_plan_overview:
            # Compatible with prefix
            if message.strip().endswith("<<END_TASK_PLAN>>"):
                self._flush_task_plan_overview()
                return
            else:
                self.task_plan_overview_buffer.append(message)
                return

        # If buffering task details, check for END marker first
        if self.is_collecting_task_details:
            if message.strip().endswith("<<END_TASK_DETAILS>>"):
                self._flush_task_details()
                return
            else:
                self.task_details_buffer.append(message)
                return

        # Start buffering if overview start
        if overview_start_pat.match(message):
            self.is_collecting_task_plan_overview = True
            self.task_plan_overview_buffer = [message]
            return

        # Start buffering if details start
        if details_start_pat.match(message):
            self.is_collecting_task_details = True
            self.task_details_buffer = [message]
            return

        # Regular message processing
        self._process_regular_message(message, sender)

    def _process_regular_message(self, message: str, sender: str):
        """Process a regular message with timestamp and sender."""
        timestamp = time.strftime("%H:%M:%S")
        lines = message.split("\n")
        if lines:
            first_line = lines[0]
            formatted_first_line = f"[dim]{timestamp}[/dim] [{sender}] {first_line}"
            self.messages.append(formatted_first_line)
            self.console.print(formatted_first_line)
            for line in lines[1:]:
                if line.strip():
                    self.messages.append(line)
                    self.console.print(line)

    def _flush_task_plan_overview(self):
        """Flush task plan overview buffer as a single panel."""
        if not self.task_plan_overview_buffer:
            return

        timestamp = time.strftime("%H:%M:%S")
        content = "\n".join(self.task_plan_overview_buffer)

        panel = Panel(
            content,
            title=f"[dim]{timestamp}[/dim] 📋 Task Plan Overview",
            border_style="blue",
            padding=(1, 2),
        )
        self.console.print(panel)

        # Reset buffer
        self.task_plan_overview_buffer = []
        self.is_collecting_task_plan_overview = False

    def _flush_task_details(self):
        """Flush task details buffer as a single panel."""
        if not self.task_details_buffer:
            return

        timestamp = time.strftime("%H:%M:%S")
        content = "\n".join(self.task_details_buffer)

        panel = Panel(
            content,
            title=f"[dim]{timestamp}[/dim] 📋 Task Details",
            border_style="green",
            padding=(1, 2),
        )
        self.console.print(panel)

        # Reset buffer
        self.task_details_buffer = []
        self.is_collecting_task_details = False

    def add_user_prompt(self, prompt: str):
        """Add user prompt to conversation."""
        self.add_message(f"User: {prompt}", "👤 User")
        self.console.print("─" * 80)

    def add_agent_message(self, message: str, agent_name: str):
        """Add agent message to conversation."""
        # Check if message contains JSON
        if self._is_json_message(message):
            self._print_json_message(message, agent_name)
        else:
            self.add_message(f"{agent_name}: {message}", f"🤖 {agent_name}")

    def add_coordinator_message(self, message: str):
        """Add coordinator message to conversation."""
        if self._is_json_message(message):
            self._print_json_message(message, "Coordinator")
        else:
            self.add_message(f"Coordinator: {message}", "🎯 Coordinator")

    def add_llm_verbose_message(self, message: str):
        """Add an LLM VERBOSE message with timestamp, icon, and pretty JSON if present."""
        import time

        from rich.console import Console
        from rich.panel import Panel

        timestamp = time.strftime("%H:%M:%S")
        icon = "🤖 LLM"
        border_style = "green"
        bg = "on grey15"
        console = Console()
        # Preserve the original json extraction content logic (if any), but do not attempt to extract/highlight JSON structure separately
        # Directly print the entire message content as is, keeping rich/Panel style
        panel = Panel(
            message,
            title=f"{icon} LLM",
            subtitle=timestamp,
            border_style=border_style,
            style=bg,
            expand=True,
        )
        console.print(panel)

    def toggle_verbose_mode(self):
        """Toggle verbose logging mode."""
        self.verbose_mode = not self.verbose_mode
        if self.verbose_mode:
            # Create logs directory if it doesn't exist
            Path("./logs").mkdir(exist_ok=True)
            self.verbose_logger.log_file = Path("./logs/verbose.log")
            self.verbose_logger.log(
                LogLevel.INFO,
                "cli",
                "Verbose logging enabled",
                data={"mode": "enabled", "log_file": str(self.verbose_logger.log_file)},
            )
            self.console.print("[green]Verbose logging enabled[/green]")
        else:
            self.verbose_logger.log_file = None
            self.verbose_logger.log(
                LogLevel.INFO,
                "cli",
                "Verbose logging disabled",
                data={"mode": "disabled"},
            )
            self.console.print("[yellow]Verbose logging disabled[/yellow]")

    def show_performance_metrics(self):
        """Display performance metrics."""
        self.verbose_logger.display_performance_summary()

    def export_verbose_logs(self, file_path: str = None):
        """Export verbose logs to file."""
        if not file_path:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            file_path = f"./logs/verbose_export_{timestamp}.json"

        try:
            self.verbose_logger.export_entries(Path(file_path))
            self.console.print(f"[green]Verbose logs exported to: {file_path}[/green]")
        except Exception as e:
            self.console.print(f"[red]Failed to export logs: {e}[/red]")

    def show_session_summary(self, session_id: str = None):
        """Display session summary."""
        if (
            not session_id
            and hasattr(self, "current_session_id")
            and self.current_session_id
        ):
            session_id = self.current_session_id

        if session_id:
            self.verbose_logger.display_session_summary(session_id)
        else:
            self.console.print("[yellow]No active session[/yellow]")

    def _is_json_message(self, message: str) -> bool:
        """Check if message contains JSON content."""
        import json

        try:
            # Look for JSON patterns in the message
            if "{" in message and "}" in message:
                # Try to extract and parse JSON
                start = message.find("{")
                end = message.rfind("}") + 1
                json_str = message[start:end]
                json.loads(json_str)
                return True
        except (json.JSONDecodeError, ValueError):
            pass
        return False

    def _print_json_message(self, message: str, sender: str):
        """Print JSON message, recursively pretty-print nested JSON in all string fields."""
        import json

        from rich.syntax import Syntax

        try:
            data = json.loads(message) if isinstance(message, str) else message
            multiline_fields = []

            def pretty_print_nested(obj, path=None):
                if path is None:
                    path = []
                if isinstance(obj, dict):
                    new_obj = {}
                    for k, v in obj.items():
                        if isinstance(v, str):
                            s = v.strip()
                            if "\n" in s:
                                new_obj[k] = "<see below>"
                                multiline_fields.append((path + [k], s))
                            else:
                                new_obj[k] = s
                        elif isinstance(v, (dict, list)):
                            new_obj[k] = pretty_print_nested(v, path + [k])
                        else:
                            new_obj[k] = v
                    return new_obj
                elif isinstance(obj, list):
                    return [
                        pretty_print_nested(i, path + [str(idx)])
                        for idx, i in enumerate(obj)
                    ]
                else:
                    return obj

            replaced = pretty_print_nested(data)
            pretty = json.dumps(replaced, indent=2, ensure_ascii=False)
            syntax = Syntax(pretty, "json", theme="monokai", line_numbers=False)
            self.console.print(syntax)
        except Exception as e:
            self.console.print(f"[red]Failed to pretty-print JSON: {e}[/red]")
            self.console.print(f"[red]Raw message:[/red] {message}")
            raise

    def print_task_update(self, task: Task):
        """Print task status update."""

        def get_enum_value(val):
            return val.value if hasattr(val, "value") else val

        status_emoji = {
            get_enum_value(TaskStatus.PENDING): "⏳",
            get_enum_value(TaskStatus.RUNNING): "🔄",
            get_enum_value(TaskStatus.COMPLETED): "✅",
            get_enum_value(TaskStatus.FAILED): "❌",
        }.get(get_enum_value(task.status), "❓")

        agent_emoji = {
            get_enum_value(AgentType.GENERATOR): "⚡",
            get_enum_value(AgentType.DEBUGGER): "🔧",
            get_enum_value(AgentType.OPTIMIZER): "🚀",
            get_enum_value(AgentType.EVALUATOR): "📊",
        }.get(get_enum_value(task.agent_type), "🤖")

        # Create panel for task update
        content = f"""
{status_emoji} {agent_emoji} {task.task_description}
Status: {get_enum_value(task.status)}
Task ID: {task.task_id}
        """.strip()

        panel = Panel(
            content,
            title="🔄 Task Update",
            border_style="yellow",
            padding=(1, 2),
        )
        self.console.print(panel)
        self.console.print("─" * 80)

    def print_task_plan(self, tasks: List[Task], plan_id: Optional[str] = None):
        """Print task plan in a single panel with one timestamp."""

        def get_enum_value(val):
            return val.value if hasattr(val, "value") else val

        current_plan = (tasks, plan_id)
        if current_plan == self.last_task_plan:
            return
        self.last_task_plan = current_plan
        timestamp = time.strftime("%H:%M:%S")
        if not tasks:
            content = "No active tasks"
        else:
            content_lines = []
            if plan_id:
                content_lines.append(f"Plan ID: {plan_id}")
            content_lines.append("")
            for i, task in enumerate(tasks, 1):
                status_emoji = {
                    get_enum_value(TaskStatus.PENDING): "⏳",
                    get_enum_value(TaskStatus.RUNNING): "🔄",
                    get_enum_value(TaskStatus.COMPLETED): "✅",
                    get_enum_value(TaskStatus.FAILED): "❌",
                }.get(get_enum_value(task.status), "❓")
                agent_emoji = {
                    get_enum_value(AgentType.GENERATOR): "⚡",
                    get_enum_value(AgentType.DEBUGGER): "🔧",
                    get_enum_value(AgentType.OPTIMIZER): "🚀",
                    get_enum_value(AgentType.EVALUATOR): "📊",
                }.get(get_enum_value(task.agent_type), "🤖")
                desc = task.task_description
                if len(desc) > 60:
                    desc = desc[:57] + "..."
                task_line = f"{i}. {status_emoji} {agent_emoji} {desc}"
                content_lines.append(task_line)
                content_lines.append(
                    f"   Status: {get_enum_value(task.status)} | ID: {task.task_id}"
                )
                # Additional info like dependencies, descriptions can be added as needed
                if hasattr(task, "dependencies") and task.dependencies:
                    deps = ", ".join(
                        [d.task_id for d in getattr(task, "dependencies", [])]
                    )
                    content_lines.append(f"   Dependencies: {deps}")
                if hasattr(task, "requirements") and task.requirements:
                    content_lines.append(f"   Requirements: {task.requirements}")
                if hasattr(task, "optimization_goals") and task.optimization_goals:
                    content_lines.append(
                        f"   Optimization Goals: {task.optimization_goals}"
                    )
                if hasattr(task, "result") and task.result:
                    content_lines.append(f"   Result: {task.result}")
                content_lines.append("")
            content = "\n".join(content_lines)
        panel = Panel(
            content,
            title=f"[dim]{timestamp}[/dim] 📋 Task Plan",
            border_style="blue",
            padding=(1, 2),
        )
        self.console.print(panel)

    def print_welcome(self):
        """Print welcome screen."""
        self.console.clear()
        self.print_logo()
        self.print_tips()
        self.print_input_border()

    async def run(self):
        """Run the new CLI."""
        self.print_welcome()

        while True:
            try:
                # Get user input
                user_input = input("> ").strip()

                if user_input.startswith("/"):
                    command = user_input.lower().strip()
                    # Use the centralized command handler
                    if not handle_commands(command, self):
                        break
                    self.print_input_border()
                    continue

                if not user_input:
                    continue

                # Add user prompt
                self.add_user_prompt(user_input)

                # Process with coordinator
                if self.coordinator:
                    async for message in self.coordinator.process_user_request(
                        user_input
                    ):
                        # Check for task plan overview first
                        if (
                            "📋 Task Plan Overview:" in message
                            or self.is_collecting_task_plan_overview
                        ):
                            self.add_message(message)
                        elif "coordinator" in message.lower():
                            self.add_coordinator_message(message)
                        elif any(
                            agent in message.lower()
                            for agent in [
                                "generator",
                                "debugger",
                                "optimizer",
                                "evaluator",
                            ]
                        ):
                            # Extract agent name from message
                            agent_name = "Agent"
                            for agent in [
                                "Generator",
                                "Debugger",
                                "Optimizer",
                                "Evaluator",
                            ]:
                                if agent.lower() in message.lower():
                                    agent_name = agent
                                    break
                            self.add_agent_message(message, agent_name)
                        elif "task" in message.lower() and (
                            "update" in message.lower() or "status" in message.lower()
                        ):
                            # Handle task updates
                            self.add_message(message, "🔄 Task System")
                        elif self._is_json_message(message):
                            # Handle JSON messages
                            self._print_json_message(message, "System")
                        else:
                            self.add_message(message)

                    # Update task plan if available
                    if hasattr(self.coordinator, "todolist_tasks"):
                        self.tasks = self.coordinator.todolist_tasks
                        self.print_task_plan(
                            self.tasks,
                            getattr(self.coordinator, "current_plan_id", None),
                        )

                    # Flush any remaining task plan overview buffer
                    if self.is_collecting_task_plan_overview:
                        self._flush_task_plan_overview()

                self.print_input_border()

            except KeyboardInterrupt:
                self.console.print("\n[yellow]Use '/quit' to exit[/yellow]")
            except EOFError:
                self.console.print("\n[yellow]Use '/quit' to exit[/yellow]")
                break
            except Exception as e:
                self.add_message(f"Error: {e}", "❌ Error")
                self.print_input_border()
        # After the loop, ensure emergency_save is called
        import asyncio

        asyncio.run(asyncio.to_thread(emergency_save))


# Legacy functions for backward compatibility
async def legacy_main():
    """Legacy main CLI entry point."""
    console = Console()
    # Load configuration
    try:
        config_manager = ConfigManager()
        config_manager.config
    except Exception as e:
        console.print(f"[red]Error loading configuration: {e}[/red]")
        return
    try:
        from pinocchio.llm.custom_llm_client import CustomLLMClient

        llm_config = config_manager.get_llm_config()
        llm_client = CustomLLMClient(llm_config)
    except Exception as e:
        console.print(f"[red]Error creating LLM client: {e}[/red]")
        return

    # Load verbose configuration
    verbose_config = config_manager.get("verbose", None)
    if verbose_config is not None:
        verbose_enabled = verbose_config.enabled
    else:
        verbose_enabled = True

    # Set up logging if verbose is enabled
    if verbose_enabled:
        from pinocchio.errors.logging import setup_logging

        logging_config = config_manager.get("logging", None)
        if logging_config is not None:
            log_level = logging_config.level
            console_output = logging_config.console_output
            file_output = logging_config.file_output
        else:
            log_level = "DEBUG"
            console_output = True
            file_output = True
        log_level_map = {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40}
        setup_logging(
            log_level=log_level_map.get(log_level, 10),
            console_output=console_output,
            file_output=file_output,
        )

    coordinator = Coordinator(llm_client)
    layout = LegacyPinocchioLayout()
    layout.set_coordinator(coordinator)

    # Start with welcome page
    console.clear()
    layout.print_welcome_screen()

    # Simple input loop without Live display
    while True:
        try:
            # Get user input - ensure cursor is visible
            console.print("\n> ", end="")
            user_input = input().strip()

            if user_input.startswith("/"):
                command = user_input.lower().strip()

                if command == "/quit":
                    console.print("[yellow]Exiting Pinocchio CLI...[/yellow]")
                    break
                elif command == "/help":
                    print_help()
                    continue
                elif command == "/history":
                    # Show command history
                    if layout.command_history:
                        console.print("\n[blue]Command History:[/blue]")
                        for i, cmd in enumerate(layout.command_history[-10:], 1):
                            console.print(f"{i}. {cmd}")
                    else:
                        console.print("\n[blue]No command history[/blue]")
                    continue
                elif command == "/chat":
                    # Switch to chat interface
                    await legacy_run_chat_interface(console, coordinator, layout)
                    # Return to welcome page after chat
                    console.clear()
                    console.print("[green]Welcome back![/green]")
                    layout.print_welcome_screen()
                    continue
                else:
                    console.print(f"[red]Unknown command: {command}[/red]")
                    console.print("Type '/help' for available commands")
                    continue

            # Add to history if not empty
            if user_input.strip():
                layout.add_command_to_history(user_input)
                console.print(f"[dim]Command added to history: {user_input}[/dim]")

        except KeyboardInterrupt:
            console.print("\n[yellow]Use '/quit' to exit[/yellow]")
        except EOFError:
            console.print("\n[yellow]Use '/quit' to exit[/yellow]")
            break
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")


async def legacy_run_chat_interface(
    console: Console, coordinator: Coordinator, layout: LegacyPinocchioLayout
):
    """Legacy chat interface."""
    console.clear()
    layout.print_chat_screen(
        show_logo=True, show_panels=True
    )  # Show logo on first display

    # Simple input loop without Live display
    while True:
        try:
            # Get user input - ensure cursor is visible
            console.print("\n> ", end="")
            user_input = input().strip()

            if user_input.startswith("/"):
                command = user_input.lower().strip()
                if command == "/cmd":
                    # Return to welcome page with message
                    console.print("[yellow]Returning to welcome page...[/yellow]")
                    return
                elif command == "/quit":
                    console.print("[yellow]Exiting Pinocchio CLI...[/yellow]")
                    exit(0)
                elif command == "/help":
                    print_help()
                    continue
                else:
                    console.print(f"[red]Unknown command: {command}[/red]")
                    console.print("Type '/help' for available commands")
                    continue

            if user_input.strip():
                layout.add_conversation_message(f" User: {user_input}")

                # Process the request and update data
                async for message in coordinator.process_user_request(user_input):
                    layout.add_conversation_message(message)
                    layout.update_todolist_from_coordinator()

                    # Simple refresh: clear and reprint
                    console.clear()
                    layout.print_chat_screen(show_logo=True, show_panels=True)

                layout.add_conversation_message("✅ Request completed!")
                # Final refresh
                console.clear()
                layout.print_chat_screen(show_logo=True, show_panels=True)

        except KeyboardInterrupt:
            console.print("\n[yellow]Use '/quit' to exit[/yellow]")
        except EOFError:
            console.print("\n[yellow]Use '/quit' to exit[/yellow]")
            break
        except Exception as e:
            layout.add_conversation_message(f"❌ Error: {e}")
            # Refresh on error
            console.clear()
            layout.print_chat_screen(show_logo=True, show_panels=True)


# New main function
async def main():
    """Main CLI entry point with new design."""
    import os
    import shutil
    import sys
    from pathlib import Path

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Pinocchio CLI - Multi-Agent Collaboration System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Configuration Modes:
  --mode development    # Development mode - raises errors instead of fallback
  --mode production     # Production mode - allows fallback for robustness

Execution Strategies:
  --strategy workflow   # Use fixed workflow patterns
  --strategy planning   # Use adaptive task planning

Testing Options:
  --dry-run            # Use mock LLM responses for fast testing

Examples:
  pinocchio --mode development --strategy workflow
  pinocchio --mode production --strategy planning
  pinocchio --strategy workflow --config configs/workflow_template.json
  pinocchio --strategy planning --config configs/planning_template.json
  pinocchio --dry-run --strategy workflow  # Fast testing with mocks
  pinocchio --dry-run --strategy planning  # Test planning logic without LLM calls
        """,
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["development", "production"],
        help="CLI mode (development/production). Development mode raises errors, production mode allows fallback",
    )

    parser.add_argument(
        "--config",
        type=str,
        default="pinocchio.json",
        help="Path to configuration file (default: pinocchio.json)",
    )

    parser.add_argument(
        "--strategy",
        type=str,
        choices=["workflow", "planning"],
        help="Execution strategy (workflow=fixed workflow, planning=adaptive planning)",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Use mock LLM responses for fast testing (no real LLM calls)",
    )

    parser.add_argument(
        "--legacy-cli", action="store_true", help="Use legacy CLI interface"
    )
    parser.add_argument(
        "--version", action="store_true", help="Show version information"
    )

    # Parse arguments, but allow unknown arguments for backward compatibility
    args, unknown = parser.parse_known_args()

    # Handle --help without executing
    if "--help" in sys.argv or "-h" in sys.argv:
        parser.print_help()
        return

    # Handle strategy switching
    if args.strategy:
        strategy_config_path = Path(f"configs/{args.strategy}_template.json")
        if strategy_config_path.exists():
            try:
                shutil.copy2(strategy_config_path, "pinocchio.json")
                console = Console()
                console.print(f"✅ Switched to {args.strategy} strategy")
                console.print(f"📁 Configuration copied from: {strategy_config_path}")

                # Show strategy info
                show_strategy_info(console, args.strategy)

            except Exception as e:
                console = Console()
                console.print(f"❌ Failed to switch to {args.strategy} strategy: {e}")
                return
        else:
            console = Console()
            console.print(f"❌ Strategy template not found: {strategy_config_path}")
            console.print("Available strategies: workflow, planning")
            return

    # Handle mode switching (can combine with strategy)
    elif args.mode:
        config_path = Path(f"configs/{args.mode}.json")
        if config_path.exists():
            try:
                shutil.copy2(config_path, "pinocchio.json")
                console = Console()
                console.print(f"✅ Switched to {args.mode} mode")
                console.print(f"📁 Configuration copied from: {config_path}")

                # Show mode info
                show_mode_info(console, args.mode)

            except Exception as e:
                console = Console()
                console.print(f"❌ Failed to switch to {args.mode} mode: {e}")
                return
        else:
            console = Console()
            console.print(f"❌ Configuration file not found: {config_path}")
            console.print("Available modes: development, production")
            return

    # Set global config file path
    os.environ["PINOCCHIO_CONFIG_FILE"] = args.config

    # Check if legacy mode is requested
    use_legacy = (
        args.legacy_cli
        or "--legacy-cli" in sys.argv
        or os.getenv("PINOCCHIO_LEGACY") == "1"
    )

    if args.version:
        console = Console()
        console.print("[bold purple]Pinocchio CLI[/bold purple]")
        console.print("[dim]Multi-Agent Collaboration System[/dim]")
        console.print("Version: 1.0.0")
        return

    if use_legacy:
        # Use legacy CLI
        await legacy_main()
        return

    # Load configuration with custom config file
    try:
        config_manager = ConfigManager(args.config)
        config_manager.config
    except Exception as e:
        console = Console()
        console.print(f"[red]Error loading configuration from {args.config}: {e}[/red]")
        return

    try:
        verbose_config = config_manager.get_verbose_config()
        verbose = verbose_config.get("enabled", False)

        # Create CLI first for verbose callback
        cli = PinocchioCLI()

        # Initialize verbose logger with config
        from pinocchio.utils.verbose_logger import VerboseLogger, set_verbose_logger

        if verbose:
            verbose_logger = VerboseLogger(
                log_file=Path(verbose_config.get("log_file", "./logs/verbose.log")),
                max_depth=verbose_config.get("max_depth", 5),
                enable_colors=verbose_config.get("enable_colors", True),
            )
            set_verbose_logger(verbose_logger)

        # Choose LLM client based on dry-run flag
        if args.dry_run:
            # Use MockLLMClient for fast testing
            from pinocchio.llm.mock_client import MockLLMClient

            llm_client = MockLLMClient(response_delay_ms=50, failure_rate=0.0)
            console = Console()
            console.print(
                "[yellow]🧪 DRY-RUN MODE: Using mock LLM client for fast testing[/yellow]"
            )
        else:
            # Try to use real LLM client, fallback to mock if it fails
            llm_client = None
            try:
                from pinocchio.llm.custom_llm_client import CustomLLMClient

                # Get default LLM config (for generator agent)
                llm_config = config_manager.get_agent_llm_config("generator")
                llm_client = CustomLLMClient(
                    llm_config,
                    verbose=verbose,
                    verbose_callback=cli.add_llm_verbose_message,
                )
                console = Console()
                console.print(
                    f"[green]✅ Connected to LLM server: {llm_config.base_url}[/green]"
                )
            except Exception as e:
                # Don't fallback to MockLLMClient, instead report the error
                console = Console()
                console.print(f"[red]❌ Failed to initialize LLM client: {e}[/red]")
                console.print(
                    "[yellow]💡 Please check your configuration in pinocchio.json[/yellow]"
                )
                console.print(
                    "[yellow]💡 You can use --dry-run for testing without LLM[/yellow]"
                )
                sys.exit(1)

        # Ensure we have a valid LLM client
        if llm_client is None:
            from pinocchio.llm.mock_client import MockLLMClient

            llm_client = MockLLMClient(response_delay_ms=100, failure_rate=0.0)
            console = Console()
            console.print(
                "[yellow]⚠️ No LLM client available, using mock LLM client[/yellow]"
            )

        # Get mode from CLI args or config
        mode = args.mode or config_manager.config.cli.mode
        console.print(f"[blue]🔧 CLI Mode: {mode}[/blue]")

        # === New: Use with Session.create_session() to wrap main flow ===
        with Session.create_session(
            task_description="Pinocchio CLI Session"
        ) as session:
            coordinator = Coordinator(llm_client, mode=mode)
            cli.set_coordinator(coordinator)
            await cli.run()
            # === On normal main flow end, batch export all agent memory and prompt ===
            if hasattr(session, "agents"):
                for agent in session.agents.values():
                    if hasattr(agent, "save_memory_to_file"):
                        agent.save_memory_to_file(session.session_id)
                    if hasattr(agent, "export_prompt_history_to_file"):
                        agent.export_prompt_history_to_file(session.session_id)

    except Exception as e:
        console = Console()
        console.print(f"[red]Error creating LLM client: {e}[/red]")
        return


def show_strategy_info(console: Console, strategy: str):
    """Show information about the given strategy in the console."""
    console.print(f"📋 {strategy.upper()} STRATEGY:")

    if strategy == "workflow":
        console.print("   Fixed workflow execution with predefined agent sequences")
        console.print("   Features:")
        console.print("   ✅ Predictable execution order")
        console.print("   ✅ Fast execution time")
        console.print("   ✅ Consistent results")
        console.print("   ✅ Easy debugging and monitoring")
        console.print(
            "   ✅ Fixed pattern: Generator → Debugger → Evaluator → Optimizer"
        )

    elif strategy == "planning":
        console.print("   Adaptive task planning with intelligent analysis")
        console.print("   Features:")
        console.print("   ✅ Intelligent request analysis")
        console.print("   ✅ Dynamic task generation")
        console.print("   ✅ Context-aware planning")
        console.print("   ✅ Flexible agent coordination")
        console.print("   ✅ Multi-round optimization cycles")


def show_mode_info(console: Console, mode: str):
    """Show information about the given mode in the console."""
    mode_info = {
        "development": {
            "description": "Development mode - raises errors instead of fallback",
            "features": [
                "✅ Raises errors on LLM failures",
                "✅ Raises errors on missing LLM client",
                "✅ Raises errors on task planning failures",
                "✅ Full verbose logging enabled",
                "✅ Performance tracking",
                "✅ Session tracking",
                "✅ Detailed agent communications",
                "✅ LLM request/response logging",
            ],
        },
        "production": {
            "description": "Production mode - allows fallback for robustness",
            "features": [
                "✅ Allows fallback on LLM failures",
                "✅ Allows fallback on missing LLM client",
                "✅ Allows fallback on task planning failures",
                "✅ Minimal logging for end users",
                "✅ Basic progress updates only",
                "✅ Clean user experience",
            ],
        },
    }

    if mode in mode_info:
        info = mode_info[mode]
        console.print(f"\n📋 {mode.upper()} MODE:")
        console.print(f"   {info['description']}")
        console.print("\n   Features:")
        for feature in info["features"]:
            console.print(f"   {feature}")


# === Global emergency save hook ===
_emergency_saved = False


def emergency_save():
    """Emergency save function to save session, memories, and prompts."""
    global _emergency_saved
    if _emergency_saved:
        return
    _emergency_saved = True
    try:
        from pinocchio.memory.manager import MemoryManager
        from pinocchio.prompt.manager import PromptManager
        from pinocchio.session.context import get_current_session

        session = get_current_session()
        if session:
            # Try to call session_logger.complete_session()
            if hasattr(session, "session_logger") and session.session_logger:
                session.session_logger.complete_session("emergency_exit")
            # Compatible with direct session.complete_session()
            if hasattr(session, "complete_session"):
                session.complete_session()
            # Force flush memory
            try:
                mm = MemoryManager()
                agent_memories = mm.get_agent_memories(session.session_id)
                for m in agent_memories:
                    mm.store_agent_memory(m)
            except Exception as e:
                print(f"[Emergency Save] Memory flush failed: {e}")
            # Force flush prompt
            try:
                pm = PromptManager()
                # === New: Batch export all agent prompt_history and memory ===
                if hasattr(session, "agents"):
                    all_prompt_history = []
                    all_memory = []
                    for agent in session.agents.values():
                        # Export each agent's prompt_history
                        if hasattr(agent, "export_prompt_history_to_file"):
                            agent.export_prompt_history_to_file(session.session_id)
                        # Export each agent's memory
                        if hasattr(agent, "save_memory_to_file"):
                            agent.save_memory_to_file(session.session_id)
                        # Merge to summary table
                        if hasattr(agent, "prompt_history"):
                            all_prompt_history.extend(agent.prompt_history)
                        if hasattr(agent, "memory"):
                            all_memory.extend(agent.memory)
                    # Merge and export
                    import json
                    from pathlib import Path

                    Path("prompts").mkdir(parents=True, exist_ok=True)
                    with open(
                        f"prompts/prompt_history_{session.session_id}_all.json",
                        "w",
                        encoding="utf-8",
                    ) as f:
                        json.dump(all_prompt_history, f, ensure_ascii=False, indent=2)
                    with open(
                        f"prompts/agent_memory_{session.session_id}_all.json",
                        "w",
                        encoding="utf-8",
                    ) as f:
                        json.dump(all_memory, f, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"[Emergency Save] Prompt flush failed: {e}")
        print("[Emergency Save] Session, memories, and prompts have been saved.")
    except Exception as e:
        print(f"[Emergency Save] Failed: {e}")


atexit.register(emergency_save)


def signal_handler(sig, frame):
    """Signal handler to trigger emergency save on Ctrl+C or kill signal."""
    print(f"\n[Signal Handler] Signal {sig} received, triggering emergency save...")
    emergency_save()
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
signal.signal(signal.SIGTERM, signal_handler)  # kill

if __name__ == "__main__":
    asyncio.run(main())


def run():
    """Entry point for the CLI script"""
    asyncio.run(main())
    emergency_save()  # Ensure this line is properly indented under the function
