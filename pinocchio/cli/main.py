"""Main CLI logic for Pinocchio."""

import asyncio
import sys
from datetime import datetime
from typing import AsyncGenerator

from prompt_toolkit import PromptSession
from prompt_toolkit.styles import Style
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# Color themes configuration
THEMES = {
    "default": {
        "primary": "#6366f1",  # Indigo
        "secondary": "#8b5cf6",  # Violet
        "success": "#10b981",  # Emerald
        "warning": "#f59e0b",  # Amber
        "error": "#ef4444",  # Red
        "info": "#3b82f6",  # Blue
        "background": "#0f172a",  # Slate 900
        "foreground": "#f8fafc",  # Slate 50
    },
    "light": {
        "primary": "#6366f1",
        "secondary": "#8b5cf6",
        "success": "#059669",
        "warning": "#d97706",
        "error": "#dc2626",
        "info": "#2563eb",
        "background": "#ffffff",
        "foreground": "#1e293b",
    },
}

# Message types and formatting
MESSAGE_TYPES = {
    "user": {"prefix": "👤 You", "color": "blue", "style": "bold"},
    "system": {"prefix": "🤖 Pinocchio", "color": "violet", "style": "italic"},
    "progress": {"prefix": "⏳", "color": "blue", "style": "dim"},
    "success": {"prefix": "✅", "color": "green", "style": "bold"},
    "error": {"prefix": "❌", "color": "red", "style": "bold"},
}


class PinocchioCLI:
    """Main class for Pinocchio CLI."""

    def __init__(self) -> None:
        """Initialize PinocchioCLI."""
        self.console: Console = Console()
        self.session: PromptSession = PromptSession()
        self.theme: str = "default"
        self.history: list = []
        self.is_running: bool = True

        # Initialize Coordinator (placeholder for now)
        # from pinocchio.coordinator import Coordinator
        # self.coordinator = Coordinator()

        # For now, we'll use a mock coordinator
        self.coordinator = MockCoordinator()

    async def start(self) -> None:
        """Start the CLI."""
        self._show_welcome()

        while self.is_running:
            try:
                # Get user input
                user_input = await self._get_user_input()

                if user_input.lower() in ["quit", "exit", "q"]:
                    await self._handle_quit()
                    break
                elif user_input.lower() == "help":
                    self._show_help()
                    continue
                elif user_input.lower() == "clear":
                    self.console.clear()
                    continue
                elif user_input.lower() == "history":
                    self._show_history()
                    continue

                # Process user request
                await self._process_request(user_input)

            except KeyboardInterrupt:
                self.console.print("\n[yellow]Use 'quit' to exit[/yellow]")
            except Exception as e:
                self.console.print(f"[red]Error: {str(e)}[/red]")

    def _show_welcome(self) -> None:
        """Show welcome screen."""
        welcome_text = Text()
        welcome_text.append("🎭 ", style="bold blue")
        welcome_text.append("Pinocchio CLI", style="bold white")
        welcome_text.append("\n\n", style="white")
        welcome_text.append(
            "Welcome to Pinocchio - Multi-Agent Collaboration System", style="white"
        )
        welcome_text.append("\n\n", style="white")
        welcome_text.append(
            "Type your request and press Enter to start...", style="dim white"
        )
        welcome_text.append("\n", style="white")
        welcome_text.append("Type 'help' for available commands", style="dim white")
        welcome_text.append("\n", style="white")
        welcome_text.append("Type 'quit' to exit", style="dim white")

        panel = Panel(welcome_text, border_style="blue", padding=(1, 2))

        self.console.print(panel)
        self.console.print()

    async def _get_user_input(self) -> str:
        """Get user input from prompt."""
        result = await self.session.prompt_async(
            "> ",
            style=Style.from_dict(
                {
                    "prompt": "bold blue",
                }
            ),
        )
        return str(result) if result else ""

    async def _process_request(self, user_input: str) -> None:
        """Process user request and display response."""
        # Record to history
        self.history.append(
            {"type": "user", "content": user_input, "timestamp": datetime.utcnow()}
        )

        # Show user input
        self._show_message("user", user_input)

        # Show progress indicator
        with Live(
            self._create_progress_panel("Processing request..."), console=self.console
        ):
            # Process request
            async for message in self.coordinator.process_user_request(user_input):
                # Show system message
                self._show_message("system", message)

        # Record system response
        self.history.append(
            {"type": "system", "content": message, "timestamp": datetime.utcnow()}
        )

    def _show_message(self, message_type: str, content: str) -> None:
        """Display a message in the console."""
        config = MESSAGE_TYPES.get(message_type, MESSAGE_TYPES["system"])

        text = Text()
        text.append(f"{config['prefix']}: ", style=f"bold {config['color']}")
        text.append(content, style=config["style"])

        self.console.print(text)
        self.console.print()

    def _create_progress_panel(self, message: str) -> Panel:
        """Create a progress panel for the UI."""
        text = Text()
        text.append("⏳ ", style="bold yellow")
        text.append(message, style="white")

        return Panel(text, border_style="yellow")

    def _show_help(self) -> None:
        """Display help information."""
        help_text = Text()
        help_text.append("Available Commands:\n\n", style="bold white")

        commands = [
            ("help", "Show this help message"),
            ("history", "Show conversation history"),
            ("clear", "Clear the screen"),
            ("quit", "Exit the application"),
            ("memory", "Show memory information"),
            ("sessions", "List recent sessions"),
        ]

        for cmd, desc in commands:
            help_text.append(f"  {cmd:<10}", style="bold blue")
            help_text.append(f"{desc}\n", style="white")

        panel = Panel(help_text, border_style="blue", title="Help")
        self.console.print(panel)
        self.console.print()

    def _show_history(self) -> None:
        """Display conversation history."""
        if not self.history:
            self.console.print("[dim]No history yet.[/dim]")
            return
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Type", style="dim", width=10)
        table.add_column("Content", style="white")
        table.add_column("Timestamp", style="dim", width=24)
        for entry in self.history:
            table.add_row(
                entry["type"],
                entry["content"],
                str(entry["timestamp"]),
            )
        self.console.print(table)
        self.console.print()

    async def _handle_quit(self) -> None:
        """Handle quit command."""
        self.console.print("[yellow]Exiting Pinocchio CLI...[/yellow]")
        self.is_running = False
        sys.exit(0)


class MockCoordinator:
    """Mock coordinator for testing"""

    async def process_user_request(self, user_input: str) -> AsyncGenerator[str, None]:
        """Mock process user request"""
        # Simulate processing steps
        yield "Analyzing your request..."
        await asyncio.sleep(0.5)

        yield "Generating code..."
        await asyncio.sleep(0.5)

        yield "Code generation complete!"
        await asyncio.sleep(0.2)


async def main() -> None:
    """Main entry point"""
    cli = PinocchioCLI()
    await cli.start()


def run() -> None:
    """Run the CLI"""
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nGoodbye! 👋")
        sys.exit(0)


if __name__ == "__main__":
    run()
