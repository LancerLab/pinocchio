#!/usr/bin/env python3
"""
Pinocchio CLI - Multi-Agent Collaboration System
"""

import asyncio

from rich.console import Console
from rich.prompt import Prompt

from pinocchio.config.config_manager import ConfigManager
from pinocchio.coordinator import Coordinator


def print_simple_logo():
    """Print simple colored PINOCCHIO logo"""
    console = Console()

    # Color gradient list (purple to deep green, 9 letters)
    colors = [
        "#8B5CF6",  # Purple
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

    # Center display
    console.print("\n" + styled.center(80) + "\n")
    console.print("🎭 Pinocchio CLI - Multi-Agent Collaboration System", style="bold")
    console.print("Type your request and press Enter to start...")
    console.print("Type '/help' for available commands")
    console.print("Type '/quit' to exit\n")


def print_block_logo():
    """Print large block font PINOCCHIO logo"""
    console = Console()

    try:
        from pyfiglet import Figlet

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
        f = Figlet(font="block")
        ascii_art = f.renderText(text).splitlines()

        for i, line in enumerate(ascii_art):
            if line.strip():  # Only color non-empty lines
                color = colors[i % len(colors)]
                console.print(f"[bold {color}]{line}[/]")

        console.print(
            "\n🎭 Pinocchio CLI - Multi-Agent Collaboration System", style="bold"
        )
        console.print("Type your request and press Enter to start...")
        console.print("Type '/help' for available commands")
        console.print("Type '/quit' to exit\n")

    except ImportError:
        console.print("[red]pyfiglet not installed. Falling back to simple logo.[/red]")
        print_simple_logo()


def print_logo():
    """Print the Pinocchio CLI logo with style selection"""
    console = Console()

    # Check if user has a preferred logo style
    logo_style = console.input(
        "[cyan]Choose logo style (1=simple, 2=block, Enter=default): [/cyan]"
    ).strip()

    if logo_style == "2":
        print_block_logo()
    else:
        print_simple_logo()


def print_help():
    """Print help information"""
    help_text = """
    📋 Available Commands:

    /help     - Show this help message
    /quit     - Exit the CLI
    /clear    - Clear the screen
    /status   - Show current session status
    /history  - Show recent session history

    💡 Usage:
    Simply type your request and press Enter to start the multi-agent workflow.
    Example: "write a matrix multiplication operator"
    """
    print(help_text)


async def main():
    """Main CLI entry point"""
    console = Console()

    # Load configuration
    try:
        config_manager = ConfigManager()
        config = config_manager.config  # Use .config instead of .get_config()
    except Exception as e:
        console.print(f"[red]Error loading configuration: {e}[/red]")
        return

    # Create coordinator
    coordinator = Coordinator(config)

    # Print logo
    print_logo()

    # Main CLI loop
    while True:
        try:
            # Get user input
            user_input = Prompt.ask("\n> ")

            # Handle commands
            if user_input.startswith("/"):
                command = user_input.lower().strip()

                if command == "/help":
                    print_help()
                    continue
                elif command == "/quit":
                    console.print("[yellow]Exiting Pinocchio CLI...[/yellow]")
                    break
                elif command == "/clear":
                    console.clear()
                    print_logo()
                    continue
                elif command == "/status":
                    # TODO: Implement status command
                    console.print("[blue]Status: Ready[/blue]")
                    continue
                elif command == "/history":
                    # TODO: Implement history command
                    console.print("[blue]No recent sessions[/blue]")
                    continue
                else:
                    console.print(f"[red]Unknown command: {command}[/red]")
                    console.print("Type '/help' for available commands")
                    continue

            # Process user request
            if user_input.strip():
                console.print(f"\n[blue]Processing: {user_input}[/blue]\n")

                async for message in coordinator.process_user_request(user_input):
                    console.print(message)

                console.print("\n[green]✅ Request completed![/green]\n")

        except KeyboardInterrupt:
            console.print("\n[yellow]Use '/quit' to exit[/yellow]")
        except Exception as e:
            console.print(f"\n[red]Error: {e}[/red]")


if __name__ == "__main__":
    asyncio.run(main())


def run():
    """Entry point for the CLI script"""
    asyncio.run(main())
