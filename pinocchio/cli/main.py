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
            "#8B5CF6",  # Purple
            "#7C3AED",  # Purple
            "#6D28D9",  # Purple
            "#4C1D95",  # Purple
            "#047857",  # Green
            "#059669",  # Green
            "#10B981",  # Green
            "#34D399",  # Green
            "#6EE7B7",  # Green
        ]
        text = "PINOCCHIO"
        f = Figlet(font="block")
        ascii_art = f.renderText(text).splitlines()

        # Find the maximum width of the logo
        max_width = max(len(line) for line in ascii_art if line.strip())

        # Print each line with left-to-right color gradient across the entire logo
        for line in ascii_art:
            if line.strip():  # Only color non-empty lines
                # Use filled block character █ instead of hollow
                filled_line = line.replace("_", "█").replace("|", "█")
                
                # Apply color gradient from left to right across the entire line
                colored_line = ""
                for i, char in enumerate(filled_line):
                    if char == "█":
                        # Calculate color index based on position across the entire logo width
                        color_index = int((i / max_width) * (len(colors) - 1))
                        color = colors[color_index]
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


def print_logo():
    """Print the Pinocchio CLI logo with block style"""
    print_block_logo()


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


def handle_commands(command: str) -> bool:
    """Handle CLI commands. Returns True if should continue, False if should quit."""
    console = Console()

    if command == "/help":
        print_help()
        return True
    elif command == "/quit":
        console.print("[yellow]Exiting Pinocchio CLI...[/yellow]")
        return False
    elif command == "/clear":
        console.clear()
        print_logo()
        return True
    elif command == "/status":
        # TODO: Implement status command
        console.print("[blue]Status: Ready[/blue]")
        return True
    elif command == "/history":
        # TODO: Implement history command
        console.print("[blue]No recent sessions[/blue]")
        return True
    else:
        console.print(f"[red]Unknown command: {command}[/red]")
        console.print("Type '/help' for available commands")
        return True


async def main():
    """Main CLI entry point"""
    console = Console()

    # Load configuration
    try:
        config_manager = ConfigManager()
        config_manager.config  # Use .config instead of .get_config()
    except Exception as e:
        console.print(f"[red]Error loading configuration: {e}[/red]")
        return

    # Create LLM client from configuration
    try:
        from pinocchio.llm.custom_llm_client import CustomLLMClient

        llm_config = config_manager.get_llm_config()
        llm_client = CustomLLMClient(llm_config)
    except Exception as e:
        console.print(f"[red]Error creating LLM client: {e}[/red]")
        return

    # Create coordinator with LLM client
    coordinator = Coordinator(llm_client)

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
                if not handle_commands(command):
                    break
                continue

            # Process user request
            if user_input.strip():
                console.print(f"\n[blue]Processing: {user_input}[/blue]\n")

                async for message in coordinator.process_user_request(user_input):
                    console.print(message)

                console.print("\n[green]✅ Request completed![/green]\n")

        except KeyboardInterrupt:
            console.print("\n[yellow]Use '/quit' to exit[/yellow]")
        except EOFError:
            console.print("\n[yellow]Use '/quit' to exit[/yellow]")
            break
        except Exception as e:
            console.print(f"\n[red]Error: {e}[/red]")


if __name__ == "__main__":
    asyncio.run(main())


def run():
    """Entry point for the CLI script"""
    asyncio.run(main())
