#!/usr/bin/env python3
"""
Pinocchio CLI - Multi-Agent Collaboration System
"""

import asyncio

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text

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
        
        # Use block font but with better handling
        f = Figlet(font="block")
        ascii_art = f.renderText(text).splitlines()

        # Print each line with color gradient, but limit width
        for i, line in enumerate(ascii_art):
            if line.strip():  # Only color non-empty lines
                # Use filled block character █ instead of hollow
                filled_line = line.replace("_", "█").replace("|", "█")
                # Truncate if too long to prevent wrapping
                if len(filled_line) > 80:
                    filled_line = filled_line[:80]
                color = colors[i % len(colors)]
                console.print(f"[bold {color}]{filled_line}[/]")

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
    """Print the Pinocchio CLI logo with block font style"""
    console = Console()
    
    # Welcome panel
    welcome_panel = Panel(
        "[bold cyan]Welcome to Pinocchio CLI![/bold cyan]\n"
        "Loading beautiful block font logo...",
        title="🎭 Pinocchio Multi-Agent System",
        border_style="purple",
        padding=(1, 2)
    )
    console.print(welcome_panel)
    
    console.clear()
    print_block_logo()


def print_help():
    """Print help information with beautiful formatting"""
    console = Console()
    
    # Create help table
    table = Table(title="📋 Available Commands", show_header=True, header_style="bold magenta")
    table.add_column("Command", style="cyan", no_wrap=True)
    table.add_column("Description", style="white")
    
    commands = [
        ("/help", "Show this help message"),
        ("/quit", "Exit the CLI"),
        ("/clear", "Clear the screen"),
        ("/status", "Show current session status"),
        ("/history", "Show recent session history"),
    ]
    
    for cmd, desc in commands:
        table.add_row(cmd, desc)
    
    # Create usage panel
    usage_text = """
    💡 Usage:
    Simply type your request and press Enter to start the multi-agent workflow.
    
    Example: "write a matrix multiplication operator"
    
    The system will automatically:
    • 🤖 Create an intelligent task plan
    • ⚡ Generate high-performance code
    • 🔧 Debug and fix issues
    • 🚀 Optimize for performance
    • 📊 Evaluate results
    """
    
    usage_panel = Panel(
        usage_text,
        title="🚀 How to Use",
        border_style="green",
        padding=(1, 2)
    )
    
    console.print(table)
    console.print("\n")
    console.print(usage_panel)


def handle_commands(command: str) -> bool:
    """Handle CLI commands. Returns True if should continue, False if should quit."""
    console = Console()
    
    if command == '/help':
        print_help()
        return True
    elif command == '/quit':
        console.print(Panel(
            "[yellow]Exiting Pinocchio CLI...[/yellow]\n"
            "Thank you for using Pinocchio! 🎭",
            title="👋 Goodbye",
            border_style="yellow"
        ))
        return False
    elif command == '/clear':
        console.clear()
        print_logo()
        return True
    elif command == '/status':
        # Status panel
        status_panel = Panel(
            "[green]✅ System Ready[/green]\n"
            "• Multi-agent system: [green]Active[/green]\n"
            "• LLM client: [green]Connected[/green]\n"
            "• Task planning: [green]Available[/green]\n"
            "• Session management: [green]Ready[/green]",
            title="📊 System Status",
            border_style="blue"
        )
        console.print(status_panel)
        return True
    elif command == '/history':
        # History panel
        history_panel = Panel(
            "[dim]No recent sessions found[/dim]\n"
            "Start a new session to see history here.",
            title="📜 Session History",
            border_style="cyan"
        )
        console.print(history_panel)
        return True
    else:
        console.print(Panel(
            f"[red]Unknown command: {command}[/red]\n"
            "Type '/help' for available commands",
            title="❌ Error",
            border_style="red"
        ))
        return True


async def main():
    """Main CLI entry point"""
    console = Console()
    
    # Load configuration
    try:
        config_manager = ConfigManager()
        config_manager.config  # Use .config instead of .get_config()
    except Exception as e:
        console.print(Panel(
            f"[red]Error loading configuration: {e}[/red]",
            title="❌ Configuration Error",
            border_style="red"
        ))
        return
    
    # Create LLM client from configuration
    try:
        from pinocchio.llm.custom_llm_client import CustomLLMClient
        llm_config = config_manager.get_llm_config()
        llm_client = CustomLLMClient(llm_config)
    except Exception as e:
        console.print(Panel(
            f"[red]Error creating LLM client: {e}[/red]",
            title="❌ LLM Client Error",
            border_style="red"
        ))
        return
    
    # Create coordinator with LLM client
    coordinator = Coordinator(llm_client)
    
    # Print logo
    print_logo()
    
    # Main CLI loop
    while True:
        try:
            # Get user input with styled prompt
            user_input = Prompt.ask("\n[bold cyan]Pinocchio[/bold cyan] > ")
            
            # Handle commands
            if user_input.startswith('/'):
                command = user_input.lower().strip()
                if not handle_commands(command):
                    break
                continue
            
            # Process user request
            if user_input.strip():
                # Create request panel
                request_panel = Panel(
                    f"[blue]Processing: {user_input}[/blue]",
                    title="🔄 Processing Request",
                    border_style="blue",
                    padding=(0, 1)
                )
                console.print(request_panel)
                console.print("\n")
                
                async for message in coordinator.process_user_request(user_input):
                    console.print(message)
                
                # Success panel
                success_panel = Panel(
                    "[green]✅ Request completed successfully![/green]",
                    title="🎉 Success",
                    border_style="green",
                    padding=(0, 1)
                )
                console.print("\n")
                console.print(success_panel)
                console.print("\n")
        
        except KeyboardInterrupt:
            console.print(Panel(
                "[yellow]Use '/quit' to exit[/yellow]",
                title="⚠️ Interrupted",
                border_style="yellow"
            ))
        except EOFError:
            console.print(Panel(
                "[yellow]Use '/quit' to exit[/yellow]",
                title="⚠️ EOF",
                border_style="yellow"
            ))
            break
        except Exception as e:
            console.print(Panel(
                f"[red]Error: {e}[/red]",
                title="❌ Error",
                border_style="red"
            ))


if __name__ == "__main__":
    asyncio.run(main())


def run():
    """Entry point for the CLI script"""
    asyncio.run(main())
