#!/usr/bin/env python3
"""
Test script to demonstrate both PINOCCHIO logo effects
"""

import os
import sys

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # noqa: E402

from rich.console import Console


def print_simple_logo():
    """Print simple colored PINOCCHIO logo"""
    console = Console()
    console.print("\n" + "=" * 60)
    console.print("EFFECT 1: Simple Colored Text")
    console.print("=" * 60)

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
    console.print("\n" + "=" * 60)
    console.print("EFFECT 2: Large Block Font (requires pyfiglet)")
    console.print("=" * 60)

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

        console.print("\n🎭 Pinocchio CLI - Multi-Agent Collaboration System", style="bold")
        console.print("Type your request and press Enter to start...")
        console.print("Type '/help' for available commands")
        console.print("Type '/quit' to exit\n")

    except ImportError:
        console.print("[red]pyfiglet not installed. Install with: pip install pyfiglet[/red]")
        console.print("[yellow]Showing fallback simple effect instead:[/yellow]")
        print_simple_logo()


if __name__ == "__main__":
    print_simple_logo()
    print_block_logo()
