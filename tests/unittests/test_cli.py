#!/usr/bin/env python3
"""Simple test script for Pinocchio CLI.

This script tests the basic functionality of the CLI.
"""

import asyncio
import os
import sys

import pytest

pytestmark = pytest.mark.asyncio

from pinocchio.cli.main import cli

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


async def test_cli():
    """Test the CLI functionality."""
    print("Testing Pinocchio CLI...")

    # Test welcome screen
    print("\n1. Testing welcome screen:")
    cli._show_welcome()

    # Test help command
    print("\n2. Testing help command:")
    cli._show_help()

    # Test message display
    print("\n3. Testing message display:")
    cli._show_message("user", "Write a conv2d operator")
    cli._show_message("system", "Analyzing your request...")
    cli._show_message("success", "Code generation complete!")

    # Test history
    print("\n4. Testing history:")
    from datetime import datetime

    cli.history = [
        {
            "type": "user",
            "content": "Write a conv2d operator",
            "timestamp": datetime.utcnow(),
        },
        {
            "type": "system",
            "content": "Analyzing your request...",
            "timestamp": datetime.utcnow(),
        },
        {
            "type": "system",
            "content": "Generating code...",
            "timestamp": datetime.utcnow(),
        },
        {
            "type": "system",
            "content": "Code generation complete!",
            "timestamp": datetime.utcnow(),
        },
    ]
    cli._show_history()

    print("\n✅ CLI test completed successfully!")


if __name__ == "__main__":
    asyncio.run(test_cli())
