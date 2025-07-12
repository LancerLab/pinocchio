#!/usr/bin/env python3
"""
Demo script to showcase the task planning system
"""

import os
import sys

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # noqa: E402

import asyncio

from pinocchio.config.config_manager import ConfigManager
from pinocchio.coordinator import Coordinator


async def demo_task_planning():
    """Demo the task planning system"""
    # Load configuration
    config_manager = ConfigManager()
    config = config_manager.get_config()

    # Create coordinator
    coordinator = Coordinator(config)

    print("🎭 Pinocchio Task Planning Demo")
    print("=" * 50)
    print()

    # Demo request
    user_request = "write a matmul for me"

    print(f"👤 User Request: {user_request}")
    print()

    # Process the request
    async for message in coordinator.process_user_request(user_request):
        print(message)

    print()
    print("🎉 Demo completed!")


if __name__ == "__main__":
    asyncio.run(demo_task_planning())
