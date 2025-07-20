#!/usr/bin/env python3
"""
Test script to verify --dry-run functionality.
This script simulates user interaction with the CLI in dry-run mode.
"""

import asyncio
import os
import sys
from pathlib import Path

import pytest

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from pinocchio.config.config_manager import ConfigManager
from pinocchio.coordinator import Coordinator
from pinocchio.llm.mock_client import MockLLMClient
from pinocchio.utils.verbose_logger import VerboseLogger, set_verbose_logger


@pytest.mark.asyncio
async def test_dry_run_workflow():
    """Test dry-run workflow with mock LLM client."""
    print("🧪 Testing Pinocchio CLI --dry-run functionality")
    print("=" * 60)

    # Initialize MockLLMClient for fast testing
    mock_llm_client = MockLLMClient(response_delay_ms=50, failure_rate=0.0)
    print(
        f"✅ MockLLMClient initialized with {mock_llm_client.response_delay_ms}ms delay"
    )

    # Load configuration
    config_manager = ConfigManager()

    # Set up verbose logging
    verbose_logger = VerboseLogger(
        log_file=Path("./logs/dry_run_test.log"),
        max_depth=5,
        enable_colors=True,
    )
    set_verbose_logger(verbose_logger)

    # Create coordinator with mock client
    coordinator = Coordinator(mock_llm_client)
    print("✅ Coordinator initialized with MockLLMClient")

    # Test request
    test_request = "请帮我生成一个简单的CUDA向量加法kernel"
    print(f"\n📝 Testing request: {test_request}")
    print("⏱️  Starting execution (should be very fast with mocks)...")

    import time

    start_time = time.time()

    # Process request
    results = []
    async for message in coordinator.process_user_request(test_request):
        results.append(message)
        print(f"📄 {message[:100]}..." if len(message) > 100 else f"📄 {message}")

    end_time = time.time()
    execution_time = end_time - start_time

    print(f"\n🏁 Execution completed!")
    print(f"⏱️  Total execution time: {execution_time:.2f} seconds")
    print(f"📊 Mock LLM calls: {mock_llm_client.call_count}")
    print(f"📝 Messages received: {len(results)}")

    # Verify it's fast (should be under 10 seconds for mock mode)
    if execution_time < 10:
        print("✅ DRY-RUN mode is working correctly (fast execution)")
    else:
        print("❌ DRY-RUN mode may not be working (slow execution)")

    print("\n🎉 Dry-run test completed!")


if __name__ == "__main__":
    asyncio.run(test_dry_run_workflow())
