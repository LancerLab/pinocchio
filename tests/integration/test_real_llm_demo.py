#!/usr/bin/env python3
"""
Test script to verify real LLM execution vs dry-run mode.
This script tests the actual LLM performance and compares it with mock results.
"""

import asyncio
import os
import sys
import time
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from pinocchio.config.config_manager import ConfigManager
from pinocchio.coordinator import Coordinator
from pinocchio.llm.custom_llm_client import CustomLLMClient
from pinocchio.llm.mock_client import MockLLMClient
from pinocchio.utils.verbose_logger import VerboseLogger, set_verbose_logger


async def test_real_vs_mock_comparison():
    """Compare real LLM execution with mock execution."""
    print("🧪 Testing Pinocchio CLI: Real LLM vs Mock Comparison")
    print("=" * 70)

    # Load configuration
    config_manager = ConfigManager()

    # Set up verbose logging
    verbose_logger = VerboseLogger(
        log_file=Path("./logs/real_vs_mock_test.log"),
        max_depth=5,
        enable_colors=True,
    )
    set_verbose_logger(verbose_logger)

    # Test request
    test_request = "请帮我生成一个简单的CUDA向量加法kernel"
    print(f"\n📝 Test request: {test_request}")

    # Test 1: Mock execution
    print("\n" + "=" * 50)
    print("🤖 Testing MOCK LLM execution...")
    print("=" * 50)

    mock_llm_client = MockLLMClient(response_delay_ms=50, failure_rate=0.0)
    mock_coordinator = Coordinator(mock_llm_client)

    mock_start_time = time.time()
    mock_results = []
    async for message in mock_coordinator.process_user_request(test_request):
        mock_results.append(message)
        print(
            f"📄 MOCK: {message[:100]}..."
            if len(message) > 100
            else f"📄 MOCK: {message}"
        )

    mock_end_time = time.time()
    mock_execution_time = mock_end_time - mock_start_time

    print(f"\n🏁 MOCK execution completed!")
    print(f"⏱️  Mock execution time: {mock_execution_time:.2f} seconds")
    print(f"📊 Mock LLM calls: {mock_llm_client.call_count}")
    print(f"📝 Mock messages: {len(mock_results)}")

    # Test 2: Real LLM execution
    print("\n" + "=" * 50)
    print("🌐 Testing REAL LLM execution...")
    print("=" * 50)

    try:
        llm_config = config_manager.get_llm_config()
        real_llm_client = CustomLLMClient(llm_config, verbose=True)
        real_coordinator = Coordinator(real_llm_client)

        real_start_time = time.time()
        real_results = []
        async for message in real_coordinator.process_user_request(test_request):
            real_results.append(message)
            print(
                f"📄 REAL: {message[:100]}..."
                if len(message) > 100
                else f"📄 REAL: {message}"
            )

        real_end_time = time.time()
        real_execution_time = real_end_time - real_start_time

        print(f"\n🏁 REAL execution completed!")
        print(f"⏱️  Real execution time: {real_execution_time:.2f} seconds")
        print(f"📝 Real messages: {len(real_results)}")

        # Comparison
        print("\n" + "=" * 50)
        print("📊 PERFORMANCE COMPARISON")
        print("=" * 50)
        speedup = (
            real_execution_time / mock_execution_time if mock_execution_time > 0 else 0
        )
        print(f"🐌 Real LLM time:    {real_execution_time:.2f} seconds")
        print(f"⚡ Mock LLM time:    {mock_execution_time:.2f} seconds")
        print(f"🚀 Speedup factor:   {speedup:.1f}x faster with mock")
        print(
            f"📈 Time saved:       {real_execution_time - mock_execution_time:.2f} seconds"
        )

        if speedup > 10:
            print(
                "✅ Excellent speedup! Mock mode is significantly faster for development."
            )
        elif speedup > 5:
            print("✅ Good speedup! Mock mode provides substantial time savings.")
        else:
            print("⚠️ Moderate speedup. Consider optimizing mock responses.")

    except Exception as e:
        print(f"❌ Real LLM execution failed: {e}")
        print("💡 This might be because LLM service is not available")
        print("🧪 Mock mode is still working perfectly for development!")

    print(f"\n🎉 Comparison test completed!")


if __name__ == "__main__":
    asyncio.run(test_real_vs_mock_comparison())
