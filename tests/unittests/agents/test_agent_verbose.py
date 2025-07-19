#!/usr/bin/env python3
"""Test script for agent verbose functionality."""

import asyncio
import json
from pathlib import Path

import pytest

from pinocchio.agents.debugger import DebuggerAgent
from pinocchio.agents.evaluator import EvaluatorAgent
from pinocchio.agents.generator import GeneratorAgent
from pinocchio.agents.optimizer import OptimizerAgent
from pinocchio.config import ConfigManager
from pinocchio.utils.verbose_logger import VerboseLogger, set_verbose_logger


@pytest.mark.asyncio
async def test_agent_verbose():
    """Test verbose functionality for all agents."""
    print("🧪 Testing Agent Verbose Functionality")
    print("=" * 50)

    # Initialize verbose logger
    verbose_logger = VerboseLogger(
        log_file=Path("./logs/agent_test.log"),
        max_depth=5,
        enable_colors=True,
    )
    set_verbose_logger(verbose_logger)

    # Test data
    test_code = """
func matrix_multiply(a: tensor, b: tensor, c: tensor) {
    for i in range(a.shape[0]) {
        for j in range(b.shape[1]) {
            for k in range(a.shape[1]) {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }
}
"""

    # Test Generator Agent
    print("\n🔧 Testing Generator Agent...")
    try:
        generator = GeneratorAgent()

        # Test simple code generation
        result = generator.generate_simple_code(
            "Generate a matrix multiplication function"
        )
        print(f"   ✅ Generator test completed - Output keys: {list(result.keys())}")

    except Exception as e:
        print(f"   ❌ Generator test failed: {e}")

    # Test Optimizer Agent
    print("\n⚡ Testing Optimizer Agent...")
    try:
        optimizer = OptimizerAgent()

        # Test code optimization
        result = optimizer.analyze_code_performance(
            test_code, ["performance", "memory"]
        )
        print(
            f"   ✅ Optimizer test completed - Suggestions: {len(result.get('optimization_suggestions', []))}"
        )

    except Exception as e:
        print(f"   ❌ Optimizer test failed: {e}")

    # Test Evaluator Agent
    print("\n📊 Testing Evaluator Agent...")
    try:
        evaluator = EvaluatorAgent()

        # Test performance evaluation
        result = evaluator.evaluate_performance(test_code)
        print(
            f"   ✅ Evaluator test completed - Bottlenecks: {len(result.get('bottlenecks', []))}"
        )

    except Exception as e:
        print(f"   ❌ Evaluator test failed: {e}")

    # Test Debugger Agent
    print("\n🐛 Testing Debugger Agent...")
    try:
        debugger = DebuggerAgent()

        # Test code debugging
        result = debugger.analyze_code_issues(
            test_code, "Potential memory access issue"
        )
        print(
            f"   ✅ Debugger test completed - Issues: {len(result.get('issues_found', []))}"
        )

    except Exception as e:
        print(f"   ❌ Debugger test failed: {e}")

    print("\n🎉 Agent verbose functionality test completed!")


def test_verbose_logging():
    """Test verbose logging functionality."""
    print("\n📝 Testing Verbose Logging")
    print("=" * 30)

    try:
        # Initialize verbose logger
        verbose_logger = VerboseLogger(
            log_file=Path("./logs/verbose_test.log"),
            max_depth=3,
            enable_colors=True,
        )
        set_verbose_logger(verbose_logger)

        # Test different log levels
        verbose_logger.log_agent_activity(
            "test_agent",
            "Test activity",
            data={"test": True, "message": "Hello World"},
            session_id="test_session",
            step_id="test_step",
        )

        verbose_logger.log_llm_activity(
            "Test LLM call",
            request_data={"prompt_length": 100},
            response_data={"response_length": 200},
            session_id="test_session",
            duration_ms=150,
        )

        verbose_logger.log(
            "INFO",
            "test_component",
            "Test message",
            data={"test": True},
            session_id="test_session",
        )

        print("   ✅ Verbose logging test completed")

    except Exception as e:
        print(f"   ❌ Verbose logging test failed: {e}")


def test_config_integration():
    """Test configuration integration with agents."""
    print("\n⚙️ Testing Configuration Integration")
    print("=" * 35)

    try:
        # Test with different config files
        config_files = [
            "configs/development.json",
            "configs/production.json",
            "configs/debug.json",
        ]

        for config_file in config_files:
            if not Path(config_file).exists():
                continue

            print(f"\n   📋 Testing {config_file}...")

            config_manager = ConfigManager(config_file)
            verbose_enabled = config_manager.is_verbose_enabled()
            verbose_mode = config_manager.get_verbose_mode()

            print(f"      Verbose enabled: {verbose_enabled}")
            print(f"      Verbose mode: {verbose_mode}")

        print("   ✅ Configuration integration test completed")

    except Exception as e:
        print(f"   ❌ Configuration integration test failed: {e}")


if __name__ == "__main__":
    # Create logs directory if it doesn't exist
    Path("./logs").mkdir(exist_ok=True)

    # Run tests
    asyncio.run(test_agent_verbose())
    test_verbose_logging()
    test_config_integration()

    print("\n🎉 All agent verbose tests completed!")
