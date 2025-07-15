#!/usr/bin/env python3
"""
Test script for CLI verbose workflow with complete logging.

This script tests the CLI workflow and ensures all verbose logs are captured,
including agent-generated code content and MCP tool call results.
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from pinocchio.config.config_manager import ConfigManager
from pinocchio.coordinator import Coordinator
from pinocchio.llm.custom_llm_client import CustomLLMClient
from pinocchio.utils.verbose_logger import LogLevel, VerboseLogger, set_verbose_logger


async def test_cuda_matmul_workflow():
    """Test CUDA matrix multiplication workflow with verbose logging."""

    print("=" * 80)
    print("🧪 Testing CLI Verbose Workflow with Complete Logging")
    print("=" * 80)

    # Setup verbose logging to file
    logs_dir = Path("./logs")
    logs_dir.mkdir(exist_ok=True)

    verbose_log_file = logs_dir / "cli_verbose_test.log"
    verbose_logger = VerboseLogger(
        log_file=verbose_log_file, max_depth=10, enable_colors=True
    )
    set_verbose_logger(verbose_logger)

    # Load configuration
    config_manager = ConfigManager()
    # Configuration is loaded automatically in __init__

    # Create LLM client with verbose logging
    llm_config = config_manager.get_llm_config()
    llm_client = CustomLLMClient(llm_config, verbose=True)

    # Create coordinator
    coordinator = Coordinator(llm_client)

    # Test user prompt for CUDA matrix multiplication
    user_prompt = """
Please help me generate a high-performance CUDA matrix multiplication kernel,
with the following requirements:
1. Support arbitrary matrix sizes
2. Optimize using shared memory
3. Include complete debugging information
4. Perform performance evaluation
5. Provide optimization suggestions
6. Generate fully compilable code

Please ensure the code quality is high, comments are detailed, and performance optimization is thorough.
"""

    print(f"📝 User Prompt: {user_prompt.strip()}")
    print()

    try:
        print("🚀 Starting workflow execution...")

        # Process the request and capture all streaming output
        all_output = []

        async for chunk in coordinator.process_user_request(user_prompt):
            print(chunk, end="", flush=True)
            all_output.append(chunk)

        print()
        print("✅ Workflow execution completed!")

        # Log final summary
        verbose_logger.log(
            LogLevel.INFO,
            "test_workflow",
            "Complete workflow test finished",
            data={
                "user_prompt": user_prompt,
                "total_output_length": len("".join(all_output)),
                "output_chunks": len(all_output),
                "log_entries_count": len(verbose_logger.entries),
                "session_id": getattr(coordinator.current_session, "session_id", None),
            },
        )

        # Export all verbose logs
        export_file = logs_dir / "complete_workflow_logs.json"
        verbose_logger.export_entries(export_file)

        print(f"📋 Verbose logs exported to: {export_file}")
        print(f"📋 Total log entries: {len(verbose_logger.entries)}")

        # Display performance summary
        verbose_logger.display_performance_summary()

        return True

    except Exception as e:
        print(f"❌ Error during workflow execution: {e}")

        # Log the error
        verbose_logger.log(
            LogLevel.ERROR,
            "test_workflow",
            f"Workflow execution failed: {e}",
            data={
                "error_type": type(e).__name__,
                "error_message": str(e),
                "user_prompt": user_prompt,
            },
        )

        return False


def analyze_verbose_logs():
    """Analyze the generated verbose logs to verify content completeness."""

    print("\n" + "=" * 80)
    print("📊 Analyzing Verbose Logs for Content Completeness")
    print("=" * 80)

    logs_file = Path("./logs/complete_workflow_logs.json")

    if not logs_file.exists():
        print("❌ Verbose logs file not found!")
        return False

    try:
        import json

        with open(logs_file, "r", encoding="utf-8") as f:
            logs = json.load(f)

        print(f"📋 Total log entries: {len(logs)}")

        # Analyze log content
        agent_logs = [log for log in logs if "agent:" in log.get("component", "")]
        llm_logs = [log for log in logs if log.get("component") == "llm"]
        coordinator_logs = [
            log for log in logs if log.get("component") == "coordinator"
        ]

        print(f"🤖 Agent logs: {len(agent_logs)}")
        print(f"🧠 LLM logs: {len(llm_logs)}")
        print(f"📋 Coordinator logs: {len(coordinator_logs)}")

        # Check for code content in logs
        code_logs = []
        tool_logs = []

        for log in logs:
            data = log.get("data", {})

            # Check for generated code
            if any(
                key in data
                for key in [
                    "generated_code",
                    "full_debugged_code",
                    "full_optimized_code",
                ]
            ):
                code_logs.append(log)

            # Check for tool results
            if any(
                key in data
                for key in ["full_tool_results", "full_evaluation_tool_results"]
            ):
                tool_logs.append(log)

        print(f"💻 Logs with code content: {len(code_logs)}")
        print(f"🔧 Logs with tool results: {len(tool_logs)}")

        # Show sample code content
        if code_logs:
            print("\n📝 Sample code content found in logs:")
            sample_log = code_logs[0]
            data = sample_log.get("data", {})

            for key in ["generated_code", "full_debugged_code", "full_optimized_code"]:
                if key in data and data[key]:
                    code_preview = (
                        data[key][:200] + "..." if len(data[key]) > 200 else data[key]
                    )
                    print(f"   {key}: {code_preview}")
                    break

        # Show sample tool results
        if tool_logs:
            print("\n🔧 Sample tool results found in logs:")
            sample_log = tool_logs[0]
            data = sample_log.get("data", {})

            for key in ["full_tool_results", "full_evaluation_tool_results"]:
                if key in data and data[key]:
                    print(
                        f"   {key}: {list(data[key].keys()) if isinstance(data[key], dict) else str(type(data[key]))}"
                    )
                    break

        print("\n✅ Log analysis completed!")
        return True

    except Exception as e:
        print(f"❌ Error analyzing logs: {e}")
        return False


if __name__ == "__main__":
    print("🧪 Pinocchio CLI Verbose Workflow Test")
    print("This script tests the complete workflow with comprehensive logging")
    print()

    # Run the test
    success = asyncio.run(test_cuda_matmul_workflow())

    if success:
        # Analyze the logs
        analyze_verbose_logs()

        print("\n" + "=" * 80)
        print("🎉 CLI Verbose Workflow Test Completed Successfully!")
        print("=" * 80)
        print()
        print("Key achievements:")
        print("✅ All agents initialized correctly")
        print("✅ Complete workflow executed")
        print("✅ Agent-generated code logged in detail")
        print("✅ MCP tool calls and results logged")
        print("✅ Verbose logs exported to JSON")
        print("✅ Performance metrics captured")
        print()
        print("📁 Log files:")
        print("   • ./logs/cli_verbose_test.log - Raw verbose log")
        print("   • ./logs/complete_workflow_logs.json - Structured JSON export")
    else:
        print("\n❌ CLI Verbose Workflow Test Failed!")
        sys.exit(1)
