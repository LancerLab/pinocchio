#!/usr/bin/env python3
"""
Test script for MCP tools integration in Pinocchio.
Tests debugging and evaluation tools functionality.
"""

import asyncio
import logging
import os
import sys

# Add the pinocchio directory to Python path
sys.path.insert(0, os.path.abspath("."))

from pinocchio.agents.debugger import DebuggerAgent
from pinocchio.agents.evaluator import EvaluatorAgent
from pinocchio.tools import CudaDebugTools, CudaEvalTools, ToolManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Sample CUDA code for testing
SAMPLE_CUDA_CODE = """
__global__ void vectorAdd(float *a, float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}
"""

SAMPLE_CUDA_CODE_WITH_ISSUES = """
__global__ int vectorAdd(float *a, float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Missing bounds check
    c[idx] = a[idx] + b[idx]  // Missing semicolon
    return 1;  // Global functions can't return non-void
}
"""


def test_tool_manager():
    """Test basic tool manager functionality."""
    print("=== Testing Tool Manager ===")

    # Create tool manager and register tools
    tool_manager = ToolManager()
    CudaDebugTools.register_tools(tool_manager)
    CudaEvalTools.register_tools(tool_manager)

    # List available tools
    tools = tool_manager.list_tools()
    print(f"Available tools: {tools}")

    # Get tool schemas
    schemas = tool_manager.get_tool_schemas()
    print(f"Tool schemas: {list(schemas.keys())}")

    return tool_manager


def test_debug_tools():
    """Test CUDA debugging tools."""
    print("\n=== Testing Debug Tools ===")

    # Create tool manager and register tools
    tool_manager = ToolManager()
    CudaDebugTools.register_tools(tool_manager)

    # Test syntax checker
    print("\n--- Testing Syntax Checker ---")
    result = tool_manager.execute_tool(
        "cuda_syntax_check", cuda_code=SAMPLE_CUDA_CODE_WITH_ISSUES, strict=True
    )
    print(f"Syntax check status: {result.status.value}")
    print(f"Output: {result.output}")
    if result.metadata:
        print(f"Issues found: {result.metadata.get('issue_count', 0)}")
        print(f"Warnings: {result.metadata.get('warning_count', 0)}")

    # Test compiler (will likely fail without nvcc)
    print("\n--- Testing Compiler ---")
    result = tool_manager.execute_tool(
        "cuda_compile", cuda_code=SAMPLE_CUDA_CODE, arch="compute_75", verbose=True
    )
    print(f"Compilation status: {result.status.value}")
    if result.error:
        print(f"Compilation error (expected if nvcc not available): {result.error}")

    # Test memory checker (will fail without cuda-memcheck)
    print("\n--- Testing Memory Checker ---")
    result = tool_manager.execute_tool(
        "cuda_memcheck", cuda_code=SAMPLE_CUDA_CODE, check_type="memcheck"
    )
    print(f"Memory check status: {result.status.value}")
    if result.error:
        print(
            f"Memory check error (expected if cuda-memcheck not available): {result.error}"
        )


def test_eval_tools():
    """Test CUDA evaluation tools."""
    print("\n=== Testing Evaluation Tools ===")

    # Create tool manager and register tools
    tool_manager = ToolManager()
    CudaEvalTools.register_tools(tool_manager)

    # Test performance analyzer
    print("\n--- Testing Performance Analyzer ---")
    result = tool_manager.execute_tool(
        "cuda_performance_analyze",
        cuda_code=SAMPLE_CUDA_CODE,
        analysis_type="comprehensive",
    )
    print(f"Performance analysis status: {result.status.value}")
    print(f"Output: {result.output}")
    if result.metadata:
        print(f"Performance score: {result.metadata.get('performance_score', 'N/A')}")

    # Test occupancy calculator
    print("\n--- Testing Occupancy Calculator ---")
    result = tool_manager.execute_tool(
        "cuda_occupancy", cuda_code=SAMPLE_CUDA_CODE, block_size=256, arch="compute_75"
    )
    print(f"Occupancy analysis status: {result.status.value}")
    print(f"Output: {result.output}")
    if result.metadata:
        occupancy_result = result.metadata.get("occupancy_result", {})
        print(
            f"Theoretical occupancy: {occupancy_result.get('occupancy_percentage', 0):.1f}%"
        )

    # Test profiler (will fail without nvprof)
    print("\n--- Testing Profiler ---")
    result = tool_manager.execute_tool(
        "cuda_profile", cuda_code=SAMPLE_CUDA_CODE, profiler="nvprof"
    )
    print(f"Profiling status: {result.status.value}")
    if result.error:
        print(f"Profiling error (expected if nvprof not available): {result.error}")


async def test_debugger_agent():
    """Test debugger agent with tools integration."""
    print("\n=== Testing Debugger Agent Integration ===")

    try:
        # Create debugger agent
        debugger = DebuggerAgent()

        # Create debug request
        request = {
            "request_id": "test_debug_001",
            "code": SAMPLE_CUDA_CODE_WITH_ISSUES,
            "language": "cuda",
            "context": {"target_arch": "compute_75", "optimization_level": "high"},
            "error_message": "Compilation failed with multiple errors",
            "step_id": "debug_step_1",
        }

        # Execute debugging
        response = await debugger.execute(request)

        print(f"Debugger response success: {response.success}")
        print(
            f"Debugger response keys: {list(response.output.keys()) if hasattr(response, 'output') and response.output else 'No output'}"
        )

        if hasattr(response, "output") and response.output:
            issues_found = response.output.get("issues_found", [])
            print(f"Issues found: {len(issues_found)}")
            if issues_found:
                print("Sample issues:")
                for i, issue in enumerate(issues_found[:3]):
                    print(f"  {i+1}. {issue}")

    except Exception as e:
        print(f"Debugger agent test failed: {e}")
        import traceback

        traceback.print_exc()


async def test_evaluator_agent():
    """Test evaluator agent with tools integration."""
    print("\n=== Testing Evaluator Agent Integration ===")

    try:
        # Create evaluator agent
        evaluator = EvaluatorAgent()

        # Create evaluation request
        request = {
            "request_id": "test_eval_001",
            "code": SAMPLE_CUDA_CODE,
            "language": "cuda",
            "context": {
                "target_arch": "compute_75",
                "block_size": 256,
                "enable_profiling": False,  # Disable profiling for test
            },
            "performance_metrics": {
                "execution_time_ms": 2.5,
                "memory_bandwidth": "150 GB/s",
                "occupancy": 0.75,
            },
            "step_id": "eval_step_1",
        }

        # Execute evaluation
        response = await evaluator.execute(request)

        print(f"Evaluator response success: {response.success}")
        print(
            f"Evaluator response keys: {list(response.output.keys()) if hasattr(response, 'output') and response.output else 'No output'}"
        )

        if hasattr(response, "output") and response.output:
            performance_score = response.output.get("performance_score")
            optimization_suggestions = response.output.get(
                "optimization_suggestions", []
            )
            print(f"Performance score: {performance_score}")
            print(f"Optimization suggestions: {len(optimization_suggestions)}")
            if optimization_suggestions:
                print("Sample suggestions:")
                for i, suggestion in enumerate(optimization_suggestions[:3]):
                    print(f"  {i+1}. {suggestion}")

    except Exception as e:
        print(f"Evaluator agent test failed: {e}")
        import traceback

        traceback.print_exc()


async def main():
    """Main test function."""
    print("Starting MCP Tools Integration Tests")
    print("=" * 50)

    # Test tool manager
    tool_manager = test_tool_manager()

    # Test individual tools
    test_debug_tools(tool_manager)
    test_eval_tools(tool_manager)

    # Test agent integration
    await test_debugger_agent()
    await test_evaluator_agent()

    print("\n" + "=" * 50)
    print("MCP Tools Integration Tests Completed")
    print("\nNote: Some tools may show errors if CUDA toolkit is not installed.")
    print(
        "This is expected behavior - the tools are designed to handle missing dependencies gracefully."
    )


if __name__ == "__main__":
    asyncio.run(main())
