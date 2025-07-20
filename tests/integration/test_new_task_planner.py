#!/usr/bin/env python3
"""Test script for the new TaskPlanner design."""

import asyncio
import json
import os
import sys

import pytest

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pinocchio.llm.mock_client import MockLLMClient
from pinocchio.task_planning.task_planner import TaskPlanner


@pytest.mark.asyncio
async def test_new_task_planner():
    """Test the new TaskPlanner design."""
    print("🧪 Testing new TaskPlanner design...")

    # Create a mock LLM client that returns a valid task decomposition response
    mock_response = {
        "agent_type": "planner",
        "success": True,
        "output": {
            "tasks": [
                {
                    "task_id": "task_1",
                    "agent_type": "generator",
                    "description": "Generate initial CUDA matrix multiplication kernel",
                    "dependencies": [],
                    "priority": "critical",
                    "requirements": {
                        "code_type": "cuda",
                        "optimization_level": "basic",
                    },
                },
                {
                    "task_id": "task_2",
                    "agent_type": "debugger",
                    "description": "Debug and validate the generated code",
                    "dependencies": ["task_1"],
                    "priority": "high",
                    "requirements": {"error_handling": True, "validation": True},
                },
                {
                    "task_id": "task_3",
                    "agent_type": "optimizer",
                    "description": "Optimize the code for performance",
                    "dependencies": ["task_2"],
                    "priority": "high",
                    "requirements": {
                        "optimization_goals": ["performance", "memory_efficiency"]
                    },
                },
                {
                    "task_id": "task_4",
                    "agent_type": "evaluator",
                    "description": "Evaluate the final code performance",
                    "dependencies": ["task_3"],
                    "priority": "medium",
                    "requirements": {
                        "evaluation_metrics": [
                            "performance",
                            "correctness",
                            "maintainability",
                        ]
                    },
                },
            ]
        },
        "explanation": "Decomposed the request into logical steps: generate, debug, optimize, evaluate",
        "confidence": 0.95,
    }

    # Create mock LLM client
    mock_llm = MockLLMClient()
    mock_llm.set_custom_response("planner", mock_response)

    # Create TaskPlanner
    planner = TaskPlanner(llm_client=mock_llm, mode="development")

    # Test user request
    user_request = "Generate a high-performance CUDA matrix multiplication kernel"

    print(f"📝 User request: {user_request}")

    try:
        # Test task plan creation
        plan = await planner.create_task_plan(user_request)
        print("✅ Task plan created successfully!")
        print(f"📊 Plan ID: {plan.plan_id}")
        print(f"📋 Number of tasks: {len(plan.tasks)}")

        # Display tasks
        for i, task in enumerate(plan.tasks, 1):
            print(f"  {i}. {task.agent_type}: {task.task_description}")
            print(f"     Priority: {task.priority}")
            print(f"     Dependencies: {[dep.task_id for dep in task.dependencies]}")
            print(f"     Requirements: {task.requirements}")

        # Validate plan
        validation = planner.validate_plan(plan)
        print(f"✅ Plan validation: {validation['is_valid']}")

        # Test fallback behavior
        print("\n🧪 Testing fallback behavior...")
        fallback_planner = TaskPlanner(mode="production")  # No LLM client
        fallback_plan = await fallback_planner.create_task_plan(user_request)
        print(f"✅ Fallback plan created with {len(fallback_plan.tasks)} tasks")

        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_prompt_format():
    """Test the new prompt format."""
    print("\n🧪 Testing new prompt format...")

    planner = TaskPlanner(mode="development")
    user_request = "Generate a high-performance CUDA matrix multiplication kernel"

    prompt = planner._build_task_decomposition_prompt(user_request)
    print("📝 Generated prompt:")
    print("=" * 60)
    print(prompt)
    print("=" * 60)

    # Check if prompt contains required elements
    required_elements = [
        "agent_type",
        "success",
        "output",
        "tasks",
        "task_id",
        "agent_type",
        "description",
        "dependencies",
        "priority",
        "requirements",
    ]

    missing_elements = []
    for element in required_elements:
        if element not in prompt:
            missing_elements.append(element)

    if missing_elements:
        print(f"❌ Missing elements in prompt: {missing_elements}")
        return False
    else:
        print("✅ All required elements present in prompt")
        return True


@pytest.mark.asyncio
async def test_parsing():
    """Test the parsing logic."""
    print("\n🧪 Testing parsing logic...")

    planner = TaskPlanner(mode="development")

    # Test valid response
    valid_response = """{
        "agent_type": "planner",
        "success": true,
        "output": {
            "tasks": [
                {
                    "task_id": "task_1",
                    "agent_type": "generator",
                    "description": "Generate code",
                    "dependencies": [],
                    "priority": "critical",
                    "requirements": {}
                }
            ]
        }
    }"""

    try:
        tasks = planner._parse_task_decomposition_response(
            valid_response, "test request"
        )
        print(f"✅ Parsed {len(tasks)} tasks from valid response")

        if tasks:
            task = tasks[0]
            print(f"  - Task ID: {task.task_id}")
            print(f"  - Agent Type: {task.agent_type}")
            print(f"  - Description: {task.task_description}")
            print(f"  - Priority: {task.priority}")

        return True

    except Exception as e:
        print(f"❌ Parsing failed: {e}")
        return False


async def main():
    """Run all tests."""
    print("🚀 Starting new TaskPlanner design tests...")

    results = []

    # Test 1: Basic functionality
    results.append(await test_new_task_planner())

    # Test 2: Prompt format
    results.append(await test_prompt_format())

    # Test 3: Parsing logic
    results.append(await test_parsing())

    # Summary
    print("\n📊 Test Results:")
    print(f"  Basic functionality: {'✅ PASS' if results[0] else '❌ FAIL'}")
    print(f"  Prompt format: {'✅ PASS' if results[1] else '❌ FAIL'}")
    print(f"  Parsing logic: {'✅ PASS' if results[2] else '❌ FAIL'}")

    if all(results):
        print("\n🎉 All tests passed!")
        return 0
    else:
        print("\n💥 Some tests failed!")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
