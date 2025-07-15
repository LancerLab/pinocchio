#!/usr/bin/env python3
"""Test script for TaskPlanner with real LLM integration."""

import asyncio
import json
import os
import sys
from typing import Optional

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pinocchio.config.settings import Settings
from pinocchio.llm.custom_llm_client import CustomLLMClient
from pinocchio.task_planning.task_planner import TaskPlanner


async def test_real_llm_task_planning():
    """Test TaskPlanner with real LLM."""
    print("🧪 Testing TaskPlanner with real LLM...")

    # Load configuration
    config = Settings()
    try:
        config.load_from_file("pinocchio.json")
        print("✅ Configuration loaded from pinocchio.json")
    except Exception as e:
        print(f"⚠️ Could not load pinocchio.json: {e}")
        print("Using default configuration")

    # Create real LLM client
    try:
        llm_client = CustomLLMClient()
        print("✅ Real LLM client created")
    except Exception as e:
        print(f"❌ Failed to create LLM client: {e}")
        return False

    # Test LLM connectivity
    try:
        test_response = await llm_client.complete("Hello", agent_type="generator")
        print(f"✅ LLM connectivity test passed: {len(test_response)} characters")
    except Exception as e:
        print(f"❌ LLM connectivity test failed: {e}")
        return False

    # Create TaskPlanner with real LLM
    planner = TaskPlanner(llm_client=llm_client, mode="development")

    # Test cases
    test_cases = [
        {
            "name": "CUDA Matrix Multiplication",
            "request": "Generate a high-performance CUDA matrix multiplication kernel",
            "expected_keywords": ["cuda", "matrix", "multiplication", "performance"],
        },
        {
            "name": "Simple Function",
            "request": "Create a simple function to calculate fibonacci numbers",
            "expected_keywords": ["fibonacci", "simple", "function"],
        },
        {
            "name": "Complex Optimization",
            "request": "Optimize a complex algorithm for maximum performance and memory efficiency",
            "expected_keywords": ["optimize", "performance", "memory", "efficiency"],
        },
        {
            "name": "Debugging Task",
            "request": "Debug and fix memory leaks in a C++ application",
            "expected_keywords": ["debug", "memory", "leaks", "c++"],
        },
    ]

    results = []

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📝 Test {i}: {test_case['name']}")
        print(f"Request: {test_case['request']}")

        try:
            # Test request analysis
            print("  🔍 Analyzing request...")
            context = await planner._analyze_request(test_case["request"])
            print(f"  ✅ Analysis successful")
            print(
                f"  📊 Primary goal: {context.requirements.get('primary_goal', 'N/A')}"
            )
            print(f"  🎯 Optimization goals: {context.optimization_goals}")
            print(f"  🔒 Constraints: {context.constraints}")
            print(f"  📋 Planning strategy: {context.planning_strategy}")

            # Test task generation
            print("  🏗️ Generating tasks...")
            tasks = await planner._generate_tasks(context)
            print(f"  ✅ Generated {len(tasks)} tasks")

            for j, task in enumerate(tasks, 1):
                print(f"    {j}. {task.agent_type}: {task.task_description[:60]}...")

            # Test full plan creation
            print("  📋 Creating full task plan...")
            plan = await planner.create_task_plan(test_case["request"])
            print(f"  ✅ Created plan with {len(plan.tasks)} tasks")

            # Validate plan
            validation = planner.validate_plan(plan)
            print(f"  ✅ Plan validation: {validation['is_valid']}")

            # Check for expected keywords in the analysis
            analysis_text = json.dumps(context.model_dump(), indent=2).lower()
            missing_keywords = []
            for keyword in test_case["expected_keywords"]:
                if keyword.lower() not in analysis_text:
                    missing_keywords.append(keyword)

            if missing_keywords:
                print(f"  ⚠️ Missing expected keywords: {missing_keywords}")
            else:
                print(f"  ✅ All expected keywords found")

            results.append(
                {
                    "test_name": test_case["name"],
                    "success": True,
                    "task_count": len(tasks),
                    "validation_passed": validation["is_valid"],
                    "missing_keywords": missing_keywords,
                }
            )

        except Exception as e:
            print(f"  ❌ Test failed: {e}")
            import traceback

            traceback.print_exc()
            results.append(
                {"test_name": test_case["name"], "success": False, "error": str(e)}
            )

    # Summary
    print(f"\n📊 Test Results Summary:")
    print(f"  Total tests: {len(test_cases)}")
    successful_tests = [r for r in results if r["success"]]
    print(f"  Successful: {len(successful_tests)}")
    print(f"  Failed: {len(results) - len(successful_tests)}")

    for result in results:
        status = "✅ PASS" if result["success"] else "❌ FAIL"
        print(f"  {result['test_name']}: {status}")
        if result["success"]:
            print(f"    - Tasks generated: {result['task_count']}")
            print(f"    - Validation passed: {result['validation_passed']}")
            if result["missing_keywords"]:
                print(f"    - Missing keywords: {result['missing_keywords']}")
        else:
            print(f"    - Error: {result['error']}")

    return len(successful_tests) == len(test_cases)


async def test_llm_response_format():
    """Test LLM response format specifically."""
    print("\n🧪 Testing LLM response format...")

    # Load configuration
    config = Settings()
    try:
        config.load_from_file("pinocchio.json")
    except Exception:
        pass

    # Create real LLM client
    try:
        llm_client = CustomLLMClient()
    except Exception as e:
        print(f"❌ Failed to create LLM client: {e}")
        return False

    # Create TaskPlanner
    planner = TaskPlanner(llm_client=llm_client, mode="development")

    # Test request
    test_request = "Generate a high-performance CUDA matrix multiplication kernel"

    try:
        # Get the prompt
        prompt = planner._build_analysis_prompt(test_request)
        print("📝 Generated prompt:")
        print("=" * 60)
        print(prompt)
        print("=" * 60)

        # Test LLM response
        print("\n🤖 Testing LLM response...")
        response = await llm_client.complete(prompt, agent_type="planner")
        print("📄 Raw LLM response:")
        print("=" * 60)
        print(response)
        print("=" * 60)

        # Test parsing
        print("\n🔍 Testing response parsing...")
        analysis = planner._parse_analysis_response(response, test_request)
        print("✅ Response parsed successfully")
        print("📊 Parsed analysis:")
        print(json.dumps(analysis, indent=2))

        # Test context creation
        print("\n🏗️ Testing context creation...")
        context = await planner._analyze_request(test_request)
        print("✅ Context created successfully")
        print("📊 Context:")
        print(json.dumps(context.model_dump(), indent=2))

        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_error_handling():
    """Test error handling with real LLM."""
    print("\n🧪 Testing error handling...")

    # Load configuration
    config = Settings()
    try:
        config.load_from_file("pinocchio.json")
    except Exception:
        pass

    # Create real LLM client
    try:
        llm_client = CustomLLMClient()
    except Exception as e:
        print(f"❌ Failed to create LLM client: {e}")
        return False

    # Test development mode (should raise errors)
    print("  🔧 Testing development mode...")
    planner_dev = TaskPlanner(llm_client=llm_client, mode="development")

    try:
        # This should work normally
        context = await planner_dev._analyze_request("Test request")
        print("  ✅ Development mode analysis successful")
    except Exception as e:
        print(f"  ❌ Development mode failed: {e}")

    # Test production mode (should use fallback)
    print("  🏭 Testing production mode...")
    planner_prod = TaskPlanner(llm_client=llm_client, mode="production")

    try:
        context = await planner_prod._analyze_request("Test request")
        print("  ✅ Production mode analysis successful")
    except Exception as e:
        print(f"  ❌ Production mode failed: {e}")

    return True


async def main():
    """Run all real LLM tests."""
    print("🚀 Starting real LLM TaskPlanner tests...")

    results = []

    # Test 1: Basic functionality with real LLM
    results.append(await test_real_llm_task_planning())

    # Test 2: LLM response format testing
    results.append(await test_llm_response_format())

    # Test 3: Error handling
    results.append(await test_error_handling())

    # Summary
    print("\n📊 Overall Test Results:")
    print(f"  Real LLM functionality: {'✅ PASS' if results[0] else '❌ FAIL'}")
    print(f"  Response format testing: {'✅ PASS' if results[1] else '❌ FAIL'}")
    print(f"  Error handling: {'✅ PASS' if results[2] else '❌ FAIL'}")

    if all(results):
        print("\n🎉 All real LLM tests passed!")
        return 0
    else:
        print("\n💥 Some real LLM tests failed!")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
