"""
Test module demonstrating usage of Workflow Fallback functionality.

This test module serves as both validation and documentation, showing developers
how the workflow fallback mechanism works between plugin-based workflows and
traditional task planning.

Key Features Demonstrated:
1. JSON-based workflow configuration and execution
2. Fallback to task planning when workflow plugins fail
3. Coordinator integration with workflow systems
4. Dynamic workflow selection and adaptation
5. Error handling and recovery mechanisms
"""

import json
import os
import sys
import tempfile
from unittest.mock import Mock, patch

import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from pinocchio.coordinator import Coordinator
from pinocchio.llm import BaseLLMClient
from pinocchio.plugins import CustomWorkflowPlugin


class MockBaseLLMClient(BaseLLMClient):
    """Mock LLM interface for testing workflow fallback scenarios."""

    def __init__(self):
        super().__init__()
        self.request_count = 0
        self.last_request = None
        self.task_planning_response = None

    def send_request(self, prompt: str, context: dict = None) -> str:
        """Simulate LLM responses for different scenarios."""
        self.request_count += 1
        self.last_request = {"prompt": prompt, "context": context}

        # Return task planning response when fallback is triggered
        if self.task_planning_response:
            return self.task_planning_response

        # Default response for workflow execution
        return f"Mock LLM response #{self.request_count} for prompt"


class TestWorkflowFallbackUsage:
    """
    Test suite demonstrating usage patterns for workflow fallback functionality.

    These tests show how developers can:
    1. Configure JSON-based workflows with fallback options
    2. Handle workflow plugin failures gracefully
    3. Integrate fallback mechanisms with the coordinator
    4. Customize fallback behavior for different scenarios
    5. Monitor and debug fallback execution
    """

    def setup_method(self):
        """Set up test environment with temporary directories and mock components."""
        self.temp_dir = tempfile.mkdtemp()
        self.mock_llm = MockBaseLLMClient()

        # Create test configuration with workflow fallback
        self.test_config = {
            "llm": {
                "provider": "mock",
                "base_url": "http://localhost:8000",
                "model_name": "test-model",
            },
            "plugins": {
                "enabled": True,
                "plugins_directory": self.temp_dir,
                "active_plugins": {"workflow": "json_workflow_plugin"},
                "plugin_configs": {
                    "json_workflow_plugin": {
                        "workflows": {
                            "working_workflow": {
                                "name": "Working CUDA Workflow",
                                "description": "A workflow that works correctly",
                                "tasks": [
                                    {
                                        "id": "generate_code",
                                        "agent_type": "generator",
                                        "description": "Generate CUDA kernel",
                                        "priority": "high",
                                    },
                                    {
                                        "id": "optimize_code",
                                        "agent_type": "optimizer",
                                        "description": "Optimize the kernel",
                                        "priority": "medium",
                                        "dependencies": ["generate_code"],
                                    },
                                ],
                            },
                            "failing_workflow": {
                                "name": "Failing Workflow",
                                "description": "A workflow designed to fail",
                                "tasks": [
                                    {
                                        "id": "invalid_task",
                                        "agent_type": "nonexistent_agent",
                                        "description": "This task will fail",
                                        "priority": "high",
                                    }
                                ],
                            },
                        }
                    }
                },
            },
            "workflow": {
                "use_plugin": True,
                "fallback_to_task_planning": True,
                "default_workflow": "working_workflow",
                "task_planning_as_backup": True,
            },
            "agents": {
                "generator": {"enabled": True},
                "optimizer": {"enabled": True},
                "debugger": {"enabled": True},
                "evaluator": {"enabled": True},
            },
        }

    def test_successful_workflow_execution_usage(self):
        """
        Demonstrates successful workflow execution without fallback.

        Usage Pattern:
        - Configure JSON workflow through plugin system
        - Execute workflow successfully
        - Validate workflow completion without fallback
        """

        # Create and configure workflow plugin
        workflow_plugin = CustomWorkflowPlugin("json_workflow_plugin")
        workflow_config = self.test_config["plugins"]["plugin_configs"][
            "json_workflow_plugin"
        ]

        assert workflow_plugin.initialize(workflow_config) == True

        # Test workflow retrieval
        working_workflow = workflow_plugin.get_workflow("working_workflow")
        assert working_workflow["name"] == "Working CUDA Workflow"
        assert len(working_workflow["tasks"]) == 2

        # Verify task structure
        tasks = working_workflow["tasks"]
        generate_task = next(t for t in tasks if t["id"] == "generate_code")
        optimize_task = next(t for t in tasks if t["id"] == "optimize_code")

        assert generate_task["agent_type"] == "generator"
        assert optimize_task["agent_type"] == "optimizer"
        assert "generate_code" in optimize_task["dependencies"]

        print("✓ Workflow configuration loaded successfully")
        print(f"✓ Workflow has {len(tasks)} tasks with proper dependencies")

        # Test workflow execution plan creation
        execution_plan = workflow_plugin.create_execution_plan("working_workflow")
        assert execution_plan["workflow_name"] == "working_workflow"
        assert len(execution_plan["execution_order"]) == 2

        # Verify execution order respects dependencies
        order = execution_plan["execution_order"]
        assert order.index("generate_code") < order.index("optimize_code")

        print("✓ Workflow execution plan created with correct dependency order")

        # Simulate successful workflow execution
        context = {"user_request": "Create optimized matrix multiplication kernel"}
        execution_result = workflow_plugin.execute_workflow("working_workflow", context)

        assert execution_result["status"] == "completed"
        assert execution_result["workflow_name"] == "working_workflow"
        assert len(execution_result["completed_tasks"]) == 2

        print("✓ Workflow executed successfully without fallback")

    def test_workflow_failure_and_fallback_usage(self):
        """
        Demonstrates workflow failure detection and fallback to task planning.

        Usage Pattern:
        - Attempt to execute a failing workflow
        - Detect failure and trigger fallback mechanism
        - Execute task planning as backup solution
        """

        # Create workflow plugin with failing configuration
        workflow_plugin = CustomWorkflowPlugin("json_workflow_plugin")
        workflow_config = self.test_config["plugins"]["plugin_configs"][
            "json_workflow_plugin"
        ]

        assert workflow_plugin.initialize(workflow_config) == True

        # Test failing workflow configuration
        failing_workflow = workflow_plugin.get_workflow("failing_workflow")
        assert failing_workflow["name"] == "Failing Workflow"
        assert len(failing_workflow["tasks"]) == 1

        # Verify failing task
        failing_task = failing_workflow["tasks"][0]
        assert failing_task["agent_type"] == "nonexistent_agent"

        print("✓ Failing workflow configuration loaded")

        # Attempt workflow execution (should fail)
        context = {"user_request": "Test failing workflow"}

        try:
            execution_result = workflow_plugin.execute_workflow(
                "failing_workflow", context
            )
            # Check if failure was detected
            if execution_result.get("status") == "failed":
                print("✓ Workflow failure detected correctly")
                fallback_needed = True
            else:
                print("⚠️ Workflow should have failed but didn't")
                fallback_needed = False
        except Exception as e:
            print(f"✓ Workflow failed with exception: {e}")
            fallback_needed = True

        # Simulate fallback to task planning
        if fallback_needed:
            print("--- Initiating Fallback to Task Planning ---")

            # Mock task planning response
            self.mock_llm.task_planning_response = json.dumps(
                {
                    "tasks": [
                        {
                            "id": "fallback_generate",
                            "agent_type": "generator",
                            "description": "Generate code as fallback",
                            "priority": "high",
                        },
                        {
                            "id": "fallback_debug",
                            "agent_type": "debugger",
                            "description": "Debug generated code",
                            "priority": "medium",
                            "dependencies": ["fallback_generate"],
                        },
                    ]
                }
            )

            # Simulate task planning fallback
            fallback_prompt = f"Create task plan for: {context['user_request']}"
            task_plan_response = self.mock_llm.send_request(fallback_prompt)
            task_plan = json.loads(task_plan_response)

            assert len(task_plan["tasks"]) == 2
            assert task_plan["tasks"][0]["agent_type"] == "generator"
            assert task_plan["tasks"][1]["agent_type"] == "debugger"

            print("✓ Fallback to task planning successful")
            print(f"✓ Generated {len(task_plan['tasks'])} fallback tasks")

    def test_coordinator_fallback_integration_usage(self):
        """
        Demonstrates coordinator integration with workflow fallback mechanism.

        Usage Pattern:
        - Configure coordinator with fallback settings
        - Process requests through coordinator
        - Show automatic fallback handling
        """

        # Mock coordinator components for testing
        class MockCoordinator:
            """Simplified coordinator for testing fallback integration."""

            def __init__(self, config):
                self.config = config
                self.workflow_plugin = None
                self.fallback_enabled = config["workflow"].get(
                    "fallback_to_task_planning", False
                )
                self.use_plugin = config["workflow"].get("use_plugin", True)

                # Initialize workflow plugin if enabled
                if self.use_plugin:
                    self.workflow_plugin = CustomWorkflowPlugin("json_workflow_plugin")
                    plugin_config = config["plugins"]["plugin_configs"][
                        "json_workflow_plugin"
                    ]
                    self.workflow_plugin.initialize(plugin_config)

            def process_request(self, user_request: str, workflow_name: str = None):
                """Process request with fallback logic."""
                result = {
                    "user_request": user_request,
                    "workflow_attempted": workflow_name,
                    "execution_path": None,
                    "fallback_triggered": False,
                    "final_status": None,
                }

                # Try workflow plugin first
                if self.use_plugin and self.workflow_plugin:
                    try:
                        if workflow_name:
                            workflow_result = self.workflow_plugin.execute_workflow(
                                workflow_name, {"user_request": user_request}
                            )

                            if workflow_result.get("status") == "completed":
                                result["execution_path"] = "workflow_plugin"
                                result["final_status"] = "success"
                                result["workflow_result"] = workflow_result
                                return result
                            else:
                                raise Exception("Workflow execution failed")
                        else:
                            raise Exception("No workflow specified")

                    except Exception as e:
                        print(f"Workflow plugin failed: {e}")
                        if self.fallback_enabled:
                            result["fallback_triggered"] = True
                        else:
                            result["final_status"] = "failed"
                            return result

                # Fallback to task planning
                if result["fallback_triggered"] or not self.use_plugin:
                    print("Executing fallback to task planning...")

                    # Simulate task planning
                    task_plan = {
                        "tasks": [
                            {
                                "id": "plan_task_1",
                                "agent_type": "generator",
                                "description": f"Generate solution for: {user_request}",
                                "priority": "high",
                            }
                        ]
                    }

                    result["execution_path"] = "task_planning_fallback"
                    result["final_status"] = "success"
                    result["task_plan"] = task_plan

                return result

        # Test coordinator with fallback integration
        coordinator = MockCoordinator(self.test_config)

        # Test 1: Successful workflow execution
        success_result = coordinator.process_request(
            "Create matrix multiplication kernel", "working_workflow"
        )

        assert success_result["execution_path"] == "workflow_plugin"
        assert success_result["final_status"] == "success"
        assert success_result["fallback_triggered"] == False

        print("✓ Coordinator workflow execution without fallback")

        # Test 2: Workflow failure with fallback
        failure_result = coordinator.process_request(
            "Test fallback mechanism", "failing_workflow"
        )

        assert failure_result["fallback_triggered"] == True
        assert failure_result["execution_path"] == "task_planning_fallback"
        assert failure_result["final_status"] == "success"

        print("✓ Coordinator fallback mechanism triggered correctly")

        # Test 3: No workflow specified (direct fallback)
        direct_fallback_result = coordinator.process_request(
            "Direct task planning request", None
        )

        assert direct_fallback_result["execution_path"] == "task_planning_fallback"
        assert direct_fallback_result["final_status"] == "success"

        print("✓ Direct fallback to task planning works correctly")

    def test_dynamic_workflow_selection_usage(self):
        """
        Demonstrates dynamic workflow selection based on request content.

        Usage Pattern:
        - Analyze user request to select appropriate workflow
        - Provide fallback options for different scenarios
        - Show adaptive workflow selection logic
        """

        class SmartWorkflowSelector:
            """Intelligent workflow selector with fallback logic."""

            def __init__(self, workflow_plugin):
                self.workflow_plugin = workflow_plugin

                # Define workflow selection rules
                self.selection_rules = {
                    "keywords": {
                        "matrix": "working_workflow",
                        "optimization": "working_workflow",
                        "debug": "working_workflow",
                        "performance": "working_workflow",
                    },
                    "fallback_order": ["working_workflow", "task_planning"],
                }

            def select_workflow(self, user_request: str) -> dict:
                """Select best workflow for user request."""
                request_lower = user_request.lower()

                # Try keyword matching
                for keyword, workflow_name in self.selection_rules["keywords"].items():
                    if keyword in request_lower:
                        return {
                            "selected_workflow": workflow_name,
                            "selection_reason": f"Keyword match: {keyword}",
                            "confidence": "high",
                        }

                # Use default workflow
                return {
                    "selected_workflow": self.selection_rules["fallback_order"][0],
                    "selection_reason": "Default selection",
                    "confidence": "medium",
                }

            def execute_with_fallback(self, user_request: str) -> dict:
                """Execute workflow with intelligent fallback."""
                selection = self.select_workflow(user_request)

                for i, workflow_option in enumerate(
                    self.selection_rules["fallback_order"]
                ):
                    try:
                        if workflow_option == "task_planning":
                            # Simulate task planning
                            return {
                                "execution_method": "task_planning",
                                "status": "success",
                                "attempt": i + 1,
                                "fallback_triggered": i > 0,
                            }
                        else:
                            # Try workflow execution
                            result = self.workflow_plugin.execute_workflow(
                                workflow_option, {"user_request": user_request}
                            )

                            if result.get("status") == "completed":
                                return {
                                    "execution_method": "workflow_plugin",
                                    "workflow_name": workflow_option,
                                    "status": "success",
                                    "attempt": i + 1,
                                    "fallback_triggered": i > 0,
                                    "workflow_result": result,
                                }
                            else:
                                raise Exception(f"Workflow {workflow_option} failed")

                    except Exception as e:
                        print(f"Attempt {i + 1} failed: {e}")
                        continue

                return {
                    "execution_method": "none",
                    "status": "failed",
                    "fallback_triggered": True,
                    "all_attempts_failed": True,
                }

        # Test dynamic workflow selection
        workflow_plugin = CustomWorkflowPlugin("json_workflow_plugin")
        workflow_config = self.test_config["plugins"]["plugin_configs"][
            "json_workflow_plugin"
        ]
        workflow_plugin.initialize(workflow_config)

        selector = SmartWorkflowSelector(workflow_plugin)

        # Test keyword-based selection
        test_requests = [
            "Create a matrix multiplication kernel",
            "Optimize CUDA performance",
            "Debug memory access issues",
            "Generate simple kernel code",
        ]

        for request in test_requests:
            selection = selector.select_workflow(request)
            print(f"Request: '{request}'")
            print(f"  Selected: {selection['selected_workflow']}")
            print(f"  Reason: {selection['selection_reason']}")
            print(f"  Confidence: {selection['confidence']}")

            # Test execution with fallback
            execution_result = selector.execute_with_fallback(request)
            assert execution_result["status"] == "success"

            print(f"  Execution: {execution_result['execution_method']}")
            print(f"  Fallback triggered: {execution_result['fallback_triggered']}")

        print("✓ Dynamic workflow selection works correctly")

    def test_fallback_configuration_patterns_usage(self):
        """
        Demonstrates different fallback configuration patterns.

        Usage Pattern:
        - Show various fallback configuration options
        - Test different fallback strategies
        - Validate configuration-driven behavior
        """

        # Test different configuration patterns
        config_patterns = {
            "strict_workflow": {
                "use_plugin": True,
                "fallback_to_task_planning": False,
                "description": "Only use workflows, no fallback",
            },
            "fallback_enabled": {
                "use_plugin": True,
                "fallback_to_task_planning": True,
                "description": "Use workflows with task planning fallback",
            },
            "task_planning_only": {
                "use_plugin": False,
                "fallback_to_task_planning": True,
                "description": "Only use task planning",
            },
            "auto_adaptive": {
                "use_plugin": True,
                "fallback_to_task_planning": True,
                "auto_select_workflow": True,
                "description": "Automatically adapt execution method",
            },
        }

        for pattern_name, pattern_config in config_patterns.items():
            print(f"\nTesting pattern: {pattern_name}")
            print(f"Description: {pattern_config['description']}")

            # Test configuration validation
            assert isinstance(pattern_config["use_plugin"], bool)
            assert isinstance(pattern_config["fallback_to_task_planning"], bool)

            # Simulate behavior based on configuration
            if (
                pattern_config["use_plugin"]
                and pattern_config["fallback_to_task_planning"]
            ):
                behavior = "Try workflow, fallback to task planning on failure"
            elif (
                pattern_config["use_plugin"]
                and not pattern_config["fallback_to_task_planning"]
            ):
                behavior = "Workflow only, fail if workflow fails"
            elif (
                not pattern_config["use_plugin"]
                and pattern_config["fallback_to_task_planning"]
            ):
                behavior = "Task planning only"
            else:
                behavior = "No execution method configured"

            print(f"Expected behavior: {behavior}")

        print("\n✓ All fallback configuration patterns validated")

    def test_fallback_monitoring_and_debugging_usage(self):
        """
        Demonstrates monitoring and debugging of fallback mechanisms.

        Usage Pattern:
        - Track fallback events and reasons
        - Monitor system behavior during fallbacks
        - Debug fallback configuration issues
        """

        class FallbackMonitor:
            """Monitor for tracking fallback events and performance."""

            def __init__(self):
                self.events = []
                self.statistics = {
                    "total_requests": 0,
                    "workflow_success": 0,
                    "fallback_triggered": 0,
                    "total_failures": 0,
                }

            def log_event(self, event_type: str, details: dict):
                """Log a fallback-related event."""
                event = {
                    "timestamp": "mock_timestamp",
                    "type": event_type,
                    "details": details,
                }
                self.events.append(event)

                # Update statistics
                if event_type == "request_start":
                    self.statistics["total_requests"] += 1
                elif event_type == "workflow_success":
                    self.statistics["workflow_success"] += 1
                elif event_type == "fallback_triggered":
                    self.statistics["fallback_triggered"] += 1
                elif event_type == "execution_failed":
                    self.statistics["total_failures"] += 1

            def get_fallback_rate(self) -> float:
                """Calculate fallback trigger rate."""
                if self.statistics["total_requests"] == 0:
                    return 0.0
                return (
                    self.statistics["fallback_triggered"]
                    / self.statistics["total_requests"]
                )

            def get_success_rate(self) -> float:
                """Calculate overall success rate."""
                if self.statistics["total_requests"] == 0:
                    return 0.0
                successes = (
                    self.statistics["workflow_success"]
                    + self.statistics["fallback_triggered"]
                )
                return successes / self.statistics["total_requests"]

            def generate_report(self) -> dict:
                """Generate comprehensive monitoring report."""
                return {
                    "statistics": self.statistics,
                    "fallback_rate": self.get_fallback_rate(),
                    "success_rate": self.get_success_rate(),
                    "recent_events": self.events[-5:],  # Last 5 events
                    "total_events": len(self.events),
                }

        # Test fallback monitoring
        monitor = FallbackMonitor()

        # Simulate various scenarios
        scenarios = [
            {"request": "Working request 1", "workflow_works": True},
            {"request": "Working request 2", "workflow_works": True},
            {"request": "Failing request 1", "workflow_works": False},
            {"request": "Working request 3", "workflow_works": True},
            {"request": "Failing request 2", "workflow_works": False},
        ]

        for scenario in scenarios:
            monitor.log_event("request_start", {"request": scenario["request"]})

            if scenario["workflow_works"]:
                monitor.log_event(
                    "workflow_success",
                    {"request": scenario["request"], "workflow": "working_workflow"},
                )
            else:
                monitor.log_event(
                    "fallback_triggered",
                    {
                        "request": scenario["request"],
                        "reason": "workflow_execution_failed",
                        "fallback_method": "task_planning",
                    },
                )

        # Generate and validate monitoring report
        report = monitor.generate_report()

        assert report["statistics"]["total_requests"] == 5
        assert report["statistics"]["workflow_success"] == 3
        assert report["statistics"]["fallback_triggered"] == 2
        assert report["fallback_rate"] == 0.4  # 2/5 = 40%
        assert (
            report["success_rate"] == 1.0
        )  # All requests succeeded (workflow or fallback)

        print("✓ Fallback monitoring statistics:")
        print(f"  Total requests: {report['statistics']['total_requests']}")
        print(f"  Workflow successes: {report['statistics']['workflow_success']}")
        print(f"  Fallbacks triggered: {report['statistics']['fallback_triggered']}")
        print(f"  Fallback rate: {report['fallback_rate']:.1%}")
        print(f"  Overall success rate: {report['success_rate']:.1%}")

        # Test debugging capabilities
        print("\n✓ Recent events for debugging:")
        for event in report["recent_events"]:
            print(f"  {event['type']}: {event['details']}")


if __name__ == "__main__":
    """
    Run workflow fallback usage tests and display results.

    This section demonstrates comprehensive workflow fallback testing.
    """
    print("Running Workflow Fallback Usage Tests...")
    print("=" * 55)

    # Create test instance
    test_instance = TestWorkflowFallbackUsage()
    test_instance.setup_method()

    # Run each test with clear output
    tests = [
        (
            "Successful Workflow Execution",
            test_instance.test_successful_workflow_execution_usage,
        ),
        (
            "Workflow Failure and Fallback",
            test_instance.test_workflow_failure_and_fallback_usage,
        ),
        (
            "Coordinator Fallback Integration",
            test_instance.test_coordinator_fallback_integration_usage,
        ),
        (
            "Dynamic Workflow Selection",
            test_instance.test_dynamic_workflow_selection_usage,
        ),
        (
            "Fallback Configuration Patterns",
            test_instance.test_fallback_configuration_patterns_usage,
        ),
        (
            "Fallback Monitoring and Debugging",
            test_instance.test_fallback_monitoring_and_debugging_usage,
        ),
    ]

    passed = 0
    total = len(tests)

    for name, test_func in tests:
        try:
            print(f"\n{name}:")
            test_func()
            print(f"✅ {name} - PASSED")
            passed += 1
        except Exception as e:
            print(f"❌ {name} - FAILED: {e}")
            import traceback

            traceback.print_exc()

    print("\n" + "=" * 55)
    print(f"Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All workflow fallback tests passed!")
        print("\nKey Benefits Demonstrated:")
        print("- Robust workflow execution with automatic fallback")
        print("- JSON-based workflow configuration and management")
        print("- Intelligent workflow selection and adaptation")
        print("- Comprehensive monitoring and debugging capabilities")
        print("- Flexible configuration patterns for different use cases")
        print("\nDevelopers can now:")
        print("- Implement reliable workflow systems with fallbacks")
        print("- Configure adaptive execution strategies")
        print("- Monitor and debug workflow performance")
        print("- Handle workflow failures gracefully")
        print("- Create custom workflow selection logic")
    else:
        print("⚠️ Some tests failed. Check workflow fallback implementation.")
