#!/usr/bin/env python3
"""Test multi-round generator→debugger→optimiser chain with dynamic insertion."""

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pinocchio.config.settings import Settings
from pinocchio.llm.mock_client import MockLLMClient
from pinocchio.task_planning.task_executor import TaskExecutor
from pinocchio.task_planning.task_planner import TaskPlanner


class BuggyGeneratorLLMClient(MockLLMClient):
    """Mock LLM client that simulates buggy generator and debugger."""

    def __init__(self, fail_pattern=None):
        """Initialize with fail pattern for generator and debugger."""
        super().__init__(response_delay_ms=10)
        self.fail_pattern = fail_pattern or [True, True, False]
        self.generator_calls = 0
        self.debugger_calls = 0

    async def complete(self, prompt, agent_type=None):
        """Simulate generator and debugger failures."""
        if agent_type == "generator":
            self.generator_calls += 1
            if (
                self.generator_calls <= len(self.fail_pattern)
                and self.fail_pattern[self.generator_calls - 1]
            ):
                return '{"agent_type": "generator", "success": false, "output": {}, "error_message": f"Simulated codegen error (attempt {self.generator_calls})", "request_id": "test_fail"}'
            else:
                return await super().complete(prompt, agent_type)
        elif agent_type == "debugger":
            self.debugger_calls += 1
            if self.debugger_calls <= 2:
                return '{"agent_type": "debugger", "success": true, "output": {"bugs_found": 3, "compilation_error": "syntax error"}, "request_id": "test_debug"}'
            else:
                return await super().complete(prompt, agent_type)
        return await super().complete(prompt, agent_type)


async def test_multi_round_optimisation():
    """Test multi-round generator→debugger→optimiser chain with dynamic insertion."""
    print("🧪 Testing Multi-Round Generator→Debugger→Optimiser Chain")
    print("=" * 70)
    test_configs = [
        {
            "name": "2 rounds with optimiser enabled",
            "config": {
                "task_planning.max_optimisation_rounds": 2,
                "task_planning.enable_optimiser": True,
                "task_planning.debug_repair.enabled": True,
                "task_planning.debug_repair.auto_insert_debugger": True,
                "task_planning.debug_repair.retry_generator_after_debug": True,
            },
            "fail_pattern": [True, False],
            "expected_rounds": 2,
        },
        {
            "name": "3 rounds with optimiser disabled",
            "config": {
                "task_planning.max_optimisation_rounds": 3,
                "task_planning.enable_optimiser": False,
                "task_planning.debug_repair.enabled": True,
                "task_planning.debug_repair.auto_insert_debugger": True,
                "task_planning.debug_repair.retry_generator_after_debug": True,
            },
            "fail_pattern": [True, True, False],
            "expected_rounds": 3,
        },
    ]
    for i, test_config in enumerate(test_configs, 1):
        print(f"\n📋 Test {i}: {test_config['name']}")
        print(
            f"   Config: max_rounds={test_config['config']['task_planning.max_optimisation_rounds']}, optimiser={test_config['config']['task_planning.enable_optimiser']}"
        )
        print(f"   Expected rounds: {test_config['expected_rounds']}")
        config = Settings()
        config.load_from_dict(test_config["config"])
        llm_client = BuggyGeneratorLLMClient(fail_pattern=test_config["fail_pattern"])
        planner = TaskPlanner(llm_client)
        executor = TaskExecutor(llm_client, config)
        user_request = "Generate a matrix multiplication kernel"
        plan = await planner.create_task_plan(user_request)
        initial_tasks = {}
        for task in plan.tasks:
            agent_type = (
                task.agent_type.value
                if hasattr(task.agent_type, "value")
                else str(task.agent_type)
            )
            if agent_type not in initial_tasks:
                initial_tasks[agent_type] = 0
            initial_tasks[agent_type] += 1
        print(f"   Initial plan: {initial_tasks}")
        messages = []
        async for msg in executor.execute_plan(plan):
            print(f"   {msg}")
            messages.append(msg)
        final_tasks = {}
        for task in plan.tasks:
            agent_type = (
                task.agent_type.value
                if hasattr(task.agent_type, "value")
                else str(task.agent_type)
            )
            if agent_type not in final_tasks:
                final_tasks[agent_type] = 0
            final_tasks[agent_type] += 1
        print(f"   Final plan: {final_tasks}")
        dynamic_insertions = [m for m in messages if "Dynamically inserted" in m]
        round_insertions = [
            m for m in messages if "Round" in m and "tasks after debugger" in m
        ]
        max_rounds_reached = [m for m in messages if "Max optimisation rounds" in m]
        print(f"   Dynamic insertions: {len(dynamic_insertions)}")
        print(f"   Round insertions: {len(round_insertions)}")
        print(f"   Max rounds reached: {len(max_rounds_reached)}")
        expected_generators = test_config["expected_rounds"]
        expected_debuggers = test_config["expected_rounds"]
        expected_optimisers = (
            test_config["expected_rounds"]
            if test_config["config"]["task_planning.enable_optimiser"]
            else 0
        )
        assert (
            final_tasks.get("generator", 0) >= expected_generators
        ), f"Expected at least {expected_generators} generators, got {final_tasks.get('generator', 0)}"
        assert (
            final_tasks.get("debugger", 0) >= expected_debuggers
        ), f"Expected at least {expected_debuggers} debuggers, got {final_tasks.get('debugger', 0)}"
        if test_config["config"]["task_planning.enable_optimiser"]:
            assert (
                final_tasks.get("optimizer", 0) >= expected_optimisers
            ), f"Expected at least {expected_optimisers} optimisers, got {final_tasks.get('optimizer', 0)}"
        print(f"   ✅ Test {i} passed!")
    print("\n🎉 Multi-round optimisation chain test completed!")


if __name__ == "__main__":
    asyncio.run(test_multi_round_optimisation())
