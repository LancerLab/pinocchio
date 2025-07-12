#!/usr/bin/env python3
"""Test debug repair loop mechanism with configurable max attempts."""
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pinocchio.config.settings import Settings
from pinocchio.llm.mock_client import MockLLMClient
from pinocchio.task_planning.task_executor import TaskExecutor
from pinocchio.task_planning.task_planner import TaskPlanner


class FailingGeneratorLLMClient(MockLLMClient):
    """Mock LLM client that simulates generator failure."""

    def __init__(self, fail_count=2):
        """Initialize with number of failures before success."""
        super().__init__(response_delay_ms=10)
        self.fail_count = fail_count
        self.generator_calls = 0

    async def complete(self, prompt, agent_type=None):
        """Simulate generator failure for first N calls."""
        if agent_type == "generator":
            self.generator_calls += 1
            if self.generator_calls <= self.fail_count:
                return '{"agent_type": "generator", "success": false, "output": {}, "error_message": f"Simulated codegen error (attempt {self.generator_calls})", "request_id": "test_fail"}'
            else:
                return await super().complete(prompt, agent_type)
        return await super().complete(prompt, agent_type)


class GeneratorOnlyPlanner(TaskPlanner):
    """Task planner that only returns a generator task."""

    async def _generate_tasks(self, context):
        """Return only a generator task."""
        generator_task = self._build_generator_task(context)
        return [generator_task]

    def _build_generator_task(self, context):
        return self._build_task(context)

    def _build_task(self, context):
        from pinocchio.data_models.task_planning import AgentType, Task, TaskPriority

        return Task(
            task_id="task_1",
            agent_type=AgentType.GENERATOR,
            task_description=context.user_request,
            requirements=context.requirements,
            optimization_goals=context.optimization_goals,
            priority=TaskPriority.CRITICAL,
            input_data={
                "user_request": context.user_request,
                "instruction": self._build_generator_instruction(context),
            },
        )


async def test_debug_repair_loop():
    """Test the debug repair loop mechanism."""
    print("🧪 Testing Debug Repair Loop Mechanism")
    print("=" * 60)

    test_configs = [
        {
            "name": "Max 2 repair attempts",
            "config": {
                "task_planning.debug_repair.enabled": True,
                "task_planning.debug_repair.max_repair_attempts": 2,
                "task_planning.debug_repair.auto_insert_debugger": True,
                "task_planning.debug_repair.retry_generator_after_debug": True,
            },
            "fail_count": 3,
            "expected_repair_attempts": 2,
        },
        {
            "name": "Max 1 repair attempt",
            "config": {
                "task_planning.debug_repair.enabled": True,
                "task_planning.debug_repair.max_repair_attempts": 1,
                "task_planning.debug_repair.auto_insert_debugger": True,
                "task_planning.debug_repair.retry_generator_after_debug": True,
            },
            "fail_count": 3,
            "expected_repair_attempts": 1,
        },
    ]

    for i, test_config in enumerate(test_configs, 1):
        print(f"\n📋 Test {i}: {test_config['name']}")
        print(
            f"   Config: max_repair_attempts={test_config['config']['task_planning.debug_repair.max_repair_attempts']}"
        )
        print(f"   Expected repair attempts: {test_config['expected_repair_attempts']}")
        config = Settings()
        config.load_from_dict(test_config["config"])
        llm_client = FailingGeneratorLLMClient(fail_count=test_config["fail_count"])
        planner = GeneratorOnlyPlanner(llm_client)
        executor = TaskExecutor(llm_client, config)
        user_request = "Generate a matrix multiplication kernel"
        plan = await planner.create_task_plan(user_request)
        print(f"   Initial plan tasks: {[t.agent_type for t in plan.tasks]}")
        messages = []
        async for msg in executor.execute_plan(plan):
            print(f"   {msg}")
            messages.append(msg)
        debugger_insertions = [
            m for m in messages if "Dynamically inserted DEBUGGER" in m
        ]
        generator_retries = [m for m in messages if "Added retry GENERATOR" in m]
        max_attempts_reached = [m for m in messages if "Max repair attempts" in m]
        print(f"   Debugger insertions: {len(debugger_insertions)}")
        print(f"   Generator retries: {len(generator_retries)}")
        print(f"   Max attempts reached: {len(max_attempts_reached)}")
        assert (
            len(debugger_insertions) == test_config["expected_repair_attempts"]
        ), f"Expected {test_config['expected_repair_attempts']} debugger insertions, got {len(debugger_insertions)}"
        assert (
            len(generator_retries) == test_config["expected_repair_attempts"]
        ), f"Expected {test_config['expected_repair_attempts']} generator retries, got {len(generator_retries)}"
        if test_config["expected_repair_attempts"] < test_config["fail_count"]:
            assert len(max_attempts_reached) > 0, "Should have reached max attempts"
        print(f"   ✅ Test {i} passed!")
    print("\n🎉 Debug repair loop mechanism test completed!")


if __name__ == "__main__":
    asyncio.run(test_debug_repair_loop())
