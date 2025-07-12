"""Test minimal dynamic debugger insertion logic."""
import asyncio

from pinocchio.config.settings import Settings
from pinocchio.llm.mock_client import MockLLMClient
from pinocchio.task_planning.task_executor import TaskExecutor
from pinocchio.task_planning.task_planner import TaskPlanner


class FailingLLMClient(MockLLMClient):
    """Mock LLM client that simulates generator failure."""

    def __init__(self):
        """Initialize the failing LLM client."""
        super().__init__(response_delay_ms=10)

    async def complete(self, prompt, agent_type=None):
        """Simulate generator failure."""
        if agent_type == "generator":
            return '{"agent_type": "generator", "success": false, "output": {}, "error_message": "Simulated codegen error", "request_id": "test_fail"}'
        return await super().complete(prompt, agent_type)


async def test_dynamic_debugger_insertion_minimal():
    """Test minimal dynamic debugger insertion logic."""
    config = Settings()
    config.load_from_dict(
        {
            "task_planning.debug_repair.enabled": True,
            "task_planning.debug_repair.max_repair_attempts": 2,
            "task_planning.debug_repair.auto_insert_debugger": True,
            "task_planning.debug_repair.retry_generator_after_debug": True,
        }
    )
    llm_client = FailingLLMClient()
    planner = TaskPlanner(llm_client)
    executor = TaskExecutor(llm_client, config)
    user_request = "Generate a matrix multiplication kernel"
    plan = await planner.create_task_plan(user_request)
    messages = []
    async for msg in executor.execute_plan(plan):
        print(msg)
        messages.append(msg)
    assert any(
        "Dynamically inserted DEBUGGER" in m for m in messages
    ), "Debugger should be dynamically inserted"
    print("✅ Minimal dynamic debugger insertion test passed!")


if __name__ == "__main__":
    asyncio.run(test_dynamic_debugger_insertion_minimal())
