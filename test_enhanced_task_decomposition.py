"""Test enhanced task decomposition logic."""
import asyncio

from pinocchio.config.settings import Settings
from pinocchio.llm.mock_client import MockLLMClient
from pinocchio.task_planning.task_executor import TaskExecutor
from pinocchio.task_planning.task_planner import TaskPlanner


async def test_enhanced_task_decomposition():
    """Test enhanced task decomposition logic."""
    config = Settings()
    config.load_from_dict(
        {
            "task_planning.max_optimisation_rounds": 2,
            "task_planning.enable_optimiser": True,
        }
    )
    llm_client = MockLLMClient()
    planner = TaskPlanner(llm_client)
    executor = TaskExecutor(llm_client, config)
    user_request = "Generate a matrix multiplication kernel with high performance"
    plan = await planner.create_task_plan(user_request)
    print(f"Initial plan tasks: {[t.agent_type for t in plan.tasks]}")
    messages = []
    async for msg in executor.execute_plan(plan):
        print(msg)
        messages.append(msg)
    print("✅ Enhanced task decomposition test passed!")


if __name__ == "__main__":
    asyncio.run(test_enhanced_task_decomposition())
