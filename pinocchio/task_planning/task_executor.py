"""Task executor for executing task plans and managing agent interactions."""

import logging
from typing import Any, AsyncGenerator, Dict, Optional

from ..agents import DebuggerAgent, EvaluatorAgent, GeneratorAgent, OptimizerAgent
from ..data_models.task_planning import (
    AgentType,
    Task,
    TaskPlan,
    TaskPriority,
    TaskResult,
)
from ..llm.mock_client import MockLLMClient

logger = logging.getLogger(__name__)


class TaskExecutor:
    """Executor for task plans with intelligent agent management."""

    def __init__(self, llm_client: Optional[Any] = None):
        """
        Initialize task executor.

        Args:
            llm_client: LLM client for agents (uses MockLLMClient if None)
        """
        self.llm_client = llm_client or MockLLMClient(response_delay_ms=200)

        # Initialize agents
        self.agents = {
            AgentType.GENERATOR: GeneratorAgent(self.llm_client),
            AgentType.OPTIMIZER: OptimizerAgent(self.llm_client),
            AgentType.DEBUGGER: DebuggerAgent(self.llm_client),
            AgentType.EVALUATOR: EvaluatorAgent(self.llm_client),
        }

        logger.info("TaskExecutor initialized with all agents")

    async def execute_plan(self, plan: TaskPlan) -> AsyncGenerator[str, None]:
        """
        Execute a task plan with progress reporting.

        Args:
            plan: Task plan to execute

        Yields:
            Progress messages during execution
        """
        plan.mark_started()
        yield f"🤖 Starting execution of plan {plan.plan_id} with {len(plan.tasks)} tasks"

        # Track execution state
        completed_tasks = []
        failed_tasks = []
        execution_results = {}

        while not plan.is_completed() and not plan.is_failed():
            # Get ready tasks
            ready_tasks = plan.get_ready_tasks()

            if not ready_tasks:
                # Check if we're stuck
                if len(completed_tasks) + len(failed_tasks) == len(plan.tasks):
                    break
                else:
                    yield "⚠️ No ready tasks, but plan not complete. This may indicate a dependency issue."
                    break

            # Execute ready tasks (could be parallel in the future)
            for task in ready_tasks:
                yield f"🔄 Executing task {task.task_id}: {task.agent_type}"

                try:
                    result = await self._execute_task(task, execution_results)

                    if result.success:
                        task.mark_completed(result)
                        completed_tasks.append(task.task_id)
                        execution_results[task.task_id] = result.output
                        yield f"✅ Task {task.task_id} completed successfully"
                    else:
                        task.mark_failed(result.error_message or "Unknown error")
                        failed_tasks.append(task.task_id)
                        yield f"❌ Task {task.task_id} failed: {result.error_message}"

                        # Check if this is a critical task
                        if task.priority == TaskPriority.CRITICAL:
                            plan.mark_failed()
                            yield f"💥 Critical task {task.task_id} failed, stopping execution"
                            break

                except Exception as e:
                    error_msg = f"Exception in task {task.task_id}: {str(e)}"
                    task.mark_failed(error_msg)
                    failed_tasks.append(task.task_id)
                    yield f"💥 Task {task.task_id} failed with exception: {str(e)}"

                    if task.priority == TaskPriority.CRITICAL:
                        plan.mark_failed()
                        break

        # Finalize plan
        if plan.is_completed():
            final_result = self._compile_final_result(execution_results, plan)
            plan.mark_completed(final_result)
            yield f"🎉 Plan {plan.plan_id} completed successfully!"
        else:
            plan.mark_failed()
            yield f"💥 Plan {plan.plan_id} failed"

        # Report final statistics
        progress = plan.get_progress()
        yield f"📊 Final statistics: {progress['completed_tasks']}/{progress['total_tasks']} tasks completed"

    async def _execute_task(
        self, task: Task, previous_results: Dict[str, Any]
    ) -> TaskResult:
        """
        Execute a single task using the appropriate agent.

        Args:
            task: Task to execute
            previous_results: Results from previous tasks

        Returns:
            TaskResult with execution results
        """
        task.mark_started()

        # Get the appropriate agent
        agent = self.agents.get(task.agent_type)
        if not agent:
            return TaskResult(
                success=False,
                output={},
                error_message=f"Unknown agent type: {task.agent_type}",
            )

        # Prepare request for agent
        request = self._prepare_agent_request(task, previous_results)

        try:
            # Execute agent
            response = await agent.execute(request)

            # Convert to TaskResult
            result = TaskResult(
                success=response.success,
                output=response.output,
                error_message=response.error_message,
                execution_time_ms=response.processing_time_ms,
                metadata={
                    "agent_type": task.agent_type,
                    "task_id": task.task_id,
                    "request_id": response.request_id,
                },
            )

            return result

        except Exception as e:
            logger.error(f"Agent execution failed for task {task.task_id}: {e}")
            return TaskResult(success=False, output={}, error_message=str(e))

    def _prepare_agent_request(
        self, task: Task, previous_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Prepare request for agent execution.

        Args:
            task: Task to execute
            previous_results: Results from previous tasks

        Returns:
            Request dictionary for agent
        """
        request = {
            "request_id": f"{task.task_id}_{task.created_at.timestamp()}",
            "task_description": task.task_description,
            "requirements": task.requirements,
            "optimization_goals": task.optimization_goals,
            "context": task.input_data,
            "timestamp": task.created_at.timestamp(),
        }

        # Add previous results as context
        if previous_results:
            request["previous_results"] = previous_results

        # Add agent-specific data
        if task.agent_type == AgentType.OPTIMIZER:
            # Pass generated code to optimizer
            if "task_1" in previous_results:
                request["code"] = previous_results["task_1"].get("code", "")

        elif task.agent_type == AgentType.DEBUGGER:
            # Pass code to debugger
            if "task_1" in previous_results:
                request["code"] = previous_results["task_1"].get("code", "")

        elif task.agent_type == AgentType.EVALUATOR:
            # Pass code to evaluator
            if "task_1" in previous_results:
                request["code"] = previous_results["task_1"].get("code", "")
            # Add optimization results if available
            if "task_2" in previous_results:
                request["optimization_results"] = previous_results["task_2"]

        return request

    def _compile_final_result(
        self, execution_results: Dict[str, Any], plan: TaskPlan
    ) -> Dict[str, Any]:
        """
        Compile final result from all task results.

        Args:
            execution_results: Results from all tasks
            plan: Original task plan

        Returns:
            Compiled final result
        """
        final_result = {
            "plan_id": plan.plan_id,
            "user_request": plan.user_request,
            "task_results": execution_results,
            "execution_summary": {
                "total_tasks": len(plan.tasks),
                "completed_tasks": len(execution_results),
                "success_rate": len(execution_results) / len(plan.tasks)
                if plan.tasks
                else 0,
            },
        }

        # Extract primary result (usually from generator)
        if "task_1" in execution_results:
            generator_result = execution_results["task_1"]
            final_result["primary_result"] = {
                "code": generator_result.get("code", ""),
                "explanation": generator_result.get("explanation", ""),
                "optimization_techniques": generator_result.get(
                    "optimization_techniques", []
                ),
            }

        # Add optimization results if available
        if "task_2" in execution_results:
            optimizer_result = execution_results["task_2"]
            final_result["optimization_results"] = {
                "optimization_suggestions": optimizer_result.get(
                    "optimization_suggestions", []
                ),
                "optimized_code": optimizer_result.get("optimized_code", ""),
                "performance_analysis": optimizer_result.get(
                    "performance_analysis", {}
                ),
            }

        # Add debugging results if available
        for task_id, result in execution_results.items():
            if "debug" in task_id.lower() or "issues" in result:
                final_result["debugging_results"] = {
                    "issues_found": result.get("issues_found", []),
                    "debugged_code": result.get("debugged_code", ""),
                    "code_analysis": result.get("code_analysis", {}),
                }

        # Add evaluation results if available
        for task_id, result in execution_results.items():
            if "eval" in task_id.lower() or "performance" in result:
                final_result["evaluation_results"] = {
                    "performance_analysis": result.get("performance_analysis", {}),
                    "optimization_recommendations": result.get(
                        "optimization_recommendations", []
                    ),
                    "performance_metrics": result.get("performance_metrics", {}),
                }

        return final_result

    async def execute_single_task(self, task: Task) -> TaskResult:
        """
        Execute a single task (for testing or isolated execution).

        Args:
            task: Task to execute

        Returns:
            TaskResult with execution results
        """
        return await self._execute_task(task, {})

    def get_agent_status(self) -> Dict[str, Any]:
        """
        Get status of all agents.

        Returns:
            Dictionary with agent status information
        """
        status = {}
        for agent_type, agent in self.agents.items():
            status[agent_type] = {
                "call_count": agent.call_count,
                "average_processing_time": agent.get_average_processing_time(),
                "total_processing_time": agent.total_processing_time,
            }
        return status

    def reset_agent_stats(self) -> None:
        """Reset statistics for all agents."""
        for agent in self.agents.values():
            agent.reset_stats()
