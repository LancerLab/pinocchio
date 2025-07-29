"""Task executor for executing task plans and managing agent interactions."""

import logging
import time
import traceback
from typing import Any, AsyncGenerator, Dict, List, Optional

from rich.console import Console
from rich.panel import Panel

from pinocchio.config.config_manager import get_config_value
from pinocchio.prompt.manager import PromptManager
from pinocchio.session.context import get_current_session
from pinocchio.task_planning.task_planner import TaskPlanner

from ..agents import DebuggerAgent, EvaluatorAgent, GeneratorAgent, OptimizerAgent
from ..config.settings import Settings
from ..data_models.task_planning import (
    AgentType,
    Task,
    TaskDependency,
    TaskPlan,
    TaskPriority,
    TaskResult,
    TaskStatus,
)
from ..llm.mock_client import MockLLMClient
from ..utils.verbose_logger import LogLevel, get_verbose_logger

logger = logging.getLogger(__name__)


class TaskExecutor:
    """Executor for task plans with intelligent agent management and debug repair loops."""

    def __init__(
        self,
        llm_client: Optional[Any] = None,
        config: Optional[Settings] = None,
        prompt_manager: Optional[PromptManager] = None,
    ):
        """
        Initialize task executor.

        Args:
            llm_client: LLM client for agents (uses MockLLMClient if None)
            config: Configuration settings (uses default if None)
        """
        # Load config
        self.llm_client = llm_client or MockLLMClient(response_delay_ms=200)
        if config is None:
            self.config = Settings()
            try:
                self.config.load_from_file("pinocchio.json")
            except Exception as e:
                logger.warning(f"Could not load config from pinocchio.json: {e}")
        else:
            self.config = config

        # Get verbose flag

        # Get LLM config

        # Initialize agents with provided LLM client or let them create their own
        if llm_client:
            # Use provided LLM client for all agents (for dry-run or custom clients)
            self.agents = {
                AgentType.GENERATOR: GeneratorAgent(llm_client=llm_client),
                AgentType.OPTIMIZER: OptimizerAgent(llm_client=llm_client),
                AgentType.DEBUGGER: DebuggerAgent(llm_client=llm_client),
                AgentType.EVALUATOR: EvaluatorAgent(llm_client=llm_client),
            }
        else:
            # Let each agent manage its own LLM client
            self.agents = {
                AgentType.GENERATOR: GeneratorAgent(),
                AgentType.OPTIMIZER: OptimizerAgent(),
                AgentType.DEBUGGER: DebuggerAgent(),
                AgentType.EVALUATOR: EvaluatorAgent(),
            }

        # Debug repair configuration
        self.debug_repair_enabled = get_config_value(
            self.config, "task_planning.debug_repair.enabled", True
        )
        self.max_repair_attempts = get_config_value(
            self.config, "task_planning.debug_repair.max_repair_attempts", 3
        )
        self.auto_insert_debugger = get_config_value(
            self.config, "task_planning.debug_repair.auto_insert_debugger", True
        )
        self.retry_generator_after_debug = get_config_value(
            self.config, "task_planning.debug_repair.retry_generator_after_debug", True
        )

        # Verbose configuration
        self.verbose_enabled = get_config_value(self.config, "verbose.enabled", True)
        self.verbose_level = get_config_value(self.config, "verbose.level", "detailed")
        self.show_agent_instructions = get_config_value(
            self.config, "verbose.show_agent_instructions", True
        )
        self.show_execution_times = get_config_value(
            self.config, "verbose.show_execution_times", True
        )
        self.show_task_details = get_config_value(
            self.config, "verbose.show_task_details", True
        )
        self.show_progress_updates = get_config_value(
            self.config, "verbose.show_progress_updates", True
        )

        # Track repair attempts
        self.repair_attempts = {}

        self.prompt_manager = prompt_manager or PromptManager()

        logger.info(
            f"TaskExecutor initialized with debug repair loop support (max_attempts={self.max_repair_attempts})"
        )
        if self.verbose_enabled:
            logger.info(f"Verbose output enabled (level: {self.verbose_level})")

    def _trace_collection(
        self,
        obj,
        name,
        context_info=None,
        log_traceback=True,
        depth=0,
        max_preview=5,
        max_depth=2,
    ):
        import traceback

        from pinocchio.utils.verbose_logger import get_verbose_logger

        data = {
            "type": str(type(obj)),
            "is_none": obj is None,
            "depth": depth,
            "context": context_info,
        }
        if isinstance(obj, dict):
            keys = list(obj.keys())
            data["keys_preview"] = keys[:max_preview]
            data["keys_count"] = len(keys)
            data["values_types"] = [str(type(obj[k])) for k in keys[:max_preview]]
            data["preview"] = {
                k: (str(obj[k])[:80] if isinstance(obj[k], str) else str(type(obj[k])))
                for k in keys[:max_preview]
            }
            # Recursively display value types and partial content (only up to max_depth)
            if depth < max_depth:
                data["values_detail"] = {}
                for k in keys[:max_preview]:
                    v = obj[k]
                    if isinstance(v, (dict, list)):
                        data["values_detail"][k] = self._trace_collection(
                            v,
                            f"{name}.{k}",
                            context_info,
                            log_traceback=False,
                            depth=depth + 1,
                            max_preview=max_preview,
                            max_depth=max_depth,
                        )
                    elif isinstance(v, str):
                        data["values_detail"][k] = {
                            "type": "str",
                            "length": len(v),
                            "preview": v[:80],
                        }
                    else:
                        data["values_detail"][k] = {
                            "type": str(type(v)),
                            "repr": repr(v),
                        }
        elif isinstance(obj, list):
            data["length"] = len(obj)
            data["preview_types"] = [str(type(x)) for x in obj[:max_preview]]
            data["preview"] = [
                str(x)[:80] if isinstance(x, str) else str(type(x))
                for x in obj[:max_preview]
            ]
            # Recursively display element content (only up to max_depth)
            if depth < max_depth:
                data["elements_detail"] = []
                for i, x in enumerate(obj[:max_preview]):
                    if isinstance(x, (dict, list)):
                        data["elements_detail"].append(
                            self._trace_collection(
                                x,
                                f"{name}[{i}]",
                                context_info,
                                log_traceback=False,
                                depth=depth + 1,
                                max_preview=max_preview,
                                max_depth=max_depth,
                            )
                        )
                    elif isinstance(x, str):
                        data["elements_detail"].append(
                            {"type": "str", "length": len(x), "preview": x[:80]}
                        )
                    else:
                        data["elements_detail"].append(
                            {"type": str(type(x)), "repr": repr(x)}
                        )
        elif isinstance(obj, str):
            data["length"] = len(obj)
            data["preview"] = obj[:200]
        else:
            data["repr"] = repr(obj)
        if log_traceback:
            data["traceback"] = "".join(traceback.format_stack(limit=8))
        get_verbose_logger().log_coordinator_activity(
            f"[Tracing] Collection access: {name}",
            data=data,
        )
        return data if depth > 0 else obj

    async def execute_plan(self, plan: TaskPlan) -> AsyncGenerator[str, None]:
        """
        Execute a task plan with progress reporting.

        Args:
            plan: Task plan to execute

        Yields:
            Progress messages during execution
        """
        start_time = time.time()

        # Log verbose plan execution start
        verbose_logger = get_verbose_logger()
        verbose_logger.log_coordinator_activity(
            "Plan execution started",
            data={
                "plan_id": plan.plan_id,
                "task_count": len(plan.tasks),
                "agent_types": [str(task.agent_type) for task in plan.tasks],
            },
            session_id=getattr(plan, "session_id", None),
        )

        # Pass SessionLogger reference to plan object
        if hasattr(self, "session_logger") and self.session_logger:
            plan.session_logger = self.session_logger
        plan.mark_started()
        yield f"🤖 Starting execution of plan {plan.plan_id} with {len(plan.tasks)} tasks"

        # Display task plan details
        async for msg in self._display_task_plan_overview(plan):
            yield msg

        # Track execution state
        completed_tasks = []
        failed_tasks = []
        execution_results = {}

        # Execute tasks
        async for msg in self._execute_tasks(
            plan, completed_tasks, failed_tasks, execution_results
        ):
            yield msg

        # Finalize plan
        async for msg in self._finalize_plan(plan, execution_results):
            yield msg

        # Log verbose plan execution completion
        verbose_logger.log_coordinator_activity(
            "Plan execution completed",
            data={
                "plan_id": plan.plan_id,
                "completed_tasks": len(completed_tasks),
                "failed_tasks": len(failed_tasks),
                "total_duration_ms": (time.time() - start_time) * 1000,
            },
            session_id=getattr(plan, "session_id", None),
            duration_ms=(time.time() - start_time) * 1000,
        )

    async def _display_task_plan_overview(
        self, plan: TaskPlan
    ) -> AsyncGenerator[str, None]:
        """Display task plan overview."""
        yield "📋 Task Plan Overview:"
        for i, task in enumerate(plan.tasks, 1):
            agent_emoji = self._get_agent_emoji(task.agent_type)
            priority_emoji = self._get_priority_emoji(task.priority)
            yield f"  {i}. {agent_emoji} {task.agent_type.upper()} ({priority_emoji} {task.priority})"
            yield f"     📝 {task.task_description}"
            yield ""
        # Add end marker for CLI panel buffering
        yield "<<END_TASK_PLAN>>"

    async def _execute_tasks(
        self,
        plan: TaskPlan,
        completed_tasks: list,
        failed_tasks: list,
        execution_results: dict,
    ) -> AsyncGenerator[str, None]:
        """Execute tasks in the plan."""
        # Enhanced termination logic: as long as all tasks are not PENDING, stop
        while (
            any(task.status == TaskStatus.PENDING or task.status == TaskStatus.SKIPPED for task in plan.tasks)
            and not plan.is_failed()
        ):
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
                async for msg in self._execute_single_task_with_verbose(
                    task, execution_results, completed_tasks, failed_tasks, plan=plan
                ):
                    yield msg

                    # Check if we should stop due to critical failure
                    if (
                        task.priority == TaskPriority.CRITICAL
                        and task.status == TaskStatus.FAILED
                    ):
                        plan.mark_failed()
                        yield f"💥 Critical task {task.task_id} failed, stopping execution"
                        return

                yield ""  # Add spacing between tasks

    async def _execute_single_task_with_verbose(
        self,
        task: Task,
        execution_results: dict,
        completed_tasks: list,
        failed_tasks: list,
        plan=None,  # pass plan for dynamic insertion
    ) -> AsyncGenerator[str, None]:
        """Execute a single task with verbose output and support dynamic debugger insertion."""
        start_time = time.time()

        # Log verbose task execution start
        verbose_logger = get_verbose_logger()
        verbose_logger.log_agent_activity(
            str(task.agent_type),
            "Task execution started",
            data={
                "task_id": task.task_id,
                "task_description": task.task_description,
                "priority": str(task.priority),
                "dependencies": [dep.task_id for dep in task.dependencies]
                if task.dependencies
                else [],
            },
            session_id=getattr(plan, "session_id", None) if plan else None,
            step_id=task.task_id,
        )

        agent_emoji = self._get_agent_emoji(task.agent_type)
        yield f"\U0001f504 Executing {agent_emoji} {task.agent_type.upper()} (Task {task.task_id})"

        # Generate task details panel content
        if self.verbose_enabled:
            task_details = self._generate_task_details_panel(task)
            if task_details:
                yield "\U0001f4cb Task Details:"
                for line in task_details:
                    yield line
                yield "<<END_TASK_DETAILS>>"

        # Show progress updates if enabled
        if self.verbose_enabled and self.show_progress_updates:
            yield "   ⏳ Starting task execution..."

        # Record input/output summary to SessionLogger
        session_logger = getattr(plan, "session_logger", None)
        safe_execution_results = execution_results or {}
        # self._trace_collection(
        #     safe_execution_results,
        #     "execution_results",
        #     {"task_id": getattr(task, "task_id", None)},
        # )
        agent_request = self._prepare_agent_request(task, safe_execution_results)
        result = None
        try:
            result = await self._execute_task(task, safe_execution_results)
        finally:
            if session_logger:
                session_logger.log_communication(
                    step_id=task.task_id,
                    agent_type=task.agent_type,
                    request=agent_request,
                    response=result
                    if result
                    else {"error": "No response"},
                )

        try:
            if result and result.success:
                task.mark_completed(result)
                completed_tasks.append(task.task_id)
                execution_results[task.task_id] = result.output

                # Log verbose task completion
                verbose_logger.log_agent_activity(
                    str(task.agent_type),
                    "Task completed successfully",
                    data={
                        "task_id": task.task_id,
                        "output_keys": list(result.output.keys())
                        if isinstance(result.output, dict)
                        else [],
                        "execution_time_ms": result.execution_time_ms,
                        "processing_time_ms": getattr(
                            result, "processing_time_ms", None
                        ),
                    },
                    session_id=getattr(plan, "session_id", None) if plan else None,
                    step_id=task.task_id,
                    duration_ms=(time.time() - start_time) * 1000,
                )

                # Show success summary
                yield f"✅ {agent_emoji} {task.agent_type.upper()} completed successfully"

                # Show output summary if verbose is enabled
                if self.verbose_enabled and result.output:
                    output_summary = self._get_output_summary(
                        task.agent_type, result.output
                    )
                    if output_summary:
                        yield f"   📊 {output_summary}"

                # Show execution time if verbose is enabled
                if (
                    self.verbose_enabled
                    and self.show_execution_times
                    and result.execution_time_ms
                ):
                    yield f"   ⏱️ Execution time: {result.execution_time_ms}ms"

                # Show additional verbose information
                if self.verbose_enabled and self.verbose_level == "detailed":
                    if hasattr(result, "processing_time_ms"):
                        yield f"   🧠 Processing time: {result.processing_time_ms}ms"
                    if hasattr(result, "llm_calls"):
                        yield f"   🤖 LLM calls: {result.llm_calls}"

            else:
                # Handle case where result is None or result.success is False
                if result is None:
                    msg = "Agent execution returned None"
                    error_message = "Agent execution returned None"
                else:
                    msg = result.error_message or "Unknown error"
                    error_message = result.error_message

                task.mark_failed(msg)
                failed_tasks.append(task.task_id)

                # Log verbose task failure
                verbose_logger.log(
                    LogLevel.ERROR,
                    f"agent:{str(task.agent_type)}",
                    "Task failed",
                    data={
                        "task_id": task.task_id,
                        "error_message": error_message,
                        "error_details": getattr(result, "error_details", None) if result else None,
                    },
                    session_id=getattr(plan, "session_id", None) if plan else None,
                    step_id=task.task_id,
                    duration_ms=(time.time() - start_time) * 1000,
                )

                yield f"❌ {agent_emoji} {task.agent_type.upper()} failed: {error_message}"

                # Show detailed error information if verbose is enabled
                if self.verbose_enabled and self.verbose_level == "detailed":
                    if hasattr(result, "error_details"):
                        yield f"   🔍 Error details: {result.error_details}"
            
            # Handle Impacts on other tasks
            if task.impacts:
                for impact in task.impacts:
                    impacted_task = next(
                        (t for t in plan.tasks if t.task_id == impact.task_id), None
                    )
                    if impacted_task:
                        # if not satify conditon, skip impact
                        if impact["condition"] and not impact["condition"] != task.status.value:
                            continue
                        
                        # Can support more actions
                        elif impact.action == "skip":
                            impacted_task.mark_skipped()
                            yield f"   ⏭️ Skipped {impacted_task.task_id} due to impact from {task.task_id}"

        except Exception as e:
            error_msg = f"Exception in task {task.task_id}: {str(e)}"
            task.mark_failed(error_msg)
            failed_tasks.append(task.task_id)

            # Log verbose task exception
            verbose_logger.log(
                LogLevel.ERROR,
                f"agent:{str(task.agent_type)}",
                "Task failed with exception",
                data={
                    "task_id": task.task_id,
                    "error": str(e),
                    "exception_type": type(e).__name__,
                },
                session_id=getattr(plan, "session_id", None) if plan else None,
                step_id=task.task_id,
                duration_ms=(time.time() - start_time) * 1000,
            )

            yield f"💥 {agent_emoji} {task.agent_type.upper()} failed with exception: {str(e)}"

            # Show detailed exception information if verbose is enabled
            if self.verbose_enabled and self.verbose_level == "detailed":
                yield f"   📋 Exception traceback: {traceback.format_exc()}"

        # Handle dynamic debugger insertion
        async for msg in self._handle_dynamic_debugger_insertion(task, result, plan):
            yield msg

        # After each task execution, pretty print the latest code
        session = get_current_session()
        if session and session.get_latest_code():
            code = session.get_latest_code()
            panel = Panel(
                code,
                title="[bold green]Current Latest Code[/bold green]",
                border_style="green",
                padding=(1, 2),
            )
            # Directly yield rich's ANSI rendered string
            console = Console()
            from io import StringIO

            buf = StringIO()
            console.file = buf
            console.print(panel)
            yield buf.getvalue()

    def _generate_task_details_panel(self, task: Task) -> List[str]:
        """Generate task details panel content."""
        details = []

        # Add task description
        details.append(f"   📋 Description: {task.task_description}")

        # Show detailed task information if enabled
        if self.show_task_details:
            if task.priority:
                priority_emoji = self._get_priority_emoji(task.priority)
                details.append(f"   🎯 Priority: {priority_emoji} {task.priority}")

            if task.dependencies:
                deps = ", ".join([dep.task_id for dep in task.dependencies])
                details.append(f"   🔗 Dependencies: {deps}")

            if task.requirements:
                reqs = ", ".join([f"{k}={v}" for k, v in task.requirements.items()])
                details.append(f"   📋 Requirements: {reqs}")

        # Show detailed instructions if enabled
        if self.show_agent_instructions and task.input_data.get("instruction"):
            details.append("   💡 Detailed Instruction:")
            instruction_lines = task.input_data["instruction"].split("\n")
            for line in instruction_lines:
                if line.strip():
                    details.append(f"      {line}")
            details.append("")

        return details

    async def _handle_dynamic_debugger_insertion(
        self, task: Task, result, plan
    ) -> AsyncGenerator[str, None]:
        """Handle dynamic debugger insertion logic."""
        # --- Dynamic debugger insertion logic for generator/optimizer failures ---
        if (
            plan is not None
            and self.debug_repair_enabled
            and self.auto_insert_debugger
            and task.agent_type in [AgentType.GENERATOR, AgentType.OPTIMIZER]
        ):
            context = task.input_data
            # Check if error detected (failure, error_message, or suspicious output)
            error_detected = (
                task.status == TaskStatus.FAILED
                or (hasattr(result, "error_message") and result.error_message)
                or (
                    hasattr(result, "output")
                    and isinstance(result.output, dict)
                    and (
                        "error" in str(result.output).lower()
                        or "exception" in str(result.output).lower()
                        or "failed" in str(result.output).lower()
                    )
                )
            )

            # Check if this is a retry task (already part of a repair cycle)
            is_retry_task = task.input_data.get("retry_after_debug", False)

            # Only track repair attempts for original tasks, not retry tasks
            if not is_retry_task:
                # Check repair attempts limit for original task
                repair_key = f"{task.task_id}_repair"
                current_repair_attempts = self.repair_attempts.get(repair_key, 0)

                # Only insert if not already a pending debugger for this task and within limits
                already_has_debugger = any(
                    t.agent_type == AgentType.DEBUGGER
                    and any(dep.task_id == task.task_id for dep in t.dependencies)
                    and t.status == TaskStatus.PENDING
                    for t in plan.tasks
                )

                if (
                    error_detected
                    and not already_has_debugger
                    and current_repair_attempts < self.max_repair_attempts
                ):
                    # Show verbose debugger insertion information
                    if self.verbose_enabled:
                        yield f"   🔧 Error detected in {task.task_id}, initiating debug repair cycle"
                        yield f"   📊 Repair attempt {current_repair_attempts + 1}/{self.max_repair_attempts}"

                    # Create context for debugger instruction
                    # prompt/context construction is handled internally by PromptManager/agent

                    planner = TaskPlanner()  # Assuming TaskPlanner is available
                    debugger_instruction = planner._build_debugger_instruction(context)

                    # Create debugger task
                    new_debugger_task = Task(
                        task_id=f"task_{len(plan.tasks)+1}",
                        agent_type=AgentType.DEBUGGER,
                        task_description=(
                            f"Dynamically inserted: Analyze and fix code error "
                            f"(attempt {current_repair_attempts + 1}/{self.max_repair_attempts})"
                        ),
                        requirements={"error_handling": True},
                        priority=TaskPriority.CRITICAL,
                        dependencies=[
                            TaskDependency(
                                task_id=task.task_id, dependency_type="required"
                            )
                        ],
                        input_data={
                            "error_handling": True,
                            "instruction": debugger_instruction,
                            "repair_attempt": current_repair_attempts + 1,
                            "max_repair_attempts": self.max_repair_attempts,
                            "original_task_id": task.task_id,
                        },
                    )
                    plan.tasks.append(new_debugger_task)

                    # Increment repair attempts counter
                    self.repair_attempts[repair_key] = current_repair_attempts + 1

                    yield (
                        f"🆕 Dynamically inserted DEBUGGER after {task.task_id} "
                        f"due to detected error (attempt {current_repair_attempts + 1}/{self.max_repair_attempts})"
                    )

                elif current_repair_attempts >= self.max_repair_attempts:
                    yield f"⚠️ Max repair attempts ({self.max_repair_attempts}) reached for {task.task_id}, stopping debug repair loop"

        # --- Dynamic insertion after debugger completion (bug detection) ---
        if (
            plan is not None
            and self.debug_repair_enabled
            and self.auto_insert_debugger
            and task.agent_type == AgentType.DEBUGGER
        ):
            context = task.input_data
            # Check if debugger found bugs that need additional repair
            bugs_detected = (
                (
                    hasattr(result, "output")
                    and isinstance(result.output, dict)
                    and (
                        "error" in str(result.output).lower()
                        or "bug" in str(result.output).lower()
                        or "issue" in str(result.output).lower()
                        or "failed" in str(result.output).lower()
                        or "compilation_error" in str(result.output).lower()
                    )
                )
                or (hasattr(result, "error_message") and result.error_message)
                or task.status == TaskStatus.FAILED
            )

            if bugs_detected and self.retry_generator_after_debug:
                # Show verbose retry information
                if self.verbose_enabled:
                    yield "   🔍 Debugger found issues, initiating retry cycle"

                # Get the original task that failed
                original_task_id = task.input_data.get("original_task_id")
                if original_task_id:
                    # Find the original task
                    original_task = next(
                        (t for t in plan.tasks if t.task_id == original_task_id), None
                    )

                    if original_task:
                        # Create retry generator task
                        # prompt/context construction is handled internally by PromptManager/agent

                        planner = TaskPlanner()  # Assuming TaskPlanner is available
                        generator_instruction = planner._build_generator_instruction(
                            context
                        )

                        new_generator_task = Task(
                            task_id=f"task_{len(plan.tasks)+1}",
                            agent_type=AgentType.GENERATOR,
                            task_description=f"Retry generation after debug fix (attempt {task.input_data.get('repair_attempt', 1)})",
                            requirements=original_task.requirements,
                            priority=TaskPriority.CRITICAL,
                            dependencies=[
                                TaskDependency(
                                    task_id=task.task_id, dependency_type="required"
                                )
                            ],
                            input_data={
                                "user_request": original_task.input_data.get(
                                    "user_request", ""
                                ),
                                "instruction": generator_instruction,
                                "retry_after_debug": True,
                                "repair_attempt": task.input_data.get(
                                    "repair_attempt", 1
                                ),
                            },
                        )
                        plan.tasks.append(new_generator_task)

                        yield "🔄 Added retry GENERATOR after debugger completion"

                        # Add another debugger for the retry
                        debugger_instruction = planner._build_debugger_instruction(
                            context
                        )

                        new_debugger_task = Task(
                            task_id=f"task_{len(plan.tasks)+1}",
                            agent_type=AgentType.DEBUGGER,
                            task_description=f"Verify retry generation (attempt {task.input_data.get('repair_attempt', 1)})",
                            requirements={"error_handling": True},
                            priority=TaskPriority.CRITICAL,
                            dependencies=[
                                TaskDependency(
                                    task_id=new_generator_task.task_id,
                                    dependency_type="required",
                                )
                            ],
                            input_data={
                                "error_handling": True,
                                "instruction": debugger_instruction,
                                "repair_attempt": task.input_data.get(
                                    "repair_attempt", 1
                                ),
                                "max_repair_attempts": self.max_repair_attempts,
                                "original_task_id": original_task_id,
                            },
                        )
                        plan.tasks.append(new_debugger_task)

                        yield "🆕 Added verification DEBUGGER for retry generation"

    async def _finalize_plan(
        self, plan: TaskPlan, execution_results: dict
    ) -> AsyncGenerator[str, None]:
        """Finalize plan and show summary."""
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

        # Show agent participation summary
        yield "🤖 Agent Participation Summary:"
        agent_stats = self._calculate_agent_stats(plan)

        for agent_type, stats in agent_stats.items():
            agent_emoji = self._get_agent_emoji(agent_type)
            success_rate = (
                (stats["completed"] / stats["total"]) * 100 if stats["total"] > 0 else 0
            )
            yield f"   {agent_emoji} {agent_type.upper()}: {stats['completed']}/{stats['total']} ({success_rate:.1f}% success)"

    def _calculate_agent_stats(self, plan: TaskPlan) -> dict:
        """Calculate agent participation statistics."""
        agent_stats = {}
        for task in plan.tasks:
            agent_type = task.agent_type
            if agent_type not in agent_stats:
                agent_stats[agent_type] = {"total": 0, "completed": 0, "failed": 0}
            agent_stats[agent_type]["total"] += 1
            if task.status == TaskStatus.COMPLETED:
                agent_stats[agent_type]["completed"] += 1
            elif task.status == TaskStatus.FAILED:
                agent_stats[agent_type]["failed"] += 1
        return agent_stats

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
        
        # return if taks is skipped
        if task.status == TaskStatus.SKIPPED:
            result = TaskResult(
                success=False,
                output=None,
                error_message="SKIPPED due to previous task failure",
                execution_time_ms=0,
                metadata={
                    "agent_type": task.agent_type,
                    "task_id": task.task_id
                },
            )

            return result
        
        task.mark_started()

        # Get the appropriate agent
        agent = self.agents.get(task.agent_type)
        if not agent:
            return TaskResult(
                success=False,
                output={},
                error_message=f"Unknown agent type: {task.agent_type}",
            )

        # prompt/context construction is handled internally by PromptManager/agent

        try:
            # Build request
            safe_previous_results = previous_results or {}
            # self._trace_collection(
            #     safe_previous_results,
            #     "previous_results",
            #     {"task_id": getattr(task, "task_id", None)},
            # )
            request = self._prepare_agent_request(task, safe_previous_results)
            # Tracing: log request code
            get_verbose_logger().log_agent_activity(
                str(task.agent_type),
                "[Tracing] _execute_task: request code",
                data={
                    "task_id": getattr(task, "task_id", None),
                    "request_code_type": str(type(request.get("code"))),
                    "request_code_length": len(request.get("code"))
                    if request.get("code")
                    else 0,
                    "request_code_preview": request.get("code", "")[:200],
                },
                session_id=getattr(task, "session_id", None),
                step_id=getattr(task, "task_id", None),
            )
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
        Build agent request using PromptManager for context-aware prompt and request structure.
        """
        # Extract code from previous results if available
        code = None
        safe_previous_results = previous_results or {}
        # self._trace_collection(
        #     safe_previous_results,
        #     "previous_results",
        #     {"task_id": getattr(task, "task_id", None)},
        #     log_traceback=True,
        #     depth=0,
        # )
        if safe_previous_results:
            for v_idx, v in enumerate(safe_previous_results.values()):
                self._trace_collection(
                    v,
                    f"previous_results.value[{v_idx}]",
                    {"task_id": getattr(task, "task_id", None)},
                    log_traceback=False,
                    depth=1,
                )
                if v is None:
                    from pinocchio.utils.verbose_logger import get_verbose_logger

                    get_verbose_logger().log_coordinator_activity(
                        f"[Tracing][ERROR] previous_results.value[{v_idx}] is NoneType!",
                        data={
                            "task_id": getattr(task, "task_id", None),
                        },
                    )
                    continue
                if (
                    isinstance(v, dict)
                    and v.get("output")
                    and isinstance(v["output"], dict)
                    and v["output"].get("code")
                ):
                    code = v["output"]["code"]
                    break
        # Tracing: log code extraction
        from pinocchio.utils.verbose_logger import get_verbose_logger

        get_verbose_logger().log_agent_activity(
            str(task.agent_type),
            "[Tracing] _prepare_agent_request: code extraction",
            data={
                "task_id": getattr(task, "task_id", None),
                "code_type": str(type(code)),
                "code_length": len(code) if code else 0,
                "code_preview": code[:200] if code else None,
                "previous_results_keys": list(safe_previous_results.keys()),
            },
            session_id=getattr(task, "session_id", None),
            step_id=getattr(task, "task_id", None),
        )
        # Use PromptManager to build request
        session_obj = get_current_session()
        request = self.prompt_manager.create_context_aware_request(
            agent_type=task.agent_type,
            task_description=task.task_description,
            session_id=getattr(task, "session_id", None),
            code=None,  # Provided by session_obj automatically
            keywords=None,
            context=getattr(task, "input_data", None) or {},
            previous_results=safe_previous_results,
            session_obj=session_obj,
        )
        # Tracing: log final request code
        get_verbose_logger().log_agent_activity(
            str(task.agent_type),
            "[Tracing] _prepare_agent_request: final request code",
            data={
                "task_id": getattr(task, "task_id", None),
                "request_code_type": str(type(request.get("code"))),
                "request_code_length": len(request.get("code"))
                if request.get("code")
                else 0,
                "request_code_preview": request.get("code", "")[:200],
            },
            session_id=getattr(task, "session_id", None),
            step_id=getattr(task, "task_id", None),
        )
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

    def _get_agent_emoji(self, agent_type: AgentType) -> str:
        """Get emoji for agent type."""
        emoji_map = {
            AgentType.GENERATOR: "⚡",
            AgentType.OPTIMIZER: "🚀",
            AgentType.DEBUGGER: "🔧",
            AgentType.EVALUATOR: "📊",
        }
        return emoji_map.get(agent_type, "🤖")

    def _get_priority_emoji(self, priority: TaskPriority) -> str:
        """Get emoji for task priority."""
        emoji_map = {
            TaskPriority.CRITICAL: "🔴",
            TaskPriority.HIGH: "🟡",
            TaskPriority.MEDIUM: "🟢",
            TaskPriority.LOW: "🔵",
        }
        return emoji_map.get(priority, "⚪")

    def _get_output_summary(self, agent_type: AgentType, output: Dict[str, Any]) -> str:
        """Get a summary of agent output."""
        if agent_type == AgentType.GENERATOR:
            if "code" in output:
                code_lines = len(output["code"].split("\n"))
                return f"Generated {code_lines} lines of code"
            return "Code generation completed"

        elif agent_type == AgentType.OPTIMIZER:
            if "optimization_suggestions" in output:
                suggestions = len(output["optimization_suggestions"])
                return f"Provided {suggestions} optimization suggestions"
            return "Optimization analysis completed"

        elif agent_type == AgentType.DEBUGGER:
            if "issues_found" in output:
                issues = len(output["issues_found"])
                return f"Found {issues} issues"
            return "Debugging analysis completed"

        elif agent_type == AgentType.EVALUATOR:
            if "performance_metrics" in output:
                return "Performance evaluation completed"
            return "Evaluation completed"

        return "Task completed"
