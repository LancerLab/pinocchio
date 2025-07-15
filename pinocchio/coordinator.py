"""Coordinator - The central orchestrator for Pinocchio multi-agent system."""

import logging
import shutil
import time
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional

from pinocchio.session.context import get_current_session

from .config.config_manager import ConfigManager, get_config_value
from .session_logger import SessionLogger
from .task_planning import TaskExecutor, TaskPlanner
from .utils.verbose_logger import LogLevel, get_verbose_logger

logger = logging.getLogger(__name__)


class Coordinator:
    """Central coordinator for managing multi-agent task planning and execution."""

    def __init__(
        self,
        llm_client: Optional[Any] = None,
        sessions_dir: str = "./sessions",
        mode: str = "production",
    ):
        """
        Initialize coordinator.

        Args:
            llm_client: LLM client instance
            sessions_dir: Directory to store session files
            mode: CLI mode (development/production). Development mode raises errors, production mode allows fallback
        """
        # Load configuration
        self.config_manager = ConfigManager("pinocchio.json")
        self.config = self.config_manager.config

        # Store mode for fallback behavior control
        self.mode = mode
        logger.info(f"Coordinator initialized in {mode} mode")

        # Initialize task planning components
        # Get planner-specific LLM config
        planner_llm_config = self.config_manager.get_agent_llm_config("planner")
        planner_llm_client = self._create_llm_client(planner_llm_config)

        self.task_planner = TaskPlanner(
            planner_llm_client, mode=mode, config=self.config
        )

        # Session management
        self.sessions_dir = sessions_dir
        self.session_logger: Optional[SessionLogger] = None

        # Plugin management
        self.plugin_manager = None
        if get_config_value(self.config, "plugins.enabled", False):
            self._initialize_plugin_manager()

        # Initialize memory and knowledge systems
        self.memory_manager = None
        self.knowledge_manager = None
        self.prompt_manager = None
        self._initialize_memory_knowledge_systems()

        # TaskExecutor needs a unified PromptManager (order adjusted to after prompt_manager initialization)
        self.task_executor = TaskExecutor(
            llm_client, self.config, prompt_manager=self.prompt_manager
        )

        # Statistics
        self.total_sessions = 0
        self.successful_sessions = 0

        logger.info("Coordinator initialized with task planning system")

    def _create_llm_client(self, llm_config):
        """Create LLM client based on configuration."""
        try:
            if llm_config.provider.value == "mock":
                from .llm.mock_client import MockLLMClient

                return MockLLMClient(response_delay_ms=50, failure_rate=0.0)
            elif llm_config.provider.value == "custom":
                from .llm.custom_llm_client import CustomLLMClient

                return CustomLLMClient(llm_config, verbose=False)
            else:
                # Fallback to mock for unknown providers
                from .llm.mock_client import MockLLMClient

                return MockLLMClient(response_delay_ms=50, failure_rate=0.0)
        except Exception as e:
            logger.warning(
                f"Failed to create LLM client for {llm_config.provider}: {e}"
            )
            # Fallback to mock
            from .llm.mock_client import MockLLMClient

            return MockLLMClient(response_delay_ms=50, failure_rate=0.0)

    def _initialize_plugin_manager(self) -> None:
        """Initialize the plugin manager."""
        from .plugins import (
            CustomAgentPlugin,
            CustomPromptPlugin,
            CustomWorkflowPlugin,
            PluginManager,
        )

        self.plugin_manager = PluginManager()

        # Register plugins if enabled
        if get_config_value(self.config, "plugins.enabled", False):
            try:
                # Register prompt plugin
                prompt_plugin = CustomPromptPlugin()
                prompt_config = get_config_value(
                    self.config, "plugins.plugin_configs.cuda_prompt_plugin", {}
                )
                self.plugin_manager.register_plugin(prompt_plugin, prompt_config)
                logger.info("Registered CUDA prompt plugin")

                # Register workflow plugin
                workflow_plugin = CustomWorkflowPlugin()
                workflow_config = get_config_value(
                    self.config, "plugins.plugin_configs.json_workflow_plugin", {}
                )
                self.plugin_manager.register_plugin(workflow_plugin, workflow_config)
                logger.info("Registered JSON workflow plugin")

                # Register agent plugin
                agent_plugin = CustomAgentPlugin()
                agent_config = get_config_value(
                    self.config, "plugins.plugin_configs.custom_agent_plugin", {}
                )
                self.plugin_manager.register_plugin(agent_plugin, agent_config)
                logger.info("Registered custom agent plugin")

            except Exception as e:
                logger.error(f"Failed to initialize plugins: {e}")
                self.plugin_manager = None

    def _initialize_memory_knowledge_systems(self) -> None:
        """Initialize memory, knowledge, and prompt management systems."""
        try:
            # Initialize memory manager
            from .memory import MemoryManager

            memories_path = get_config_value(
                self.config, "storage.memories_path", "./memories"
            )
            self.memory_manager = MemoryManager(store_dir=memories_path)
            logger.info("Memory manager initialized")

            # Initialize knowledge manager
            from .knowledge import KnowledgeManager

            knowledge_path = get_config_value(
                self.config, "storage.knowledge_path", "./knowledge"
            )
            self.knowledge_manager = KnowledgeManager(storage_path=knowledge_path)

            # Add CUDA knowledge base
            self.knowledge_manager.add_cuda_knowledge_base()
            logger.info("Knowledge manager initialized with CUDA knowledge base")

            # Initialize prompt manager
            from .prompt import PromptManager

            prompt_storage_path = get_config_value(
                self.config, "storage.prompt_storage_path", "./prompt_storage"
            )
            self.prompt_manager = PromptManager(storage_path=prompt_storage_path)

            # Integrate memory and knowledge with prompt manager
            self.prompt_manager.integrate_memory_and_knowledge(
                memory_manager=self.memory_manager,
                knowledge_manager=self.knowledge_manager,
            )
            logger.info(
                "Prompt manager initialized with memory and knowledge integration"
            )

        except Exception as e:
            logger.error(f"Failed to initialize memory/knowledge systems: {e}")
            self.memory_manager = None
            self.knowledge_manager = None
            self.prompt_manager = None

    async def process_user_request(self, user_prompt: str) -> AsyncGenerator[str, None]:
        """
        Process user request through intelligent task planning and execution.

        Args:
            user_prompt: User's input prompt

        Yields:
            Progress messages during processing
        """
        self.total_sessions += 1
        self._final_result = None

        try:
            # Initialize session logger (for logging only)
            async for msg in self._initialize_session_logger(user_prompt):
                yield msg

            # Create task plan
            async for msg in self._create_task_plan(user_prompt):
                yield msg

            # Execute task plan
            async for msg in self._execute_task_plan():
                yield msg

            # Process results
            async for msg in self._process_results():
                yield msg

            # Save session
            async for msg in self._save_session():
                yield msg

        except Exception as e:
            async for msg in self._handle_error(e):
                yield msg

    async def _initialize_session_logger(
        self, user_prompt: str
    ) -> AsyncGenerator[str, None]:
        """Initialize new session."""
        start_time = time.time()

        self.session_logger = SessionLogger(user_prompt, self.sessions_dir)

        # Log verbose session initialization
        verbose_logger = get_verbose_logger()
        verbose_logger.log_coordinator_activity(
            "Session initialized",
            data={
                "user_prompt": user_prompt,
                "session_id": self.session_logger.session_id,
                "sessions_dir": self.sessions_dir,
            },
            session_id=self.session_logger.session_id,
            duration_ms=(time.time() - start_time) * 1000,
        )

        yield self.session_logger.log_summary("Session started")

    async def _create_task_plan(self, user_prompt: str) -> AsyncGenerator[str, None]:
        start_time = time.time()
        verbose_logger = get_verbose_logger()
        yield self.session_logger.log_summary("🤖 Creating intelligent task plan...")
        try:
            plan = None
            workflow_used = False
            verbose_logger.log(
                LogLevel.INFO,
                "coordinator",
                "Start task plan creation",
                {
                    "user_prompt": user_prompt,
                    "session_id": self.session_logger.session_id,
                },
            )
            if (
                get_config_value(self.config, "workflow.use_plugin", False)
                and self.plugin_manager is not None
            ):
                try:
                    verbose_logger.log(
                        LogLevel.INFO,
                        "coordinator",
                        "Trying workflow plugin",
                        {"plugin_manager": str(self.plugin_manager)},
                    )
                    plan = await self._create_workflow_plan(user_prompt)
                    workflow_used = True
                    verbose_logger.log(
                        LogLevel.INFO,
                        "coordinator",
                        "Workflow plugin used for plan",
                        {"plan": str(plan)},
                    )
                    yield self.session_logger.log_summary(
                        "🔌 Using custom workflow plugin"
                    )
                except Exception as e:
                    logger.warning(f"Workflow plugin failed: {e}")
                    verbose_logger.log(
                        LogLevel.WARNING,
                        "coordinator",
                        "Workflow plugin failed",
                        {"error": str(e)},
                    )
                    yield self.session_logger.log_summary(
                        f"⚠️ Workflow plugin failed: {str(e)}"
                    )
                    if not get_config_value(
                        self.config, "workflow.fallback_to_task_planning", True
                    ):
                        raise
                    else:
                        verbose_logger.log(
                            LogLevel.INFO,
                            "coordinator",
                            "Falling back to task planning",
                        )
                        yield self.session_logger.log_summary(
                            "🔄 Falling back to task planning"
                        )
            if plan is None:
                verbose_logger.log(
                    LogLevel.INFO, "coordinator", "Using task planning system"
                )
                try:
                    session = get_current_session()
                    plan = await self.task_planner.create_task_plan(
                        user_prompt, session_id=session.session_id if session else None
                    )
                    verbose_logger.log(
                        LogLevel.INFO,
                        "coordinator",
                        "Task plan created by task_planner",
                        {"plan_id": plan.plan_id, "task_count": len(plan.tasks)},
                    )
                    yield self.session_logger.log_summary(
                        "📋 Using task planning system"
                    )
                except Exception as e:
                    verbose_logger.log(
                        LogLevel.ERROR,
                        "coordinator",
                        "Task planning failed",
                        {"error": str(e)},
                    )
                    if self.mode == "development":
                        # In development mode, propagate the error
                        logger.error(f"Task planning failed in development mode: {e}")
                        raise
                    else:
                        # In production mode, provide fallback message
                        yield self.session_logger.log_summary(
                            f"⚠️ Task planning failed, using fallback: {str(e)}"
                        )
                        # Create a minimal fallback plan
                        from .data_models.task_planning import (
                            AgentType,
                            Task,
                            TaskPlan,
                            TaskPriority,
                        )

                        plan = TaskPlan(
                            plan_id=f"fallback_plan_{self.session_logger.session_id}",
                            user_request=user_prompt,
                            tasks=[
                                Task(
                                    task_id="fallback_task",
                                    agent_type=AgentType.GENERATOR,
                                    task_description=f"Generate code for: {user_prompt}",
                                    requirements={},
                                    optimization_goals=["performance"],
                                    priority=TaskPriority.CRITICAL,
                                    dependencies=[],
                                    input_data={"user_request": user_prompt},
                                )
                            ],
                            context={},
                        )
            self.current_plan = plan
            validation = self.task_planner.validate_plan(self.current_plan)
            verbose_logger.log(
                LogLevel.INFO,
                "coordinator",
                "Task plan validated",
                {"validation": validation},
            )
            if validation["valid"]:
                workflow_type = "workflow plugin" if workflow_used else "task planning"
                yield self.session_logger.log_summary(
                    f"✅ Task plan created using {workflow_type}: {len(self.current_plan.tasks)} tasks"
                )
                if validation["warnings"]:
                    for warning in validation["warnings"]:
                        yield self.session_logger.log_summary(f"⚠️ {warning}")
            else:
                error_msg = f"Invalid task plan: {', '.join(validation['errors'])}"
                verbose_logger.log(
                    LogLevel.ERROR,
                    "coordinator",
                    "Invalid task plan",
                    {"errors": validation["errors"]},
                )
                yield self.session_logger.log_summary(f"❌ {error_msg}")
                raise ValueError(error_msg)
        except Exception as e:
            verbose_logger.log(
                LogLevel.ERROR, "coordinator", "Task planning failed", {"error": str(e)}
            )
            yield self.session_logger.log_summary(f"💥 Task planning failed: {str(e)}")
            logger.exception("Task planning failed")
            raise

    async def _create_workflow_plan(self, user_prompt: str):
        """Create task plan using workflow plugin."""
        # Get workflow plugin configuration
        workflow_plugin_name = get_config_value(
            self.config, "plugins.active_plugins.workflow"
        )
        if not workflow_plugin_name:
            raise ValueError("No workflow plugin configured")

        # Get workflow plugin
        workflow_plugin = self.plugin_manager.get_plugin(workflow_plugin_name)
        if not workflow_plugin:
            raise ValueError(f"Workflow plugin not found: {workflow_plugin_name}")

        # Get workflow configuration
        default_workflow = get_config_value(
            self.config, "workflow.default_workflow", "cuda_development"
        )
        workflow_config = {
            "workflow_name": default_workflow,
            "session_id": self.session_logger.session_id,
        }

        # Create workflow plan
        plan = workflow_plugin.execute(
            "create_workflow", user_request=user_prompt, config=workflow_config
        )

        # Set session attributes
        plan.session_id = self.session_logger.session_id
        plan.session_logger = self.session_logger

        return plan

    async def _execute_task_plan(self) -> AsyncGenerator[str, None]:
        """Execute the task plan with progress reporting."""
        start_time = time.time()

        if not hasattr(self, "current_plan"):
            yield self.session_logger.log_summary("❌ No task plan available")
            return

        yield self.session_logger.log_summary("🚀 Starting task execution...")

        # Log verbose execution start
        verbose_logger = get_verbose_logger()
        verbose_logger.log_coordinator_activity(
            "Task execution started",
            data={
                "task_count": len(self.current_plan.tasks),
                "plan_id": getattr(self.current_plan, "id", None),
            },
            session_id=self.session_logger.session_id,
        )

        # Pass SessionLogger to TaskExecutor
        self.task_executor.session_logger = self.session_logger

        # Execute plan and collect results
        async for progress_msg in self.task_executor.execute_plan(self.current_plan):
            yield self.session_logger.log_summary(progress_msg)

            # Store final result if plan completed
            if "🎉 Plan" in progress_msg and "completed successfully" in progress_msg:
                if hasattr(self.current_plan, "final_result"):
                    self._final_result = self.current_plan.final_result

                # Log verbose execution completion
                verbose_logger.log_coordinator_activity(
                    "Task execution completed",
                    data={
                        "final_result": self._final_result,
                        "plan_id": getattr(self.current_plan, "id", None),
                    },
                    session_id=self.session_logger.session_id,
                    duration_ms=(time.time() - start_time) * 1000,
                )

    async def _process_results(self) -> AsyncGenerator[str, None]:
        """Process and display final results."""
        session = get_current_session()
        if self._final_result and self.current_plan.is_completed():
            async for msg in self._display_success_results(self._final_result):
                yield msg
            if session:
                session.complete_session()
            self.successful_sessions += 1
            yield self.session_logger.log_summary("Session completed successfully")
        else:
            if session:
                session.fail_session()
            yield self.session_logger.log_summary("Session failed")

    async def _display_success_results(
        self, final_result: Dict[str, Any]
    ) -> AsyncGenerator[str, None]:
        """Display successful results with enhanced information."""
        # Display primary result (code generation)
        if "primary_result" in final_result:
            primary = final_result["primary_result"]
            code = primary.get("code", "")
            if code:
                yield f"\n🎉 Code generation completed!\n\n```choreo\n{code}\n```"

                explanation = primary.get("explanation", "")
                if explanation:
                    yield f"\n📋 Explanation: {explanation}"

                optimization_techniques = primary.get("optimization_techniques", [])
                if optimization_techniques:
                    yield f"\n⚡ Optimizations applied: {', '.join(optimization_techniques)}"

        # Display optimization results if available
        if "optimization_results" in final_result:
            opt_results = final_result["optimization_results"]
            suggestions = opt_results.get("optimization_suggestions", [])
            if suggestions:
                yield "\n🔧 Optimization suggestions:"
                for i, suggestion in enumerate(suggestions[:3], 1):  # Show top 3
                    yield f"  {i}. {suggestion.get('description', 'N/A')}"

        # Display debugging results if available
        if "debugging_results" in final_result:
            debug_results = final_result["debugging_results"]
            issues = debug_results.get("issues_found", [])
            if issues:
                yield f"\n🐛 Issues found: {len(issues)} issues detected"
                for i, issue in enumerate(issues[:2], 1):  # Show top 2
                    yield f"  {i}. {issue.get('description', 'N/A')}"

        # Display evaluation results if available
        if "evaluation_results" in final_result:
            eval_results = final_result["evaluation_results"]
            performance = eval_results.get("performance_analysis", {})
            if performance:
                yield "\n📊 Performance analysis:"
                yield f"  Time complexity: {performance.get('time_complexity', 'N/A')}"
                yield f"  Memory efficiency: {performance.get('memory_efficiency', 'N/A')}"

        # Display execution summary
        if "execution_summary" in final_result:
            summary = final_result["execution_summary"]
            yield "\n📈 Execution summary:"
            yield f"  Tasks completed: {summary.get('completed_tasks', 0)}/{summary.get('total_tasks', 0)}"
            yield f"  Success rate: {summary.get('success_rate', 0):.1%}"

    async def _save_session(self) -> AsyncGenerator[str, None]:
        """Save session to file."""
        try:
            session_file = self.session_logger.save_to_file()
            yield self.session_logger.log_summary(
                f"Session saved to: {getattr(session_file, 'name', str(session_file))}"
            )
            # === New: If the session directory is empty, delete it ===
            sessions_dir = Path(self.session_logger.sessions_dir)
            if sessions_dir.exists() and not any(sessions_dir.iterdir()):
                shutil.rmtree(sessions_dir)
        except Exception as e:
            logger.warning(f"Failed to save session: {e}")
            yield self.session_logger.log_summary("Warning: Failed to save session")

    async def _handle_error(self, error: Exception) -> AsyncGenerator[str, None]:
        """Handle session errors."""
        error_message = f"Coordinator error: {str(error)}"
        logger.error(error_message)

        # === New: Write error_message to logs_path/error_msg/{session_id}.txt ===
        logs_path = get_config_value(self.config, "storage.logs_path", "./logs")
        error_dir = Path(logs_path) / "error_msg"
        error_dir.mkdir(parents=True, exist_ok=True)
        session_id = (
            self.session_logger.session_id if self.session_logger else "unknown"
        )
        error_file = error_dir / f"{session_id}.txt"
        with open(error_file, "w", encoding="utf-8") as f:
            f.write(error_message)

        if self.session_logger:
            self.session_logger.complete_session("error")
            yield self.session_logger.log_summary(
                f"Session failed with error: {str(error)}"
            )
        else:
            yield f"❌ Error: {str(error)}"

    def get_stats(self) -> Dict[str, Any]:
        """
        Get coordinator statistics.

        Returns:
            Dictionary with statistics
        """
        return {
            "total_sessions": self.total_sessions,
            "successful_sessions": self.successful_sessions,
            "success_rate": (
                self.successful_sessions / self.total_sessions * 100
                if self.total_sessions > 0
                else 0
            ),
            "agent_stats": self.task_executor.get_agent_status(),
        }

    def get_session_history(self) -> List[Dict[str, Any]]:
        """
        Get session history.

        Returns:
            List of session information
        """
        try:
            return SessionLogger.list_sessions(self.sessions_dir)
        except Exception as e:
            logger.warning(f"Failed to list sessions: {e}")
            return []

    async def load_session(self, session_id: str) -> Optional[SessionLogger]:
        """
        Load a specific session.

        Args:
            session_id: Session identifier

        Returns:
            SessionLogger if found, None otherwise
        """
        try:
            session = SessionLogger.load_from_file(session_id, self.sessions_dir)
            return session
        except Exception as e:
            logger.warning(f"Failed to load session {session_id}: {e}")
            return None

    def reset_stats(self) -> None:
        """Reset coordinator statistics."""
        self.total_sessions = 0
        self.successful_sessions = 0
        self.task_executor.reset_agent_stats()


async def process_simple_request(user_prompt: str) -> str:
    """
    Convenience function for simple request processing.

    Args:
        user_prompt: User's input prompt

    Returns:
        Final result as string
    """
    coordinator = Coordinator()
    result = ""

    async for message in coordinator.process_user_request(user_prompt):
        result += message + "\n"

    return result
