"""Coordinator - The central orchestrator for Pinocchio multi-agent system."""

import logging
from typing import Any, AsyncGenerator, Dict, List, Optional

from .session_logger import SessionLogger
from .task_planning import TaskExecutor, TaskPlanner

logger = logging.getLogger(__name__)


class Coordinator:
    """Central coordinator for managing multi-agent task planning and execution."""

    def __init__(
        self, llm_client: Optional[Any] = None, sessions_dir: str = "./sessions"
    ):
        """
        Initialize coordinator.

        Args:
            llm_client: LLM client instance
            sessions_dir: Directory to store session files
        """
        # Initialize task planning components
        self.task_planner = TaskPlanner(llm_client)
        self.task_executor = TaskExecutor(llm_client)

        # Session management
        self.sessions_dir = sessions_dir
        self.current_session: Optional[SessionLogger] = None

        # Statistics
        self.total_sessions = 0
        self.successful_sessions = 0

        logger.info("Coordinator initialized with task planning system")

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
            # Initialize session
            async for msg in self._initialize_session(user_prompt):
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

    async def _initialize_session(self, user_prompt: str) -> AsyncGenerator[str, None]:
        """Initialize new session."""
        self.current_session = SessionLogger(user_prompt, self.sessions_dir)
        yield self.current_session.log_summary("Session started")

    async def _create_task_plan(self, user_prompt: str) -> AsyncGenerator[str, None]:
        """Create intelligent task plan from user request."""
        yield self.current_session.log_summary("🤖 Creating intelligent task plan...")

        try:
            # Create task plan using task planner
            self.current_plan = await self.task_planner.create_task_plan(
                user_prompt, session_id=self.current_session.session_id
            )

            # Validate plan
            validation = self.task_planner.validate_plan(self.current_plan)

            if validation["valid"]:
                yield self.current_session.log_summary(
                    f"✅ Task plan created: {len(self.current_plan.tasks)} tasks"
                )

                if validation["warnings"]:
                    yield self.current_session.log_summary(
                        f"⚠️ Plan warnings: {', '.join(validation['warnings'])}"
                    )
            else:
                yield self.current_session.log_summary(
                    f"❌ Plan validation failed: {', '.join(validation['issues'])}"
                )
                raise Exception("Task plan validation failed")

        except Exception as e:
            yield self.current_session.log_summary(
                f"❌ Failed to create task plan: {str(e)}"
            )
            raise

    async def _execute_task_plan(self) -> AsyncGenerator[str, None]:
        """Execute the task plan with progress reporting."""
        if not hasattr(self, "current_plan"):
            yield self.current_session.log_summary("❌ No task plan available")
            return

        yield self.current_session.log_summary("🚀 Starting task execution...")

        # Execute plan and collect results
        async for progress_msg in self.task_executor.execute_plan(self.current_plan):
            yield self.current_session.log_summary(progress_msg)

            # Store final result if plan completed
            if "🎉 Plan" in progress_msg and "completed successfully" in progress_msg:
                if hasattr(self.current_plan, "final_result"):
                    self._final_result = self.current_plan.final_result

    async def _process_results(self) -> AsyncGenerator[str, None]:
        """Process and display final results."""
        if self._final_result and self.current_plan.is_completed():
            async for msg in self._display_success_results(self._final_result):
                yield msg
            self.current_session.complete_session("completed")
            self.successful_sessions += 1
            yield self.current_session.log_summary("Session completed successfully")
        else:
            self.current_session.complete_session("failed")
            yield self.current_session.log_summary("Session failed")

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
            session_file = self.current_session.save_to_file()
            yield self.current_session.log_summary(
                f"Session saved to: {getattr(session_file, 'name', str(session_file))}"
            )
        except Exception as e:
            logger.warning(f"Failed to save session: {e}")
            yield self.current_session.log_summary("Warning: Failed to save session")

    async def _handle_error(self, error: Exception) -> AsyncGenerator[str, None]:
        """Handle session errors."""
        error_message = f"Coordinator error: {str(error)}"
        logger.error(error_message)

        if self.current_session:
            self.current_session.complete_session("error")
            yield self.current_session.log_summary(
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
