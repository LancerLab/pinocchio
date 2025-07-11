"""Coordinator - The central orchestrator for Pinocchio multi-agent system."""

import logging
from typing import Any, AsyncGenerator, Dict, List, Optional

from .agents.generator import GeneratorAgent
from .llm.mock_client import MockLLMClient
from .session_logger import SessionLogger

logger = logging.getLogger(__name__)


class Coordinator:
    """Central coordinator for managing multi-agent workflows."""

    def __init__(
        self, llm_client: Optional[Any] = None, sessions_dir: str = "./sessions"
    ):
        """
        Initialize coordinator.

        Args:
            llm_client: LLM client instance (uses MockLLMClient if None)
            sessions_dir: Directory to store session files
        """
        # Initialize LLM client
        self.llm_client = llm_client or MockLLMClient(response_delay_ms=200)

        # Initialize agents
        self.generator_agent = GeneratorAgent(self.llm_client)

        # Session management
        self.sessions_dir = sessions_dir
        self.current_session: Optional[SessionLogger] = None

        # Statistics
        self.total_sessions = 0
        self.successful_sessions = 0

        logger.info("Coordinator initialized")

    async def process_user_request(self, user_prompt: str) -> AsyncGenerator[str, None]:
        """
        Process user request through multi-agent workflow.

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

            # Execute workflow
            async for msg in self._execute_workflow(user_prompt):
                yield msg
            final_result = self._final_result

            # Process results
            async for msg in self._process_results(final_result):
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

    async def _execute_workflow(self, user_prompt: str) -> AsyncGenerator[str, None]:
        """Execute the workflow steps."""
        plan = self._generate_simple_plan(user_prompt)
        yield self.current_session.log_summary(f"Plan generated: {len(plan)} steps")

        self._final_result = None
        for i, step in enumerate(plan):
            step_id = f"step_{i+1}"
            agent_type = step["agent_type"]
            task_description = step["task_description"]

            yield self.current_session.log_summary(
                f"Executing step {i+1}: {agent_type}"
            )

            result = await self._execute_agent_step(
                step_id, agent_type, task_description
            )

            if result.success:
                yield self.current_session.log_summary(
                    f"Step {i+1} completed successfully"
                )
                self._final_result = result
            else:
                yield self.current_session.log_summary(
                    f"Step {i+1} failed: {result.error_message}"
                )
                self._final_result = result
                break

    async def _process_results(self, final_result: Any) -> AsyncGenerator[str, None]:
        """Process and display final results."""
        if final_result and final_result.success:
            async for msg in self._display_success_results(final_result):
                yield msg
            self.current_session.complete_session("completed")
            self.successful_sessions += 1
            yield self.current_session.log_summary("Session completed successfully")
        else:
            self.current_session.complete_session("failed")
            yield self.current_session.log_summary("Session failed")

    async def _display_success_results(
        self, final_result: Any
    ) -> AsyncGenerator[str, None]:
        """Display successful results."""
        code = final_result.output.get("code", "")
        if code:
            yield f"\n🎉 Code generation completed!\n\n```choreo\n{code}\n```"

            explanation = final_result.output.get("explanation", "")
            if explanation:
                yield f"\n📋 Explanation: {explanation}"

            optimization_techniques = final_result.output.get(
                "optimization_techniques", []
            )
            if optimization_techniques:
                yield f"\n⚡ Optimizations applied: {', '.join(optimization_techniques)}"

    async def _save_session(self) -> AsyncGenerator[str, None]:
        """Save session to file."""
        try:
            session_file = self.current_session.save_to_file()
            yield self.current_session.log_summary(
                f"Session saved to: {session_file.name}"
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

    def _generate_simple_plan(self, user_prompt: str) -> List[Dict[str, Any]]:
        """
        Generate simple execution plan based on user prompt.

        Args:
            user_prompt: User's input prompt

        Returns:
            List of execution steps
        """
        plan = []

        # For MVP, we only implement the generator step
        # Later this will be expanded to include debugging, optimization, and evaluation

        # Always start with code generation
        plan.append(
            {
                "agent_type": "generator",
                "task_description": user_prompt,
                "requirements": self._extract_requirements(user_prompt),
                "optimization_goals": self._extract_optimization_goals(user_prompt),
            }
        )

        # TODO: Add conditional steps based on user prompt analysis
        # if "debug" in user_prompt.lower():
        #     plan.append({"agent_type": "debugger", ...})
        # if "optimize" in user_prompt.lower():
        #     plan.append({"agent_type": "optimizer", ...})
        # if "evaluate" in user_prompt.lower():
        #     plan.append({"agent_type": "evaluator", ...})

        return plan

    def _extract_requirements(self, user_prompt: str) -> Dict[str, Any]:
        """
        Extract requirements from user prompt.

        Args:
            user_prompt: User's input prompt

        Returns:
            Dictionary of extracted requirements
        """
        requirements = {}

        # Simple keyword-based requirement extraction
        prompt_lower = user_prompt.lower()

        # Performance requirements
        if (
            "fast" in prompt_lower
            or "performance" in prompt_lower
            or "快速" in user_prompt
            or "高性能" in user_prompt
        ):
            requirements["performance"] = "high"
        if "memory" in prompt_lower or "内存" in user_prompt:
            requirements["memory_efficient"] = True

        # Operation type
        if "conv" in prompt_lower or "conv2d" in prompt_lower:
            requirements["operation_type"] = "convolution"
        elif (
            "matmul" in prompt_lower or "matrix" in prompt_lower or "矩阵" in user_prompt
        ):
            requirements["operation_type"] = "matrix_multiplication"
        elif "add" in prompt_lower or "加法" in user_prompt:
            requirements["operation_type"] = "element_wise_addition"

        # Input/output requirements
        if "tensor" in prompt_lower or "tensor" in user_prompt:
            requirements["data_type"] = "tensor"

        return requirements

    def _extract_optimization_goals(self, user_prompt: str) -> List[str]:
        """
        Extract optimization goals from user prompt.

        Args:
            user_prompt: User's input prompt

        Returns:
            List of optimization goals
        """
        goals = []
        prompt_lower = user_prompt.lower()

        if "fast" in prompt_lower or "speed" in prompt_lower or "快速" in user_prompt:
            goals.append("maximize_throughput")
        if "memory" in prompt_lower or "内存" in user_prompt:
            goals.append("minimize_memory_usage")
        if "cache" in prompt_lower or "cache" in user_prompt:
            goals.append("optimize_cache_locality")
        if "parallel" in prompt_lower or "并行" in user_prompt:
            goals.append("enable_parallelization")

        # Default goals if none specified
        if not goals:
            goals = ["maximize_throughput", "optimize_cache_locality"]

        return goals

    async def _execute_agent_step(
        self, step_id: str, agent_type: str, task_description: str
    ) -> Any:
        """
        Execute a single agent step.

        Args:
            step_id: Unique step identifier
            agent_type: Type of agent to execute
            task_description: Task description for the agent

        Returns:
            Agent response
        """
        # Build request
        request = {
            "request_id": f"{self.current_session.session_id}_{step_id}",
            "agent_type": agent_type,
            "task_description": task_description,
            "context": self.current_session.get_context(),
            "session_id": self.current_session.session_id,
        }

        # Execute based on agent type
        if agent_type == "generator":
            result = await self.generator_agent.execute(request)
        else:
            # For MVP, only generator is implemented
            raise NotImplementedError(f"Agent type '{agent_type}' not yet implemented")

        # Log communication
        self.current_session.log_communication(
            step_id=step_id,
            agent_type=agent_type,
            request=request,
            response=result.model_dump(),
        )

        return result

    def get_stats(self) -> Dict[str, Any]:
        """
        Get coordinator statistics.

        Returns:
            Statistics dictionary
        """
        success_rate = 0.0
        if self.total_sessions > 0:
            success_rate = self.successful_sessions / self.total_sessions

        return {
            "total_sessions": self.total_sessions,
            "successful_sessions": self.successful_sessions,
            "success_rate": success_rate,
            "current_session_id": self.current_session.session_id
            if self.current_session
            else None,
            "agent_stats": {"generator": self.generator_agent.get_stats()},
        }

    def get_session_history(self) -> List[Dict[str, Any]]:
        """
        Get session history.

        Returns:
            List of session summaries
        """
        return SessionLogger.list_sessions(self.sessions_dir)

    async def load_session(self, session_id: str) -> Optional[SessionLogger]:
        """
        Load a previous session.

        Args:
            session_id: Session ID to load

        Returns:
            Loaded session or None if not found
        """
        sessions = self.get_session_history()

        for session_info in sessions:
            if session_info["session_id"] == session_id:
                try:
                    session = SessionLogger.load_from_file(session_info["file_path"])
                    logger.info(f"Session loaded: {session_id}")
                    return session
                except Exception as e:
                    logger.error(f"Failed to load session {session_id}: {e}")
                    return None

        logger.warning(f"Session not found: {session_id}")
        return None

    def reset_stats(self) -> None:
        """Reset coordinator statistics."""
        self.total_sessions = 0
        self.successful_sessions = 0
        self.generator_agent.reset_stats()
        logger.info("Coordinator stats reset")


# Convenience function for simple usage
async def process_simple_request(user_prompt: str) -> str:
    """
    Process a simple request and return the result.

    Args:
        user_prompt: User's input prompt

    Returns:
        Final result string
    """
    coordinator = Coordinator()
    messages = []

    async for message in coordinator.process_user_request(user_prompt):
        messages.append(message)

    return "\n".join(messages)
