"""Task planner for intelligent task decomposition and planning."""

import logging
import uuid
from typing import Any, Dict, List, Optional

from ..data_models.task_planning import (
    AgentType,
    Task,
    TaskDependency,
    TaskPlan,
    TaskPlanningContext,
    TaskPriority,
)
from ..llm.mock_client import MockLLMClient

logger = logging.getLogger(__name__)


class TaskPlanner:
    """Intelligent task planner for decomposing user requests into executable tasks."""

    def __init__(self, llm_client: Optional[Any] = None):
        """
        Initialize task planner.

        Args:
            llm_client: LLM client for intelligent planning (uses MockLLMClient if None)
        """
        self.llm_client = llm_client or MockLLMClient(response_delay_ms=100)
        logger.info("TaskPlanner initialized")

    async def create_task_plan(
        self, user_request: str, session_id: Optional[str] = None
    ) -> TaskPlan:
        """
        Create a task plan from user request.

        Args:
            user_request: User's input request
            session_id: Optional session identifier

        Returns:
            TaskPlan with decomposed tasks
        """
        # Create planning context
        context = await self._analyze_request(user_request)
        context.session_id = session_id

        # Generate tasks based on analysis
        tasks = await self._generate_tasks(context)

        # Create task plan
        plan_id = f"plan_{uuid.uuid4().hex[:8]}"
        plan = TaskPlan(
            plan_id=plan_id,
            user_request=user_request,
            tasks=tasks,
            session_id=session_id,
            context=context.model_dump(),
        )

        logger.info(f"Created task plan {plan_id} with {len(tasks)} tasks")
        return plan

    async def _analyze_request(self, user_request: str) -> TaskPlanningContext:
        """
        Analyze user request to extract requirements and goals.

        Args:
            user_request: User's input request

        Returns:
            TaskPlanningContext with extracted information
        """
        # Build analysis prompt
        prompt = self._build_analysis_prompt(user_request)

        try:
            # Call LLM for intelligent analysis
            response = await self.llm_client.complete(prompt, agent_type="task_planner")

            # Parse response (simplified for now)
            analysis = self._parse_analysis_response(response, user_request)

        except Exception as e:
            logger.warning(f"LLM analysis failed, using fallback: {e}")
            analysis = self._fallback_analysis(user_request)

        return TaskPlanningContext(
            user_request=user_request,
            requirements=analysis.get("requirements", {}),
            optimization_goals=analysis.get("optimization_goals", []),
            constraints=analysis.get("constraints", []),
            user_preferences=analysis.get("user_preferences", {}),
            planning_strategy=analysis.get("planning_strategy", "standard"),
        )

    def _build_analysis_prompt(self, user_request: str) -> str:
        """Build prompt for request analysis."""
        return f"""
You are a task planner in the Pinocchio multi-agent system. Analyze the following user request and extract:

1. Requirements: What needs to be accomplished
2. Optimization goals: Performance, memory, etc. goals
3. Constraints: Limitations or requirements
4. User preferences: Any specific preferences mentioned
5. Planning strategy: Recommended approach (standard, aggressive, conservative)

User request: {user_request}

Please provide your analysis in JSON format:
{{
    "requirements": {{
        "primary_goal": "main objective",
        "secondary_goals": ["goal1", "goal2"],
        "code_requirements": ["requirement1", "requirement2"]
    }},
    "optimization_goals": ["performance", "memory_efficiency", "scalability"],
    "constraints": ["constraint1", "constraint2"],
    "user_preferences": {{
        "preference1": "value1"
    }},
    "planning_strategy": "standard|aggressive|conservative"
}}
"""

    def _parse_analysis_response(
        self, response: str, user_request: str
    ) -> Dict[str, Any]:
        """Parse LLM response for request analysis."""
        try:
            import json

            # Try to extract JSON from response
            if "{" in response and "}" in response:
                start = response.find("{")
                end = response.rfind("}") + 1
                json_str = response[start:end]
                return json.loads(json_str)
        except Exception as e:
            logger.warning(f"Failed to parse analysis response: {e}")

        # Fallback parsing
        return self._fallback_analysis(user_request)

    def _fallback_analysis(self, user_request: str) -> Dict[str, Any]:
        """Fallback analysis when LLM fails."""
        user_request_lower = user_request.lower()

        requirements = {
            "primary_goal": "code_generation",
            "secondary_goals": [],
            "code_requirements": [],
        }

        optimization_goals = []
        if any(
            word in user_request_lower
            for word in ["optimize", "performance", "fast", "efficient"]
        ):
            optimization_goals.append("performance")
        if any(
            word in user_request_lower for word in ["memory", "efficient", "compact"]
        ):
            optimization_goals.append("memory_efficiency")
        if any(word in user_request_lower for word in ["scale", "parallel", "thread"]):
            optimization_goals.append("scalability")

        constraints = []
        if any(word in user_request_lower for word in ["debug", "error", "fix"]):
            constraints.append("error_handling")

        return {
            "requirements": requirements,
            "optimization_goals": optimization_goals,
            "constraints": constraints,
            "user_preferences": {},
            "planning_strategy": "standard",
        }

    async def _generate_tasks(self, context: TaskPlanningContext) -> List[Task]:
        """
        Generate tasks based on planning context.

        Args:
            context: Task planning context

        Returns:
            List of tasks to execute
        """
        tasks = []
        task_counter = 1

        # Always start with code generation
        generator_instruction = self._build_generator_instruction(context)
        generator_task = Task(
            task_id=f"task_{task_counter}",
            agent_type=AgentType.GENERATOR,
            task_description=context.user_request,
            requirements=context.requirements,
            optimization_goals=context.optimization_goals,
            priority=TaskPriority.CRITICAL,
            input_data={
                "user_request": context.user_request,
                "instruction": generator_instruction,
            },
        )
        tasks.append(generator_task)
        task_counter += 1

        # Add optimization task if optimization goals exist
        if context.optimization_goals:
            optimizer_instruction = self._build_optimizer_instruction(context)
            optimizer_task = Task(
                task_id=f"task_{task_counter}",
                agent_type=AgentType.OPTIMIZER,
                task_description=f"Optimize generated code for: {', '.join(context.optimization_goals)}",
                requirements={"optimization_goals": context.optimization_goals},
                priority=TaskPriority.HIGH,
                dependencies=[
                    TaskDependency(task_id="task_1", dependency_type="required")
                ],
                input_data={
                    "optimization_goals": context.optimization_goals,
                    "instruction": optimizer_instruction,
                },
            )
            tasks.append(optimizer_task)
            task_counter += 1

        # Add debugging task if error handling is needed
        if "error_handling" in context.constraints:
            debugger_instruction = self._build_debugger_instruction(context)
            debugger_task = Task(
                task_id=f"task_{task_counter}",
                agent_type=AgentType.DEBUGGER,
                task_description="Analyze code for potential issues and errors",
                requirements={"error_handling": True},
                priority=TaskPriority.HIGH,
                dependencies=[
                    TaskDependency(task_id="task_1", dependency_type="required")
                ],
                input_data={
                    "error_handling": True,
                    "instruction": debugger_instruction,
                },
            )
            tasks.append(debugger_task)
            task_counter += 1

        # Add evaluation task for performance assessment
        if context.optimization_goals or context.requirements.get(
            "performance_requirements"
        ):
            evaluator_instruction = self._build_evaluator_instruction(context)
            evaluator_task = Task(
                task_id=f"task_{task_counter}",
                agent_type=AgentType.EVALUATOR,
                task_description="Evaluate code performance and provide assessment",
                requirements={"evaluation_criteria": context.optimization_goals},
                priority=TaskPriority.MEDIUM,
                dependencies=[
                    TaskDependency(task_id="task_1", dependency_type="required")
                ],
                input_data={
                    "evaluation_criteria": context.optimization_goals,
                    "instruction": evaluator_instruction,
                },
            )
            tasks.append(evaluator_task)

        return tasks

    def _build_generator_instruction(self, context: TaskPlanningContext) -> str:
        """Build detailed instruction for generator agent."""
        instruction_parts = [
            "Generate high-performance Choreo DSL operator code based on the user request.",
            "",
            "Key Requirements:",
        ]

        if context.requirements:
            for key, value in context.requirements.items():
                instruction_parts.append(f"- {key}: {value}")

        if context.optimization_goals:
            instruction_parts.extend(
                [
                    "",
                    "Optimization Goals:",
                    "- " + "\n- ".join(context.optimization_goals),
                ]
            )

        if context.constraints:
            instruction_parts.extend(
                ["", "Constraints:", "- " + "\n- ".join(context.constraints)]
            )

        instruction_parts.extend(
            [
                "",
                "Focus on:",
                "- Performance optimization (loop tiling, vectorization, memory coalescing)",
                "- Memory efficiency and access patterns",
                "- Correctness and safety with proper error checking",
                "- Code readability and maintainability",
                "- Following Choreo DSL syntax and conventions",
            ]
        )

        return "\n".join(instruction_parts)

    def _build_optimizer_instruction(self, context: TaskPlanningContext) -> str:
        """Build detailed instruction for optimizer agent."""
        instruction_parts = [
            "Analyze and optimize the generated Choreo DSL code for better performance.",
            "",
            "Optimization Goals:",
            "- " + "\n- ".join(context.optimization_goals),
        ]

        if context.requirements:
            instruction_parts.extend(
                [
                    "",
                    "Additional Requirements:",
                    "- "
                    + "\n- ".join(
                        [f"{k}: {v}" for k, v in context.requirements.items()]
                    ),
                ]
            )

        instruction_parts.extend(
            [
                "",
                "Optimization Focus:",
                "- Identify performance bottlenecks",
                "- Apply advanced optimization techniques",
                "- Maintain code correctness",
                "- Provide detailed optimization explanations",
                "- Suggest hyperparameter tuning",
            ]
        )

        return "\n".join(instruction_parts)

    def _build_debugger_instruction(self, context: TaskPlanningContext) -> str:
        """Build detailed instruction for debugger agent."""
        instruction_parts = [
            "Analyze the generated code for potential issues, errors, and improvements.",
            "",
            "Debugging Focus:",
            "- Syntax errors and compatibility issues",
            "- Logic errors and edge cases",
            "- Performance bottlenecks",
            "- Memory access patterns",
            "- Error handling and validation",
        ]

        if context.constraints:
            instruction_parts.extend(
                [
                    "",
                    "Specific Constraints to Check:",
                    "- " + "\n- ".join(context.constraints),
                ]
            )

        instruction_parts.extend(
            [
                "",
                "Provide:",
                "- Detailed analysis of issues found",
                "- Specific fixes with explanations",
                "- Improved code version",
                "- Recommendations for robustness",
            ]
        )

        return "\n".join(instruction_parts)

    def _build_evaluator_instruction(self, context: TaskPlanningContext) -> str:
        """Build detailed instruction for evaluator agent."""
        instruction_parts = [
            "Evaluate the generated code for performance, correctness, and quality.",
            "",
            "Evaluation Criteria:",
        ]

        if context.optimization_goals:
            instruction_parts.append("- " + "\n- ".join(context.optimization_goals))

        if context.requirements.get("performance_requirements"):
            instruction_parts.extend(
                [
                    "",
                    "Performance Requirements:",
                    f"- {context.requirements['performance_requirements']}",
                ]
            )

        instruction_parts.extend(
            [
                "",
                "Evaluation Focus:",
                "- Code quality and maintainability",
                "- Performance characteristics",
                "- Memory usage patterns",
                "- Correctness and safety",
                "- Optimization effectiveness",
                "- Scalability considerations",
            ]
        )

        return "\n".join(instruction_parts)

    async def create_adaptive_plan(
        self, user_request: str, previous_results: Dict[str, Any]
    ) -> TaskPlan:
        """
        Create an adaptive task plan based on previous results.

        Args:
            user_request: User's input request
            previous_results: Results from previous executions

        Returns:
            Adaptive TaskPlan
        """
        context = await self._analyze_request(user_request)
        context.previous_results = previous_results

        # Generate adaptive tasks
        tasks = await self._generate_adaptive_tasks(context, previous_results)

        plan_id = f"adaptive_plan_{uuid.uuid4().hex[:8]}"
        plan = TaskPlan(
            plan_id=plan_id,
            user_request=user_request,
            tasks=tasks,
            context=context.dict(),
        )

        logger.info(f"Created adaptive task plan {plan_id} with {len(tasks)} tasks")
        return plan

    async def _generate_adaptive_tasks(
        self, context: TaskPlanningContext, previous_results: Dict[str, Any]
    ) -> List[Task]:
        """Generate adaptive tasks based on previous results."""
        tasks = []
        task_counter = 1

        # Check if previous generation failed
        if previous_results.get("generator_failed"):
            # Start with debugging to understand the issue
            debugger_task = Task(
                task_id=f"task_{task_counter}",
                agent_type=AgentType.DEBUGGER,
                task_description="Analyze previous generation failure and provide insights",
                requirements={"analyze_failure": True},
                priority=TaskPriority.CRITICAL,
                input_data={"previous_results": previous_results},
            )
            tasks.append(debugger_task)
            task_counter += 1

        # Add generation task
        generator_task = Task(
            task_id=f"task_{task_counter}",
            agent_type=AgentType.GENERATOR,
            task_description=context.user_request,
            requirements=context.requirements,
            optimization_goals=context.optimization_goals,
            priority=TaskPriority.CRITICAL,
            input_data={
                "user_request": context.user_request,
                "previous_results": previous_results,
            },
        )
        tasks.append(generator_task)
        task_counter += 1

        # Add optimization if needed
        if context.optimization_goals:
            optimizer_task = Task(
                task_id=f"task_{task_counter}",
                agent_type=AgentType.OPTIMIZER,
                task_description=f"Optimize generated code for: {', '.join(context.optimization_goals)}",
                requirements={"optimization_goals": context.optimization_goals},
                priority=TaskPriority.HIGH,
                dependencies=[
                    TaskDependency(
                        task_id=f"task_{task_counter-1}", dependency_type="required"
                    )
                ],
                input_data={"optimization_goals": context.optimization_goals},
            )
            tasks.append(optimizer_task)
            task_counter += 1

        return tasks

    def validate_plan(self, plan: TaskPlan) -> Dict[str, Any]:
        """
        Validate task plan for consistency and completeness.

        Args:
            plan: Task plan to validate

        Returns:
            Validation results
        """
        issues = []
        warnings = []

        # Check for circular dependencies
        for task in plan.tasks:
            for dep in task.dependencies:
                if dep.task_id == task.task_id:
                    issues.append(f"Circular dependency in task {task.task_id}")

        # Check for missing dependencies
        task_ids = {task.task_id for task in plan.tasks}
        for task in plan.tasks:
            for dep in task.dependencies:
                if dep.task_id not in task_ids:
                    issues.append(
                        f"Missing dependency {dep.task_id} for task {task.task_id}"
                    )

        # Check for critical tasks
        critical_tasks = [
            task for task in plan.tasks if task.priority == TaskPriority.CRITICAL
        ]
        if not critical_tasks:
            warnings.append("No critical tasks in plan")

        # Check plan size
        if len(plan.tasks) > 10:
            warnings.append("Plan has many tasks, consider simplification")

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "task_count": len(plan.tasks),
            "critical_task_count": len(critical_tasks),
        }
