"""
Task planning module for Pinocchio multi-agent system.

This module provides intelligent task planning capabilities for coordinating
multiple agents in code generation and optimization workflows.
"""

import logging
from typing import Any, Dict, List, Optional

from ..data_models.task_planning import (
    AgentType,
    Task,
    TaskDependency,
    TaskPlan,
    TaskPlanningContext,
    TaskPriority,
)
from ..utils import parse_structured_output, safe_json_parse, validate_json_structure

logger = logging.getLogger(__name__)


class TaskPlanner:
    """Intelligent task planner for multi-agent coordination."""

    def __init__(self, llm_client: Optional[Any] = None):
        """
        Initialize task planner.

        Args:
            llm_client: LLM client for request analysis
        """
        self.llm_client = llm_client
        logger.info("TaskPlanner initialized")

    async def create_task_plan(
        self, user_request: str, session_id: Optional[str] = None
    ) -> TaskPlan:
        """
        Create a comprehensive task plan for user request.

        Args:
            user_request: User's request description
            session_id: Optional session ID for context

        Returns:
            TaskPlan with generated tasks
        """
        logger.info(f"Creating task plan for request: {user_request[:50]}...")

        # Analyze user request
        context = await self._analyze_request(user_request)

        # Generate tasks
        tasks = await self._generate_tasks(context)

        # Create task plan
        plan = TaskPlan(
            plan_id=f"plan_{session_id or 'default'}",
            user_request=user_request,
            tasks=tasks,
            context=context.model_dump(),
        )

        logger.info(f"Created task plan with {len(tasks)} tasks")
        return plan

    async def _analyze_request(self, user_request: str) -> TaskPlanningContext:
        """
        Analyze user request to determine requirements and strategy.

        Args:
            user_request: User's request description

        Returns:
            TaskPlanningContext with analysis results
        """
        if self.llm_client:
            # Use LLM for intelligent analysis
            prompt = self._build_analysis_prompt(user_request)
            try:
                response = await self.llm_client.complete(prompt)
                analysis = self._parse_analysis_response(response, user_request)
            except Exception as e:
                logger.warning(f"LLM analysis failed, using fallback: {e}")
                analysis = self._fallback_analysis(user_request)
        else:
            # Use fallback analysis
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
Analyze the following user request for code generation and optimization:

Request: {user_request}

Please provide a structured analysis in JSON format:
{{
    "requirements": {{
        "primary_goal": "main objective",
        "secondary_goals": ["list", "of", "secondary", "goals"],
        "code_requirements": ["efficient_data_structures", "performance_optimization"]
    }},
    "optimization_goals": ["performance", "memory_efficiency", "scalability"],
    "constraints": ["simplicity", "safety", "compatibility"],
    "user_preferences": {{
        "complexity_level": "simple|moderate|advanced",
        "optimization_aggressiveness": "conservative|standard|aggressive"
    }},
    "planning_strategy": "conservative|standard|aggressive"
}}

Focus on high-performance computing and code optimization requirements.
"""

    def _parse_analysis_response(
        self, response: str, user_request: str
    ) -> Dict[str, Any]:
        """Parse LLM response for request analysis."""
        try:
            # Use utils for JSON parsing
            parsed = safe_json_parse(response)
            if parsed is not None:
                # Validate the structure
                required_keys = ["requirements", "optimization_goals", "constraints"]
                if validate_json_structure(parsed, required_keys):
                    return parsed
        except Exception as e:
            logger.warning(f"Failed to parse analysis response: {e}")

        # Fallback parsing
        return self._fallback_analysis(user_request)

    def _fallback_analysis(self, user_request: str) -> Dict[str, Any]:
        """Provide fallback analysis when LLM fails."""
        # Extract basic requirements
        requirements = self._extract_basic_requirements(user_request)

        # Determine optimization goals
        optimization_goals = self._determine_optimization_goals(user_request)

        # Identify constraints
        constraints = self._identify_constraints(user_request)

        # Determine planning strategy
        planning_strategy = self._determine_planning_strategy(user_request)

        return {
            "requirements": requirements,
            "optimization_goals": optimization_goals,
            "constraints": constraints,
            "user_preferences": {},
            "planning_strategy": planning_strategy,
        }

    def _extract_basic_requirements(self, user_request: str) -> Dict[str, Any]:
        """Extract basic requirements from user request."""
        requirements = {
            "primary_goal": user_request,
            "secondary_goals": [],
            "code_requirements": [],
        }

        # Add common requirements based on keywords
        if any(word in user_request.lower() for word in ["matrix", "vector", "array"]):
            requirements["code_requirements"].append("efficient_data_structures")

        if any(
            word in user_request.lower() for word in ["performance", "fast", "optimize"]
        ):
            requirements["code_requirements"].append("performance_optimization")

        return requirements

    def _determine_optimization_goals(self, user_request: str) -> List[str]:
        """Determine optimization goals from user request."""
        goals = ["performance"]

        if any(word in user_request.lower() for word in ["memory", "efficient"]):
            goals.append("memory_efficiency")

        if any(word in user_request.lower() for word in ["scale", "large"]):
            goals.append("scalability")

        return goals

    def _identify_constraints(self, user_request: str) -> List[str]:
        """Identify constraints from user request."""
        constraints = []

        if any(word in user_request.lower() for word in ["simple", "basic"]):
            constraints.append("simplicity")

        if any(word in user_request.lower() for word in ["safe", "robust"]):
            constraints.append("safety")

        return constraints

    def _determine_planning_strategy(self, user_request: str) -> str:
        """Determine planning strategy based on user request."""
        if any(
            word in user_request.lower() for word in ["complex", "advanced", "optimize"]
        ):
            return "aggressive"
        elif any(word in user_request.lower() for word in ["simple", "basic"]):
            return "conservative"
        else:
            return "standard"

    async def _generate_tasks(self, context: TaskPlanningContext) -> List[Task]:
        """
        Generate tasks as a multi-round generator→debugger→optimiser chain.
        """
        from ..config.settings import Settings

        config = Settings()
        try:
            config.load_from_file("pinocchio.json")
        except Exception:
            pass
        max_rounds = int(config.get("task_planning.max_optimisation_rounds", 3))
        enable_optimiser = bool(config.get("task_planning.enable_optimiser", True))

        tasks = []
        task_counter = 1
        prev_task_id = None
        for round_idx in range(1, max_rounds + 1):
            # Generator
            generator_instruction = self._build_generator_instruction(context)
            generator_task_id = f"task_{task_counter}"
            generator_task = Task(
                task_id=generator_task_id,
                agent_type=AgentType.GENERATOR,
                task_description=f"[Round {round_idx}] {context.user_request}",
                requirements=context.requirements,
                optimization_goals=context.optimization_goals,
                priority=TaskPriority.CRITICAL,
                dependencies=[
                    TaskDependency(task_id=prev_task_id, dependency_type="required")
                ]
                if prev_task_id
                else [],
                input_data={
                    "user_request": context.user_request,
                    "instruction": generator_instruction,
                    "optimisation_round": round_idx,
                },
            )
            tasks.append(generator_task)
            task_counter += 1
            prev_task_id = generator_task_id

            # Debugger (must follow every generator)
            debugger_instruction = self._build_debugger_instruction(context)
            debugger_task_id = f"task_{task_counter}"
            debugger_task = Task(
                task_id=debugger_task_id,
                agent_type=AgentType.DEBUGGER,
                task_description=f"[Round {round_idx}] Compile and debug generated code",
                requirements={"error_handling": True},
                priority=TaskPriority.CRITICAL,
                dependencies=[
                    TaskDependency(
                        task_id=generator_task_id, dependency_type="required"
                    )
                ],
                input_data={
                    "error_handling": True,
                    "instruction": debugger_instruction,
                    "optimisation_round": round_idx,
                },
            )
            tasks.append(debugger_task)
            task_counter += 1
            prev_task_id = debugger_task_id

            # Optimiser (optional)
            if enable_optimiser:
                optimizer_instruction = self._build_optimizer_instruction(context)
                optimizer_task_id = f"task_{task_counter}"
                optimizer_task = Task(
                    task_id=optimizer_task_id,
                    agent_type=AgentType.OPTIMIZER,
                    task_description=(
                        f"[Round {round_idx}] Optimise code for: "
                        f"{', '.join(context.optimization_goals) if context.optimization_goals else 'performance and efficiency'}"
                    ),
                    requirements={
                        "optimization_goals": context.optimization_goals
                        or ["performance", "memory_efficiency"]
                    },
                    priority=TaskPriority.HIGH,
                    dependencies=[
                        TaskDependency(
                            task_id=debugger_task_id, dependency_type="required"
                        )
                    ],
                    input_data={
                        "optimization_goals": context.optimization_goals
                        or ["performance", "memory_efficiency"],
                        "instruction": optimizer_instruction,
                        "optimisation_round": round_idx,
                    },
                )
                tasks.append(optimizer_task)
                task_counter += 1
                prev_task_id = optimizer_task_id

        return tasks

    def _build_generator_instruction(self, context: TaskPlanningContext) -> str:
        """Build detailed instruction for generator agent."""
        instruction_parts = [
            "Generate high-performance Choreo DSL operator code based on the user request.",
            "",
            "Key Requirements:",
            f"- Primary Goal: {context.user_request}",
            f"- Optimization Goals: {', '.join(context.optimization_goals)}",
            f"- Constraints: {', '.join(context.constraints)}",
            "",
            "Code Requirements:",
            "- Use efficient data structures and algorithms",
            "- Implement proper error handling",
            "- Include performance optimizations",
            "- Add comprehensive comments",
            "- Ensure code is production-ready",
            "",
            "Output Format:",
            "- Provide complete, compilable code",
            "- Include usage examples",
            "- Explain optimization techniques used",
            "- List any assumptions or limitations",
        ]

        return "\n".join(instruction_parts)

    def _build_optimizer_instruction(self, context: TaskPlanningContext) -> str:
        """Build detailed instruction for optimizer agent."""
        instruction_parts = [
            "Optimize the generated code for maximum performance and efficiency.",
            "",
            "Optimization Goals:",
            f"- {', '.join(context.optimization_goals)}",
            "",
            "Optimization Techniques to Consider:",
            "- Loop unrolling and vectorization",
            "- Memory access pattern optimization",
            "- Cache-friendly data structures",
            "- Algorithm complexity reduction",
            "- Parallel processing opportunities",
            "- Memory usage optimization",
            "",
            "Output Format:",
            "- Provide optimized code with explanations",
            "- Include performance benchmarks",
            "- Document optimization techniques used",
            "- Highlight expected performance improvements",
        ]

        return "\n".join(instruction_parts)

    def _build_debugger_instruction(self, context: TaskPlanningContext) -> str:
        """Build detailed instruction for debugger agent."""
        instruction_parts = [
            "Analyze and debug the generated code for potential issues.",
            "",
            "Debugging Focus Areas:",
            "- Compilation errors and syntax issues",
            "- Runtime errors and exceptions",
            "- Logic errors and edge cases",
            "- Performance bottlenecks",
            "- Memory leaks and resource management",
            "- Thread safety and concurrency issues",
            "",
            "Output Format:",
            "- List all issues found with severity levels",
            "- Provide specific fixes for each issue",
            "- Include corrected code snippets",
            "- Explain the root cause of each issue",
            "- Suggest preventive measures",
        ]

        return "\n".join(instruction_parts)

    def _build_evaluator_instruction(self, context: TaskPlanningContext) -> str:
        """Build detailed instruction for evaluator agent."""
        instruction_parts = [
            "Evaluate the generated and optimized code for quality and correctness.",
            "",
            "Evaluation Criteria:",
            "- Code correctness and functionality",
            "- Performance characteristics",
            "- Code maintainability and readability",
            "- Error handling and robustness",
            "- Documentation quality",
            "- Test coverage and reliability",
            "",
            "Output Format:",
            "- Provide comprehensive evaluation report",
            "- Include performance metrics",
            "- Rate code quality on multiple dimensions",
            "- Suggest improvements and best practices",
            "- Provide overall score and recommendations",
        ]

        return "\n".join(instruction_parts)

    async def create_adaptive_plan(
        self, user_request: str, previous_results: Dict[str, Any]
    ) -> TaskPlan:
        """
        Create an adaptive task plan based on previous results.

        Args:
            user_request: User's request description
            previous_results: Results from previous execution

        Returns:
            Adaptive TaskPlan
        """
        logger.info("Creating adaptive task plan")

        # Analyze request with previous context
        context = await self._analyze_request(user_request)

        # Generate adaptive tasks
        tasks = await self._generate_adaptive_tasks(context, previous_results)

        # Create adaptive plan
        plan = TaskPlan(
            plan_id=f"adaptive_plan_{len(previous_results)}",
            user_request=user_request,
            tasks=tasks,
            context=context.model_dump(),
        )

        return plan

    async def _generate_adaptive_tasks(
        self, context: TaskPlanningContext, previous_results: Dict[str, Any]
    ) -> List[Task]:
        """Generate adaptive tasks based on previous results."""
        tasks = []

        # Analyze previous failures
        failed_tasks = []
        for key, result in previous_results.items():
            if isinstance(result, dict) and not result.get("success", True):
                failed_tasks.append(result)
            elif isinstance(result, bool) and not result:
                # Handle boolean results (like "generator_failed": True)
                failed_tasks.append(
                    {"task_id": key, "success": False, "error_details": {}}
                )

        # Check for specific failure patterns
        if (
            "generator_failed" in previous_results
            and previous_results["generator_failed"]
        ):
            failed_tasks.append(
                {
                    "task_id": "generator",
                    "success": False,
                    "error_details": {
                        "error_message": previous_results.get(
                            "error_message", "Generator failed"
                        )
                    },
                }
            )

        if failed_tasks:
            # Add debugging and repair tasks - ensure DEBUGGER is first
            for i, failed_task in enumerate(failed_tasks):
                debug_task = Task(
                    task_id=f"debug_task_{i}",
                    agent_type=AgentType.DEBUGGER,
                    task_description=f"Debug and repair failed task: {failed_task.get('task_id', 'unknown')}",
                    requirements={"error_analysis": True},
                    priority=TaskPriority.CRITICAL,
                    input_data={
                        "failed_task": failed_task,
                        "error_details": failed_task.get("error_details", {}),
                    },
                )
                tasks.append(debug_task)

            # Add continuation task after debugger tasks
            continuation_task = Task(
                task_id="continuation_task",
                agent_type=AgentType.GENERATOR,
                task_description="Continue code generation after bug fix",
                requirements={"continuation": True},
                priority=TaskPriority.HIGH,
                dependencies=[
                    TaskDependency(
                        task_id=f"debug_task_{i}", dependency_type="required"
                    )
                    for i in range(len(failed_tasks))
                ],
                input_data={
                    "user_request": context.user_request,
                    "previous_results": previous_results,
                    "instruction": "Continue with improved code generation based on previous debugging results",
                },
            )
            tasks.append(continuation_task)
        else:
            # No failures, just add a generator task
            continuation_task = Task(
                task_id="continuation_task",
                agent_type=AgentType.GENERATOR,
                task_description="Continue code generation",
                requirements={"continuation": True},
                priority=TaskPriority.HIGH,
                input_data={
                    "user_request": context.user_request,
                    "previous_results": previous_results,
                    "instruction": "Continue with code generation",
                },
            )
            tasks.append(continuation_task)

        return tasks

    def validate_plan(self, plan: TaskPlan) -> Dict[str, Any]:
        """
        Validate task plan for completeness and consistency.

        Args:
            plan: TaskPlan to validate

        Returns:
            Validation results
        """
        validation_results = {
            "valid": True,  # Keep backward compatibility
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "task_count": len(plan.tasks),
            "dependency_issues": [],
            "issues": [],  # Keep backward compatibility
        }

        # Check for required tasks - only GENERATOR is absolutely required
        agent_types = [task.agent_type for task in plan.tasks]
        required_agents = [AgentType.GENERATOR]  # Only GENERATOR is required

        for agent_type in required_agents:
            if agent_type not in agent_types:
                validation_results["valid"] = False
                validation_results["is_valid"] = False
                error_msg = f"Missing required agent type: {agent_type}"
                validation_results["errors"].append(error_msg)
                validation_results["issues"].append(
                    error_msg
                )  # Keep backward compatibility

        # Check dependencies
        task_ids = {task.task_id for task in plan.tasks}
        for task in plan.tasks:
            for dependency in task.dependencies:
                if dependency.task_id not in task_ids:
                    error_msg = f"Task {task.task_id} depends on non-existent task {dependency.task_id}"
                    validation_results["dependency_issues"].append(error_msg)
                    validation_results["errors"].append(error_msg)
                    validation_results["issues"].append(
                        error_msg
                    )  # Keep backward compatibility
                elif dependency.task_id == task.task_id:
                    # Circular dependency
                    error_msg = f"Circular dependency in task {task.task_id}"
                    validation_results["dependency_issues"].append(error_msg)
                    validation_results["errors"].append(error_msg)
                    validation_results["issues"].append(
                        error_msg
                    )  # Keep backward compatibility

        if validation_results["dependency_issues"]:
            validation_results["valid"] = False
            validation_results["is_valid"] = False

        return validation_results

    @staticmethod
    def _create_context_from_task(task: Task) -> TaskPlanningContext:
        """Create TaskPlanningContext from a single task."""
        return TaskPlanningContext(
            user_request=task.task_description,
            requirements=task.requirements or {},
            optimization_goals=task.optimization_goals or [],
            constraints=[],
            user_preferences={},
            planning_strategy="standard",
        )
