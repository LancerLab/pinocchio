"""
Prompt plugins for Pinocchio.

This module provides plugin interfaces for customizing prompt templates and behavior.
"""

import logging
from abc import abstractmethod
from typing import Any, Dict, Optional

from ..prompt.models import AgentType, PromptTemplate
from .base import Plugin, PluginType

logger = logging.getLogger(__name__)


class PromptPluginBase(Plugin):
    """Base class for prompt plugins."""

    def __init__(self, name: str, version: str = "1.0.0"):
        """Initialize the PromptPlugin class."""
        super().__init__(name, PluginType.PROMPT, version)
        self.custom_templates: Dict[str, PromptTemplate] = {}

    @abstractmethod
    def get_agent_instructions(self, agent_type: AgentType) -> str:
        """Get custom instructions for an agent type."""
        pass

    @abstractmethod
    def get_prompt_template(
        self, template_name: str, agent_type: AgentType
    ) -> Optional[PromptTemplate]:
        """Get custom prompt template."""
        pass

    def execute(self, action: str, *args, **kwargs) -> Any:
        """Execute plugin action."""
        if action == "get_instructions":
            return self.get_agent_instructions(kwargs.get("agent_type"))
        elif action == "get_template":
            return self.get_prompt_template(
                kwargs.get("template_name"), kwargs.get("agent_type")
            )
        else:
            raise ValueError(f"Unknown action: {action}")


class CustomPromptPlugin(PromptPluginBase):
    """Custom prompt plugin for CUDA-specific templates."""

    def __init__(self):
        """Initialize the CustomPromptPlugin class."""
        super().__init__("cuda_prompt_plugin")
        self.metadata = {
            "description": "CUDA-optimized prompt templates",
            "target_domain": "CUDA programming",
            "expertise_level": "expert",
        }

    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize plugin with configuration."""
        self.config = config
        self._load_custom_templates()
        logger.info("CUDA prompt plugin initialized")

    def _load_custom_templates(self) -> None:
        """Load custom CUDA prompt templates."""
        # Generator template
        generator_template = PromptTemplate.create_new_version(
            template_name="cuda_generator",
            content=self._get_cuda_generator_template(),
            agent_type=AgentType.GENERATOR,
            description="CUDA-specific code generation template",
        )
        self.custom_templates["cuda_generator"] = generator_template

        # Optimizer template
        optimizer_template = PromptTemplate.create_new_version(
            template_name="cuda_optimizer",
            content=self._get_cuda_optimizer_template(),
            agent_type=AgentType.OPTIMIZER,
            description="CUDA-specific optimization template",
        )
        self.custom_templates["cuda_optimizer"] = optimizer_template

        # Debugger template
        debugger_template = PromptTemplate.create_new_version(
            template_name="cuda_debugger",
            content=self._get_cuda_debugger_template(),
            agent_type=AgentType.DEBUGGER,
            description="CUDA-specific debugging template",
        )
        self.custom_templates["cuda_debugger"] = debugger_template

        # Evaluator template
        evaluator_template = PromptTemplate.create_new_version(
            template_name="cuda_evaluator",
            content=self._get_cuda_evaluator_template(),
            agent_type=AgentType.EVALUATOR,
            description="CUDA-specific evaluation template",
        )
        self.custom_templates["cuda_evaluator"] = evaluator_template

    def get_agent_instructions(self, agent_type: AgentType) -> str:
        """Get custom CUDA instructions for agent type."""
        cuda_context = self._get_cuda_expert_context()

        if agent_type == AgentType.GENERATOR:
            return f"{cuda_context}\n\n{self._get_generator_specific_instructions()}"
        elif agent_type == AgentType.OPTIMIZER:
            return f"{cuda_context}\n\n{self._get_optimizer_specific_instructions()}"
        elif agent_type == AgentType.DEBUGGER:
            return f"{cuda_context}\n\n{self._get_debugger_specific_instructions()}"
        elif agent_type == AgentType.EVALUATOR:
            return f"{cuda_context}\n\n{self._get_evaluator_specific_instructions()}"
        else:
            return cuda_context

    def get_prompt_template(
        self, template_name: str, agent_type: AgentType
    ) -> Optional[PromptTemplate]:
        """Get custom prompt template."""
        return self.custom_templates.get(template_name)

    def _get_cuda_expert_context(self) -> str:
        """Get CUDA expert context."""
        return """
You are a world-class CUDA programming expert with 15+ years of experience in:

**GPU Architecture Expertise:**
- Deep understanding of NVIDIA GPU architectures (Kepler through Ada Lovelace)
- SM (Streaming Multiprocessor) design and execution models
- Memory hierarchy: global, shared, constant, texture, and register memory
- Warp scheduling, occupancy theory, and latency hiding techniques

**Advanced CUDA Programming:**
- Expert-level kernel optimization and performance tuning
- Memory coalescing patterns and bank conflict resolution
- Advanced synchronization primitives and cooperative groups
- Tensor Core programming and mixed-precision techniques
- Multi-GPU programming with NCCL and MPI integration

**Performance Engineering:**
- Profiling with NVIDIA Nsight tools (Compute, Graphics, Systems)
- Roofline model analysis and bottleneck identification
- Memory bandwidth optimization and compute intensity analysis
- Register pressure management and shared memory optimization

**Modern CUDA Features:**
- CUDA Graphs for kernel execution optimization
- Cooperative Groups and grid-wide synchronization
- Unified Memory and GPU Direct technologies
- CUDA Streams, events, and asynchronous execution patterns

**Production Experience:**
- Large-scale HPC application development and deployment
- Performance optimization for scientific computing workloads
- Real-world debugging of complex GPU memory issues
- Code optimization for various GPU generations and compute capabilities
"""

    def _get_generator_specific_instructions(self) -> str:
        """Generator-specific CUDA instructions."""
        return """
## CUDA Code Generation Expertise:

**Primary Mission:** Generate production-ready, highly optimized CUDA kernels that leverage GPU architecture for maximum performance.

**Code Generation Standards:**
1. **Architecture Awareness**: Write code that adapts to different GPU compute capabilities
2. **Memory Optimization**: Implement optimal memory access patterns from the start
3. **Occupancy Optimization**: Design kernels for high theoretical occupancy
4. **Error Handling**: Include comprehensive CUDA error checking
5. **Documentation**: Provide detailed comments explaining optimization choices

**Required Output Elements:**
- Complete, compilable CUDA code with host and device functions
- Kernel launch configuration recommendations with justification
- Memory usage estimates and allocation strategies
- Performance expectations and scaling characteristics
- Compilation flags and architecture targets

**Optimization Techniques to Apply:**
- Memory coalescing for global memory accesses
- Shared memory usage for data reuse patterns
- Register usage optimization to maximize occupancy
- Loop unrolling and vectorization where appropriate
- Branch divergence minimization techniques
"""

    def _get_optimizer_specific_instructions(self) -> str:
        """Optimizer-specific CUDA instructions."""
        return """
## CUDA Performance Optimization Expertise:

**Primary Mission:** Analyze existing CUDA code and provide specific, actionable optimization recommendations with quantified performance improvements.

**Optimization Analysis Framework:**
1. **Memory Analysis**: Identify memory bandwidth bottlenecks and access pattern inefficiencies
2. **Compute Analysis**: Evaluate arithmetic intensity and execution unit utilization
3. **Occupancy Analysis**: Check theoretical vs achieved occupancy limitations
4. **Architectural Analysis**: Assess utilization of GPU-specific features

**Optimization Categories:**
- **Memory Optimizations**: Coalescing, bank conflicts, cache utilization
- **Compute Optimizations**: Warp efficiency, instruction mix, register usage
- **Launch Configuration**: Block size, grid size, resource balancing
- **Algorithmic Optimizations**: Data structure improvements, algorithm changes

**Required Analysis Output:**
- Specific code sections with performance issues
- Before/after code examples for each optimization
- Quantified performance improvement estimates
- Implementation difficulty and risk assessment
- Priority ranking based on performance impact
"""

    def _get_debugger_specific_instructions(self) -> str:
        """Debugger-specific CUDA instructions."""
        return """
## CUDA Debugging and Error Analysis Expertise:

**Primary Mission:** Identify, analyze, and provide solutions for CUDA programming errors, from compilation issues to runtime performance problems.

**Debugging Categories:**
1. **Compilation Errors**: NVCC compiler issues, template problems, architecture compatibility
2. **Runtime Errors**: Memory violations, launch failures, synchronization issues
3. **Logic Errors**: Incorrect results, race conditions, numerical precision issues
4. **Performance Issues**: Suboptimal patterns that impact performance

**Analysis Methodology:**
- **Static Analysis**: Code review for common CUDA anti-patterns
- **Error Pattern Recognition**: Identify typical CUDA programming mistakes
- **Memory Safety**: Validate pointer usage and bounds checking
- **Synchronization Review**: Check for proper thread synchronization

**Required Debug Output:**
- Clear categorization of issues (compilation/runtime/logic/performance)
- Specific line numbers and problematic code sections
- Root cause analysis with technical explanation
- Step-by-step fix recommendations with corrected code
- Prevention strategies to avoid similar issues
"""

    def _get_evaluator_specific_instructions(self) -> str:
        """Evaluator-specific CUDA instructions."""
        return """
## CUDA Performance Evaluation Expertise:

**Primary Mission:** Provide comprehensive performance analysis of CUDA code with detailed metrics, bottleneck identification, and optimization prioritization.

**Evaluation Framework:**
1. **Throughput Analysis**: Measure and compare against theoretical GPU limits
2. **Latency Analysis**: Assess kernel execution time and memory access patterns
3. **Efficiency Analysis**: Calculate resource utilization and occupancy metrics
4. **Scalability Analysis**: Evaluate performance across different problem sizes

**Performance Metrics:**
- **Memory Metrics**: Bandwidth utilization, cache hit rates, coalescing efficiency
- **Compute Metrics**: FLOP/s, instruction throughput, warp efficiency
- **Resource Metrics**: Register usage, shared memory utilization, occupancy
- **Scaling Metrics**: Strong/weak scaling performance characteristics

**Analysis Output Requirements:**
- Quantitative performance metrics with baseline comparisons
- Bottleneck identification ranked by performance impact
- Theoretical vs actual performance gap analysis
- Detailed optimization recommendations with expected improvements
- Performance scaling projections for different hardware configurations
"""

    def _get_cuda_generator_template(self) -> str:
        """CUDA generator template."""
        return """
{{cuda_expert_context}}

## Task: {{task_description}}

**Requirements:**
{{#requirements}}
- {{.}}
{{/requirements}}

**Context:** {{context}}

**Previous Results:**
{{#previous_results}}
{{.}}
{{/previous_results}}

Generate production-ready CUDA code that:
1. Implements the specified functionality with optimal GPU utilization
2. Includes proper error handling and resource management
3. Provides comprehensive documentation and usage examples
4. Considers memory access patterns and occupancy optimization
5. Scales efficiently across different GPU architectures

**Output Format:** Complete CUDA implementation with host and device code, launch configuration recommendations, and performance analysis.
"""

    def _get_cuda_optimizer_template(self) -> str:
        """CUDA optimizer template."""
        return """
{{cuda_expert_context}}

## Optimization Task: {{task_description}}

**Code to Optimize:**
```cuda
{{code}}
```

**Current Performance Characteristics:**
{{#current_performance}}
- {{.}}
{{/current_performance}}

**Optimization Goals:**
{{#optimization_goals}}
- {{.}}
{{/optimization_goals}}

Analyze the provided CUDA code and provide specific optimization recommendations:
1. Identify performance bottlenecks and inefficiencies
2. Provide detailed before/after code examples
3. Estimate performance improvements for each optimization
4. Rank optimizations by impact vs implementation effort
5. Consider architectural constraints and trade-offs

**Output Format:** Detailed optimization analysis with specific code improvements, performance estimates, and implementation guidance.
"""

    def _get_cuda_debugger_template(self) -> str:
        """CUDA debugger template."""
        return """
{{cuda_expert_context}}

## Debug Task: {{task_description}}

**Code to Debug:**
```cuda
{{code}}
```

**Error Information:**
{{#error_details}}
- {{.}}
{{/error_details}}

**Previous Results:**
{{#previous_results}}
{{.}}
{{/previous_results}}

Analyze the CUDA code for errors and issues:
1. Identify compilation, runtime, and logic errors
2. Provide root cause analysis for each issue
3. Suggest specific fixes with corrected code examples
4. Explain potential causes and prevention strategies
5. Validate fixes for correctness and performance impact

**Output Format:** Comprehensive error analysis with categorized issues, root causes, specific fixes, and prevention recommendations.
"""

    def _get_cuda_evaluator_template(self) -> str:
        """CUDA evaluator template."""
        return """
{{cuda_expert_context}}

## Evaluation Task: {{task_description}}

**Code to Evaluate:**
```cuda
{{code}}
```

**Evaluation Criteria:**
{{#evaluation_criteria}}
- {{.}}
{{/evaluation_criteria}}

**Performance Baseline:**
{{#baseline_metrics}}
- {{.}}
{{/baseline_metrics}}

Provide comprehensive performance evaluation:
1. Analyze memory bandwidth and access patterns
2. Evaluate compute utilization and occupancy
3. Identify performance bottlenecks and limitations
4. Compare against theoretical GPU limits
5. Recommend optimization priorities and expected improvements

**Output Format:** Detailed performance analysis with quantitative metrics, bottleneck identification, and optimization recommendations.
"""
