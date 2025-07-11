"""
Integration tests for multi-agent workflow between Prompt, Session, Memory, and Knowledge modules.

This test suite demonstrates how the modules work together in a high-performance
kernel optimization workflow, similar to the prompt_usage_example.py.
"""

from datetime import datetime

import pytest

from pinocchio.knowledge import (
    KnowledgeCategory,
    KnowledgeContentType,
    KnowledgeFragment,
    KnowledgeManager,
)
from pinocchio.memory import AgentMemory, CodeVersion, MemoryManager
from pinocchio.prompt import AgentType, PromptManager, PromptType, StructuredInput
from pinocchio.session import SessionManager, SessionStatus


@pytest.fixture
def temp_workflow_dir(tmp_path):
    """Create temporary directory for workflow testing."""
    workflow_dir = tmp_path / "workflow_test"
    workflow_dir.mkdir()
    return workflow_dir


@pytest.fixture
def prompt_manager(temp_workflow_dir):
    """Create prompt manager for testing."""
    prompt_dir = temp_workflow_dir / "prompts"
    prompt_dir.mkdir()
    return PromptManager(storage_path=str(prompt_dir))


@pytest.fixture
def session_manager(temp_workflow_dir):
    """Create session manager for testing."""
    session_dir = temp_workflow_dir / "sessions"
    session_dir.mkdir()
    return SessionManager(store_dir=str(session_dir))


@pytest.fixture
def memory_manager(temp_workflow_dir):
    """Create memory manager for testing."""
    memory_dir = temp_workflow_dir / "memory"
    memory_dir.mkdir()
    return MemoryManager(store_dir=str(memory_dir))


@pytest.fixture
def knowledge_manager(temp_workflow_dir):
    """Create knowledge manager for testing."""
    knowledge_dir = temp_workflow_dir / "knowledge"
    knowledge_dir.mkdir()
    return KnowledgeManager(storage_path=str(knowledge_dir))


class TestMultiAgentKernelOptimizationWorkflow:
    """Test multi-agent workflow for kernel optimization."""

    def test_workflow_initialization(
        self, prompt_manager, session_manager, memory_manager, knowledge_manager
    ):
        """Test initialization of all managers for workflow."""
        # Create a session for kernel optimization
        session = session_manager.create_session(
            task_description="Optimize matrix multiplication kernel for AVX-512",
            target_performance={
                "target_gflops": 100.0,
                "target_memory_bandwidth": 80.0,
                "target_cache_hit_rate": 0.9,
            },
        )

        # Verify session creation
        assert session.session_id is not None
        assert session.status == SessionStatus.ACTIVE
        assert session.target_performance is not None

        # Verify managers are ready
        assert prompt_manager is not None
        assert memory_manager is not None
        assert knowledge_manager is not None

        return session

    def test_kernel_generator_workflow(
        self, prompt_manager, session_manager, memory_manager, knowledge_manager
    ):
        """Test kernel generator agent workflow."""
        # Create session
        session = session_manager.create_session(
            task_description="Generate optimized matrix multiplication kernel"
        )

        # Create kernel generator template
        prompt_manager.create_template(
            template_name="kernel_generator",
            content="""
Generate optimized kernel for the following computation:
Algorithm: {{algorithm_type}}
Input dimensions: {{input_dims}}
Target architecture: {{target_arch}}
Performance requirements: {{performance_targets}}

Optimization constraints:
- Memory bandwidth: {{memory_constraints}}
- Cache utilization: {{cache_constraints}}
- SIMD vectorization: {{vectorization_level}}
- Parallelization strategy: {{parallel_strategy}}

Please provide:
- Optimized kernel implementation
- Vectorization techniques used (AVX/NEON/SVE)
- Memory access patterns and cache optimization
- Parallelization approach (OpenMP/CUDA/SYCL)
- Performance analysis and bottlenecks
- Confidence score for optimization
""",
            agent_type=AgentType.GENERATOR,
            prompt_type=PromptType.CODE_GENERATION,
            description="High-performance kernel generator",
            tags=["kernel-optimization", "vectorization", "hpc"],
        )

        # Create structured input for matrix multiplication
        generator_input = StructuredInput(
            code_snippet="""
// Baseline matrix multiplication kernel
void matmul_baseline(float* A, float* B, float* C, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < N; k++) {
                sum += A[i*N + k] * B[k*N + j];
            }
            C[i*N + j] = sum;
        }
    }
}
""",
            requirements={
                "algorithm": "matrix_multiplication",
                "data_type": "float",
                "target_arch": "x86_64_avx512",
                "performance": "high_throughput",
            },
            constraints=[
                "memory_bandwidth_limited",
                "cache_friendly_access",
                "simd_vectorization",
                "thread_parallelization",
            ],
            optimization_targets=[
                "flops_per_second",
                "memory_bandwidth_utilization",
                "cache_hit_rate",
                "vectorization_efficiency",
            ],
            performance_metrics={
                "execution_time": 2.5,
                "memory_bandwidth": 45.2,
                "cache_miss_rate": 0.15,
                "vectorization_coverage": 0.3,
                "gflops": 12.8,
            },
        )

        # Format prompt
        formatted_prompt = prompt_manager.format_structured_prompt(
            "kernel_generator", generator_input, agent_type=AgentType.GENERATOR
        )

        # Record agent interaction in session
        session_manager.add_agent_interaction(
            session.session_id,
            "generator",
            {
                "template_used": "kernel_generator",
                "input_data": generator_input.to_dict(),
                "formatted_prompt": formatted_prompt[:500],  # Truncate for storage
                "timestamp": datetime.utcnow().isoformat(),
            },
        )

        # Simulate generated optimized kernel
        optimized_kernel = """
// Optimized matrix multiplication kernel with AVX-512
#include <immintrin.h>

void matmul_optimized(float* A, float* B, float* C, int N) {
    const int BLOCK_SIZE = 32;

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < N; i += BLOCK_SIZE) {
        for (int j = 0; j < N; j += BLOCK_SIZE) {
            for (int k = 0; k < N; k += BLOCK_SIZE) {
                // Blocked computation with vectorization
                for (int ii = i; ii < min(i + BLOCK_SIZE, N); ii++) {
                    for (int jj = j; jj < min(j + BLOCK_SIZE, N); jj += 16) {
                        __m512 sum = _mm512_setzero_ps();
                        for (int kk = k; kk < min(k + BLOCK_SIZE, N); kk++) {
                            __m512 a = _mm512_set1_ps(A[ii * N + kk]);
                            __m512 b = _mm512_loadu_ps(&B[kk * N + jj]);
                            sum = _mm512_fmadd_ps(a, b, sum);
                        }
                        _mm512_storeu_ps(&C[ii * N + jj], sum);
                    }
                }
            }
        }
    }
}
"""

        # Store optimized kernel in memory
        code_version = CodeVersion.create_new_version(
            session_id=session.session_id,
            code=optimized_kernel,
            language="c",
            kernel_type="matrix_multiplication",
            source_agent="generator",
            description="Optimized matrix multiplication with AVX-512 and blocking",
            optimization_techniques=["vectorization", "blocking", "parallelization"],
        )

        memory_manager.add_code_version(session.session_id, code_version)

        # Add version reference to session
        session_manager.add_version_reference(
            session.session_id, "memory", code_version.version_id
        )

        # Store knowledge about optimization techniques
        knowledge_fragment = KnowledgeFragment.create_fragment(
            session_id=session.session_id,
            agent_type="generator",
            category=KnowledgeCategory.OPTIMIZATION,
            title="Matrix multiplication optimization with AVX-512 vectorization and blocking",
            content="Matrix multiplication optimization with AVX-512 vectorization and blocking",
            content_type=KnowledgeContentType.TEXT,
            metadata={
                "technique": "blocking_vectorization",
                "architecture": "x86_64_avx512",
                "algorithm": "matrix_multiplication",
                "performance_improvement": "6.7x speedup",
            },
        )

        knowledge_manager.add_fragment(knowledge_fragment)

        # Add knowledge version reference to session
        session_manager.add_version_reference(
            session.session_id, "knowledge", knowledge_fragment.fragment_id
        )

        # Verify workflow data
        code_memory = memory_manager.get_code_memory(session.session_id)
        assert code_version.version_id in code_memory.versions
        assert knowledge_fragment.fragment_id in [
            f.fragment_id
            for f in knowledge_manager.get_session_fragments(session.session_id)
        ]

        # Verify session tracking
        session_data = session_manager.get_session(session.session_id)
        assert len(session_data.agent_interactions) == 1
        assert len(session_data.memory_versions) == 1
        assert len(session_data.knowledge_versions) == 1

        return session, code_version, knowledge_fragment

    def test_kernel_debugger_workflow(
        self, prompt_manager, session_manager, memory_manager, knowledge_manager
    ):
        """Test kernel debugger agent workflow."""
        # Get the session and code from previous test
        session, code_version, knowledge_fragment = self.test_kernel_generator_workflow(
            prompt_manager, session_manager, memory_manager, knowledge_manager
        )

        # Create kernel debugger template
        prompt_manager.create_template(
            template_name="kernel_debugger",
            content="""
Debug and optimize the following kernel:
Kernel code: {{kernel_code}}
Performance profile: {{performance_profile}}
Bottleneck analysis: {{bottleneck_analysis}}
Hardware counters: {{hw_counters}}

Issues to address:
- Memory access patterns: {{memory_issues}}
- Vectorization efficiency: {{vectorization_issues}}
- Cache miss rates: {{cache_issues}}
- Load balancing: {{load_balancing_issues}}

Please provide:
- Performance bottleneck identification
- Memory access optimization suggestions
- Vectorization improvements
- Cache-friendly restructuring
- Load balancing optimizations
- Root cause analysis of performance issues
""",
            agent_type=AgentType.DEBUGGER,
            prompt_type=PromptType.CODE_DEBUGGING,
            description="Kernel performance debugging",
            tags=["performance-debugging", "memory-optimization", "hpc"],
        )

        # Create debugger input
        debugger_input = StructuredInput(
            code_snippet=code_version.code,
            requirements={
                "analysis_type": "performance_debugging",
                "target_metrics": [
                    "memory_bandwidth",
                    "cache_efficiency",
                    "vectorization",
                ],
            },
            constraints=[
                "maintain_functionality",
                "preserve_optimization_techniques",
                "focus_on_bottlenecks",
            ],
            optimization_targets=[
                "reduce_cache_misses",
                "improve_memory_bandwidth",
                "increase_vectorization_efficiency",
            ],
            performance_metrics={
                "current_gflops": 85.2,
                "current_bandwidth": 72.1,
                "cache_miss_rate": 0.12,
                "vectorization_coverage": 0.85,
                "load_imbalance": 0.15,
            },
        )

        # Record debugger interaction
        session_manager.add_agent_interaction(
            session.session_id,
            "debugger",
            {
                "template_used": "kernel_debugger",
                "input_data": debugger_input.to_dict(),
                "analysis_target": code_version.version_id,
                "timestamp": datetime.utcnow().isoformat(),
            },
        )

        # Simulate debugger analysis and improvements
        improved_kernel = """
// Further optimized matrix multiplication kernel
#include <immintrin.h>

void matmul_optimized_v2(float* A, float* B, float* C, int N) {
    const int BLOCK_SIZE = 64;  // Increased block size for better cache utilization

    // Prefetch data for better cache performance
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < N; i += BLOCK_SIZE) {
        for (int j = 0; j < N; j += BLOCK_SIZE) {
            // Prefetch next blocks
            _mm_prefetch(&A[i * N], _MM_HINT_T0);
            _mm_prefetch(&B[j], _MM_HINT_T0);

            for (int k = 0; k < N; k += BLOCK_SIZE) {
                for (int ii = i; ii < min(i + BLOCK_SIZE, N); ii++) {
                    for (int jj = j; jj < min(j + BLOCK_SIZE, N); jj += 16) {
                        __m512 sum = _mm512_setzero_ps();

                        // Unrolled loop for better instruction-level parallelism
                        for (int kk = k; kk < min(k + BLOCK_SIZE, N); kk += 4) {
                            __m512 a1 = _mm512_set1_ps(A[ii * N + kk]);
                            __m512 a2 = _mm512_set1_ps(A[ii * N + kk + 1]);
                            __m512 a3 = _mm512_set1_ps(A[ii * N + kk + 2]);
                            __m512 a4 = _mm512_set1_ps(A[ii * N + kk + 3]);

                            __m512 b1 = _mm512_loadu_ps(&B[kk * N + jj]);
                            __m512 b2 = _mm512_loadu_ps(&B[(kk + 1) * N + jj]);
                            __m512 b3 = _mm512_loadu_ps(&B[(kk + 2) * N + jj]);
                            __m512 b4 = _mm512_loadu_ps(&B[(kk + 3) * N + jj]);

                            sum = _mm512_fmadd_ps(a1, b1, sum);
                            sum = _mm512_fmadd_ps(a2, b2, sum);
                            sum = _mm512_fmadd_ps(a3, b3, sum);
                            sum = _mm512_fmadd_ps(a4, b4, sum);
                        }
                        _mm512_storeu_ps(&C[ii * N + jj], sum);
                    }
                }
            }
        }
    }
}
"""

        # Create improved code version
        improved_version = CodeVersion.create_new_version(
            session_id=session.session_id,
            code=improved_kernel,
            language="c",
            kernel_type="matrix_multiplication",
            source_agent="debugger",
            description="Debugger-optimized kernel with prefetching and loop unrolling",
            optimization_techniques=[
                "prefetching",
                "loop_unrolling",
                "cache_optimization",
            ],
            parent_version_id=code_version.version_id,
        )

        memory_manager.add_code_version(session.session_id, improved_version)
        session_manager.add_version_reference(
            session.session_id, "memory", improved_version.version_id
        )

        # Add optimization knowledge
        debug_knowledge = KnowledgeFragment.create_fragment(
            session_id=session.session_id,
            agent_type="debugger",
            category=KnowledgeCategory.OPTIMIZATION,
            title="Cache optimization through prefetching and loop unrolling for matrix multiplication",
            content="Cache optimization through prefetching and loop unrolling for matrix multiplication",
            content_type=KnowledgeContentType.TEXT,
            metadata={
                "technique": "prefetching_unrolling",
                "improvement": "8.6% performance increase",
                "cache_miss_reduction": "33%",
                "bandwidth_improvement": "8.6%",
            },
        )

        knowledge_manager.add_fragment(debug_knowledge)
        session_manager.add_version_reference(
            session.session_id, "knowledge", debug_knowledge.fragment_id
        )

        # Verify debugger workflow
        code_memory = memory_manager.get_code_memory(session.session_id)
        assert improved_version.version_id in code_memory.versions
        assert debug_knowledge.fragment_id in [
            f.fragment_id
            for f in knowledge_manager.get_session_fragments(session.session_id)
        ]

        session_data = session_manager.get_session(session.session_id)
        assert len(session_data.agent_interactions) == 2  # generator + debugger
        assert len(session_data.memory_versions) == 2
        assert len(session_data.knowledge_versions) == 2

        return session, improved_version, debug_knowledge

    def test_kernel_evaluator_workflow(
        self, prompt_manager, session_manager, memory_manager, knowledge_manager
    ):
        """Test kernel evaluator agent workflow."""
        # Get session and improved code from previous tests
        session, improved_version, debug_knowledge = self.test_kernel_debugger_workflow(
            prompt_manager, session_manager, memory_manager, knowledge_manager
        )

        # Create evaluator template
        prompt_manager.create_template(
            template_name="kernel_evaluator",
            content="""
Evaluate the following optimized kernel:
Kernel implementation: {{kernel_code}}
Benchmark results: {{benchmark_results}}
Performance metrics: {{performance_metrics}}
Target requirements: {{target_requirements}}

Evaluation criteria:
- Computational efficiency: {{comp_efficiency}}
- Memory bandwidth utilization: {{memory_efficiency}}
- Cache hit rates: {{cache_efficiency}}
- Vectorization coverage: {{vectorization_coverage}}
- Scalability: {{scalability_metrics}}

Please provide:
- Performance evaluation summary
- Efficiency analysis (FLOPS, bandwidth, cache)
- Scalability assessment
- Optimization quality score
- Further improvement suggestions
- Confidence in optimization results
""",
            agent_type=AgentType.EVALUATOR,
            prompt_type=PromptType.CODE_EVALUATION,
            description="Kernel performance evaluation",
            tags=["performance-evaluation", "benchmarking", "hpc"],
        )

        # Create evaluator input
        evaluator_input = StructuredInput(
            code_snippet=improved_version.code,
            requirements={
                "evaluation_type": "comprehensive_performance",
                "target_metrics": [
                    "gflops",
                    "bandwidth",
                    "cache_efficiency",
                    "scalability",
                ],
            },
            constraints=[
                "objective_assessment",
                "quantitative_analysis",
                "scalability_evaluation",
            ],
            optimization_targets=[
                "performance_score",
                "efficiency_rating",
                "scalability_score",
            ],
            performance_metrics={
                "measured_gflops": 92.5,
                "measured_bandwidth": 78.3,
                "cache_hit_rate": 0.92,
                "vectorization_coverage": 0.88,
                "scalability_factor": 0.85,
                "power_efficiency": 0.78,
            },
        )

        # Record evaluator interaction
        session_manager.add_agent_interaction(
            session.session_id,
            "evaluator",
            {
                "template_used": "kernel_evaluator",
                "input_data": evaluator_input.to_dict(),
                "evaluation_target": improved_version.version_id,
                "timestamp": datetime.utcnow().isoformat(),
            },
        )

        # Simulate evaluation results
        evaluation_results = {
            "performance_score": 8.5,
            "efficiency_rating": 0.85,
            "scalability_score": 0.82,
            "optimization_quality": "excellent",
            "recommendations": [
                "Consider NUMA-aware parallelization for larger matrices",
                "Explore mixed precision for further performance gains",
                "Investigate power-aware optimization techniques",
            ],
            "confidence_score": 0.88,
        }

        # Store evaluation in memory
        evaluation_memory = AgentMemory(
            session_id=session.session_id,
            agent_type="evaluator",
            version_id="evaluation_v1",
            input_data={},
            output_data={
                "evaluation_results": evaluation_results,
                "performance_metrics": evaluator_input.performance_metrics,
                "recommendations": evaluation_results["recommendations"],
            },
            processing_time_ms=150,
            status="success",
        )

        memory_manager.store_agent_memory(evaluation_memory)

        # Add evaluation knowledge
        evaluation_knowledge = KnowledgeFragment.create_fragment(
            session_id=session.session_id,
            agent_type="evaluator",
            category=KnowledgeCategory.OPTIMIZATION,
            title="Comprehensive evaluation of optimized matrix multiplication kernel",
            content="Comprehensive evaluation of optimized matrix multiplication kernel",
            content_type=KnowledgeContentType.TEXT,
            metadata={
                "performance_score": 8.5,
                "efficiency_rating": 0.85,
                "scalability_score": 0.82,
                "optimization_quality": "excellent",
                "confidence_score": 0.88,
            },
        )

        knowledge_manager.add_fragment(evaluation_knowledge)
        session_manager.add_version_reference(
            session.session_id, "knowledge", evaluation_knowledge.fragment_id
        )

        # Add performance metrics
        memory_manager.add_performance_metrics(
            session_id=session.session_id,
            code_version_id=improved_version.version_id,
            agent_type="evaluator",
            execution_time_ms=45.2,
            memory_usage_mb=128.5,
            cache_miss_rate=0.08,
            cpu_utilization=85.3,
            throughput=92.5,
            latency=0.001,
            power_consumption=78.3,
        )

        # Verify complete workflow
        session_data = session_manager.get_session(session.session_id)
        assert (
            len(session_data.agent_interactions) == 3
        )  # generator + debugger + evaluator
        assert len(session_data.memory_versions) == 2
        assert len(session_data.knowledge_versions) == 3

        # Verify all components are properly linked
        code_memory = memory_manager.get_code_memory(session.session_id)
        assert len(code_memory.versions) == 2

        knowledge_fragments = [
            f.fragment_id
            for f in knowledge_manager.get_session_fragments(session.session_id)
        ]
        assert len(knowledge_fragments) == 3

        agent_memories = memory_manager.get_agent_memories(session.session_id)
        assert len(agent_memories) >= 1  # At least evaluator memory

        return session, evaluation_results

    def test_workflow_optimization_iteration(
        self, prompt_manager, session_manager, memory_manager, knowledge_manager
    ):
        """Test complete optimization iteration workflow."""
        # Run complete workflow
        session, evaluation_results = self.test_kernel_evaluator_workflow(
            prompt_manager, session_manager, memory_manager, knowledge_manager
        )

        # Add optimization iteration to session
        iteration_data = {
            "iteration_number": 1,
            "agents_involved": ["generator", "debugger", "evaluator"],
            "performance_improvement": {
                "gflops": "7.2x improvement",
                "bandwidth": "1.7x improvement",
                "cache_efficiency": "33% improvement",
            },
            "optimization_techniques": [
                "AVX-512 vectorization",
                "Blocking/tiling",
                "Loop unrolling",
                "Memory prefetching",
            ],
            "evaluation_score": evaluation_results["performance_score"],
        }

        session_manager.add_optimization_iteration(session.session_id, iteration_data)

        # Add performance trend point
        performance_trend = {
            "gflops": 92.5,
            "memory_bandwidth": 78.3,
            "cache_hit_rate": 0.92,
            "vectorization_coverage": 0.88,
            "optimization_quality": evaluation_results["optimization_quality"],
        }

        session_manager.add_performance_metrics(session.session_id, performance_trend)

        # Verify complete workflow data
        session_data = session_manager.get_session(session.session_id)

        # Check session summary
        summary = session_data.get_optimization_summary()
        assert summary["total_iterations"] == 1
        assert summary["total_agent_interactions"] == 3
        assert summary["performance_trend_length"] == 1
        assert summary["status"] == "active"

        # Check agent interactions by type
        generator_interactions = session_data.get_agent_interactions_by_type(
            "generator"
        )
        debugger_interactions = session_data.get_agent_interactions_by_type("debugger")
        evaluator_interactions = session_data.get_agent_interactions_by_type(
            "evaluator"
        )

        assert len(generator_interactions) == 1
        assert len(debugger_interactions) == 1
        assert len(evaluator_interactions) == 1

        # Check version references
        assert len(session_data.memory_versions) == 2
        assert len(session_data.knowledge_versions) == 3
        assert len(session_data.prompt_versions) == 0  # No prompt versions in this test

        # Check performance trend
        latest_performance = session_data.get_latest_performance_metrics()
        assert latest_performance is not None
        assert latest_performance["metrics"]["gflops"] == 92.5

        return session, summary

    def test_workflow_session_export(
        self, prompt_manager, session_manager, memory_manager, knowledge_manager
    ):
        """Test exporting complete workflow session."""
        # Run complete workflow
        session, summary = self.test_workflow_optimization_iteration(
            prompt_manager, session_manager, memory_manager, knowledge_manager
        )

        # Export session
        export = session_manager.export_session(
            session.session_id, include_module_data=True
        )

        # Verify export contains all workflow data
        assert export.session.session_id == session.session_id
        # The task description comes from the kernel_generator_workflow test
        assert (
            export.session.task_description
            == "Generate optimized matrix multiplication kernel"
        )
        assert len(export.session.agent_interactions) == 3
        assert len(export.session.optimization_iterations) == 1
        assert len(export.session.performance_trend) == 1

        # Verify session statistics
        stats = session_manager.get_statistics()
        assert stats["total_sessions"] >= 1
        assert stats["total_agent_interactions"] >= 3
        assert stats["total_optimization_iterations"] >= 1

        return export

    def test_workflow_session_analysis(
        self, prompt_manager, session_manager, memory_manager, knowledge_manager
    ):
        """Test analysis of workflow session data."""
        # Run complete workflow
        session, summary = self.test_workflow_optimization_iteration(
            prompt_manager, session_manager, memory_manager, knowledge_manager
        )

        # Analyze session performance
        analysis = session_manager.analyze_session_performance(session.session_id)

        # Verify analysis results
        # Note: analyze_session_performance returns empty dict for now
        # The test should be updated when this method is implemented
        assert isinstance(analysis, dict)
        assert analysis["total_iterations"] == 1
        assert analysis["performance_points"] == 1
        assert "agent_interaction_counts" in analysis
        assert analysis["agent_interaction_counts"]["generator"] == 1
        assert analysis["agent_interaction_counts"]["debugger"] == 1
        assert analysis["agent_interaction_counts"]["evaluator"] == 1

        # Generate session report
        report = session_manager.generate_session_report(session)

        # Verify report structure
        assert report["session_id"] == session.session_id
        # The task description comes from the kernel_generator_workflow test
        assert (
            report["task_description"]
            == "Generate optimized matrix multiplication kernel"
        )
        assert report["status"] == "active"
        assert "optimization_summary" in report
        assert "performance_analysis" in report
        assert "version_references" in report

        # Verify version references in report
        version_refs = report["version_references"]
        assert version_refs["memory_versions"] == 2
        assert version_refs["knowledge_versions"] == 3

        return analysis, report
