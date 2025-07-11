"""
Prompt module usage example for high-performance kernel optimization.

This example demonstrates the new prompt module features in the context
of high-performance computing, kernel optimization, and numerical algorithms.
"""

import shutil
import tempfile
from pathlib import Path

from pinocchio.prompt import (
    AgentType,
    FileTemplateLoader,
    PromptManager,
    PromptTemplate,
    PromptType,
    StructuredInput,
    VersionControl,
    VersionStatus,
)


def main():
    """Run example for high-performance kernel optimization."""
    print("=== Pinocchio Prompt Module: High-Performance Kernel Optimization ===\n")

    # Create temporary directory for storage
    temp_dir = tempfile.mkdtemp()
    try:
        # Initialize prompt manager
        manager = PromptManager(storage_path=temp_dir)

        print("1. Creating templates for kernel optimization agents...")

        # Create kernel generator template
        generator_template = manager.create_template(
            template_name="kernel_generator",
            content="""
Generate optimized kernel for the following computation:
Algorithm: {{algorithm_type}}
Input dimensions: {{input_dims}}
Output dimensions: {{output_dims}}
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
            description="High-performance kernel generator for numerical algorithms",
            tags=["kernel-optimization", "vectorization", "parallelization", "hpc"],
        )

        # Create kernel debugger template
        debugger_template = manager.create_template(
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
            description="Kernel performance debugging and optimization",
            tags=[
                "performance-debugging",
                "memory-optimization",
                "cache-analysis",
                "hpc",
            ],
        )

        # Create kernel evaluator template
        evaluator_template = manager.create_template(
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
            description="Kernel performance evaluation and analysis",
            tags=["performance-evaluation", "benchmarking", "scalability", "hpc"],
        )

        print(
            f"Created {generator_template.template_name} for {generator_template.agent_type.value}"
        )
        print(
            f"Created {debugger_template.template_name} for {debugger_template.agent_type.value}"
        )
        print(
            f"Created {evaluator_template.template_name} for {evaluator_template.agent_type.value}"
        )

        print("\n2. Testing structured input/output for kernel optimization...")

        # Create structured input for matrix multiplication kernel
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

        # Format template with structured input
        formatted_prompt = manager.format_structured_prompt(
            "kernel_generator", generator_input, agent_type=AgentType.GENERATOR
        )

        print("Formatted kernel generator prompt:")
        print(
            formatted_prompt[:300] + "..."
            if len(formatted_prompt) > 300
            else formatted_prompt
        )

        print("\n3. Testing template management for HPC workflows...")

        # List all templates
        all_templates = manager.list_templates()
        print(f"All HPC templates: {list(all_templates.keys())}")

        # List templates by agent type
        generator_templates = manager.list_templates(agent_type=AgentType.GENERATOR)
        print(f"Kernel generator templates: {list(generator_templates.keys())}")

        # Search templates for optimization techniques
        search_results = manager.search_templates("vectorization")
        print(f"Templates with vectorization: {len(search_results)} found")

        print("\n4. Testing version control for kernel optimization...")

        # Initialize version control
        vc = VersionControl(storage_path=temp_dir + "/version_control")

        # Create initial version
        version1 = vc.create_version(
            generator_template,
            branch_name="main",
            description="Initial kernel generator with basic optimization",
            created_by="hpc_engineer",
            status=VersionStatus.DRAFT,
        )

        # Create updated template with advanced optimization
        updated_template = PromptTemplate.create_new_version(
            template_name="kernel_generator",
            content="""
Generate optimized kernel with advanced techniques:
Algorithm: {{algorithm_type}}
Input dimensions: {{input_dims}}
Target architecture: {{target_arch}}

Advanced optimization requirements:
- Blocking/tiling for cache optimization
- SIMD vectorization with AVX-512/NEON/SVE
- Memory prefetching strategies
- NUMA-aware parallelization
- Load balancing for heterogeneous cores
- Power-aware optimization

Please provide:
- Blocked/tiled kernel implementation
- Advanced vectorization with intrinsics
- Memory access pattern optimization
- NUMA-aware parallelization
- Performance analysis with roofline model
- Power efficiency considerations
""",
            agent_type=AgentType.GENERATOR,
            prompt_type=PromptType.CODE_GENERATION,
            parent_version_id=version1.version_id,
        )

        version2 = vc.create_version(
            updated_template,
            branch_name="main",
            description="Advanced kernel optimization with blocking and NUMA awareness",
            created_by="hpc_engineer",
            status=VersionStatus.REVIEW,
        )

        # List versions
        versions = vc.list_versions("kernel_generator")
        print(f"Versions of kernel_generator: {len(versions)}")

        # Get version history
        history = vc.get_version_history("kernel_generator", version2.version_id)
        print(f"Version history length: {len(history)}")

        print("\n5. Testing template loading for HPC workflows...")

        # Create file loader
        loader_dir = temp_dir + "/templates"
        Path(loader_dir).mkdir(exist_ok=True)

        # Save kernel template to file
        import json

        template_file = Path(loader_dir) / "kernel_optimization_template.json"
        with open(template_file, "w") as f:
            json.dump(generator_template.to_dict(), f, indent=2)

        # Create loader and load template
        loader = FileTemplateLoader(loader_dir)
        loaded_template = loader.load_template("kernel_optimization_template")

        if loaded_template:
            print(f"Loaded kernel template: {loaded_template.template_name}")
            print(f"Agent type: {loaded_template.agent_type.value}")

        print("\n6. Testing performance tracking for kernel optimization...")

        # Update template statistics with realistic HPC metrics
        manager.update_template_stats("kernel_generator", True, 0.8)  # 80% success rate
        manager.update_template_stats("kernel_generator", True, 0.6)  # 60% success rate
        manager.update_template_stats(
            "kernel_generator", False, 0.9
        )  # 90% response time

        # Get performance stats
        stats = manager.get_performance_stats()
        print(f"Overall success rate: {stats['overall_success_rate']:.2f}")
        print(f"Average response time: {stats['average_response_time']:.2f}s")

        print("\n7. Testing template export/import for HPC workflows...")

        # Export template
        exported_json = manager.export_template("kernel_generator", format="json")
        print(f"Exported kernel template size: {len(exported_json)} characters")

        # Import template
        imported_template = manager.import_template(exported_json, format="json")
        print(f"Imported kernel template: {imported_template.template_name}")

        print("\n8. Demonstrating kernel optimization workflow...")

        # Simulate a complete kernel optimization workflow
        print("Simulating kernel optimization workflow:")
        print("- Generator creates optimized matrix multiplication kernel")
        print("- Debugger identifies memory bandwidth bottlenecks")
        print("- Evaluator assesses performance improvements")
        print("- Version control tracks optimization iterations")
        print("- Performance tracking monitors optimization success")

        print("\n=== High-Performance Kernel Optimization Example Completed! ===")
        print("This example demonstrates how the prompt module supports")
        print("advanced HPC workflows with kernel optimization, vectorization,")
        print("parallelization, and performance analysis.")

    finally:
        # Clean up
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    main()
