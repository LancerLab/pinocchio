"""
CUDA performance evaluation tools for MCP integration.
Provides tools for performance profiling, occupancy analysis, and optimization guidance.
"""

import json
import logging
import math
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from .base import MCPTool, ToolExecutionStatus, ToolResult

logger = logging.getLogger(__name__)


class CudaProfilerTool(MCPTool):
    """Tool for CUDA performance profiling using nvprof/nsys."""

    def __init__(self, timeout: int = 180):
        """Initialize the CudaProfilerTool class."""
        super().__init__(
            name="cuda_profile",
            description="Profile CUDA application performance",
            timeout=timeout,
        )

    def execute(
        self,
        cuda_code: str,
        host_code: Optional[str] = None,
        profiler: str = "nvprof",
        metrics: Optional[List[str]] = None,
        arch: str = "compute_75",
    ) -> ToolResult:
        """
        Profile CUDA application performance.

        Args:
            cuda_code: CUDA kernel code
            host_code: Host code to run the kernel
            profiler: Profiler to use (nvprof, nsys)
            metrics: Specific metrics to collect
            arch: Target architecture

        Returns:
            ToolResult: Profiling result
        """
        # Create complete CUDA program
        if host_code is None:
            host_code = self._generate_performance_host_code()

        complete_program = f"{cuda_code}\n\n{host_code}"

        # Create temporary files
        cu_file = self._create_temp_file(complete_program, ".cu")
        exe_file = cu_file.replace(".cu", ".exe")

        try:
            # Compile with optimization flags
            compile_cmd = [
                "nvcc",
                cu_file,
                "-o",
                exe_file,
                f"-arch={arch}",
                "-O3",
                "-lineinfo",
            ]
            compile_result = self._run_command(compile_cmd)

            if compile_result.status != ToolExecutionStatus.SUCCESS:
                return ToolResult(
                    status=ToolExecutionStatus.ERROR,
                    output="",
                    error=f"Compilation failed: {compile_result.error}",
                    metadata={"compilation_output": compile_result.output},
                )

            # Run profiler
            if profiler == "nvprof":
                result = self._run_nvprof(exe_file, metrics)
            elif profiler == "nsys":
                result = self._run_nsys(exe_file)
            else:
                return ToolResult(
                    status=ToolExecutionStatus.ERROR,
                    output="",
                    error=f"Unknown profiler: {profiler}",
                )

            # Parse profiling output
            if result.status == ToolExecutionStatus.SUCCESS:
                result.metadata = {
                    "profiler": profiler,
                    "executable": exe_file,
                    "architecture": arch,
                    "metrics_requested": metrics or [],
                    "performance_analysis": self._parse_profiling_output(
                        result.output, profiler
                    ),
                }

            return result

        finally:
            # Cleanup temporary files
            self._cleanup_temp_file(cu_file)
            if os.path.exists(exe_file):
                self._cleanup_temp_file(exe_file)

    def _generate_performance_host_code(self) -> str:
        """Generate host code for performance testing."""
        return """
#include <cuda_runtime.h>
#include <stdio.h>
#include <time.h>

int main() {
    // Setup for performance measurement
    const int size = 1024 * 1024;
    const int bytes = size * sizeof(float);

    // Allocate host memory
    float *h_a = (float*)malloc(bytes);
    float *h_b = (float*)malloc(bytes);
    float *h_c = (float*)malloc(bytes);

    // Initialize data
    for (int i = 0; i < size; i++) {
        h_a[i] = (float)i;
        h_b[i] = (float)(i * 2);
    }

    // Allocate device memory
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    // Copy data to device
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    // Setup execution configuration
    dim3 block(256);
    dim3 grid((size + block.x - 1) / block.x);

    // Warm up run
    // testKernel<<<grid, block>>>(d_a, d_b, d_c, size);
    cudaDeviceSynchronize();

    // Create events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Time multiple kernel runs
    cudaEventRecord(start);
    for (int i = 0; i < 100; i++) {
        // testKernel<<<grid, block>>>(d_a, d_b, d_c, size);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("Average kernel time: %.3f ms\\n", milliseconds / 100.0f);

    // Copy result back
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
"""

    def _run_nvprof(
        self, exe_file: str, metrics: Optional[List[str]] = None
    ) -> ToolResult:
        """Run nvprof profiler."""
        cmd = ["nvprof"]

        # Add common metrics
        default_metrics = [
            "gld_efficiency",
            "gst_efficiency",
            "sm_efficiency",
            "achieved_occupancy",
            "branch_efficiency",
        ]

        metrics_to_use = metrics if metrics else default_metrics
        for metric in metrics_to_use:
            cmd.extend(["--metrics", metric])

        # Add events
        cmd.extend(["--events", "elapsed_cycles_sm"])

        # Add executable
        cmd.append(exe_file)

        return self._run_command(cmd)

    def _run_nsys(self, exe_file: str) -> ToolResult:
        """Run nsys profiler."""
        cmd = [
            "nsys",
            "profile",
            "--stats=true",
            "--output=/tmp/profile_output",
            exe_file,
        ]

        return self._run_command(cmd)

    def _parse_profiling_output(self, output: str, profiler: str) -> Dict[str, Any]:
        """Parse profiling output for structured data."""
        if profiler == "nvprof":
            return self._parse_nvprof_output(output)
        elif profiler == "nsys":
            return self._parse_nsys_output(output)
        else:
            return {"raw_output": output}

    def _parse_nvprof_output(self, output: str) -> Dict[str, Any]:
        """Parse nvprof output."""
        metrics = {}
        kernel_stats = {}

        lines = output.split("\n")
        current_section = None

        for line in lines:
            line = line.strip()

            # Parse metrics
            if "Metric result:" in line:
                current_section = "metrics"
            elif current_section == "metrics" and "=" in line:
                parts = line.split("=")
                if len(parts) == 2:
                    metric_name = parts[0].strip()
                    metric_value = parts[1].strip()
                    metrics[metric_name] = metric_value

            # Parse kernel timing
            if "GPU activities:" in line:
                current_section = "gpu_activities"
            elif current_section == "gpu_activities" and "%" in line:
                # Parse timing information
                parts = line.split()
                if len(parts) >= 4:
                    try:
                        time_percent = parts[0].replace("%", "")
                        avg_time = parts[1]
                        kernel_stats["time_percentage"] = float(time_percent)
                        kernel_stats["average_time"] = avg_time
                    except Exception:
                        pass

        return {
            "metrics": metrics,
            "kernel_statistics": kernel_stats,
            "profiler": "nvprof",
        }

    def _parse_nsys_output(self, output: str) -> Dict[str, Any]:
        """Parse nsys output."""
        # Simplified nsys parsing
        return {
            "raw_output": output,
            "profiler": "nsys",
            "note": "Detailed nsys parsing requires additional processing",
        }

    def get_schema(self) -> Dict[str, Any]:
        """Return the JSON schema for the tool's input parameters."""
        return {
            "type": "object",
            "properties": {
                "cuda_code": {
                    "type": "string",
                    "description": "CUDA kernel code to profile",
                },
                "host_code": {
                    "type": "string",
                    "description": "Host code to run the kernel (optional)",
                },
                "profiler": {
                    "type": "string",
                    "enum": ["nvprof", "nsys"],
                    "description": "Profiler to use",
                    "default": "nvprof",
                },
                "metrics": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Specific metrics to collect",
                },
                "arch": {
                    "type": "string",
                    "description": "Target GPU architecture",
                    "default": "compute_75",
                },
            },
            "required": ["cuda_code"],
        }


class CudaOccupancyCalculator(MCPTool):
    """Tool for calculating theoretical CUDA kernel occupancy."""

    def __init__(self, timeout: int = 30):
        """Initialize the CudaOccupancyCalculator class."""
        super().__init__(
            name="cuda_occupancy",
            description="Calculate theoretical kernel occupancy",
            timeout=timeout,
        )

    def execute(
        self,
        cuda_code: str,
        block_size: int = 256,
        arch: str = "compute_75",
        registers_per_thread: Optional[int] = None,
        shared_memory_per_block: Optional[int] = None,
    ) -> ToolResult:
        """
        Calculate theoretical kernel occupancy.

        Args:
            cuda_code: CUDA kernel code
            block_size: Threads per block
            arch: Target architecture
            registers_per_thread: Registers used per thread
            shared_memory_per_block: Shared memory used per block

        Returns:
            ToolResult: Occupancy calculation result
        """
        try:
            # Get device properties for architecture
            device_props = self._get_device_properties(arch)

            # Estimate resource usage if not provided
            if registers_per_thread is None:
                registers_per_thread = self._estimate_register_usage(cuda_code)

            if shared_memory_per_block is None:
                shared_memory_per_block = self._estimate_shared_memory_usage(cuda_code)

            # Calculate occupancy
            occupancy_result = self._calculate_occupancy(
                device_props, block_size, registers_per_thread, shared_memory_per_block
            )

            # Generate optimization suggestions
            suggestions = self._generate_optimization_suggestions(
                occupancy_result,
                device_props,
                block_size,
                registers_per_thread,
                shared_memory_per_block,
            )

            output = self._format_occupancy_result(occupancy_result, suggestions)

            return ToolResult(
                status=ToolExecutionStatus.SUCCESS,
                output=output,
                metadata={
                    "architecture": arch,
                    "block_size": block_size,
                    "registers_per_thread": registers_per_thread,
                    "shared_memory_per_block": shared_memory_per_block,
                    "occupancy_result": occupancy_result,
                    "suggestions": suggestions,
                    "device_properties": device_props,
                },
            )

        except Exception as e:
            return ToolResult(
                status=ToolExecutionStatus.ERROR,
                output="",
                error=f"Occupancy calculation failed: {str(e)}",
            )

    def _get_device_properties(self, arch: str) -> Dict[str, int]:
        """Get device properties for given architecture."""
        # Simplified device properties for common architectures
        properties = {
            "compute_75": {  # RTX 20xx, Tesla V100
                "max_threads_per_sm": 2048,
                "max_blocks_per_sm": 16,
                "max_registers_per_sm": 65536,
                "max_shared_memory_per_sm": 65536,
                "max_shared_memory_per_block": 49152,
                "register_file_size": 65536,
                "warp_size": 32,
            },
            "compute_80": {  # RTX 30xx, A100
                "max_threads_per_sm": 2048,
                "max_blocks_per_sm": 16,
                "max_registers_per_sm": 65536,
                "max_shared_memory_per_sm": 102400,
                "max_shared_memory_per_block": 49152,
                "register_file_size": 65536,
                "warp_size": 32,
            },
            "compute_86": {  # RTX 30xx mobile
                "max_threads_per_sm": 1536,
                "max_blocks_per_sm": 16,
                "max_registers_per_sm": 65536,
                "max_shared_memory_per_sm": 102400,
                "max_shared_memory_per_block": 49152,
                "register_file_size": 65536,
                "warp_size": 32,
            },
        }

        return properties.get(arch, properties["compute_75"])

    def _estimate_register_usage(self, cuda_code: str) -> int:
        """Estimate register usage from kernel code."""
        # Simple heuristic based on variable declarations and operations
        lines = cuda_code.split("\n")
        register_estimate = 16  # Base estimate

        for line in lines:
            # Count variable declarations
            if any(keyword in line for keyword in ["float", "int", "double"]):
                register_estimate += 2

            # Count mathematical operations
            if any(op in line for op in ["+", "-", "*", "/", "sqrt", "sin", "cos"]):
                register_estimate += 1

        return min(register_estimate, 63)  # Cap at typical max per thread

    def _estimate_shared_memory_usage(self, cuda_code: str) -> int:
        """Estimate shared memory usage from kernel code."""
        shared_memory = 0
        lines = cuda_code.split("\n")

        for line in lines:
            if "__shared__" in line:
                # Try to extract array size
                if "[" in line and "]" in line:
                    try:
                        size_str = line[line.find("[") + 1 : line.find("]")]
                        if size_str.isdigit():
                            shared_memory += int(size_str) * 4  # Assume float
                        else:
                            shared_memory += 1024  # Default estimate
                    except Exception:
                        shared_memory += 1024

        return shared_memory

    def _calculate_occupancy(
        self,
        device_props: Dict[str, int],
        block_size: int,
        registers_per_thread: int,
        shared_memory_per_block: int,
    ) -> Dict[str, Any]:
        """Calculate occupancy based on resource limitations."""
        max_threads_per_sm = device_props["max_threads_per_sm"]
        max_blocks_per_sm = device_props["max_blocks_per_sm"]
        max_registers_per_sm = device_props["max_registers_per_sm"]
        max_shared_memory_per_sm = device_props["max_shared_memory_per_sm"]
        warp_size = device_props["warp_size"]

        # Calculate warps per block
        warps_per_block = math.ceil(block_size / warp_size)

        # Calculate limitations
        blocks_by_threads = max_threads_per_sm // block_size
        blocks_by_registers = (
            max_registers_per_sm // (registers_per_thread * block_size)
            if registers_per_thread > 0
            else float("inf")
        )
        blocks_by_shared_memory = (
            max_shared_memory_per_sm // shared_memory_per_block
            if shared_memory_per_block > 0
            else float("inf")
        )
        blocks_by_sm_limit = max_blocks_per_sm

        # Find limiting factor
        max_blocks = min(
            blocks_by_threads,
            blocks_by_registers,
            blocks_by_shared_memory,
            blocks_by_sm_limit,
        )
        max_blocks = max(0, int(max_blocks))

        # Calculate occupancy
        theoretical_warps = max_blocks * warps_per_block
        max_warps_per_sm = max_threads_per_sm // warp_size
        occupancy = (
            min(theoretical_warps / max_warps_per_sm, 1.0)
            if max_warps_per_sm > 0
            else 0
        )

        return {
            "occupancy_percentage": occupancy * 100,
            "active_warps": theoretical_warps,
            "max_warps_per_sm": max_warps_per_sm,
            "active_blocks": max_blocks,
            "warps_per_block": warps_per_block,
            "limiting_factors": {
                "threads": blocks_by_threads,
                "registers": blocks_by_registers,
                "shared_memory": blocks_by_shared_memory,
                "sm_limit": blocks_by_sm_limit,
            },
        }

    def _generate_optimization_suggestions(
        self,
        occupancy_result: Dict[str, Any],
        device_props: Dict[str, int],
        block_size: int,
        registers_per_thread: int,
        shared_memory_per_block: int,
    ) -> List[str]:
        """Generate optimization suggestions based on occupancy analysis."""
        suggestions = []
        occupancy = occupancy_result["occupancy_percentage"]
        limiting = occupancy_result["limiting_factors"]

        if occupancy < 50:
            suggestions.append("Low occupancy detected. Consider optimization.")

        # Find most restrictive limitation
        min_limit = min(
            limiting["threads"],
            limiting["registers"],
            limiting["shared_memory"],
            limiting["sm_limit"],
        )

        if (
            limiting["registers"] == min_limit
            and limiting["registers"] < limiting["threads"]
        ):
            suggestions.append(
                f"Register usage is limiting (estimated {registers_per_thread} per thread). "
                "Consider reducing local variables or using shared memory."
            )

        if limiting["shared_memory"] == min_limit and shared_memory_per_block > 0:
            suggestions.append(
                f"Shared memory usage is limiting ({shared_memory_per_block} bytes per block). "
                "Consider reducing shared memory usage or using smaller blocks."
            )

        if limiting["threads"] == min_limit:
            suggestions.append(
                "Thread count per block may be suboptimal. "
                "Try different block sizes (multiples of warp size)."
            )

        # Block size suggestions
        if block_size % 32 != 0:
            suggestions.append(
                "Block size should be a multiple of warp size (32) for optimal efficiency."
            )

        if block_size < 128:
            suggestions.append("Consider larger block sizes for better occupancy.")
        elif block_size > 512:
            suggestions.append(
                "Very large block sizes may limit occupancy due to resource constraints."
            )

        return suggestions

    def _format_occupancy_result(
        self, occupancy_result: Dict[str, Any], suggestions: List[str]
    ) -> str:
        """Format occupancy calculation results."""
        output = []
        output.append("=== OCCUPANCY ANALYSIS ===")
        output.append(
            f"Theoretical Occupancy: {occupancy_result['occupancy_percentage']:.1f}%"
        )
        output.append(f"Active Warps: {occupancy_result['active_warps']}")
        output.append(f"Active Blocks: {occupancy_result['active_blocks']}")
        output.append(f"Warps per Block: {occupancy_result['warps_per_block']}")

        output.append("\n=== RESOURCE LIMITATIONS ===")
        limiting = occupancy_result["limiting_factors"]
        output.append(f"Max blocks by thread count: {limiting['threads']}")
        output.append(f"Max blocks by register usage: {limiting['registers']:.1f}")
        output.append(f"Max blocks by shared memory: {limiting['shared_memory']:.1f}")
        output.append(f"Max blocks by SM limit: {limiting['sm_limit']}")

        if suggestions:
            output.append("\n=== OPTIMIZATION SUGGESTIONS ===")
            for i, suggestion in enumerate(suggestions, 1):
                output.append(f"{i}. {suggestion}")

        return "\n".join(output)

    def get_schema(self) -> Dict[str, Any]:
        """Return the JSON schema for the tool's input parameters."""
        return {
            "type": "object",
            "properties": {
                "cuda_code": {
                    "type": "string",
                    "description": "CUDA kernel code to analyze",
                },
                "block_size": {
                    "type": "integer",
                    "description": "Threads per block",
                    "default": 256,
                },
                "arch": {
                    "type": "string",
                    "description": "Target GPU architecture",
                    "default": "compute_75",
                },
                "registers_per_thread": {
                    "type": "integer",
                    "description": "Registers used per thread (estimated if not provided)",
                },
                "shared_memory_per_block": {
                    "type": "integer",
                    "description": "Shared memory used per block in bytes (estimated if not provided)",
                },
            },
            "required": ["cuda_code"],
        }


class CudaPerformanceAnalyzer(MCPTool):
    """Tool for comprehensive CUDA performance analysis."""

    def __init__(self, timeout: int = 60):
        """Initialize the CudaPerformanceAnalyzer class."""
        super().__init__(
            name="cuda_performance_analyze",
            description="Analyze CUDA code for performance characteristics",
            timeout=timeout,
        )

    def execute(
        self, cuda_code: str, analysis_type: str = "comprehensive"
    ) -> ToolResult:
        """
        Analyze CUDA code for performance characteristics.

        Args:
            cuda_code: CUDA kernel code to analyze
            analysis_type: Type of analysis (comprehensive, memory, compute)

        Returns:
            ToolResult: Performance analysis result
        """
        try:
            analysis_results = {}

            if analysis_type in ["comprehensive", "memory"]:
                analysis_results["memory_analysis"] = self._analyze_memory_patterns(
                    cuda_code
                )

            if analysis_type in ["comprehensive", "compute"]:
                analysis_results["compute_analysis"] = self._analyze_compute_patterns(
                    cuda_code
                )

            if analysis_type == "comprehensive":
                analysis_results["general_analysis"] = self._analyze_general_patterns(
                    cuda_code
                )

            # Generate performance score
            performance_score = self._calculate_performance_score(analysis_results)

            output = self._format_analysis_results(analysis_results, performance_score)

            return ToolResult(
                status=ToolExecutionStatus.SUCCESS,
                output=output,
                metadata={
                    "analysis_type": analysis_type,
                    "performance_score": performance_score,
                    "analysis_results": analysis_results,
                },
            )

        except Exception as e:
            return ToolResult(
                status=ToolExecutionStatus.ERROR,
                output="",
                error=f"Performance analysis failed: {str(e)}",
            )

    def _analyze_memory_patterns(self, code: str) -> Dict[str, Any]:
        """Analyze memory access patterns."""
        issues = []
        good_practices = []

        lines = code.split("\n")

        for i, line in enumerate(lines, 1):
            stripped = line.strip()

            # Check for coalesced memory access
            if any(pattern in stripped for pattern in ["threadIdx.x", "blockIdx.x"]):
                if any(
                    bad_pattern in stripped
                    for bad_pattern in ["* threadIdx.x", "threadIdx.x *"]
                ):
                    if "[" in stripped and "]" in stripped:
                        issues.append(
                            {
                                "line": i,
                                "type": "memory_coalescing",
                                "message": "Potential non-coalesced memory access pattern",
                            }
                        )
                elif "threadIdx.x" in stripped and "[" in stripped:
                    good_practices.append(
                        {
                            "line": i,
                            "type": "memory_coalescing",
                            "message": "Good coalesced memory access pattern",
                        }
                    )

            # Check for shared memory usage
            if "__shared__" in stripped:
                good_practices.append(
                    {
                        "line": i,
                        "type": "shared_memory",
                        "message": "Using shared memory for data reuse",
                    }
                )

            # Check for bank conflicts
            if "__shared__" in stripped and "threadIdx.x" in stripped:
                if "* threadIdx.x" in stripped or "threadIdx.x *" in stripped:
                    issues.append(
                        {
                            "line": i,
                            "type": "bank_conflicts",
                            "message": "Potential shared memory bank conflicts",
                        }
                    )

        return {
            "issues": issues,
            "good_practices": good_practices,
            "memory_score": max(0, 100 - len(issues) * 20 + len(good_practices) * 10),
        }

    def _analyze_compute_patterns(self, code: str) -> Dict[str, Any]:
        """Analyze compute patterns and efficiency."""
        issues = []
        good_practices = []

        lines = code.split("\n")

        for i, line in enumerate(lines, 1):
            stripped = line.strip()

            # Check for branch divergence
            if "if" in stripped and any(
                thread_var in stripped for thread_var in ["threadIdx", "blockIdx"]
            ):
                issues.append(
                    {
                        "line": i,
                        "type": "branch_divergence",
                        "message": "Potential branch divergence based on thread ID",
                    }
                )

            # Check for expensive operations
            expensive_ops = ["sin", "cos", "tan", "log", "exp", "pow", "sqrt"]
            if any(op in stripped for op in expensive_ops):
                if "f" not in stripped:  # Using double precision
                    issues.append(
                        {
                            "line": i,
                            "type": "precision",
                            "message": "Using double precision math - consider single precision",
                        }
                    )
                else:
                    good_practices.append(
                        {
                            "line": i,
                            "type": "precision",
                            "message": "Using single precision math functions",
                        }
                    )

            # Check for loop unrolling opportunities
            if "for" in stripped and any(
                small_num in stripped for small_num in ["< 4", "< 8", "< 16"]
            ):
                good_practices.append(
                    {
                        "line": i,
                        "type": "loop_unrolling",
                        "message": "Small loop - consider unrolling",
                    }
                )

        return {
            "issues": issues,
            "good_practices": good_practices,
            "compute_score": max(0, 100 - len(issues) * 15 + len(good_practices) * 10),
        }

    def _analyze_general_patterns(self, code: str) -> Dict[str, Any]:
        """Analyze general CUDA programming patterns."""
        issues = []
        good_practices = []

        # Check for proper thread indexing
        if "threadIdx" in code and "blockIdx" in code and "blockDim" in code:
            good_practices.append(
                {
                    "type": "thread_indexing",
                    "message": "Proper thread indexing pattern detected",
                }
            )

        # Check for bounds checking
        if any(check in code for check in ["if (", "if("]) and any(
            bound in code for bound in ["<", ">=", "return"]
        ):
            good_practices.append(
                {"type": "bounds_checking", "message": "Bounds checking implemented"}
            )

        # Check for register pressure
        float_count = code.count("float")
        int_count = code.count("int")
        if float_count + int_count > 20:
            issues.append(
                {
                    "type": "register_pressure",
                    "message": f"Many variables declared ({float_count + int_count}) - potential register pressure",
                }
            )

        return {
            "issues": issues,
            "good_practices": good_practices,
            "general_score": max(0, 100 - len(issues) * 10 + len(good_practices) * 15),
        }

    def _calculate_performance_score(self, analysis_results: Dict[str, Any]) -> int:
        """Calculate overall performance score."""
        scores = []

        if "memory_analysis" in analysis_results:
            scores.append(analysis_results["memory_analysis"]["memory_score"])

        if "compute_analysis" in analysis_results:
            scores.append(analysis_results["compute_analysis"]["compute_score"])

        if "general_analysis" in analysis_results:
            scores.append(analysis_results["general_analysis"]["general_score"])

        return int(sum(scores) / len(scores)) if scores else 50

    def _format_analysis_results(
        self, analysis_results: Dict[str, Any], performance_score: int
    ) -> str:
        """Format analysis results for output."""
        output = []
        output.append("=== CUDA PERFORMANCE ANALYSIS ===")
        output.append(f"Overall Performance Score: {performance_score}/100")

        for category, results in analysis_results.items():
            output.append(f"\n=== {category.upper().replace('_', ' ')} ===")

            if "issues" in results and results["issues"]:
                output.append("Issues Found:")
                for issue in results["issues"]:
                    if "line" in issue:
                        output.append(f"  Line {issue['line']}: {issue['message']}")
                    else:
                        output.append(f"  {issue['message']}")

            if "good_practices" in results and results["good_practices"]:
                output.append("Good Practices:")
                for practice in results["good_practices"]:
                    if "line" in practice:
                        output.append(
                            f"  Line {practice['line']}: {practice['message']}"
                        )
                    else:
                        output.append(f"  {practice['message']}")

            if f"{category.split('_')[0]}_score" in results:
                score_key = f"{category.split('_')[0]}_score"
                output.append(f"Category Score: {results[score_key]}/100")

        return "\n".join(output)

    def get_schema(self) -> Dict[str, Any]:
        """Return the JSON schema for the tool's input parameters."""
        return {
            "type": "object",
            "properties": {
                "cuda_code": {
                    "type": "string",
                    "description": "CUDA kernel code to analyze",
                },
                "analysis_type": {
                    "type": "string",
                    "enum": ["comprehensive", "memory", "compute"],
                    "description": "Type of performance analysis to perform",
                    "default": "comprehensive",
                },
            },
            "required": ["cuda_code"],
        }


class CudaEvalTools:
    """
    Container class for all CUDA evaluation tools.

    Provides a unified interface to register all evaluation tools.
    """

    def __init__(self):
        """Initialize the CudaEvalTools class."""
        pass

    @staticmethod
    def get_all_tools() -> List[MCPTool]:
        """Get all available CUDA evaluation tools."""
        return [
            CudaProfilerTool(),
            CudaOccupancyCalculator(),
            CudaPerformanceAnalyzer(),
        ]

    @staticmethod
    def register_tools(tool_manager) -> None:
        """Register all CUDA evaluation tools with a tool manager."""
        tools = CudaEvalTools.get_all_tools()
        for tool in tools:
            tool_manager.register_tool(tool)
