"""
CUDA debugging tools for MCP integration.
Provides tools for CUDA code compilation, validation, and debugging.
"""

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

from .base import MCPTool, ToolExecutionStatus, ToolResult

logger = logging.getLogger(__name__)


class CudaCompilerTool(MCPTool):
    """Tool for CUDA code compilation and syntax checking."""

    def __init__(self, timeout: int = 60):
        """Initialize the CudaCompilerTool with a timeout."""
        super().__init__(
            name="cuda_compile",
            description="Compile CUDA code and check for syntax errors",
            timeout=timeout,
        )

    def execute(
        self,
        cuda_code: str,
        arch: str = "compute_75",
        include_paths: Optional[List[str]] = None,
        verbose: bool = False,
    ) -> ToolResult:
        """
        Compile CUDA code and return compilation results.

        Args:
            cuda_code: CUDA source code to compile
            arch: Target architecture (default: compute_75)
            include_paths: Additional include paths
            verbose: Enable verbose compilation output

        Returns:
            ToolResult: Compilation result
        """
        # Create temporary file
        temp_file = self._create_temp_file(cuda_code, ".cu")
        output_file = temp_file.replace(".cu", ".o")

        try:
            # Build nvcc command
            command = ["nvcc", "-c", temp_file, "-o", output_file, f"-arch={arch}"]

            if include_paths:
                for include_path in include_paths:
                    command.extend(["-I", include_path])

            if verbose:
                command.append("-v")

            # Add common flags for better error reporting
            command.extend(["-Xcompiler", "-Wall", "--ptxas-options=-v"])

            result = self._run_command(command)

            # Parse compilation output for detailed information
            metadata = {
                "temp_file": temp_file,
                "output_file": output_file,
                "architecture": arch,
                "include_paths": include_paths or [],
            }

            if result.status == ToolExecutionStatus.SUCCESS:
                metadata["compilation_successful"] = True
                if os.path.exists(output_file):
                    metadata["object_file_size"] = os.path.getsize(output_file)
            else:
                metadata["compilation_successful"] = False
                # Parse error messages for better debugging
                if result.error:
                    metadata["error_analysis"] = self._parse_compilation_errors(
                        result.error
                    )

            result.metadata = metadata
            return result

        finally:
            # Cleanup temporary files
            self._cleanup_temp_file(temp_file)
            if os.path.exists(output_file):
                self._cleanup_temp_file(output_file)

    def _parse_compilation_errors(self, error_output: str) -> Dict[str, Any]:
        """Parse nvcc error output for structured error information."""
        errors = []
        warnings = []

        for line in error_output.split("\n"):
            if "error:" in line.lower():
                errors.append(line.strip())
            elif "warning:" in line.lower():
                warnings.append(line.strip())

        return {
            "error_count": len(errors),
            "warning_count": len(warnings),
            "errors": errors,
            "warnings": warnings,
        }

    def get_schema(self) -> Dict[str, Any]:
        """Return the JSON schema for the tool's input parameters."""
        return {
            "type": "object",
            "properties": {
                "cuda_code": {
                    "type": "string",
                    "description": "CUDA source code to compile",
                },
                "arch": {
                    "type": "string",
                    "description": "Target GPU architecture (e.g., compute_75, compute_80)",
                    "default": "compute_75",
                },
                "include_paths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Additional include directories",
                },
                "verbose": {
                    "type": "boolean",
                    "description": "Enable verbose compilation output",
                    "default": False,
                },
            },
            "required": ["cuda_code"],
        }


class CudaMemcheckTool(MCPTool):
    """Tool for CUDA memory checking using cuda-memcheck."""

    def __init__(self, timeout: int = 120):
        """Initialize the CudaMemcheckTool with a timeout."""
        super().__init__(
            name="cuda_memcheck",
            description="Run cuda-memcheck on CUDA executable",
            timeout=timeout,
        )

    def execute(
        self,
        cuda_code: str,
        host_code: Optional[str] = None,
        check_type: str = "memcheck",
        arch: str = "compute_75",
    ) -> ToolResult:
        """
        Run cuda-memcheck on CUDA code.

        Args:
            cuda_code: CUDA kernel code
            host_code: Host code to run the kernel
            check_type: Type of check (memcheck, racecheck, synccheck, initcheck)
            arch: Target architecture

        Returns:
            ToolResult: Memory check result
        """
        # Create complete CUDA program
        if host_code is None:
            host_code = self._generate_default_host_code()

        complete_program = f"{cuda_code}\n\n{host_code}"

        # Create temporary files
        cu_file = self._create_temp_file(complete_program, ".cu")
        exe_file = cu_file.replace(".cu", ".exe")

        try:
            # First compile the program
            compile_cmd = ["nvcc", cu_file, "-o", exe_file, f"-arch={arch}"]
            compile_result = self._run_command(compile_cmd)

            if compile_result.status != ToolExecutionStatus.SUCCESS:
                return ToolResult(
                    status=ToolExecutionStatus.ERROR,
                    output="",
                    error=f"Compilation failed: {compile_result.error}",
                    metadata={"compilation_output": compile_result.output},
                )

            # Run cuda-memcheck
            memcheck_cmd = ["cuda-memcheck", f"--tool={check_type}", exe_file]
            result = self._run_command(memcheck_cmd)

            # Parse memcheck output
            metadata = {
                "check_type": check_type,
                "executable": exe_file,
                "architecture": arch,
            }

            if result.output:
                metadata["memcheck_analysis"] = self._parse_memcheck_output(
                    result.output
                )

            result.metadata = metadata
            return result

        finally:
            # Cleanup temporary files
            self._cleanup_temp_file(cu_file)
            if os.path.exists(exe_file):
                self._cleanup_temp_file(exe_file)

    def _generate_default_host_code(self) -> str:
        """Generate default host code for testing kernels."""
        return """
int main() {
    // Allocate device memory for testing
    float *d_data;
    cudaMalloc(&d_data, 1024 * sizeof(float));

    // Launch kernel with simple configuration
    dim3 block(256);
    dim3 grid((1024 + block.x - 1) / block.x);

    // Note: This assumes a simple kernel signature
    // Real implementation would need kernel-specific parameters

    // Synchronize and check for errors
    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s\\n", cudaGetErrorString(error));
        return 1;
    }

    // Cleanup
    cudaFree(d_data);
    return 0;
}
"""

    def _parse_memcheck_output(self, output: str) -> Dict[str, Any]:
        """Parse cuda-memcheck output for structured information."""
        issues = []
        summary = {
            "memory_leaks": 0,
            "invalid_accesses": 0,
            "race_conditions": 0,
            "uninitialized_reads": 0,
        }

        lines = output.split("\n")
        for line in lines:
            if "leak" in line.lower():
                summary["memory_leaks"] += 1
                issues.append({"type": "memory_leak", "description": line.strip()})
            elif "invalid" in line.lower() and "access" in line.lower():
                summary["invalid_accesses"] += 1
                issues.append({"type": "invalid_access", "description": line.strip()})
            elif "race" in line.lower():
                summary["race_conditions"] += 1
                issues.append({"type": "race_condition", "description": line.strip()})
            elif "uninitialized" in line.lower():
                summary["uninitialized_reads"] += 1
                issues.append(
                    {"type": "uninitialized_read", "description": line.strip()}
                )

        return {"summary": summary, "issues": issues, "total_issues": len(issues)}

    def get_schema(self) -> Dict[str, Any]:
        """Return the JSON schema for the tool's input parameters."""
        return {
            "type": "object",
            "properties": {
                "cuda_code": {
                    "type": "string",
                    "description": "CUDA kernel code to check",
                },
                "host_code": {
                    "type": "string",
                    "description": "Host code to run the kernel (optional)",
                },
                "check_type": {
                    "type": "string",
                    "enum": ["memcheck", "racecheck", "synccheck", "initcheck"],
                    "description": "Type of memory check to perform",
                    "default": "memcheck",
                },
                "arch": {
                    "type": "string",
                    "description": "Target GPU architecture",
                    "default": "compute_75",
                },
            },
            "required": ["cuda_code"],
        }


class CudaSyntaxChecker(MCPTool):
    """Tool for CUDA syntax and semantic checking."""

    def __init__(self, timeout: int = 30):
        """Initialize the CudaSyntaxChecker with a timeout."""
        super().__init__(
            name="cuda_syntax_check",
            description="Check CUDA code for basic syntax errors",
            timeout=timeout,
        )

    def execute(self, cuda_code: str, strict: bool = False) -> ToolResult:
        """
        Check CUDA code syntax and semantic issues.

        Args:
            cuda_code: CUDA source code to check
            strict: Enable strict checking mode

        Returns:
            ToolResult: Syntax check result
        """
        issues = []
        warnings = []

        # Basic syntax checks
        syntax_issues = self._check_basic_syntax(cuda_code)
        issues.extend(syntax_issues)

        # CUDA-specific checks
        cuda_issues = self._check_cuda_semantics(cuda_code)
        issues.extend(cuda_issues)

        # Performance warnings
        if strict:
            perf_warnings = self._check_performance_issues(cuda_code)
            warnings.extend(perf_warnings)

        # Determine overall status
        status = ToolExecutionStatus.SUCCESS
        if any(issue["severity"] == "error" for issue in issues):
            status = ToolExecutionStatus.ERROR

        output = self._format_check_results(issues, warnings)

        return ToolResult(
            status=status,
            output=output,
            metadata={
                "issue_count": len(issues),
                "warning_count": len(warnings),
                "issues": issues,
                "warnings": warnings,
                "strict_mode": strict,
            },
        )

    def _check_basic_syntax(self, code: str) -> List[Dict[str, Any]]:
        """Check basic C/C++ syntax issues."""
        issues = []
        lines = code.split("\n")

        for i, line in enumerate(lines, 1):
            # Check for common syntax issues
            if line.strip().endswith(";") and "{" in line and "}" not in line:
                issues.append(
                    {
                        "line": i,
                        "severity": "warning",
                        "type": "syntax",
                        "message": "Semicolon after opening brace",
                    }
                )

            # Check for missing semicolons (simple heuristic)
            stripped = line.strip()
            if (
                stripped
                and not stripped.endswith((";", "{", "}", ":", "\\"))
                and not stripped.startswith(("#", "//", "/*"))
                and "<<" not in stripped
                and ">>" not in stripped
            ):
                if any(
                    keyword in stripped for keyword in ["return", "break", "continue"]
                ):
                    issues.append(
                        {
                            "line": i,
                            "severity": "error",
                            "type": "syntax",
                            "message": "Missing semicolon",
                        }
                    )

        return issues

    def _check_cuda_semantics(self, code: str) -> List[Dict[str, Any]]:
        """Check CUDA-specific semantic issues."""
        issues = []
        lines = code.split("\n")

        for i, line in enumerate(lines, 1):
            stripped = line.strip()

            # Check for __global__ function issues
            if "__global__" in stripped:
                if "void" not in stripped:
                    issues.append(
                        {
                            "line": i,
                            "severity": "error",
                            "type": "cuda_semantic",
                            "message": "__global__ functions must return void",
                        }
                    )

            # Check for memory allocation without error checking
            if "cudaMalloc" in stripped and "cudaError" not in code:
                issues.append(
                    {
                        "line": i,
                        "severity": "warning",
                        "type": "cuda_semantic",
                        "message": "cudaMalloc without error checking",
                    }
                )

            # Check for kernel launch without synchronization
            if "<<<" in stripped and ">>>" in stripped:
                # Look for cudaDeviceSynchronize in the following lines
                sync_found = False
                for j in range(i, min(i + 10, len(lines))):
                    if (
                        "cudaDeviceSynchronize" in lines[j]
                        or "cudaStreamSynchronize" in lines[j]
                    ):
                        sync_found = True
                        break

                if not sync_found:
                    issues.append(
                        {
                            "line": i,
                            "severity": "warning",
                            "type": "cuda_semantic",
                            "message": "Kernel launch without synchronization",
                        }
                    )

        return issues

    def _check_performance_issues(self, code: str) -> List[Dict[str, Any]]:
        """Check for potential performance issues."""
        warnings = []
        lines = code.split("\n")

        for i, line in enumerate(lines, 1):
            stripped = line.strip()

            # Check for potential memory coalescing issues
            if any(
                pattern in stripped for pattern in ["[threadIdx.x *", "* threadIdx.x]"]
            ):
                warnings.append(
                    {
                        "line": i,
                        "type": "performance",
                        "message": "Potential memory coalescing issue - consider stride patterns",
                    }
                )

            # Check for excessive register usage patterns
            if stripped.count("float") > 10:
                warnings.append(
                    {
                        "line": i,
                        "type": "performance",
                        "message": "Many variables declared - consider register pressure",
                    }
                )

            # Check for branch divergence patterns
            if "if" in stripped and "threadIdx" in stripped:
                warnings.append(
                    {
                        "line": i,
                        "type": "performance",
                        "message": "Thread-dependent branching may cause divergence",
                    }
                )

        return warnings

    def _format_check_results(
        self, issues: List[Dict[str, Any]], warnings: List[Dict[str, Any]]
    ) -> str:
        """Format check results for output."""
        output = []

        if issues:
            output.append("=== ISSUES FOUND ===")
            for issue in issues:
                severity = issue["severity"].upper()
                output.append(f"Line {issue['line']}: [{severity}] {issue['message']}")

        if warnings:
            output.append("\n=== PERFORMANCE WARNINGS ===")
            for warning in warnings:
                output.append(f"Line {warning['line']}: [WARNING] {warning['message']}")

        if not issues and not warnings:
            output.append("No issues found.")

        return "\n".join(output)

    def get_schema(self) -> Dict[str, Any]:
        """Return the JSON schema for the tool's input parameters."""
        return {
            "type": "object",
            "properties": {
                "cuda_code": {
                    "type": "string",
                    "description": "CUDA source code to check",
                },
                "strict": {
                    "type": "boolean",
                    "description": "Enable strict checking including performance warnings",
                    "default": False,
                },
            },
            "required": ["cuda_code"],
        }


class CudaDebugTools:
    """Collection of CUDA debugging tools for MCP integration."""

    def __init__(self):
        """Initialize the CudaDebugTools collection."""
        pass

    @staticmethod
    def get_all_tools() -> List[MCPTool]:
        """Get all available CUDA debugging tools."""
        return [CudaCompilerTool(), CudaMemcheckTool(), CudaSyntaxChecker()]

    @staticmethod
    def register_tools(tool_manager) -> None:
        """Register all CUDA debugging tools with a tool manager."""
        tools = CudaDebugTools.get_all_tools()
        for tool in tools:
            tool_manager.register_tool(tool)
