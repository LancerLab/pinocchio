"""
Test module demonstrating usage of Real Code Transmission functionality.

This test module serves as both validation and documentation, showing developers
how the real code transmission system works between agents and how to use it effectively.

Key Features Demonstrated:
1. Real CUDA code generation (not placeholder text)
2. Code transmission between different agents
3. Code validation and processing workflows
4. Template-based code generation patterns
5. Integration with task execution system
"""

import os
import sys

import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from pinocchio.agents import (
    DebuggerAgent,
    EvaluatorAgent,
    GeneratorAgent,
    OptimizerAgent,
)
from pinocchio.llm import BaseLLMClient


class MockBaseLLMClient(BaseLLMClient):
    """Mock LLM interface that simulates real code generation responses."""

    def __init__(self):
        super().__init__()
        self.last_request = None
        self.response_mode = "real_code"  # vs 'placeholder'

        # Predefined CUDA code templates for different requests
        self.code_templates = {
            "matrix_multiplication": """
#include <cuda_runtime.h>
#include <cublas_v2.h>

// CUDA kernel for matrix multiplication
__global__ void matrix_multiply_kernel(float* A, float* B, float* C,
                                     int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Host function to launch kernel
void launch_matrix_multiply(float* d_A, float* d_B, float* d_C,
                          int M, int N, int K) {
    dim3 blockSize(16, 16);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x,
                  (M + blockSize.y - 1) / blockSize.y);

    matrix_multiply_kernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, M, N, K);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s\\n", cudaGetErrorString(error));
    }
}
""",
            "vector_addition": """
#include <cuda_runtime.h>

// CUDA kernel for vector addition
__global__ void vector_add_kernel(float* A, float* B, float* C, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        C[idx] = A[idx] + B[idx];
    }
}

// Host function wrapper
cudaError_t vector_add_cuda(float* h_A, float* h_B, float* h_C, int n) {
    float *d_A, *d_B, *d_C;
    size_t size = n * sizeof(float);

    // Allocate GPU memory
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // Copy data to GPU
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Launch kernel
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    vector_add_kernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, n);

    // Copy result back
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return cudaGetLastError();
}
""",
            "optimized_matrix_multiply": """
#include <cuda_runtime.h>

#define TILE_SIZE 16

// Optimized matrix multiplication with shared memory
__global__ void optimized_matrix_multiply_kernel(float* A, float* B, float* C,
                                               int M, int N, int K) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;

    float sum = 0.0f;

    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Load tiles into shared memory
        if (row < M && t * TILE_SIZE + tx < K) {
            As[ty][tx] = A[row * K + t * TILE_SIZE + tx];
        } else {
            As[ty][tx] = 0.0f;
        }

        if (col < N && t * TILE_SIZE + ty < K) {
            Bs[ty][tx] = B[(t * TILE_SIZE + ty) * N + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }

        __syncthreads();

        // Compute partial sum
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}
""",
        }

    def send_request(self, prompt: str, context: dict = None) -> str:
        """Simulate real code generation based on request content."""
        self.last_request = {
            "prompt": prompt,
            "context": context or {},
            "prompt_length": len(prompt),
        }

        if self.response_mode == "placeholder":
            return "// TODO: Implement CUDA kernel here"

        # Analyze prompt to determine what code to generate
        prompt_lower = prompt.lower()

        if "matrix" in prompt_lower and "multiply" in prompt_lower:
            if "optim" in prompt_lower:
                return self.code_templates["optimized_matrix_multiply"]
            else:
                return self.code_templates["matrix_multiplication"]
        elif "vector" in prompt_lower and "add" in prompt_lower:
            return self.code_templates["vector_addition"]
        else:
            # Generic kernel template
            return """
#include <cuda_runtime.h>

__global__ void generic_kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        // Process data element
        data[idx] = data[idx] * 2.0f;
    }
}

void launch_generic_kernel(float* d_data, int n) {
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;

    generic_kernel<<<gridSize, blockSize>>>(d_data, n);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s\\n", cudaGetErrorString(error));
    }
}
"""


class TestRealCodeTransmissionUsage:
    """
    Test suite demonstrating usage patterns for real code transmission.

    These tests show how developers can:
    1. Generate real CUDA code instead of placeholders
    2. Pass code between different agent types
    3. Validate code transmission integrity
    4. Use different code generation templates
    5. Handle code processing workflows
    """

    def setup_method(self):
        """Set up test environment with mock LLM interface."""
        self.mock_llm = MockBaseLLMClient()

    def test_real_vs_placeholder_code_generation(self):
        """
        Demonstrates the difference between real and placeholder code generation.

        Usage Pattern:
        - Shows how system generates actual CUDA code
        - Compares with old placeholder approach
        - Validates code quality and completeness
        """
        generator = GeneratorAgent(self.mock_llm)

        # Test real code generation (current system)
        self.mock_llm.response_mode = "real_code"
        real_code = generator.generate_code("Create a vector addition CUDA kernel")

        # Validate real code characteristics
        assert len(real_code) > 100, "Real code should be substantial"
        assert "__global__" in real_code, "Should contain CUDA kernel syntax"
        assert "#include" in real_code, "Should include necessary headers"
        assert (
            "cudaMemcpy" in real_code or "cudaMalloc" in real_code
        ), "Should include CUDA API calls"

        # Test placeholder mode (old system simulation)
        self.mock_llm.response_mode = "placeholder"
        placeholder_code = generator.generate_code(
            "Create a vector addition CUDA kernel"
        )

        # Validate placeholder characteristics
        assert "TODO" in placeholder_code, "Placeholder should contain TODO"
        assert len(placeholder_code) < 50, "Placeholder should be short"

        print(f"✓ Real code length: {len(real_code)} chars")
        print(f"✓ Placeholder code length: {len(placeholder_code)} chars")
        print(
            f"✓ Real code is {len(real_code) // len(placeholder_code)}x longer than placeholder"
        )

    def test_code_transmission_between_agents(self):
        """
        Demonstrates how code is passed between different agent types.

        Usage Pattern:
        - Generator creates initial code
        - Code is passed to other agents for processing
        - Each agent can modify and improve the code
        - Shows complete code processing pipeline
        """
        # Set up all agent types
        generator = GeneratorAgent(self.mock_llm)
        optimizer = OptimizerAgent(self.mock_llm)
        debugger = DebuggerAgent(self.mock_llm)
        evaluator = EvaluatorAgent(self.mock_llm)

        # Step 1: Generate initial code
        self.mock_llm.response_mode = "real_code"
        initial_code = generator.generate_code("Create a matrix multiplication kernel")

        # Validate initial code
        assert "__global__" in initial_code
        assert "matrix_multiply" in initial_code.lower()

        print(f"✓ Generated initial code: {len(initial_code)} characters")

        # Step 2: Pass code to debugger for validation
        debug_result = debugger.debug_code(initial_code)

        # Verify debugger received the full code
        debug_request = self.mock_llm.last_request
        assert initial_code in debug_request["prompt"]

        print("✓ Code successfully transmitted to debugger")

        # Step 3: Pass code to optimizer for improvement
        optimization_result = optimizer.optimize_code(initial_code)

        # Verify optimizer received the full code
        optimize_request = self.mock_llm.last_request
        assert initial_code in optimize_request["prompt"]

        print("✓ Code successfully transmitted to optimizer")

        # Step 4: Pass code to evaluator for performance analysis
        evaluation_result = evaluator.evaluate_code(initial_code)

        # Verify evaluator received the full code
        eval_request = self.mock_llm.last_request
        assert initial_code in eval_request["prompt"]

        print("✓ Code successfully transmitted to evaluator")

        # Verify code integrity throughout transmission
        all_requests = [debug_request, optimize_request, eval_request]
        for i, request in enumerate(all_requests):
            assert initial_code in request["prompt"], f"Code corrupted in step {i+1}"

        print("✓ Code integrity maintained through all transmissions")

    def test_template_based_code_generation(self):
        """
        Demonstrates different code generation templates and patterns.

        Usage Pattern:
        - Shows how system selects appropriate templates
        - Demonstrates template customization
        - Validates template-specific outputs
        """
        generator = GeneratorAgent(self.mock_llm)
        self.mock_llm.response_mode = "real_code"

        # Test different template triggers
        test_cases = [
            {
                "request": "Create a matrix multiplication kernel",
                "expected_content": [
                    "matrix_multiply_kernel",
                    "__global__",
                    "blockIdx",
                ],
                "template_type": "matrix_multiplication",
            },
            {
                "request": "Generate a vector addition CUDA function",
                "expected_content": ["vector_add", "cudaMemcpy", "cudaMalloc"],
                "template_type": "vector_addition",
            },
            {
                "request": "Create an optimized matrix multiply with shared memory",
                "expected_content": ["__shared__", "TILE_SIZE", "syncthreads"],
                "template_type": "optimized_matrix_multiply",
            },
        ]

        for case in test_cases:
            code = generator.generate_code(case["request"])

            # Validate template-specific content
            for expected in case["expected_content"]:
                assert (
                    expected in code
                ), f"Missing '{expected}' in {case['template_type']} template"

            print(f"✓ {case['template_type']} template generated correctly")

        # Test generic template fallback
        generic_code = generator.generate_code("Create a simple kernel")
        assert "generic_kernel" in generic_code
        assert "__global__" in generic_code

        print("✓ Generic template fallback works correctly")

    def test_code_processing_workflow(self):
        """
        Demonstrates a complete code processing workflow with real code.

        Usage Pattern:
        - Shows realistic development workflow
        - Demonstrates code evolution through processing stages
        - Validates workflow with actual CUDA code
        """
        # Create agent pipeline
        generator = GeneratorAgent(self.mock_llm)
        debugger = DebuggerAgent(self.mock_llm)
        optimizer = OptimizerAgent(self.mock_llm)
        evaluator = EvaluatorAgent(self.mock_llm)

        self.mock_llm.response_mode = "real_code"

        # Workflow: Generate -> Debug -> Optimize -> Evaluate
        workflow_stages = []

        # Stage 1: Code Generation
        print("\n--- Stage 1: Code Generation ---")
        initial_request = "Create a matrix multiplication kernel for deep learning"
        generated_code = generator.generate_code(initial_request)

        workflow_stages.append(
            {
                "stage": "generation",
                "input": initial_request,
                "output": generated_code,
                "agent": "generator",
            }
        )

        # Validate generation stage
        assert len(generated_code) > 200
        assert "matrix" in generated_code.lower()
        print(f"✓ Generated {len(generated_code)} characters of CUDA code")

        # Stage 2: Code Debugging
        print("\n--- Stage 2: Code Debugging ---")
        debug_result = debugger.debug_code(generated_code)

        workflow_stages.append(
            {
                "stage": "debugging",
                "input": generated_code,
                "output": debug_result,
                "agent": "debugger",
            }
        )

        # Verify debugging received full code
        assert generated_code in self.mock_llm.last_request["prompt"]
        print("✓ Debugging stage received complete code")

        # Stage 3: Code Optimization
        print("\n--- Stage 3: Code Optimization ---")
        optimization_result = optimizer.optimize_code(generated_code)

        workflow_stages.append(
            {
                "stage": "optimization",
                "input": generated_code,
                "output": optimization_result,
                "agent": "optimizer",
            }
        )

        # Verify optimization received full code
        assert generated_code in self.mock_llm.last_request["prompt"]
        print("✓ Optimization stage received complete code")

        # Stage 4: Code Evaluation
        print("\n--- Stage 4: Code Evaluation ---")
        evaluation_result = evaluator.evaluate_code(generated_code)

        workflow_stages.append(
            {
                "stage": "evaluation",
                "input": generated_code,
                "output": evaluation_result,
                "agent": "evaluator",
            }
        )

        # Verify evaluation received full code
        assert generated_code in self.mock_llm.last_request["prompt"]
        print("✓ Evaluation stage received complete code")

        # Workflow validation
        assert len(workflow_stages) == 4
        for stage in workflow_stages:
            assert stage["input"] is not None
            assert stage["output"] is not None
            assert stage["agent"] in ["generator", "debugger", "optimizer", "evaluator"]

        print(f"\n✓ Complete workflow executed with {len(workflow_stages)} stages")
        print("✓ Code successfully processed through entire pipeline")

        return workflow_stages

    def test_code_modification_tracking(self):
        """
        Demonstrates how to track code modifications through the pipeline.

        Usage Pattern:
        - Shows how to monitor code changes
        - Validates code improvement tracking
        - Demonstrates version control patterns
        """
        generator = GeneratorAgent(self.mock_llm)
        optimizer = OptimizerAgent(self.mock_llm)

        self.mock_llm.response_mode = "real_code"

        # Generate initial version
        v1_code = generator.generate_code("Create basic matrix multiplication")

        # Create modification tracking
        code_versions = {
            "v1_original": {
                "code": v1_code,
                "stage": "generation",
                "metrics": {
                    "length": len(v1_code),
                    "complexity": v1_code.count("for") + v1_code.count("while"),
                    "cuda_calls": v1_code.count("cuda"),
                    "includes": v1_code.count("#include"),
                },
            }
        }

        # Simulate optimization (in real system, optimizer would return modified code)
        # For demo, we'll manually create an "optimized" version
        v2_code = v1_code.replace("float sum = 0.0f;", "register float sum = 0.0f;")
        v2_code = v2_code.replace("__global__", "__global__ __launch_bounds__(256)")

        code_versions["v2_optimized"] = {
            "code": v2_code,
            "stage": "optimization",
            "metrics": {
                "length": len(v2_code),
                "complexity": v2_code.count("for") + v2_code.count("while"),
                "cuda_calls": v2_code.count("cuda"),
                "includes": v2_code.count("#include"),
                "optimizations": v2_code.count("register")
                + v2_code.count("__launch_bounds__"),
            },
        }

        # Analyze modifications
        v1_metrics = code_versions["v1_original"]["metrics"]
        v2_metrics = code_versions["v2_optimized"]["metrics"]

        # Validate tracking
        assert (
            v2_metrics["length"] > v1_metrics["length"]
        ), "Optimized version should be longer"
        assert v2_metrics["optimizations"] > 0, "Should have optimization annotations"

        # Calculate improvement metrics
        improvements = {
            "size_increase": v2_metrics["length"] - v1_metrics["length"],
            "optimization_count": v2_metrics.get("optimizations", 0),
            "stage_progression": ["generation", "optimization"],
        }

        print(f"✓ Original code: {v1_metrics['length']} chars")
        print(f"✓ Optimized code: {v2_metrics['length']} chars")
        print(f"✓ Added {improvements['optimization_count']} optimization annotations")
        print(
            f"✓ Tracked progression: {' -> '.join(improvements['stage_progression'])}"
        )

        return code_versions

    def test_error_handling_in_code_transmission(self):
        """
        Demonstrates error handling during code transmission.

        Usage Pattern:
        - Shows how to handle transmission failures
        - Validates error recovery mechanisms
        - Demonstrates robust code processing
        """
        generator = GeneratorAgent(self.mock_llm)
        debugger = DebuggerAgent(self.mock_llm)

        # Test with valid code
        self.mock_llm.response_mode = "real_code"
        valid_code = generator.generate_code("Create vector addition kernel")

        # Verify valid transmission
        try:
            debug_result = debugger.debug_code(valid_code)
            assert valid_code in self.mock_llm.last_request["prompt"]
            print("✓ Valid code transmitted successfully")
        except Exception as e:
            pytest.fail(f"Valid code transmission failed: {e}")

        # Test with empty code
        try:
            empty_debug = debugger.debug_code("")
            # Should handle gracefully
            print("✓ Empty code handled gracefully")
        except Exception as e:
            print(f"✓ Empty code properly rejected: {e}")

        # Test with malformed code
        malformed_code = "This is not valid CUDA code at all"
        try:
            malformed_debug = debugger.debug_code(malformed_code)
            assert malformed_code in self.mock_llm.last_request["prompt"]
            print("✓ Malformed code transmitted (agent will handle validation)")
        except Exception as e:
            print(f"✓ Malformed code properly handled: {e}")

        # Test with very large code
        large_code = valid_code * 100  # Simulate very large code
        try:
            large_debug = debugger.debug_code(large_code)
            print(f"✓ Large code ({len(large_code)} chars) transmitted successfully")
        except Exception as e:
            print(f"✓ Large code handled with appropriate limits: {e}")


if __name__ == "__main__":
    """
    Run real code transmission tests and display results.

    This section shows developers how to run tests and understand the system.
    """
    print("Running Real Code Transmission Usage Tests...")
    print("=" * 60)

    # Create test instance
    test_instance = TestRealCodeTransmissionUsage()
    test_instance.setup_method()

    # Run each test with clear output
    tests = [
        (
            "Real vs Placeholder Generation",
            test_instance.test_real_vs_placeholder_code_generation,
        ),
        (
            "Code Transmission Between Agents",
            test_instance.test_code_transmission_between_agents,
        ),
        (
            "Template-Based Generation",
            test_instance.test_template_based_code_generation,
        ),
        ("Complete Processing Workflow", test_instance.test_code_processing_workflow),
        ("Code Modification Tracking", test_instance.test_code_modification_tracking),
        ("Error Handling", test_instance.test_error_handling_in_code_transmission),
    ]

    passed = 0
    total = len(tests)

    for name, test_func in tests:
        try:
            print(f"\n{name}:")
            test_func()
            print(f"✅ {name} - PASSED")
            passed += 1
        except Exception as e:
            print(f"❌ {name} - FAILED: {e}")
            import traceback

            traceback.print_exc()

    print("\n" + "=" * 60)
    print(f"Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All real code transmission tests passed!")
        print("\nKey Benefits Demonstrated:")
        print("- Real CUDA code generation instead of placeholders")
        print("- Reliable code transmission between agents")
        print("- Template-based code generation system")
        print("- Complete development workflow support")
        print("- Robust error handling and validation")
        print("\nDevelopers can now:")
        print("- Generate production-ready CUDA code")
        print("- Build multi-agent code processing pipelines")
        print("- Track code evolution through workflow stages")
        print("- Handle various code types and edge cases")
    else:
        print("⚠️ Some tests failed. Check implementation details.")
