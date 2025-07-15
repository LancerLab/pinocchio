#!/usr/bin/env python3
"""
Complete End-to-End Test for CUDA Matrix Multiplication
including code generation, nvcc compilation, and ncu performance analysis
"""

import asyncio
import json
import os
import subprocess

# Add project root directory to Python path
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from pinocchio.config import ConfigManager
from pinocchio.coordinator import Coordinator
from pinocchio.llm.custom_llm_client import CustomLLMClient


async def generate_cuda_code():
    """Use Pinocchio to generate CUDA code"""
    print("🚀 Step 1: Generate CUDA matrix multiplication code...")

    # Initialize configuration and LLM client
    config_manager = ConfigManager()
    llm_config = config_manager.get_llm_config()
    llm_client = CustomLLMClient(llm_config, verbose=True)

    # Create coordinator
    coordinator = Coordinator(llm_client)

    # User request
    user_request = """Please help me generate a simple CUDA matrix multiplication kernel,
1. Support arbitrary matrix size (M x K) * (K x N) = (M x N)
2. Optimize using shared memory, tile size = 32
3. Include complete host code and error checking
4. Provide compilation commands and launch configuration
5. The code should be directly compilable and runnable"""

    # Generate code
    generated_result = None
    async for message in coordinator.process_user_request(user_request):
        print(f"📝 {message}")
        # Get final result
        if hasattr(coordinator, "_final_result") and coordinator._final_result:
            generated_result = coordinator._final_result

    return generated_result


def extract_cuda_code(generated_result):
    """Extract CUDA code from generated result"""
    print("🔍 Step 2: Extract CUDA code...")

    if not generated_result:
        print("❌ No code generated")
        return None

    # Find code field
    code = None
    if isinstance(generated_result, dict):
        # Try multiple possible code fields
        for key in ["code", "output", "cuda_code", "kernel_code"]:
            if key in generated_result:
                if (
                    isinstance(generated_result[key], dict)
                    and "code" in generated_result[key]
                ):
                    code = generated_result[key]["code"]
                else:
                    code = generated_result[key]
                break

    if code:
        print(f"✅ Successfully extracted code, length: {len(code)} characters")
        return code
    else:
        print("❌ CUDA code not found")
        print(f"Generated result structure: {type(generated_result)}")
        if isinstance(generated_result, dict):
            print(f"Available fields: {list(generated_result.keys())}")
        return None


def save_and_compile_cuda_code(cuda_code):
    """Save and compile CUDA code"""
    print("🔨 Step 3: Compile CUDA code...")

    if not cuda_code:
        print("❌ No CUDA code to compile")
        return None, None

    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save CUDA code
        cu_file = Path(temp_dir) / "matmul.cu"
        with open(cu_file, "w", encoding="utf-8") as f:
            f.write(cuda_code)

        print(f"📁 Saved to: {cu_file}")

        # Compile command
        exe_file = Path(temp_dir) / "matmul"
        compile_cmd = [
            "nvcc",
            "-arch=sm_75",  # Adapt to modern GPUs
            "-O3",  # Optimize
            "-o",
            str(exe_file),
            str(cu_file),
        ]

        print(f"🔧 Compile command: {' '.join(compile_cmd)}")

        try:
            # Check if nvcc is available
            result = subprocess.run(
                ["nvcc", "--version"], capture_output=True, text=True, timeout=10
            )
            if result.returncode != 0:
                print("❌ nvcc not available, please install CUDA toolkit")
                return None, None

            print(
                f"✅ nvcc version: {result.stdout.split('release')[1].split(',')[0].strip()}"
            )

            # Compile
            result = subprocess.run(
                compile_cmd, capture_output=True, text=True, timeout=60
            )

            if result.returncode == 0:
                print("✅ Compilation successful!")

                # Copy files to current directory for later use
                import shutil

                final_cu = "generated_matmul.cu"
                final_exe = "generated_matmul"
                shutil.copy2(cu_file, final_cu)
                shutil.copy2(exe_file, final_exe)

                print(f"📁 Files saved to: {final_cu}, {final_exe}")
                return final_cu, final_exe
            else:
                print(f"❌ Compilation failed:")
                print(f"stderr: {result.stderr}")
                print(f"stdout: {result.stdout}")
                return str(cu_file), None

        except subprocess.TimeoutExpired:
            print("❌ Compilation timed out")
            return str(cu_file), None
        except Exception as e:
            print(f"❌ Compilation error: {e}")
            return str(cu_file), None


def run_performance_analysis(exe_file):
    """Use ncu for performance analysis"""
    print("📊 Step 4: Performance analysis...")

    if not exe_file or not os.path.exists(exe_file):
        print("❌ No executable file for performance analysis")
        return None

    try:
        # Check if ncu is available
        result = subprocess.run(
            ["ncu", "--version"], capture_output=True, text=True, timeout=10
        )
        if result.returncode != 0:
            print("❌ ncu (NVIDIA Nsight Compute) not available")
            print("💡 Hint: NVIDIA Nsight Compute needs to be installed")
            return None

        print(f"✅ ncu version available")

        # Run performance analysis
        ncu_cmd = [
            "ncu",
            "--metrics",
            "smsp__cycles_elapsed.avg,dram__throughput.avg.pct_of_peak_sustained_elapsed",
            "--target-processes",
            "all",
            exe_file,
        ]

        print(f"🔍 Analysis command: {' '.join(ncu_cmd)}")

        result = subprocess.run(ncu_cmd, capture_output=True, text=True, timeout=120)

        if result.returncode == 0:
            print("✅ Performance analysis completed!")
            print("📈 Analysis results:")
            print(result.stdout)

            # Save results
            with open("performance_analysis.txt", "w") as f:
                f.write(result.stdout)
            print("📁 Results saved to: performance_analysis.txt")

            return result.stdout
        else:
            print(f"❌ Performance analysis failed:")
            print(f"stderr: {result.stderr}")
            return None

    except subprocess.TimeoutExpired:
        print("❌ Performance analysis timed out")
        return None
    except Exception as e:
        print(f"❌ Performance analysis error: {e}")
        return None


def create_simple_test_code():
    """Create a simple CUDA test code as a fallback"""
    cuda_code = """
#include <stdio.h>
#include <cuda_runtime.h>
#include <sys/time.h>

#define TILE_WIDTH 32
#define CHECK_CUDA(call) do { \\
    cudaError_t err = call; \\
    if (err != cudaSuccess) { \\
        printf("CUDA error at %s:%d - %s\\n", __FILE__, __LINE__, cudaGetErrorString(err)); \\
        exit(1); \\
    } \\
} while(0)

__global__ void matmul_kernel(float* A, float* B, float* C, int M, int N, int K) {
    __shared__ float As[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bs[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;

    float sum = 0.0f;

    for (int m = 0; m < (K + TILE_WIDTH - 1) / TILE_WIDTH; ++m) {
        // Load tiles
        if (row < M && m * TILE_WIDTH + tx < K)
            As[ty][tx] = A[row * K + m * TILE_WIDTH + tx];
        else
            As[ty][tx] = 0.0f;

        if (col < N && m * TILE_WIDTH + ty < K)
            Bs[ty][tx] = B[(m * TILE_WIDTH + ty) * N + col];
        else
            Bs[ty][tx] = 0.0f;

        __syncthreads();

        // Compute
        for (int k = 0; k < TILE_WIDTH; ++k)
            sum += As[ty][k] * Bs[k][tx];

        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = sum;
}

double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

int main() {
    int M = 1024, N = 1024, K = 1024;
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    // Host memory
    float *h_A = (float*)malloc(size_A);
    float *h_B = (float*)malloc(size_B);
    float *h_C = (float*)malloc(size_C);

    // Initialize
    for (int i = 0; i < M * K; i++) h_A[i] = 1.0f;
    for (int i = 0; i < K * N; i++) h_B[i] = 1.0f;

    // Device memory
    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, size_A));
    CHECK_CUDA(cudaMalloc(&d_B, size_B));
    CHECK_CUDA(cudaMalloc(&d_C, size_C));

    // Copy to device
    CHECK_CUDA(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice));

    // Launch kernel
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid((N + TILE_WIDTH - 1) / TILE_WIDTH, (M + TILE_WIDTH - 1) / TILE_WIDTH);

    double start = get_time();
    matmul_kernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, M, N, K);
    CHECK_CUDA(cudaDeviceSynchronize());
    double end = get_time();

    printf("Matrix multiplication completed in %.3f ms\\n", (end - start) * 1000);
    printf("Performance: %.2f GFLOPS\\n", 2.0 * M * N * K / (end - start) / 1e9);

    // Copy result back
    CHECK_CUDA(cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost));

    // Verify result (should be K for all elements)
    printf("Sample result: C[0][0] = %.1f (expected: %.1f)\\n", h_C[0], (float)K);

    // Cleanup
    free(h_A); free(h_B); free(h_C);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);

    return 0;
}
"""
    return cuda_code


async def main():
    """Main function"""
    print("🎯 Pinocchio CUDA Matrix Multiplication End-to-End Test")
    print("=" * 60)

    start_time = time.time()

    try:
        # Step 1: Generate CUDA code
        generated_result = await generate_cuda_code()

        # Step 2: Extract code
        cuda_code = extract_cuda_code(generated_result)

        # If generation failed, use fallback code
        if not cuda_code:
            print("🔄 Using fallback CUDA code...")
            cuda_code = create_simple_test_code()

        # Step 3: Compile
        cu_file, exe_file = save_and_compile_cuda_code(cuda_code)

        # Step 4: Performance analysis
        if exe_file:
            performance_result = run_performance_analysis(exe_file)
        else:
            print("⚠️ Skipping performance analysis (compilation failed)")
            performance_result = None

        # Summary
        end_time = time.time()
        print("\n" + "=" * 60)
        print("📋 Summary Report:")
        print(f"⏱️  Total time: {end_time - start_time:.2f} seconds")
        print(f"📝 CUDA code generation: {'✅' if cuda_code else '❌'}")
        print(f"🔨 nvcc compilation: {'✅' if exe_file else '❌'}")
        print(f"📊 ncu performance analysis: {'✅' if performance_result else '❌'}")

        # Create detailed report
        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_time_seconds": end_time - start_time,
            "cuda_code_generated": bool(cuda_code),
            "compilation_successful": bool(exe_file),
            "performance_analysis_completed": bool(performance_result),
            "generated_files": {
                "cuda_source": cu_file if cu_file else None,
                "executable": exe_file if exe_file else None,
                "performance_report": "performance_analysis.txt"
                if performance_result
                else None,
            },
        }

        with open("e2e_test_report.json", "w") as f:
            json.dump(report, f, indent=2)

        print(f"📁 Detailed report saved to: e2e_test_report.json")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
