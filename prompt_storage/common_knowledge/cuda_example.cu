#include <stdio.h>
#include <cuda_runtime.h>
#include <sys/time.h>

#define CHECK_CUDA(call) do { \\
    cudaError_t err = call; \\
    if (err != cudaSuccess) { \\
        printf("CUDA error at %s:%d - %s\\n", __FILE__, __LINE__, cudaGetErrorString(err)); \\
        exit(1); \\
    } \\
} while(0)

__global__ void vector_add_kernel(float* A, float* B, float* C, int N) {
    // Grid-stride loop
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = idx; i < N; i += stride) {
        C[i] = A[i] + B[i];
    }
}

double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

int main() {
    int N = 32 * 1024; // 32K elements
    size_t size = N * sizeof(float);

    printf("Vector size: %d elements (%.2f MB)\\n", N, size / 1024.0 / 1024.0);

    // Host memory
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);

    // Initialize vectors
    for (int i = 0; i < N; i++) {
        h_A[i] = (float)i;
        h_B[i] = (float)(i * 2);
    }

    // Device memory
    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, size));
    CHECK_CUDA(cudaMalloc(&d_B, size));
    CHECK_CUDA(cudaMalloc(&d_C, size));

    // Copy to device
    CHECK_CUDA(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    // Launch kernel
    int dimBlock = 256;
    int dimGrid = (N + dimBlock - 1) / dimBlock;

    double start_kernel = get_time();
    vector_add_kernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, N);
    CHECK_CUDA(cudaDeviceSynchronize());
    double end_kernel = get_time();

    // Copy result back
    CHECK_CUDA(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

    // Performance metrics
    double kernel_time = end_kernel - start_kernel;

    printf("\\nPerformance Results:\\n");
    printf("Kernel time: %.3f ms\\n", kernel_time * 1000);

    // Verify result
    printf("\\nVerifying results...\\n");
    bool correct = true;
    const int max_errors_to_show = 5;

    for (int i = 0; i < N; i++) {
        float expected = h_A[i] + h_B[i];
        if (abs(h_C[i] - expected) > 1e-5) {
            correct = false;
            break;
        }
    }

    if (!correct) {
        printf("❌ Verification failed!\n");
    }

    // Cleanup
    free(h_A); free(h_B); free(h_C);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);

    return 0;
}
