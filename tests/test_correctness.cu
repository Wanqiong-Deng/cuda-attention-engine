#include "attention.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>

// Generate random float in [0, 1)
float rand_float() {
    return (float)rand() / RAND_MAX;
}

// Compare GPU output vs CPU reference
bool check_correctness(
    const float* gpu_output,
    const float* cpu_output,
    int size,
    float tolerance
) {
    float max_error = 0.0f;
    float mean_error = 0.0f;
    int worst_idx = 0;

    for (int i = 0; i < size; i++) {
        float err = fabsf(gpu_output[i] - cpu_output[i]);
        mean_error += err;
        if (err > max_error) {
            max_error = err;
            worst_idx = i;
        }
    }
    mean_error /= size;

    printf("  Max error:  %e (at index %d)\n", max_error, worst_idx);
    printf("  Mean error: %e\n", mean_error);

    if (max_error > tolerance) {
        printf("  FAILED: max error exceeds tolerance %e\n", tolerance);
        return false;
    }
    printf("  PASSED\n");
    return true;
}

int main() {
    // Test configuration
    int seq_len = 256;
    int head_dim = 64;
    int qkv_size = seq_len * head_dim;

    printf("=== Attention Correctness Test ===\n");
    printf("seq_len=%d, head_dim=%d\n\n", seq_len, head_dim);

    // Allocate host memory
    float* h_Q = (float*)malloc(qkv_size * sizeof(float));
    float* h_K = (float*)malloc(qkv_size * sizeof(float));
    float* h_V = (float*)malloc(qkv_size * sizeof(float));
    float* h_O_cpu = (float*)malloc(qkv_size * sizeof(float));
    float* h_O_gpu = (float*)malloc(qkv_size * sizeof(float));

    // Fill with random data
    srand(42);
    for (int i = 0; i < qkv_size; i++) {
        h_Q[i] = rand_float() - 0.5f;
        h_K[i] = rand_float() - 0.5f;
        h_V[i] = rand_float() - 0.5f;
    }

    // CPU reference
    printf("[CPU] Computing reference attention...\n");
    attention_forward_cpu(h_Q, h_K, h_V, h_O_cpu, seq_len, head_dim);
    printf("[CPU] Done.\n\n");

    // GPU naive FP32
    printf("[GPU] Computing naive FP32 attention...\n");
    float *d_Q, *d_K, *d_V, *d_O;
    cudaMalloc(&d_Q, qkv_size * sizeof(float));
    cudaMalloc(&d_K, qkv_size * sizeof(float));
    cudaMalloc(&d_V, qkv_size * sizeof(float));
    cudaMalloc(&d_O, qkv_size * sizeof(float));

    cudaMemcpy(d_Q, h_Q, qkv_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K, qkv_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V, qkv_size * sizeof(float), cudaMemcpyHostToDevice);

    attention_forward_naive_fp32(d_Q, d_K, d_V, d_O, seq_len, head_dim);
    cudaDeviceSynchronize();

    cudaMemcpy(h_O_gpu, d_O, qkv_size * sizeof(float), cudaMemcpyDeviceToHost);
    printf("[GPU] Done.\n\n");

    // Correctness check
    printf("[CHECK] Naive FP32 vs CPU reference:\n");
    bool passed_naive = check_correctness(h_O_gpu, h_O_cpu, qkv_size, 1e-4f);

    // GPU tiled FP32
    printf("\n[GPU] Computing tiled FP32 attention...\n");
    attention_forward_tiled_fp32(d_Q, d_K, d_V, d_O, seq_len, head_dim);
    cudaDeviceSynchronize();

    cudaMemcpy(h_O_gpu, d_O, qkv_size * sizeof(float), cudaMemcpyDeviceToHost);
    printf("[GPU] Done.\n\n");

    printf("[CHECK] Tiled FP32 vs CPU reference:\n");
    bool passed_tiled = check_correctness(h_O_gpu, h_O_cpu, qkv_size, 1e-4f);

    // GPU FP16
    printf("\n[GPU] Computing FP16 attention...\n");
    attention_forward_fp16(d_Q, d_K, d_V, d_O, seq_len, head_dim);
    cudaDeviceSynchronize();

    cudaMemcpy(h_O_gpu, d_O, qkv_size * sizeof(float), cudaMemcpyDeviceToHost);
    printf("[GPU] Done.\n\n");

    printf("[CHECK] FP16 vs CPU reference (tolerance 1e-2 due to half precision):\n");
    bool passed_fp16 = check_correctness(h_O_gpu, h_O_cpu, qkv_size, 1e-2f);

    // GPU Tensor Core
    printf("\n[GPU] Computing Tensor Core attention...\n");
    attention_forward_tensor_core(d_Q, d_K, d_V, d_O, seq_len, head_dim);
    cudaDeviceSynchronize();

    cudaMemcpy(h_O_gpu, d_O, qkv_size * sizeof(float), cudaMemcpyDeviceToHost);
    printf("[GPU] Done.\n\n");

    printf("[CHECK] Tensor Core vs CPU reference (tolerance 1e-2):\n");
    bool passed_tc = check_correctness(h_O_gpu, h_O_cpu, qkv_size, 1e-2f);

    // Cleanup
    free(h_Q); free(h_K); free(h_V);
    free(h_O_cpu); free(h_O_gpu);
    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_O);

    bool all_passed = passed_naive && passed_tiled && passed_fp16 && passed_tc;
    printf("\n=== %s ===\n", all_passed ? "ALL TESTS PASSED" : "SOME TESTS FAILED");
    return all_passed ? 0 : 1;
}
