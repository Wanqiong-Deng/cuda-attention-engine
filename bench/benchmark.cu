#include "attention.h"
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

float rand_float() {
    return (float)rand() / RAND_MAX;
}

struct BenchResult {
    const char* name;
    float avg_ms;
    float tflops;
};

BenchResult run_benchmark(
    void (*kernel_fn)(const float*, const float*, const float*, float*, int, int),
    const char* name,
    float* d_Q, float* d_K, float* d_V, float* d_O,
    int seq_len, int head_dim, long long flops,
    int warmup_runs, int bench_runs,
    cudaEvent_t start, cudaEvent_t stop
) {
    printf("[%s]\n", name);

    for (int i = 0; i < warmup_runs; i++) {
        kernel_fn(d_Q, d_K, d_V, d_O, seq_len, head_dim);
    }
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for (int i = 0; i < bench_runs; i++) {
        kernel_fn(d_Q, d_K, d_V, d_O, seq_len, head_dim);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    float avg_ms = ms / bench_runs;
    float tflops = (float)flops / (avg_ms * 1e9f);

    printf("  Avg time:  %.3f ms\n", avg_ms);
    printf("  TFLOPS:    %.4f\n\n", tflops);

    return {name, avg_ms, tflops};
}

int main() {
    int seq_len = 512;
    int head_dim = 64;
    int qkv_size = seq_len * head_dim;
    int warmup_runs = 5;
    int bench_runs = 20;

    printf("╔══════════════════════════════════════╗\n");
    printf("║     CUDA Attention Engine Benchmark  ║\n");
    printf("╠══════════════════════════════════════╣\n");
    printf("║  seq_len=%d, head_dim=%d            ║\n", seq_len, head_dim);
    printf("║  warmup=%d, bench_runs=%d           ║\n", warmup_runs, bench_runs);
    printf("╚══════════════════════════════════════╝\n\n");

    // Allocate host memory and fill with random data
    float* h_Q = (float*)malloc(qkv_size * sizeof(float));
    float* h_K = (float*)malloc(qkv_size * sizeof(float));
    float* h_V = (float*)malloc(qkv_size * sizeof(float));

    srand(42);
    for (int i = 0; i < qkv_size; i++) {
        h_Q[i] = rand_float() - 0.5f;
        h_K[i] = rand_float() - 0.5f;
        h_V[i] = rand_float() - 0.5f;
    }

    // Allocate device memory
    float *d_Q, *d_K, *d_V, *d_O;
    cudaMalloc(&d_Q, qkv_size * sizeof(float));
    cudaMalloc(&d_K, qkv_size * sizeof(float));
    cudaMalloc(&d_V, qkv_size * sizeof(float));
    cudaMalloc(&d_O, qkv_size * sizeof(float));

    cudaMemcpy(d_Q, h_Q, qkv_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K, qkv_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V, qkv_size * sizeof(float), cudaMemcpyHostToDevice);

    // CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Compute total FLOPS
    long long flops = 2LL * seq_len * seq_len * head_dim  // QK^T
                    + 2LL * seq_len * seq_len * head_dim  // scores*V
                    + 5LL * seq_len * seq_len;            // softmax

    // Run all benchmarks
    BenchResult results[4];
    results[0] = run_benchmark(attention_forward_naive_fp32, "Naive FP32",
                               d_Q, d_K, d_V, d_O, seq_len, head_dim, flops,
                               warmup_runs, bench_runs, start, stop);
    results[1] = run_benchmark(attention_forward_tiled_fp32, "Tiled FP32 (Shared Mem)",
                               d_Q, d_K, d_V, d_O, seq_len, head_dim, flops,
                               warmup_runs, bench_runs, start, stop);
    results[2] = run_benchmark(attention_forward_fp16, "FP16 Mixed Precision",
                               d_Q, d_K, d_V, d_O, seq_len, head_dim, flops,
                               warmup_runs, bench_runs, start, stop);
    results[3] = run_benchmark(attention_forward_tensor_core, "Tensor Core (WMMA)",
                               d_Q, d_K, d_V, d_O, seq_len, head_dim, flops,
                               warmup_runs, bench_runs, start, stop);

    // Print summary table
    float baseline_ms = results[0].avg_ms;

    printf("╔═══════════════════════════╦══════════╦══════════╦══════════╗\n");
    printf("║ Kernel                    ║ Time(ms) ║  TFLOPS  ║ Speedup  ║\n");
    printf("╠═══════════════════════════╬══════════╬══════════╬══════════╣\n");
    for (int i = 0; i < 4; i++) {
        printf("║ %-25s ║ %7.3f  ║ %7.4f  ║ %6.2fx  ║\n",
               results[i].name, results[i].avg_ms, results[i].tflops,
               baseline_ms / results[i].avg_ms);
    }
    printf("╚═══════════════════════════╩══════════╩══════════╩══════════╝\n");

    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    free(h_Q); free(h_K); free(h_V);
    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_O);

    return 0;
}
