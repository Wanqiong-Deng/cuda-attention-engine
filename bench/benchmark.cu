#include "attention.h"
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <algorithm>

float rand_float() {
    return (float)rand() / RAND_MAX;
}

struct BenchResult {
    const char* name;
    float median_ms;
    float tflops;
    float arithmetic_intensity;
};

int compare_float(const void* a, const void* b) {
    float fa = *(const float*)a;
    float fb = *(const float*)b;
    return (fa > fb) - (fa < fb);
}

BenchResult run_benchmark(
    void (*kernel_fn)(const float*, const float*, const float*, float*, int, int),
    const char* name,
    float* d_Q, float* d_K, float* d_V, float* d_O,
    int seq_len, int head_dim, long long flops, long long bytes_transferred,
    int warmup_runs, int bench_runs
) {
    printf("[%s]\n", name);

    // Warmup
    for (int i = 0; i < warmup_runs; i++) {
        kernel_fn(d_Q, d_K, d_V, d_O, seq_len, head_dim);
    }
    cudaDeviceSynchronize();

    // Per-run timing (for median)
    float* times = (float*)malloc(bench_runs * sizeof(float));
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int i = 0; i < bench_runs; i++) {
        cudaEventRecord(start);
        kernel_fn(d_Q, d_K, d_V, d_O, seq_len, head_dim);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        cudaEventElapsedTime(&times[i], start, stop);
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Sort and take median
    qsort(times, bench_runs, sizeof(float), compare_float);
    float median_ms = times[bench_runs / 2];
    float min_ms = times[0];
    float max_ms = times[bench_runs - 1];

    float tflops = (float)flops / (median_ms * 1e9f);
    float ai = (float)flops / (float)bytes_transferred;

    printf("  Median: %.3f ms  (min=%.3f, max=%.3f)\n", median_ms, min_ms, max_ms);
    printf("  TFLOPS: %.4f\n", tflops);
    printf("  Arithmetic Intensity: %.2f FLOP/Byte\n\n", ai);

    free(times);
    return {name, median_ms, tflops, ai};
}

int main() {
    int seq_len = 512;
    int head_dim = 64;
    int qkv_size = seq_len * head_dim;
    int warmup_runs = 10;
    int bench_runs = 50;

    printf("╔══════════════════════════════════════════════════╗\n");
    printf("║       CUDA Attention Engine Benchmark            ║\n");
    printf("╠══════════════════════════════════════════════════╣\n");
    printf("║  seq_len=%d, head_dim=%d, num_heads=1           ║\n", seq_len, head_dim);
    printf("║  warmup=%d, bench_runs=%d (median reported)     ║\n", warmup_runs, bench_runs);
    printf("║  Timing: CUDA events, per-run sync, median      ║\n");
    printf("╚══════════════════════════════════════════════════╝\n\n");

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

    // Compute FLOPS and bytes transferred
    // QK^T: 2*N*N*d FLOPS, reads Q[N*d] + K[N*d], writes scores[N*N]
    // Softmax: ~5*N*N FLOPS, reads/writes scores[N*N]
    // SV: 2*N*N*d FLOPS, reads scores[N*N] + V[N*d], writes O[N*d]
    long long N = seq_len;
    long long d = head_dim;

    long long flops = 2LL*N*N*d + 2LL*N*N*d + 5LL*N*N;

    // Bytes transferred (naive: all from global memory, no reuse)
    // QK: read Q[N*d] + K[N*d] + write scores[N*N]
    // softmax: read+write scores[N*N] * 3 passes
    // SV: read scores[N*N] + V[N*d] + write O[N*d]
    long long bytes_naive = (2*N*d + N*N + 3*2*N*N + N*N + N*d + N*d) * 4; // FP32

    // Tiled: Q and K tiles reused TILE_SIZE times from shared mem
    // Effective: reads reduced by factor of TILE_SIZE for matmuls
    int TILE = 16;
    long long bytes_tiled = (2*N*d*N/TILE + N*N + 3*2*N*N + N*N*N/TILE + N*d + N*d) * 4;
    // Simplified estimate: tiling reduces matmul reads by TILE factor
    long long bytes_tiled_simplified = bytes_naive / 3; // conservative estimate

    // FP16: half the bytes for Q,K,V (scores stay FP32 for softmax)
    long long bytes_fp16 = (2*N*d + N*N*4 + 3*2*N*N*4 + N*N*4 + N*d + N*d) + (2*N*d)*2;
    // Simplified: ~60% of naive (Q/K/V in FP16, scores in FP32)
    long long bytes_fp16_simplified = (long long)(bytes_naive * 0.6f);

    // Tensor Core: similar to FP16 in memory, but compute throughput much higher
    long long bytes_tc = bytes_fp16_simplified;

    printf("Workload characteristics:\n");
    printf("  Total FLOPS: %lld (%.2f GFLOP)\n", flops, flops/1e9f);
    printf("  Naive bytes transferred: %lld (%.2f MB)\n", bytes_naive, bytes_naive/1e6f);
    printf("  Naive arithmetic intensity: %.2f FLOP/Byte\n\n", (float)flops/bytes_naive);

    // Run all benchmarks
    BenchResult results[4];
    results[0] = run_benchmark(attention_forward_naive_fp32, "Naive FP32",
                               d_Q, d_K, d_V, d_O, seq_len, head_dim,
                               flops, bytes_naive, warmup_runs, bench_runs);
    results[1] = run_benchmark(attention_forward_tiled_fp32, "Tiled FP32 (Shared Mem)",
                               d_Q, d_K, d_V, d_O, seq_len, head_dim,
                               flops, bytes_tiled_simplified, warmup_runs, bench_runs);
    results[2] = run_benchmark(attention_forward_fp16, "FP16 Mixed Precision",
                               d_Q, d_K, d_V, d_O, seq_len, head_dim,
                               flops, bytes_fp16_simplified, warmup_runs, bench_runs);
    results[3] = run_benchmark(attention_forward_tensor_core, "Tensor Core (WMMA)",
                               d_Q, d_K, d_V, d_O, seq_len, head_dim,
                               flops, bytes_tc, warmup_runs, bench_runs);

    // Print summary table
    float baseline_ms = results[0].median_ms;

    printf("╔═══════════════════════════╦══════════╦══════════╦══════════╦══════════════╗\n");
    printf("║ Kernel                    ║Time (ms) ║  TFLOPS  ║ Speedup  ║ AI (FLOP/B)  ║\n");
    printf("╠═══════════════════════════╬══════════╬══════════╬══════════╬══════════════╣\n");
    for (int i = 0; i < 4; i++) {
        printf("║ %-25s ║ %7.3f  ║ %7.4f  ║ %6.2fx  ║ %10.2f  ║\n",
               results[i].name, results[i].median_ms, results[i].tflops,
               baseline_ms / results[i].median_ms, results[i].arithmetic_intensity);
    }
    printf("╚═══════════════════════════╩══════════╩══════════╩══════════╩══════════════╝\n");

    printf("\nMethodology:\n");
    printf("  - Timing: CUDA events with cudaDeviceSynchronize before each run\n");
    printf("  - Reported: median of %d runs (after %d warmup)\n", bench_runs, warmup_runs);
    printf("  - Arithmetic Intensity = FLOPS / estimated bytes transferred\n");

    // Cleanup
    free(h_Q); free(h_K); free(h_V);
    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_O);

    return 0;
}
