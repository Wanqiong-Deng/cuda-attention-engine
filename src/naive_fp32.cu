#include "attention.h"
#include <cfloat>

// Naive FP32 attention kernel — Phase 1 baseline
// Each thread computes one element of the output matrix O[row][col]
// No optimization: all reads from global memory, no tiling, no shared memory

// Kernel 1: Compute Q × K^T and apply scale, store attention scores
__global__ void naive_qk_kernel(
    const float* Q,
    const float* K,
    float* scores,
    int seq_len,
    int head_dim,
    float scale
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < seq_len && col < seq_len) {
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            dot += Q[row * head_dim + d] * K[col * head_dim + d];
        }
        scores[row * seq_len + col] = dot * scale;
    }
}

// Kernel 2: Row-wise softmax on the scores matrix
// Two-pass: first find max per row, then compute exp and normalize
__global__ void naive_softmax_kernel(
    float* scores,
    int seq_len
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= seq_len) return;

    // Pass 1: find max
    float max_val = -FLT_MAX;
    for (int j = 0; j < seq_len; j++) {
        float val = scores[row * seq_len + j];
        if (val > max_val) max_val = val;
    }

    // Pass 2: exp and sum
    float sum_exp = 0.0f;
    for (int j = 0; j < seq_len; j++) {
        float val = expf(scores[row * seq_len + j] - max_val);
        scores[row * seq_len + j] = val;
        sum_exp += val;
    }

    // Pass 3: normalize
    for (int j = 0; j < seq_len; j++) {
        scores[row * seq_len + j] /= sum_exp;
    }
}

// Kernel 3: scores × V = Output
__global__ void naive_sv_kernel(
    const float* scores,
    const float* V,
    float* O,
    int seq_len,
    int head_dim
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < seq_len && col < head_dim) {
        float val = 0.0f;
        for (int j = 0; j < seq_len; j++) {
            val += scores[row * seq_len + j] * V[j * head_dim + col];
        }
        O[row * head_dim + col] = val;
    }
}

// Host function: orchestrates the three kernels
void attention_forward_naive_fp32(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    int seq_len,
    int head_dim
) {
    float scale = 1.0f / sqrtf((float)head_dim);

    // Allocate intermediate scores matrix on device
    float* scores;
    cudaMalloc(&scores, seq_len * seq_len * sizeof(float));

    // Kernel 1: Q × K^T
    dim3 block1(16, 16);
    dim3 grid1((seq_len + 15) / 16, (seq_len + 15) / 16);
    naive_qk_kernel<<<grid1, block1>>>(Q, K, scores, seq_len, head_dim, scale);

    // Kernel 2: Softmax
    int threads = 256;
    int blocks = (seq_len + threads - 1) / threads;
    naive_softmax_kernel<<<blocks, threads>>>(scores, seq_len);

    // Kernel 3: scores × V
    dim3 block3(16, 16);
    dim3 grid3((head_dim + 15) / 16, (seq_len + 15) / 16);
    naive_sv_kernel<<<grid3, block3>>>(scores, V, O, seq_len, head_dim);

    cudaFree(scores);
}
