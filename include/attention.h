#pragma once

#include <cuda_runtime.h>

enum AttentionPrecision {
    PRECISION_FP32 = 0,
    PRECISION_FP16 = 1,
    PRECISION_TENSOR_CORE = 2
};

// Forward declaration of attention kernels
// All matrices are in row-major layout:
//   Q: [seq_len, head_dim]
//   K: [seq_len, head_dim]
//   V: [seq_len, head_dim]
//   O: [seq_len, head_dim] (output)

// Naive FP32 attention (Phase 1 baseline)
void attention_forward_naive_fp32(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    int seq_len,
    int head_dim
);

// Tensor Core attention using WMMA (Phase 4)
void attention_forward_tensor_core(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    int seq_len,
    int head_dim
);

// FP16 attention with mixed precision (Phase 3)
// Input/output in FP32 on host side; internally computes in FP16
void attention_forward_fp16(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    int seq_len,
    int head_dim
);

// Tiled FP32 attention with shared memory (Phase 2)
void attention_forward_tiled_fp32(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    int seq_len,
    int head_dim
);

// CPU reference implementation for correctness validation
void attention_forward_cpu(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    int seq_len,
    int head_dim
);
