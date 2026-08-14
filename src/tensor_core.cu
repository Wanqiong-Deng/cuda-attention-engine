#include "attention.h"
#include <cuda_fp16.h>
#include <mma.h>
#include <cfloat>

using namespace nvcuda;

// =============================================================
// Phase 4: Tensor Core Attention (WMMA API)
//
// Tensor Cores are specialized hardware units on the GPU that
// can do a 16×16×16 matrix multiply-accumulate in ONE clock cycle.
//
// Normal CUDA cores: each thread does one multiply-add.
//   256 threads × 1 multiply-add = 256 operations per cycle
//
// Tensor Core: one warp (32 threads) collectively does:
//   D = A × B + C   where A is 16×16, B is 16×16, C is 16×16
//   = 16×16×16 = 4096 multiply-add operations in one go
//
// That's why Tensor Cores are 10-16x faster for matrix math.
//
// WMMA = Warp Matrix Multiply Accumulate
//   - A warp (32 threads) cooperates to hold matrix "fragments"
//   - Each thread holds a piece of the 16×16 matrix
//   - You don't index individual elements — you load/store/compute whole fragments
// =============================================================

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

// Kernel 1: Q × K^T using Tensor Cores
// Each warp computes a 16×16 tile of the output scores matrix
__global__ void tc_qk_kernel(
    const __half* Q,
    const __half* K,
    float* scores,
    int seq_len,
    int head_dim,
    float scale
) {
    // Which 16×16 output tile does this warp handle?
    int warpM = (blockIdx.y * blockDim.y + threadIdx.y) / 32;  // not used this way
    // Simpler: one warp per 16×16 output tile
    int warp_row = blockIdx.y * WMMA_M;
    int warp_col = blockIdx.x * WMMA_N;

    // Declare fragments (pieces of 16×16 matrices distributed across 32 threads)
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    // Initialize accumulator to zero
    wmma::fill_fragment(c_frag, 0.0f);

    // Loop over head_dim in chunks of 16
    int num_tiles = (head_dim + WMMA_K - 1) / WMMA_K;
    for (int t = 0; t < num_tiles; t++) {
        int k_offset = t * WMMA_K;

        // Load Q tile [warp_row : warp_row+16, k_offset : k_offset+16]
        if (warp_row < seq_len && k_offset < head_dim) {
            wmma::load_matrix_sync(a_frag, Q + warp_row * head_dim + k_offset, head_dim);
        }

        // Load K tile — col_major because we want K^T
        // K is [seq_len, head_dim] row-major, loading as col_major gives us K^T
        if (warp_col < seq_len && k_offset < head_dim) {
            wmma::load_matrix_sync(b_frag, K + warp_col * head_dim + k_offset, head_dim);
        }

        // Tensor Core: c_frag += a_frag × b_frag
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    // Apply scale to accumulator
    for (int i = 0; i < c_frag.num_elements; i++) {
        c_frag.x[i] *= scale;
    }

    // Store result to scores matrix
    if (warp_row < seq_len && warp_col < seq_len) {
        wmma::store_matrix_sync(scores + warp_row * seq_len + warp_col, c_frag, seq_len, wmma::mem_row_major);
    }
}

// Kernel 2: Softmax (same as Phase 3 — stays in FP32)
__global__ void tc_softmax_kernel(
    float* scores,
    int seq_len
) {
    extern __shared__ float sdata[];

    int row = blockIdx.x;
    int tid = threadIdx.x;
    int block_size = blockDim.x;

    float local_max = -FLT_MAX;
    for (int j = tid; j < seq_len; j += block_size) {
        float val = scores[row * seq_len + j];
        if (val > local_max) local_max = val;
    }
    sdata[tid] = local_max;
    __syncthreads();

    for (int stride = block_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (sdata[tid + stride] > sdata[tid])
                sdata[tid] = sdata[tid + stride];
        }
        __syncthreads();
    }
    float row_max = sdata[0];

    float local_sum = 0.0f;
    for (int j = tid; j < seq_len; j += block_size) {
        float val = expf(scores[row * seq_len + j] - row_max);
        scores[row * seq_len + j] = val;
        local_sum += val;
    }
    sdata[tid] = local_sum;
    __syncthreads();

    for (int stride = block_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }
    float row_sum = sdata[0];

    for (int j = tid; j < seq_len; j += block_size) {
        scores[row * seq_len + j] /= row_sum;
    }
}

// Kernel 3: scores × V using Tensor Cores
// scores is FP32, need to convert to FP16 for Tensor Core input
__global__ void tc_sv_kernel(
    const float* scores,
    const __half* V,
    float* O,
    int seq_len,
    int head_dim
) {
    int warp_row = blockIdx.y * WMMA_M;
    int warp_col = blockIdx.x * WMMA_N;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);

    // We need scores in FP16 for Tensor Core
    // Load FP32 scores into a temporary FP16 fragment
    int num_tiles = (seq_len + WMMA_K - 1) / WMMA_K;

    // Shared memory for FP32→FP16 conversion of scores tile
    __shared__ __half scores_half[WMMA_M][WMMA_K];

    int lane_id = threadIdx.x % 32;

    for (int t = 0; t < num_tiles; t++) {
        int k_offset = t * WMMA_K;

        // Convert scores tile from FP32 to FP16 (cooperative within warp)
        for (int i = lane_id; i < WMMA_M * WMMA_K; i += 32) {
            int r = i / WMMA_K;
            int c = i % WMMA_K;
            int global_row = warp_row + r;
            int global_col = k_offset + c;
            if (global_row < seq_len && global_col < seq_len)
                scores_half[r][c] = __float2half(scores[global_row * seq_len + global_col]);
            else
                scores_half[r][c] = __float2half(0.0f);
        }
        __syncwarp();

        wmma::load_matrix_sync(a_frag, &scores_half[0][0], WMMA_K);

        // Load V tile
        if (k_offset < seq_len && warp_col < head_dim) {
            wmma::load_matrix_sync(b_frag, V + k_offset * head_dim + warp_col, head_dim);
        }

        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    // Store output (FP32)
    if (warp_row < seq_len && warp_col < head_dim) {
        wmma::store_matrix_sync(O + warp_row * head_dim + warp_col, c_frag, head_dim, wmma::mem_row_major);
    }
}

// Helper: convert float to half (same as Phase 3)
__global__ void tc_float_to_half(const float* input, __half* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = __float2half(input[idx]);
    }
}

// Host function
void attention_forward_tensor_core(
    const float* Q,      // device pointer, FP32
    const float* K,      // device pointer, FP32
    const float* V,      // device pointer, FP32
    float* O,            // device pointer, FP32
    int seq_len,
    int head_dim
) {
    int qkv_size = seq_len * head_dim;
    float scale = 1.0f / sqrtf((float)head_dim);

    // Convert inputs to FP16 (Tensor Cores require FP16 input)
    __half *d_Q_half, *d_K_half, *d_V_half;
    cudaMalloc(&d_Q_half, qkv_size * sizeof(__half));
    cudaMalloc(&d_K_half, qkv_size * sizeof(__half));
    cudaMalloc(&d_V_half, qkv_size * sizeof(__half));

    int threads = 256;
    int blocks = (qkv_size + threads - 1) / threads;
    tc_float_to_half<<<blocks, threads>>>(Q, d_Q_half, qkv_size);
    tc_float_to_half<<<blocks, threads>>>(K, d_K_half, qkv_size);
    tc_float_to_half<<<blocks, threads>>>(V, d_V_half, qkv_size);

    // Allocate scores (FP32)
    float* scores;
    cudaMalloc(&scores, seq_len * seq_len * sizeof(float));

    // Kernel 1: Tensor Core Q × K^T
    // One warp per 16×16 output tile, one warp = 32 threads
    // Block: 128 threads = 4 warps (but we use 1 warp per tile for simplicity)
    dim3 grid1((seq_len + WMMA_N - 1) / WMMA_N, (seq_len + WMMA_M - 1) / WMMA_M);
    dim3 block1(32, 1);  // one warp per block (simple version)
    tc_qk_kernel<<<grid1, block1>>>(d_Q_half, d_K_half, scores, seq_len, head_dim, scale);

    // Kernel 2: Softmax (FP32)
    int softmax_threads = 256;
    int shared_mem_size = softmax_threads * sizeof(float);
    tc_softmax_kernel<<<seq_len, softmax_threads, shared_mem_size>>>(scores, seq_len);

    // Kernel 3: Tensor Core scores × V
    dim3 grid3((head_dim + WMMA_N - 1) / WMMA_N, (seq_len + WMMA_M - 1) / WMMA_M);
    dim3 block3(32, 1);
    tc_sv_kernel<<<grid3, block3>>>(scores, d_V_half, O, seq_len, head_dim);

    // Cleanup
    cudaFree(d_Q_half);
    cudaFree(d_K_half);
    cudaFree(d_V_half);
    cudaFree(scores);
}
