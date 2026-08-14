#include "attention.h"
#include <cfloat>

#define TILE_SIZE 16

// =============================================================
// Phase 2: Tiled Attention with Shared Memory
//
// Key insight: naive kernel 里每个 thread 独立从 global memory 读取
// 整行/整列数据，大量重复读取。Tiling 的思路是：
//   一个 block 里的 thread 合作把一块数据搬到 shared memory，
//   然后大家从 shared memory（快100x）重复读取。
// =============================================================

// Kernel 1: Tiled Q × K^T
// 和矩阵乘法的经典 tiling 一模一样
__global__ void tiled_qk_kernel(
    const float* Q,
    const float* K,
    float* scores,
    int seq_len,
    int head_dim,
    float scale
) {
    __shared__ float tile_Q[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_K[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float dot = 0.0f;

    // 沿 head_dim 方向分块
    int num_tiles = (head_dim + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < num_tiles; t++) {
        // 合作加载：每个 thread 搬一个元素到 shared memory
        int q_col = t * TILE_SIZE + threadIdx.x;
        if (row < seq_len && q_col < head_dim)
            tile_Q[threadIdx.y][threadIdx.x] = Q[row * head_dim + q_col];
        else
            tile_Q[threadIdx.y][threadIdx.x] = 0.0f;

        int k_col = t * TILE_SIZE + threadIdx.y;
        if (col < seq_len && k_col < head_dim)
            tile_K[threadIdx.y][threadIdx.x] = K[col * head_dim + k_col];
        else
            tile_K[threadIdx.y][threadIdx.x] = 0.0f;

        // 等所有 thread 都加载完
        __syncthreads();

        // 从 shared memory 算部分点积
        for (int d = 0; d < TILE_SIZE; d++) {
            dot += tile_Q[threadIdx.y][d] * tile_K[d][threadIdx.x];
        }

        // 等所有 thread 都算完，再加载下一块
        __syncthreads();
    }

    if (row < seq_len && col < seq_len) {
        scores[row * seq_len + col] = dot * scale;
    }
}

// Kernel 2: Row-wise softmax with shared memory reduction
// 用一个 block 处理一行，block 内 thread 协作找 max 和 sum
__global__ void tiled_softmax_kernel(
    float* scores,
    int seq_len
) {
    extern __shared__ float sdata[];

    int row = blockIdx.x;
    int tid = threadIdx.x;
    int block_size = blockDim.x;

    // Phase 1: 找这一行的最大值（parallel reduction）
    float local_max = -FLT_MAX;
    for (int j = tid; j < seq_len; j += block_size) {
        float val = scores[row * seq_len + j];
        if (val > local_max) local_max = val;
    }
    sdata[tid] = local_max;
    __syncthreads();

    // 树形归约找全局 max
    for (int stride = block_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (sdata[tid + stride] > sdata[tid])
                sdata[tid] = sdata[tid + stride];
        }
        __syncthreads();
    }
    float row_max = sdata[0];

    // Phase 2: 计算 exp(x - max) 并求和（parallel reduction）
    float local_sum = 0.0f;
    for (int j = tid; j < seq_len; j += block_size) {
        float val = expf(scores[row * seq_len + j] - row_max);
        scores[row * seq_len + j] = val;
        local_sum += val;
    }
    sdata[tid] = local_sum;
    __syncthreads();

    // 树形归约求总和
    for (int stride = block_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }
    float row_sum = sdata[0];

    // Phase 3: 归一化
    for (int j = tid; j < seq_len; j += block_size) {
        scores[row * seq_len + j] /= row_sum;
    }
}

// Kernel 3: Tiled scores × V
__global__ void tiled_sv_kernel(
    const float* scores,
    const float* V,
    float* O,
    int seq_len,
    int head_dim
) {
    __shared__ float tile_S[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_V[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float val = 0.0f;

    int num_tiles = (seq_len + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < num_tiles; t++) {
        int s_col = t * TILE_SIZE + threadIdx.x;
        if (row < seq_len && s_col < seq_len)
            tile_S[threadIdx.y][threadIdx.x] = scores[row * seq_len + s_col];
        else
            tile_S[threadIdx.y][threadIdx.x] = 0.0f;

        int v_row = t * TILE_SIZE + threadIdx.y;
        if (v_row < seq_len && col < head_dim)
            tile_V[threadIdx.y][threadIdx.x] = V[v_row * head_dim + col];
        else
            tile_V[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for (int d = 0; d < TILE_SIZE; d++) {
            val += tile_S[threadIdx.y][d] * tile_V[d][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < seq_len && col < head_dim) {
        O[row * head_dim + col] = val;
    }
}

// Host function
void attention_forward_tiled_fp32(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    int seq_len,
    int head_dim
) {
    float scale = 1.0f / sqrtf((float)head_dim);

    float* scores;
    cudaMalloc(&scores, seq_len * seq_len * sizeof(float));

    // Kernel 1: Tiled Q × K^T
    dim3 block1(TILE_SIZE, TILE_SIZE);
    dim3 grid1((seq_len + TILE_SIZE - 1) / TILE_SIZE, (seq_len + TILE_SIZE - 1) / TILE_SIZE);
    tiled_qk_kernel<<<grid1, block1>>>(Q, K, scores, seq_len, head_dim, scale);

    // Kernel 2: Softmax (one block per row, 256 threads)
    int softmax_threads = 256;
    int shared_mem_size = softmax_threads * sizeof(float);
    tiled_softmax_kernel<<<seq_len, softmax_threads, shared_mem_size>>>(scores, seq_len);

    // Kernel 3: Tiled scores × V
    dim3 block3(TILE_SIZE, TILE_SIZE);
    dim3 grid3((head_dim + TILE_SIZE - 1) / TILE_SIZE, (seq_len + TILE_SIZE - 1) / TILE_SIZE);
    tiled_sv_kernel<<<grid3, block3>>>(scores, V, O, seq_len, head_dim);

    cudaFree(scores);
}
