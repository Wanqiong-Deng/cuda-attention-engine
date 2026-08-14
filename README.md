# CUDA Attention Engine

A profiling-driven study of attention kernel performance under LLM inference workloads, with emphasis on Agentic RL rollout patterns.

## What This Is

- From-scratch CUDA attention kernels (naive → shared memory tiling → FP16 → Tensor Core) as a learning foundation
- Benchmarking framework comparing custom kernels vs PyTorch SDPA vs FlashAttention-2
- RL-representative workload characterization: variable-length batching, prefix sharing, context scaling
- Nsight Compute / Nsight Systems profiling analysis

## Quick Start

```bash
# Build CUDA kernels
mkdir build && cd build && cmake .. && make -j && cd ..

# Run correctness tests
./build/test_correctness

# Run kernel benchmark
./build/benchmark

# Run RL workload benchmark (requires PyTorch + flash-attn)
pip install torch flash-attn
python bench/workload_bench.py

# Profile with Nsight
nsys profile --trace=cuda python bench/workload_bench.py
ncu --set full ./build/benchmark
```

## Project Structure

```
├── include/attention.h         — API declarations
├── src/
│   ├── naive_fp32.cu           — Baseline (global memory only)
│   ├── tiled_fp32.cu           — Shared memory tiling
│   ├── fp16.cu                 — FP16 mixed precision
│   └── tensor_core.cu          — Tensor Core (WMMA)
├── bench/
│   ├── benchmark.cu            — CUDA kernel comparison
│   └── workload_bench.py       — RL workload profiling (PyTorch + FA2)
├── tests/test_correctness.cu   — GPU vs CPU correctness validation
├── scripts/                    — Build & profiling scripts
└── docs/                       — Profiling analysis reports
```

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | Volta+ (Tensor Core) | A100 / H100 |
| CUDA | 11.0+ | 12.x |
| CMake | 3.18+ | — |

## Status

- [x] Phase 1-4: Kernel implementations (naive, tiled, FP16, Tensor Core)
- [x] Correctness tests & benchmark harness
- [x] RL workload benchmark script
- [ ] On-GPU profiling & analysis (pending GPU access)
- [ ] Nsight Compute roofline analysis
- [ ] RL workload characterization report
