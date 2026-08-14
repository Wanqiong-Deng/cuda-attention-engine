# Phase 5: Profiling Analysis Template

## GPU Info
- GPU Model: [e.g., A100 80GB]
- CUDA Version: [e.g., 12.2]
- Driver Version: [fill in]

## Benchmark Results

| Kernel | Time (ms) | TFLOPS | Speedup | % of Peak |
|--------|-----------|--------|---------|-----------|
| Naive FP32 | | | 1.00x | |
| Tiled FP32 | | | | |
| FP16 Mixed | | | | |
| Tensor Core | | | | |

Peak TFLOPS for this GPU: FP32 = ___, FP16 = ___, TC = ___

## Nsight Compute Analysis

### Naive FP32 Kernel (QK)

| Metric | Value | Assessment |
|--------|-------|------------|
| Compute Throughput | % | |
| Memory Throughput | % | |
| Achieved Occupancy | % | |
| L2 Cache Hit Rate | % | |

**Diagnosis:** [memory bound / compute bound / latency bound]
**Bottleneck:** [describe]
**Action taken:** → Phase 2 tiling

---

### Tiled FP32 Kernel (QK)

| Metric | Value | Improvement vs Naive |
|--------|-------|---------------------|
| Compute Throughput | % | |
| Memory Throughput | % | |
| Achieved Occupancy | % | |
| L2 Cache Hit Rate | % | |

**Diagnosis:**
**Remaining bottleneck:**
**Action taken:** → Phase 3 FP16

---

### FP16 Kernel (QK)

| Metric | Value | Improvement vs Tiled |
|--------|-------|---------------------|
| Compute Throughput | % | |
| Memory Throughput | % | |
| Achieved Occupancy | % | |

**Diagnosis:**
**Remaining bottleneck:**
**Action taken:** → Phase 4 Tensor Core

---

### Tensor Core Kernel (QK)

| Metric | Value | Improvement vs FP16 |
|--------|-------|---------------------|
| Compute Throughput | % | |
| Memory Throughput | % | |
| Achieved Occupancy | % | |

**Diagnosis:**
**Remaining bottleneck:**
**Possible further optimizations:**
- [ ] Increase occupancy (reduce registers per thread)
- [ ] Double buffering (overlap compute and memory load)
- [ ] Larger tile sizes
- [ ] Fuse softmax with QK kernel

## Roofline Analysis

```
         ▲ FLOPS
         │
Compute  │─────────────────────────╱ peak compute
Ceiling  │                       ╱
         │                     ╱
         │        ●TC        ╱
         │                 ╱
         │      ●FP16    ╱
         │             ╱
         │   ●Tiled  ╱  memory bandwidth ceiling
         │         ╱
         │●Naive ╱
         │     ╱
         └───╱──────────────────→ Arithmetic Intensity (FLOPS/byte)
```

Arithmetic Intensity of our attention kernel:
- QK matmul: 2*N*d FLOPS / (2*N*d + N*N) bytes ≈ ___
- Assessment: [memory bound / compute bound / balanced]

## Summary & Conclusions

1. Biggest single optimization: [Phase ? → Phase ?] gave ___x speedup because ___
2. Current bottleneck after all optimizations: ___
3. If I had more time, I would: ___
4. Comparison to FlashAttention/PyTorch: ___
