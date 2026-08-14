#!/bin/bash
# Phase 5: Nsight Compute Profiling Script
# Run this on the GPU server after compiling

# Build first
mkdir -p build && cd build && cmake .. && make -j && cd ..

echo "=== Phase 5: Profiling All Kernels ==="
echo ""

# Create output directory for profiling reports
mkdir -p profiling_reports

# Profile each kernel individually with Nsight Compute
# ncu = Nsight Compute CLI tool

echo "[1/4] Profiling Naive FP32..."
ncu --set full \
    --export profiling_reports/naive_fp32 \
    ./build/benchmark --kernel naive

echo "[2/4] Profiling Tiled FP32..."
ncu --set full \
    --export profiling_reports/tiled_fp32 \
    ./build/benchmark --kernel tiled

echo "[3/4] Profiling FP16..."
ncu --set full \
    --export profiling_reports/fp16 \
    ./build/benchmark --kernel fp16

echo "[4/4] Profiling Tensor Core..."
ncu --set full \
    --export profiling_reports/tensor_core \
    ./build/benchmark --kernel tc

echo ""
echo "=== Profiling Complete ==="
echo "Reports saved to profiling_reports/"
echo "Open with: ncu-ui profiling_reports/naive_fp32.ncu-rep"
echo ""

# Quick summary: key metrics to look at
echo "=== Key Metrics to Analyze ==="
echo "1. Memory Throughput (%)     — how much bandwidth are we using?"
echo "2. Compute Throughput (%)    — how much compute are we using?"
echo "3. Achieved Occupancy (%)    — how full are the SMs?"
echo "4. L1/L2 Cache Hit Rate (%)  — are we reusing data?"
echo "5. Warp Stall Reasons        — why are threads waiting?"
echo ""
echo "Decision framework:"
echo "  Memory Throughput HIGH + Compute LOW = memory bound → more tiling/caching"
echo "  Compute Throughput HIGH + Memory LOW = compute bound → algorithmic change"
echo "  Both LOW = latency bound → increase occupancy or overlap"
