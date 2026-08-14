#!/bin/bash
# Quick start: build and run everything on a fresh GPU server
# Usage: bash scripts/run_all.sh

set -e

echo "=== Building ==="
mkdir -p build
cd build
cmake ..
make -j$(nproc)
cd ..

echo ""
echo "=== Correctness Tests ==="
./build/test_correctness

echo ""
echo "=== Benchmark ==="
./build/benchmark

echo ""
echo "=== Done ==="
echo "Next: run scripts/profile.sh for Nsight Compute analysis"
