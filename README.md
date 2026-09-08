# Rollout-Aware Attention Lab

An experiment harness for understanding how attention behavior affects the cost of
agentic RL rollouts. It combines small from-scratch CUDA attention kernels with
production-facing experiments over causal SDPA and FlashAttention.

> **Scope:** this is a learning and measurement project, not a replacement for
> FlashAttention or a production inference server. Custom kernels use a dense,
> single-head attention API; rollout experiments use causal PyTorch attention.

## Why this exists

In post-training and agentic RL, the important attention workload is often not a
single fixed-shape training batch:

- Rollouts finish at very different lengths, so padding can dominate work.
- System prompts and trajectory prefixes are shared across many samples.
- Context grows over an episode, making prefill and decode bottlenecks different.

This repository turns those claims into reproducible experiments. The goal is to
measure the cost of three scheduling choices: padding, length bucketing, and
prefix-KV reuse.

## What is implemented today

### Kernel learning track

- Dense attention forward pass in CUDA: naive FP32, shared-memory tiled FP32,
  FP16 mixed precision, and a WMMA Tensor Core prototype.
- CPU-reference correctness tests.
- CUDA-event benchmark reporting median latency.

These implementations materialize the `N x N` score matrix and are intentionally
educational. They are not fused FlashAttention kernels and are **not** compared
as like-for-like alternatives to FlashAttention.

### Rollout workload track

- Causal PyTorch SDPA measurements across context lengths.
- Optional FlashAttention-2 measurements when installed.
- A variable-length rollout workload measured both as a padded batch and as
  sequential length buckets.
- Prefix-cache accounting that reports KV-memory and projection-work savings
  separately from attention compute.
- Machine-readable JSON output for runs and environment metadata.

See [the experiment guide](docs/experiments.md) for methodology and
[the target design](docs/target_design.md) for the next implementation milestones.

## Quick start

### 1. Build the CUDA learning kernels

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/test_correctness
./build/benchmark
```

### 2. Run rollout experiments

```bash
# Required: CUDA-enabled PyTorch. FlashAttention is optional.
python -m pip install torch
python bench/workload_bench.py --output reports/rollout_a100.json

# Reduce memory use while checking setup.
python bench/workload_bench.py --quick --output reports/rollout_smoke.json
```

The benchmark exits early with a clear message if CUDA is unavailable. Each result
file contains the GPU, PyTorch/CUDA versions, configuration, raw latency samples,
and derived efficiency metrics.

## Results policy

No throughput or speedup is claimed until it is measured on a named GPU and the
corresponding JSON result is checked into `reports/`. A valid report records:

- GPU / driver / CUDA / PyTorch versions;
- exact command and benchmark configuration;
- warmup count, sample count, and median/p95 latency;
- whether FlashAttention was installed and which backend was exercised.

This makes negative results useful too: for example, bucketing may lose at small
batch sizes because additional launches outweigh reduced padding.

## Repository layout

```
include/attention.h        Dense single-head attention API
src/                       CUDA learning kernels and CPU reference
tests/test_correctness.cu  GPU vs CPU correctness checks
bench/benchmark.cu         CUDA kernel microbenchmark
bench/workload_bench.py    Rollout-aware causal-attention experiments
docs/experiments.md        Measurement methodology and interpretation
docs/target_design.md      Target-version design and milestones
reports/                   Checked-in JSON results; large profiler artifacts stay ignored
```

## Resume wording

Use only wording that matches the state you have actually reached. Once the
rollout benchmark has been run and its result JSON is committed, a defensible
version is:

> Built a CUDA attention performance lab for agentic-RL rollout workloads;
> benchmarked causal SDPA/FlashAttention under variable-length and shared-prefix
> traces, using CUDA-event and Nsight measurements to quantify padding, context
> growth, and launch-overhead trade-offs.

If the varlen/prefix-cache implementation milestone is complete, add the measured
result rather than an unverified percentage, e.g. “reduced padded attention work
by **X%** on a recorded rollout trace.”

## Hardware

CUDA kernels require a CUDA-capable GPU. The WMMA prototype requires Volta or
newer hardware and dimensions divisible by 16. The rollout benchmark requires
CUDA-enabled PyTorch; A100, H100, or L40S are sensible profiling targets.
