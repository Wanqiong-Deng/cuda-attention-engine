# Experiment guide

## Reproducible command

```bash
python bench/workload_bench.py \
  --output reports/rollout_$(hostname).json \
  --runs 30 --warmup 10
```

Use `--quick` only to validate the harness; do not cite its output as a scaling
result. Keep result files with the commit that generated them.

## What each experiment means

### Uniform causal attention

Measures a conventional fixed-length baseline at several context lengths. Latency
should grow approximately quadratically for full-sequence causal attention, but
kernel selection and memory effects mean the observed curve need not be exactly
`4x` for every doubling.

### Variable-length rollout: padded vs length buckets

The padded baseline computes every item at the trace maximum length. The bucketed
baseline groups items with equal length and launches each group separately. Its
wall-clock result includes those extra launches; this is intentional.

The JSON field `attention_work_efficiency` reports
`sum(length^2) / (batch * max(length)^2)`. It estimates useful causal-attention
work before accounting for GPU implementation details. It is not a measured
speedup.

### Shared-prefix accounting

For `B` requests sharing a prefix of length `P`, a cache stores the prefix KV
once rather than `B` times. The harness reports the avoided KV elements and K/V
projection work. Attention for suffix queries still reads/attends to the prefix,
so treat this as a prefill/memory model rather than an attention-kernel benchmark.

## Profiling follow-up

```bash
nsys profile --trace=cuda,nvtx -o reports/rollout_timeline \
  python bench/workload_bench.py --quick

ncu --set full --target-processes all --export reports/sdpa_profile \
  python bench/workload_bench.py --quick
```

Record kernel-launch gaps, memory allocation/copies, achieved occupancy, and
memory throughput. Compare only matching causal paths and shapes.
