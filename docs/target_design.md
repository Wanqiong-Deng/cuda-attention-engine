# Target version: rollout-aware attention measurement layer

## Problem statement

Agentic RL produces a mixture of prefill and decode requests with variable
trajectory lengths, shared system prompts, and growing contexts. Dense attention
microbenchmarks explain CUDA fundamentals, but do not establish how request shape
and scheduling change end-to-end attention cost.

The target version of this repository is a small, reproducible measurement layer
for those trade-offs. It is intentionally narrower than vLLM, SGLang, or
FlashAttention: it generates controlled traces, executes attention backends, and
explains the observed cost.

## Deliverables

| Deliverable | Definition of done | Status |
|---|---|---|
| Dense-kernel learning track | CPU correctness checks and CUDA-event median benchmark | Present |
| Padded rollout baseline | Causal SDPA/optional FA2 over a fixed synthetic trace | Present |
| Length-bucket experiment | Same trace measured as padded and grouped-by-length execution; JSON includes latency and theoretical padding work | Present |
| Prefix-cache cost model | Reports shared-KV storage and K/V projection savings; does not mislabel it as saved attention FLOPs | Present |
| Varlen backend | Run FlashAttention varlen API or an equivalent packed backend on the exact trace | Planned |
| Real prefix-KV cache | Reuse prefix K/V tensors across requests and measure prefill/decode separately | Planned |
| Evidence pack | Raw JSON, Nsight exports, chart, and one-page findings for a named GPU | Planned |

## Experiment contract

All experiments must report three distinct quantities:

1. **Observed wall-clock latency** — CUDA event median and p95, including the
   chosen batching/scheduling strategy.
2. **Attention work estimate** — proportional to `sum_i L_i^2` for causal
   self-attention, compared with padded `B * max(L)^2`. This is a shape-derived
   efficiency metric, not a timing result.
3. **Prefix reuse model** — KV storage and K/V projection savings. Reusing a KV
   prefix does not eliminate suffix queries attending to that prefix, so it must
   never be reported as equivalent attention-FLOP savings.

## Minimal next implementation

Implement packed varlen FlashAttention on `DEFAULT_ROLLOUT_LENGTHS`, then compare
it to padded SDPA and length bucketing under the same dtype, heads, head dimension,
causal flag, warmups, and repetition count. Write one JSON output per GPU.

The expected outcome is deliberately not specified: varlen should remove padding
work, but small inputs may still favor a simple padded launch because of packing
and scheduling overhead.

## Non-goals

- Claiming a custom kernel beats FlashAttention.
- Calling an expanded PyTorch prefix tensor a production KV cache.
- Reporting “TFLOPS” for an entire attention pipeline without saying what data
  movement, conversion, allocation, and fusion are included.
- Benchmarking causal and non-causal paths against each other.
