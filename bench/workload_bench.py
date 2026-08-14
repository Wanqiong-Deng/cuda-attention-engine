"""
Phase 5.5: RL Rollout Inference Workload Benchmark

Compare attention implementations across RL-representative workload patterns.
Goal: Find which implementation works best under which conditions,
      and explain WHY using profiling data.

Implementations compared:
  1. PyTorch SDPA (torch.nn.functional.scaled_dot_product_attention)
  2. FlashAttention-2 (pip install flash-attn)

Workload patterns:
  1. Uniform: all sequences same length (baseline)
  2. High-variance: RL rollout lengths vary wildly (512 to 8192)
  3. Shared-prefix: all sequences share a long prefix (RL system prompt)

Usage:
  pip install torch flash-attn
  python bench/workload_bench.py

  With profiling:
  nsys profile --trace=cuda python bench/workload_bench.py
"""

import torch
import torch.nn.functional as F
import time
import argparse

# Try importing flash_attn
try:
    from flash_attn import flash_attn_func
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False
    print("[WARNING] flash-attn not installed. Install with: pip install flash-attn")
    print("         Only PyTorch SDPA will be benchmarked.\n")


def benchmark_fn(fn, warmup=5, runs=20):
    """Benchmark a function with CUDA event timing."""
    # Warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    # Timed runs
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(runs):
        fn()
    end.record()
    torch.cuda.synchronize()

    avg_ms = start.elapsed_time(end) / runs
    return avg_ms


def run_sdpa(Q, K, V, is_causal=True):
    """PyTorch's built-in scaled dot-product attention."""
    return F.scaled_dot_product_attention(Q, K, V, is_causal=is_causal)


def run_flash_attn(Q, K, V, causal=True):
    """FlashAttention-2. Expects (batch, seq_len, num_heads, head_dim)."""
    # flash_attn expects (B, S, H, D) not (B, H, S, D)
    Q_t = Q.transpose(1, 2)
    K_t = K.transpose(1, 2)
    V_t = V.transpose(1, 2)
    out = flash_attn_func(Q_t, K_t, V_t, causal=causal)
    return out.transpose(1, 2)


def workload_uniform(batch_size, seq_len, num_heads, head_dim, dtype):
    """All sequences same length."""
    Q = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda', dtype=dtype)
    K = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda', dtype=dtype)
    V = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda', dtype=dtype)
    return Q, K, V


def workload_high_variance(num_heads, head_dim, dtype):
    """
    RL rollout: batch of 8 sequences with wildly different lengths.
    Simulates: some rollouts finish quickly, some go very long.

    Since standard attention requires same seq_len in a batch,
    we pad to max and note the waste.
    """
    # Realistic RL rollout lengths
    seq_lens = [512, 1024, 2048, 512, 4096, 1024, 2048, 8192]
    max_len = max(seq_lens)
    batch_size = len(seq_lens)

    # Create padded tensors
    Q = torch.zeros(batch_size, num_heads, max_len, head_dim, device='cuda', dtype=dtype)
    K = torch.zeros(batch_size, num_heads, max_len, head_dim, device='cuda', dtype=dtype)
    V = torch.zeros(batch_size, num_heads, max_len, head_dim, device='cuda', dtype=dtype)

    for i, sl in enumerate(seq_lens):
        Q[i, :, :sl, :] = torch.randn(num_heads, sl, head_dim, device='cuda', dtype=dtype)
        K[i, :, :sl, :] = torch.randn(num_heads, sl, head_dim, device='cuda', dtype=dtype)
        V[i, :, :sl, :] = torch.randn(num_heads, sl, head_dim, device='cuda', dtype=dtype)

    total_tokens = sum(seq_lens)
    padded_tokens = batch_size * max_len
    waste_pct = (1 - total_tokens / padded_tokens) * 100

    return Q, K, V, seq_lens, waste_pct


def workload_shared_prefix(batch_size, prefix_len, unique_len, num_heads, head_dim, dtype):
    """
    All rollouts share the same system prompt (prefix),
    but have different unique suffixes.

    Without prefix caching: prefix KV is recomputed for each batch item.
    This measures the "naive" cost.
    """
    total_len = prefix_len + unique_len

    # Shared prefix (same for all batch items)
    prefix_K = torch.randn(1, num_heads, prefix_len, head_dim, device='cuda', dtype=dtype)
    prefix_V = torch.randn(1, num_heads, prefix_len, head_dim, device='cuda', dtype=dtype)

    # Expand prefix to full batch (simulates recomputation)
    K_prefix = prefix_K.expand(batch_size, -1, -1, -1)
    V_prefix = prefix_V.expand(batch_size, -1, -1, -1)

    # Unique suffixes
    K_unique = torch.randn(batch_size, num_heads, unique_len, head_dim, device='cuda', dtype=dtype)
    V_unique = torch.randn(batch_size, num_heads, unique_len, head_dim, device='cuda', dtype=dtype)

    # Full K, V = [prefix | unique]
    K_full = torch.cat([K_prefix, K_unique], dim=2)
    V_full = torch.cat([V_prefix, V_unique], dim=2)

    # Q attends to full sequence
    Q = torch.randn(batch_size, num_heads, total_len, head_dim, device='cuda', dtype=dtype)

    redundant_compute_pct = (prefix_len / total_len) * 100

    return Q, K_full, V_full, redundant_compute_pct


def print_header(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def print_result(name, ms, extra=""):
    tflops_str = ""
    print(f"  {name:<30} {ms:>8.3f} ms  {extra}")


def main():
    parser = argparse.ArgumentParser(description="RL Workload Attention Benchmark")
    parser.add_argument("--num-heads", type=int, default=32)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16")
    args = parser.parse_args()

    num_heads = args.num_heads
    head_dim = args.head_dim
    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16

    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Config: num_heads={num_heads}, head_dim={head_dim}, dtype={args.dtype}")
    print(f"FlashAttention available: {HAS_FLASH_ATTN}")

    # ============================================================
    # Workload 1: Uniform (baseline)
    # ============================================================
    print_header("Workload 1: Uniform Seq Length (Baseline)")

    for seq_len in [512, 2048, 4096, 8192]:
        batch_size = 8
        Q, K, V = workload_uniform(batch_size, seq_len, num_heads, head_dim, dtype)

        print(f"\n  seq_len={seq_len}, batch={batch_size}")

        ms = benchmark_fn(lambda: run_sdpa(Q, K, V))
        print_result("PyTorch SDPA", ms)

        if HAS_FLASH_ATTN:
            ms = benchmark_fn(lambda: run_flash_attn(Q, K, V))
            print_result("FlashAttention-2", ms)

    # ============================================================
    # Workload 2: High Variance (RL rollout pattern)
    # ============================================================
    print_header("Workload 2: High-Variance Seq Length (RL Rollout)")

    Q, K, V, seq_lens, waste_pct = workload_high_variance(num_heads, head_dim, dtype)
    print(f"\n  Seq lengths: {seq_lens}")
    print(f"  Max (padded to): {max(seq_lens)}")
    print(f"  Padding waste: {waste_pct:.1f}% of compute is on PAD tokens")
    print(f"  (In production, ragged/varlen layout eliminates this waste)")

    ms = benchmark_fn(lambda: run_sdpa(Q, K, V))
    print_result("PyTorch SDPA (padded)", ms)

    if HAS_FLASH_ATTN:
        ms = benchmark_fn(lambda: run_flash_attn(Q, K, V))
        print_result("FlashAttention-2 (padded)", ms)

    # Compare: what if all sequences were just max_len? (uniform)
    Q_u, K_u, V_u = workload_uniform(8, max(seq_lens), num_heads, head_dim, dtype)
    ms_uniform = benchmark_fn(lambda: run_sdpa(Q_u, K_u, V_u))
    print_result("PyTorch SDPA (uniform max)", ms_uniform, "(no variance, same total)")
    print(f"\n  Insight: padded high-variance vs uniform max → similar time,")
    print(f"  meaning {waste_pct:.0f}% of compute is wasted on padding.")

    # ============================================================
    # Workload 3: Shared Prefix (RL system prompt)
    # ============================================================
    print_header("Workload 3: Shared Prefix (RL System Prompt)")

    batch_size = 8
    for prefix_len, unique_len in [(512, 1536), (2048, 2048), (2048, 512)]:
        Q, K, V, redundant_pct = workload_shared_prefix(
            batch_size, prefix_len, unique_len, num_heads, head_dim, dtype
        )
        total_len = prefix_len + unique_len

        print(f"\n  prefix={prefix_len}, unique={unique_len}, total={total_len}")
        print(f"  Prefix compute redundancy: {redundant_pct:.0f}%")
        print(f"  (With prefix caching, this {redundant_pct:.0f}% would be free)")

        ms = benchmark_fn(lambda: run_sdpa(Q, K, V))
        print_result("PyTorch SDPA", ms)

        if HAS_FLASH_ATTN:
            ms = benchmark_fn(lambda: run_flash_attn(Q, K, V))
            print_result("FlashAttention-2", ms)

    # ============================================================
    # Workload 4: Scaling — how does latency grow with seq_len?
    # ============================================================
    print_header("Workload 4: Latency Scaling (seq_len growth during rollout)")
    print("  Simulates context growing as Agent takes more steps")

    batch_size = 1
    results_sdpa = []
    results_fa = []

    for seq_len in [256, 512, 1024, 2048, 4096, 8192]:
        Q, K, V = workload_uniform(batch_size, seq_len, num_heads, head_dim, dtype)

        ms = benchmark_fn(lambda: run_sdpa(Q, K, V))
        results_sdpa.append((seq_len, ms))

        if HAS_FLASH_ATTN:
            ms = benchmark_fn(lambda: run_flash_attn(Q, K, V))
            results_fa.append((seq_len, ms))

    print(f"\n  {'seq_len':<10} {'SDPA (ms)':<12} {'FA2 (ms)':<12} {'SDPA scaling':<14}")
    base_sdpa = results_sdpa[0][1]
    for i, (sl, ms) in enumerate(results_sdpa):
        fa_ms = f"{results_fa[i][1]:.3f}" if results_fa else "N/A"
        scaling = f"{ms/base_sdpa:.1f}x"
        print(f"  {sl:<10} {ms:<12.3f} {fa_ms:<12} {scaling:<14}")

    print(f"\n  Insight: O(N^2) attention → 2x seq_len ≈ 4x latency")
    print(f"  Long RL trajectories hit this quadratic wall hard.")

    # ============================================================
    # Summary
    # ============================================================
    print_header("Summary & Profiling Next Steps")
    print("""
  What we measured:
    1. Uniform baseline — establishes per-implementation speed
    2. High-variance — quantifies padding waste in RL batches
    3. Shared prefix — quantifies redundant KV compute without caching
    4. Scaling curve — shows quadratic cost of growing context

  Next: run with nsys/ncu to explain WHY:
    nsys profile --trace=cuda python bench/workload_bench.py
    → Look for: kernel launch gaps, idle time, memory transfer overhead

    ncu --kernel-name regex:".*attention.*" python bench/workload_bench.py
    → Look for: memory throughput, compute throughput, occupancy
    → Compare numbers across workload patterns

  Key questions to answer with profiling:
    - Does padding waste show up as low compute throughput? (wasted FLOPS on zeros)
    - Is there a seq_len threshold where FA2 starts beating SDPA?
    - In shared-prefix workload, does L2 cache help at all? (same prefix data reaccessed)
""")


if __name__ == "__main__":
    main()
