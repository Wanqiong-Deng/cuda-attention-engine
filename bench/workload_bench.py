"""Rollout-aware causal-attention benchmark.

This harness intentionally separates measured latency from analytical cost models:
  * padded and length-bucketed causal attention are timed with CUDA events;
  * prefix sharing reports KV/projection savings, not fictitious attention savings.

FlashAttention-2 is optional. Its fixed-length API is compared only on matching
causal shapes; packed varlen attention is the next milestone (see target_design).
"""

import argparse
import json
import math
import platform
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import median

import torch
import torch.nn.functional as F

try:
    from flash_attn import flash_attn_func

    HAS_FLASH_ATTN = True
except ImportError:
    flash_attn_func = None
    HAS_FLASH_ATTN = False


DEFAULT_ROLLOUT_LENGTHS = [512, 1024, 2048, 512, 4096, 1024, 2048, 8192]
QUICK_ROLLOUT_LENGTHS = [128, 256, 512, 128, 512, 256, 384, 512]


def percentile(values, q):
    """Nearest-rank percentile for a non-empty sequence."""
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(len(ordered) * q) - 1))
    return ordered[index]


def benchmark_cuda(fn, warmup, runs):
    """Return CUDA-event samples in milliseconds; synchronization is intentional."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    samples_ms = []
    for _ in range(runs):
        start.record()
        fn()
        stop.record()
        stop.synchronize()
        samples_ms.append(start.elapsed_time(stop))

    return {
        "samples_ms": samples_ms,
        "median_ms": median(samples_ms),
        "p95_ms": percentile(samples_ms, 0.95),
        "min_ms": min(samples_ms),
    }


def make_qkv(batch, seq_len, num_heads, head_dim, dtype):
    shape = (batch, num_heads, seq_len, head_dim)
    return tuple(torch.randn(shape, device="cuda", dtype=dtype) for _ in range(3))


def sdpa(Q, K, V):
    return F.scaled_dot_product_attention(Q, K, V, is_causal=True)


def flash_attention(Q, K, V):
    """FA2 fixed-length API uses [batch, sequence, heads, dimension]."""
    return flash_attn_func(
        Q.transpose(1, 2), K.transpose(1, 2), V.transpose(1, 2), causal=True
    ).transpose(1, 2)


def implementation_results(Q, K, V, warmup, runs):
    """Time every available backend on exactly the same causal tensors."""
    results = {"sdpa": benchmark_cuda(lambda: sdpa(Q, K, V), warmup, runs)}
    if HAS_FLASH_ATTN:
        try:
            results["flash_attention_2"] = benchmark_cuda(
                lambda: flash_attention(Q, K, V), warmup, runs
            )
        except (RuntimeError, NotImplementedError) as error:
            results["flash_attention_2"] = {
                "status": "unavailable_for_shape",
                "error": str(error),
            }
    return results


def uniform_experiment(args, dtype, seq_lens):
    results = []
    for seq_len in seq_lens:
        Q, K, V = make_qkv(args.batch_size, seq_len, args.num_heads, args.head_dim, dtype)
        backends = implementation_results(Q, K, V, args.warmup, args.runs)
        results.append(
            {
                "shape": {
                    "batch": args.batch_size,
                    "seq_len": seq_len,
                    "num_heads": args.num_heads,
                    "head_dim": args.head_dim,
                    "causal": True,
                },
                "backends": backends,
            }
        )
        del Q, K, V
    return results


def variable_length_experiment(args, dtype, lengths):
    """Compare one padded launch with a sequential schedule of exact-length groups."""
    batch = len(lengths)
    maximum = max(lengths)
    groups = Counter(lengths)
    useful_work = sum(length * length for length in lengths)
    padded_work = batch * maximum * maximum

    Q, K, V = make_qkv(batch, maximum, args.num_heads, args.head_dim, dtype)
    padded = implementation_results(Q, K, V, args.warmup, args.runs)
    del Q, K, V

    bucket_tensors = {
        length: make_qkv(count, length, args.num_heads, args.head_dim, dtype)
        for length, count in sorted(groups.items())
    }

    def bucketed_sdpa():
        for bucket_qkv in bucket_tensors.values():
            sdpa(*bucket_qkv)

    bucketed = {"sdpa": benchmark_cuda(bucketed_sdpa, args.warmup, args.runs)}
    if HAS_FLASH_ATTN:
        def bucketed_flash():
            for bucket_qkv in bucket_tensors.values():
                flash_attention(*bucket_qkv)

        try:
            bucketed["flash_attention_2"] = benchmark_cuda(
                bucketed_flash, args.warmup, args.runs
            )
        except (RuntimeError, NotImplementedError) as error:
            bucketed["flash_attention_2"] = {
                "status": "unavailable_for_shape",
                "error": str(error),
            }

    return {
        "trace_lengths": lengths,
        "length_buckets": [{"seq_len": length, "batch": count} for length, count in sorted(groups.items())],
        "padded_shape": {"batch": batch, "seq_len": maximum},
        "attention_work": {
            "useful_relative_units": useful_work,
            "padded_relative_units": padded_work,
            "attention_work_efficiency": useful_work / padded_work,
            "padding_work_fraction": 1 - useful_work / padded_work,
            "note": "Relative causal-attention work derived from shape, not measured speedup.",
        },
        "padded": padded,
        "length_bucketed_sequential": bucketed,
        "methodology_note": "Bucketed timing includes all per-bucket launches; it is not a packed-varlen kernel.",
    }


def prefix_cache_model(batch, prefix_len, suffix_len, num_heads, head_dim, dtype):
    """Account for what prefix KV reuse avoids, without claiming attention savings."""
    bytes_per_element = torch.tensor([], dtype=dtype).element_size()
    kv_elements_per_token = 2 * num_heads * head_dim
    prefix_kv_elements = prefix_len * kv_elements_per_token
    without_cache_elements = batch * prefix_kv_elements
    with_cache_elements = prefix_kv_elements
    avoided_elements = without_cache_elements - with_cache_elements
    # K and V projections each multiply a hidden state by a head-sized output.
    # This tracks only shared-prefix projection work; attention remains necessary.
    projection_relative_flops_avoided = 2 * avoided_elements
    return {
        "batch": batch,
        "prefix_len": prefix_len,
        "suffix_len": suffix_len,
        "dtype_bytes": bytes_per_element,
        "prefix_kv_bytes_without_cache": without_cache_elements * bytes_per_element,
        "prefix_kv_bytes_with_cache": with_cache_elements * bytes_per_element,
        "prefix_kv_bytes_avoided": avoided_elements * bytes_per_element,
        "prefix_kv_storage_reduction": avoided_elements / without_cache_elements,
        "shared_prefix_kv_projection_relative_flops_avoided": projection_relative_flops_avoided,
        "note": "This model excludes attention for suffix queries attending to the prefix. It is not an attention latency result.",
    }


def environment():
    properties = torch.cuda.get_device_properties(0)
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "gpu_compute_capability": f"{properties.major}.{properties.minor}",
        "flash_attention_2_importable": HAS_FLASH_ATTN,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Rollout-aware causal-attention benchmark")
    parser.add_argument("--num-heads", type=int, default=32)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--quick", action="store_true", help="Use reduced shapes for a harness smoke test.")
    parser.add_argument("--output", type=Path, help="Write the full JSON result to this path.")
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA-enabled PyTorch is required: torch.cuda.is_available() is false.")
    if args.warmup < 0 or args.runs < 1:
        raise SystemExit("--warmup must be non-negative and --runs must be at least 1.")

    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    uniform_lengths = [128, 512] if args.quick else [512, 2048, 4096, 8192]
    rollout_lengths = QUICK_ROLLOUT_LENGTHS if args.quick else DEFAULT_ROLLOUT_LENGTHS
    prefix_cases = [(args.batch_size, 512, 1536), (args.batch_size, 2048, 512)]
    if args.quick:
        prefix_cases = [(args.batch_size, 128, 384)]

    report = {
        "environment": environment(),
        "config": {
            "num_heads": args.num_heads,
            "head_dim": args.head_dim,
            "dtype": args.dtype,
            "warmup": args.warmup,
            "runs": args.runs,
            "quick": args.quick,
            "causal": True,
        },
        "uniform_causal_attention": uniform_experiment(args, dtype, uniform_lengths),
        "variable_length_rollout": variable_length_experiment(args, dtype, rollout_lengths),
        "prefix_cache_cost_model": [
            prefix_cache_model(batch, prefix, suffix, args.num_heads, args.head_dim, dtype)
            for batch, prefix, suffix in prefix_cases
        ],
    }

    variable = report["variable_length_rollout"]
    work = variable["attention_work"]
    print(f"GPU: {report['environment']['gpu']} | PyTorch CUDA: {report['environment']['torch_cuda']}")
    print(f"FlashAttention-2 importable: {HAS_FLASH_ATTN}")
    print("\nVariable-length rollout")
    print(f"  trace lengths: {rollout_lengths}")
    print(f"  shape-derived padding work: {work['padding_work_fraction']:.1%}")
    for strategy in ("padded", "length_bucketed_sequential"):
        print(f"  {strategy}:")
        for backend, result in variable[strategy].items():
            if "median_ms" in result:
                print(f"    {backend}: median={result['median_ms']:.3f} ms, p95={result['p95_ms']:.3f} ms")
            else:
                print(f"    {backend}: {result['status']}")

    encoded = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
        print(f"\nWrote report: {args.output}")
    else:
        print("\nNo --output provided; pass one to retain raw samples and environment metadata.")


if __name__ == "__main__":
    main()
