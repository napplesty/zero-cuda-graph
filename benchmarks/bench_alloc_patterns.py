#!/usr/bin/env python3
"""
bench_alloc_patterns.py -- Benchmark realistic allocation patterns.

Tests various allocation/free sequences to compare the default
CUDACachingAllocator with the zcg VMMAllocator under realistic
workload patterns.

Usage:
  # Default allocator
  python bench_alloc_patterns.py

  # zcg allocator
  python bench_alloc_patterns.py --use-zcg

  # Auto-compare
  python bench_alloc_patterns.py --compare
"""

import argparse
import json
import os
import random
import subprocess
import sys
import time

import torch

# ====================================================================
# Configuration
# ====================================================================

NUM_CYCLES   = 1000
WARMUP_CYCLES = 50

SMALL_SIZES = [
    1 * 1024,           # 1 KB
    4 * 1024,           # 4 KB
    16 * 1024,          # 16 KB
    64 * 1024,          # 64 KB
    256 * 1024,         # 256 KB
    1 * 1024 * 1024,    # 1 MB
    4 * 1024 * 1024,    # 4 MB
    8 * 1024 * 1024,    # 8 MB
]

LARGE_SIZES = [
    16 * 1024 * 1024,   # 16 MB
    32 * 1024 * 1024,   # 32 MB
    64 * 1024 * 1024,   # 64 MB
    128 * 1024 * 1024,  # 128 MB
    256 * 1024 * 1024,  # 256 MB
]

MIXED_SIZES = SMALL_SIZES + LARGE_SIZES

DTYPE = torch.float32

# ====================================================================
# Pattern benchmarks
# ====================================================================

def pattern_alloc_free_cycle(sizes, num_cycles, label=""):
    """Allocate-then-free pattern: allocate all, then free all, repeat."""
    # Warmup.
    for _ in range(WARMUP_CYCLES):
        tensors = []
        for sz in sizes:
            n = sz // DTYPE.itemsize
            tensors.append(torch.empty(n, dtype=DTYPE, device="cuda"))
        del tensors
    torch.cuda.synchronize()

    # Timed run.
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    for _ in range(num_cycles):
        tensors = []
        for sz in sizes:
            n = sz // DTYPE.itemsize
            tensors.append(torch.empty(n, dtype=DTYPE, device="cuda"))
        del tensors

    torch.cuda.synchronize()
    t1 = time.perf_counter()

    total_ms = (t1 - t0) * 1000
    per_cycle_us = total_ms * 1000 / num_cycles
    return {
        "pattern": f"alloc_free_cycle ({label})",
        "num_cycles": num_cycles,
        "tensors_per_cycle": len(sizes),
        "total_ms": round(total_ms, 2),
        "per_cycle_us": round(per_cycle_us, 2),
    }


def pattern_random_alloc_free(sizes, num_ops, label=""):
    """Random interleaved alloc/free: 70% alloc, 30% free."""
    rng = random.Random(42)
    live_tensors = []

    # Warmup.
    for _ in range(WARMUP_CYCLES):
        t = torch.empty(1024, dtype=DTYPE, device="cuda")
        del t
    torch.cuda.synchronize()

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    alloc_count = 0
    free_count = 0

    for _ in range(num_ops):
        if len(live_tensors) == 0 or (rng.random() < 0.7 and len(live_tensors) < 200):
            # Allocate.
            sz = rng.choice(sizes)
            n = sz // DTYPE.itemsize
            live_tensors.append(torch.empty(n, dtype=DTYPE, device="cuda"))
            alloc_count += 1
        else:
            # Free a random tensor.
            idx = rng.randint(0, len(live_tensors) - 1)
            live_tensors.pop(idx)
            free_count += 1

    # Free remaining.
    del live_tensors

    torch.cuda.synchronize()
    t1 = time.perf_counter()

    total_ms = (t1 - t0) * 1000
    return {
        "pattern": f"random_interleaved ({label})",
        "num_ops": num_ops,
        "allocs": alloc_count,
        "frees": free_count,
        "total_ms": round(total_ms, 2),
        "per_op_us": round(total_ms * 1000 / num_ops, 2),
    }


def pattern_peak_then_steady(peak_size_mb, steady_size_mb, num_steady_cycles):
    """Allocate a large peak, free it, then do many steady-state cycles."""
    # Warmup.
    t = torch.empty(1024, dtype=DTYPE, device="cuda")
    del t
    torch.cuda.synchronize()

    peak_n = (peak_size_mb * 1024 * 1024) // DTYPE.itemsize
    steady_n = (steady_size_mb * 1024 * 1024) // DTYPE.itemsize

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    # Phase 1: peak allocation.
    peak_tensor = torch.empty(peak_n, dtype=DTYPE, device="cuda")
    del peak_tensor

    # Phase 2: steady-state small allocations.
    for _ in range(num_steady_cycles):
        t = torch.empty(steady_n, dtype=DTYPE, device="cuda")
        del t

    torch.cuda.synchronize()
    t1 = time.perf_counter()

    total_ms = (t1 - t0) * 1000
    return {
        "pattern": f"peak({peak_size_mb}MB)_then_steady({steady_size_mb}MB)",
        "num_steady_cycles": num_steady_cycles,
        "total_ms": round(total_ms, 2),
    }


def pattern_simulate_inference(batch_sizes, hidden_dim, num_layers, num_batches):
    """
    Simulate a simplified transformer inference pattern:
    For each batch, allocate activations for each layer, then free them.
    """
    # Warmup.
    for _ in range(5):
        t = torch.empty(1024, dtype=DTYPE, device="cuda")
        del t
    torch.cuda.synchronize()

    rng = random.Random(123)

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    for i in range(num_batches):
        bs = rng.choice(batch_sizes)
        layer_tensors = []

        for layer in range(num_layers):
            # Simulate: attention QKV, attention output, FFN intermediate, FFN output
            qkv = torch.empty(bs, 3, hidden_dim, dtype=DTYPE, device="cuda")
            attn_out = torch.empty(bs, hidden_dim, dtype=DTYPE, device="cuda")
            ffn_mid = torch.empty(bs, hidden_dim * 4, dtype=DTYPE, device="cuda")
            ffn_out = torch.empty(bs, hidden_dim, dtype=DTYPE, device="cuda")
            layer_tensors.extend([qkv, attn_out, ffn_mid, ffn_out])

        # Free all layer tensors (simulates end of forward pass).
        del layer_tensors

    torch.cuda.synchronize()
    t1 = time.perf_counter()

    total_ms = (t1 - t0) * 1000
    return {
        "pattern": "simulated_inference",
        "num_batches": num_batches,
        "num_layers": num_layers,
        "hidden_dim": hidden_dim,
        "batch_sizes": batch_sizes,
        "total_ms": round(total_ms, 2),
        "per_batch_us": round(total_ms * 1000 / num_batches, 2),
    }


# ====================================================================
# Memory stats helper
# ====================================================================

def get_memory_info(use_zcg):
    """Get memory stats from the appropriate allocator."""
    info = {}
    if use_zcg:
        import zcg
        stats = zcg.memory_stats()
        info["total_physical_mb"] = stats["total_physical"] / (1024 * 1024)
        info["total_cached_mb"] = stats["total_physical_cached"] / (1024 * 1024)
        info["purge_count"] = stats.get("purge_count", 0)
        info["total_purged_mb"] = stats.get("total_purged_bytes", 0) / (1024 * 1024)
        info["num_alloc_retries"] = stats["num_alloc_retries"]
        info["num_ooms"] = stats["num_ooms"]
    else:
        stats = torch.cuda.memory_stats()
        info["total_physical_mb"] = stats.get("reserved_bytes.all.current", 0) / (1024 * 1024)
        info["total_cached_mb"] = stats.get("inactive_split_bytes.all.current", 0) / (1024 * 1024)
        info["num_alloc_retries"] = stats.get("num_alloc_retries", 0)
        info["num_ooms"] = stats.get("num_ooms", 0)
    return info


# ====================================================================
# Main benchmark runner
# ====================================================================

def run_all_benchmarks(use_zcg=False):
    results = {
        "allocator": "zcg" if use_zcg else "default",
        "device_name": torch.cuda.get_device_name(0),
        "cuda_version": torch.version.cuda,
        "pytorch_version": torch.__version__,
        "benchmarks": [],
    }

    print("\n--- Pattern 1: Alloc-Free Cycle (small tensors) ---")
    r = pattern_alloc_free_cycle(SMALL_SIZES, NUM_CYCLES, "small")
    print(f"    {r['total_ms']:.1f} ms total, {r['per_cycle_us']:.1f} us/cycle")
    results["benchmarks"].append(r)

    print("\n--- Pattern 2: Alloc-Free Cycle (large tensors) ---")
    r = pattern_alloc_free_cycle(LARGE_SIZES, NUM_CYCLES, "large")
    print(f"    {r['total_ms']:.1f} ms total, {r['per_cycle_us']:.1f} us/cycle")
    results["benchmarks"].append(r)

    print("\n--- Pattern 3: Alloc-Free Cycle (mixed) ---")
    r = pattern_alloc_free_cycle(MIXED_SIZES, NUM_CYCLES, "mixed")
    print(f"    {r['total_ms']:.1f} ms total, {r['per_cycle_us']:.1f} us/cycle")
    results["benchmarks"].append(r)

    print("\n--- Pattern 4: Random Interleaved (small) ---")
    r = pattern_random_alloc_free(SMALL_SIZES, NUM_CYCLES * 10, "small")
    print(f"    {r['total_ms']:.1f} ms total, {r['per_op_us']:.1f} us/op")
    results["benchmarks"].append(r)

    print("\n--- Pattern 5: Random Interleaved (mixed) ---")
    r = pattern_random_alloc_free(MIXED_SIZES, NUM_CYCLES * 10, "mixed")
    print(f"    {r['total_ms']:.1f} ms total, {r['per_op_us']:.1f} us/op")
    results["benchmarks"].append(r)

    print("\n--- Pattern 6: Peak then Steady ---")
    r = pattern_peak_then_steady(
        peak_size_mb=512, steady_size_mb=8, num_steady_cycles=NUM_CYCLES,
    )
    print(f"    {r['total_ms']:.1f} ms total")
    results["benchmarks"].append(r)

    print("\n--- Pattern 7: Simulated Inference ---")
    r = pattern_simulate_inference(
        batch_sizes=[1, 2, 4, 8, 16],
        hidden_dim=4096,
        num_layers=32,
        num_batches=100,
    )
    print(f"    {r['total_ms']:.1f} ms total, {r['per_batch_us']:.1f} us/batch")
    results["benchmarks"].append(r)

    # Collect memory stats.
    print("\n--- Memory Stats ---")
    mem = get_memory_info(use_zcg)
    results["memory_stats"] = mem
    for k, v in mem.items():
        print(f"    {k}: {v}")

    return results


# ====================================================================
# Comparison mode
# ====================================================================

def run_comparison():
    script = os.path.abspath(__file__)

    print("=" * 70)
    print("Running with DEFAULT allocator...")
    print("=" * 70)
    r1 = subprocess.run(
        [sys.executable, script, "--output", "/tmp/_bench_patterns_default.json"],
        capture_output=True, text=True,
    )
    print(r1.stdout)
    if r1.returncode != 0:
        print("ERROR:", r1.stderr)
        return

    print("=" * 70)
    print("Running with ZCG allocator...")
    print("=" * 70)
    r2 = subprocess.run(
        [sys.executable, script, "--use-zcg",
         "--output", "/tmp/_bench_patterns_zcg.json"],
        capture_output=True, text=True,
    )
    print(r2.stdout)
    if r2.returncode != 0:
        print("ERROR:", r2.stderr)
        return

    # Load and compare.
    with open("/tmp/_bench_patterns_default.json") as f:
        default = json.load(f)
    with open("/tmp/_bench_patterns_zcg.json") as f:
        zcg_res = json.load(f)

    print("\n" + "=" * 80)
    print("  Allocation Pattern Benchmark Comparison")
    print(f"  Device: {default['device_name']}")
    print("=" * 80)

    print(f"\n  {'Pattern':<45s}  {'Default(ms)':>12s}  {'zcg(ms)':>10s}  {'Ratio':>8s}")
    print("  " + "-" * 78)

    for d, z in zip(default["benchmarks"], zcg_res["benchmarks"]):
        d_ms = d["total_ms"]
        z_ms = z["total_ms"]
        ratio = z_ms / d_ms if d_ms > 0 else float('inf')
        name = d["pattern"][:44]
        print(f"  {name:<45s}  {d_ms:>10.1f}  {z_ms:>10.1f}  {ratio:>7.2f}x")

    print()


# ====================================================================
# Entry point
# ====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark allocation patterns")
    parser.add_argument("--use-zcg", action="store_true",
                        help="Use zcg VMMAllocator")
    parser.add_argument("--output", type=str, default=None,
                        help="Save results to JSON")
    parser.add_argument("--compare", action="store_true",
                        help="Run both allocators and compare")
    args = parser.parse_args()

    if args.compare:
        run_comparison()
        return

    if args.use_zcg:
        import zcg  # noqa: F401
        print("Allocator: zcg VMMAllocator")
    else:
        print("Allocator: default CUDACachingAllocator")

    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA {torch.version.cuda}, PyTorch {torch.__version__}")

    results = run_all_benchmarks(use_zcg=args.use_zcg)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
