#!/usr/bin/env python3
"""
bench_torch_empty.py -- Benchmark torch.empty() latency.

Compares the default CUDACachingAllocator with the zcg VMMAllocator.

Usage:
  # Default allocator
  python bench_torch_empty.py --output results_default.json

  # zcg allocator
  python bench_torch_empty.py --use-zcg --output results_zcg.json

  # Auto-compare (spawns both as subprocesses)
  python bench_torch_empty.py --compare
"""

import argparse
import json
import os
import subprocess
import sys
import time

import torch

# ====================================================================
# Configuration
# ====================================================================

SIZES = [
    (     1024, "1 KB"),
    (    64 * 1024, "64 KB"),
    (  1024 * 1024, "1 MB"),
    (  8 * 1024 * 1024, "8 MB"),
    ( 16 * 1024 * 1024, "16 MB"),
    ( 64 * 1024 * 1024, "64 MB"),
    (256 * 1024 * 1024, "256 MB"),
    (1024 * 1024 * 1024, "1 GB"),
]

WARMUP_ITERS = 20
BENCH_ITERS  = 200
DTYPE        = torch.float32

# ====================================================================
# Benchmark helpers
# ====================================================================

def bench_empty_cold(size: int, iters: int, device: str = "cuda:0"):
    """Measure allocation time for cold-start (no cache)."""
    latencies = []
    for _ in range(iters):
        # Empty cache to force cold allocation.
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end   = torch.cuda.Event(enable_timing=True)

        start.record()
        t = torch.empty(size // DTYPE.itemsize, dtype=DTYPE, device=device)
        end.record()

        torch.cuda.synchronize()
        latencies.append(start.elapsed_time(end) * 1000)  # ms -> us

        del t

    return latencies


def bench_empty_warm(size: int, iters: int, device: str = "cuda:0"):
    """Measure allocation time for warm-state (cache populated)."""
    # Prime the cache: allocate and free once to populate free lists.
    t = torch.empty(size // DTYPE.itemsize, dtype=DTYPE, device=device)
    del t
    torch.cuda.synchronize()

    latencies = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end   = torch.cuda.Event(enable_timing=True)

        start.record()
        t = torch.empty(size // DTYPE.itemsize, dtype=DTYPE, device=device)
        end.record()

        torch.cuda.synchronize()
        latencies.append(start.elapsed_time(end) * 1000)  # ms -> us

        del t

    return latencies


def compute_stats(latencies):
    import statistics
    if not latencies:
        return {}
    latencies_sorted = sorted(latencies)
    n = len(latencies_sorted)
    return {
        "min":    round(latencies_sorted[0], 2),
        "max":    round(latencies_sorted[-1], 2),
        "avg":    round(statistics.mean(latencies_sorted), 2),
        "median": round(statistics.median(latencies_sorted), 2),
        "p99":    round(latencies_sorted[min(n - 1, int(n * 0.99))], 2),
    }


# ====================================================================
# Main benchmark runner
# ====================================================================

def run_benchmarks(device: str = "cuda:0"):
    """Run all benchmarks, return results as a dict."""
    results = {
        "allocator": "unknown",
        "device_name": torch.cuda.get_device_name(0),
        "cuda_version": torch.version.cuda,
        "pytorch_version": torch.__version__,
        "warmup_iters": WARMUP_ITERS,
        "bench_iters": BENCH_ITERS,
        "benchmarks": [],
    }

    # Global warmup.
    for _ in range(WARMUP_ITERS):
        t = torch.empty(1024, device=device)
        del t
    torch.cuda.synchronize()

    for size, label in SIZES:
        # Skip sizes that exceed device memory (rough check).
        free_mem, _ = torch.cuda.mem_get_info()
        if size > free_mem * 0.8:
            print(f"  Skipping {label} (not enough memory)")
            continue

        print(f"  Benchmarking {label}...")

        cold = bench_empty_cold(size, BENCH_ITERS, device)
        warm = bench_empty_warm(size, BENCH_ITERS, device)

        entry = {
            "size_bytes": size,
            "label": label,
            "cold_us": compute_stats(cold),
            "warm_us": compute_stats(warm),
        }
        results["benchmarks"].append(entry)

    return results


# ====================================================================
# Comparison mode
# ====================================================================

def run_comparison():
    """Spawn two subprocesses (default and zcg) and compare results."""
    script = os.path.abspath(__file__)

    print("=" * 70)
    print("Running with DEFAULT allocator...")
    print("=" * 70)
    r1 = subprocess.run(
        [sys.executable, script, "--output", "/tmp/_bench_default.json"],
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
        [sys.executable, script, "--use-zcg", "--output", "/tmp/_bench_zcg.json"],
        capture_output=True, text=True,
    )
    print(r2.stdout)
    if r2.returncode != 0:
        print("ERROR:", r2.stderr)
        return

    # Load results.
    with open("/tmp/_bench_default.json") as f:
        default = json.load(f)
    with open("/tmp/_bench_zcg.json") as f:
        zcg_res = json.load(f)

    # Print comparison table.
    print("\n" + "=" * 90)
    print("  torch.empty() Latency Comparison (median, us)")
    print(f"  Device: {default['device_name']}")
    print(f"  CUDA: {default['cuda_version']}, PyTorch: {default['pytorch_version']}")
    print("=" * 90)

    header = f"  {'Size':>8s}  │  {'Default(cold)':>14s}  {'Default(warm)':>14s}"
    header += f"  │  {'zcg(cold)':>14s}  {'zcg(warm)':>14s}  │  {'warm ratio':>10s}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    d_map = {b["label"]: b for b in default["benchmarks"]}
    z_map = {b["label"]: b for b in zcg_res["benchmarks"]}

    for _, label in SIZES:
        if label not in d_map or label not in z_map:
            continue
        d = d_map[label]
        z = z_map[label]

        dc = d["cold_us"]["median"]
        dw = d["warm_us"]["median"]
        zc = z["cold_us"]["median"]
        zw = z["warm_us"]["median"]
        ratio = zw / dw if dw > 0 else float('inf')

        print(f"  {label:>8s}  │  {dc:>12.2f} us  {dw:>12.2f} us"
              f"  │  {zc:>12.2f} us  {zw:>12.2f} us  │  {ratio:>9.2f}x")

    print()


# ====================================================================
# Entry point
# ====================================================================

def main():
    parser = argparse.ArgumentParser(description="Benchmark torch.empty()")
    parser.add_argument("--use-zcg", action="store_true",
                        help="Import zcg to use the VMM allocator")
    parser.add_argument("--output", type=str, default=None,
                        help="Save results to JSON file")
    parser.add_argument("--compare", action="store_true",
                        help="Run both allocators and print comparison")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    if args.compare:
        run_comparison()
        return

    # Import zcg if requested (must happen before any CUDA allocation).
    if args.use_zcg:
        import zcg  # noqa: F401
        allocator_name = "zcg"
        print(f"Using zcg VMMAllocator")
    else:
        allocator_name = "default"
        print(f"Using default CUDACachingAllocator")

    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA {torch.version.cuda}, PyTorch {torch.__version__}")
    print(f"Iterations: {BENCH_ITERS} (warmup: {WARMUP_ITERS})")
    print()

    results = run_benchmarks(args.device)
    results["allocator"] = allocator_name

    # Print summary table.
    print(f"\n{'Size':>8s}  {'cold median(us)':>16s}  {'warm median(us)':>16s}")
    print("-" * 44)
    for b in results["benchmarks"]:
        print(f"{b['label']:>8s}  {b['cold_us']['median']:>14.2f}  "
              f"{b['warm_us']['median']:>14.2f}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
