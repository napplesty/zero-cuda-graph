# zero-cuda-graph (zcg)

A VMM-based PyTorch CUDA allocator that eliminates static GPU memory reservation for CUDA Graphs.

## The Problem

CUDA Graphs require all tensor memory addresses to remain constant during replay. PyTorch's default solution — pool isolation — dedicates a separate memory pool for each graph, keeping that memory resident throughout the graph's lifetime. With 10 graph variants at 500MB each, that's 5GB permanently reserved, even when idle.

## The Solution

`zcg` replaces PyTorch's `CUDACachingAllocator` with a VMM (CUDA Virtual Memory Management) based allocator. VMM decouples virtual addresses from physical memory:

- **Capture time**: reserve virtual address space (cheap, no physical memory)
- **Replay time**: map/remap physical memory on-demand without changing virtual addresses
- **Idle time**: unmap physical blocks and return them to a shared pool

Result: CUDA Graph replay works because virtual addresses are stable, but physical memory is freely shared across graph instances.

## Architecture

```
┌─ Python Layer (zcg/) ────────────────────────────┐
│  __init__.py      – import-time allocator swap    │
│  config.py        – runtime configuration         │
│  memory.py        – stats & diagnostics           │
│  ops.py           – pointer manipulation API      │
└──────────────────────┬───────────────────────────-┘
                       │ pybind11
┌──────────────────────▼───────────────────────────-┐
│  C++ Extension (zcg_c)                             │
│                                                    │
│  VMMAllocator (c10::cuda::CUDAAllocator)           │
│  ├─ SmallAllocator   – sub-alloc in 32GB VA range  │
│  ├─ LargeAllocator   – greedy block packing        │
│  └─ PhysicalMemoryManager – block pool lifecycle   │
│                                                    │
│  Substitution        – data_ptr swap & VMM remap   │
│  CUDADriverAPI       – dlopen'd VMM API wrapper    │
└──────────────────────┬───────────────────────────-┘
                       │ cuMemXxx()
                 CUDA Driver (VMM)
```

## Key Features

- **Zero code changes** — `import zcg` before any CUDA op; all `torch.empty()`, autograd allocations use VMM transparently
- **Dual-channel allocation** — SmallAllocator (< 16MB) sub-allocates from pre-reserved 32GB VA with 2MB physical blocks; LargeAllocator (>= 16MB) uses greedy packing from 5 size classes (16/32/64/128/256 MB)
- **Physical block caching** — freed blocks stay in free-lists for reuse, avoiding costly `cuMemCreate`/`cuMemRelease` round-trips
- **3-tier OOM recovery** — selective purge (largest blocks first) → full trim → error
- **Stream-aware deferred free** — event-based recycling respects `recordStream()` semantics
- **CUDA Graph pool isolation** — full `beginAllocateToPool` / `endAllocateToPool` / `releasePool` support
- **VMM remap** — swap physical memory behind a virtual address without changing tensor pointers
- **Multi-GPU peer access** — cross-device visibility via `CUmemSetAccess`
- **Comprehensive diagnostics** — `memory_stats()`, `memory_summary()`, peak/accumulated counters, OOM/purge tracking

## Requirements

- Python >= 3.8
- PyTorch >= 2.5.0
- CUDA >= 10.2 (VMM API support)
- NVIDIA GPU

## Installation

```bash
pip install -e .
```

## Quick Start

```python
import zcg          # Must be before any CUDA operation
import torch

# All CUDA allocations now use VMMAllocator
x = torch.randn(1024, 1024, device="cuda")
y = torch.randn(1024, 1024, device="cuda")
z = x @ y

# Check memory usage
print(zcg.memory_summary())

# CUDA Graph usage — physical memory is no longer permanently reserved
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    out = model(input)
graph.replay()  # VA unchanged, physical memory remapped
```

## API Reference

### `zcg.configure(*, threshold=None)`

Set the routing threshold (bytes) between small and large allocator paths. Call after `import zcg` but before any CUDA allocation.

```python
zcg.configure(threshold=32 * 1024 * 1024)  # 32 MB
```

### `zcg.memory_stats(device=0) -> dict`

Returns detailed memory statistics including per-pool (small/large) current, peak, accumulated bytes, physical memory counters, OOM/purge counts.

### `zcg.memory_summary(device=0) -> str`

Human-readable memory usage summary.

### `zcg.empty_cache()`

Release all cached (unused) physical memory back to the driver.

### `zcg.reset_accumulated_stats(device=0)` / `zcg.reset_peak_stats(device=0)`

Reset lifetime or peak statistics for profiling windows.

### `zcg.enable_peer_access(dev, dev_to_access)`

Enable cross-device memory access via both CUDA P2P and VMM access descriptors.

### `zcg.replace_data_ptr(tensor, new_ptr)`

Hot-swap the underlying device pointer of a tensor without changing its metadata.

### `zcg.vmm_remap(va, size, old_handles, new_handles, new_sizes, device=0, peer_devices=None)`

Remap physical memory behind a virtual address — the core primitive for CUDA Graph memory management.

## Performance

### VMM Raw API Latency (H100 80GB HBM3)

| Operation | 2 MB | 16 MB | 64 MB | 256 MB |
|-----------|------|-------|-------|--------|
| `cuMemAddressReserve` | 0.98 us | 1.01 us | 0.96 us | 0.95 us |
| `cuMemCreate` | 9.75 us | 11.96 us | 10.55 us | 14.70 us |
| `cuMemMap` | 0.43 us | 0.44 us | 0.45 us | 0.47 us |
| `cuMemSetAccess` | 77.00 us | 77.16 us | 77.48 us | 97.61 us |
| `cuMemUnmap` | 36.30 us | 36.50 us | 36.23 us | 36.69 us |
| `cuMemRelease` | 35.85 us | 38.89 us | 49.41 us | 90.83 us |
| **Full chain** | **88.77 us** | **88.39 us** | **156.51 us** | **629.02 us** |

### Key Optimization Insights

- **`cuMemSetAccess` dominates** (80%+ of full chain) — minimize call count
- **Block caching** eliminates `cuMemCreate`/`cuMemRelease` round-trips (46-106 us saved per hit)
- **Large block packing** reduces `cuMemSetAccess` calls: 256MB as 1 block = 98 us vs. 128x 2MB blocks = 9856 us (**100x difference**)
- **SmallAllocator warm path** is pure CPU (zero driver calls) — free-list lookup only
- **`cuMemMap` is nearly free** (0.43 us) but `cuMemUnmap` is expensive (36 us) — map eagerly, unmap lazily

## Benchmarks

```bash
# Allocator latency comparison (cold & warm paths)
python benchmarks/bench_torch_empty.py --compare

# Realistic workload patterns
python benchmarks/bench_alloc_patterns.py --compare

# Raw VMM API latency (standalone C++)
# See benchmarks/bench_vmm_raw_api.cc
```

## Project Structure

```
zero-cuda-graph/
├── zcg/                        # Python package
│   ├── __init__.py             # Auto-registers VMMAllocator on import
│   ├── config.py               # Runtime configuration
│   ├── memory.py               # Memory diagnostics API
│   └── ops.py                  # Pointer manipulation API
├── include/                    # C++ headers
│   ├── common.h                # Constants, alignment helpers
│   ├── cuda_driver_api.h       # VMM API wrapper (dlopen)
│   ├── physical_memory.h       # Physical block pool
│   ├── small_allocator.h       # Sub-allocator (< threshold)
│   ├── large_allocator.h       # Block-packing allocator (>= threshold)
│   ├── vmm_allocator.h         # Top-level PyTorch allocator
│   └── substitution.h          # Pointer swap & VMM remap
├── csrc/                       # C++ implementation
│   ├── bindings.cc             # pybind11 Python bindings
│   ├── cuda_driver_api.cc
│   ├── physical_memory.cc
│   ├── small_allocator.cc
│   ├── large_allocator.cc
│   ├── vmm_allocator.cc
│   └── substitution.cc
├── benchmarks/                 # Performance benchmarks
│   ├── bench_torch_empty.py
│   ├── bench_alloc_patterns.py
│   └── bench_vmm_raw_api.cc
├── zero-cuda-graph.md          # Design blog post (Chinese)
├── DESIGN.md                   # Technical architecture doc (Chinese)
├── setup.py
└── LICENSE                     # MIT
```

## Documentation

- **[zero-cuda-graph.md](zero-cuda-graph.md)** — Design motivation, VMM API analysis, and optimization strategies
- **[DESIGN.md](DESIGN.md)** — Full technical architecture and implementation phases

## License

MIT
