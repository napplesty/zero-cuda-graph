"""
zcg.memory – memory diagnostics and cache management.
"""

from __future__ import annotations

from typing import Any


def memory_stats(device: int = 0) -> dict[str, Any]:
    """Return a dictionary of memory statistics for the given device.

    The returned dict has keys:

    - ``small_pool`` – stats for the small-tensor (< threshold) path,
      including ``current_bytes``, ``peak_bytes``, ``accumulated_alloc``,
      ``accumulated_freed``, ``num_allocs``, ``num_frees``, and physical
      memory counters.
    - ``large_pool`` – stats for the large-tensor (>= threshold) path
      (same sub-keys as ``small_pool``).
    - ``total_allocated`` – bytes currently in use across both pools.
    - ``total_physical``  – total physical bytes held (in-use + cached).
    - ``total_physical_cached`` – physical bytes sitting idle in free lists.
    - ``threshold`` – current routing threshold in bytes.
    - ``num_alloc_retries`` – number of OOM-retry attempts.
    - ``num_ooms`` – number of unrecoverable OOM events.
    """
    import zcg_c
    return zcg_c.get_memory_stats(device)


def empty_cache() -> None:
    """Release all cached (unused) physical memory back to the driver.

    This does **not** free tensors that are still alive – it only
    releases memory that was freed by PyTorch but retained in the
    allocator's free lists for reuse.  Also forces all pending
    deferred frees to complete.
    """
    import zcg_c
    zcg_c.empty_cache()


def reset_accumulated_stats(device: int = 0) -> None:
    """Reset accumulated (lifetime) allocation statistics.

    Resets ``accumulated_alloc``, ``accumulated_freed``, ``num_allocs``,
    ``num_frees``, ``num_alloc_retries``, and ``num_ooms`` counters
    for the given device.
    """
    import zcg_c
    zcg_c.reset_accumulated_stats(device)


def reset_peak_stats(device: int = 0) -> None:
    """Reset peak allocation statistics to the current level.

    After this call ``peak_bytes`` equals ``current_bytes`` for both
    the small and large pool on the given device.
    """
    import zcg_c
    zcg_c.reset_peak_stats(device)


def enable_peer_access(dev: int, dev_to_access: int) -> None:
    """Enable peer access between two CUDA devices.

    After this call, device *dev* can directly read/write memory that
    resides on device *dev_to_access*.  Both the CUDA runtime P2P path
    and the VMM ``CUmemAccessDesc`` are updated so that existing and
    future VMM mappings on *dev_to_access* are accessible from *dev*.
    """
    import zcg_c
    zcg_c.enable_peer_access(dev, dev_to_access)


def memory_summary(device: int = 0) -> str:
    """Return a human-readable summary of memory usage."""
    stats = memory_stats(device)

    def _fmt(n: int | float) -> str:
        """Format bytes as a human-friendly string."""
        for unit in ("B", "KB", "MB", "GB"):
            if abs(n) < 1024:
                return f"{n:.1f} {unit}"
            n /= 1024
        return f"{n:.1f} TB"

    sp = stats["small_pool"]
    lp = stats["large_pool"]

    lines = [
        f"=== VMMAllocator memory summary (device {device}) ===",
        f"Threshold:              {_fmt(stats['threshold'])}",
        f"Total allocated:        {_fmt(stats['total_allocated'])}",
        f"Total physical held:    {_fmt(stats['total_physical'])}",
        f"Total physical cached:  {_fmt(stats['total_physical_cached'])}",
        f"OOM retries:            {stats['num_alloc_retries']}",
        f"OOM failures:           {stats['num_ooms']}",
        "",
        "--- Small pool ---",
        f"  Current:              {_fmt(sp['current_bytes'])}",
        f"  Peak:                 {_fmt(sp['peak_bytes'])}",
        f"  Accumulated alloc:    {_fmt(sp['accumulated_alloc'])}",
        f"  Accumulated freed:    {_fmt(sp['accumulated_freed'])}",
        f"  Num allocs:           {sp['num_allocs']}",
        f"  Num frees:            {sp['num_frees']}",
        f"  Mapped (reserved):    {_fmt(sp['reserved_bytes'])}",
        f"  VA range:             {_fmt(sp['va_range_bytes'])}",
        f"  Physical cached:      {_fmt(sp['physical_cached'])}",
        f"  Physical in use:      {_fmt(sp['physical_in_use'])}",
        "",
        "--- Large pool ---",
        f"  Current:              {_fmt(lp['current_bytes'])}",
        f"  Peak:                 {_fmt(lp['peak_bytes'])}",
        f"  Accumulated alloc:    {_fmt(lp['accumulated_alloc'])}",
        f"  Accumulated freed:    {_fmt(lp['accumulated_freed'])}",
        f"  Num allocs:           {lp['num_allocs']}",
        f"  Num frees:            {lp['num_frees']}",
        f"  Physical cached:      {_fmt(lp['physical_cached'])}",
        f"  Physical in use:      {_fmt(lp['physical_in_use'])}",
    ]
    return "\n".join(lines)
