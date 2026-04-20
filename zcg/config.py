"""
zcg.config – runtime configuration for VMMAllocator.
"""

from __future__ import annotations

# Default values (mirrored from C++ common.h).
DEFAULT_THRESHOLD: int = 16 * 1024 * 1024        # 16 MB
DEFAULT_SMALL_VA_SIZE: int = 32 * 1024**3         # 32 GB
SMALL_BLOCK_SIZE: int = 2 * 1024 * 1024           # 2 MB
LARGE_BLOCK_SIZES: list[int] = [256, 128, 64, 32, 16]  # in MB


def configure(
    *,
    threshold: int | None = None,
) -> None:
    """Adjust VMMAllocator parameters at runtime.

    Parameters
    ----------
    threshold : int, optional
        The boundary (in bytes) between the small-tensor path and the
        large-tensor path.  Allocations smaller than *threshold* are
        sub-allocated from 2 MB physical blocks; larger ones use the
        greedy block-pool strategy.

    .. note::
        This function should be called **after** ``import zcg`` but
        **before** allocating any CUDA tensors for the change to take
        full effect.
    """
    import zcg_c

    if threshold is not None:
        zcg_c.set_threshold(threshold)
