"""
zcg – zero-cuda-graph
~~~~~~~~~~~~~~~~~~~~~

VMM-based CUDA allocator for PyTorch.

Importing this package **automatically** replaces the default CUDA caching
allocator with VMMAllocator.  This must happen before any CUDA tensor is
created, so ``import zcg`` should be one of the first lines in your script.

Example::

    import zcg          # ← allocator is now active
    import torch

    x = torch.randn(1024, device="cuda")  # uses VMMAllocator
"""

from zcg.config import configure  # noqa: F401 – re-export
from zcg.memory import (  # noqa: F401
    memory_stats,
    empty_cache,
    memory_summary,
    reset_accumulated_stats,
    reset_peak_stats,
    enable_peer_access,
)
from zcg.ops import replace_data_ptr, vmm_remap  # noqa: F401


def _auto_register() -> None:
    """Register VMMAllocator as the global CUDA allocator."""
    import torch  # noqa: F811

    if not torch.cuda.is_available():
        import warnings
        warnings.warn(
            "zcg: CUDA is not available – VMMAllocator will NOT be registered. "
            "GPU operations will use the default allocator.",
            stacklevel=2,
        )
        return

    import zcg_c
    zcg_c.register_vmm_allocator()


_auto_register()
