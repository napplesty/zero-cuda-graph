"""
zcg.ops – tensor pointer manipulation and VMM remapping.
"""

from __future__ import annotations

from typing import Optional, Callable

import torch


def replace_data_ptr(
    tensor: torch.Tensor,
    new_ptr: int,
) -> None:
    """Replace the underlying device pointer of *tensor* in-place.

    After this call the tensor's metadata (shape, stride, dtype …) is
    unchanged, but every element access goes through *new_ptr*.

    Parameters
    ----------
    tensor : torch.Tensor
        Must be a CUDA tensor with valid storage.
    new_ptr : int
        The device-virtual-address of the new memory region, obtained
        e.g. from ``tensor.data_ptr()`` of another tensor, from a
        custom allocator, or from a VMM mapping.

    .. warning::
        The caller is responsible for ensuring that *new_ptr* points
        to a region at least as large as ``tensor.nelement() *
        tensor.element_size()`` bytes and that it remains valid for
        the lifetime of *tensor*.
    """
    import zcg_c
    zcg_c.replace_data_ptr(tensor, new_ptr)


def vmm_remap(
    va: int,
    size: int,
    old_handles: list[int],
    new_handles: list[int],
    new_sizes: list[int],
    device: int = 0,
    peer_devices: list[int] | None = None,
) -> None:
    """Remap physical memory behind a virtual address.

    This is the core primitive for CUDA-Graph-friendly memory
    management: the virtual address that was recorded during graph
    capture remains the same, but the physical pages behind it are
    swapped out.

    Parameters
    ----------
    va : int
        Device virtual address (page-aligned) to remap.
    size : int
        Total byte length of the region.
    old_handles : list[int]
        Physical allocation handles currently mapped (will be unmapped).
    new_handles : list[int]
        Physical allocation handles to map in their place.
    new_sizes : list[int]
        Byte size of each handle in *new_handles*.  Must have the same
        length as *new_handles*, and their sum must equal *size*.
    device : int
        CUDA device ordinal.
    peer_devices : list[int], optional
        Additional devices that should have read/write access to the
        remapped region.  If ``None`` or empty, only *device* has access.
    """
    import zcg_c
    zcg_c.vmm_remap(va, size, old_handles, new_handles, new_sizes,
                     device, peer_devices or [])
