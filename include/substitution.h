#pragma once
// ============================================================
// substitution.h – utilities for hot-swapping the memory that
// backs a PyTorch tensor and for VMM-level remapping.
// ============================================================

#include <ATen/Tensor.h>
#include <cuda.h>
#include <cstddef>
#include <vector>

namespace zcg {

/// Replace the data pointer inside `tensor`'s Storage with `new_ptr`.
///
/// A new DataPtr is constructed using the supplied deleter (or the
/// default VMM deleter when `deleter == nullptr`).  The previous
/// DataPtr is returned so the caller can manage its lifetime.
///
/// This is the "pointer hot-swap" path – the tensor's metadata
/// (sizes, strides, dtype …) stays unchanged; only the underlying
/// memory address changes.
at::DataPtr replaceDataPtr(at::Tensor& tensor,
                           void* new_ptr,
                           void (*deleter)(void*) = nullptr);

/// VMM-level remap: unmap the old physical blocks at virtual address
/// `va` and map `new_handles` in their place.  The virtual address
/// seen by every tensor that references this region stays the same,
/// but the physical memory behind it changes.
///
/// @param va            device virtual address (must be page-aligned)
/// @param size          total byte length of the mapping
/// @param old_handles   physical handles currently mapped (will be unmapped)
/// @param new_handles   physical handles to map in their place
/// @param new_sizes     byte size of each handle in new_handles (must sum to `size`)
/// @param device        CUDA device ordinal
/// @param peer_devices  additional devices that should have R/W access (may be empty)
void vmmRemap(CUdeviceptr va, size_t size,
              const std::vector<CUmemGenericAllocationHandle>& old_handles,
              const std::vector<CUmemGenericAllocationHandle>& new_handles,
              const std::vector<size_t>& new_sizes,
              int device,
              const std::vector<int>& peer_devices = {});

} // namespace zcg
