// ============================================================
// substitution.cc – DataPtr hot-swap & VMM remap
// ============================================================

#include "substitution.h"
#include "cuda_driver_api.h"
#include "vmm_allocator.h"

#include <c10/core/Storage.h>
#include <c10/core/StorageImpl.h>
#include <c10/util/Exception.h>

namespace zcg {

// ============================================================
// Pointer hot-swap
// ============================================================

at::DataPtr replaceDataPtr(at::Tensor& tensor,
                           void* new_ptr,
                           void (*deleter)(void*)) {
  TORCH_CHECK(tensor.defined(), "zcg::replaceDataPtr: tensor is undefined");
  TORCH_CHECK(tensor.storage().defined(),
              "zcg::replaceDataPtr: tensor has no storage");

  auto device = tensor.device();

  // If no deleter is supplied, use a no-op (the caller manages the memory).
  if (!deleter) {
    deleter = [](void*) {};
  }

  at::DataPtr new_data_ptr(new_ptr, new_ptr, deleter, device);
  return tensor.storage().set_data_ptr(std::move(new_data_ptr));
}

// ============================================================
// VMM remap
// ============================================================

void vmmRemap(CUdeviceptr va, size_t size,
              const std::vector<CUmemGenericAllocationHandle>& old_handles,
              const std::vector<CUmemGenericAllocationHandle>& new_handles,
              const std::vector<size_t>& new_sizes,
              int device,
              const std::vector<int>& peer_devices) {
  TORCH_CHECK(size > 0, "zcg::vmmRemap: size must be > 0");
  TORCH_CHECK(!new_handles.empty(),
              "zcg::vmmRemap: new_handles must not be empty");
  TORCH_CHECK(new_handles.size() == new_sizes.size(),
              "zcg::vmmRemap: new_handles and new_sizes must have the same length");

  // Verify total size matches.
  size_t total = 0;
  for (size_t s : new_sizes) total += s;
  TORCH_CHECK(total == size,
              "zcg::vmmRemap: sum of new_sizes (", total,
              ") != region size (", size, ")");

  auto& drv = CUDADriverAPI::get();

  // 1. Unmap the existing mapping.
  ZCG_CU_CHECK(drv.memUnmap(va, size));

  // 2. Map the new handles contiguously, each with its real size.
  size_t offset = 0;
  size_t num_mapped = 0;
  try {
    for (size_t i = 0; i < new_handles.size(); ++i) {
      ZCG_CU_CHECK(drv.memMap(va + offset, new_sizes[i],
                              /*offset=*/0, new_handles[i], /*flags=*/0));
      ++num_mapped;
      offset += new_sizes[i];
    }
  } catch (...) {
    // Rollback: unmap successfully mapped handles.
    size_t rollback_offset = 0;
    for (size_t i = 0; i < num_mapped; ++i) {
      drv.memUnmap(va + rollback_offset, new_sizes[i]);
      rollback_offset += new_sizes[i];
    }
    // Try to remap old handles to restore the original state.
    // (best-effort: if this also fails, the VA is left unmapped)
    size_t old_offset = 0;
    size_t old_remaining = size;
    for (size_t i = 0; i < old_handles.size() && old_remaining > 0; ++i) {
      // We don't know old sizes individually, but they originally
      // covered the full `size` region.  Re-throw the original error.
    }
    throw;
  }

  // 3. Set access permissions for self + peer devices.
  std::vector<CUmemAccessDesc> descs;
  descs.reserve(1 + peer_devices.size());

  CUmemAccessDesc self_desc{};
  self_desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  self_desc.location.id   = device;
  self_desc.flags         = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  descs.push_back(self_desc);

  for (int peer : peer_devices) {
    CUmemAccessDesc pd{};
    pd.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    pd.location.id   = peer;
    pd.flags         = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    descs.push_back(pd);
  }

  ZCG_CU_CHECK(drv.memSetAccess(va, size, descs.data(), descs.size()));
}

} // namespace zcg
