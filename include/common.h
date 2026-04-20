#pragma once

#include <cuda.h>
#include <cstddef>
#include <cstdint>

namespace zcg {

// ============================================================
// Alignment & sizing helpers
// ============================================================

inline size_t roundUp(size_t n, size_t align) {
  return (n + align - 1) & ~(align - 1);
}

// ============================================================
// Constants
// ============================================================

// Sub-allocation alignment inside the small allocator (matches PyTorch)
constexpr size_t kMinBlockAlign = 512;

// Physical block granularity for the small-tensor path
constexpr size_t kSmallBlockSize = 2ULL << 20; // 2 MB

// Physical block size classes for the large-tensor path (descending)
constexpr size_t kLargeBlockSizes[] = {
    256ULL << 20, // 256 MB
    128ULL << 20, // 128 MB
     64ULL << 20, //  64 MB
     32ULL << 20, //  32 MB
     16ULL << 20, //  16 MB
};
constexpr size_t kNumLargeClasses =
    sizeof(kLargeBlockSizes) / sizeof(kLargeBlockSizes[0]);
constexpr size_t kMinLargeBlock = kLargeBlockSizes[kNumLargeClasses - 1];

// Default routing threshold: < threshold → small, >= threshold → large
constexpr size_t kDefaultThreshold = kMinLargeBlock; // 16 MB

// Default virtual-address pool reserved for the small allocator per device
constexpr size_t kDefaultSmallVASize = 32ULL << 30; // 32 GB

// ============================================================
// Lightweight value type for a physical memory handle
// ============================================================

struct PhysicalBlock {
  CUmemGenericAllocationHandle handle = 0;
  size_t size   = 0;
  int    device = -1;
};

} // namespace zcg
