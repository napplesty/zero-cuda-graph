#pragma once
// ============================================================
// physical_memory.h – manages creation / caching / release of
// CUDA VMM physical memory blocks (CUmemGenericAllocationHandle).
//
// Two separate pools:
//   • small pool  – fixed 2 MB blocks
//   • large pool  – blocks in {16,32,64,128,256} MB size classes
//
// Pools are strictly isolated: a large block is never carved up
// for small allocations and vice-versa.
// ============================================================

#include "common.h"
#include "cuda_driver_api.h"

#include <map>
#include <mutex>
#include <vector>

namespace zcg {

class PhysicalMemoryManager {
 public:
  explicit PhysicalMemoryManager(int device);
  ~PhysicalMemoryManager();

  // ---- small blocks (2 MB) -------------------------------------
  PhysicalBlock allocSmallBlock();
  void          freeSmallBlock(PhysicalBlock blk);

  // ---- large blocks (by size class) ----------------------------
  /// @param size_class  one of kLargeBlockSizes[]
  PhysicalBlock allocLargeBlock(size_t size_class);
  void          freeLargeBlock(PhysicalBlock blk);

  // ---- housekeeping --------------------------------------------
  /// Release all *unused* blocks back to the driver.
  void trim();

  // ---- statistics ----------------------------------------------
  size_t totalPhysicalAllocated() const;   // bytes currently held (in-use + cached)
  size_t totalPhysicalCached()    const;   // bytes sitting idle in free lists
  size_t smallPoolCached()        const;
  size_t largePoolCached()        const;
  size_t smallPoolInUse()         const;
  size_t largePoolInUse()         const;

 private:
  int device_;

  /// Allocation properties (reused for every cuMemCreate call on this device).
  CUmemAllocationProp alloc_prop_{};

  mutable std::mutex mu_;

  // Free lists (ready to be reused without a cuMemCreate round-trip)
  std::vector<PhysicalBlock>                     small_free_;
  std::map<size_t, std::vector<PhysicalBlock>>   large_free_;  // key = size_class

  // Accounting
  size_t small_in_use_bytes_ = 0;
  size_t small_cached_bytes_ = 0;
  size_t large_in_use_bytes_ = 0;
  size_t large_cached_bytes_ = 0;

  // Internal helpers
  PhysicalBlock createBlock(size_t size);
  void          destroyBlock(PhysicalBlock& blk);
};

} // namespace zcg
