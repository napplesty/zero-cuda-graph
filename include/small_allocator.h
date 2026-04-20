#pragma once
// ============================================================
// small_allocator.h – sub-allocator for tensors < threshold.
//
// Design:
//   • Reserves a large contiguous VA range at construction time.
//   • Internally manages a free-list (address-ordered, best-fit)
//     of variable-sized sub-ranges within that VA.
//   • Maps 2 MB physical blocks on demand as allocations touch
//     previously unmapped pages.
//   • Physical blocks are only unmapped on explicit emptyCache().
// ============================================================

#include "common.h"
#include "physical_memory.h"

#include <map>
#include <mutex>
#include <set>
#include <unordered_map>

#include <cuda_runtime.h>

namespace zcg {

class SmallAllocator {
 public:
  SmallAllocator(int device, PhysicalMemoryManager& pmm, size_t va_pool_size);
  ~SmallAllocator();

  /// Allocate `size` bytes (will be rounded up to kMinBlockAlign).
  /// Returns a device pointer inside the pre-reserved VA range.
  void* allocate(size_t size, cudaStream_t stream);

  /// Free a pointer previously returned by allocate().
  void  free(void* ptr);

  /// Release all physical blocks whose pages are completely free.
  void  emptyCache();

  /// Does this pointer fall within our VA range?
  bool  owns(const void* ptr) const;

  /// Register a peer device that should have read/write access to
  /// this allocator's mapped memory.  Already-mapped pages will be
  /// updated; future mappings will include the peer automatically.
  void  addPeerDevice(int peer_device);

  // ---- statistics -----------------------------------------------
  size_t allocatedBytes() const;
  size_t reservedBytes()  const;   // physical bytes mapped
  size_t vaRangeSize()    const { return va_size_; }

 private:
  int device_;
  PhysicalMemoryManager& pmm_;

  // VA range managed by this allocator
  CUdeviceptr va_base_ = 0;
  size_t      va_size_ = 0;

  // Physical page tracking
  size_t num_pages_ = 0;            // va_size_ / kSmallBlockSize
  std::vector<bool>          page_mapped_;   // is page i backed by physical mem?
  std::vector<PhysicalBlock> page_blocks_;   // handle for each mapped page

  // High-water mark (bump pointer for new allocations)
  size_t high_water_ = 0;

  // Free list – address-ordered for coalescing
  std::map<CUdeviceptr, size_t>            free_by_addr_;  // va -> size
  std::map<size_t, std::set<CUdeviceptr>>  free_by_size_;  // size -> set<va>

  // Active allocations
  std::unordered_map<CUdeviceptr, size_t> allocated_;      // va -> size

  mutable std::mutex mu_;

  // Accounting
  size_t allocated_bytes_ = 0;
  size_t mapped_bytes_    = 0;

  // Peer device list (devices granted read/write access to our memory)
  std::vector<int> peer_devices_;

  // helpers
  void ensurePagesMapped(CUdeviceptr start, size_t size);
  void addFreeBlock(CUdeviceptr va, size_t size);
  void removeFreeBlock(CUdeviceptr va, size_t size);
  void coalesce(CUdeviceptr va, size_t size);

  /// Set access permissions for self + all registered peer devices.
  void setAccess(CUdeviceptr va, size_t size);
};

} // namespace zcg
