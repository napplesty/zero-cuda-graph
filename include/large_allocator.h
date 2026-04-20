#pragma once
// ============================================================
// large_allocator.h – allocator for tensors >= threshold.
//
// Each allocation gets its own VA reservation.  Physical blocks
// (16–256 MB) are greedily packed to cover the request size and
// mapped contiguously into the VA range.
//
// On free the VA is unmapped and released; physical blocks are
// returned to PhysicalMemoryManager for reuse.
// ============================================================

#include "common.h"
#include "physical_memory.h"

#include <mutex>
#include <unordered_map>
#include <vector>

#include <cuda_runtime.h>

namespace zcg {

class LargeAllocator {
 public:
  LargeAllocator(int device, PhysicalMemoryManager& pmm);
  ~LargeAllocator();

  /// Allocate at least `size` bytes.  The actual mapped size may be
  /// slightly larger due to rounding to block boundaries.
  void* allocate(size_t size, cudaStream_t stream);

  /// Free a pointer previously returned by allocate().
  void  free(void* ptr);

  /// Return all cached resources (no-op today – nothing is cached
  /// at this level; blocks go back to PhysicalMemoryManager).
  void  emptyCache();

  /// Does this pointer correspond to an active large allocation?
  bool  owns(const void* ptr) const;

  /// Register a peer device that should have read/write access to
  /// this allocator's mapped memory.  Already-mapped allocations will
  /// be updated; future allocations will include the peer automatically.
  void  addPeerDevice(int peer_device);

  // ---- statistics -----------------------------------------------
  size_t allocatedBytes() const;

 private:
  int device_;
  PhysicalMemoryManager& pmm_;

  /// One record per live allocation.
  struct Allocation {
    CUdeviceptr              va_base;
    size_t                   va_size;        // total VA reserved
    size_t                   requested_size; // original request
    std::vector<PhysicalBlock> blocks;
  };

  mutable std::mutex mu_;
  std::unordered_map<CUdeviceptr, Allocation> allocs_;
  size_t allocated_bytes_ = 0;

  // Peer device list (devices granted read/write access to our memory)
  std::vector<int> peer_devices_;

  // helpers
  /// Greedy packing: return the list of block sizes whose sum >= size.
  static std::vector<size_t> computeLayout(size_t size);

  /// Set access permissions for self + all registered peer devices.
  void setAccess(CUdeviceptr va, size_t size);
};

} // namespace zcg
