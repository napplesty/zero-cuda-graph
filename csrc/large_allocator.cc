// ============================================================
// large_allocator.cc – LargeAllocator implementation
// ============================================================

#include "large_allocator.h"
#include "cuda_driver_api.h"

#include <algorithm>
#include <c10/util/Exception.h>

namespace zcg {

// ============================================================
// Construction / destruction
// ============================================================

LargeAllocator::LargeAllocator(int device, PhysicalMemoryManager& pmm)
    : device_(device), pmm_(pmm) {}

LargeAllocator::~LargeAllocator() {
  // Free every live allocation (shouldn't happen in normal shutdown
  // because Python should have freed all tensors, but be safe).
  std::lock_guard<std::mutex> lk(mu_);
  auto& drv = CUDADriverAPI::get();
  for (auto& [va, alloc] : allocs_) {
    drv.memUnmap(alloc.va_base, alloc.va_size);
    for (auto& blk : alloc.blocks) {
      pmm_.freeLargeBlock(blk);
    }
    drv.memAddressFree(alloc.va_base, alloc.va_size);
  }
  allocs_.clear();
  allocated_bytes_ = 0;
}

// ============================================================
// Allocate
// ============================================================

void* LargeAllocator::allocate(size_t size, cudaStream_t /*stream*/) {
  TORCH_CHECK(size > 0, "zcg LargeAllocator: zero-size allocation");

  // 1. Compute which block sizes to combine.
  auto layout = computeLayout(size);

  // Total mapped size is the sum of all blocks in the layout.
  size_t total = 0;
  for (size_t s : layout) total += s;

  auto& drv = CUDADriverAPI::get();

  // 2. Reserve a contiguous VA range for this allocation.
  CUdeviceptr va = 0;
  ZCG_CU_CHECK(drv.memAddressReserve(
      &va, total, /*alignment=*/kSmallBlockSize,
      /*addr=*/0, /*flags=*/0));

  // 3. Obtain physical blocks and map them into the VA.
  //    Hold mu_ throughout mapping + setAccess to avoid a data race
  //    on peer_devices_ (which setAccess reads).
  std::lock_guard<std::mutex> lk(mu_);

  std::vector<PhysicalBlock> blocks;
  blocks.reserve(layout.size());
  size_t offset = 0;
  size_t num_mapped = 0;
  try {
    for (size_t blk_size : layout) {
      PhysicalBlock blk = pmm_.allocLargeBlock(blk_size);
      blocks.push_back(blk);
      ZCG_CU_CHECK(drv.memMap(
          va + offset, blk_size,
          /*offset=*/0, blk.handle, /*flags=*/0));
      ++num_mapped;
      offset += blk_size;
    }
    // 4. Set access permissions (reads peer_devices_, safe under mu_).
    setAccess(va, total);
  } catch (...) {
    size_t unmap_offset = 0;
    for (size_t i = 0; i < blocks.size(); ++i) {
      if (i < num_mapped) {
        drv.memUnmap(va + unmap_offset, blocks[i].size);
        unmap_offset += blocks[i].size;
      }
      pmm_.freeLargeBlock(blocks[i]);
    }
    drv.memAddressFree(va, total);
    throw;
  }

  // 5. Record the allocation (already under mu_).
  allocs_[va] = Allocation{va, total, size, std::move(blocks)};
  allocated_bytes_ += total;

  return reinterpret_cast<void*>(va);
}

// ============================================================
// Free
// ============================================================

void LargeAllocator::free(void* ptr) {
  if (!ptr) return;
  CUdeviceptr va = reinterpret_cast<CUdeviceptr>(ptr);

  Allocation alloc;
  {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = allocs_.find(va);
    TORCH_CHECK(it != allocs_.end(),
                "zcg LargeAllocator::free: unknown pointer ", ptr);
    alloc = std::move(it->second);
    allocs_.erase(it);
    allocated_bytes_ -= alloc.va_size;
  }

  auto& drv = CUDADriverAPI::get();

  // Unmap physical blocks.
  size_t offset = 0;
  for (auto& blk : alloc.blocks) {
    drv.memUnmap(alloc.va_base + offset, blk.size);
    pmm_.freeLargeBlock(blk);
    offset += blk.size;
  }

  // Release the VA range.
  drv.memAddressFree(alloc.va_base, alloc.va_size);
}

// ============================================================
// Empty cache (no-op at this level)
// ============================================================

void LargeAllocator::emptyCache() {
  // Large allocator has no per-allocation caching.
  // The PhysicalMemoryManager caches the underlying blocks.
}

// ============================================================
// Owns check
// ============================================================

bool LargeAllocator::owns(const void* ptr) const {
  std::lock_guard<std::mutex> lk(mu_);
  return allocs_.count(reinterpret_cast<CUdeviceptr>(ptr)) > 0;
}

// ============================================================
// Statistics
// ============================================================

size_t LargeAllocator::allocatedBytes() const {
  std::lock_guard<std::mutex> lk(mu_);
  return allocated_bytes_;
}

// ============================================================
// Greedy block-packing algorithm
// ============================================================

// static
std::vector<size_t> LargeAllocator::computeLayout(size_t size) {
  std::vector<size_t> layout;
  size_t remaining = size;

  // Greedily pick the largest block class that fits.
  for (size_t i = 0; i < kNumLargeClasses; ++i) {
    size_t cls = kLargeBlockSizes[i];
    while (remaining >= cls) {
      layout.push_back(cls);
      remaining -= cls;
    }
  }

  // If there's a leftover, round up to the smallest block class that fits.
  if (remaining > 0) {
    size_t best = kMinLargeBlock; // fallback to smallest class
    for (int i = static_cast<int>(kNumLargeClasses) - 1; i >= 0; --i) {
      if (kLargeBlockSizes[i] >= remaining) {
        best = kLargeBlockSizes[i];
        break;  // smallest class that covers the remainder
      }
    }
    layout.push_back(best);
  }

  return layout;
}

// ============================================================
// Internal: set access permissions
// ============================================================

void LargeAllocator::setAccess(CUdeviceptr va, size_t size) {
  // Build descriptors for self + all registered peer devices.
  std::vector<CUmemAccessDesc> descs;
  descs.reserve(1 + peer_devices_.size());

  CUmemAccessDesc self_desc{};
  self_desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  self_desc.location.id   = device_;
  self_desc.flags         = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  descs.push_back(self_desc);

  for (int peer : peer_devices_) {
    CUmemAccessDesc pd{};
    pd.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    pd.location.id   = peer;
    pd.flags         = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    descs.push_back(pd);
  }

  ZCG_CU_CHECK(CUDADriverAPI::get().memSetAccess(
      va, size, descs.data(), descs.size()));
}

// ============================================================
// Peer device access
// ============================================================

void LargeAllocator::addPeerDevice(int peer_device) {
  std::lock_guard<std::mutex> lk(mu_);

  // Deduplicate.
  for (int d : peer_devices_) {
    if (d == peer_device) return;
  }
  peer_devices_.push_back(peer_device);

  // Grant the new peer access to every live allocation.
  CUmemAccessDesc desc{};
  desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  desc.location.id   = peer_device;
  desc.flags         = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

  for (auto& [va, alloc] : allocs_) {
    ZCG_CU_CHECK(CUDADriverAPI::get().memSetAccess(
        alloc.va_base, alloc.va_size, &desc, 1));
  }
}

} // namespace zcg
