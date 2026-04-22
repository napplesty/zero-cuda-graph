// ============================================================
// physical_memory.cc – PhysicalMemoryManager implementation
// ============================================================

#include "physical_memory.h"
#include "cuda_driver_api.h"

#include <algorithm>
#include <cstring>
#include <c10/util/Exception.h>

namespace zcg {

// ============================================================
// Construction / destruction
// ============================================================

PhysicalMemoryManager::PhysicalMemoryManager(int device) : device_(device) {
  // Prepare the allocation property template that will be reused for
  // every cuMemCreate call on this device.
  std::memset(&alloc_prop_, 0, sizeof(alloc_prop_));
  alloc_prop_.type               = CU_MEM_ALLOCATION_TYPE_PINNED;
  alloc_prop_.location.type      = CU_MEM_LOCATION_TYPE_DEVICE;
  alloc_prop_.location.id        = device;

  // Sanity-check: query granularity to make sure VMM is available.
  size_t granularity = 0;
  ZCG_CU_CHECK(CUDADriverAPI::get().memGetAllocationGranularity(
      &granularity, &alloc_prop_,
      CU_MEM_ALLOC_GRANULARITY_MINIMUM));
  TORCH_CHECK(granularity > 0,
              "zcg: VMM allocation granularity is 0 – device may not support VMM");
  TORCH_CHECK(kSmallBlockSize % granularity == 0,
              "zcg: kSmallBlockSize (", kSmallBlockSize,
              ") is not a multiple of the device granularity (", granularity, ")");
}

PhysicalMemoryManager::~PhysicalMemoryManager() {
  // Release everything still in the free lists.
  trim();
}

// ============================================================
// Small blocks
// ============================================================

PhysicalBlock PhysicalMemoryManager::allocSmallBlock() {
  std::lock_guard<std::mutex> lk(mu_);
  if (!small_free_.empty()) {
    auto blk = small_free_.back();
    small_free_.pop_back();
    small_cached_bytes_ -= blk.size;
    small_in_use_bytes_ += blk.size;
    return blk;
  }
  // No cached block – create a new one.
  auto blk = createBlock(kSmallBlockSize);
  small_in_use_bytes_ += blk.size;
  return blk;
}

void PhysicalMemoryManager::freeSmallBlock(PhysicalBlock blk) {
  std::lock_guard<std::mutex> lk(mu_);
  small_in_use_bytes_ -= blk.size;
  small_cached_bytes_ += blk.size;
  small_free_.push_back(blk);
}

// ============================================================
// Large blocks
// ============================================================

PhysicalBlock PhysicalMemoryManager::allocLargeBlock(size_t size_class) {
  std::lock_guard<std::mutex> lk(mu_);
  auto& pool = large_free_[size_class];
  if (!pool.empty()) {
    auto blk = pool.back();
    pool.pop_back();
    large_cached_bytes_ -= blk.size;
    large_in_use_bytes_ += blk.size;
    return blk;
  }
  auto blk = createBlock(size_class);
  large_in_use_bytes_ += blk.size;
  return blk;
}

void PhysicalMemoryManager::freeLargeBlock(PhysicalBlock blk) {
  std::lock_guard<std::mutex> lk(mu_);
  large_in_use_bytes_ -= blk.size;
  large_cached_bytes_ += blk.size;
  large_free_[blk.size].push_back(blk);
}

// ============================================================
// Purge – release *enough* cached blocks (largest first)
// ============================================================

size_t PhysicalMemoryManager::purge(size_t min_bytes) {
  if (min_bytes == 0) return 0;

  // Phase 1: collect blocks to release under the lock.
  std::vector<PhysicalBlock> to_destroy;

  size_t freed = 0;
  {
    std::lock_guard<std::mutex> lk(mu_);

    // 1a. Release large blocks, largest size class first.
    for (auto it = large_free_.rbegin();
         it != large_free_.rend() && freed < min_bytes; ) {
      auto& pool = it->second;
      while (!pool.empty() && freed < min_bytes) {
        auto blk = pool.back();
        pool.pop_back();
        large_cached_bytes_ -= blk.size;
        freed += blk.size;
        to_destroy.push_back(blk);
      }
      ++it;
    }

    // Remove empty buckets.
    for (auto it = large_free_.begin(); it != large_free_.end(); ) {
      if (it->second.empty()) {
        it = large_free_.erase(it);
      } else {
        ++it;
      }
    }

    // 1b. If still not enough, release small blocks.
    while (!small_free_.empty() && freed < min_bytes) {
      auto blk = small_free_.back();
      small_free_.pop_back();
      small_cached_bytes_ -= blk.size;
      freed += blk.size;
      to_destroy.push_back(blk);
    }

    // Only count effective purges (where something was actually freed).
    if (freed > 0) {
      purge_count_++;
      total_purged_bytes_ += freed;
    }
  }

  // Phase 2: actually release blocks OUTSIDE the lock so that
  // concurrent alloc/free operations are not blocked during the
  // (potentially slow) cuMemRelease driver calls.
  for (auto& blk : to_destroy) {
    destroyBlock(blk);
  }

  return freed;
}

// ============================================================
// Trim – release *all* cached blocks
// ============================================================

void PhysicalMemoryManager::trim() {
  std::lock_guard<std::mutex> lk(mu_);
  for (auto& blk : small_free_) {
    destroyBlock(blk);
  }
  small_cached_bytes_ = 0;
  small_free_.clear();

  for (auto& [sz, pool] : large_free_) {
    for (auto& blk : pool) {
      destroyBlock(blk);
    }
  }
  large_cached_bytes_ = 0;
  large_free_.clear();
}

// ============================================================
// Statistics
// ============================================================

size_t PhysicalMemoryManager::totalPhysicalAllocated() const {
  std::lock_guard<std::mutex> lk(mu_);
  return small_in_use_bytes_ + small_cached_bytes_
       + large_in_use_bytes_ + large_cached_bytes_;
}

size_t PhysicalMemoryManager::totalPhysicalCached() const {
  std::lock_guard<std::mutex> lk(mu_);
  return small_cached_bytes_ + large_cached_bytes_;
}

size_t PhysicalMemoryManager::smallPoolCached() const {
  std::lock_guard<std::mutex> lk(mu_);
  return small_cached_bytes_;
}

size_t PhysicalMemoryManager::largePoolCached() const {
  std::lock_guard<std::mutex> lk(mu_);
  return large_cached_bytes_;
}

size_t PhysicalMemoryManager::smallPoolInUse() const {
  std::lock_guard<std::mutex> lk(mu_);
  return small_in_use_bytes_;
}

size_t PhysicalMemoryManager::largePoolInUse() const {
  std::lock_guard<std::mutex> lk(mu_);
  return large_in_use_bytes_;
}

size_t PhysicalMemoryManager::purgeCount() const {
  std::lock_guard<std::mutex> lk(mu_);
  return purge_count_;
}

size_t PhysicalMemoryManager::totalPurgedBytes() const {
  std::lock_guard<std::mutex> lk(mu_);
  return total_purged_bytes_;
}

// ============================================================
// Internal helpers
// ============================================================

PhysicalBlock PhysicalMemoryManager::createBlock(size_t size) {
  // NOTE: caller must hold mu_
  PhysicalBlock blk;
  blk.size   = size;
  blk.device = device_;
  ZCG_CU_CHECK(CUDADriverAPI::get().memCreate(
      &blk.handle, size, &alloc_prop_, /*flags=*/0));
  return blk;
}

void PhysicalMemoryManager::destroyBlock(PhysicalBlock& blk) {
  if (blk.handle) {
    // Ignore errors during cleanup (e.g. during process shutdown).
    CUDADriverAPI::get().memRelease(blk.handle);
    blk.handle = 0;
  }
}

} // namespace zcg
