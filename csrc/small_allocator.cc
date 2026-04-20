// ============================================================
// small_allocator.cc – SmallAllocator implementation
// ============================================================

#include "small_allocator.h"
#include "cuda_driver_api.h"

#include <algorithm>
#include <c10/util/Exception.h>

namespace zcg {

// ============================================================
// Construction / destruction
// ============================================================

SmallAllocator::SmallAllocator(int device,
                               PhysicalMemoryManager& pmm,
                               size_t va_pool_size)
    : device_(device), pmm_(pmm), va_size_(va_pool_size) {
  // Round up to a multiple of kSmallBlockSize.
  va_size_ = roundUp(va_size_, kSmallBlockSize);
  num_pages_ = va_size_ / kSmallBlockSize;

  page_mapped_.resize(num_pages_, false);
  page_blocks_.resize(num_pages_);

  // Reserve a contiguous VA range from the driver.
  ZCG_CU_CHECK(CUDADriverAPI::get().memAddressReserve(
      &va_base_, va_size_,
      /*alignment=*/kSmallBlockSize,
      /*addr=*/0, /*flags=*/0));
}

SmallAllocator::~SmallAllocator() {
  // Unmap everything, then release the VA range.
  for (size_t i = 0; i < num_pages_; ++i) {
    if (page_mapped_[i]) {
      CUDADriverAPI::get().memUnmap(
          va_base_ + i * kSmallBlockSize, kSmallBlockSize);
      pmm_.freeSmallBlock(page_blocks_[i]);
      page_mapped_[i] = false;
    }
  }
  if (va_base_) {
    CUDADriverAPI::get().memAddressFree(va_base_, va_size_);
  }
}

// ============================================================
// Allocate
// ============================================================

void* SmallAllocator::allocate(size_t size, cudaStream_t /*stream*/) {
  size = roundUp(std::max(size, (size_t)1), kMinBlockAlign);

  std::lock_guard<std::mutex> lk(mu_);

  // 1. Try to find a suitable free block (best-fit).
  CUdeviceptr result = 0;
  auto it = free_by_size_.lower_bound(size);
  if (it != free_by_size_.end()) {
    // Pick the first address in the smallest-sufficient bucket.
    auto& addrs = it->second;
    CUdeviceptr va = *addrs.begin();
    size_t blk_size = it->first;

    removeFreeBlock(va, blk_size);

    result = va;
    // If the free block is larger than needed, split it.
    if (blk_size > size) {
      addFreeBlock(va + size, blk_size - size);
    }
  } else {
    // 2. Bump allocate from the high-water mark.
    TORCH_CHECK(high_water_ + size <= va_size_,
                "zcg SmallAllocator: VA space exhausted (", va_size_,
                " bytes).  Consider increasing the VA pool size.");
    result = va_base_ + high_water_;
    high_water_ += size;
  }

  // 3. Ensure the physical pages are mapped.
  ensurePagesMapped(result, size);

  // 4. Track the allocation.
  allocated_[result] = size;
  allocated_bytes_ += size;

  return reinterpret_cast<void*>(result);
}

// ============================================================
// Free
// ============================================================

void SmallAllocator::free(void* ptr) {
  if (!ptr) return;
  CUdeviceptr va = reinterpret_cast<CUdeviceptr>(ptr);

  std::lock_guard<std::mutex> lk(mu_);

  auto it = allocated_.find(va);
  TORCH_CHECK(it != allocated_.end(),
              "zcg SmallAllocator::free: unknown pointer ", ptr);

  size_t size = it->second;
  allocated_.erase(it);
  allocated_bytes_ -= size;

  // Return to free list and coalesce.
  coalesce(va, size);
}

// ============================================================
// emptyCache – unmap fully-free pages
// ============================================================

void SmallAllocator::emptyCache() {
  std::lock_guard<std::mutex> lk(mu_);

  // For each 2 MB page, check if it is entirely covered by free ranges.
  // We do this by scanning the free list and computing per-page free bytes.
  std::vector<size_t> page_free_bytes(num_pages_, 0);

  for (auto& [va, sz] : free_by_addr_) {
    size_t first_page = (va - va_base_) / kSmallBlockSize;
    CUdeviceptr end = va + sz;
    for (size_t p = first_page; p < num_pages_; ++p) {
      CUdeviceptr page_start = va_base_ + p * kSmallBlockSize;
      CUdeviceptr page_end   = page_start + kSmallBlockSize;
      if (page_start >= end) break;
      CUdeviceptr overlap_start = (va > page_start) ? va : page_start;
      CUdeviceptr overlap_end   = (end < page_end)  ? end : page_end;
      if (overlap_end > overlap_start) {
        page_free_bytes[p] += (overlap_end - overlap_start);
      }
    }
  }

  for (size_t p = 0; p < num_pages_; ++p) {
    if (page_mapped_[p] && page_free_bytes[p] == kSmallBlockSize) {
      CUdeviceptr page_va = va_base_ + p * kSmallBlockSize;
      CUDADriverAPI::get().memUnmap(page_va, kSmallBlockSize);
      pmm_.freeSmallBlock(page_blocks_[p]);
      page_mapped_[p] = false;
      page_blocks_[p] = {};
      mapped_bytes_ -= kSmallBlockSize;
    }
  }
}

// ============================================================
// Owns check
// ============================================================

bool SmallAllocator::owns(const void* ptr) const {
  auto va = reinterpret_cast<CUdeviceptr>(ptr);
  return va >= va_base_ && va < va_base_ + va_size_;
}

// ============================================================
// Statistics
// ============================================================

size_t SmallAllocator::allocatedBytes() const {
  std::lock_guard<std::mutex> lk(mu_);
  return allocated_bytes_;
}

size_t SmallAllocator::reservedBytes() const {
  std::lock_guard<std::mutex> lk(mu_);
  return mapped_bytes_;
}

// ============================================================
// Internal: ensure pages are backed by physical memory
// ============================================================

void SmallAllocator::ensurePagesMapped(CUdeviceptr start, size_t size) {
  size_t first = (start - va_base_) / kSmallBlockSize;
  size_t last  = (start + size - 1 - va_base_) / kSmallBlockSize;

  for (size_t p = first; p <= last; ++p) {
    if (!page_mapped_[p]) {
      PhysicalBlock blk = pmm_.allocSmallBlock();
      CUdeviceptr page_va = va_base_ + p * kSmallBlockSize;

      ZCG_CU_CHECK(CUDADriverAPI::get().memMap(
          page_va, kSmallBlockSize,
          /*offset=*/0, blk.handle, /*flags=*/0));

      setAccess(page_va, kSmallBlockSize);

      page_mapped_[p] = true;
      page_blocks_[p] = blk;
      mapped_bytes_ += kSmallBlockSize;
    }
  }
}

// ============================================================
// Internal: free-list management with coalescing
// ============================================================

void SmallAllocator::addFreeBlock(CUdeviceptr va, size_t size) {
  free_by_addr_[va] = size;
  free_by_size_[size].insert(va);
}

void SmallAllocator::removeFreeBlock(CUdeviceptr va, size_t size) {
  free_by_addr_.erase(va);
  auto& bucket = free_by_size_[size];
  bucket.erase(va);
  if (bucket.empty()) free_by_size_.erase(size);
}

void SmallAllocator::coalesce(CUdeviceptr va, size_t size) {
  CUdeviceptr merged_va = va;
  size_t merged_size = size;

  // Try to merge with the predecessor.
  auto it = free_by_addr_.lower_bound(va);
  if (it != free_by_addr_.begin()) {
    auto prev = std::prev(it);
    if (prev->first + prev->second == va) {
      merged_va = prev->first;
      merged_size += prev->second;
      removeFreeBlock(prev->first, prev->second);
    }
  }

  // Try to merge with the successor.
  it = free_by_addr_.lower_bound(va + size);
  if (it != free_by_addr_.end() && it->first == merged_va + merged_size) {
    merged_size += it->second;
    removeFreeBlock(it->first, it->second);
  }

  addFreeBlock(merged_va, merged_size);
}

// ============================================================
// Internal: set access permissions after mapping
// ============================================================

void SmallAllocator::setAccess(CUdeviceptr va, size_t size) {
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

void SmallAllocator::addPeerDevice(int peer_device) {
  std::lock_guard<std::mutex> lk(mu_);

  // Deduplicate.
  for (int d : peer_devices_) {
    if (d == peer_device) return;
  }
  peer_devices_.push_back(peer_device);

  // Grant the new peer access to every already-mapped page.
  CUmemAccessDesc desc{};
  desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  desc.location.id   = peer_device;
  desc.flags         = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

  for (size_t p = 0; p < num_pages_; ++p) {
    if (page_mapped_[p]) {
      CUdeviceptr page_va = va_base_ + p * kSmallBlockSize;
      ZCG_CU_CHECK(CUDADriverAPI::get().memSetAccess(
          page_va, kSmallBlockSize, &desc, 1));
    }
  }
}

} // namespace zcg
