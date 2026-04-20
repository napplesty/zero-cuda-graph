// ============================================================
// vmm_allocator.cc – VMMAllocator implementation (Phase 4)
//
// Features:
//   - Event-based stream-aware deferred recycling
//   - OOM retry with cache trimming
//   - CUDA Graph pool isolation
//   - Full DeviceStats tracking (peak / accumulated)
//   - Multi-GPU peer access
// ============================================================

#include "vmm_allocator.h"

#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <cuda_runtime.h>

namespace zcg {

// ============================================================
// EventPool – reusable pool of CUDA events
// ============================================================

cudaEvent_t EventPool::acquire() {
  std::lock_guard<std::mutex> lk(mu_);
  if (!pool_.empty()) {
    cudaEvent_t e = pool_.back();
    pool_.pop_back();
    return e;
  }
  // Create a lightweight event (no timing overhead).
  cudaEvent_t e;
  C10_CUDA_CHECK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
  return e;
}

void EventPool::release(cudaEvent_t e) {
  std::lock_guard<std::mutex> lk(mu_);
  pool_.push_back(e);
}

EventPool::~EventPool() {
  for (auto e : pool_) {
    cudaEventDestroy(e);  // ignore errors during shutdown
  }
}

// ============================================================
// PoolStats – atomic stat counters
// ============================================================

void PoolStats::recordAlloc(int64_t n) {
  auto cur = current_bytes.fetch_add(n, std::memory_order_relaxed) + n;
  accumulated_alloc.fetch_add(n, std::memory_order_relaxed);
  num_allocs.fetch_add(1, std::memory_order_relaxed);

  // CAS loop to update peak.
  auto prev = peak_bytes.load(std::memory_order_relaxed);
  while (cur > prev &&
         !peak_bytes.compare_exchange_weak(
             prev, cur,
             std::memory_order_relaxed, std::memory_order_relaxed)) {
  }
}

void PoolStats::recordFree(int64_t n) {
  current_bytes.fetch_sub(n, std::memory_order_relaxed);
  accumulated_freed.fetch_add(n, std::memory_order_relaxed);
  num_frees.fetch_add(1, std::memory_order_relaxed);
}

void PoolStats::resetAccumulated() {
  accumulated_alloc.store(0, std::memory_order_relaxed);
  accumulated_freed.store(0, std::memory_order_relaxed);
  num_allocs.store(0, std::memory_order_relaxed);
  num_frees.store(0, std::memory_order_relaxed);
}

void PoolStats::resetPeak() {
  peak_bytes.store(current_bytes.load(std::memory_order_relaxed),
                   std::memory_order_relaxed);
}

// ============================================================
// Global singleton
// ============================================================

static std::shared_ptr<VMMAllocator> g_vmm_allocator;

// Deleter called by DataPtr when a tensor is freed.
static void vmm_delete(void* ptr) {
  if (auto* alloc = getVMMAllocator()) {
    alloc->free_impl(ptr);
  }
}

// Expose vmm_delete for raw_deleter() override.
at::DeleterFnPtr VMMAllocator::raw_deleter() const {
  return &vmm_delete;
}

// ============================================================
// Registration
// ============================================================

void registerVMMAllocator() {
  TORCH_CHECK(!g_vmm_allocator,
              "zcg: VMMAllocator is already registered");

  auto* current = cca::allocator.load();
  TORCH_CHECK(!current || !current->initialized(),
              "zcg: cannot replace an already-initialized CUDA allocator. "
              "Make sure to `import zcg` before any CUDA usage.");

  g_vmm_allocator = std::make_shared<VMMAllocator>();

  int count = 0;
  cudaError_t err = cudaGetDeviceCount(&count);
  if (err != cudaSuccess) count = 0;
  g_vmm_allocator->init(count);

  cca::allocator.store(g_vmm_allocator.get());
}

VMMAllocator* getVMMAllocator() {
  return g_vmm_allocator.get();
}

// ============================================================
// Construction / destruction
// ============================================================

VMMAllocator::VMMAllocator() = default;

VMMAllocator::~VMMAllocator() {
  // Force-process any remaining pending frees before tearing down.
  for (int d = 0; d < device_count_; ++d) {
    forceProcessPendingFrees(d);
  }
}

// ============================================================
// init / initialized
// ============================================================

void VMMAllocator::init(int device_count) {
  std::call_once(init_flag_, [&] {
    device_count_ = device_count;
    devices_.resize(device_count);
    memory_fraction_.resize(device_count, 1.0);
    active_pools_.resize(device_count);

    for (int i = 0; i < device_count; ++i) {
      auto& ds = devices_[i];
      ds.pmm         = std::make_unique<PhysicalMemoryManager>(i);
      ds.small_alloc = std::make_unique<SmallAllocator>(
          i, *ds.pmm, kDefaultSmallVASize);
      ds.large_alloc = std::make_unique<LargeAllocator>(i, *ds.pmm);
      ds.initialized = true;
    }

    initialized_.store(true);
  });
}

bool VMMAllocator::initialized() {
  return initialized_.load();
}

// ============================================================
// Helper: device validation
// ============================================================

void VMMAllocator::checkDevice(int device) const {
  TORCH_CHECK(device >= 0 && device < device_count_,
              "zcg: invalid device ", device,
              " (device_count=", device_count_, ")");
}

int VMMAllocator::currentDevice() const {
  int dev = 0;
  cudaGetDevice(&dev);
  return dev;
}

// ============================================================
// Core allocation with OOM retry & pool isolation
// ============================================================

void* VMMAllocator::alloc_impl(size_t size, cudaStream_t stream) {
  if (size == 0) return nullptr;
  TORCH_CHECK(enabled_.load(), "zcg: allocator is disabled");

  int dev = currentDevice();
  checkDevice(dev);

  auto& ds = devices_[dev];
  bool is_large = (size >= threshold_.load(std::memory_order_relaxed));

  // Determine active pool id (for CUDA Graph isolation).
  // Only assign to the pool if the stream matches the pool's filter.
  MempoolId_t pool_id{0, 0};
  {
    std::lock_guard<std::mutex> lk(meta_mu_);
    auto& stack = active_pools_[dev];
    if (!stack.empty() && stack.back().filter &&
        stack.back().filter(stream)) {
      pool_id = stack.back().id;
    }
  }

  void* ptr = nullptr;

  // ---- first attempt -------------------------------------------
  try {
    processPendingFrees();
    if (is_large) {
      ptr = ds.large_alloc->allocate(size, stream);
    } else {
      ptr = ds.small_alloc->allocate(size, stream);
    }
  } catch (const std::exception&) {
    // ---- OOM retry: trim caches and try once more ----------------
    ds.num_alloc_retries.fetch_add(1, std::memory_order_relaxed);
    forceProcessPendingFrees(dev);
    ds.small_alloc->emptyCache();
    ds.large_alloc->emptyCache();
    ds.pmm->trim();

    try {
      if (is_large) {
        ptr = ds.large_alloc->allocate(size, stream);
      } else {
        ptr = ds.small_alloc->allocate(size, stream);
      }
    } catch (...) {
      ds.num_ooms.fetch_add(1, std::memory_order_relaxed);
      throw;
    }
  }

  // ---- stats ---------------------------------------------------
  auto& stats = is_large ? ds.large_stats : ds.small_stats;
  stats.recordAlloc(static_cast<int64_t>(size));

  // ---- metadata ------------------------------------------------
  {
    std::lock_guard<std::mutex> lk(meta_mu_);
    meta_[ptr] = AllocMeta{size, dev, is_large, stream, pool_id, {}};
    if (pool_id != MempoolId_t{0, 0}) {
      pool_allocs_[pool_id].insert(ptr);
    }
  }

  return ptr;
}

// ============================================================
// Free with event-based deferred recycling
// ============================================================

void VMMAllocator::free_impl(void* ptr) {
  if (!ptr) return;

  AllocMeta meta;
  {
    std::lock_guard<std::mutex> lk(meta_mu_);
    auto it = meta_.find(ptr);
    if (it == meta_.end()) return;  // double-free guard
    meta = std::move(it->second);
    meta_.erase(it);

    // Remove from pool tracking.
    if (meta.pool_id != MempoolId_t{0, 0}) {
      auto pit = pool_allocs_.find(meta.pool_id);
      if (pit != pool_allocs_.end()) {
        pit->second.erase(ptr);
      }
    }
  }

  // If the tensor was used on other streams (via recordStream),
  // we must defer the actual free until those streams complete.
  if (!meta.recorded_streams.empty()) {
    PendingFree pf;
    pf.ptr  = ptr;
    pf.meta = meta;

    // Record an event on each recorded stream.
    for (cudaStream_t s : meta.recorded_streams) {
      cudaEvent_t e = event_pool_.acquire();
      C10_CUDA_CHECK(cudaEventRecord(e, s));
      pf.events.push_back(e);
    }

    // Also record on the allocation stream if it was not already
    // among the recorded streams (the alloc stream may still have
    // in-flight work that references this memory).
    // NOTE: we always record, even for stream 0 (default stream),
    // because cudaEventRecord(e, 0) is valid and necessary.
    {
      bool already_recorded = false;
      for (cudaStream_t s : meta.recorded_streams) {
        if (s == meta.alloc_stream) { already_recorded = true; break; }
      }
      if (!already_recorded) {
        cudaEvent_t e = event_pool_.acquire();
        C10_CUDA_CHECK(cudaEventRecord(e, meta.alloc_stream));
        pf.events.push_back(e);
      }
    }

    std::lock_guard<std::mutex> lk(meta_mu_);
    pending_frees_.push_back(std::move(pf));
    return;
  }

  // No recorded streams → safe to free immediately.
  // The CUDA stream ordering guarantee ensures that subsequent
  // allocations on the same stream happen after this memory was
  // last used.
  doFree(ptr, meta);
}

// ============================================================
// Deferred-free processing
// ============================================================

void VMMAllocator::processPendingFrees() {
  // Quickly bail if nothing is pending (common fast path).
  {
    std::lock_guard<std::mutex> lk(meta_mu_);
    if (pending_frees_.empty()) return;
  }

  std::vector<PendingFree> ready;
  {
    std::lock_guard<std::mutex> lk(meta_mu_);
    auto it = pending_frees_.begin();
    while (it != pending_frees_.end()) {
      bool all_done = true;
      for (auto& ev : it->events) {
        if (cudaEventQuery(ev) != cudaSuccess) {
          all_done = false;
          break;
        }
      }
      if (all_done) {
        ready.push_back(std::move(*it));
        it = pending_frees_.erase(it);
      } else {
        ++it;
      }
    }
  }

  // Actually free outside the lock.
  for (auto& pf : ready) {
    for (auto& ev : pf.events) {
      event_pool_.release(ev);
    }
    doFree(pf.ptr, pf.meta);
  }
}

void VMMAllocator::forceProcessPendingFrees(int device) {
  // Step 1: Collect entries for this device under the lock (no sync here).
  std::vector<PendingFree> to_free;
  {
    std::lock_guard<std::mutex> lk(meta_mu_);
    auto it = pending_frees_.begin();
    while (it != pending_frees_.end()) {
      if (it->meta.device == device) {
        to_free.push_back(std::move(*it));
        it = pending_frees_.erase(it);
      } else {
        ++it;
      }
    }
  }

  // Step 2: Synchronize events OUTSIDE the lock (avoids blocking allocator).
  for (auto& pf : to_free) {
    for (auto& ev : pf.events) {
      cudaEventSynchronize(ev);
      event_pool_.release(ev);
    }
    doFree(pf.ptr, pf.meta);
  }
}

void VMMAllocator::doFree(void* ptr, const AllocMeta& meta) {
  auto& ds = devices_[meta.device];
  auto& stats = meta.is_large ? ds.large_stats : ds.small_stats;
  stats.recordFree(static_cast<int64_t>(meta.size));

  if (meta.is_large) {
    ds.large_alloc->free(ptr);
  } else {
    ds.small_alloc->free(ptr);
  }
}

// ============================================================
// c10::Allocator interface
// ============================================================

at::DataPtr VMMAllocator::allocate(size_t n) {
  int dev = currentDevice();
  if (n == 0) {
    return at::DataPtr(nullptr, at::Device(at::kCUDA, dev));
  }

  cudaStream_t stream = c10::cuda::getCurrentCUDAStream(dev).stream();
  void* ptr = alloc_impl(n, stream);

  return at::DataPtr(ptr, ptr, &vmm_delete,
                     at::Device(at::kCUDA, dev));
}

void VMMAllocator::copy_data(
    void* dest, const void* src, std::size_t count) const {
  C10_CUDA_CHECK(cudaMemcpy(dest, src, count, cudaMemcpyDefault));
}

// ============================================================
// Raw alloc / free
// ============================================================

void* VMMAllocator::raw_alloc(size_t nbytes) {
  int dev = currentDevice();
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream(dev).stream();
  return alloc_impl(nbytes, stream);
}

void* VMMAllocator::raw_alloc_with_stream(size_t nbytes, cudaStream_t stream) {
  return alloc_impl(nbytes, stream);
}

void VMMAllocator::raw_delete(void* ptr) {
  free_impl(ptr);
}

// ============================================================
// Stream awareness
// ============================================================

void VMMAllocator::recordStream(
    const at::DataPtr& ptr, c10::cuda::CUDAStream stream) {
  if (!ptr.get()) return;

  std::lock_guard<std::mutex> lk(meta_mu_);
  auto it = meta_.find(ptr.get());
  if (it == meta_.end()) return;

  cudaStream_t raw = stream.stream();
  auto& streams = it->second.recorded_streams;

  // Avoid duplicates.
  for (auto s : streams) {
    if (s == raw) return;
  }
  streams.push_back(raw);
}

// ============================================================
// Cache management
// ============================================================

void VMMAllocator::emptyCache(MempoolId_t /*id*/) {
  for (int d = 0; d < device_count_; ++d) {
    forceProcessPendingFrees(d);
    auto& ds = devices_[d];
    if (!ds.initialized) continue;
    ds.small_alloc->emptyCache();
    ds.large_alloc->emptyCache();
    ds.pmm->trim();
  }
}

// ============================================================
// Memory fraction
// ============================================================

double VMMAllocator::getMemoryFraction(c10::DeviceIndex device) {
  checkDevice(device);
  std::lock_guard<std::mutex> lk(meta_mu_);
  return memory_fraction_[device];
}

void VMMAllocator::setMemoryFraction(double fraction, c10::DeviceIndex device) {
  checkDevice(device);
  std::lock_guard<std::mutex> lk(meta_mu_);
  memory_fraction_[device] = fraction;
}

// ============================================================
// Statistics (full DeviceStats tracking)
// ============================================================

DeviceStats VMMAllocator::getDeviceStats(c10::DeviceIndex device) {
  checkDevice(device);
  DeviceStats stats{};

  auto& ds = devices_[device];
  if (!ds.initialized) return stats;

  // Index: 0 = AGGREGATE, 1 = SMALL_POOL, 2 = LARGE_POOL

  // ---- allocated_bytes (logical bytes in use) -------------------
  stats.allocated_bytes[1].current   = ds.small_stats.current_bytes.load();
  stats.allocated_bytes[1].peak      = ds.small_stats.peak_bytes.load();
  stats.allocated_bytes[1].allocated = ds.small_stats.accumulated_alloc.load();
  stats.allocated_bytes[1].freed     = ds.small_stats.accumulated_freed.load();

  stats.allocated_bytes[2].current   = ds.large_stats.current_bytes.load();
  stats.allocated_bytes[2].peak      = ds.large_stats.peak_bytes.load();
  stats.allocated_bytes[2].allocated = ds.large_stats.accumulated_alloc.load();
  stats.allocated_bytes[2].freed     = ds.large_stats.accumulated_freed.load();

  stats.allocated_bytes[0].current   =
      stats.allocated_bytes[1].current + stats.allocated_bytes[2].current;
  stats.allocated_bytes[0].peak      =
      stats.allocated_bytes[1].peak + stats.allocated_bytes[2].peak;
  stats.allocated_bytes[0].allocated =
      stats.allocated_bytes[1].allocated + stats.allocated_bytes[2].allocated;
  stats.allocated_bytes[0].freed     =
      stats.allocated_bytes[1].freed + stats.allocated_bytes[2].freed;

  // ---- reserved_bytes (physical memory held) --------------------
  size_t small_mapped  = ds.small_alloc->reservedBytes();
  size_t small_cached  = ds.pmm->smallPoolCached();
  size_t large_in_use  = ds.large_alloc->allocatedBytes();
  size_t large_cached  = ds.pmm->largePoolCached();

  stats.reserved_bytes[1].current =
      static_cast<int64_t>(small_mapped + small_cached);
  stats.reserved_bytes[2].current =
      static_cast<int64_t>(large_in_use + large_cached);
  stats.reserved_bytes[0].current =
      stats.reserved_bytes[1].current + stats.reserved_bytes[2].current;

  // ---- allocation count -----------------------------------------
  stats.allocation[1].current   =
      ds.small_stats.num_allocs.load() - ds.small_stats.num_frees.load();
  stats.allocation[1].allocated = ds.small_stats.num_allocs.load();
  stats.allocation[1].freed     = ds.small_stats.num_frees.load();

  stats.allocation[2].current   =
      ds.large_stats.num_allocs.load() - ds.large_stats.num_frees.load();
  stats.allocation[2].allocated = ds.large_stats.num_allocs.load();
  stats.allocation[2].freed     = ds.large_stats.num_frees.load();

  stats.allocation[0].current   =
      stats.allocation[1].current + stats.allocation[2].current;
  stats.allocation[0].allocated =
      stats.allocation[1].allocated + stats.allocation[2].allocated;
  stats.allocation[0].freed     =
      stats.allocation[1].freed + stats.allocation[2].freed;

  // ---- OOM counters ---------------------------------------------
  stats.num_alloc_retries = ds.num_alloc_retries.load();
  stats.num_ooms          = ds.num_ooms.load();

  return stats;
}

void VMMAllocator::resetAccumulatedStats(c10::DeviceIndex device) {
  checkDevice(device);
  auto& ds = devices_[device];
  if (!ds.initialized) return;
  ds.small_stats.resetAccumulated();
  ds.large_stats.resetAccumulated();
  ds.num_alloc_retries.store(0, std::memory_order_relaxed);
  ds.num_ooms.store(0, std::memory_order_relaxed);
}

void VMMAllocator::resetPeakStats(c10::DeviceIndex device) {
  checkDevice(device);
  auto& ds = devices_[device];
  if (!ds.initialized) return;
  ds.small_stats.resetPeak();
  ds.large_stats.resetPeak();
}

// ============================================================
// Introspection
// ============================================================

SnapshotInfo VMMAllocator::snapshot(MempoolId_t /*id*/, bool /*traces*/) {
  return SnapshotInfo{};
}

void VMMAllocator::cacheInfo(c10::DeviceIndex device, size_t* largest) {
  if (largest) {
    *largest = kDefaultSmallVASize;
  }
}

void* VMMAllocator::getBaseAllocation(void* ptr, size_t* out_size) {
  std::lock_guard<std::mutex> lk(meta_mu_);
  auto it = meta_.find(ptr);
  if (out_size) {
    *out_size = (it != meta_.end()) ? it->second.size : 0;
  }
  return ptr;
}

std::vector<StreamSegmentSize>
VMMAllocator::getExpandableSegmentSizes(c10::DeviceIndex /*device*/) {
  return {};
}

// ============================================================
// Enable / disable
// ============================================================

void VMMAllocator::enable(bool value) {
  enabled_.store(value);
}

bool VMMAllocator::isEnabled() const {
  return enabled_.load();
}

// ============================================================
// Pool isolation (for CUDA Graphs)
// ============================================================

void VMMAllocator::beginAllocateToPool(
    c10::DeviceIndex device, MempoolId_t id,
    std::function<bool(cudaStream_t)> filter) {
  checkDevice(device);
  std::lock_guard<std::mutex> lk(meta_mu_);
  active_pools_[device].push_back(
      ActivePoolEntry{id, std::move(filter)});
}

void VMMAllocator::endAllocateToPool(
    c10::DeviceIndex device, MempoolId_t id) {
  checkDevice(device);
  std::lock_guard<std::mutex> lk(meta_mu_);
  auto& stack = active_pools_[device];
  // Remove the most recent matching entry (stack semantics).
  for (auto it = stack.rbegin(); it != stack.rend(); ++it) {
    if (it->id == id) {
      stack.erase(std::next(it).base());
      return;
    }
  }
}

void VMMAllocator::releasePool(
    c10::DeviceIndex device, MempoolId_t id) {
  checkDevice(device);

  std::vector<void*> ptrs_to_free;
  std::vector<AllocMeta> metas_to_free;
  std::vector<PendingFree> pool_pending;

  {
    std::lock_guard<std::mutex> lk(meta_mu_);
    auto pit = pool_allocs_.find(id);
    if (pit == pool_allocs_.end()) return;

    for (void* ptr : pit->second) {
      auto mit = meta_.find(ptr);
      if (mit != meta_.end()) {
        ptrs_to_free.push_back(ptr);
        metas_to_free.push_back(mit->second);
        meta_.erase(mit);
      }
    }
    pool_allocs_.erase(pit);

    // Collect pending frees belonging to this pool (move out of list).
    auto pfit = pending_frees_.begin();
    while (pfit != pending_frees_.end()) {
      if (pfit->meta.pool_id == id) {
        pool_pending.push_back(std::move(*pfit));
        pfit = pending_frees_.erase(pfit);
      } else {
        ++pfit;
      }
    }
  }

  // Synchronize and free pending entries OUTSIDE the lock.
  for (auto& pf : pool_pending) {
    for (auto& ev : pf.events) {
      cudaEventSynchronize(ev);
      event_pool_.release(ev);
    }
    ptrs_to_free.push_back(pf.ptr);
    metas_to_free.push_back(pf.meta);
  }

  for (size_t i = 0; i < ptrs_to_free.size(); ++i) {
    doFree(ptrs_to_free[i], metas_to_free[i]);
  }
}

// ============================================================
// IPC – not yet implemented
// ============================================================

ShareableHandle VMMAllocator::shareIpcHandle(void* /*ptr*/) {
  TORCH_CHECK(false, "VMMAllocator: shareIpcHandle not yet implemented");
}

std::shared_ptr<void> VMMAllocator::getIpcDevPtr(std::string /*handle*/) {
  TORCH_CHECK(false, "VMMAllocator: getIpcDevPtr not yet implemented");
}

// ============================================================
// Tracing / observers – stored but not actively used yet
// ============================================================

void VMMAllocator::recordHistory(
    bool /*enabled*/, CreateContextFn /*ctx*/, size_t /*max_entries*/,
    RecordContext /*when*/, bool /*clear*/,
    const std::vector<std::string>& /*skip*/) {}

void VMMAllocator::attachOutOfMemoryObserver(OutOfMemoryObserver /*obs*/) {}
void VMMAllocator::attachOomRejectionObserver(OomRejectionObserver /*obs*/) {}
void VMMAllocator::attachAllocatorTraceTracker(AllocatorTraceTracker /*t*/) {}

// ============================================================
// Peer access / memcpy
// ============================================================

void VMMAllocator::enablePeerAccess(
    c10::DeviceIndex dev, c10::DeviceIndex dev_to_access) {
  checkDevice(dev);
  checkDevice(dev_to_access);

  // 1. Enable CUDA runtime peer access.
  {
    c10::cuda::CUDAGuard guard(dev);
    cudaError_t err = cudaDeviceEnablePeerAccess(dev_to_access, 0);
    if (err != cudaSuccess && err != cudaErrorPeerAccessAlreadyEnabled) {
      C10_CUDA_CHECK(err);
    }
  }

  // 2. Register `dev` as a peer on the target device's sub-allocators
  //    so that all current and future VMM mappings on `dev_to_access`
  //    include a CUmemAccessDesc for `dev`.
  auto& target = devices_[dev_to_access];
  {
    std::lock_guard<std::mutex> lk(target.peers_mu);
    if (target.peer_devices.count(dev)) return;  // already registered
    target.peer_devices.insert(dev);
  }

  target.small_alloc->addPeerDevice(dev);
  target.large_alloc->addPeerDevice(dev);
}

cudaError_t VMMAllocator::memcpyAsync(
    void* dst, int /*dstDev*/,
    const void* src, int /*srcDev*/,
    size_t count, cudaStream_t stream, bool /*p2p*/) {
  return cudaMemcpyAsync(dst, src, count, cudaMemcpyDefault, stream);
}

// ============================================================
// Checkpoint – not yet implemented
// ============================================================

std::shared_ptr<AllocatorState>
VMMAllocator::getCheckpointState(c10::DeviceIndex /*dev*/, MempoolId_t /*id*/) {
  TORCH_CHECK(false, "VMMAllocator: getCheckpointState not yet implemented");
}

CheckpointDelta VMMAllocator::setCheckpointPoolState(
    c10::DeviceIndex /*dev*/, std::shared_ptr<AllocatorState> /*pps*/) {
  TORCH_CHECK(false, "VMMAllocator: setCheckpointPoolState not yet implemented");
}

// ============================================================
// Name
// ============================================================

std::string VMMAllocator::name() {
  return "VMMAllocator";
}

// ============================================================
// Configuration
// ============================================================

void VMMAllocator::setThreshold(size_t bytes) {
  threshold_.store(bytes, std::memory_order_relaxed);
}

const PerDeviceState& VMMAllocator::deviceState(int device) const {
  checkDevice(device);
  return devices_[device];
}

} // namespace zcg
