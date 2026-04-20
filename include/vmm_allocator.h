#pragma once
// ============================================================
// vmm_allocator.h – the top-level CUDAAllocator implementation
// that plugs into PyTorch via changeCurrentAllocator().
//
// Phase 4 features:
//   • Event-based stream-aware deferred recycling
//   • OOM retry with cache trimming
//   • CUDA Graph pool isolation
//   • Full DeviceStats tracking (peak / accumulated)
//   • Multi-GPU peer access
// ============================================================

#include "small_allocator.h"
#include "large_allocator.h"

#include <c10/core/Allocator.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAStream.h>

#include <atomic>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <cuda_runtime.h>

namespace zcg {

// Convenience aliases into the PyTorch namespace
namespace cca = c10::cuda::CUDACachingAllocator;
using CUDAAllocator       = cca::CUDAAllocator;
using DeviceStats          = c10::CachingDeviceAllocator::DeviceStats;
using SnapshotInfo         = cca::SnapshotInfo;
using ShareableHandle      = cca::ShareableHandle;
using AllocatorState       = cca::AllocatorState;
using CheckpointDelta      = cca::CheckpointDelta;
using StreamSegmentSize    = cca::StreamSegmentSize;
using OutOfMemoryObserver  = cca::OutOfMemoryObserver;
using OomRejectionObserver = cca::OomRejectionObserver;
using AllocatorTraceTracker= cca::AllocatorTraceTracker;
using MempoolId_t          = c10::MempoolId_t;
using RecordContext        = c10::CachingDeviceAllocator::RecordContext;
using CreateContextFn      = c10::CachingDeviceAllocator::CreateContextFn;
using GatheredContext      = c10::CachingDeviceAllocator::GatheredContext;

// ============================================================
// EventPool – reusable pool of CUDA events (avoids create/destroy churn)
// ============================================================

class EventPool {
 public:
  cudaEvent_t acquire();
  void        release(cudaEvent_t e);
  ~EventPool();
 private:
  std::mutex mu_;
  std::vector<cudaEvent_t> pool_;
};

// ============================================================
// PoolStats – atomic counters for one sub-pool (small or large)
// ============================================================

struct PoolStats {
  std::atomic<int64_t> current_bytes{0};
  std::atomic<int64_t> peak_bytes{0};
  std::atomic<int64_t> accumulated_alloc{0};
  std::atomic<int64_t> accumulated_freed{0};
  std::atomic<int64_t> num_allocs{0};
  std::atomic<int64_t> num_frees{0};

  void recordAlloc(int64_t n);
  void recordFree(int64_t n);
  void resetAccumulated();
  void resetPeak();
};

// ============================================================
// Per-device state
// ============================================================

struct PerDeviceState {
  std::unique_ptr<PhysicalMemoryManager> pmm;
  std::unique_ptr<SmallAllocator>        small_alloc;
  std::unique_ptr<LargeAllocator>        large_alloc;

  // Statistics
  PoolStats small_stats;
  PoolStats large_stats;
  std::atomic<int64_t> num_alloc_retries{0};
  std::atomic<int64_t> num_ooms{0};

  // Peer devices that have been granted access to this device's memory
  std::mutex peers_mu;
  std::unordered_set<int> peer_devices;

  bool initialized = false;
};

// ============================================================
// Allocation metadata kept by the top-level allocator
// ============================================================

struct AllocMeta {
  size_t       size;
  int          device;
  bool         is_large;
  cudaStream_t alloc_stream;
  MempoolId_t  pool_id{0, 0};                    // {0,0} = no pool
  std::vector<cudaStream_t> recorded_streams;     // from recordStream()
};

// ============================================================
// Pending free entry (deferred until stream events complete)
// ============================================================

struct PendingFree {
  void*                   ptr;
  AllocMeta               meta;
  std::vector<cudaEvent_t> events;   // one per recorded stream
};

// ============================================================
// Active pool entry (for CUDA Graph capture isolation)
// ============================================================

struct ActivePoolEntry {
  MempoolId_t id;
  std::function<bool(cudaStream_t)> filter;
};

// ============================================================
// VMMAllocator
// ============================================================

class VMMAllocator : public CUDAAllocator {
 public:
  VMMAllocator();
  ~VMMAllocator() override;

  // ---- core allocation (c10::Allocator) -------------------------
  at::DataPtr allocate(size_t n) override;
  at::DeleterFnPtr raw_deleter() const override;
  void copy_data(
      void* dest, const void* src, std::size_t count) const override;

  // ---- CUDAAllocator: raw alloc/free ----------------------------
  void* raw_alloc(size_t nbytes) override;
  void* raw_alloc_with_stream(size_t nbytes, cudaStream_t stream) override;
  void  raw_delete(void* ptr) override;

  // ---- lifecycle ------------------------------------------------
  void init(int device_count) override;
  bool initialized() override;

  // ---- stream awareness -----------------------------------------
  void recordStream(const at::DataPtr& ptr,
                    c10::cuda::CUDAStream stream) override;

  // ---- cache management -----------------------------------------
  void emptyCache(MempoolId_t id = {0, 0}) override;

  // ---- memory limits --------------------------------------------
  double getMemoryFraction(c10::DeviceIndex device) override;
  void   setMemoryFraction(double fraction,
                           c10::DeviceIndex device) override;

  // ---- statistics -----------------------------------------------
  DeviceStats getDeviceStats(c10::DeviceIndex device) override;
  void resetAccumulatedStats(c10::DeviceIndex device) override;
  void resetPeakStats(c10::DeviceIndex device) override;

  // ---- introspection --------------------------------------------
  SnapshotInfo snapshot(MempoolId_t id = {0, 0},
                        bool include_traces = true) override;
  void  cacheInfo(c10::DeviceIndex device, size_t* largest) override;
  void* getBaseAllocation(void* ptr, size_t* size) override;
  std::vector<StreamSegmentSize>
        getExpandableSegmentSizes(c10::DeviceIndex device) override;

  // ---- enable/disable -------------------------------------------
  void enable(bool value) override;
  bool isEnabled() const override;

  // ---- pool isolation -------------------------------------------
  void beginAllocateToPool(c10::DeviceIndex device, MempoolId_t id,
                           std::function<bool(cudaStream_t)> filter) override;
  void endAllocateToPool(c10::DeviceIndex device,
                         MempoolId_t id) override;
  void releasePool(c10::DeviceIndex device, MempoolId_t id) override;

  // ---- IPC (stubbed) --------------------------------------------
  ShareableHandle shareIpcHandle(void* ptr) override;
  std::shared_ptr<void> getIpcDevPtr(std::string handle) override;

  // ---- tracing / observers (stubbed) ----------------------------
  void recordHistory(bool enabled, CreateContextFn ctx_creator,
                     size_t alloc_trace_max_entries,
                     RecordContext when, bool clear,
                     const std::vector<std::string>& skip) override;
  void attachOutOfMemoryObserver(OutOfMemoryObserver obs) override;
  void attachOomRejectionObserver(OomRejectionObserver obs) override;
  void attachAllocatorTraceTracker(AllocatorTraceTracker t) override;

  // ---- peer access / memcpy -------------------------------------
  void enablePeerAccess(c10::DeviceIndex dev,
                        c10::DeviceIndex dev_to_access) override;
  cudaError_t memcpyAsync(void* dst, int dstDev,
                          const void* src, int srcDev,
                          size_t count, cudaStream_t stream,
                          bool p2p_enabled) override;

  // ---- checkpoint (stubbed) -------------------------------------
  std::shared_ptr<AllocatorState> getCheckpointState(
      c10::DeviceIndex device, MempoolId_t id) override;
  CheckpointDelta setCheckpointPoolState(
      c10::DeviceIndex device,
      std::shared_ptr<AllocatorState> pps) override;

  // ---- name -----------------------------------------------------
  std::string name() override;

  // ---- internal helpers exposed for bindings --------------------
  void  free_impl(void* ptr);
  const PerDeviceState& deviceState(int device) const;

  // Configuration
  void   setThreshold(size_t bytes);
  size_t threshold() const { return threshold_.load(std::memory_order_relaxed); }

 private:
  std::atomic<bool> initialized_{false};
  std::once_flag init_flag_;
  std::atomic<bool> enabled_{true};
  int device_count_ = 0;

  std::atomic<size_t> threshold_{kDefaultThreshold};
  std::vector<double> memory_fraction_;  // per device, guarded by meta_mu_

  std::vector<PerDeviceState> devices_;

  // Global allocation metadata (all devices)
  mutable std::mutex meta_mu_;
  std::unordered_map<void*, AllocMeta> meta_;

  // Event-based deferred free
  EventPool event_pool_;
  std::vector<PendingFree> pending_frees_;  // guarded by meta_mu_

  // Pool isolation: per-device stack of active pools
  std::vector<std::vector<ActivePoolEntry>> active_pools_;  // [device]
  // Pool id → set of live allocation ptrs
  std::map<MempoolId_t, std::unordered_set<void*>> pool_allocs_;

  // Internal
  void*  alloc_impl(size_t size, cudaStream_t stream);
  int    currentDevice() const;
  void   processPendingFrees();
  void   forceProcessPendingFrees(int device);
  void   doFree(void* ptr, const AllocMeta& meta);
  void   checkDevice(int device) const;
};

// ============================================================
// Registration helpers
// ============================================================

void registerVMMAllocator();
VMMAllocator* getVMMAllocator();

} // namespace zcg
