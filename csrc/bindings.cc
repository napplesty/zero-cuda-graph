// ============================================================
// bindings.cc – pybind11 module definition for zcg_c
// ============================================================

#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

#include "vmm_allocator.h"
#include "substitution.h"

namespace py = pybind11;

namespace zcg {

// ---- Python-friendly wrappers ---------------------------------

static void py_register() {
  registerVMMAllocator();
}

static void py_empty_cache() {
  auto* alloc = getVMMAllocator();
  TORCH_CHECK(alloc, "zcg: allocator not registered");
  alloc->emptyCache();
}

static void py_set_threshold(int64_t bytes) {
  auto* alloc = getVMMAllocator();
  TORCH_CHECK(alloc, "zcg: allocator not registered");
  TORCH_CHECK(bytes > 0, "zcg: threshold must be positive");
  alloc->setThreshold(static_cast<size_t>(bytes));
}

static int64_t py_get_threshold() {
  auto* alloc = getVMMAllocator();
  TORCH_CHECK(alloc, "zcg: allocator not registered");
  return static_cast<int64_t>(alloc->threshold());
}

/// Return a dict with memory stats for the given device.
static py::dict py_get_memory_stats(int device) {
  auto* alloc = getVMMAllocator();
  TORCH_CHECK(alloc, "zcg: allocator not registered");

  auto& ds = alloc->deviceState(device);
  TORCH_CHECK(ds.initialized,
              "zcg: device ", device, " is not initialized");
  py::dict result;

  // Small pool
  py::dict small;
  small["allocated_bytes"] = ds.small_alloc->allocatedBytes();
  small["reserved_bytes"]  = ds.small_alloc->reservedBytes();
  small["va_range_bytes"]  = ds.small_alloc->vaRangeSize();
  small["physical_cached"] = ds.pmm->smallPoolCached();
  small["physical_in_use"] = ds.pmm->smallPoolInUse();

  // Phase 4 stats
  small["current_bytes"]       = ds.small_stats.current_bytes.load();
  small["peak_bytes"]          = ds.small_stats.peak_bytes.load();
  small["accumulated_alloc"]   = ds.small_stats.accumulated_alloc.load();
  small["accumulated_freed"]   = ds.small_stats.accumulated_freed.load();
  small["num_allocs"]          = ds.small_stats.num_allocs.load();
  small["num_frees"]           = ds.small_stats.num_frees.load();
  result["small_pool"] = small;

  // Large pool
  py::dict large;
  large["allocated_bytes"] = ds.large_alloc->allocatedBytes();
  large["physical_cached"] = ds.pmm->largePoolCached();
  large["physical_in_use"] = ds.pmm->largePoolInUse();

  large["current_bytes"]       = ds.large_stats.current_bytes.load();
  large["peak_bytes"]          = ds.large_stats.peak_bytes.load();
  large["accumulated_alloc"]   = ds.large_stats.accumulated_alloc.load();
  large["accumulated_freed"]   = ds.large_stats.accumulated_freed.load();
  large["num_allocs"]          = ds.large_stats.num_allocs.load();
  large["num_frees"]           = ds.large_stats.num_frees.load();
  result["large_pool"] = large;

  // Aggregate
  result["total_allocated"] =
      ds.small_alloc->allocatedBytes() + ds.large_alloc->allocatedBytes();
  result["total_physical"] = ds.pmm->totalPhysicalAllocated();
  result["total_physical_cached"] = ds.pmm->totalPhysicalCached();
  result["threshold"] = alloc->threshold();

  // OOM counters
  result["num_alloc_retries"] = ds.num_alloc_retries.load();
  result["num_ooms"]          = ds.num_ooms.load();

  return result;
}

/// Replace the data pointer of a tensor.
static void py_replace_data_ptr(at::Tensor& tensor, int64_t new_ptr) {
  replaceDataPtr(tensor, reinterpret_cast<void*>(new_ptr), nullptr);
}

/// VMM-level remap.
static void py_vmm_remap(int64_t va, int64_t size,
                          std::vector<int64_t> old_handles,
                          std::vector<int64_t> new_handles,
                          std::vector<int64_t> new_sizes,
                          int device,
                          std::vector<int> peer_devices) {
  std::vector<CUmemGenericAllocationHandle> old_h, new_h;
  std::vector<size_t> sizes;
  for (auto h : old_handles) old_h.push_back(static_cast<CUmemGenericAllocationHandle>(h));
  for (auto h : new_handles) new_h.push_back(static_cast<CUmemGenericAllocationHandle>(h));
  for (auto s : new_sizes) sizes.push_back(static_cast<size_t>(s));
  vmmRemap(static_cast<CUdeviceptr>(va),
           static_cast<size_t>(size),
           old_h, new_h, sizes, device, peer_devices);
}

/// Enable peer access: allow `dev` to read/write memory on `dev_to_access`.
static void py_enable_peer_access(int dev, int dev_to_access) {
  auto* alloc = getVMMAllocator();
  TORCH_CHECK(alloc, "zcg: allocator not registered");
  alloc->enablePeerAccess(
      static_cast<c10::DeviceIndex>(dev),
      static_cast<c10::DeviceIndex>(dev_to_access));
}

/// Reset accumulated (lifetime) statistics for a device.
static void py_reset_accumulated_stats(int device) {
  auto* alloc = getVMMAllocator();
  TORCH_CHECK(alloc, "zcg: allocator not registered");
  alloc->resetAccumulatedStats(static_cast<c10::DeviceIndex>(device));
}

/// Reset peak statistics for a device.
static void py_reset_peak_stats(int device) {
  auto* alloc = getVMMAllocator();
  TORCH_CHECK(alloc, "zcg: allocator not registered");
  alloc->resetPeakStats(static_cast<c10::DeviceIndex>(device));
}

} // namespace zcg

// ============================================================
// Module definition
// ============================================================

PYBIND11_MODULE(zcg_c, m) {
  m.doc() = "zero-cuda-graph: VMM-based CUDA allocator for PyTorch";

  // ---- allocator lifecycle ------------------------------------
  m.def("register_vmm_allocator", &zcg::py_register,
        "Register the VMMAllocator as the global CUDA allocator.\n"
        "Must be called before any CUDA tensor allocation.");

  m.def("empty_cache", &zcg::py_empty_cache,
        "Release all cached physical memory back to the driver.");

  // ---- configuration ------------------------------------------
  m.def("set_threshold", &zcg::py_set_threshold,
        py::arg("bytes"),
        "Set the small/large allocation routing threshold (bytes).");

  m.def("get_threshold", &zcg::py_get_threshold,
        "Get the current routing threshold (bytes).");

  // ---- diagnostics --------------------------------------------
  m.def("get_memory_stats", &zcg::py_get_memory_stats,
        py::arg("device") = 0,
        "Return a dict of memory statistics for the given device.");

  m.def("reset_accumulated_stats", &zcg::py_reset_accumulated_stats,
        py::arg("device") = 0,
        "Reset accumulated (lifetime) allocation statistics.");

  m.def("reset_peak_stats", &zcg::py_reset_peak_stats,
        py::arg("device") = 0,
        "Reset peak allocation statistics to the current level.");

  // ---- tensor pointer manipulation ----------------------------
  m.def("replace_data_ptr", &zcg::py_replace_data_ptr,
        py::arg("tensor"), py::arg("new_ptr"),
        "Replace the underlying data pointer of a tensor.\n"
        "The new_ptr must be a valid device address (as int).\n"
        "The old DataPtr is released.");

  // ---- VMM remap ----------------------------------------------
  m.def("vmm_remap", &zcg::py_vmm_remap,
        py::arg("va"), py::arg("size"),
        py::arg("old_handles"), py::arg("new_handles"),
        py::arg("new_sizes"),
        py::arg("device") = 0,
        py::arg("peer_devices") = std::vector<int>{},
        "Remap physical memory at a virtual address.\n"
        "Unmap old handles and map new handles at the same VA.\n"
        "new_sizes must contain the byte size of each new handle.\n"
        "peer_devices: list of device IDs that should also have access.");

  // ---- peer access --------------------------------------------
  m.def("enable_peer_access", &zcg::py_enable_peer_access,
        py::arg("dev"), py::arg("dev_to_access"),
        "Enable peer access: allow device `dev` to read/write\n"
        "memory that resides on device `dev_to_access`.\n"
        "Updates both CUDA runtime P2P and VMM access descriptors.");
}
