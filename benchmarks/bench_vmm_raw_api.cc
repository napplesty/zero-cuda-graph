// ============================================================
// bench_vmm_raw_api.cc -- Standalone benchmark for CUDA VMM API
// call latencies.
//
// Measures cuMemAddressReserve, cuMemCreate, cuMemMap,
// cuMemSetAccess, cuMemUnmap, cuMemRelease, cuMemAddressFree,
// and the full reserve->create->map->access pipeline.
//
// Build:
//   g++ -std=c++17 -O2 -I${CUDA_HOME}/include \
//       bench_vmm_raw_api.cc -o bench_vmm_raw_api -ldl
//
// Run:
//   ./bench_vmm_raw_api [--iters N] [--device D] [--csv]
// ============================================================

#include <cuda.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <numeric>
#include <string>
#include <vector>

// ============================================================
// Driver API function typedefs
// ============================================================

using pfn_cuInit = CUresult (*)(unsigned int);
using pfn_cuDeviceGet = CUresult (*)(CUdevice*, int);
using pfn_cuCtxCreate = CUresult (*)(CUcontext*, unsigned int, CUdevice);
using pfn_cuCtxDestroy = CUresult (*)(CUcontext);
using pfn_cuDeviceGetName = CUresult (*)(char*, int, CUdevice);
using pfn_cuMemAddressReserve =
    CUresult (*)(CUdeviceptr*, size_t, size_t, CUdeviceptr, unsigned long long);
using pfn_cuMemAddressFree = CUresult (*)(CUdeviceptr, size_t);
using pfn_cuMemCreate =
    CUresult (*)(CUmemGenericAllocationHandle*, size_t,
                 const CUmemAllocationProp*, unsigned long long);
using pfn_cuMemRelease = CUresult (*)(CUmemGenericAllocationHandle);
using pfn_cuMemMap =
    CUresult (*)(CUdeviceptr, size_t, size_t,
                 CUmemGenericAllocationHandle, unsigned long long);
using pfn_cuMemUnmap = CUresult (*)(CUdeviceptr, size_t);
using pfn_cuMemSetAccess =
    CUresult (*)(CUdeviceptr, size_t, const CUmemAccessDesc*, size_t);
using pfn_cuMemGetAllocationGranularity =
    CUresult (*)(size_t*, const CUmemAllocationProp*, CUmemAllocationGranularity_flags);

// ============================================================
// Global function pointers
// ============================================================

static pfn_cuInit                       fn_cuInit;
static pfn_cuDeviceGet                  fn_cuDeviceGet;
static pfn_cuCtxCreate                  fn_cuCtxCreate;
static pfn_cuCtxDestroy                 fn_cuCtxDestroy;
static pfn_cuDeviceGetName              fn_cuDeviceGetName;
static pfn_cuMemAddressReserve          fn_cuMemAddressReserve;
static pfn_cuMemAddressFree             fn_cuMemAddressFree;
static pfn_cuMemCreate                  fn_cuMemCreate;
static pfn_cuMemRelease                 fn_cuMemRelease;
static pfn_cuMemMap                     fn_cuMemMap;
static pfn_cuMemUnmap                   fn_cuMemUnmap;
static pfn_cuMemSetAccess               fn_cuMemSetAccess;
static pfn_cuMemGetAllocationGranularity fn_cuMemGetAllocationGranularity;

#define LOAD_SYM(handle, name)                                           \
  do {                                                                   \
    fn_##name = reinterpret_cast<pfn_##name>(dlsym(handle, #name));      \
    if (!fn_##name) {                                                    \
      fprintf(stderr, "Failed to load %s: %s\n", #name, dlerror());     \
      return false;                                                      \
    }                                                                    \
  } while (0)

#define CU_CHECK(expr)                                                   \
  do {                                                                   \
    CUresult _r = (expr);                                                \
    if (_r != CUDA_SUCCESS) {                                            \
      const char* _msg = nullptr;                                        \
      fprintf(stderr, "CUDA error %d at %s:%d\n", _r, __FILE__, __LINE__); \
      exit(1);                                                           \
    }                                                                    \
  } while (0)

// ============================================================
// Load driver library
// ============================================================

static void* g_lib = nullptr;

static bool loadDriver() {
  const char* paths[] = {"libcuda.so.1", "libcuda.so", "libcuda.dylib", nullptr};
  for (int i = 0; paths[i]; ++i) {
    g_lib = dlopen(paths[i], RTLD_LAZY);
    if (g_lib) break;
  }
  if (!g_lib) {
    fprintf(stderr, "Cannot dlopen libcuda: %s\n", dlerror());
    return false;
  }

  LOAD_SYM(g_lib, cuInit);
  LOAD_SYM(g_lib, cuDeviceGet);
  LOAD_SYM(g_lib, cuCtxCreate);
  LOAD_SYM(g_lib, cuCtxDestroy);
  LOAD_SYM(g_lib, cuDeviceGetName);
  LOAD_SYM(g_lib, cuMemAddressReserve);
  LOAD_SYM(g_lib, cuMemAddressFree);
  LOAD_SYM(g_lib, cuMemCreate);
  LOAD_SYM(g_lib, cuMemRelease);
  LOAD_SYM(g_lib, cuMemMap);
  LOAD_SYM(g_lib, cuMemUnmap);
  LOAD_SYM(g_lib, cuMemSetAccess);
  LOAD_SYM(g_lib, cuMemGetAllocationGranularity);
  return true;
}

// ============================================================
// Timing utilities
// ============================================================

using Clock = std::chrono::high_resolution_clock;

struct TimingStats {
  double min_us;
  double max_us;
  double avg_us;
  double median_us;
  double p99_us;
};

static TimingStats computeStats(std::vector<double>& samples) {
  std::sort(samples.begin(), samples.end());
  size_t n = samples.size();
  double sum = std::accumulate(samples.begin(), samples.end(), 0.0);

  TimingStats s;
  s.min_us    = samples.front();
  s.max_us    = samples.back();
  s.avg_us    = sum / n;
  s.median_us = (n % 2 == 0)
                    ? (samples[n / 2 - 1] + samples[n / 2]) / 2.0
                    : samples[n / 2];
  s.p99_us    = samples[std::min(n - 1, (size_t)(n * 0.99))];
  return s;
}

// ============================================================
// Benchmark kernels
// ============================================================

static int g_device = 0;
static CUmemAllocationProp g_prop{};
static CUmemAccessDesc g_access_desc{};

static void setupProp() {
  memset(&g_prop, 0, sizeof(g_prop));
  g_prop.type          = CU_MEM_ALLOCATION_TYPE_PINNED;
  g_prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  g_prop.location.id   = g_device;

  memset(&g_access_desc, 0, sizeof(g_access_desc));
  g_access_desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  g_access_desc.location.id   = g_device;
  g_access_desc.flags         = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
}

// Benchmark: cuMemAddressReserve + cuMemAddressFree
static void benchReserveFree(size_t size, int iters,
                             std::vector<double>& reserve_us,
                             std::vector<double>& free_us) {
  reserve_us.resize(iters);
  free_us.resize(iters);

  for (int i = 0; i < iters; ++i) {
    CUdeviceptr va = 0;

    auto t0 = Clock::now();
    CU_CHECK(fn_cuMemAddressReserve(&va, size, 2ULL << 20, 0, 0));
    auto t1 = Clock::now();

    auto t2 = Clock::now();
    CU_CHECK(fn_cuMemAddressFree(va, size));
    auto t3 = Clock::now();

    reserve_us[i] =
        std::chrono::duration<double, std::micro>(t1 - t0).count();
    free_us[i] =
        std::chrono::duration<double, std::micro>(t3 - t2).count();
  }
}

// Benchmark: cuMemCreate + cuMemRelease
static void benchCreateRelease(size_t size, int iters,
                               std::vector<double>& create_us,
                               std::vector<double>& release_us) {
  create_us.resize(iters);
  release_us.resize(iters);

  for (int i = 0; i < iters; ++i) {
    CUmemGenericAllocationHandle h = 0;

    auto t0 = Clock::now();
    CU_CHECK(fn_cuMemCreate(&h, size, &g_prop, 0));
    auto t1 = Clock::now();

    auto t2 = Clock::now();
    CU_CHECK(fn_cuMemRelease(h));
    auto t3 = Clock::now();

    create_us[i] =
        std::chrono::duration<double, std::micro>(t1 - t0).count();
    release_us[i] =
        std::chrono::duration<double, std::micro>(t3 - t2).count();
  }
}

// Benchmark: cuMemMap + cuMemSetAccess + cuMemUnmap
// (requires a pre-reserved VA and pre-created handle)
static void benchMapAccessUnmap(size_t size, int iters,
                                std::vector<double>& map_us,
                                std::vector<double>& access_us,
                                std::vector<double>& unmap_us) {
  map_us.resize(iters);
  access_us.resize(iters);
  unmap_us.resize(iters);

  // Reserve VA and create physical handle once.
  CUdeviceptr va = 0;
  CU_CHECK(fn_cuMemAddressReserve(&va, size, 2ULL << 20, 0, 0));
  CUmemGenericAllocationHandle h = 0;
  CU_CHECK(fn_cuMemCreate(&h, size, &g_prop, 0));

  for (int i = 0; i < iters; ++i) {
    auto t0 = Clock::now();
    CU_CHECK(fn_cuMemMap(va, size, 0, h, 0));
    auto t1 = Clock::now();

    auto t2 = Clock::now();
    CU_CHECK(fn_cuMemSetAccess(va, size, &g_access_desc, 1));
    auto t3 = Clock::now();

    auto t4 = Clock::now();
    CU_CHECK(fn_cuMemUnmap(va, size));
    auto t5 = Clock::now();

    map_us[i] =
        std::chrono::duration<double, std::micro>(t1 - t0).count();
    access_us[i] =
        std::chrono::duration<double, std::micro>(t3 - t2).count();
    unmap_us[i] =
        std::chrono::duration<double, std::micro>(t5 - t4).count();
  }

  CU_CHECK(fn_cuMemRelease(h));
  CU_CHECK(fn_cuMemAddressFree(va, size));
}

// Benchmark: full pipeline (reserve -> create -> map -> access ->
//                           unmap -> release -> free)
static void benchFullPipeline(size_t size, int iters,
                              std::vector<double>& full_us) {
  full_us.resize(iters);

  for (int i = 0; i < iters; ++i) {
    CUdeviceptr va = 0;
    CUmemGenericAllocationHandle h = 0;

    auto t0 = Clock::now();
    CU_CHECK(fn_cuMemAddressReserve(&va, size, 2ULL << 20, 0, 0));
    CU_CHECK(fn_cuMemCreate(&h, size, &g_prop, 0));
    CU_CHECK(fn_cuMemMap(va, size, 0, h, 0));
    CU_CHECK(fn_cuMemSetAccess(va, size, &g_access_desc, 1));
    auto t1 = Clock::now();

    // Teardown (not timed).
    CU_CHECK(fn_cuMemUnmap(va, size));
    CU_CHECK(fn_cuMemRelease(h));
    CU_CHECK(fn_cuMemAddressFree(va, size));

    full_us[i] =
        std::chrono::duration<double, std::micro>(t1 - t0).count();
  }
}

// ============================================================
// Reporting
// ============================================================

static bool g_csv = false;

static void printHeader() {
  if (g_csv) {
    printf("operation,size_mb,min_us,avg_us,median_us,p99_us,max_us\n");
  }
}

static void printRow(const char* op, size_t size, TimingStats& s) {
  double mb = size / (1024.0 * 1024.0);
  if (g_csv) {
    printf("%s,%.0f,%.2f,%.2f,%.2f,%.2f,%.2f\n",
           op, mb, s.min_us, s.avg_us, s.median_us, s.p99_us, s.max_us);
  } else {
    printf("  %-28s %6.0f MB  │ %8.2f  %8.2f  %8.2f  %8.2f  %8.2f\n",
           op, mb, s.min_us, s.avg_us, s.median_us, s.p99_us, s.max_us);
  }
}

// ============================================================
// Main
// ============================================================

int main(int argc, char** argv) {
  int iters = 200;

  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == "--iters" && i + 1 < argc)
      iters = atoi(argv[++i]);
    else if (std::string(argv[i]) == "--device" && i + 1 < argc)
      g_device = atoi(argv[++i]);
    else if (std::string(argv[i]) == "--csv")
      g_csv = true;
    else if (std::string(argv[i]) == "--help") {
      printf("Usage: %s [--iters N] [--device D] [--csv]\n", argv[0]);
      return 0;
    }
  }

  if (!loadDriver()) return 1;

  CU_CHECK(fn_cuInit(0));

  CUdevice dev;
  CU_CHECK(fn_cuDeviceGet(&dev, g_device));

  char name[256] = {};
  CU_CHECK(fn_cuDeviceGetName(name, sizeof(name), dev));

  CUcontext ctx;
  CU_CHECK(fn_cuCtxCreate(&ctx, 0, dev));

  setupProp();

  // Verify VMM is supported.
  size_t granularity = 0;
  CU_CHECK(fn_cuMemGetAllocationGranularity(
      &granularity, &g_prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
  if (granularity == 0) {
    fprintf(stderr, "Device %s does not support VMM\n", name);
    return 1;
  }

  // ---- Sizes to test -------------------------------------------
  const size_t MB = 1ULL << 20;
  size_t sizes[] = {2 * MB, 16 * MB, 64 * MB, 256 * MB};
  int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

  // ---- Warmup --------------------------------------------------
  {
    CUdeviceptr va = 0;
    CUmemGenericAllocationHandle h = 0;
    fn_cuMemAddressReserve(&va, 2 * MB, 2 * MB, 0, 0);
    fn_cuMemCreate(&h, 2 * MB, &g_prop, 0);
    fn_cuMemMap(va, 2 * MB, 0, h, 0);
    fn_cuMemSetAccess(va, 2 * MB, &g_access_desc, 1);
    fn_cuMemUnmap(va, 2 * MB);
    fn_cuMemRelease(h);
    fn_cuMemAddressFree(va, 2 * MB);
  }

  // ---- Print banner --------------------------------------------
  if (!g_csv) {
    printf("============================================================\n");
    printf("CUDA VMM Raw API Benchmark\n");
    printf("Device: %s (id=%d)\n", name, g_device);
    printf("Granularity: %zu bytes\n", granularity);
    printf("Iterations per test: %d\n", iters);
    printf("============================================================\n\n");
    printf("  %-28s %9s  │ %8s  %8s  %8s  %8s  %8s\n",
           "Operation", "Size", "min(us)", "avg(us)", "med(us)",
           "p99(us)", "max(us)");
    printf("  %s\n",
           std::string(28 + 10 + 3 + 5 * 10, '-').c_str());
  } else {
    printHeader();
  }

  // ---- Run benchmarks ------------------------------------------
  for (int si = 0; si < num_sizes; ++si) {
    size_t sz = sizes[si];

    std::vector<double> reserve_us, free_us;
    benchReserveFree(sz, iters, reserve_us, free_us);
    auto rs = computeStats(reserve_us);
    auto fs = computeStats(free_us);
    printRow("cuMemAddressReserve", sz, rs);
    printRow("cuMemAddressFree", sz, fs);

    std::vector<double> create_us, release_us;
    benchCreateRelease(sz, iters, create_us, release_us);
    auto cs = computeStats(create_us);
    auto rls = computeStats(release_us);
    printRow("cuMemCreate", sz, cs);
    printRow("cuMemRelease", sz, rls);

    std::vector<double> map_us, access_us, unmap_us;
    benchMapAccessUnmap(sz, iters, map_us, access_us, unmap_us);
    auto ms = computeStats(map_us);
    auto as = computeStats(access_us);
    auto us = computeStats(unmap_us);
    printRow("cuMemMap", sz, ms);
    printRow("cuMemSetAccess", sz, as);
    printRow("cuMemUnmap", sz, us);

    std::vector<double> full_us;
    benchFullPipeline(sz, iters, full_us);
    auto fls = computeStats(full_us);
    printRow("Full pipeline (alloc)", sz, fls);

    if (!g_csv) printf("\n");
  }

  fn_cuCtxDestroy(ctx);
  if (g_lib) dlclose(g_lib);
  return 0;
}
