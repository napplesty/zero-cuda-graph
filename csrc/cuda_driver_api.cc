// ============================================================
// cuda_driver_api.cc – dlopen wrapper implementation
// ============================================================

#include "cuda_driver_api.h"

#include <dlfcn.h>
#include <c10/util/Exception.h>

namespace zcg {

// static
CUDADriverAPI& CUDADriverAPI::get() {
  static CUDADriverAPI instance;
  return instance;
}

CUDADriverAPI::CUDADriverAPI() {
  // Try the versioned soname first, then the unversioned link.
#if defined(__linux__)
  lib_ = dlopen("libcuda.so.1", RTLD_LAZY);
  if (!lib_) {
    lib_ = dlopen("libcuda.so", RTLD_LAZY);
  }
#elif defined(__APPLE__)
  lib_ = dlopen("libcuda.dylib", RTLD_LAZY);
#else
  lib_ = nullptr;
#endif
  TORCH_CHECK(lib_ != nullptr,
              "zcg: failed to dlopen libcuda – is the NVIDIA driver installed?  "
              "dlerror: ", dlerror());

  // ---- load symbols -------------------------------------------
  loadSym(memAddressReserve,           "cuMemAddressReserve");
  loadSym(memAddressFree,              "cuMemAddressFree");
  loadSym(memCreate,                   "cuMemCreate");
  loadSym(memRelease,                  "cuMemRelease");
  loadSym(memMap,                      "cuMemMap");
  loadSym(memUnmap,                    "cuMemUnmap");
  loadSym(memSetAccess,                "cuMemSetAccess");
  loadSym(memGetAllocationGranularity, "cuMemGetAllocationGranularity");
  loadSym(ctxGetDevice,               "cuCtxGetDevice");
  loadSym(deviceGet,                   "cuDeviceGet");
}

CUDADriverAPI::~CUDADriverAPI() {
  // We intentionally do NOT dlclose – the library may have registered
  // atexit handlers or hold GPU state that must outlive this object.
}

template <typename FnPtr>
void CUDADriverAPI::loadSym(FnPtr& out, const char* name) {
  void* sym = dlsym(lib_, name);
  TORCH_CHECK(sym != nullptr,
              "zcg: failed to load symbol '", name,
              "' from libcuda.  dlerror: ", dlerror());
  out = reinterpret_cast<FnPtr>(sym);
}

} // namespace zcg
