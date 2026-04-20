#pragma once
// ============================================================
// cuda_driver_api.h – thin, type-safe dlopen wrapper around the
// CUDA driver VMM APIs.  Loaded lazily as a singleton.
// ============================================================

#include <cuda.h>
#include <c10/util/Exception.h>

namespace zcg {

// ------------------------------------------------------------
// Error-check macro – throws c10::Error on failure
// ------------------------------------------------------------
#define ZCG_CU_CHECK(expr)                                               \
  do {                                                                   \
    CUresult _err = (expr);                                              \
    TORCH_CHECK(_err == CUDA_SUCCESS,                                    \
                "CUDA driver error in " #expr ": code ", (int)_err);     \
  } while (0)

// ------------------------------------------------------------
// CUDADriverAPI singleton
// ------------------------------------------------------------
class CUDADriverAPI {
 public:
  /// Access the process-wide singleton (created on first call).
  static CUDADriverAPI& get();

  // --- Virtual Memory Management ---------------------------------
  decltype(&cuMemAddressReserve)           memAddressReserve;
  decltype(&cuMemAddressFree)              memAddressFree;
  decltype(&cuMemCreate)                   memCreate;
  decltype(&cuMemRelease)                  memRelease;
  decltype(&cuMemMap)                      memMap;
  decltype(&cuMemUnmap)                    memUnmap;
  decltype(&cuMemSetAccess)                memSetAccess;
  decltype(&cuMemGetAllocationGranularity) memGetAllocationGranularity;

  // --- Misc driver helpers we also need --------------------------
  decltype(&cuCtxGetDevice)                ctxGetDevice;
  decltype(&cuDeviceGet)                   deviceGet;

 private:
  CUDADriverAPI();
  ~CUDADriverAPI();
  CUDADriverAPI(const CUDADriverAPI&)            = delete;
  CUDADriverAPI& operator=(const CUDADriverAPI&) = delete;

  void* lib_ = nullptr;

  template <typename FnPtr>
  void loadSym(FnPtr& out, const char* name);
};

} // namespace zcg
