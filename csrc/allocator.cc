#include <c10/core/Allocator.h>

#include <torch/csrc/Device.h>
#include <torch/extension.h>

#include <ATen/native/cpu/Loops.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/EmptyTensor.h>

struct CUDAVMMAllocator final : at::Allocator {
  CUDAVMMAllocator() = default;
  at::DataPtr allocate(size_t n) override {
    void* data
    return {data, data, &ReportAndDelete, at::Device(at::DeviceType::PrivateUse1, 0)};
  }

  static void Delete(void* ptr) {
    
  }

  at::DeleterFnPtr raw_deleter() const override {
    return &ReportAndDelete;
  }
};

static DummyCustomAllocator global_custom_alloc;
REGISTER_ALLOCATOR(c10::DeviceType::CUDA, &global_custom_alloc);
