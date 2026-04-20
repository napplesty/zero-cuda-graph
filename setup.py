import os
import torch
from setuptools import setup, find_packages
from torch.utils.cpp_extension import CppExtension, BuildExtension

# ---------------------------------------------------------------------------
# Locate CUDA toolkit headers (needed for cuda.h type definitions).
# We do NOT link libcuda at build-time – it is dlopen'd at runtime.
# ---------------------------------------------------------------------------

cuda_home = getattr(torch.utils.cpp_extension, "CUDA_HOME", None)
if cuda_home is None:
    cuda_home = os.environ.get("CUDA_HOME", "/usr/local/cuda")

cuda_include_dir = os.path.join(cuda_home, "include")
torch_lib_dir = os.path.join(os.path.dirname(torch.__file__), "lib")

# ---------------------------------------------------------------------------
# Extension sources
# ---------------------------------------------------------------------------

sources = [
    "csrc/cuda_driver_api.cc",
    "csrc/physical_memory.cc",
    "csrc/small_allocator.cc",
    "csrc/large_allocator.cc",
    "csrc/vmm_allocator.cc",
    "csrc/substitution.cc",
    "csrc/bindings.cc",
]

setup(
    name="zero_cuda_graph",
    version="1.0",
    author="nastyapple",
    author_email="napplesty@outlook.com",
    description="VMM-based CUDA allocator for zero-overhead CUDA Graph in torch.compile",
    packages=find_packages(),
    ext_modules=[
        CppExtension(
            name="zcg_c",
            sources=sources,
            include_dirs=["include", cuda_include_dir],
            library_dirs=[torch_lib_dir],
            libraries=["c10_cuda", "dl"],
            extra_compile_args={
                "cxx": ["-std=c++17", "-O2"],
            },
            extra_link_args=[
                f"-Wl,-rpath,{torch_lib_dir}",
            ],
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.8",
    install_requires=["torch>=2.5.0"],
)
