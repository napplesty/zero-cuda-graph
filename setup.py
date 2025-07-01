from setuptools import setup, find_packages
from torch.utils.cpp_extension import CppExtension, BuildExtension

setup(
    name="zero_cuda_graph",
    version="1.0",
    author="nastyapple",
    author_email="napplesty@outlook.com",
    description="support no overhead cudagraph in torch.compile",
    packages=find_packages(where="zcg"),
    ext_modules=[
        CppExtension(
            name="zcg_c",
            sources=[
                "csrc/allocator.cc",
                "csrc/substitution.cc"
            ],
            include_dirs=["include"]
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    install_requires=["torch>=2.5.0"]
)
