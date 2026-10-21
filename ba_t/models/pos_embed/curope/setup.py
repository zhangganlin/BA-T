# Copyright (C) 2022-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Explicit CUDA architectures (no GPU needed at build time, e.g. inside `docker build`).
# Add a '-gencode' entry for other GPUs, or use "-arch=native" (build machine GPU only).
all_cuda_archs = [
    '-gencode', 'arch=compute_75,code=sm_75',  # T4, RTX 20xx
    '-gencode', 'arch=compute_80,code=sm_80',  # A100
    '-gencode', 'arch=compute_86,code=sm_86',  # A40, RTX 30xx, RTX A6000
    '-gencode', 'arch=compute_89,code=sm_89',  # L4, L40S, RTX 40xx, RTX 6000 Ada
    '-gencode', 'arch=compute_90,code=sm_90',  # H100
    '-gencode', 'arch=compute_90,code=compute_90',  # PTX for newer GPUs (needs a matching torch/CUDA)
]

setup(
    name = 'curope',
    ext_modules = [
        CUDAExtension(
                name='curope',
                sources=[
                    "curope.cpp",
                    "kernels.cu",
                ],
                extra_compile_args = dict(
                    nvcc=['-O3','--ptxas-options=-v',"--use_fast_math"]+all_cuda_archs, 
                    cxx=['-O3'])
                )
    ],
    cmdclass = {
        'build_ext': BuildExtension
    })
