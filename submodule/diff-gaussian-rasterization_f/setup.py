#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os
os.path.dirname(os.path.abspath(__file__))
# import torch.utils.cpp_extension
# # Hack to force C++14 instead of PyTorch's default C++17
# if '-std=c++17' in torch.utils.cpp_extension.COMMON_NVCC_FLAGS:
#     torch.utils.cpp_extension.COMMON_NVCC_FLAGS.remove('-std=c++17')
# if '-std=c++14' not in torch.utils.cpp_extension.COMMON_NVCC_FLAGS:
#     torch.utils.cpp_extension.COMMON_NVCC_FLAGS.append('-std=c++14')

setup(
    name="diff_gaussian_rasterization_f",
    packages=['diff_gaussian_rasterization_f'],
    ext_modules=[
        CUDAExtension(
            name="diff_gaussian_rasterization_f._C",
            sources=[
                "cuda_rasterizer/rasterizer_impl.cu",
                "cuda_rasterizer/forward.cu",
                "cuda_rasterizer/backward.cu",
                "cuda_rasterizer/adam.cu",
                "rasterize_points.cu",
                "conv.cu",
                "ext.cpp",
            ],
            extra_compile_args={
                "cxx": ["-std=c++17"],
                "nvcc": [
                    "-std=c++17",
                    "-I" + os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party/glm/"),
                ],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
