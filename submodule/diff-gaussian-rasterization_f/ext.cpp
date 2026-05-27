/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include <torch/extension.h>
#include "rasterize_points.h"
#include "cuda_rasterizer/config.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rasterize_gaussians", &RasterizeGaussiansCUDA);
  m.def("rasterize_gaussians_simp", &RasterizeGaussiansSimpCUDA);
  m.def("rasterize_gaussians_depth", &RasterizeGaussiansDepthCUDA);
  m.def("rasterize_gaussians_backward", &RasterizeGaussiansBackwardCUDA);
  m.def("mark_visible", &markVisible);
  m.def("adamUpdate", &adamUpdate);
  m.def("fusedssim", &fusedssim);
  m.def("fusedssim_backward", &fusedssim_backward);  
  m.def("preprocess_gaussians", &PreprocessGaussiansCUDA);
  m.def("preprocess_gaussians_backward", &PreprocessGaussiansBackwardCUDA);
  m.def("render_gaussians", &RenderGaussiansCUDA);
  m.def("render_gaussians_backward", &RenderGaussiansBackwardCUDA);
  m.def("get_local2j_ids_bool", &GetLocal2jIdsBoolCUDA);
  m.def("get_local2j_ids_bool_adjust_mode6", &GetLocal2jIdsBoolAdjustMode6CUDA);
}

