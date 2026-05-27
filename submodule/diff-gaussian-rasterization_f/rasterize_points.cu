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

#include <math.h>
#include <torch/extension.h>
#include <cstdio>
#include <sstream>
#include <iostream>
#include <tuple>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <memory>
#include "cuda_rasterizer/config.h"
#include "cuda_rasterizer/rasterizer.h"
#include "cuda_rasterizer/adam.h"
#include <fstream>
#include <string>
#include <functional>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

#include "cuda_rasterizer/auxiliary.h"
#define ONE_DIM_BLOCK_SIZE 256
// #define NUM_CHANNELS 3 // Default 3, RGB
// #define NUM_ALL_MAP 5


std::function<char*(size_t N)> resizeFunctional(torch::Tensor& t) {
    auto lambda = [&t](size_t N) {
        t.resize_({(long long)N});
    return reinterpret_cast<char*>(t.contiguous().data_ptr());
    };
    return lambda;
}

std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RenderGaussiansCUDA(
	const torch::Tensor& background,
    const int image_height,
    const int image_width,// image setting
	torch::Tensor& means2D,// (P, 2)
	torch::Tensor& depths,
	torch::Tensor& radii,
	torch::Tensor& conic_opacity,
	torch::Tensor& rgb,//3dgs intermediate results
	const torch::Tensor& compute_locally,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& campos,
	const float tan_fovx,
	const float tan_fovy,
	const torch::Tensor& all_map,
	const bool render_geo,
	const bool debug,
	const pybind11::dict &args)
{
  const int P = means2D.size(0);
  const int H = image_height;
  const int W = image_width;

  auto int_opts = means2D.options().dtype(torch::kInt32);
  auto float_opts = means2D.options().dtype(torch::kFloat32);

  torch::Tensor out_color = torch::full({NUM_CHANNELS, H, W}, 0.0, float_opts);

  torch::Tensor out_observe = torch::full({P}, 0, means2D.options().dtype(torch::kInt32));
  torch::Tensor out_all_map = torch::full({NUM_ALL_MAP, H, W}, 0, float_opts);
  torch::Tensor out_plane_depth = torch::full({1, H, W}, 0, float_opts);




  const int TILE_Y = (H + BLOCK_Y - 1) / BLOCK_Y;
  const int TILE_X = (W + BLOCK_X - 1) / BLOCK_X;
  const int tile_num = TILE_Y * TILE_X;
  torch::Tensor n_render = torch::full({tile_num}, 0, int_opts);
  torch::Tensor n_consider = torch::full({tile_num}, 0, int_opts);
  torch::Tensor n_contrib = torch::full({tile_num}, 0, int_opts);

  torch::Device device(torch::kCUDA);
  torch::TensorOptions options(torch::kByte);
  torch::Tensor geomBuffer = torch::empty({0}, options.device(device));
  torch::Tensor binningBuffer = torch::empty({0}, options.device(device));
  torch::Tensor imgBuffer = torch::empty({0}, options.device(device));
  std::function<char*(size_t)> geomFunc = resizeFunctional(geomBuffer);
  std::function<char*(size_t)> binningFunc = resizeFunctional(binningBuffer);
  std::function<char*(size_t)> imgFunc = resizeFunctional(imgBuffer);

  const float focal_y = H / (2.0f * tan_fovy);
  const float focal_x = W / (2.0f * tan_fovx);
  
  int rendered = 0;
  if(P != 0)
  {
	  rendered = CudaRasterizer::Rasterizer::renderForward(
		geomFunc,
		binningFunc,
		imgFunc,//buffer
	    P,
		background.contiguous().data<float>(),
		W, H,//image setting
		focal_x, focal_y,
		float(W*0.5f), float(H*0.5f),
		viewmatrix.contiguous().data<float>(), 
		campos.contiguous().data<float>(),
		all_map.contiguous().data<float>(), 
		reinterpret_cast<float2*>(means2D.contiguous().data<float>()),
		depths.contiguous().data<float>(),
		radii.contiguous().data<int>(),
		reinterpret_cast<float4*>(conic_opacity.contiguous().data<float>()),
		rgb.contiguous().data<float>(),//3dgs intermediate results
		compute_locally.contiguous().data<bool>(),
		out_color.contiguous().data<float>(),
		n_render.contiguous().data<int>(),
		n_consider.contiguous().data<int>(),
		n_contrib.contiguous().data<int>(),//output
		out_observe.contiguous().data<int>(),
		out_all_map.contiguous().data<float>(),
		out_plane_depth.contiguous().data<float>(),
		render_geo,
		debug,
		args);
  }
  return std::make_tuple(rendered, out_color, n_render, n_consider, n_contrib, out_observe, out_all_map, out_plane_depth, geomBuffer, binningBuffer, imgBuffer);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RenderGaussiansBackwardCUDA(
 	const torch::Tensor& background,
	const float tan_fovx,
	const float tan_fovy,
	const torch::Tensor& all_map_pixels,
	const torch::Tensor& all_maps,
	const int R,
	const torch::Tensor& geomBuffer,
	const torch::Tensor& binningBuffer,
	const torch::Tensor& imageBuffer,
	const torch::Tensor& compute_locally,
    const torch::Tensor& dL_dout_color,
	const torch::Tensor& dL_dout_all_map,
	const torch::Tensor& dL_dout_plane_depth,
	const torch::Tensor& means2D,// (P, 2)
	const torch::Tensor& conic_opacity,
	const torch::Tensor& rgb,
	const bool render_geo,
	const bool debug,
	const pybind11::dict &args)
{
  const int P = means2D.size(0);
  const int H = dL_dout_color.size(1);
  const int W = dL_dout_color.size(2);
  
  torch::Tensor dL_dmeans2D = torch::zeros({P, 3}, means2D.options());
  torch::Tensor dL_dcolors = torch::zeros({P, NUM_CHANNELS}, means2D.options());//if we use mixed precision, dtype in options() is different now. If we also do swapping, device could be different. 
  torch::Tensor dL_dconic = torch::zeros({P, 2, 2}, means2D.options());//The requires_grad property for the gradient tensor is typically False
  torch::Tensor dL_dopacity = torch::zeros({P, 1}, means2D.options());
  torch::Tensor dL_dmeans2D_abs = torch::zeros({P, 3}, means2D.options());
  torch::Tensor dL_dall_map = torch::zeros({P, NUM_ALL_MAP}, means2D.options());
  if(P != 0)
  {
	  CudaRasterizer::Rasterizer::renderBackward(
		P, R,
		background.contiguous().data<float>(),
		W, H,   //rasterization settings.  
		tan_fovx, tan_fovy,
		reinterpret_cast<char*>(geomBuffer.contiguous().data_ptr()),
		reinterpret_cast<char*>(binningBuffer.contiguous().data_ptr()),
		reinterpret_cast<char*>(imageBuffer.contiguous().data_ptr()),//buffer that contains intermedia results
		compute_locally.contiguous().data<bool>(),
		dL_dout_color.contiguous().data<float>(),//gradient of output
		dL_dout_all_map.contiguous().data<float>(),
		dL_dout_plane_depth.contiguous().data<float>(),
		dL_dmeans2D.contiguous().data<float>(),
		dL_dmeans2D_abs.contiguous().data<float>(),
		dL_dconic.contiguous().data<float>(),
		dL_dopacity.contiguous().data<float>(),
		dL_dcolors.contiguous().data<float>(),//gradient of inputs
		dL_dall_map.contiguous().data<float>(),
		reinterpret_cast<float2*>(means2D.contiguous().data<float>()),
		reinterpret_cast<float4*>(conic_opacity.contiguous().data<float>()),
		rgb.contiguous().data<float>(),
		all_maps.contiguous().data<float>(),
		all_map_pixels.contiguous().data<float>(),
		render_geo,
		debug,
		args);
  }

  torch::Tensor dL_dconic_opacity = torch::zeros({P, 4}, means2D.options());
  // set dL_dconic_opacity[..., 0] = dL_dconic[..., 0, 0]
  dL_dconic_opacity.select(1, 0).copy_(dL_dconic.select(1, 0).select(1, 0));
  // set dL_dconic_opacity[..., 1] = dL_dconic[..., 0, 1]
  dL_dconic_opacity.select(1, 1).copy_(dL_dconic.select(1, 0).select(1, 1));
  // set dL_dconic_opacity[..., 2] = dL_dconic[..., 1, 1]
  dL_dconic_opacity.select(1, 2).copy_(dL_dconic.select(1, 1).select(1, 1));
  // set dL_dconic_opacity[..., 3] = dL_dopacity[..., 0]
  dL_dconic_opacity.select(1, 3).copy_(dL_dopacity.select(1, 0));

  return std::make_tuple(dL_dmeans2D, dL_dmeans2D_abs, dL_dconic_opacity, dL_dcolors, dL_dall_map);
}


__global__ void getTouchedIdsBool(
	int P,
	int height,
	int width,
	int world_size,
	const float2* means2D,
	const int* radii,// NOTE: radii is not const in getRect()
	const int* dist_global_strategy,
	bool* touchedIdsBool,
	bool avoid_pixel_all2all)
{
	auto i = cg::this_grid().thread_rank();
	if (i < P)
	{
		uint2 rect_min, rect_max;
		dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);

		getRect(means2D[i], radii[i], rect_min, rect_max, tile_grid);
		
		// method 1:
		int touched_min_tile_idx = rect_min.y * tile_grid.x + rect_min.x;
		int touched_max_tile_idx = (rect_max.y - 1 ) * tile_grid.x + rect_max.x - 1;

		if ( touched_max_tile_idx < touched_min_tile_idx )
			return;
			
		for (int rk = 0; rk < world_size; rk++)
		{
			int tile_l = *(dist_global_strategy+rk);
			int tile_r = *(dist_global_strategy+rk+1);
			if (avoid_pixel_all2all) {
				tile_l -= tile_grid.x+1;
				tile_r += tile_grid.x+1;
			}

			if (touched_max_tile_idx < tile_l || touched_min_tile_idx >= tile_r)
				continue;
			
			touchedIdsBool[i * world_size + rk] = true;
		}
		

		
	}
}

torch::Tensor GetLocal2jIdsBoolCUDA(
	int image_height,
	int image_width,
	int mp_rank,
	int mp_world_size,
	const torch::Tensor& means2D,
	const torch::Tensor& radii,
	const torch::Tensor& dist_global_strategy,
	const pybind11::dict &args)
{	
	const int P = means2D.size(0);
	const int H = image_height;
	const int W = image_width;
	bool avoid_pixel_all2all = args["avoid_pixel_all2all"].cast<bool>();

	torch::Tensor local2jIdsBool = torch::full({P, mp_world_size}, false, means2D.options().dtype(torch::kBool));

	getTouchedIdsBool << <(P + ONE_DIM_BLOCK_SIZE - 1) / ONE_DIM_BLOCK_SIZE, ONE_DIM_BLOCK_SIZE >> >(
		P,
		H,
		W,
		mp_world_size,
		reinterpret_cast<float2*>(means2D.contiguous().data<float>()),
		radii.contiguous().data<int>(),
		dist_global_strategy.contiguous().data<int>(),
		local2jIdsBool.contiguous().data<bool>(),
		avoid_pixel_all2all
	);

	return local2jIdsBool;
}


__global__ void getTouchedIdsBoolAdjustMode6(
	int P,
	int height,
	int width,
	int world_size,
	const float2* means2D,
	const int* radii,// NOTE: radii is not const in getRect()
	const int* rectangles,
	bool* touchedIdsBool,
	bool avoid_pixel_all2all)
{
	auto i = cg::this_grid().thread_rank();
	if (i < P)
	{
		uint2 rect_min, rect_max;
		dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);

		getRect(means2D[i], radii[i], rect_min, rect_max, tile_grid);

		for (int rk = 0; rk < world_size; rk++)
		{
			// local_tile_y_l, local_tile_y_r, local_tile_x_l, local_tile_x_r
			const int* rectangles_offset = rectangles+(rk*4);
			int local_tile_y_l = *(rectangles_offset);
			int local_tile_y_r = *(rectangles_offset+1);
			int local_tile_x_l = *(rectangles_offset+2);
			int local_tile_x_r = *(rectangles_offset+3);



			if (avoid_pixel_all2all) {
				if (local_tile_y_l>0) local_tile_y_l-=1;
				if (local_tile_x_l>0) local_tile_x_l-=1;//WERID: If local_tile_x_l changes to -1, then it gives weird behavior and I have not figure it out yet. 
				local_tile_y_r+=1;
				local_tile_x_r+=1;
			}
			if (rect_max.y <= local_tile_y_l || 
				local_tile_y_r <= rect_min.y || 
				rect_max.x <= local_tile_x_l || 
				local_tile_x_r <= rect_min.x) continue;

			touchedIdsBool[i * world_size + rk] = true;
		}
	}
}

torch::Tensor GetLocal2jIdsBoolAdjustMode6CUDA(
	int image_height,
	int image_width,
	int mp_rank,
	int mp_world_size,
	const torch::Tensor& means2D,
	const torch::Tensor& radii,
	const torch::Tensor& rectangles,
	const pybind11::dict &args)
{
	const int P = means2D.size(0);
	const int H = image_height;
	const int W = image_width;
	bool avoid_pixel_all2all = args["avoid_pixel_all2all"].cast<bool>();

	torch::Tensor local2jIdsBool = torch::full({P, mp_world_size}, false, means2D.options().dtype(torch::kBool));

	getTouchedIdsBoolAdjustMode6 << <(P + ONE_DIM_BLOCK_SIZE - 1) / ONE_DIM_BLOCK_SIZE, ONE_DIM_BLOCK_SIZE >> >(
		P,
		H,
		W,
		mp_world_size,
		reinterpret_cast<float2*>(means2D.contiguous().data<float>()),
		radii.contiguous().data<int>(),
		rectangles.contiguous().data<int>(),
		local2jIdsBool.contiguous().data<bool>(),
		avoid_pixel_all2all
	);

	return local2jIdsBool;
}


std::tuple<int, int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, 
torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansCUDA(
  const torch::Tensor& background,
  const torch::Tensor& means3D,
    const torch::Tensor& colors,
    const torch::Tensor& opacity,
  const torch::Tensor& scales,
  const torch::Tensor& rotations,
  const float scale_modifier,
  const torch::Tensor& cov3D_precomp,
  const torch::Tensor& viewmatrix,
  const torch::Tensor& projmatrix,
  const float tan_fovx, 
  const float tan_fovy,
    const int image_height,
    const int image_width,
	const torch::Tensor& dc,
	const torch::Tensor& sh,
  const int degree,
  const torch::Tensor& campos,
  const bool prefiltered,
  const bool flag_max_count,
  const torch::Tensor& culling,
  const bool debug
  )
{
  if (means3D.ndimension() != 2 || means3D.size(1) != 3) {
    AT_ERROR("means3D must have dimensions (num_points, 3)");
  }
  
  const int P = means3D.size(0);
  const int H = image_height;
  const int W = image_width;

  auto int_opts = means3D.options().dtype(torch::kInt32);
  auto float_opts = means3D.options().dtype(torch::kFloat32);

  torch::Tensor out_color = torch::full({NUM_CHAFFELS, H, W}, 0.0, float_opts);
  torch::Tensor radii = torch::full({P}, 0, means3D.options().dtype(torch::kInt32));

  torch::Tensor accum_max_count = torch::full({P}, 0, float_opts);
  
  torch::Device device(torch::kCUDA);
  torch::TensorOptions options(torch::kByte);
  torch::Tensor geomBuffer = torch::empty({0}, options.device(device));
  torch::Tensor binningBuffer = torch::empty({0}, options.device(device));
  torch::Tensor imgBuffer = torch::empty({0}, options.device(device));
  torch::Tensor sampleBuffer = torch::empty({0}, options.device(device));
  std::function<char*(size_t)> geomFunc = resizeFunctional(geomBuffer);
  std::function<char*(size_t)> binningFunc = resizeFunctional(binningBuffer);
  std::function<char*(size_t)> imgFunc = resizeFunctional(imgBuffer);
  std::function<char*(size_t)> sampleFunc = resizeFunctional(sampleBuffer);
  
  int rendered = 0;
  int num_buckets = 0;
  if(P != 0)
  {
    int M = 0;
    if(sh.size(0) != 0)
    {
      M = sh.size(1);
    }

    auto tup = CudaRasterizer::Rasterizer::forward(
      geomFunc,
      binningFunc,
      imgFunc,
      sampleFunc,
      P, degree, M,
      background.contiguous().data<float>(),
      W, H,
      means3D.contiguous().data<float>(),
      dc.contiguous().data_ptr<float>(),
      sh.contiguous().data_ptr<float>(),
      colors.contiguous().data<float>(), 
      opacity.contiguous().data<float>(), 
      scales.contiguous().data_ptr<float>(),
      scale_modifier,
      rotations.contiguous().data_ptr<float>(),
      cov3D_precomp.contiguous().data<float>(), 
      viewmatrix.contiguous().data<float>(), 
      projmatrix.contiguous().data<float>(),
      campos.contiguous().data<float>(),
      tan_fovx,
      tan_fovy,
      prefiltered,
      out_color.contiguous().data<float>(),

      flag_max_count,  
      accum_max_count.contiguous().data<float>(),  

      radii.contiguous().data<int>(),
      culling.contiguous().data<bool>(),
      debug);

		rendered = std::get<0>(tup);
		num_buckets = std::get<1>(tup);      
  }
  

  return std::make_tuple(rendered, num_buckets, out_color, accum_max_count, radii, geomBuffer, 
  binningBuffer, imgBuffer, sampleBuffer);
  
}


std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, 
torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansSimpCUDA(
  const torch::Tensor& background,
  const torch::Tensor& means3D,
    const torch::Tensor& colors,
    const torch::Tensor& opacity,
  const torch::Tensor& scales,
  const torch::Tensor& rotations,
  const float scale_modifier,
  const torch::Tensor& cov3D_precomp,
  const torch::Tensor& viewmatrix,
  const torch::Tensor& projmatrix,
  const float tan_fovx, 
  const float tan_fovy,
    const int image_height,
    const int image_width,
  const torch::Tensor& dc,
  const torch::Tensor& sh,
  const int degree,
  const torch::Tensor& campos,
  const bool prefiltered,
  const torch::Tensor& culling,
  const bool debug
  )
{
  if (means3D.ndimension() != 2 || means3D.size(1) != 3) {
    AT_ERROR("means3D must have dimensions (num_points, 3)");
  }
  
  const int P = means3D.size(0);
  const int H = image_height;
  const int W = image_width;

  auto int_opts = means3D.options().dtype(torch::kInt32);
  auto float_opts = means3D.options().dtype(torch::kFloat32);

  torch::Tensor out_color = torch::full({NUM_CHAFFELS, H, W}, 0.0, float_opts);
  torch::Tensor radii = torch::full({P}, 0, means3D.options().dtype(torch::kInt32));

  torch::Tensor accum_weights_ptr = torch::full({P}, 0, float_opts);
  torch::Tensor accum_weights_count = torch::full({P}, 0, int_opts);
  torch::Tensor accum_max_count = torch::full({P}, 0, float_opts);
  
  torch::Device device(torch::kCUDA);
  torch::TensorOptions options(torch::kByte);
  torch::Tensor geomBuffer = torch::empty({0}, options.device(device));
  torch::Tensor binningBuffer = torch::empty({0}, options.device(device));
  torch::Tensor imgBuffer = torch::empty({0}, options.device(device));
  std::function<char*(size_t)> geomFunc = resizeFunctional(geomBuffer);
  std::function<char*(size_t)> binningFunc = resizeFunctional(binningBuffer);
  std::function<char*(size_t)> imgFunc = resizeFunctional(imgBuffer);
  
  int rendered = 0;
  if(P != 0)
  {
    int M = 0;
    if(sh.size(0) != 0)
    {
      M = sh.size(1);
    }

    rendered = CudaRasterizer::Rasterizer::forward_simp(
      geomFunc,
      binningFunc,
      imgFunc,
      P, degree, M,
      background.contiguous().data<float>(),
      W, H,
      means3D.contiguous().data<float>(),
      dc.contiguous().data_ptr<float>(),
      sh.contiguous().data_ptr<float>(),
      colors.contiguous().data<float>(), 
      opacity.contiguous().data<float>(), 
      scales.contiguous().data_ptr<float>(),
      scale_modifier,
      rotations.contiguous().data_ptr<float>(),
      cov3D_precomp.contiguous().data<float>(), 
      viewmatrix.contiguous().data<float>(), 
      projmatrix.contiguous().data<float>(),
      campos.contiguous().data<float>(),
      tan_fovx,
      tan_fovy,
      prefiltered,
      out_color.contiguous().data<float>(),

      accum_weights_ptr.contiguous().data<float>(),  
      accum_weights_count.contiguous().data<int>(),  
      accum_max_count.contiguous().data<float>(),  

      radii.contiguous().data<int>(),
      culling.contiguous().data<bool>(),
      debug);
  }

  return std::make_tuple(rendered, out_color, accum_weights_ptr, accum_weights_count, accum_max_count, radii, geomBuffer, 
  binningBuffer, imgBuffer);
  
}









std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, 
torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, 
torch::Tensor>
RasterizeGaussiansDepthCUDA(
  const torch::Tensor& background,
  const torch::Tensor& means3D,
    const torch::Tensor& colors,
    const torch::Tensor& opacity,
  const torch::Tensor& scales,
  const torch::Tensor& rotations,
  const float scale_modifier,
  const torch::Tensor& cov3D_precomp,
  const torch::Tensor& viewmatrix,
  const torch::Tensor& projmatrix,
  const float tan_fovx, 
  const float tan_fovy,
    const int image_height,
    const int image_width,
  const torch::Tensor& dc,
  const torch::Tensor& sh,
  const int degree,
  const torch::Tensor& campos,
  const bool prefiltered,
  const torch::Tensor& culling,
  const bool debug)
{
  if (means3D.ndimension() != 2 || means3D.size(1) != 3) {
    AT_ERROR("means3D must have dimensions (num_points, 3)");
  }
  
  const int P = means3D.size(0);
  const int H = image_height;
  const int W = image_width;

  auto int_opts = means3D.options().dtype(torch::kInt32);
  auto float_opts = means3D.options().dtype(torch::kFloat32);

  torch::Tensor out_color = torch::full({NUM_CHAFFELS, H, W}, 0.0, float_opts);
  torch::Tensor out_pts = torch::full({3, H, W}, 0.0, float_opts);
  torch::Tensor radii = torch::full({P}, 0, means3D.options().dtype(torch::kInt32));

  torch::Tensor out_depth = torch::full({1, H, W}, 0.0, float_opts);

  torch::Tensor accum_alpha = torch::full({1, H, W}, 0.0, float_opts);
  torch::Tensor gidx = torch::full({1, H, W}, 0.0, int_opts);
  
  torch::Device device(torch::kCUDA);
  torch::TensorOptions options(torch::kByte);
  torch::Tensor geomBuffer = torch::empty({0}, options.device(device));
  torch::Tensor binningBuffer = torch::empty({0}, options.device(device));
  torch::Tensor imgBuffer = torch::empty({0}, options.device(device));
  std::function<char*(size_t)> geomFunc = resizeFunctional(geomBuffer);
  std::function<char*(size_t)> binningFunc = resizeFunctional(binningBuffer);
  std::function<char*(size_t)> imgFunc = resizeFunctional(imgBuffer);
  

  torch::Tensor discriminants = torch::full({1, H, W}, 0.0, float_opts);


  int rendered = 0;
  if(P != 0)
  {
    int M = 0;
    if(sh.size(0) != 0)
    {
      M = sh.size(1);
    }

    rendered = CudaRasterizer::Rasterizer::forward_depth(
      geomFunc,
      binningFunc,
      imgFunc,
      P, degree, M,
      background.contiguous().data<float>(),
      W, H,
      means3D.contiguous().data<float>(),
      dc.contiguous().data_ptr<float>(),
      sh.contiguous().data_ptr<float>(),
      colors.contiguous().data<float>(), 
      opacity.contiguous().data<float>(), 
      scales.contiguous().data_ptr<float>(),
      scale_modifier,
      rotations.contiguous().data_ptr<float>(),
      cov3D_precomp.contiguous().data<float>(), 
      viewmatrix.contiguous().data<float>(), 
      projmatrix.contiguous().data<float>(),
      campos.contiguous().data<float>(),
      tan_fovx,
      tan_fovy,
      prefiltered,
      out_color.contiguous().data<float>(),
      out_pts.contiguous().data<float>(),

      out_depth.contiguous().data<float>(),
      accum_alpha.contiguous().data<float>(),  
      gidx.contiguous().data<int>(),
      discriminants.contiguous().data<float>(),  

      radii.contiguous().data<int>(),
      culling.contiguous().data<bool>(),
      debug);
  }

  return std::make_tuple(rendered, out_color, out_pts, out_depth, 
  accum_alpha, gidx, discriminants, radii, geomBuffer, 
  binningBuffer, imgBuffer);  
}  





std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
 RasterizeGaussiansBackwardCUDA(
   const torch::Tensor& background,
  const torch::Tensor& means3D,
  const torch::Tensor& radii,
    const torch::Tensor& colors,
  const torch::Tensor& scales,
  const torch::Tensor& rotations,
  const float scale_modifier,
  const torch::Tensor& cov3D_precomp,
  const torch::Tensor& viewmatrix,
    const torch::Tensor& projmatrix,
  const float tan_fovx,
  const float tan_fovy,
    const torch::Tensor& dL_dout_color,
	const torch::Tensor& dc,
	const torch::Tensor& sh,
  const int degree,
  const torch::Tensor& campos,
  const torch::Tensor& geomBuffer,
  const int R,
  const torch::Tensor& binningBuffer,
  const torch::Tensor& imageBuffer,
	const int B,
	const torch::Tensor& sampleBuffer,  
  const bool debug) 
{
  const int P = means3D.size(0);
  const int H = dL_dout_color.size(1);
  const int W = dL_dout_color.size(2);
  
  int M = 0;
  if(sh.size(0) != 0)
  {  
  M = sh.size(1);
  }

  torch::Tensor dL_dmeans3D = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dmeans2D = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dcolors = torch::zeros({P, NUM_CHAFFELS}, means3D.options());
  torch::Tensor dL_dconic = torch::zeros({P, 2, 2}, means3D.options());
  torch::Tensor dL_dopacity = torch::zeros({P, 1}, means3D.options());
  torch::Tensor dL_dcov3D = torch::zeros({P, 6}, means3D.options());
  torch::Tensor dL_ddc = torch::zeros({P, 1, 3}, means3D.options());
  torch::Tensor dL_dsh = torch::zeros({P, M, 3}, means3D.options());
  torch::Tensor dL_dscales = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_drotations = torch::zeros({P, 4}, means3D.options());
  
  if(P != 0)
  {  
	  CudaRasterizer::Rasterizer::backward(P, degree, M, R, B,
    background.contiguous().data<float>(),
    W, H, 
    means3D.contiguous().data<float>(),
	  dc.contiguous().data<float>(),
	  sh.contiguous().data<float>(),
    colors.contiguous().data<float>(),
    scales.data_ptr<float>(),
    scale_modifier,
    rotations.data_ptr<float>(),
    cov3D_precomp.contiguous().data<float>(),
    viewmatrix.contiguous().data<float>(),
    projmatrix.contiguous().data<float>(),
    campos.contiguous().data<float>(),
    tan_fovx,
    tan_fovy,
    radii.contiguous().data<int>(),
    reinterpret_cast<char*>(geomBuffer.contiguous().data_ptr()),
    reinterpret_cast<char*>(binningBuffer.contiguous().data_ptr()),
    reinterpret_cast<char*>(imageBuffer.contiguous().data_ptr()),
	  reinterpret_cast<char*>(sampleBuffer.contiguous().data_ptr()),
    dL_dout_color.contiguous().data<float>(),
    dL_dmeans2D.contiguous().data<float>(),
    dL_dconic.contiguous().data<float>(),  
    dL_dopacity.contiguous().data<float>(),
    dL_dcolors.contiguous().data<float>(),
    dL_dmeans3D.contiguous().data<float>(),
    dL_dcov3D.contiguous().data<float>(),
	  dL_ddc.contiguous().data<float>(),
	  dL_dsh.contiguous().data<float>(),
    dL_dscales.contiguous().data<float>(),
    dL_drotations.contiguous().data<float>(),
    debug);
  }

  return std::make_tuple(dL_dmeans2D, dL_dcolors, dL_dopacity, dL_dmeans3D, dL_dcov3D, dL_ddc, dL_dsh, dL_dscales, dL_drotations);
}

torch::Tensor markVisible(
    torch::Tensor& means3D,
    torch::Tensor& viewmatrix,
    torch::Tensor& projmatrix)
{ 
  const int P = means3D.size(0);
  
  torch::Tensor present = torch::full({P}, false, means3D.options().dtype(at::kBool));
 
  if(P != 0)
  {
  CudaRasterizer::Rasterizer::markVisible(P,
    means3D.contiguous().data<float>(),
    viewmatrix.contiguous().data<float>(),
    projmatrix.contiguous().data<float>(),
    present.contiguous().data<bool>());
  }
  
  return present;
}


void adamUpdate(
	torch::Tensor &param,
	torch::Tensor &param_grad,
	torch::Tensor &exp_avg,
	torch::Tensor &exp_avg_sq,
	torch::Tensor &visible,
	const float lr,
	const float b1,
	const float b2,
	const float eps,
	const uint32_t N,
	const uint32_t M
){
	ADAM::adamUpdate(
		param.contiguous().data<float>(),
		param_grad.contiguous().data<float>(),
		exp_avg.contiguous().data<float>(),
		exp_avg_sq.contiguous().data<float>(),
		visible.contiguous().data<bool>(),
		lr,
		b1,
		b2,
		eps,
		N,
		M);
}

std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
PreprocessGaussiansCUDA(
	const torch::Tensor& means3D,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const torch::Tensor& sh,
    const torch::Tensor& opacity,//3dgs' parametes.
	const float scale_modifier,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& projmatrix,
	const float tan_fovx, 
	const float tan_fovy,
    const int image_height,
    const int image_width,
	const int degree,
	const torch::Tensor& campos,
	const bool prefiltered,//raster_settings
	const bool debug,
	const pybind11::dict &args) {

	if (means3D.ndimension() != 2 || means3D.size(1) != 3) {
		AT_ERROR("means3D must have dimensions (num_points, 3)");
	}

	const int P = means3D.size(0);
	const int H = image_height;
	const int W = image_width;

	// of shape (P, 2). means2D is (P, 2) in cuda. It will be converted to (P, 3) when is sent back to python to meet torch graph's requirement.
	torch::Tensor means2D = torch::full({P, 2}, 0.0, means3D.options());
	// of shape (P)
	torch::Tensor depths = torch::full({P}, 0.0, means3D.options());
	// of shape (P)
	torch::Tensor radii = torch::full({P}, 0, means3D.options().dtype(torch::kInt32));
	// of shape (P, 6)
	torch::Tensor cov3D = torch::full({P, 6}, 0.0, means3D.options());
	// of shape (P, 4)
	torch::Tensor conic_opacity = torch::full({P, 4}, 0.0, means3D.options());
	// of shape (P, 3)
	torch::Tensor rgb = torch::full({P, 3}, 0.0, means3D.options());
	// of shape (P)
	torch::Tensor clamped = torch::full({P, 3}, false, means3D.options().dtype(at::kBool));

	int rendered = 0;
	if(P != 0)
	{
    torch::Tensor dc;
    torch::Tensor sh_rest;
		int M = 0;
		if(sh.size(0) != 0)
		{
      if (sh.ndimension() != 3 || sh.size(2) != 3 || sh.size(1) < 1) {
        AT_ERROR("sh must have dimensions (num_points, num_sh_coeffs, 3) with num_sh_coeffs >= 1");
      }
      dc = sh.select(1, 0).contiguous();
      sh_rest = sh.slice(1, 1, sh.size(1)).contiguous();
      M = sh_rest.size(1);
		}

    const float* dc_ptr = dc.defined() ? dc.data_ptr<float>() : nullptr;
    const float* sh_ptr = sh_rest.defined() ? sh_rest.data_ptr<float>() : nullptr;

		rendered = CudaRasterizer::Rasterizer::preprocessForward(
			reinterpret_cast<float2*>(means2D.contiguous().data<float>()),
			depths.contiguous().data<float>(),
			radii.contiguous().data<int>(),
			cov3D.contiguous().data<float>(),
			reinterpret_cast<float4*>(conic_opacity.contiguous().data<float>()),
			rgb.contiguous().data<float>(),
			clamped.contiguous().data<bool>(),
			P, degree, M,
			W, H,
			means3D.contiguous().data<float>(),
			scales.contiguous().data_ptr<float>(),
			rotations.contiguous().data_ptr<float>(),
      dc_ptr,
      sh_ptr,
			opacity.contiguous().data<float>(), 
			scale_modifier,
			viewmatrix.contiguous().data<float>(), 
			projmatrix.contiguous().data<float>(),
			campos.contiguous().data<float>(),
			tan_fovx,
			tan_fovy,
			prefiltered,
			debug,
			args);
	}
	return std::make_tuple(rendered, means2D, depths, radii, cov3D, conic_opacity, rgb, clamped);
}


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
  PreprocessGaussiansBackwardCUDA(
	const torch::Tensor& radii,
	const torch::Tensor& cov3D,
	const torch::Tensor& clamped,//the above are all per-Gaussian intemediate results.
	const torch::Tensor& means3D,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const torch::Tensor& sh,//input of this operator
	const float scale_modifier,
	const torch::Tensor& viewmatrix,
    const torch::Tensor& projmatrix,
	const float tan_fovx,
	const float tan_fovy,
    const int image_height,
    const int image_width,
	const int degree,
	const torch::Tensor& campos,//rasterization setting.
	const torch::Tensor& dL_dmeans2D,// (P, 3)
	const torch::Tensor& dL_dconic_opacity,
	const torch::Tensor& dL_dcolors,//gradients of output of this operator
	const int R,
	const bool debug,
	const pybind11::dict &args)
{
  const int P = means3D.size(0);
  const int H = image_height;
  const int W = image_width;

  torch::Tensor dc;
  torch::Tensor sh_rest;
  
  int M = 0;
  if(sh.size(0) != 0)
  {
	if (sh.ndimension() != 3 || sh.size(2) != 3 || sh.size(1) < 1) {
		AT_ERROR("sh must have dimensions (num_points, num_sh_coeffs, 3) with num_sh_coeffs >= 1");
	}
	dc = sh.select(1, 0).contiguous();
	sh_rest = sh.slice(1, 1, sh.size(1)).contiguous();
	M = sh_rest.size(1);
  }

  const float* dc_ptr = dc.defined() ? dc.data_ptr<float>() : nullptr;
  const float* sh_ptr = sh_rest.defined() ? sh_rest.data_ptr<float>() : nullptr;

  torch::Tensor dL_dconic = torch::zeros({P, 2, 2}, means3D.options());
  // set dL_dconic[..., 0, 0] = dL_dconic_opacity[..., 0]
  dL_dconic.select(1, 0).select(1, 0).copy_(dL_dconic_opacity.select(1, 0));// select() is kind of view, it does not allocate new memory.
  // set dL_dconic[..., 0, 1] = dL_dconic_opacity[..., 1]
  dL_dconic.select(1, 0).select(1, 1).copy_(dL_dconic_opacity.select(1, 1));
  // set dL_dconic[..., 1, 1] = dL_dconic_opacity[..., 2]
  dL_dconic.select(1, 1).select(1, 1).copy_(dL_dconic_opacity.select(1, 2));
  dL_dconic = dL_dconic.contiguous();

  torch::Tensor dL_dopacity = torch::zeros({P, 1}, means3D.options());
  // set dL_dopacity[..., 0] = dL_dconic_opacity[..., 3]
  dL_dopacity.select(1, 0).copy_(dL_dconic_opacity.select(1, 3));
  dL_dopacity = dL_dopacity.contiguous();

  torch::Tensor dL_dmeans3D = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dcov3D = torch::zeros({P, 6}, means3D.options());
  //dL_dcov3D is itermidiate result to compute dL_drotations and dL_dscales, do not need to return to python.
  torch::Tensor dL_dscales = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_drotations = torch::zeros({P, 4}, means3D.options());
  torch::Tensor dL_ddc = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dsh_rest = torch::zeros({P, M, 3}, means3D.options());

  if(P != 0)
  {  
	  CudaRasterizer::Rasterizer::preprocessBackward(
		radii.contiguous().data<int>(),
		cov3D.contiguous().data<float>(),
		clamped.contiguous().data<bool>(),//the above are all per-Gaussian intermediate results.
		P, degree, M, R,
		W, H, //rasterization setting.
		means3D.contiguous().data<float>(),
		scales.data_ptr<float>(),
  	    rotations.data_ptr<float>(),
    dc_ptr,
    sh_ptr,//input of this operator
		scale_modifier,
		viewmatrix.contiguous().data<float>(),
	    projmatrix.contiguous().data<float>(),
	    campos.contiguous().data<float>(),
	    tan_fovx,
	    tan_fovy,//rasterization setting.
	    dL_dmeans2D.contiguous().data<float>(),
	    dL_dconic.contiguous().data<float>(),
	    dL_dcolors.contiguous().data<float>(),//gradients of output of this operator
	    dL_dmeans3D.contiguous().data<float>(),
	    dL_dcov3D.contiguous().data<float>(),
	    dL_dscales.contiguous().data<float>(),
	    dL_drotations.contiguous().data<float>(),
    dL_ddc.contiguous().data<float>(),
      dL_dsh_rest.contiguous().data<float>(),//gradients of input of this operator
		debug,
		args);
  }

  torch::Tensor dL_dsh = torch::cat({dL_ddc.unsqueeze(1), dL_dsh_rest}, 1).contiguous();

  return std::make_tuple(dL_dmeans3D, dL_dscales, dL_drotations, dL_dsh, dL_dopacity);
}
