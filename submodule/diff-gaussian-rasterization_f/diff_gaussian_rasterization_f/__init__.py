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

from typing import NamedTuple
import torch.nn as nn
import torch
from . import _C

try:
    import diff_gaussian_rasterization_r as _split_backend
    _SPLIT_C = _split_backend._C
except Exception:
    _SPLIT_C = None


def _get_split_op_backend(op_name: str):
    # The distributed render pipeline requires the split backend
    # (diff-gaussian-rasterization_r style operators) for render ops.
    # Falling back to f render ops can silently produce tiled/block artifacts.
    # Keep preprocess on the f backend because some split variants return SH
    # grads with shape [P, 3] while this code expects [P, M, 3].
    prefer_split_ops = {
        "render_gaussians",
        "render_gaussians_backward",
    }
    if op_name in prefer_split_ops:
        if _SPLIT_C is not None and hasattr(_SPLIT_C, op_name):
            return _SPLIT_C
        raise AttributeError(
            "Missing split backend op '%s'. Install diff_gaussian_rasterization_r "
            "from submodule/diff-gaussian-rasterization_r so the multi-GPU "
            "render path remains consistent." % op_name
        )
    if hasattr(_C, op_name):
        return _C
    if _SPLIT_C is not None and hasattr(_SPLIT_C, op_name):
        return _SPLIT_C
    raise AttributeError(f"No backend provides split op '{op_name}'")



def cpu_deep_copy_tuple(input_tuple):
    copied_tensors = [item.cpu().clone() if isinstance(item, torch.Tensor) else item for item in input_tuple]
    return tuple(copied_tensors)


def preprocess_gaussians(
    means3D,
    scales,
    rotations,
    sh,
    opacities,
    raster_settings,
    cuda_args,
):
    return _PreprocessGaussians.apply(
        means3D,
        scales,
        rotations,
        sh,
        opacities,
        raster_settings,
        cuda_args,
    )


class _PreprocessGaussians(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means3D,
        scales,
        rotations,
        sh,
        opacities,
        raster_settings,
        cuda_args,
    ):

        # Restructure arguments the way that the C++ lib expects them
        args = (
            means3D,
            scales,
            rotations,
            sh,
            opacities,# 3dgs' parametes.
            raster_settings.scale_modifier,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.image_height,
            raster_settings.image_width,
            raster_settings.sh_degree,
            raster_settings.campos,
            raster_settings.prefiltered,
            raster_settings.debug,#raster_settings
            cuda_args
        )

        split_backend = _get_split_op_backend("preprocess_gaussians")
        num_rendered, means2D, depths, radii, cov3D, conic_opacity, rgb, clamped = split_backend.preprocess_gaussians(*args)

        # Keep relevant tensors for backward
        ctx.raster_settings = raster_settings
        ctx.cuda_args = cuda_args
        ctx.num_rendered = num_rendered
        ctx.save_for_backward(means3D, scales, rotations, sh, means2D, depths, radii, cov3D, conic_opacity, rgb, clamped)
        ctx.mark_non_differentiable(radii, depths)

        return means2D, rgb, conic_opacity, radii, depths

    @staticmethod
    def backward(ctx, grad_means2D, grad_rgb, grad_conic_opacity, grad_radii, grad_depths):
        # grad_radii, grad_depths should be all None. 

        # Restore necessary values from context
        num_rendered = ctx.num_rendered
        raster_settings = ctx.raster_settings
        cuda_args = ctx.cuda_args
        means3D, scales, rotations, sh, means2D, depths, radii, cov3D, conic_opacity, rgb, clamped = ctx.saved_tensors

        # change dL_dmeans2D from (P, 2) to (P, 3)
        # grad_means2D is (P, 2) now. Need to pad it to (P, 3) because preprocess_gaussians_backward's cuda implementation.
        grad_means2D_pad = torch.zeros((grad_means2D.shape[0], 1), dtype = grad_means2D.dtype, device = grad_means2D.device)
        grad_means2D = torch.cat((grad_means2D, grad_means2D_pad), dim = 1).contiguous()

        # Restructure args as C++ method expects them
        args = (radii,
                cov3D,
                clamped,#the above are all per-Gaussian intemediate results.
                means3D,
                scales,
                rotations, 
                sh, #input of this operator
                raster_settings.scale_modifier, 
                raster_settings.viewmatrix,
                raster_settings.projmatrix,
                raster_settings.tanfovx,
                raster_settings.tanfovy,
                raster_settings.image_height,
                raster_settings.image_width,
                raster_settings.sh_degree,
                raster_settings.campos,#rasterization setting.
                grad_means2D,
                grad_conic_opacity,
                grad_rgb,#gradients of output of this operator
                num_rendered,
                raster_settings.debug,
                cuda_args)

        split_backend = _get_split_op_backend("preprocess_gaussians_backward")
        dL_dmeans3D, dL_dscales, dL_drotations, dL_dsh, dL_dopacity = split_backend.preprocess_gaussians_backward(*args)

        grads = (
            dL_dmeans3D.contiguous(),
            dL_dscales.contiguous(),
            dL_drotations.contiguous(),
            dL_dsh.contiguous(),
            dL_dopacity.contiguous(),
            None,#raster_settings
            None,#raster_settings
        )

        return grads



def render_gaussians(
    means2D,
    means2D_abs,
    conic_opacity,
    rgb,
    all_map,
    depths,
    radii,
    compute_locally,
    extended_compute_locally,
    raster_settings,
    cuda_args,
):
    return _RenderGaussians.apply(
        means2D,
        means2D_abs,
        conic_opacity,
        rgb,
        all_map,
        depths,
        radii,
        compute_locally,
        extended_compute_locally,
        raster_settings,
        cuda_args,
    )

class _RenderGaussians(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means2D,
        means2D_abs,
        conic_opacity,
        rgb,
        all_maps,
        depths,
        radii,
        compute_locally,
        extended_compute_locally,
        raster_settings,
        cuda_args,
    ):

        # Restructure arguments the way that the C++ lib expects them
        args = (
            raster_settings.bg,
            raster_settings.image_height,
            raster_settings.image_width,# image setting
            means2D,
            depths,
            radii,
            conic_opacity,
            rgb,# 3dgs intermediate results
            extended_compute_locally if cuda_args["avoid_pixel_all2all"] else compute_locally,
            raster_settings.viewmatrix, 
            raster_settings.campos,
            raster_settings.tanfovx, 
            raster_settings.tanfovy, 
            all_maps,
            getattr(raster_settings, "render_geo", False),
            raster_settings.debug,
            cuda_args
        )

        split_backend = _get_split_op_backend("render_gaussians")
        num_rendered, color, n_render, n_consider, n_contrib, out_observe, out_all_map, out_plane_depth, geomBuffer, binningBuffer, imgBuffer = split_backend.render_gaussians(*args)

        # Keep relevant tensors for backward
        ctx.raster_settings = raster_settings
        ctx.cuda_args = cuda_args
        ctx.num_rendered = num_rendered
        # ctx.render_forward_start_time = render_forward_start_time
        ctx.save_for_backward(means2D, conic_opacity, rgb, all_maps, out_all_map, geomBuffer, binningBuffer, imgBuffer, compute_locally, extended_compute_locally)
        ctx.mark_non_differentiable(n_render, n_consider, n_contrib)

        return color, n_render, n_consider, n_contrib, out_observe, out_all_map, out_plane_depth

    @staticmethod
    def backward(ctx, grad_color, grad_n_render, grad_n_consider, grad_n_contrib, grad_out_observe, grad_out_all_map, grad_out_plane_depth):
        # grad_n_render, grad_n_consider, grad_n_contrib should be all None. 

        # Restore necessary values from context
        num_rendered = ctx.num_rendered
        raster_settings = ctx.raster_settings
        cuda_args = ctx.cuda_args
        means2D, conic_opacity, rgb, all_maps, all_map_pixels ,geomBuffer, binningBuffer, imgBuffer, compute_locally, extended_compute_locally = ctx.saved_tensors

        grad_color = grad_color.contiguous()
        if grad_out_all_map is None:
            grad_out_all_map = torch.zeros_like(all_map_pixels)
        else:
            grad_out_all_map = grad_out_all_map.contiguous()
        if grad_out_plane_depth is None:
            grad_out_plane_depth = torch.zeros(
                (raster_settings.image_height, raster_settings.image_width),
                dtype=grad_color.dtype,
                device=grad_color.device,
            )
        else:
            grad_out_plane_depth = grad_out_plane_depth.contiguous()

        # Restructure args as C++ method expects them
        args = (raster_settings.bg,
                raster_settings.tanfovx, 
                raster_settings.tanfovy, 
                all_map_pixels,
                all_maps,
                num_rendered,
                geomBuffer,
                binningBuffer,
                imgBuffer,
                compute_locally,# buffer
                grad_color,# gradient of output of this operator
                grad_out_all_map,
                grad_out_plane_depth,
                means2D,
                conic_opacity,
                rgb,# 3dgs intermediate results
                getattr(raster_settings, "render_geo", False),
                raster_settings.debug,
                cuda_args)

        split_backend = _get_split_op_backend("render_gaussians_backward")
        dL_dmeans2D, dL_dmeans2D_abs, dL_dconic_opacity, dL_dcolors, dL_dall_map = split_backend.render_gaussians_backward(*args)
        dL_dmeans2D = dL_dmeans2D[:,:2]

        grads = (
            dL_dmeans2D.contiguous(),
            dL_dmeans2D_abs.contiguous(),
            dL_dconic_opacity.contiguous(),
            dL_dcolors.contiguous(),
            dL_dall_map.contiguous(),
            None,
            None,
            None,
            None,
            None,
            None # this is for cuda_args
        )

        return grads




def rasterize_gaussians(
    means3D,
    means2D,
    dc,
    sh,
    colors_precomp,
    opacities,
    scales,
    rotations,
    cov3Ds_precomp,
    flag_max_count,
    culling,
    raster_settings,
):
    return _RasterizeGaussians.apply(
        means3D,
        means2D,
        dc,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        flag_max_count,
        culling,
        raster_settings,
    )

def rasterize_gaussians_simp(
    means3D,
    means2D,
    dc,
    sh,
    colors_precomp,
    opacities,
    scales,
    rotations,
    cov3Ds_precomp,
    culling,
    raster_settings,
):
    return _RasterizeGaussians.render_simp(
        means3D,
        means2D,
        dc,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        culling,
        raster_settings,
    )



def rasterize_gaussians_depth(
    means3D,
    means2D,
    dc,
    sh,
    colors_precomp,
    opacities,
    scales,
    rotations,
    cov3Ds_precomp,
    culling,
    raster_settings,
):
    return _RasterizeGaussians.render_depth(
        means3D,
        means2D,
        dc,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        culling,
        raster_settings,
    )     





class _RasterizeGaussians(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means3D,
        means2D,
        dc,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        flag_max_count,
        culling,
        raster_settings,
    ):

        # Restructure arguments the way that the C++ lib expects them
        args = (
            raster_settings.bg, 
            means3D,
            colors_precomp,
            opacities,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3Ds_precomp,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.image_height,
            raster_settings.image_width,
            dc,
            sh,
            raster_settings.sh_degree,
            raster_settings.campos,
            raster_settings.prefiltered,
            flag_max_count,
            culling,
            raster_settings.debug,
        )

        # Invoke C++/CUDA rasterizer
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
            try:
                num_rendered, num_buckets, color, accum_max_count, radii, geomBuffer, binningBuffer, imgBuffer, sampleBuffer = _C.rasterize_gaussians(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_fw.dump")
                print("\nAn error occured in forward. Please forward snapshot_fw.dump for debugging.")
                raise ex
        else:
            num_rendered, num_buckets, color, accum_max_count, radii, geomBuffer, binningBuffer, imgBuffer, sampleBuffer = _C.rasterize_gaussians(*args)

        # Keep relevant tensors for backward
        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        ctx.num_buckets = num_buckets
        ctx.save_for_backward(colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, dc, sh, geomBuffer, binningBuffer, imgBuffer, sampleBuffer)

        return color, radii, accum_max_count
    

    def render_simp(
        means3D,
        means2D,
        dc,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        culling,
        raster_settings,
    ):

        # Restructure arguments the way that the C++ lib expects them
        args = (
            raster_settings.bg, 
            means3D,
            colors_precomp,
            opacities,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3Ds_precomp,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.image_height,
            raster_settings.image_width,
            dc,
            sh,
            raster_settings.sh_degree,
            raster_settings.campos,
            raster_settings.prefiltered,
            culling,
            raster_settings.debug,
        )


        num_rendered, color, accum_weights_ptr, accum_weights_count, accum_max_count, radii, geomBuffer, binningBuffer, imgBuffer = _C.rasterize_gaussians_simp(*args)


        return color, radii, accum_weights_ptr, accum_weights_count, accum_max_count




    def render_depth(
        means3D,
        means2D,
        dc,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        culling,
        raster_settings,
    ):

        # Restructure arguments the way that the C++ lib expects them
        args = (
            raster_settings.bg, 
            means3D,
            colors_precomp,
            opacities,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3Ds_precomp,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.image_height,
            raster_settings.image_width,
            dc,
            sh,
            raster_settings.sh_degree,
            raster_settings.campos,
            raster_settings.prefiltered,
            culling,
            raster_settings.debug,
        )

        # Invoke C++/CUDA rasterizer
        num_rendered, color, out_pts, depth, accum_alpha, gidx, discriminants, radii, geomBuffer, binningBuffer, imgBuffer = _C.rasterize_gaussians_depth(*args)

        res =  {"render": color,
                "out_pts": out_pts,
                "rendered_depth": depth,
                "discriminants": discriminants,
                "gidx": gidx,
                "accum_alpha": accum_alpha,
                }

        return res
    






    @staticmethod
    # def backward(ctx, grad_out_color, a,b,c,d):
    def backward(ctx, grad_out_color, *_):
        # Restore necessary values from context
        num_rendered = ctx.num_rendered
        num_buckets = ctx.num_buckets
        raster_settings = ctx.raster_settings
        colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, dc, sh, geomBuffer, binningBuffer, imgBuffer, sampleBuffer = ctx.saved_tensors

        # Restructure args as C++ method expects them
        args = (raster_settings.bg,
                means3D, 
                radii, 
                colors_precomp, 
                scales, 
                rotations, 
                raster_settings.scale_modifier, 
                cov3Ds_precomp, 
                raster_settings.viewmatrix, 
                raster_settings.projmatrix, 
                raster_settings.tanfovx, 
                raster_settings.tanfovy, 
                grad_out_color, 
                dc,
                sh, 
                raster_settings.sh_degree, 
                raster_settings.campos,
                geomBuffer,
                num_rendered,
                binningBuffer,
                imgBuffer,
                num_buckets,
                sampleBuffer,
                raster_settings.debug)

        # Compute gradients for relevant tensors by invoking backward method
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
            try:
                grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_dc, grad_sh, grad_scales, grad_rotations = _C.rasterize_gaussians_backward(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_bw.dump")
                print("\nAn error occured in backward. Writing snapshot_bw.dump for debugging.\n")
                raise ex
        else:
             grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_dc, grad_sh, grad_scales, grad_rotations = _C.rasterize_gaussians_backward(*args)

        grads = (
            grad_means3D,
            grad_means2D,
            grad_dc,
            grad_sh,
            grad_colors_precomp,
            grad_opacities,
            grad_scales,
            grad_rotations,
            grad_cov3Ds_precomp,
            None,
            None,
            None,
        )

        return grads

class GaussianRasterizationSettings(NamedTuple):
    image_height: int
    image_width: int 
    tanfovx : float
    tanfovy : float
    bg : torch.Tensor
    scale_modifier : float
    viewmatrix : torch.Tensor
    projmatrix : torch.Tensor
    sh_degree : int
    campos : torch.Tensor
    prefiltered : bool
    debug : bool

class GaussianRasterizer(nn.Module):
    def __init__(self, raster_settings):
        super().__init__()
        self.raster_settings = raster_settings

    def markVisible(self, positions):
        # Mark visible points (based on frustum culling for camera) with a boolean 
        with torch.no_grad():
            raster_settings = self.raster_settings
            visible = _C.mark_visible(
                positions,
                raster_settings.viewmatrix,
                raster_settings.projmatrix)
            
        return visible

    def preprocess_gaussians(self, means3D, scales, rotations, shs, opacities, cuda_args = None):
        
        raster_settings = self.raster_settings

        # Invoke C++/CUDA rasterization routine
        return preprocess_gaussians(
            means3D,
            scales,
            rotations,
            shs,
            opacities,
            raster_settings,
            cuda_args)

    def forward(self, means3D, means2D, opacities, culling, dc = None, shs = None, colors_precomp = None, scales = None, rotations = None, cov3D_precomp = None, flag_max_count=False):
        
        raster_settings = self.raster_settings

        if (shs is None and colors_precomp is None) or (shs is not None and colors_precomp is not None):
            raise Exception('Please provide excatly one of either SHs or precomputed colors!')
        
        if ((scales is None or rotations is None) and cov3D_precomp is None) or ((scales is not None or rotations is not None) and cov3D_precomp is not None):
            raise Exception('Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!')
        
        if dc is None:
            dc = torch.Tensor([])        
        if shs is None:
            shs = torch.Tensor([])
        if colors_precomp is None:
            colors_precomp = torch.Tensor([])

        if scales is None:
            scales = torch.Tensor([])
        if rotations is None:
            rotations = torch.Tensor([])
        if cov3D_precomp is None:
            cov3D_precomp = torch.Tensor([])

        # Invoke C++/CUDA rasterization routine
        return rasterize_gaussians(
            means3D,
            means2D,
            dc,
            shs,
            colors_precomp,
            opacities,
            scales, 
            rotations,
            cov3D_precomp,
            flag_max_count,
            culling,
            raster_settings,
        )
    

    def render_simp(self, means3D, means2D, opacities, culling, dc = None, shs = None, colors_precomp = None, scales = None, rotations = None, cov3D_precomp = None):
        
        raster_settings = self.raster_settings

        if (shs is None and colors_precomp is None) or (shs is not None and colors_precomp is not None):
            raise Exception('Please provide excatly one of either SHs or precomputed colors!')
        
        if ((scales is None or rotations is None) and cov3D_precomp is None) or ((scales is not None or rotations is not None) and cov3D_precomp is not None):
            raise Exception('Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!')
        
        if dc is None:
            dc = torch.Tensor([])            
        if shs is None:
            shs = torch.Tensor([])
        if colors_precomp is None:
            colors_precomp = torch.Tensor([])

        if scales is None:
            scales = torch.Tensor([])
        if rotations is None:
            rotations = torch.Tensor([])
        if cov3D_precomp is None:
            cov3D_precomp = torch.Tensor([])

        # Invoke C++/CUDA rasterization routine
        return rasterize_gaussians_simp(
            means3D,
            means2D,
            dc,
            shs,
            colors_precomp,
            opacities,
            scales, 
            rotations,
            cov3D_precomp,
            culling,
            raster_settings,
        )
    

    
    def render_depth(self, means3D, means2D, opacities, culling, dc = None, shs = None, colors_precomp = None, scales = None, rotations = None, cov3D_precomp = None):
        
        raster_settings = self.raster_settings

        if (shs is None and colors_precomp is None) or (shs is not None and colors_precomp is not None):
            raise Exception('Please provide excatly one of either SHs or precomputed colors!')
        
        if ((scales is None or rotations is None) and cov3D_precomp is None) or ((scales is not None or rotations is not None) and cov3D_precomp is not None):
            raise Exception('Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!')
        
        if dc is None:
            dc = torch.Tensor([])            
        if shs is None:
            shs = torch.Tensor([])
        if colors_precomp is None:
            colors_precomp = torch.Tensor([])

        if scales is None:
            scales = torch.Tensor([])
        if rotations is None:
            rotations = torch.Tensor([])
        if cov3D_precomp is None:
            cov3D_precomp = torch.Tensor([])

        # Invoke C++/CUDA rasterization routine
        return rasterize_gaussians_depth(
            means3D,
            means2D,
            dc,
            shs,
            colors_precomp,
            opacities,
            scales, 
            rotations,
            cov3D_precomp,
            culling,
            raster_settings,
        )   

    def render_gaussians(self, means2D, means2D_abs, conic_opacity, rgb, all_map, depths, radii, compute_locally, extended_compute_locally, cuda_args = None):

        raster_settings = self.raster_settings

        # Invoke C++/CUDA rasterization routine
        return render_gaussians(
            means2D,
            means2D_abs,
            conic_opacity,
            rgb,
            all_map, 
            depths,
            radii,
            compute_locally,
            extended_compute_locally,
            raster_settings,
            cuda_args
        ) 
    

class SparseGaussianAdam(torch.optim.Adam):
    def __init__(self, params, lr, eps):
        super().__init__(params=params, lr=lr, eps=eps)
    
    @torch.no_grad()
    def step(self, visibility, N):
        for group in self.param_groups:
            lr = group["lr"]
            eps = group["eps"]

            assert len(group["params"]) == 1, "more than one tensor in group"
            param = group["params"][0]
            if param.grad is None or torch.prod(torch.tensor(param.grad.shape))==0:
                continue

            # Lazy state initialization
            state = self.state[param]
            if len(state) == 0:
                state['step'] = torch.tensor(0.0, dtype=torch.float32)
                state['exp_avg'] = torch.zeros_like(param, memory_format=torch.preserve_format)
                state['exp_avg_sq'] = torch.zeros_like(param, memory_format=torch.preserve_format)

            stored_state = self.state.get(param, None)
            exp_avg = stored_state["exp_avg"]
            exp_avg_sq = stored_state["exp_avg_sq"]

            # compensate lr for sparse adam, (1-b2**step)**0.5/(1-b1**step)
            state['step']+=1
            step=state['step']

            M = param.numel() // N
            _C.adamUpdate(param, param.grad, exp_avg, exp_avg_sq, visibility, lr*(1-0.999**step)**0.5/(1-0.9**step), 0.9, 0.999, eps, N, M)



