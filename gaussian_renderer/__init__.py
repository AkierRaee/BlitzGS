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

import torch
import math
from scene.gaussian_model import GaussianModel
import utils.general_utils as utils
from utils.sh_utils import eval_sh
from diff_gaussian_rasterization_f import GaussianRasterizationSettings, GaussianRasterizer
import torch.distributed.nn.functional as dist_func
from pytorch3d.transforms import quaternion_to_matrix


def all_to_all_communication_final(
    batched_rasterizers,
    batched_screenspace_params,
    batched_cuda_args,
    batched_strategies,
):
    num_cameras = len(batched_rasterizers)
    # gpui_to_gpuj_camk_size
    # gpui_to_gpuj_camk_send_ids

    local_to_gpuj_camk_size = [[] for j in range(utils.DEFAULT_GROUP.size())]  
    local_to_gpuj_camk_send_ids = [[] for j in range(utils.DEFAULT_GROUP.size())] 
    for k in range(num_cameras):
        strategy = batched_strategies[k]
        means2D, rgb, conic_opacity, radii, depths, means3D, scales, rotation = batched_screenspace_params[k]
        local2j_ids, local2j_ids_bool = batched_strategies[k].get_local2j_ids(
            means2D, radii, batched_rasterizers[k].raster_settings, batched_cuda_args[k]
        ) 

        for local_id, global_id in enumerate(strategy.gpu_ids):
            local_to_gpuj_camk_size[global_id].append(len(local2j_ids[local_id]))
            local_to_gpuj_camk_send_ids[global_id].append(local2j_ids[local_id]) #

        for j in range(utils.DEFAULT_GROUP.size()):
            if len(local_to_gpuj_camk_size[j]) == k:
                local_to_gpuj_camk_size[j].append(0)
                local_to_gpuj_camk_send_ids[j].append(
                    torch.empty((0,), dtype=torch.int64, device=means2D.device)
                )

    gpui_to_gpuj_imgk_size = torch.zeros(
        (utils.DEFAULT_GROUP.size(), utils.DEFAULT_GROUP.size(), num_cameras),
        dtype=torch.int,
        device="cuda",
    )
    local_to_gpuj_camk_size_tensor = torch.tensor(
        local_to_gpuj_camk_size, dtype=torch.int, device="cuda"
    )
    torch.distributed.all_gather_into_tensor(
        gpui_to_gpuj_imgk_size,
        local_to_gpuj_camk_size_tensor,  #
        group=utils.DEFAULT_GROUP,
    )
    gpui_to_gpuj_imgk_size = gpui_to_gpuj_imgk_size.cpu().numpy().tolist()

    def one_all_to_all(batched_tensors, use_function_version=False):
        tensor_to_rki = []
        tensor_from_rki = []
        for i in range(utils.DEFAULT_GROUP.size()):
            tensor_to_rki_list = []
            tensor_from_rki_size = 0
            for k in range(num_cameras):
                tensor_to_rki_list.append(
                    batched_tensors[k][local_to_gpuj_camk_send_ids[i][k]]  #
                )
                #
                tensor_from_rki_size += gpui_to_gpuj_imgk_size[i][
                    utils.DEFAULT_GROUP.rank()
                ][k]
            tensor_to_rki.append(torch.cat(tensor_to_rki_list, dim=0).contiguous())
            tensor_from_rki.append(
                torch.empty(
                    (tensor_from_rki_size,) + batched_tensors[0].shape[1:],
                    dtype=batched_tensors[0].dtype,
                    device="cuda",
                )
            )#
        if (
            use_function_version
        ):  
            dist_func.all_to_all(
                output_tensor_list=tensor_from_rki,
                input_tensor_list=tensor_to_rki,
                group=utils.DEFAULT_GROUP,
            )  # The function version could naturally enable communication during backward.
        else:
            torch.distributed.all_to_all(
                output_tensor_list=tensor_from_rki,
                input_tensor_list=tensor_to_rki,
                group=utils.DEFAULT_GROUP,
            )

        # tensor_from_rki: (world_size, (all data received from all other GPUs))
        for i in range(utils.DEFAULT_GROUP.size()):
            # -> (world_size, num_cameras, *)
            tensor_from_rki[i] = tensor_from_rki[i].split(
                gpui_to_gpuj_imgk_size[i][utils.DEFAULT_GROUP.rank()], dim=0
            )

        tensors_per_camera = []
        for k in range(num_cameras):
            tensors_per_camera.append(
                torch.cat(
                    [tensor_from_rki[i][k] for i in range(utils.DEFAULT_GROUP.size())],
                    dim=0,
                ).contiguous()
            )

        return tensors_per_camera

    # Merge means2D, rgb, conic_opacity into one functional all-to-all communication call.
    batched_catted_screenspace_states = []
    batched_catted_screenspace_auxiliary_states = []
    
    for k in range(num_cameras):
        means2D, rgb, conic_opacity, radii, depths, means3D, scales, rotation = batched_screenspace_params[k]

        # Keep communication tensors 2D for stable downstream CUDA contracts.
        if rgb.dim() > 2:
            rgb = rgb.contiguous().view(rgb.shape[0], -1)

        if k == 0:
            mean2d_dim1 = means2D.shape[1]
            rgb_dim1 = rgb.shape[1]
            conic_opacity_dim1 = conic_opacity.shape[1]
        batched_catted_screenspace_states.append(
            torch.cat([means2D, rgb, conic_opacity, means3D, scales, rotation], dim=1).contiguous()
        )
        batched_catted_screenspace_auxiliary_states.append(
            torch.cat(
                [radii.float().unsqueeze(1), depths.unsqueeze(1)], dim=1
            ).contiguous()
        )

    batched_params_redistributed = one_all_to_all(
        batched_catted_screenspace_states, use_function_version=True
    )
    batched_means2D_redistributed = []
    batched_rgb_redistributed = []
    batched_conic_opacity_redistributed = []
    batched_means3D_redistributed = []
    batched_scales_redistributed = []
    batched_rotations_redistributed = []

    for k in range(num_cameras):
        means2D_redistributed, rgb_redistributed, conic_opacity_redistributed, means3D_redistributed, scales_redistributed, rotations_redistributed= (
            torch.split(
                batched_params_redistributed[k],
                [mean2d_dim1, rgb_dim1, conic_opacity_dim1, 3, 3, 4],
                dim=1,
            )
        )

        batched_means2D_redistributed.append(means2D_redistributed)
        batched_rgb_redistributed.append(rgb_redistributed)
        batched_conic_opacity_redistributed.append(conic_opacity_redistributed)
        batched_means3D_redistributed.append(means3D_redistributed)
        batched_scales_redistributed.append(scales_redistributed)
        batched_rotations_redistributed.append(rotations_redistributed)

    batched_radii_depth_redistributed = one_all_to_all(
        batched_catted_screenspace_auxiliary_states, use_function_version=False
    )
    batched_radii_redistributed = []
    batched_depths_redistributed = []
    for k in range(num_cameras):
        radii_redistributed, depths_redistributed  = torch.split(
            batched_radii_depth_redistributed[k], [1, 1], dim=1
        )

        batched_radii_redistributed.append(radii_redistributed.squeeze(1).int())
        batched_depths_redistributed.append(depths_redistributed.squeeze(1))

    return (
        batched_means2D_redistributed,
        batched_rgb_redistributed,
        batched_conic_opacity_redistributed,
        batched_radii_redistributed,
        batched_depths_redistributed,
        batched_means3D_redistributed,
        batched_scales_redistributed,
        batched_rotations_redistributed,
        gpui_to_gpuj_imgk_size,
        local_to_gpuj_camk_send_ids,
    )

def get_cuda_args_final(strategy, mode="train"):
    args = utils.get_args()
    iteration = utils.get_cur_iter()

    if mode == "train":
        for x in range(args.bsz):
            if (iteration + x) % args.log_interval == 1:
                iteration += x
                break
    elif mode == "test":
        iteration = -1
    else:
        raise ValueError("mode should be train or test.")

    cuda_args = {
        "mode": mode,
        "world_size": str(utils.WORLD_SIZE),
        "global_rank": str(utils.GLOBAL_RANK),
        "local_rank": str(utils.LOCAL_RANK),
        "mp_world_size": str(strategy.world_size),
        "mp_rank": str(strategy.rank),
        "log_folder": args.log_folder,
        "log_interval": str(args.log_interval),
        "iteration": str(iteration),
        "avoid_pixel_all2all": False,
        "stats_collector": {},
    }
    return cuda_args

def distributed_preprocess3dgs_and_all2all_final(
    batched_viewpoint_cameras,
    pc,  # standard GaussianModel
    pipe,
    bg_color: torch.Tensor,
    scaling_modifier=1.0,
    batched_voxel_mask=None, 
    batched_nearest_cameras=None,
    batched_nearest_voxel_mask=None,
    retain_grad=False,
    batched_strategies=None,
    mode="train",
    return_plane: bool = True,
    iterations=0
):
    timers = utils.get_timers()
    args = utils.get_args()

    assert utils.DEFAULT_GROUP.size() == 1 or (
        args.gaussians_distribution and args.image_distribution
    ), "Ensure distributed training given multiple GPU. "

    if timers is not None:
        timers.start("forward_prepare_gaussians")

    batched_means3D, batched_opacity, batched_scales = [], [], []
    batched_rotations, batched_colors = [],[]
    batched_neural_opacity, batched_mask = [],[]
    batched_rasterizers, batched_cuda_args, batched_screenspace_params = [], [],[]
    batched_means2D, batched_radii, batched_cuda_args_test = [], [],[]
    batched_lod_masks = []

    batched_means3D_nearest, batched_opacity_nearest, batched_scales_nearest = [], [],[]
    batched_rotations_nearest, batched_colors_nearest = [],[]
    batched_means2D_nearest, batched_rasterizers_nearest = [],[]
    batched_screenspace_params_nearest, batched_radii_nearest = [],[]

    if timers is not None:
        timers.start("forward_preprocess_gaussians")

    xyz = pc.get_xyz
    color = pc.get_features
    opacity = pc.get_opacity
    scaling = pc.get_scaling
    rot = pc.get_rotation

    for viewpoint_camera, viewpoint_camera_nearest, visible_mask, visible_mask_nearest, strategy in zip(
        batched_viewpoint_cameras, batched_nearest_cameras, batched_voxel_mask, batched_nearest_voxel_mask, batched_strategies):

        _densify_until = getattr(args, 'densify_until_iter', 0)
        if mode == "train" and iterations < _densify_until:
            lod_indices = None
        else:
            lod_indices = pc.get_lod_indices(viewpoint_camera.camera_center, iterations)
        if (mode == "train"
                and getattr(args, 'ms_culling_in_render', False)
                and getattr(args, 'use_ms_culling', False)
                and getattr(pc, '_culling', None) is not None
                and iterations >= _densify_until):
            assert viewpoint_camera.uid < pc._culling_num_views, (
                f"camera uid {viewpoint_camera.uid} out of range "
                f"for _culling_num_views={pc._culling_num_views}")
            view_cull = pc._get_culling_view(viewpoint_camera.uid)  # bool [N_full]
            if lod_indices is None:
                lod_indices = (~view_cull).nonzero(as_tuple=False).squeeze(1)
            else:
                lod_indices = lod_indices[~view_cull[lod_indices]]

        if lod_indices is not None:
            xyz_cam     = xyz.index_select(0, lod_indices)
            color_cam   = color.index_select(0, lod_indices)
            opacity_cam = opacity.index_select(0, lod_indices)
            scaling_cam = scaling.index_select(0, lod_indices)
            rot_cam     = rot.index_select(0, lod_indices)
        else:
            xyz_cam, color_cam, opacity_cam, scaling_cam, rot_cam = xyz, color, opacity, scaling, rot
        batched_lod_masks.append(lod_indices)

        batched_means3D.append(xyz_cam)
        batched_colors.append(color_cam)
        batched_opacity.append(opacity_cam)
        batched_scales.append(scaling_cam)
        batched_rotations.append(rot_cam)
        batched_neural_opacity.append(None) # Padded for compatibility with downstream dict unpacking
        batched_mask.append(None)

        cuda_args = get_cuda_args_final(strategy, mode)
        batched_cuda_args.append(cuda_args)

        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx, tanfovy=tanfovy, bg=bg_color,
            scale_modifier=scaling_modifier, viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform, sh_degree=pc.active_sh_degree,
            campos=viewpoint_camera.camera_center, prefiltered=False, debug=pipe.debug,
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        means2D, rgb_precomp, conic_opacity, radii, depths = rasterizer.preprocess_gaussians(
            means3D=xyz_cam, scales=scaling_cam, rotations=rot_cam, shs=color_cam, opacities=opacity_cam, cuda_args=cuda_args,
        )

        if mode == "train":
            means2D.retain_grad()

        batched_means2D.append(means2D)
        screenspace_params =[means2D, rgb_precomp, conic_opacity, radii, depths, xyz_cam, scaling_cam, rot_cam]
        batched_rasterizers.append(rasterizer)
        batched_screenspace_params.append(screenspace_params)
        batched_radii.append(radii)

    utils.check_initial_gpu_memory_usage("after forward_preprocess_gaussians")
    if timers is not None: timers.stop("forward_preprocess_gaussians")

    if utils.DEFAULT_GROUP.size() == 1:
        pass 

    if timers is not None: timers.start("forward_all_to_all_communication")
    
    (
        batched_means2D_redistributed, batched_rgb_redistributed, batched_conic_opacity_redistributed,
        batched_radii_redistributed, batched_depths_redistributed, batched_means3D_redistributed,
        batched_scales_redistributed, batched_rotations_redistributed,
        gpui_to_gpuj_imgk_size, local_to_gpuj_camk_send_ids,
    ) = all_to_all_communication_final(
        batched_rasterizers, batched_screenspace_params, batched_cuda_args, batched_strategies,
    )

    if len(batched_rasterizers_nearest) != 0:
        (
            batched_means2D_redistributed_nearest, batched_rgb_redistributed_nearest, batched_conic_opacity_redistributed_nearest,
            batched_radii_redistributed_nearest, batched_depths_redistributed_nearest, batched_means3D_redistributed_nearest,
            batched_scales_redistributed_nearest, batched_rotations_redistributed_nearest, _, _,
        ) = all_to_all_communication_final(
            batched_rasterizers_nearest, batched_screenspace_params_nearest, batched_cuda_args_test, batched_strategies,
        )
    else: 
        batched_means2D_redistributed_nearest = batched_rgb_redistributed_nearest = batched_conic_opacity_redistributed_nearest = None
        batched_radii_redistributed_nearest = batched_depths_redistributed_nearest = batched_means3D_redistributed_nearest = None
        batched_scales_redistributed_nearest = batched_rotations_redistributed_nearest = batched_rasterizers_nearest = batched_cuda_args_test = None

    utils.check_initial_gpu_memory_usage("after forward_all_to_all_communication")
    if timers is not None: timers.stop("forward_all_to_all_communication")
    n_total = pc.get_xyz.shape[0]
    full_radii_list = []
    full_means2D_list = []
    for _r, _m2d, _lod in zip(batched_radii, batched_means2D, batched_lod_masks):
        if _lod is not None:
            _full_r = torch.zeros(n_total, dtype=_r.dtype, device=_r.device)
            _full_r[_lod] = _r
            _full_m2d = torch.zeros(n_total, *_m2d.shape[1:], dtype=_m2d.dtype, device=_m2d.device)
            _full_m2d[_lod] = _m2d.detach()  # detach: grad tracked via original filtered tensor
            full_radii_list.append(_full_r)
            full_means2D_list.append(_full_m2d)
        else:
            full_radii_list.append(_r)
            full_means2D_list.append(_m2d)

    batched_screenspace_pkg = {
        "batched_locally_preprocessed_mean2D": batched_means2D,  # original (grad-enabled)
        "batched_locally_preprocessed_visibility_filter": [r > 0 for r in full_radii_list],
        "batched_locally_preprocessed_radii": full_radii_list,
        "batched_locally_opacity": batched_neural_opacity, # Now Nones, won't break dictionary extraction
        "batched_locally_offset_mask": batched_mask,       # Now Nones
        "batched_locally_voxel_mask": batched_voxel_mask,
        "batched_lod_masks": batched_lod_masks,            # Per-camera LOD index tensors (None when LOD inactive)
        "batched_rasterizers": batched_rasterizers,
        "batched_cuda_args": batched_cuda_args,
        "batched_cuda_args_test": batched_cuda_args_test,
        "batched_means2D_redistributed": batched_means2D_redistributed,
        "batched_rgb_redistributed": batched_rgb_redistributed,
        "batched_conic_opacity_redistributed": batched_conic_opacity_redistributed,
        "batched_radii_redistributed": batched_radii_redistributed,
        "batched_depths_redistributed": batched_depths_redistributed,
        "batched_means3D_redistributed": batched_means3D_redistributed,
        "batched_scales_redistributed": batched_scales_redistributed,
        "batched_rotations_redistributed": batched_rotations_redistributed,
        "batched_means2D_redistributed_nearest":  batched_means2D_redistributed_nearest,
        "batched_rgb_redistributed_nearest" : batched_rgb_redistributed_nearest, 
        "batched_conic_opacity_redistributed_nearest": batched_conic_opacity_redistributed_nearest,
        "batched_radii_redistributed_nearest": batched_radii_redistributed_nearest,
        "batched_depths_redistributed_nearest": batched_depths_redistributed_nearest,
        "batched_means3D_redistributed_nearest": batched_means3D_redistributed_nearest,
        "batched_scales_redistributed_nearest": batched_scales_redistributed_nearest,
        "batched_rotations_redistributed_nearest": batched_rotations_redistributed_nearest,
        "batched_rasterizers_nearest": batched_rasterizers_nearest,
        "gpui_to_gpuj_imgk_size": gpui_to_gpuj_imgk_size,
        "local_to_gpuj_camk_send_ids": local_to_gpuj_camk_send_ids,
    }
    return batched_screenspace_pkg

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    from diff_gaussian_rasterization_r import GaussianRasterizationSettings, GaussianRasterizer
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii}


def render_imp(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, flag_max_count=True, culling=None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    from diff_gaussian_rasterization_f import GaussianRasterizationSettings, GaussianRasterizer
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            dc, shs = pc.get_features_dc, pc.get_features_rest
    else:
        colors_precomp = override_color

    if culling==None:
        culling=torch.zeros(means3D.shape[0], dtype=torch.bool, device='cuda')

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii, accum_max_count  = rasterizer(
        means3D = means3D,
        means2D = means2D,
        dc = dc,
        shs = shs,
        culling = culling,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp,
        flag_max_count=flag_max_count)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : (radii > 0).nonzero(),
            "radii": radii,
            "area_max": accum_max_count,
            }


def render_simp(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, culling=None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    from diff_gaussian_rasterization_f import GaussianRasterizationSettings, GaussianRasterizer
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            dc, shs = pc.get_features_dc, pc.get_features_rest
    else:
        colors_precomp = override_color

    if culling==None:
        culling=torch.zeros(means3D.shape[0], dtype=torch.bool, device='cuda')

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii, \
    accum_weights_ptr, accum_weights_count, accum_max_count  = rasterizer.render_simp(
        means3D = means3D,
        means2D = means2D,
        dc = dc,
        shs = shs,
        culling = culling,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : (radii > 0).nonzero(),
            "radii": radii,
            "accum_weights": accum_weights_ptr,
            "area_proj": accum_weights_count,
            "area_max": accum_max_count,
            }



def render_final(batched_cameras, batched_screenspace_pkg, batched_strategies,  tile_size=16 , batched_cameras_nearest =None, rasterizer_nearest = None):
    """
    Render the scene.
    """
    timers = utils.get_timers()
    args = utils.get_args()

    batched_rendered_image = []
    batched_compute_locally = []
    batched_out_observe = []
    batched_out_all_map = []
    batched_out_plane_depth = []
    batched_return_dict = []
    batched_return_dict_nearest = []

    def _normalize_and_check_render_inputs(means2d, rgb, conic_opacity, radii, depths, name_prefix):
        if rgb.dim() == 3 and rgb.shape[1] == 1:
            rgb = rgb.squeeze(1)
        if rgb.dim() != 2:
            raise RuntimeError(f"{name_prefix}: rgb must be 2D [P, C], got {tuple(rgb.shape)}")
        if conic_opacity.dim() == 3 and conic_opacity.shape[1] == 1:
            conic_opacity = conic_opacity.squeeze(1)
        if conic_opacity.dim() != 2:
            raise RuntimeError(f"{name_prefix}: conic_opacity must be 2D [P, C], got {tuple(conic_opacity.shape)}")
        if means2d.dim() != 2:
            raise RuntimeError(f"{name_prefix}: means2D must be 2D [P, C], got {tuple(means2d.shape)}")
        if radii.dim() != 1:
            raise RuntimeError(f"{name_prefix}: radii must be 1D [P], got {tuple(radii.shape)}")
        if depths.dim() != 1:
            raise RuntimeError(f"{name_prefix}: depths must be 1D [P], got {tuple(depths.shape)}")

        p = means2d.shape[0]
        if not (rgb.shape[0] == conic_opacity.shape[0] == radii.shape[0] == depths.shape[0] == p):
            raise RuntimeError(
                f"{name_prefix}: inconsistent batch length P, got means2D={means2d.shape[0]}, rgb={rgb.shape[0]}, conic={conic_opacity.shape[0]}, radii={radii.shape[0]}, depths={depths.shape[0]}"
            )
        if rgb.shape[1] != 3:
            raise RuntimeError(f"{name_prefix}: rgb channels must be 3, got {rgb.shape[1]}")
        if conic_opacity.shape[1] != 4:
            raise RuntimeError(f"{name_prefix}: conic_opacity channels must be 4, got {conic_opacity.shape[1]}")

        return (
            means2d.contiguous(),
            rgb.contiguous(),
            conic_opacity.contiguous(),
            radii.contiguous(),
            depths.contiguous(),
        )

    for cam_id in range(len(batched_screenspace_pkg["batched_rasterizers"])):
        strategy = batched_strategies[cam_id]
        if utils.GLOBAL_RANK not in strategy.gpu_ids:
            batched_rendered_image.append(None)
            batched_compute_locally.append(None)
            batched_out_observe.append(None)
            return_dict =  {
                "viewspace_points": None,
                "viewspace_points_abs": None,
                "visibility_filter" : None,
                "radii": None,
                "out_observe": None,
                "rendered_normal": None,
                "plane_depth": None,
                "rendered_distance": None,
                "depth_normal": None
                }
            batched_return_dict.append(return_dict)
            batched_return_dict_nearest.append(None)
            continue

        # get compute_locally to know local workload in the end2end distributed training.
        if timers is not None:
            timers.start("forward_compute_locally")
        compute_locally = strategy.get_compute_locally()
        extended_compute_locally = strategy.get_extended_compute_locally()
        if timers is not None:
            timers.stop("forward_compute_locally")

        rasterizer = batched_screenspace_pkg["batched_rasterizers"][cam_id]
        cuda_args = batched_screenspace_pkg["batched_cuda_args"][cam_id]
        means2D_redistributed = batched_screenspace_pkg[
            "batched_means2D_redistributed"
        ][cam_id]
        rgb_redistributed = batched_screenspace_pkg["batched_rgb_redistributed"][cam_id]
        conic_opacity_redistributed = batched_screenspace_pkg[
            "batched_conic_opacity_redistributed"
        ][cam_id]
        radii_redistributed = batched_screenspace_pkg["batched_radii_redistributed"][
            cam_id
        ]
        depths_redistributed = batched_screenspace_pkg["batched_depths_redistributed"][
            cam_id
        ]
        xyz_redistributed = batched_screenspace_pkg["batched_means3D_redistributed"][cam_id]
        scales_redistributed = batched_screenspace_pkg["batched_scales_redistributed"][cam_id]
        rotations_redistributed = batched_screenspace_pkg["batched_rotations_redistributed"][cam_id]

        (
            means2D_redistributed,
            rgb_redistributed,
            conic_opacity_redistributed,
            radii_redistributed,
            depths_redistributed,
        ) = _normalize_and_check_render_inputs(
            means2D_redistributed,
            rgb_redistributed,
            conic_opacity_redistributed,
            radii_redistributed,
            depths_redistributed,
            f"cam={cam_id}",
        )
        if means2D_redistributed.shape[0] < 10:
            # That means we do not have enough gaussians locally for rendering, that mainly happens because of insufficient initial points.
            rendered_image = (
                means2D_redistributed.sum()
                + conic_opacity_redistributed.sum()
                + rgb_redistributed.sum()
            )
            cuda_args["stats_collector"]["forward_render_time"] = 0.0
            cuda_args["stats_collector"]["backward_render_time"] = 0.0
            cuda_args["stats_collector"]["forward_loss_time"] = 0.0

            return_dict =  {
                "viewspace_points": None,
                "viewspace_points_abs": None,
                "visibility_filter" : None,
                "radii": None,
                "out_observe": None,
                "rendered_normal": None,
                "plane_depth": None,
                "rendered_distance": None,
                "depth_normal": None
                }
            out_all_map = None
            out_observe = None
            out_plane_depth = None
            nearest_render_pkg = None
        else:
            nearest_render_pkg = None
            input_all_map = torch.zeros((means2D_redistributed.shape[0], 5)).cuda().float()
            # render
            if timers is not None:
                timers.start("forward_render_gaussians")

            screenspace_points_abs = torch.zeros_like(xyz_redistributed, dtype=means2D_redistributed.dtype, requires_grad=True, device="cuda") + 0
            try:
                screenspace_points_abs.retain_grad()
                # screenspace_points_abs.retain_grad()
            except:
                pass
            means2D_abs = screenspace_points_abs
            rendered_image, n_render, n_consider, n_contrib, out_observe, out_all_map, out_plane_depth = (
                rasterizer.render_gaussians(
                    means2D=means2D_redistributed,
                    means2D_abs= means2D_abs,
                    conic_opacity=conic_opacity_redistributed,
                    rgb=rgb_redistributed,
                    all_map = input_all_map,
                    depths=depths_redistributed,
                    radii=radii_redistributed,
                    compute_locally=compute_locally,
                    extended_compute_locally=extended_compute_locally,
                    cuda_args=cuda_args,
                )
            )
            rendered_normal = None
            rendered_alpha = None
            rendered_distance = None
            depth_normal = None
            return_dict =  {
                        "viewspace_points": means2D_redistributed,
                        "viewspace_points_abs": means2D_abs,
                        "visibility_filter" : radii_redistributed > 0,
                        "radii": radii_redistributed,
                        "out_observe": out_observe,
                        "rendered_normal": rendered_normal,
                        "plane_depth": None,
                        "rendered_distance": rendered_distance,
                        "depth_normal": depth_normal,
                        "scales_redistributed": scales_redistributed,
                        "rendered_alpha": rendered_alpha
                        }
        batched_rendered_image.append(rendered_image)
        batched_compute_locally.append(True)
        batched_out_all_map.append(out_all_map)
        batched_out_observe.append(out_observe)
        batched_out_plane_depth.append(out_plane_depth)
        batched_return_dict.append(return_dict)
        batched_return_dict_nearest.append(nearest_render_pkg)

        if timers is not None:
            timers.stop("forward_render_gaussians")
    utils.check_initial_gpu_memory_usage("after forward_render_gaussians")

    ########## [END] CUDA Rasterization Call ##########
    return batched_rendered_image, batched_compute_locally, batched_out_all_map, batched_out_observe, batched_out_plane_depth, batched_return_dict, batched_return_dict_nearest
