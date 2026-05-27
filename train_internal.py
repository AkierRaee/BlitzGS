import os
import time
import torch
import json
from utils.loss_utils import l1_loss

from torch.cuda import nvtx
from scene import Scene, GaussianModel, SceneDataset
from gaussian_renderer.workload_division import (
    start_strategy_final,
    finish_strategy_final,
    DivisionStrategyHistoryFinal,
)
from gaussian_renderer.loss_distribution import (
    load_camera_from_cpu_to_all_gpu,
    load_camera_from_cpu_to_all_gpu_for_eval,
    batched_loss_computation,
)
from utils.general_utils import prepare_output_and_logger, globally_sync_for_timer
import utils.general_utils as utils
from utils.timer import Timer, End2endTimer
from tqdm import tqdm
from utils.image_utils import psnr
import torch.distributed as dist
import torchvision
from os import makedirs
import cv2
import numpy as np
from utils.loss_utils import ssim
from lpipsPyTorch import lpips
from gaussian_renderer import distributed_preprocess3dgs_and_all2all_final, render_final, render_simp

_FINITE_CHECK = os.environ.get("BLITZGS_FINITE_CHECK", "0") == "1"

def _finite_check(gaussians, label, iteration):
    if not _FINITE_CHECK:
        return
    for name, t in (("_xyz", gaussians._xyz), ("_scaling", gaussians._scaling),
                    ("_rotation", gaussians._rotation), ("_opacity", gaussians._opacity)):
        if not torch.isfinite(t).all():
            bad = (~torch.isfinite(t)).sum().item()
            raise RuntimeError(f"[finite_check {label} iter={iteration} rank={utils.LOCAL_RANK}] "
                               f"{name} has {bad} non-finite values (shape={tuple(t.shape)})")

def visualize_scalars(scalar_tensor: torch.Tensor) -> np.ndarray:
    to_use = scalar_tensor.view(-1)
    while to_use.shape[0] > 2 ** 24:
        to_use = to_use[::2]

    mi = torch.quantile(to_use, 0.05)
    ma = torch.quantile(to_use, 0.95)

    scalar_tensor = (scalar_tensor - mi) / max(ma - mi, 1e-8)  # normalize to 0~1
    scalar_tensor = scalar_tensor.clamp_(0, 1)

    scalar_tensor = ((1 - scalar_tensor) * 255).byte().numpy()  # inverse heatmap
    return cv2.cvtColor(cv2.applyColorMap(scalar_tensor, cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB)


def training(dataset_args, opt_args, pipe_args, args, log_file):

    # Init auxiliary tools
    timers = Timer(args)
    utils.set_timers(timers)
    prepare_output_and_logger(dataset_args)
    utils.log_cpu_memory_usage("at the beginning of training")
    start_from_this_iteration = 1

    # Init parameterized scene (standard 3DGS init)
    gaussians = GaussianModel(dataset_args.sh_degree)
    
    with torch.no_grad():
        scene = Scene(args, gaussians, args.load_iteration)
        gaussians.training_setup(opt_args)
        
        # Per-camera culling initialization
        if hasattr(gaussians, 'init_culling'):
            gaussians.init_culling(len(scene.getTrainCameras()))
            
        if args.start_checkpoint != "":
            model_params, start_from_this_iteration = utils.load_checkpoint(args)
            gaussians.restore(model_params, opt_args)
            utils.print_rank_0("Restored from checkpoint: {}".format(args.start_checkpoint))
            log_file.write("Restored from checkpoint: {}\n".format(args.start_checkpoint))

        scene.log_scene_info_to_file(log_file, "Scene Info Before Training")
    utils.check_initial_gpu_memory_usage("after init and before training loop")

    train_dataset = SceneDataset(scene.getTrainCameras())
    if args.adjust_strategy_warmp_iterations == -1:
        args.adjust_strategy_warmp_iterations = len(train_dataset.cameras)

    strategy_history = DivisionStrategyHistoryFinal(
        train_dataset, utils.DEFAULT_GROUP.size(), utils.DEFAULT_GROUP.rank()
    )

    background = None
    bg_color = [1, 1, 1] if dataset_args.white_background else[0, 0, 0]
    if bg_color is not None:
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    end2end_timers = End2endTimer(args)
    end2end_timers.start()
    progress_bar = tqdm(
        range(1, opt_args.iterations + 1),
        desc="Training progress",
        disable=(utils.LOCAL_RANK != 0),
    )
    progress_bar.update(start_from_this_iteration - 1)
    num_trained_batches = 0
    saved_milestones = set()

    ema_loss_for_log = 0

    # --- Perf instrumentation (memory + speed) ---
    torch.cuda.reset_peak_memory_stats()
    _perf_mem_sum_mb = 0.0
    _perf_mem_samples = 0
    _perf_iters_done = 0
    torch.cuda.synchronize()
    _perf_t0 = time.time()

    for iteration in range(start_from_this_iteration, opt_args.iterations + 1, args.bsz):
        if iteration // args.bsz % 30 == 0:
            progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
        progress_bar.update(args.bsz)
        utils.set_cur_iter(iteration)
        gaussians.update_learning_rate(iteration)
        num_trained_batches += 1
        timers.clear()
        
        if args.nsys_profile:
            nvtx.range_push(f"iteration[{iteration},{iteration+args.bsz})")
            
        if utils.check_update_at_this_iter(iteration, args.bsz, 1000, 0):
            gaussians.oneupSHdegree()

        if args.local_sampling:
            assert args.bsz % utils.WORLD_SIZE == 0
            batched_cameras_idx = train_dataset.get_batched_cameras_idx(args.bsz // utils.WORLD_SIZE)
            batched_all_cameras_idx = torch.zeros((utils.WORLD_SIZE, len(batched_cameras_idx)), device="cuda", dtype=int)
            batched_cameras_idx = torch.tensor(batched_cameras_idx, device="cuda", dtype=int)
            torch.distributed.all_gather_into_tensor(batched_all_cameras_idx, batched_cameras_idx, group=utils.DEFAULT_GROUP)
            batched_all_cameras_idx = batched_all_cameras_idx.cpu().numpy().squeeze()
            batched_cameras = train_dataset.get_batched_cameras_from_idx(batched_all_cameras_idx)
        else:
            batched_cameras, batched_nearest_cameras = train_dataset.get_batched_cameras(args.bsz)

        with torch.no_grad():
            timers.start("prepare_strategies")
            batched_strategies, gpuid2tasks = start_strategy_final(batched_cameras, strategy_history)
            timers.stop("prepare_strategies")

            timers.start("load_cameras")
            load_camera_from_cpu_to_all_gpu(batched_cameras, batched_strategies, gpuid2tasks)
            timers.stop("load_cameras")

        batched_voxel_mask =[None for _ in batched_cameras]
        batched_nearest_voxel_mask =[None for _ in batched_nearest_cameras]
        
        retain_grad = (iteration < opt_args.update_until and iteration >= 0)

        batched_screenspace_pkg = distributed_preprocess3dgs_and_all2all_final(
            batched_cameras,
            gaussians,
            pipe_args,
            background,
            batched_voxel_mask = batched_voxel_mask,
            batched_nearest_cameras = batched_nearest_cameras,
            batched_nearest_voxel_mask = batched_nearest_voxel_mask,
            retain_grad=retain_grad,
            batched_strategies=batched_strategies,
            mode="train",
            return_plane = False,
            iterations = iteration,
        )
        
        batched_image, batched_compute_locally, batched_out_all_map, batched_out_observe, batched_out_plane_depth, batched_return_dict, batched_return_dict_nearest = render_final(
            batched_cameras,
            batched_screenspace_pkg, 
            batched_strategies, 
            gaussians, 
            batched_cameras_nearest = batched_nearest_cameras
        )
        
        batch_statistic_collector = [cuda_args["stats_collector"] for cuda_args in batched_screenspace_pkg["batched_cuda_args"]]
        
        loss_sum, batched_losses = batched_loss_computation(
            batched_image,
            batched_return_dict,
            batched_cameras,
            batched_strategies,
            batch_statistic_collector,
            iterations=iteration,
            opt=opt_args,
        )

        timers.start("backward")
        loss_sum.backward()
        timers.stop("backward")
        utils.check_initial_gpu_memory_usage("after backward")

        with torch.no_grad():
            globally_sync_for_timer()
            timers.start("finish_strategy_final")
            finish_strategy_final(batched_cameras, strategy_history, batched_strategies, batch_statistic_collector)
            timers.stop("finish_strategy_final")

            timers.start("sync_loss_and_log")
            batched_losses = torch.tensor(batched_losses, device="cuda")
            if utils.DEFAULT_GROUP.size() > 1:
                dist.all_reduce(batched_losses, op=dist.ReduceOp.SUM, group=utils.DEFAULT_GROUP)
            batched_loss = (1.0 - args.lambda_dssim) * batched_losses[:, 0] + args.lambda_dssim * (1.0 - batched_losses[:, 1])
            batched_loss_cpu = batched_loss.cpu().numpy()
            ema_loss_for_log = batched_loss_cpu.mean() if ema_loss_for_log is None else 0.6 * ema_loss_for_log + 0.4 * batched_loss_cpu.mean()
            
            train_dataset.update_losses(batched_loss_cpu)
            batched_loss_cpu =[round(loss, 6) for loss in batched_loss_cpu]
            log_string = "iteration[{},{}) loss: {} image: {}\n".format(iteration, iteration + args.bsz, batched_loss_cpu,[cam.image_name for cam in batched_cameras])
            log_file.write(log_string)
            timers.stop("sync_loss_and_log")

            end2end_timers.stop()
            training_report(iteration, l1_loss, args.test_iterations, scene, pipe_args, background, args.backend, dataset_args.model_path)
            end2end_timers.start()

            # LOD diagnostics
            diag_interval = getattr(opt_args, 'octree_diag_interval', 0)
            if (getattr(opt_args, 'use_octree_lod', False)
                    and diag_interval > 0
                    and gaussians.octree is not None
                    and (iteration // diag_interval) != ((iteration - args.bsz) // diag_interval)
                    and utils.LOCAL_RANK == 0):
                n_kept = getattr(gaussians, '_lod_last_kept', None)
                n_total = getattr(gaussians, '_lod_last_total', None)
                active = gaussians.octree.get_active_level(iteration)
                if n_kept is not None and n_total is not None and n_total > 0:
                    ratio = n_kept / n_total
                    lod_line = f"[LOD iter={iteration}] active_level={active}/{gaussians.octree.levels - 1}  kept={n_kept}/{n_total} ({ratio:.1%})"
                else:
                    lod_line = f"[LOD iter={iteration}] active_level={active}/{gaussians.octree.levels - 1}  (no mask computed this iter)"
                print(lod_line)
                log_file.write(lod_line + "\n")

                with torch.no_grad():
                    centroid = gaussians._xyz.detach().mean(dim=0)
                    cams = train_dataset.cameras
                    dists = torch.stack([
                        (c.camera_center.to(centroid.device) - centroid).norm()
                        for c in cams
                    ])
                    order = torch.argsort(dists)
                    n_pick = min(5, len(cams))
                    pick_idx = [order[int(i * (len(cams) - 1) / max(1, n_pick - 1))].item()
                                for i in range(n_pick)]
                    n_total_p = gaussians._xyz.shape[0]
                    for ci in pick_idx:
                        cam = cams[ci]
                        mask = gaussians.octree.compute_lod_mask(
                            cam.camera_center.cuda(),
                            gaussians._xyz.detach(),
                            gaussians._lod_level,
                            iteration=iteration,
                        )
                        kept = int(mask.sum().item())
                        probe_line = (f"[LOD probe iter={iteration}] cam={cam.image_name} "
                                      f"dist_to_centroid={dists[ci].item():.1f}  "
                                      f"kept={kept}/{n_total_p} ({kept/n_total_p:.1%})")
                        print(probe_line)
                        log_file.write(probe_line + "\n")

            if iteration < opt_args.densify_until_iter:
                batched_viewspace_points = batched_screenspace_pkg.get(
                    "batched_locally_preprocessed_mean2D", []
                )
                batched_visibility_filter = batched_screenspace_pkg.get(
                    "batched_locally_preprocessed_visibility_filter", []
                )
                batched_radii = batched_screenspace_pkg.get(
                    "batched_locally_preprocessed_radii", []
                )
                batched_lod_masks = batched_screenspace_pkg.get(
                    "batched_lod_masks", []
                )

                # 1. Accumulate point statistics over the batch
                for i, camera in enumerate(batched_cameras):
                    if i >= len(batched_compute_locally) or not batched_compute_locally[i]:
                        continue # Skip if computed on a remote GPU
                    if i >= len(batched_viewspace_points) or i >= len(batched_visibility_filter) or i >= len(batched_radii):
                        continue # Skip if computed on a remote GPU

                    viewspace_points = batched_viewspace_points[i]
                    visibility_filter = batched_visibility_filter[i]
                    radii = batched_radii[i]
                    lod_indices = batched_lod_masks[i] if i < len(batched_lod_masks) else None

                    if lod_indices is not None:
                        full_vis = visibility_filter  # already full-size bool mask
                        gaussians.max_radii2D[full_vis] = torch.max(
                            gaussians.max_radii2D[full_vis], radii[full_vis]
                        )

                        if viewspace_points.grad is not None:
                            filtered_vis = radii[lod_indices] > 0
                            grad_norms = torch.norm(viewspace_points.grad[filtered_vis, :2], dim=-1, keepdim=True)
                            gaussians.xyz_gradient_accum[lod_indices[filtered_vis]] += grad_norms
                            gaussians.denom[lod_indices[filtered_vis]] += 1
                    else:
                        gaussians.max_radii2D[visibility_filter] = torch.max(
                            gaussians.max_radii2D[visibility_filter], radii[visibility_filter]
                        )
                        if opt_args.use_ms_culling and not opt_args.disable_phi_weighting and gaussians.factor_culling is not None:
                            gaussians.add_densification_stats_culling(viewspace_points, visibility_filter, gaussians.factor_culling)
                        else:
                            gaussians.add_densification_stats(viewspace_points, visibility_filter)

                gaussians.sync_gradients_for_replicated_3dgs_storage(batched_screenspace_pkg)

                if iteration > opt_args.densify_from_iter and (iteration // opt_args.densification_interval) != ((iteration - args.bsz) // opt_args.densification_interval):
                    size_threshold = 20 if iteration > opt_args.opacity_reset_interval else None

                    _n_before_local = int(gaussians.get_xyz.shape[0])
                    _n_before_total = _n_before_local
                    if utils.WORLD_SIZE > 1:
                        _t = torch.tensor([_n_before_local], dtype=torch.long, device="cuda")
                        dist.all_reduce(_t, op=dist.ReduceOp.SUM)
                        _n_before_total = int(_t.item())

                    gaussians.densify_and_prune(opt_args.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                    _finite_check(gaussians, "after_densify", iteration)

                    _n_after_local = int(gaussians.get_xyz.shape[0])
                    _n_after_total = _n_after_local
                    if utils.WORLD_SIZE > 1:
                        _t = torch.tensor([_n_after_local], dtype=torch.long, device="cuda")
                        dist.all_reduce(_t, op=dist.ReduceOp.SUM)
                        _n_after_total = int(_t.item())
                    _delta = _n_after_total - _n_before_total
                    _sign = "+" if _delta >= 0 else ""
                    if utils.LOCAL_RANK == 0:
                        _msg = (
                            "[GSCount iter={}] before={} after={} delta={}{} "
                            "(local before={} after={})\n"
                        ).format(
                            iteration, _n_before_total, _n_after_total, _sign, _delta,
                            _n_before_local, _n_after_local,
                        )
                        print(_msg, end="")
                        log_file.write(_msg)

                    if utils.WORLD_SIZE > 1:
                        gaussians.redistribute_gaussians()
                    _finite_check(gaussians, "after_redistribute", iteration)

            if (opt_args.use_ms_culling
                    and (iteration <= opt_args.simp_iteration1 < iteration + args.bsz)):
                gaussians.culling_with_interesction_sampling(scene, render_simp, iteration, opt_args, pipe_args, background)
                if utils.WORLD_SIZE > 1:
                    gaussians.redistribute_gaussians()
                utils.print_rank_0(f"[simp1 iter={iteration}] new N={gaussians._xyz.shape[0]}")

            if (opt_args.use_ms_culling
                    and (iteration <= opt_args.simp_iteration2 < iteration + args.bsz)):
                gaussians.culling_with_interesction_preserving(scene, render_simp, iteration, opt_args, pipe_args, background)
                gaussians.prune_points(gaussians.get_opacity.squeeze() < opt_args.min_opacity)
                if utils.WORLD_SIZE > 1:
                    gaussians.redistribute_gaussians()
                utils.print_rank_0(f"[simp2 iter={iteration}] new N={gaussians._xyz.shape[0]}")

            mid_phase3 = (opt_args.simp_iteration2 + opt_args.iterations) // 2
            if opt_args.use_ms_culling and (iteration <= mid_phase3 < iteration + args.bsz):
                gaussians.init_culling(len(scene.getTrainCameras()))

            matched_save_iterations = sorted(
                {
                    save_iteration
                    for save_iteration in args.save_iterations
                    if iteration <= save_iteration < iteration + args.bsz
                    and save_iteration not in saved_milestones
                }
            )
            if len(matched_save_iterations) > 0:
                end2end_timers.stop()
                for save_iteration in matched_save_iterations:
                    end2end_timers.print_time(log_file, save_iteration)
                    utils.print_rank_0("\n[ITER {}] Saving Gaussians".format(save_iteration))
                    log_file.write("[ITER {}] Saving Gaussians\n".format(save_iteration))
                    scene.save(save_iteration)
                    saved_milestones.add(save_iteration)
                end2end_timers.start()

            if any([iteration <= checkpoint_iteration < iteration + args.bsz for checkpoint_iteration in args.checkpoint_iterations]):
                end2end_timers.stop()
                utils.print_rank_0("\n[ITER {}] Saving Checkpoint".format(iteration))
                log_file.write("[ITER {}] Saving Checkpoint\n".format(iteration))
                save_folder = scene.model_path + "/checkpoints/" + str(iteration) + "/"
                if utils.DEFAULT_GROUP.rank() == 0:
                    os.makedirs(save_folder, exist_ok=True)
                if utils.DEFAULT_GROUP.size() > 1:
                    torch.distributed.barrier(group=utils.DEFAULT_GROUP)
                torch.save(
                    (gaussians.capture(), iteration + args.bsz),
                    save_folder + "/chkpnt_ws=" + str(utils.WORLD_SIZE) + "_rk=" + str(utils.GLOBAL_RANK) + ".pth",
                )
                end2end_timers.start()

            # Optimizer step
            if iteration < opt_args.iterations:
                timers.start("optimizer_step")
                if args.lr_scale_mode != "accumu":  
                    if hasattr(gaussians, "all_parameters"):
                        params_iter = gaussians.all_parameters()
                    else:
                        params_iter = [
                            p
                            for group in gaussians.optimizer.param_groups
                            for p in group["params"]
                        ]
                    for param in params_iter:
                        if param.grad is not None:
                            param.grad /= args.bsz
                if not args.stop_update_param:
                    if gaussians.optimizer.__class__.__name__ == "SparseGaussianAdam":
                        visibility = batched_screenspace_pkg.get(
                            "sum_batched_locally_preprocessed_visibility_filter_int", None
                        )
                        if visibility is None:
                            vis_list = batched_screenspace_pkg.get(
                                "batched_locally_preprocessed_visibility_filter", []
                            )
                            if len(vis_list) > 0:
                                visibility = torch.sum(
                                    torch.stack([x.int() for x in vis_list]), dim=0
                                )
                            else:
                                visibility = torch.ones(
                                    gaussians.get_xyz.shape[0],
                                    dtype=torch.int,
                                    device="cuda",
                                )
                        # Sparse CUDA Adam expects a 1D bool mask of length N.
                        visibility = visibility.reshape(-1)
                        if visibility.shape[0] != gaussians.get_xyz.shape[0]:
                            visibility = torch.ones(
                                gaussians.get_xyz.shape[0],
                                dtype=torch.bool,
                                device="cuda",
                            )
                        elif visibility.dtype != torch.bool:
                            visibility = visibility > 0
                        visibility = visibility.contiguous()
                        gaussians.optimizer.step(visibility, gaussians.get_xyz.shape[0])
                    else:
                        gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)
                timers.stop("optimizer_step")

        # Finish iteration clean up
        torch.cuda.synchronize()
        for viewpoint_cam in batched_cameras:
            viewpoint_cam.original_image = None
            if args.distributed_dataset_storage:
                viewpoint_cam.image_gray = None 
                viewpoint_cam.invdepthmap = None
        if args.nsys_profile:
            nvtx.range_pop()
        if utils.check_enable_python_timer():
            timers.printTimers(iteration, mode="sum")
        log_file.flush()

        # Per-iteration memory sample (MB, currently-allocated, cheap)
        _perf_mem_sum_mb += torch.cuda.memory_allocated() / (1024 * 1024)
        _perf_mem_samples += 1
        _perf_iters_done += 1

    torch.cuda.synchronize()
    _perf_elapsed = max(time.time() - _perf_t0, 1e-9)
    _perf_peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    _perf_avg_mb = (_perf_mem_sum_mb / _perf_mem_samples) if _perf_mem_samples > 0 else 0.0
    _perf_iters_per_sec = _perf_iters_done / _perf_elapsed
    _perf_sec_per_iter = _perf_elapsed / max(_perf_iters_done, 1)
    _perf_summary = (
        "[Perf] peak_mem={:.1f} MB  avg_mem={:.1f} MB  "
        "elapsed={:.1f} s  iters={}  speed={:.2f} it/s ({:.4f} s/it)\n"
    ).format(
        _perf_peak_mb, _perf_avg_mb, _perf_elapsed,
        _perf_iters_done, _perf_iters_per_sec, _perf_sec_per_iter,
    )
    print(_perf_summary, end="")
    log_file.write(_perf_summary)

    # Final Gaussian count summary (local shard and global total).
    _n_final_local = int(gaussians.get_xyz.shape[0])
    _n_final_total = _n_final_local
    if utils.WORLD_SIZE > 1:
        _t = torch.tensor([_n_final_local], dtype=torch.long, device="cuda")
        dist.all_reduce(_t, op=dist.ReduceOp.SUM)
        _n_final_total = int(_t.item())
    _final_msg = "[GSCount final] local={} total={}\n".format(
        _n_final_local, _n_final_total
    )
    if utils.LOCAL_RANK == 0:
        print(_final_msg, end="")
    log_file.write(_final_msg)

    if opt_args.iterations not in args.save_iterations:
        end2end_timers.print_time(log_file, opt_args.iterations)
    log_file.write("Max Memory usage: {} GB.\n".format(torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024))
    progress_bar.close()


def training_report(iteration, l1_loss, testing_iterations, scene: Scene, pipe_args, background, backend, model_path):
    args = utils.get_args()
    log_file = utils.get_log_file()
    
    while len(testing_iterations) > 0 and iteration > testing_iterations[0]:
        testing_iterations.pop(0)
        
    if len(testing_iterations) > 0 and utils.check_update_at_this_iter(iteration, utils.get_args().bsz, testing_iterations[0], 0):
        testing_iterations.pop(0)
        utils.print_rank_0("\n[ITER {}] Start Testing".format(iteration))

        validation_configs = (
            {"name": "test", "cameras": scene.getTestCameras(), "num_cameras": len(scene.getTestCameras())},
            {"name": "train", "cameras": scene.getTrainCameras(), "num_cameras": max(20, args.bsz)},
        )

        for config in validation_configs:
            render_path = os.path.join(model_path, config["name"], "ours_{}".format(iteration), "renders")
            gts_path = os.path.join(model_path, config["name"], "ours_{}".format(iteration), "gt")
            depths_path = os.path.join(model_path, config["name"], "ours_{}".format(iteration), "depths")
            render_normal_path = os.path.join(model_path, config["name"], "ours_{}".format(iteration), "normals")
            rendered_distance_path = os.path.join(model_path, config["name"], "ours_{}".format(iteration), "distance")
            
            makedirs(render_path, exist_ok=True)
            makedirs(gts_path, exist_ok=True)
            makedirs(depths_path, exist_ok=True)
            makedirs(render_normal_path, exist_ok=True)
            makedirs(rendered_distance_path, exist_ok=True)
            
            if config["cameras"] and len(config["cameras"]) > 0:
                l1_test = torch.scalar_tensor(0.0, device="cuda")
                psnr_test = torch.scalar_tensor(0.0, device="cuda")
                num_cameras = config["num_cameras"] 
                eval_dataset = SceneDataset(config["cameras"])
                strategy_history = DivisionStrategyHistoryFinal(eval_dataset, utils.DEFAULT_GROUP.size(), utils.DEFAULT_GROUP.rank())
                
                for idx in range(1, num_cameras + 1, args.bsz):
                    num_camera_to_load = min(args.bsz, num_cameras - idx + 1)
                    if args.local_sampling:
                        batched_cameras_idx = eval_dataset.get_batched_cameras_idx(args.bsz // utils.WORLD_SIZE)
                        batched_all_cameras_idx = torch.zeros((utils.WORLD_SIZE, len(batched_cameras_idx)), device="cuda", dtype=int)
                        batched_cameras_idx = torch.tensor(batched_cameras_idx, device="cuda", dtype=int)
                        torch.distributed.all_gather_into_tensor(batched_all_cameras_idx, batched_cameras_idx, group=utils.DEFAULT_GROUP)
                        batched_all_cameras_idx = batched_all_cameras_idx.cpu().numpy().squeeze()
                        batched_cameras, _ = eval_dataset.get_batched_cameras_from_idx(batched_all_cameras_idx)
                    else:
                        batched_cameras, _ = eval_dataset.get_batched_cameras(num_camera_to_load , eval=True)
                        
                    batched_strategies, gpuid2tasks = start_strategy_final(batched_cameras, strategy_history)
                    load_camera_from_cpu_to_all_gpu_for_eval(batched_cameras, batched_strategies, gpuid2tasks)

                    batched_voxel_mask =[None for _ in batched_cameras]
                    batched_nearest_voxel_mask=[None for _ in batched_cameras]
                    batched_nearest_cameras= [None for _ in batched_cameras]

                    batched_screenspace_pkg = distributed_preprocess3dgs_and_all2all_final(
                        batched_cameras,
                        scene.gaussians,
                        pipe_args,
                        batched_voxel_mask = batched_voxel_mask,
                        bg_color = background,
                        batched_strategies=batched_strategies,
                        batched_nearest_cameras = batched_nearest_cameras,
                        batched_nearest_voxel_mask = batched_nearest_voxel_mask,
                        mode="test",
                    )
                    
                    batched_image, batched_compute_locally, batched_out_all_map, batched_out_observe, batched_out_plane_depth, batched_return_dict, _ = render_final(
                        batched_cameras, batched_screenspace_pkg, batched_strategies, scene.gaussians, batched_nearest_cameras
                    )
                    
                    for camera_id, (image, gt_camera, render_pkg) in enumerate(zip(batched_image, batched_cameras, batched_return_dict)):
                        # Ensure reduction inputs are always valid CUDA tensors.
                        h, w = gt_camera.original_image.shape[1], gt_camera.original_image.shape[2]

                        def _ensure_cuda_tensor(x, shape):
                            if not isinstance(x, torch.Tensor) or x.numel() == 0:
                                return torch.zeros(shape, device="cuda", dtype=torch.float32)
                            return x

                        depth = _ensure_cuda_tensor(render_pkg.get("plane_depth"), (1, h, w))
                        normal = _ensure_cuda_tensor(render_pkg.get("rendered_normal"), (3, h, w))
                        rendered_distance = _ensure_cuda_tensor(render_pkg.get("rendered_distance"), (1, h, w))
                        image = _ensure_cuda_tensor(image, tuple(gt_camera.original_image.shape))

                        if utils.DEFAULT_GROUP.size() > 1:
                            torch.distributed.all_reduce(depth, op=dist.ReduceOp.SUM, group=utils.DEFAULT_GROUP)
                            torch.distributed.all_reduce(rendered_distance, op=dist.ReduceOp.SUM, group=utils.DEFAULT_GROUP)
                            torch.distributed.all_reduce(image, op=dist.ReduceOp.SUM, group=utils.DEFAULT_GROUP)
                            torch.distributed.all_reduce(normal, op=dist.ReduceOp.SUM, group=utils.DEFAULT_GROUP)

                        image = torch.clamp(image, 0.0, 1.0)
                        gt_image = torch.clamp(gt_camera.original_image / 255.0, 0.0, 1.0)

                        if idx + camera_id < num_cameras + 1:
                            l1_test += l1_loss(image, gt_image).mean().double()
                            psnr_test += psnr(image, gt_image).mean().double()

                        if utils.GLOBAL_RANK == 0:
                            depth = depth.detach().cpu().numpy().squeeze(0)
                            depth_i = (depth - depth.min()) / (depth.max() - depth.min() + 1e-20)
                            depth_i = (depth_i * 255).clip(0, 255).astype(np.uint8)
                            depth_color = cv2.applyColorMap(depth_i, cv2.COLORMAP_JET)

                            distance = rendered_distance.squeeze().detach().cpu().numpy()
                            distance_i = (distance - distance.min()) / (distance.max() - distance.min() + 1e-20)
                            distance_i = (distance_i * 255).clip(0, 255).astype(np.uint8)
                            distance_color = cv2.applyColorMap(distance_i, cv2.COLORMAP_JET)

                            normal = normal.permute(1,2,0)
                            normal = normal/(normal.norm(dim=-1, keepdim=True)+1.0e-8)
                            normal = normal.detach().cpu().numpy()
                            normal = ((normal+1) * 127.5).astype(np.uint8).clip(0, 255)
                            
                            torchvision.utils.save_image(image, os.path.join(render_path, gt_camera.image_name + ".png"))
                            torchvision.utils.save_image(gt_image, os.path.join(gts_path, gt_camera.image_name + ".png"))
                            cv2.imwrite(os.path.join(rendered_distance_path,  gt_camera.image_name + ".png"), distance_color)
                            cv2.imwrite(os.path.join(depths_path,  gt_camera.image_name + ".png"), depth_color)
                            cv2.imwrite(os.path.join(render_normal_path,  gt_camera.image_name + ".png"), normal)

                        gt_camera.original_image = None
                        
                psnr_test /= num_cameras
                l1_test /= num_cameras
                utils.print_rank_0("\n[ITER {}] Evaluating {}: L1 {} PSNR {}, images {}, ".format(iteration, config["name"], l1_test, psnr_test, num_cameras))
                log_file.write("[ITER {}] Evaluating {}: L1 {} PSNR {}\n".format(iteration, config["name"], l1_test, psnr_test))

        torch.cuda.empty_cache()