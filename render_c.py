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

import os
from argparse import ArgumentParser
from os import makedirs

import torch
import torch.distributed as dist
import torchvision
from tqdm import tqdm

from arguments import (
    AuxiliaryParams,
    BenchmarkParams,
    DebugParams,
    DistributionParams,
    ModelParams,
    OptimizationParams,
    PipelineParams,
    get_combined_args,
)
from gaussian_renderer import GaussianModel, render_imp
from scene import Scene
from utils.system_utils import searchForMaxIteration
from utils.general_utils import init_distributed, safe_state, set_args
import utils.general_utils as utils


def _build_candidate_indices(num_cameras, args):
    indices = []
    for idx in range(num_cameras):
        actual_idx = idx + 1
        if args.sample_freq != -1 and actual_idx % args.sample_freq != 0:
            continue
        if args.l != -1 and args.r != -1:
            if actual_idx < args.l or actual_idx >= args.r:
                continue
        indices.append(idx)

    if args.generate_num != -1:
        indices = indices[: args.generate_num]

    return indices


def _get_gt_image_tensor(view, device):
    gt = None
    if hasattr(view, "original_image") and view.original_image is not None:
        gt = view.original_image
    elif hasattr(view, "original_image_backup") and view.original_image_backup is not None:
        gt = view.original_image_backup
    else:
        raise RuntimeError(
            f"No image tensor found for camera {getattr(view, 'image_name', 'unknown')}"
        )

    gt = gt[0:3, :, :]
    if gt.dtype == torch.uint8:
        gt = gt.float() / 255.0
    elif gt.dtype.is_floating_point and torch.max(gt) > 1.0:
        gt = gt / 255.0

    return gt.to(device)


def render_set(model_path, name, iteration, views, gaussians, pipeline, background, args):
    render_path = os.path.join(model_path, name, f"ours_{iteration}", "renders")
    gts_path = os.path.join(model_path, name, f"ours_{iteration}", "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    world_size = utils.DEFAULT_GROUP.size()
    rank = utils.DEFAULT_GROUP.rank()

    candidate_indices = _build_candidate_indices(len(views), args)
    local_indices = candidate_indices[rank::world_size]

    progress = None
    if rank == 0:
        progress = tqdm(total=len(candidate_indices), desc=f"Rendering progress ({name})")

    total_steps = (len(local_indices) + args.bsz - 1) // args.bsz
    if world_size > 1:
        total_steps_tensor = torch.tensor([total_steps], device="cuda", dtype=torch.int64)
        dist.all_reduce(total_steps_tensor, op=dist.ReduceOp.MAX, group=utils.DEFAULT_GROUP)
        total_steps = int(total_steps_tensor.item())

    for step in range(total_steps):
        start = step * args.bsz
        batch_indices = local_indices[start : start + args.bsz]

        for idx in batch_indices:
            view = views[idx]
            rendering = render_imp(view, gaussians, pipeline, background)["render"]
            gt = _get_gt_image_tensor(view, rendering.device)
            torchvision.utils.save_image(
                rendering, os.path.join(render_path, f"{view.image_name}.png")
            )
            torchvision.utils.save_image(
                gt, os.path.join(gts_path, f"{view.image_name}.png")
            )

        if world_size > 1:
            local_done = torch.tensor([len(batch_indices)], device="cuda", dtype=torch.int64)
            dist.all_reduce(local_done, op=dist.ReduceOp.SUM, group=utils.DEFAULT_GROUP)
            if progress is not None:
                progress.update(int(local_done.item()))
        elif progress is not None:
            progress.update(len(batch_indices))

    if progress is not None:
        progress.close()


def render_sets(
    dataset: ModelParams,
    iteration: int,
    pipeline: PipelineParams,
    skip_train: bool,
    skip_test: bool,
    args,
):
    with torch.no_grad():
        if not hasattr(pipeline, "convert_SHs_python"):
            pipeline.convert_SHs_python = False
        if not hasattr(pipeline, "compute_cov3D_python"):
            pipeline.compute_cov3D_python = False
        if not hasattr(pipeline, "debug"):
            pipeline.debug = False

        gaussians = GaussianModel(dataset.sh_degree)
        if iteration == -1:
            resolved_iteration = searchForMaxIteration(
                os.path.join(dataset.model_path, "point_cloud")
            )
        else:
            resolved_iteration = iteration

        dataset.load_ply_path = os.path.join(
            dataset.model_path,
            "point_cloud",
            f"iteration_{resolved_iteration}",
            "point_cloud.ply",
        )

        scene = Scene(dataset, gaussians, load_iteration=None, shuffle=False)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
            render_set(
                dataset.model_path,
                "train",
                resolved_iteration,
                scene.getTrainCameras(),
                gaussians,
                pipeline,
                background,
                args,
            )

        if not skip_test:
            render_set(
                dataset.model_path,
                "test",
                resolved_iteration,
                scene.getTestCameras(),
                gaussians,
                pipeline,
                background,
                args,
            )


if __name__ == "__main__":
    parser = ArgumentParser(description="Multi-GPU render")
    ap = AuxiliaryParams(parser)
    model = ModelParams(parser, sentinel=True)
    op = OptimizationParams(parser)
    pipeline = PipelineParams(parser)
    dist_p = DistributionParams(parser)
    bench_p = BenchmarkParams(parser)
    debug_p = DebugParams(parser)

    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--generate_num", default=-1, type=int)
    parser.add_argument("--sample_freq", default=-1, type=int)
    parser.add_argument("--l", default=-1, type=int)
    parser.add_argument("--r", default=-1, type=int)
    args = get_combined_args(parser)
    args.distributed_dataset_storage = False
    args.local_sampling = False
    init_distributed(args)
    set_args(args)

    if torch.cuda.is_available():
        torch.cuda.set_device(utils.LOCAL_RANK)

    if utils.DEFAULT_GROUP.size() > 1:
        torch.distributed.barrier(group=utils.DEFAULT_GROUP)

    if utils.GLOBAL_RANK == 0:
        print("Rendering " + args.model_path)

    safe_state(args.quiet)

    render_sets(
        model.extract(args),
        args.iteration,
        pipeline.extract(args),
        args.skip_train,
        args.skip_test,
        args,
    )
