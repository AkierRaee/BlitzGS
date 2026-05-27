import time
import torch
import torch.distributed as dist

import utils.general_utils as utils
from utils.loss_utils import pixelwise_l1_with_mask, pixelwise_ssim_with_mask


def get_coverage_y_min_max(tile_ids_l, tile_ids_r):
    return tile_ids_l * utils.BLOCK_Y, min(tile_ids_r * utils.BLOCK_Y, utils.IMG_H)


def get_coverage_y_min(tile_ids_l):
    return tile_ids_l * utils.BLOCK_Y


def get_coverage_y_max(tile_ids_r):
    return min(tile_ids_r * utils.BLOCK_Y, utils.IMG_H)


def load_camera_from_cpu_to_all_gpu_for_eval(
    batched_cameras, batched_strategies, gpuid2tasks
):
    timers = utils.get_timers()
    args = utils.get_args()

    if args.distributed_dataset_storage:
        if args.local_sampling:
            for idx, camera in enumerate(batched_cameras):
                if camera.original_image_backup is not None:
                    camera.original_image = camera.original_image_backup.cuda()
                    scatter_list = [
                        camera.original_image for _ in range(utils.IN_NODE_GROUP.size())
                    ]
                    torch.distributed.scatter(
                        camera.original_image,
                        scatter_list=scatter_list,
                        src=utils.GLOBAL_RANK,
                        group=utils.IN_NODE_GROUP,
                    )
                else:
                    camera.original_image = torch.zeros(
                        (3, utils.IMG_H, utils.IMG_W), dtype=torch.uint8, device="cuda"
                    )
                    bsz_per_gpu = args.bsz // utils.WORLD_SIZE
                    torch.distributed.scatter(
                        camera.original_image,
                        scatter_list=None,
                        src=idx // bsz_per_gpu,
                        group=utils.IN_NODE_GROUP,
                    )
                torch.distributed.barrier(group=utils.DEFAULT_GROUP)
            return

        if utils.IN_NODE_GROUP.rank() == 0:
            for camera in batched_cameras:
                camera.original_image = camera.original_image_backup.cuda()
                scatter_list = [
                    camera.original_image for _ in range(utils.IN_NODE_GROUP.size())
                ]
                torch.distributed.scatter(
                    camera.original_image,
                    scatter_list=scatter_list,
                    src=utils.get_first_rank_on_cur_node(),
                    group=utils.IN_NODE_GROUP,
                )
        else:
            for camera in batched_cameras:
                camera.original_image = torch.zeros(
                    (3, utils.IMG_H, utils.IMG_W), dtype=torch.uint8, device="cuda"
                )
                torch.distributed.scatter(
                    camera.original_image,
                    scatter_list=None,
                    src=utils.get_first_rank_on_cur_node(),
                    group=utils.IN_NODE_GROUP,
                )
    else:
        for camera in batched_cameras:
            camera.original_image = camera.original_image_backup.cuda()


def load_camera_from_cpu_to_all_gpu(batched_cameras, batched_strategies, gpuid2tasks):
    timers = utils.get_timers()
    args = utils.get_args()

    timers.start("load_gt_image_to_gpu")

    def load_camera_from_cpu_to_gpu(first_task, last_task):
        coverage_min_max_y = {}
        coverage_min_y_first_task = get_coverage_y_min(first_task[1])
        coverage_max_y_last_task = get_coverage_y_max(last_task[2])
        for camera_id_in_batch in range(first_task[0], last_task[0] + 1):
            coverage_min_y = 0
            if camera_id_in_batch == first_task[0]:
                coverage_min_y = coverage_min_y_first_task
            coverage_max_y = utils.IMG_H
            if camera_id_in_batch == last_task[0]:
                coverage_max_y = coverage_max_y_last_task

            batched_cameras[camera_id_in_batch].original_image = (
                batched_cameras[camera_id_in_batch]
                .original_image_backup[:, coverage_min_y:coverage_max_y, :]
                .cuda()
            )
            coverage_min_max_y[camera_id_in_batch] = (coverage_min_y, coverage_max_y)
        return coverage_min_max_y

    if args.distributed_dataset_storage:
        if args.local_sampling:
            first_task = gpuid2tasks[utils.GLOBAL_RANK][0]
            last_task = gpuid2tasks[utils.GLOBAL_RANK][-1]
            _ = load_camera_from_cpu_to_gpu(first_task, last_task)
        elif utils.IN_NODE_GROUP.rank() == 0:
            in_node_first_rank = utils.GLOBAL_RANK
            in_node_last_rank = in_node_first_rank + utils.IN_NODE_GROUP.size() - 1
            first_task = gpuid2tasks[in_node_first_rank][0]
            last_task = gpuid2tasks[in_node_last_rank][-1]
            coverage_min_max_y_gpu0 = load_camera_from_cpu_to_gpu(first_task, last_task)
    else:
        first_task = gpuid2tasks[utils.GLOBAL_RANK][0]
        last_task = gpuid2tasks[utils.GLOBAL_RANK][-1]
        _ = load_camera_from_cpu_to_gpu(first_task, last_task)

    timers.stop("load_gt_image_to_gpu")

    if args.local_sampling:
        return

    timers.start("scatter_gt_image")
    if args.distributed_dataset_storage:
        comm_ops = []
        if utils.IN_NODE_GROUP.rank() == 0:
            in_node_first_rank = utils.get_first_rank_on_cur_node()
            in_node_last_rank = in_node_first_rank + utils.IN_NODE_GROUP.size() - 1
            for rank in range(in_node_first_rank, in_node_last_rank + 1):
                if rank == utils.GLOBAL_RANK:
                    continue
                for task in gpuid2tasks[rank]:
                    camera_id = task[0]
                    coverage_min_y = get_coverage_y_min(task[1])
                    coverage_max_y = get_coverage_y_max(task[2])

                    coverage_min_y_gpu0, coverage_max_y_gpu0 = coverage_min_max_y_gpu0[
                        camera_id
                    ]
                    if (
                        coverage_min_y == coverage_min_y_gpu0
                        and coverage_max_y == coverage_max_y_gpu0
                    ):
                        op = torch.distributed.P2POp(
                            dist.isend,
                            batched_cameras[camera_id].original_image.contiguous(),
                            rank,
                        )
                    else:
                        send_tensor = (
                            batched_cameras[camera_id]
                            .original_image[
                                :,
                                coverage_min_y
                                - coverage_min_y_gpu0 : coverage_max_y
                                - coverage_min_y_gpu0,
                                :,
                            ]
                            .contiguous()
                        )
                        op = torch.distributed.P2POp(dist.isend, send_tensor, rank)
                    comm_ops.append(op)

            reqs = torch.distributed.batch_isend_irecv(comm_ops)
            for req in reqs:
                req.wait()

            for task in gpuid2tasks[utils.GLOBAL_RANK]:
                camera_id = task[0]
                coverage_min_y_gpu0, coverage_max_y_gpu0 = coverage_min_max_y_gpu0[
                    camera_id
                ]
                coverage_min_y = get_coverage_y_min(task[1])
                coverage_max_y = get_coverage_y_max(task[2])
                batched_cameras[camera_id].original_image = (
                    batched_cameras[camera_id]
                    .original_image[
                        :,
                        coverage_min_y
                        - coverage_min_y_gpu0 : coverage_max_y
                        - coverage_min_y_gpu0,
                        :,
                    ]
                    .contiguous()
                )
        else:
            in_node_first_rank = utils.get_first_rank_on_cur_node()
            recv_buffer_list = []
            for task in gpuid2tasks[utils.GLOBAL_RANK]:
                coverage_min_y = get_coverage_y_min(task[1])
                coverage_max_y = get_coverage_y_max(task[2])
                recv_buffer = torch.zeros(
                    (3, coverage_max_y - coverage_min_y, utils.IMG_W),
                    dtype=torch.uint8,
                    device="cuda",
                )
                recv_buffer_list.append(recv_buffer)
                op = torch.distributed.P2POp(
                    dist.irecv, recv_buffer, in_node_first_rank
                )
                comm_ops.append(op)

            reqs = torch.distributed.batch_isend_irecv(comm_ops)
            for req in reqs:
                req.wait()

            for idx, task in enumerate(gpuid2tasks[utils.GLOBAL_RANK]):
                batched_cameras[task[0]].original_image = recv_buffer_list[idx]

    timers.stop("scatter_gt_image")


def final_system_loss_computation(
    image, viewpoint_cam, strategy, statistic_collector
):
    timers = utils.get_timers()
    args = utils.get_args()

    timers.start("prepare_image_rect_and_mask")
    assert (
        utils.GLOBAL_RANK in strategy.gpu_ids
    ), "The current gpu must be used to render this camera."
    rank = strategy.gpu_ids.index(utils.GLOBAL_RANK)
    tile_ids_l, tile_ids_r = (
        strategy.division_pos[rank],
        strategy.division_pos[rank + 1],
    )
    coverage_min_y, coverage_max_y = get_coverage_y_min_max(tile_ids_l, tile_ids_r)

    local_image_rect = image[:, coverage_min_y:coverage_max_y, :].contiguous()
    local_image_rect_pixels_compute_locally = torch.ones(
        (coverage_max_y - coverage_min_y, utils.IMG_W), dtype=torch.bool, device="cuda"
    )
    timers.stop("prepare_image_rect_and_mask")

    timers.start("prepare_gt_image")
    local_image_rect_gt = torch.clamp(viewpoint_cam.original_image / 255.0, 0.0, 1.0)
    timers.stop("prepare_gt_image")

    timers.start("local_loss_computation")
    torch.cuda.synchronize()
    start_time = time.time()
    pixelwise_Ll1 = pixelwise_l1_with_mask(
        local_image_rect, local_image_rect_gt, local_image_rect_pixels_compute_locally
    )
    Ll1 = pixelwise_Ll1.sum() / (utils.get_num_pixels() * 3)
    pixelwise_ssim_loss = pixelwise_ssim_with_mask(
        local_image_rect, local_image_rect_gt, local_image_rect_pixels_compute_locally
    )
    ssim_loss = pixelwise_ssim_loss.sum() / (utils.get_num_pixels() * 3)

    torch.cuda.synchronize()
    statistic_collector["forward_loss_time"] = (time.time() - start_time) * 1000
    timers.stop("local_loss_computation")

    return Ll1, ssim_loss


def batched_loss_computation(
    batched_image,
    batched_return_dict,
    batched_cameras,
    batched_strategies,
    batched_statistic_collector,
    iterations,
    opt,
):
    args = utils.get_args()
    timers = utils.get_timers()

    timers.start("loss_computation")
    batched_losses = []
    loss_sum = 0
    for idx, (
        image,
        render_pkg,
        camera,
        strategy,
        statistic_collector,
    ) in enumerate(
        zip(
            batched_image,
            batched_return_dict,
            batched_cameras,
            batched_strategies,
            batched_statistic_collector,
        )
    ):
        if image is None:
            loss = 0
            batched_losses.append([0.0, 0.0])
        elif len(image.shape) == 0:
            loss = image * 0
            batched_losses.append([loss, 0.0])
        else:
            Ll1, ssim_loss = final_system_loss_computation(
                image, camera, strategy, statistic_collector
            )

            loss = (1.0 - args.lambda_dssim) * Ll1 + args.lambda_dssim * (
                1.0 - ssim_loss
            )
            if torch.isnan(loss):
                print(f'NAN with rgb:{camera.image_name}')
                continue
            batched_losses.append([Ll1, ssim_loss])
            if iterations > opt.scale_loss_from_iter:
                visibility_filter = render_pkg["visibility_filter"]
                if visibility_filter is not None:
                    if visibility_filter.sum() > 0:
                        weight = 10.0
                        scale = (render_pkg["scales_redistributed"])[visibility_filter]
                        sorted_scale, _ = torch.sort(scale, dim=-1)
                        min_scale_loss = sorted_scale[..., 0]
                        loss += weight * min_scale_loss.mean()

            timers.stop("loss_computation")

        loss_sum += loss

    assert loss_sum.dim() == 0, "The loss_sum must be a scalar tensor."
    return loss_sum * args.lr_scale_loss, batched_losses
