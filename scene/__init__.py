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
import random
import json
from random import randint
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks, storePly
from scene.gaussian_model import GaussianModel
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON, _get_resolution
import utils.general_utils as utils
import torch
import numpy as np
from utils.graphics_utils import BasicPointCloud


class _NullLogFile:
    def write(self, _msg):
        return


_NULL_LOG_FILE = _NullLogFile()


class Scene:

    gaussians: GaussianModel

    def __init__(
        self, args, gaussians: GaussianModel, load_iteration=None, shuffle=True
    ):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians
        log_file = utils.get_log_file() or _NULL_LOG_FILE
        

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(
                    os.path.join(self.model_path, "point_cloud")
                )
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        utils.log_cpu_memory_usage("before loading images meta data")
        llffhold = getattr(args, "llffhold", 83)
        src_lower = args.source_path.lower()
        _is_mc = "matrix_city" in src_lower or "matrixcity" in src_lower
        if _is_mc and "train" in args.source_path:
            scene_info = sceneLoadTypeCallbacks["City"](
                args.source_path, args.images, args.eval, llffhold
            )
        elif os.path.exists(
            os.path.join(args.source_path, "sparse")
        ):
            scene_info = sceneLoadTypeCallbacks["Colmap"](
                args.source_path, args.images, args.eval, llffhold
            )
        else:
            raise ValueError("No valid dataset found in the source path")


        if not self.loaded_iter:
            points = self.save_ply(scene_info.point_cloud, args.ratio, os.path.join(self.model_path, "input.ply"))
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), "w") as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(
                scene_info.train_cameras
            )
            random.shuffle(
                scene_info.test_cameras
            )

        utils.log_cpu_memory_usage("before decoding images")

        self.cameras_extent = scene_info.nerf_normalization["radius"]
        print(f'camera_extent: {self.cameras_extent}')
        orig_w, orig_h = (
            scene_info.train_cameras[0].width,
            scene_info.train_cameras[0].height,
        )
        res_w, res_h = _get_resolution(orig_w, orig_h, args.resolution)
        utils.set_img_size(res_h, res_w)
        dataset_size_in_GB = (
            1.0
            * (len(scene_info.train_cameras) + len(scene_info.test_cameras))
            * res_w
            * res_h
            * (3)
            / 1e9
        )
        log_file.write(f"Dataset size: {dataset_size_in_GB} GB\n")
        preload_threshold = getattr(args, "preload_dataset_to_gpu_threshold", 3)
        if (
            dataset_size_in_GB < preload_threshold
        ):
            log_file.write(
                f"[NOTE]: Preloading dataset({dataset_size_in_GB}GB) to GPU. Disable local_sampling and distributed_dataset_storage.\n"
            )
            print(
                f"[NOTE]: Preloading dataset({dataset_size_in_GB}GB) to GPU. Disable local_sampling and distributed_dataset_storage."
            )
            args.preload_dataset_to_gpu = True
            args.local_sampling = False
            args.distributed_dataset_storage = False

        utils.print_rank_0("Decoding Training Cameras")

        self.train_cameras = None
        self.test_cameras = None
        self.multi_view_num = args.multi_view_num
        num_train_cameras = getattr(args, "num_train_cameras", -1)
        if not args.eval:
            if num_train_cameras >= 0:
                train_cameras = scene_info.train_cameras[: num_train_cameras]
            else:
                train_cameras = scene_info.train_cameras
            self.train_cameras = cameraList_from_camInfos(train_cameras, args)
            log_file.write(
                "Number of local training cameras: {}\n".format(len(self.train_cameras))
            )
            if len(self.train_cameras) > 0:
                log_file.write(
                    "Image size: {}x{}\n".format(
                        self.train_cameras[0].image_height,
                        self.train_cameras[0].image_width,
                    )
                )
            

        utils.print_rank_0("Decoding Test Cameras")
        num_test_cameras = getattr(args, "num_test_cameras", -1)
        if num_test_cameras >= 0:
            test_cameras = scene_info.test_cameras[: num_test_cameras]
        else:
            test_cameras = scene_info.test_cameras
        self.test_cameras = cameraList_from_camInfos(test_cameras, args)
        log_file.write(
            "Number of local test cameras: {}\n".format(len(self.test_cameras))
        )
        if len(self.test_cameras) > 0:
            log_file.write(
                "Image size: {}x{}\n".format(
                    self.test_cameras[0].image_height,
                    self.test_cameras[0].image_width,
                )
            )
        print("computing nearest_id")
        self.world_view_transforms = []
        camera_centers = []
        center_rays = []
        if not args.eval:
            for id, cur_cam in enumerate(self.train_cameras):
                self.world_view_transforms.append(cur_cam.world_view_transform)
                camera_centers.append(cur_cam.camera_center)
                R = torch.tensor(cur_cam.R).float().cuda()
                T = torch.tensor(cur_cam.T).float().cuda()
                center_ray = torch.tensor([0.0,0.0,1.0]).float().cuda()
                center_ray = center_ray@R.transpose(-1,-2)
                center_rays.append(center_ray)
            self.world_view_transforms = torch.stack(self.world_view_transforms)
            camera_centers = torch.stack(camera_centers, dim=0)
            center_rays = torch.stack(center_rays, dim=0)
            center_rays = torch.nn.functional.normalize(center_rays, dim=-1)
            diss = torch.norm(camera_centers[:,None] - camera_centers[None], dim=-1).detach().cpu().numpy()
            tmp = torch.sum(center_rays[:,None]*center_rays[None], dim=-1)
            angles = torch.arccos(tmp)*180/3.14159
            angles = angles.detach().cpu().numpy()
            with open(os.path.join(self.model_path, "multi_view.json"), 'w') as file:
                for id, cur_cam in enumerate(self.train_cameras):
                    sorted_indices = np.lexsort((angles[id], diss[id]))
                    mask = (angles[id][sorted_indices] < args.multi_view_max_angle) & \
                        (diss[id][sorted_indices] > args.multi_view_min_dis) & \
                        (diss[id][sorted_indices] < args.multi_view_max_dis)
                    sorted_indices = sorted_indices[mask]
                    multi_view_num = min(self.multi_view_num, len(sorted_indices))
                    json_d = {'ref_name' : cur_cam.image_name, 'nearest_name': []}
                    for index in sorted_indices[:multi_view_num]:
                        cur_cam.nearest_id.append(index)
                        cur_cam.nearest_names.append(self.train_cameras[index].image_name)
                        json_d["nearest_name"].append(self.train_cameras[index].image_name)
                    json_str = json.dumps(json_d, separators=(',', ':'))
                    file.write(json_str)
                    file.write('\n')


        utils.check_initial_gpu_memory_usage("after Loading all images")
        utils.log_cpu_memory_usage("after decoding images")

        if self.loaded_iter:
            self.gaussians.load_ply(
                os.path.join(
                    self.model_path, "point_cloud", "iteration_" + str(self.loaded_iter)
                )
            )
        elif hasattr(args, "load_ply_path"):
            self.gaussians.load_ply(args.load_ply_path)
        else:
            self.gaussians.create_from_pcd(
                points, self.cameras_extent, cameras=self.train_cameras
            )

        utils.check_initial_gpu_memory_usage("after initializing point cloud")
        utils.log_cpu_memory_usage("after loading initial 3dgs points")
    def save(self, iteration):
        point_cloud_path = os.path.join(
            self.model_path, "point_cloud/iteration_{}".format(iteration)
        )
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def save_ply(self, pcd, ratio, path):
        points = torch.tensor(pcd.points[::ratio]).float().cuda()
        colors = torch.tensor(pcd.colors[::ratio]).float().cuda()
        storePly(path, points.cpu().numpy(), colors.cpu().numpy())
        return BasicPointCloud(
            points=points.cpu().numpy(),
            colors=colors.cpu().numpy(),
            normals=np.zeros_like(points.cpu().numpy()),
        )

    def getTrainCameras(self):
        return self.train_cameras

    def getTestCameras(self):
        return self.test_cameras

    def log_scene_info_to_file(self, log_file, prefix_str=""):
        log_file.write("scaling shape: {}\n".format(self.gaussians._scaling.shape))
        log_file.write("opacity shape: {}\n".format(self.gaussians._opacity.shape))
        log_file.write("rotation shape: {}\n".format(self.gaussians._rotation.shape))


class SceneDataset:
    def __init__(self, cameras):
        self.cameras = cameras
        self.camera_size = len(self.cameras)
        self.sample_camera_idx = []
        for i in range(self.camera_size):
            if self.cameras[i].original_image_backup is not None:
                self.sample_camera_idx.append(i)

        self.cur_epoch_cameras = []
        self.cur_iteration = 0

        self.iteration_loss = []
        self.epoch_loss = []

        self.log_file = utils.get_log_file()
        self.args = utils.get_args()

        self.last_time_point = None
        self.epoch_time = []
        self.epoch_n_sample = []

        self.per_view_residual = torch.zeros(self.camera_size)
        self.per_view_seen = torch.zeros(self.camera_size, dtype=torch.long)
        self._uid_to_pos = {c.uid: i for i, c in enumerate(self.cameras)}

    @property
    def cur_epoch(self):
        return len(self.epoch_loss)

    @property
    def cur_iteration_in_epoch(self):
        return len(self.iteration_loss)

    def get_one_camera(self, batched_cameras_uid, shuffle):
        args = utils.get_args()
        if len(self.cur_epoch_cameras) == 0:
            if args.local_sampling:
                self.cur_epoch_cameras = self.sample_camera_idx.copy()
            else:
                self.cur_epoch_cameras = list(range(self.camera_size))
            if shuffle:
                indices = self._epoch_permutation(len(self.cur_epoch_cameras))
            else :
                indices = torch.arange(len(self.cur_epoch_cameras))
            self.cur_epoch_cameras = [self.cur_epoch_cameras[i] for i in indices]

        self.cur_iteration += 1

        idx = 0
        while self.cameras[self.cur_epoch_cameras[idx]].uid in batched_cameras_uid:
            idx += 1
        camera_idx = self.cur_epoch_cameras.pop(idx)
        viewpoint_cam = self.cameras[camera_idx]
        return camera_idx, viewpoint_cam

    def _epoch_permutation(self, n):
        args = self.args
        use_a2 = getattr(args, "use_error_sampling", False)
        if not use_a2 or self.cur_iteration < args.densify_from_iter:
            return torch.randperm(n)

        pos_tensor = torch.tensor(self.cur_epoch_cameras, dtype=torch.long)
        r = self.per_view_residual[pos_tensor].clone()
        seen = self.per_view_seen[pos_tensor]
        if (seen == 0).any():
            r_max = r.max() if (seen > 0).any() else torch.tensor(1.0)
            r[seen == 0] = r_max
        alpha = self._compute_alpha()
        if alpha <= 0.0:
            return torch.randperm(n)

        std = r.std().clamp(min=1e-6)
        logits = alpha * (r - r.mean()) / std
        w = torch.softmax(logits, dim=0)
        cap = args.sampling_weight_cap / max(1, n)
        w = w.clamp(max=cap)
        w = w / w.sum()
        return torch.multinomial(w, n, replacement=False)

    def _compute_alpha(self):
        args = self.args
        it = self.cur_iteration
        if it < args.densify_from_iter or it > args.densify_until_iter:
            return 0.0
        warmup_end = args.densify_from_iter + args.sampling_alpha_warmup_iters
        if it < warmup_end:
            return args.sampling_alpha_final * (it - args.densify_from_iter) / max(1, args.sampling_alpha_warmup_iters)
        return args.sampling_alpha_final

    def get_batched_cameras(self, batch_size, shuffle = True , eval = False):
        assert (
            batch_size <= self.camera_size
        ), "Batch size is larger than the number of cameras in the scene."
        batched_cameras = []
        batched_cameras_uid = []
        batched_nearest_cameras = []
        batched_nearest_cameras_uid = []
        for i in range(batch_size):
            _, camera = self.get_one_camera(batched_cameras_uid, shuffle = shuffle)
            batched_cameras.append(camera)
            batched_cameras_uid.append(camera.uid)
            if eval == False:
                if len(camera.nearest_id) > 4:
                    for idx in random.sample(camera.nearest_id,1):
                        while self.cameras[idx].uid in batched_cameras_uid: 
                            idx = random.sample(camera.nearest_id,1)[0]
                        batched_nearest_cameras.append(self.cameras[idx])
                        batched_nearest_cameras_uid.append(self.cameras[idx].uid)
                else:
                    batched_nearest_cameras.append(camera)
                    batched_nearest_cameras_uid.append(camera.uid)
            else:
                continue
        return batched_cameras, batched_nearest_cameras


    def get_batched_cameras_idx(self, batch_size):
        assert (
            batch_size <= self.camera_size
        ), "Batch size is larger than the number of cameras in the scene."
        batched_cameras_idx = []
        batched_cameras_uid = []
        for i in range(batch_size):
            idx, camera = self.get_one_camera(batched_cameras_uid)
            batched_cameras_uid.append(camera.uid)
            batched_cameras_idx.append(idx)

        return batched_cameras_idx

    def get_batched_cameras_from_idx(self, idx_list):
        return [self.cameras[i] for i in idx_list]

    def update_losses(self, losses, cameras=None):
        if cameras is not None and getattr(self.args, "use_error_sampling", False):
            ema = float(self.args.residual_ema)
            for loss, cam in zip(losses, cameras):
                pos = self._uid_to_pos.get(cam.uid)
                if pos is None:
                    continue
                r = float(loss)
                if self.per_view_seen[pos] == 0:
                    self.per_view_residual[pos] = r
                else:
                    self.per_view_residual[pos] = ema * float(self.per_view_residual[pos]) + (1.0 - ema) * r
                self.per_view_seen[pos] += 1

        for loss in losses:
            self.iteration_loss.append(loss)
            if len(self.iteration_loss) % self.camera_size == 0:
                self.epoch_loss.append(
                    sum(self.iteration_loss[-self.camera_size :]) / self.camera_size
                )
                self.log_file.write(
                    "epoch {} loss: {}\n".format(
                        len(self.epoch_loss), self.epoch_loss[-1]
                    )
                )
                self.iteration_loss = []
