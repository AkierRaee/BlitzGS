#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#

import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from utils.sh_utils import SH2RGB

import utils.general_utils as utils
import torch.distributed as dist

try:
    from diff_gaussian_rasterization_f import SparseGaussianAdam
except:
    pass

def init_cdf_mask(importance, thres=1.0):
    importance = importance.flatten()   
    if thres!=1.0:
        percent_sum = thres
        vals,idx = torch.sort(importance+(1e-6))
        cumsum_val = torch.cumsum(vals, dim=0)
        split_index = ((cumsum_val/vals.sum()) > (1-percent_sum)).nonzero().min()
        split_val_nonprune = vals[split_index]

        non_prune_mask = importance>split_val_nonprune 
    else: 
        non_prune_mask = torch.ones_like(importance).bool()
        
    return non_prune_mask


class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.covariance_activation = build_covariance_from_scaling_rotation
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self, sh_degree : int):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)

        self._culling = None
        self._culling_num_views = 0
        self.factor_culling = None

        self._lod_level = torch.empty(0, dtype=torch.int8)
        self.octree = None

        self._ids = None
        self._birth_iter = None
        self._id_prefix = 0
        self._next_local_id = 0
        self.spawn_gate = None
        self._lod_last_kept = None
        self._lod_last_total = None
        self._lod_last_active_level = None

        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

    def capture(self):
        octree_state = self.octree.state_dict() if self.octree is not None else None
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
            self._lod_level,
            octree_state,
        )

    def restore(self, model_args, training_args):
        (self.active_sh_degree,
        self._xyz,
        self._features_dc,
        self._features_rest,
        self._scaling,
        self._rotation,
        self._opacity,
        self.max_radii2D,
        xyz_gradient_accum,
        denom,
        opt_dict,
        self.spatial_lr_scale,
        *optional) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)
        if len(optional) >= 2 and optional[1] is not None:
            from scene.octree_lod import OctreeLOD
            self._lod_level = optional[0]
            self.octree = OctreeLOD.from_state_dict(optional[1])
        elif len(optional) >= 1 and optional[0] is not None:
            self._lod_level = optional[0]

    def save_id_state(self, folder):
        """E40: persist the id-tracking state next to a training checkpoint
        (capture() predates ids and its tuple format is left untouched)."""
        if self._ids is None:
            return
        torch.save({
            "ids": self._ids.cpu(),
            "birth_iter": self._birth_iter.cpu(),
            "next_local_id": self._next_local_id,
            "id_prefix": self._id_prefix,
        }, os.path.join(folder, "id_state_ws={}_rk={}.pt".format(
            utils.WORLD_SIZE, utils.GLOBAL_RANK)))

    def load_id_state(self, checkpoint_folder):
        """E40: restore id state saved by save_id_state for the checkpoint at
        <model>/checkpoints/<it>/ (sidecar lives in checkpoints_idstate/<it>/).
        Returns True when restored — the signal that the spawn gate may be
        enabled on a resumed run."""
        ckpt = checkpoint_folder.rstrip("/")
        folder = os.path.join(os.path.dirname(os.path.dirname(ckpt)),
                              "checkpoints_idstate", os.path.basename(ckpt))
        path = os.path.join(folder, "id_state_ws={}_rk={}.pt".format(
            utils.WORLD_SIZE, utils.GLOBAL_RANK))
        if not os.path.exists(path):
            return False
        blob = torch.load(path, map_location="cpu")
        self._ids = blob["ids"].cuda()
        self._birth_iter = blob["birth_iter"].cuda()
        self._next_local_id = int(blob["next_local_id"])
        self._id_prefix = int(blob["id_prefix"])
        assert self._ids.shape[0] == self._xyz.shape[0], \
            "id_state size {} != model size {}".format(
                self._ids.shape[0], self._xyz.shape[0])
        return True

    def _init_id_tracking(self):
        """Persistent globally-unique IDs + birth iterations for the current
        population. Substrate of the spawn gate's parent_age feature."""
        rank = utils.GLOBAL_RANK
        n = self._xyz.shape[0]
        self._id_prefix = rank << 40
        self._ids = torch.arange(n, dtype=torch.int64, device="cuda") + self._id_prefix
        self._next_local_id = n
        self._birth_iter = torch.zeros(n, dtype=torch.int32, device="cuda")

    def init_spawn_gate(self, gate):
        """Enable the spawn pruning gate. Needs ID tracking for parent_age."""
        if self._ids is None:
            self._init_id_tracking()
        self.spawn_gate = gate

    def _apply_spawn_gate(self, selected_pts_mask, grad_norm, is_split):
        """Veto spawn candidates the gate predicts will not survive.
        Returns selected_pts_mask with vetoed candidates cleared."""
        if self.spawn_gate is None:
            return selected_pts_mask
        n_sel = int(selected_pts_mask.sum())
        if n_sel == 0:
            return selected_pts_mask
        it = utils.get_cur_iter() or 0
        if self.octree is not None:
            lv = self._lod_level[selected_pts_mask]
            if is_split:
                lv = (lv.long() + 1).clamp(0, self.octree.levels - 1)
        else:
            lv = None
        feats = self._spawn_features(grad_norm, selected_pts_mask, lv)
        veto = self.spawn_gate.veto(feats, it, is_split)
        if veto is None:
            return selected_pts_mask
        out = selected_pts_mask.clone()
        sel_idx = selected_pts_mask.nonzero(as_tuple=True)[0]
        out[sel_idx[veto]] = False
        print(f"[SpawnGate iter={it} rk{utils.GLOBAL_RANK}] "
              f"{'split' if is_split else 'clone'}: cand={n_sel} "
              f"vetoed={int(veto.sum())}", flush=True)
        return out

    def _extend_ids(self, n_new):
        if self._ids is None or n_new <= 0:
            return
        it = utils.get_cur_iter() or 0
        new_ids = torch.arange(n_new, dtype=torch.int64, device="cuda") \
            + (self._id_prefix + self._next_local_id)
        self._next_local_id += n_new
        self._ids = torch.cat([self._ids, new_ids])
        self._birth_iter = torch.cat([
            self._birth_iter,
            torch.full((n_new,), it, dtype=torch.int32, device="cuda")])

    def _spawn_features(self, grad_norm, selected_pts_mask, parent_levels):
        """Spawn-time feature dict for the selected parents (read BEFORE
        densification_postfix reallocates denom)."""
        if self._ids is None:
            return None
        it = utils.get_cur_iter() or 0
        n = int(selected_pts_mask.sum())
        if parent_levels is not None:
            lod = parent_levels.float()
        else:
            lod = torch.full((n,), -1.0, device="cuda")
        return {
            "parent_grad": grad_norm[selected_pts_mask],
            "parent_opacity": self.get_opacity[selected_pts_mask].squeeze(-1),
            "parent_scale": self.get_scaling[selected_pts_mask].mean(dim=1),
            "parent_age": (it - self._birth_iter[selected_pts_mask]).float(),
            "parent_denom": self.denom[selected_pts_mask].squeeze(-1),
            "lod_level": lod,
        }

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_features_dc(self):
        return self._features_dc
    
    @property
    def get_features_rest(self):
        return self._features_rest    
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd, spatial_lr_scale: float, cameras=None):
        """
        Initialize Gaussians from the point cloud.

        When `--use_octree_lod` is set and `cameras` is provided, replaces the flat
        PCD init with octree sampling: each Gaussian is an anchor at a specific
        level's unique voxel cell, with level intrinsic to its sampling voxel.
        Otherwise the standard flat PCD init is used.
        """
        self.spatial_lr_scale = spatial_lr_scale
        args = utils.get_args()
        use_lod = getattr(args, 'use_octree_lod', False) and cameras is not None

        if isinstance(pcd, torch.Tensor):
            pcd_points = pcd.float().cuda()
            pcd_colors = torch.rand_like(pcd_points)
        else:
            pcd_points = torch.tensor(np.asarray(pcd.points)).float().cuda()
            pcd_colors = torch.tensor(np.asarray(pcd.colors)).float().cuda()

        if use_lod:
            fused_point_cloud, fused_color = self._build_octree_lod_points(
                pcd_points, pcd_colors, cameras, args
            )
        else:
            fused_point_cloud = pcd_points
            fused_color = RGB2SH(pcd_colors)

        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        dist2 = torch.clamp_min(distCUDA2(fused_point_cloud), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        if args.gaussians_distribution:
            shard_world_size = utils.DEFAULT_GROUP.size()
            shard_rank = utils.DEFAULT_GROUP.rank()
            the_idx = shard_rank
            random_idx = torch.arange(fused_point_cloud.shape[0], device="cuda")
            mask = random_idx % shard_world_size == the_idx
            index = random_idx[mask]

            fused_point_cloud = fused_point_cloud[index].contiguous()
            features = features[index].contiguous()
            scales = scales[index].contiguous()
            rots = rots[index].contiguous()
            opacities = opacities[index].contiguous()
            if use_lod:
                self._lod_level = self._lod_level[index].contiguous()

        print("Number of points at initialisation on this GPU : ", fused_point_cloud.shape[0])

        self._xyz = nn.Parameter(fused_point_cloud.contiguous().requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        if use_lod:
            coarse_iter = getattr(args, 'densify_until_iter', self.octree.levels * 5000)
            coarse_factor = getattr(args, 'octree_coarse_factor', 1.5)
            init_level = getattr(args, 'octree_init_level', -1)
            self.octree.set_coarse_interval(coarse_iter, coarse_factor, init_level)
            dist_str = {int(l): int((self._lod_level == l).sum()) for l in range(self.octree.levels)}
            print(f"[OctreeLOD] levels={self.octree.levels}  base_layer={self.octree.base_layer}  "
                  f"voxel_size={float(self.octree.voxel_size):.4f}  "
                  f"standard_dist={float(self.octree.standard_dist):.2f}  "
                  f"init_active={self.octree.init_level}  dist2level={self.octree.dist2level}  "
                  f"per_level(pre_shard)={dist_str}")

    def _build_octree_lod_points(self, pcd_points, pcd_colors, cameras, args):

        from scene.octree_lod import OctreeLOD

        with torch.no_grad():
            q_lo = torch.quantile(pcd_points, 0.01, dim=0)
            q_hi = torch.quantile(pcd_points, 0.99, dim=0)
            bbox_min = q_lo.clone()
            bbox_max = q_hi.clone()
            if args.gaussians_distribution and utils.DEFAULT_GROUP.size() > 1:
                dist.all_reduce(bbox_min, op=dist.ReduceOp.MIN)
                dist.all_reduce(bbox_max, op=dist.ReduceOp.MAX)

            self.octree = OctreeLOD(
                levels=getattr(args, 'octree_levels', -1),
                fork=getattr(args, 'octree_fork', 2),
                base_layer=getattr(args, 'octree_base_layer', -1),
                extend=getattr(args, 'octree_extend', 1.1),
                visible_threshold=getattr(args, 'octree_visible_threshold', 0.9),
                dist_ratio=getattr(args, 'octree_dist_ratio', 0.999),
                dist2level=getattr(args, 'octree_dist2level', 'round'),
            )

            cam_centers = torch.stack(
                [c.camera_center.float().cuda() for c in cameras], dim=0
            )
            self.octree.set_level(pcd_points, cam_centers, scales=(1.0,))
            self.octree.set_bbox_and_voxel_size(bbox_min, bbox_max)

            positions, levels, colors = self.octree.octree_sample(pcd_points, pcd_colors)
            positions, levels, mean_visible, mask = self.octree.auto_calibrate_and_weed(
                positions, levels
            )
            colors = colors[mask]
            print(f"[OctreeLOD] sampled={positions.shape[0]}  "
                  f"weed_threshold={self.octree.visible_threshold:.3f}  "
                  f"mean_visible={float(mean_visible):.3f}")

            self._lod_level = levels.to(torch.int8).contiguous()
            return positions.contiguous(), RGB2SH(colors).contiguous()

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        try:
            self.optimizer = SparseGaussianAdam(l, lr=0.0, eps=1e-15)
        except:
            self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l =['x', 'y', 'z', 'nx', 'ny', 'nz']
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        args = utils.get_args()
        group = utils.DEFAULT_GROUP
        if args.gaussians_distribution and not getattr(args, 'distributed_save', False):
            def gather_uneven_tensors(tensor):
                tensor_sizes = torch.zeros((group.size()), dtype=torch.int, device="cuda")
                tensor_sizes[group.rank()] = tensor.shape[0]
                dist.all_reduce(tensor_sizes, op=dist.ReduceOp.SUM)
                tensor_sizes = tensor_sizes.cpu().numpy().tolist()

                gathered_tensors =[]
                if group.rank() == 0:
                    for i in range(group.size()):
                        if i == group.rank():
                            gathered_tensors.append(tensor)
                        else:
                            tensor_from_rk_i = torch.zeros(
                                (tensor_sizes[i],) + tensor.shape[1:],
                                dtype=tensor.dtype,
                                device="cuda",
                            )
                            dist.recv(tensor_from_rk_i, src=i)
                            gathered_tensors.append(tensor_from_rk_i)
                    gathered_tensors = torch.cat(gathered_tensors, dim=0)
                else:
                    dist.send(tensor, dst=0)
                return gathered_tensors if group.rank() == 0 else tensor

            xyz = gather_uneven_tensors(self._xyz).detach().cpu().numpy()
            normals = np.zeros_like(xyz)
            f_dc = gather_uneven_tensors(self._features_dc).detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
            f_rest = gather_uneven_tensors(self._features_rest).detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
            opacities = gather_uneven_tensors(self._opacity).detach().cpu().numpy()
            scale = gather_uneven_tensors(self._scaling).detach().cpu().numpy()
            rotation = gather_uneven_tensors(self._rotation).detach().cpu().numpy()
            lod_level_np = None
            if self.octree is not None and self._lod_level.numel() > 0:
                lod_gathered = gather_uneven_tensors(self._lod_level.float().unsqueeze(1))
                if group.rank() == 0:
                    lod_level_np = lod_gathered.detach().cpu().numpy().astype(np.float32)

            if group.rank() != 0:
                return
        else:
            if not getattr(args, 'gaussians_distribution', False) and utils.DEFAULT_GROUP is not None and utils.DEFAULT_GROUP.size() > 1 and utils.DEFAULT_GROUP.rank() != 0:
                return
            xyz = self._xyz.detach().cpu().numpy()
            normals = np.zeros_like(xyz)
            f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
            f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
            opacities = self._opacity.detach().cpu().numpy()
            scale = self._scaling.detach().cpu().numpy()
            rotation = self._rotation.detach().cpu().numpy()
            lod_level_np = None
            if self.octree is not None and self._lod_level.numel() > 0:
                lod_level_np = self._lod_level.cpu().numpy().astype(np.float32).reshape(-1, 1)

        mkdir_p(os.path.dirname(path))

        attribute_names = self.construct_list_of_attributes()
        attributes_list = [xyz, normals, f_dc, f_rest, opacities, scale, rotation]
        if lod_level_np is not None:
            attribute_names = attribute_names + ['lod_level']
            attributes_list.append(lod_level_np)

        dtype_full = [(attr, 'f4') for attr in attribute_names]
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate(attributes_list, axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

        if self.octree is not None:
            torch.save(self.octree.state_dict(), os.path.join(os.path.dirname(path), "octree_state.pt"))

    def load_ply(self, path):
        plydata = PlyData.read(path)
        xyz = np.stack((np.asarray(plydata.elements[0]["x"]), np.asarray(plydata.elements[0]["y"]), np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").contiguous().requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        self.active_sh_degree = self.max_sh_degree

        args = utils.get_args()
        if getattr(args, 'use_octree_lod', False):
            from scene.octree_lod import OctreeLOD
            if 'lod_level' in [p.name for p in plydata.elements[0].properties]:
                lod_np = np.asarray(plydata.elements[0]['lod_level']).astype(np.int8)
                self._lod_level = torch.tensor(lod_np, dtype=torch.int8, device="cuda")
            else:
                self._lod_level = torch.empty(0, dtype=torch.int8, device="cuda")

            state_path = os.path.join(os.path.dirname(path), "octree_state.pt")
            if os.path.exists(state_path):
                state = torch.load(state_path, map_location="cpu")
                self.octree = OctreeLOD.from_state_dict(state)
            else:
                print(f"[OctreeLOD] {state_path} not found — LOD will be inactive "
                      "until re-initialized from cameras (save-time only).")
                self.octree = None

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]
                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state
                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        if self._ids is not None and self._ids.shape[0] == valid_points_mask.shape[0]:
            self._ids = self._ids[valid_points_mask]
            self._birth_iter = self._birth_iter[valid_points_mask]
        optimizable_tensors = self._prune_optimizer(valid_points_mask)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

        if hasattr(self, '_culling') and self._culling is not None:
            if self._culling.shape[0] == valid_points_mask.shape[0]:
                self._culling = self._culling[valid_points_mask]
                self.factor_culling = self.factor_culling[valid_points_mask]
            else:
                self._culling = None
                self.factor_culling = None

        if self.octree is not None and self._lod_level.shape[0] == valid_points_mask.shape[0]:
            self._lod_level = self._lod_level[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)
                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state
                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, parent_levels=None):
        self._extend_ids(new_xyz.shape[0])
        d = {"xyz": new_xyz, "f_dc": new_features_dc, "f_rest": new_features_rest, "opacity": new_opacities, "scaling" : new_scaling, "rotation" : new_rotation}
        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        if self.octree is not None:
            if parent_levels is not None:
                new_levels = parent_levels.to(self._lod_level.device).to(torch.int8)
            else:
                new_levels = torch.full(
                    (new_xyz.shape[0],), self.octree.levels - 1,
                    dtype=torch.int8, device=self._lod_level.device,
                )
            self._lod_level = torch.cat([self._lod_level, new_levels])

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * 0.01))
        for group in self.optimizer.param_groups:
            if group["name"] == "opacity":
                stored_state = self.optimizer.state.get(group["params"][0], None)
                if stored_state is not None:
                    stored_state["exp_avg"] = torch.zeros_like(opacities_new)
                    stored_state["exp_avg_sq"] = torch.zeros_like(opacities_new)
                    del self.optimizer.state[group["params"][0]]
                    group["params"][0] = nn.Parameter(opacities_new.requires_grad_(True))
                    self.optimizer.state[group["params"][0]] = stored_state
                else:
                    group["params"][0] = nn.Parameter(opacities_new.requires_grad_(True))
                self._opacity = group["params"][0]
                break

    @torch.no_grad()
    def reset_scaling(self, factor: float):
        if factor <= 0.0 or factor == 1.0:
            return
        log_factor = float(np.log(factor))
        for group in self.optimizer.param_groups:
            if group["name"] == "scaling":
                param = group["params"][0]
                param.data.add_(log_factor)
                stored_state = self.optimizer.state.get(param, None)
                if stored_state is not None:
                    if "exp_avg" in stored_state:
                        stored_state["exp_avg"].zero_()
                    if "exp_avg_sq" in stored_state:
                        stored_state["exp_avg_sq"].zero_()
                break

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0
        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)
        count_after_grow = self.get_xyz.shape[0]
        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)
        torch.cuda.empty_cache()
        return count_after_grow

    def densify_and_prune_topk(self, K_global, min_opacity, extent, max_screen_size, score_mode="grad"):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0
        grad_norm = torch.norm(grads, dim=-1)
        is_split = torch.max(self.get_scaling, dim=1).values > self.percent_dense * extent
        score = self._compute_topk_score(grads, score_mode)

        if K_global > 0:
            selected = self._global_topk_mask(score, K_global)
        else:
            selected = torch.zeros_like(score, dtype=torch.bool)

        clone_mask = selected & ~is_split
        split_mask = selected & is_split
        split_grad = grad_norm
        self._densify_and_clone_masked(clone_mask, grad_norm)
        n_after_clone = self._xyz.shape[0]
        if n_after_clone > split_mask.shape[0]:
            n_pad = n_after_clone - split_mask.shape[0]
            split_mask = torch.cat([split_mask, torch.zeros(n_pad, dtype=torch.bool, device=split_mask.device)])
            split_grad = torch.cat([split_grad, torch.zeros(n_pad, device=split_grad.device)])
        self._densify_and_split_masked(split_mask, split_grad)
        count_after_grow = self.get_xyz.shape[0]

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)
        torch.cuda.empty_cache()
        return count_after_grow

    def _compute_topk_score(self, grads, mode):
        g = torch.norm(grads, dim=-1)
        if mode == "grad":
            return g
        if mode == "grad_times_denom":
            return g * self.denom.squeeze(-1).clamp(min=1.0)
        if mode == "grad_times_opacity":
            return g * self.get_opacity.squeeze(-1)
        raise ValueError(f"unknown topk_score mode: {mode}")

    def _global_topk_mask(self, score, K_global):
        if utils.WORLD_SIZE <= 1:
            K = int(min(K_global, score.numel()))
            if K == 0:
                return torch.zeros_like(score, dtype=torch.bool)
            threshold = score.topk(K).values[-1]
            return score >= threshold

        finite = torch.isfinite(score)
        if finite.any():
            local_min = score[finite].min()
            local_max = score[finite].max()
        else:
            local_min = torch.tensor(0.0, device=score.device)
            local_max = torch.tensor(0.0, device=score.device)
        lo_t = local_min.detach().clone()
        hi_t = local_max.detach().clone()
        dist.all_reduce(lo_t, op=dist.ReduceOp.MIN)
        dist.all_reduce(hi_t, op=dist.ReduceOp.MAX)
        lo, hi = float(lo_t.item()), float(hi_t.item())
        if not (hi > lo):
            return torch.zeros_like(score, dtype=torch.bool)

        tol = max(1, K_global // 100)
        threshold = 0.5 * (lo + hi)
        for _ in range(24):
            threshold = 0.5 * (lo + hi)
            local_count = int((score >= threshold).sum().item())
            ct = torch.tensor([local_count], device=score.device, dtype=torch.long)
            dist.all_reduce(ct, op=dist.ReduceOp.SUM)
            global_count = int(ct.item())
            if abs(global_count - K_global) <= tol:
                break
            if global_count > K_global:
                lo = threshold
            else:
                hi = threshold
        return score >= threshold

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask, torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        selected_pts_mask = self._apply_spawn_gate(selected_pts_mask, torch.norm(grads, dim=-1), is_split=False)
        self._densify_and_clone_masked(selected_pts_mask, torch.norm(grads, dim=-1))

    def _densify_and_clone_masked(self, selected_pts_mask, grad_norm):
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        parent_levels = self._lod_level[selected_pts_mask] if self.octree is not None else None
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, parent_levels=parent_levels)

        if hasattr(self, '_culling') and self._culling is not None:
            new_culling = self._culling[selected_pts_mask]
            self._culling = torch.cat((self._culling, new_culling))
            new_factor_culling = self.factor_culling[selected_pts_mask]
            self.factor_culling = torch.cat((self.factor_culling, new_factor_culling))

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask, torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)
        selected_pts_mask = self._apply_spawn_gate(selected_pts_mask, padded_grad, is_split=True)
        self._densify_and_split_masked(selected_pts_mask, padded_grad, N=N)

    def _densify_and_split_masked(self, selected_pts_mask, padded_grad, N=2):
        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)

        if self.octree is not None:
            parent_levels = self._lod_level[selected_pts_mask].repeat(N)
            parent_levels = (parent_levels.long() + 1).clamp(0, self.octree.levels - 1).to(torch.int8)
        else:
            parent_levels = None
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, parent_levels=parent_levels)

        if hasattr(self, '_culling') and self._culling is not None:
            new_culling = self._culling[selected_pts_mask].repeat(N,1)
            self._culling = torch.cat((self._culling, new_culling))
            new_factor_culling = self.factor_culling[selected_pts_mask].repeat(N,1)
            self.factor_culling = torch.cat((self.factor_culling, new_factor_culling))

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def _alloc_culling(self, n_xyz, num_views):
        self._culling_num_views = int(num_views)
        n_bytes = (int(num_views) + 7) // 8
        self._culling = torch.zeros((n_xyz, n_bytes), dtype=torch.uint8, device='cuda')

    def _get_culling_view(self, uid):
        byte_idx = int(uid) >> 3
        bit_mask = 1 << (int(uid) & 7)
        return (self._culling[:, byte_idx] & bit_mask).bool()

    def _set_culling_view(self, uid, mask):
        byte_idx = int(uid) >> 3
        bit = 1 << (int(uid) & 7)
        col = self._culling[:, byte_idx]
        col &= (~bit) & 0xFF
        col |= mask.to(torch.uint8) * bit

    def init_culling(self, num_views):
        self._alloc_culling(self._xyz.shape[0], num_views)
        self.factor_culling=torch.ones((self._xyz.shape[0],1), device='cuda')

    def add_densification_stats_culling(self, viewspace_point_tensor, update_filter, factor):
        self.xyz_gradient_accum[update_filter] += (torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True) * factor[update_filter])
        self.denom[update_filter] += 1

    def culling_with_interesction_preserving(self, scene, render_simp, iteration, args, pipe, background):
        imp_score = torch.zeros(self._xyz.shape[0]).cuda()
        accum_area_max = torch.zeros(self._xyz.shape[0]).cuda()
        views = scene.getTrainCameras().copy()

        self._alloc_culling(self._xyz.shape[0], len(views))
        count_rad = torch.zeros((self._xyz.shape[0], 1)).cuda()
        count_vis = torch.zeros((self._xyz.shape[0], 1)).cuda()

        for view in views:
            render_pkg = render_simp(view, self, pipe, background, culling=self._get_culling_view(view.uid))
            accum_weights = render_pkg["accum_weights"]
            area_proj = render_pkg["area_proj"]
            area_max = render_pkg["area_max"]

            accum_area_max = accum_area_max + area_max
            if getattr(args, 'imp_metric', 'outdoor') == 'outdoor':
                mask_t = area_max != 0
                temp = imp_score + accum_weights / area_proj
                imp_score[mask_t] = temp[mask_t]
            else:
                imp_score = imp_score + accum_weights

            non_prune_mask = init_cdf_mask(importance=accum_weights, thres=0.99)
            self._set_culling_view(view.uid, non_prune_mask == False)
            count_rad[render_pkg["radii"] > 0] += 1
            count_vis[non_prune_mask] += 1

        imp_score[accum_area_max == 0] = 0
        if getattr(args, 'view_normalized_importance', False):
            imp_score = imp_score / (count_rad[:, 0] + 1e-6)
        non_prune_mask = init_cdf_mask(importance=imp_score,
                                       thres=getattr(args, 'simp2_cdf_thres', 0.99))
        self.factor_culling = count_vis / (count_rad + 1e-1)

        prune_mask = non_prune_mask == False
        if getattr(args, 'count_vis_prune', False):
            prune_mask = torch.logical_or(prune_mask, (count_vis <= 1)[:, 0])
        if not getattr(args, 'disable_simp2_prune', False):
            self.prune_points(prune_mask)

    def culling_with_interesction_sampling(self, scene, render_simp, iteration, args, pipe, background):
        imp_score = torch.zeros(self._xyz.shape[0]).cuda()
        accum_area_max = torch.zeros(self._xyz.shape[0]).cuda()
        views = scene.getTrainCameras().copy()

        self._alloc_culling(self._xyz.shape[0], len(views))
        count_rad = torch.zeros((self._xyz.shape[0], 1)).cuda()
        count_vis = torch.zeros((self._xyz.shape[0], 1)).cuda()

        for view in views:
            render_pkg = render_simp(view, self, pipe, background, culling=self._get_culling_view(view.uid))
            accum_weights = render_pkg["accum_weights"]
            area_proj = render_pkg["area_proj"]
            area_max = render_pkg["area_max"]

            accum_area_max = accum_area_max + area_max
            if getattr(args, 'imp_metric', 'outdoor') == 'outdoor':
                mask_t = area_max != 0
                temp = imp_score + accum_weights / area_proj
                imp_score[mask_t] = temp[mask_t]
            else:
                imp_score = imp_score + accum_weights

            non_prune_mask = init_cdf_mask(importance=accum_weights, thres=0.99)
            self._set_culling_view(view.uid, non_prune_mask == False)
            count_rad[render_pkg["radii"] > 0] += 1
            count_vis[non_prune_mask] += 1

        imp_score[accum_area_max == 0] = 0
        if getattr(args, 'view_normalized_importance', False):
            imp_score = imp_score / (count_rad[:, 0] + 1e-6)
        prob = imp_score / imp_score.sum()
        prob = prob.cpu().numpy()

        sampling_factor = getattr(args, 'sampling_factor', 0.8)
        N_xyz = self._xyz.shape[0]
        num_nonzero = int((prob != 0).sum())
        num_sampled = min(int(N_xyz * sampling_factor), num_nonzero)
        indices = np.random.choice(N_xyz, size=num_sampled, p=prob, replace=False)

        non_prune_mask = np.zeros(N_xyz, dtype=bool)
        non_prune_mask[indices] = True

        self.factor_culling = count_vis / (count_rad + 1e-1)

        prune_mask = torch.tensor(non_prune_mask == False, device='cuda')
        if getattr(args, 'count_vis_prune', False):
            prune_mask = torch.logical_or(prune_mask, (count_vis <= 1)[:, 0])
        if not getattr(args, 'disable_simp1_prune', False):
            self.prune_points(prune_mask)

    def clone(self, selected_pts_mask):
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]

        temp_opacity_old = self.get_opacity[selected_pts_mask]
        new_opacity = 1-(1-temp_opacity_old)**0.5

        temp_scale_old = self.get_scaling[selected_pts_mask]
        new_scaling = (temp_opacity_old / (2*new_opacity-0.5**0.5*new_opacity**2)) * temp_scale_old

        new_opacity = torch.clamp(new_opacity, max=1.0 - torch.finfo(torch.float32).eps, min=0.0051)
        new_opacity = self.inverse_opacity_activation(new_opacity)
        new_scaling = self.scaling_inverse_activation(new_scaling)   

        self._opacity[selected_pts_mask] = new_opacity
        self._scaling[selected_pts_mask] = new_scaling
        new_rotation = self._rotation[selected_pts_mask]

        parent_levels = self._lod_level[selected_pts_mask] if self.octree is not None else None
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, parent_levels=parent_levels)

        if hasattr(self, '_culling') and self._culling is not None:
            new_culling = self._culling[selected_pts_mask]
            self._culling = torch.cat((self._culling, new_culling))
            new_factor_culling = self.factor_culling[selected_pts_mask]
            self.factor_culling = torch.cat((self.factor_culling, new_factor_culling))

    def parameters(self):
        params =[]
        if hasattr(self, '_xyz') and self._xyz is not None: params.append(self._xyz)
        if hasattr(self, '_features_dc') and self._features_dc is not None: params.append(self._features_dc)
        if hasattr(self, '_features_rest') and self._features_rest is not None: params.append(self._features_rest)
        if hasattr(self, '_scaling') and self._scaling is not None: params.append(self._scaling)
        if hasattr(self, '_rotation') and self._rotation is not None: params.append(self._rotation)
        if hasattr(self, '_opacity') and self._opacity is not None: params.append(self._opacity)
        return iter(params)

    def log_gaussian_stats(self):
        num_3dgs = self._xyz.shape[0]
        avg_size = torch.mean(torch.max(self.get_scaling, dim=1).values).item()
        avg_opacity = torch.mean(self.get_opacity).item()
        stats = {"num_3dgs": num_3dgs, "avg_size": avg_size, "avg_opacity": avg_opacity}
        exp_avg_dict = {}
        exp_avg_sq_dict = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None and "exp_avg" in stored_state:
                exp_avg_dict[group["name"]] = torch.mean(torch.norm(stored_state["exp_avg"], dim=-1)).item()
                exp_avg_sq_dict[group["name"]] = torch.mean(torch.norm(stored_state["exp_avg_sq"], dim=-1)).item()
        return stats, exp_avg_dict, exp_avg_sq_dict

    def sync_gradients_for_replicated_3dgs_storage(self, batched_screenspace_pkg):
        args = utils.get_args()
        if "visible_count" in args.grad_normalization_mode:
            batched_locally_preprocessed_visibility_filter_int = [x.int() for x in batched_screenspace_pkg["batched_locally_preprocessed_visibility_filter"]]
            sum_batched_locally_preprocessed_visibility_filter_int = torch.sum(torch.stack(batched_locally_preprocessed_visibility_filter_int), dim=0)
            batched_screenspace_pkg["sum_batched_locally_preprocessed_visibility_filter_int"] = sum_batched_locally_preprocessed_visibility_filter_int

        if args.sync_grad_mode == "dense": sync_func = sync_gradients_densely
        elif args.sync_grad_mode == "sparse": sync_func = sync_gradients_sparsely
        elif args.sync_grad_mode == "fused_dense": sync_func = sync_gradients_fused_densely
        elif args.sync_grad_mode == "fused_sparse": sync_func = sync_gradients_fused_sparsely
        
        if not args.gaussians_distribution and utils.DEFAULT_GROUP.size() > 1:
            sync_func(self, utils.DEFAULT_GROUP)

    def group_for_redistribution(self):
        args = utils.get_args()
        if args.gaussians_distribution: return utils.DEFAULT_GROUP
        else: return utils.SingleGPUGroup()

    def all2all_gaussian_state(self, state, destination, i2j_send_size):
        comm_group = self.group_for_redistribution()
        state_to_gpuj = []
        state_from_gpuj =[]
        for j in range(comm_group.size()):
            state_to_gpuj.append(state[destination == j, ...].contiguous())
            state_from_gpuj.append(torch.zeros((i2j_send_size[j][comm_group.rank()], *state.shape[1:]), device="cuda", dtype=state.dtype))
        torch.distributed.all_to_all(state_from_gpuj, state_to_gpuj, group=comm_group)
        return torch.cat(state_from_gpuj, dim=0).contiguous()

    def all2all_tensors_in_optimizer(self, destination, i2j_send_size):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                if "exp_avg" not in stored_state:
                    stored_state["momentum_buffer"] = self.all2all_gaussian_state(stored_state["momentum_buffer"], destination, i2j_send_size)
                else:
                    stored_state["exp_avg"] = self.all2all_gaussian_state(stored_state["exp_avg"], destination, i2j_send_size)
                    stored_state["exp_avg_sq"] = self.all2all_gaussian_state(stored_state["exp_avg_sq"], destination, i2j_send_size)

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(self.all2all_gaussian_state(group["params"][0], destination, i2j_send_size), requires_grad=True)
                self.optimizer.state[group["params"][0]] = stored_state
                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(self.all2all_gaussian_state(group["params"][0], destination, i2j_send_size), requires_grad=True)
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def get_destination_1(self, world_size):
        return torch.arange(self.get_xyz.shape[0], device="cuda") % world_size

    def need_redistribute_gaussians(self, group):
        args = utils.get_args()
        if group.size() == 1: return False
        if utils.get_denfify_iter() == args.redistribute_anchors_frequency: return True
        
        local_n_gaussians = self.get_xyz.shape[0]
        all_local_n_gaussians =[None for _ in range(group.size())]
        torch.distributed.all_gather_object(all_local_n_gaussians, local_n_gaussians, group=group)
        if min(all_local_n_gaussians) * args.redistribute_gaussians_threshold < max(all_local_n_gaussians):
            return True
        return False

    def redistribute_gaussians(self):
        args = utils.get_args()
        if args.redistribute_anchor_mode == "no_redistribute": return

        comm_group_for_redistribution = self.group_for_redistribution()
        if not self.need_redistribute_gaussians(comm_group_for_redistribution): return

        destination = self.get_destination_1(comm_group_for_redistribution.size())
        local2j_send_size = torch.bincount(destination, minlength=comm_group_for_redistribution.size()).int()
        
        i2j_send_size = torch.zeros((comm_group_for_redistribution.size(), comm_group_for_redistribution.size()), dtype=torch.int, device="cuda")
        torch.distributed.all_gather_into_tensor(i2j_send_size, local2j_send_size, group=comm_group_for_redistribution)
        i2j_send_size = i2j_send_size.cpu().numpy().tolist()

        optimizable_tensors = self.all2all_tensors_in_optimizer(destination, i2j_send_size)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        if hasattr(self, '_culling') and self._culling is not None and self._culling.numel() > 0:
            self._culling = self.all2all_gaussian_state(self._culling, destination, i2j_send_size)
        if hasattr(self, 'factor_culling') and self.factor_culling is not None and self.factor_culling.numel() > 0:
            self.factor_culling = self.all2all_gaussian_state(self.factor_culling, destination, i2j_send_size)

        if self.octree is not None and self._lod_level.numel() > 0:
            lod_float = self._lod_level.to(torch.float32).unsqueeze(1)
            self._lod_level = self.all2all_gaussian_state(lod_float, destination, i2j_send_size).squeeze(1).to(torch.int8)

        if self._ids is not None and self._ids.numel() > 0:
            self._ids = self.all2all_gaussian_state(
                self._ids.unsqueeze(1), destination, i2j_send_size).squeeze(1)
            self._birth_iter = self.all2all_gaussian_state(
                self._birth_iter.unsqueeze(1), destination, i2j_send_size).squeeze(1)

        self.send_to_gpui_cnt = torch.zeros((self.get_xyz.shape[0], comm_group_for_redistribution.size()), dtype=torch.int, device="cuda")
        torch.cuda.empty_cache()

    def get_lod_indices(self, camera_center: torch.Tensor, iteration: int = -1):
        """Return an int64 index tensor of Gaussians to render, or None if LOD is inactive.

        Computes a fresh per-camera mask every call — no caching.
        LOD is active from iteration 0 with progressive level unlocking.
        Returns None if the filter ratio is below `octree_min_filter_ratio`,
        so we don't pay LOD overhead for negligible savings.
        """
        if self.octree is None or self._lod_level.numel() == 0:
            return None

        args = utils.get_args()
        n_total = self._xyz.shape[0]

        with torch.no_grad():
            bool_mask = self.octree.compute_lod_mask(
                camera_center,
                self._xyz.detach(),
                self._lod_level,
                iteration=iteration,
            )
            n_kept = int(bool_mask.sum().item())

        min_ratio = getattr(args, 'octree_min_filter_ratio', 0.20)
        if n_kept >= n_total * (1.0 - min_ratio):
            self._lod_last_kept = n_kept
            self._lod_last_total = n_total
            return None

        with torch.no_grad():
            indices = bool_mask.nonzero(as_tuple=False).squeeze(1)

        self._lod_last_kept = n_kept
        self._lod_last_total = n_total
        self._lod_last_active_level = self.octree.get_active_level(iteration)

        return indices


def get_sparse_ids(tensors):
    sparse_ids = None
    with torch.no_grad():
        for tensor in tensors:
            nonzero_indices = torch.nonzero(tensor)
            row_indices = nonzero_indices[:, 0]
            if sparse_ids is None:
                sparse_ids = row_indices
            else:
                sparse_ids = torch.cat((sparse_ids, row_indices))
        sparse_ids = torch.unique(sparse_ids, sorted=True)
        return sparse_ids

def sync_gradients_sparsely(gaussians, group):
    with torch.no_grad():
        sparse_ids = get_sparse_ids([gaussians._xyz.grad.data])
        sparse_ids_mask = torch.zeros((gaussians._xyz.shape[0]), dtype=torch.bool, device="cuda")
        sparse_ids_mask[sparse_ids] = True
        torch.distributed.all_reduce(sparse_ids_mask, op=dist.ReduceOp.SUM, group=group)

        def sync_grads(data):
            sparse_grads = data.grad.data[sparse_ids_mask].contiguous()
            torch.distributed.all_reduce(sparse_grads, op=dist.ReduceOp.SUM, group=group)
            data.grad.data[sparse_ids_mask] = sparse_grads

        sync_grads(gaussians._xyz)
        sync_grads(gaussians._features_dc)
        sync_grads(gaussians._features_rest)
        sync_grads(gaussians._opacity)
        sync_grads(gaussians._scaling)
        sync_grads(gaussians._rotation)

    log_file = utils.get_log_file()
    if log_file:
        non_zero_indices_cnt = sparse_ids_mask.sum().item()
        total_indices_cnt = sparse_ids_mask.shape[0]
        log_file.write(
            "iterations:[{}, {}) non_zero_indices_cnt: {} total_indices_cnt: {} ratio: {}\n".format(
                utils.get_cur_iter(), utils.get_cur_iter() + utils.get_args().bsz,
                non_zero_indices_cnt, total_indices_cnt, non_zero_indices_cnt / total_indices_cnt,
            )
        )

def sync_gradients_densely(gaussians, group):
    with torch.no_grad():
        def sync_grads(data):
            torch.distributed.all_reduce(data.grad.data, op=dist.ReduceOp.SUM, group=group)
        sync_grads(gaussians._xyz)
        sync_grads(gaussians._features_dc)
        sync_grads(gaussians._features_rest)
        sync_grads(gaussians._opacity)
        sync_grads(gaussians._scaling)
        sync_grads(gaussians._rotation)

def sync_gradients_fused_densely(gaussians, group):
    with torch.no_grad():
        all_params_grads =[
            param.grad.data for param in[
                gaussians._xyz, gaussians._features_dc, gaussians._features_rest,
                gaussians._opacity, gaussians._scaling, gaussians._rotation,
            ]
        ]
        all_params_grads_dim1 = [param_grad.shape[1] for param_grad in all_params_grads]
        catted_params_grads = torch.cat(all_params_grads, dim=1).contiguous()
        torch.distributed.all_reduce(catted_params_grads, op=dist.ReduceOp.SUM, group=group)
        split_params_grads = torch.split(catted_params_grads, all_params_grads_dim1, dim=1)
        for param_grad, split_param_grad in zip(all_params_grads, split_params_grads):
            param_grad.copy_(split_param_grad)

def sync_gradients_fused_sparsely(gaussians, group):
    raise NotImplementedError("Fused sparse sync gradients is not implemented yet.")