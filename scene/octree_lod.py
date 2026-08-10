
import math

import torch


def _safe_quantile(t: torch.Tensor, q: float) -> torch.Tensor:
    MAX = 16_000_000
    if t.numel() > MAX:
        idx = torch.randint(0, t.numel(), (MAX,), device=t.device)
        t = t.flatten()[idx]
    return torch.quantile(t, q)


class OctreeLOD:

    def __init__(
        self,
        levels: int = -1,
        fork: int = 2,
        base_layer: int = -1,
        extend: float = 1.1,
        visible_threshold: float = 0.9,
        dist_ratio: float = 0.999,
        dist2level: str = "round",
        padding: float = 0.0,
    ):
        self.levels = int(levels)
        self.fork = int(fork)
        self.base_layer = int(base_layer)
        self.extend = float(extend)
        self.visible_threshold = float(visible_threshold)
        self.dist_ratio = float(dist_ratio)
        self.dist2level = str(dist2level)
        self.padding = float(padding)

        self.standard_dist = None
        self.voxel_size = None
        self.init_pos = None
        self.bbox_min = None
        self.bbox_max = None
        self.cam_infos = None

        self._active_level = 0
        self.init_level = 0
        self.coarse_intervals = []

    def set_level(self, points: torch.Tensor, cam_centers: torch.Tensor, scales=(1.0,)):
        device = points.device
        points = points.to(device).float()
        cam_centers = cam_centers.to(device).float()

        cam_infos = []
        all_dist = []
        for scale in scales:
            s = float(scale)
            for cc in cam_centers:
                cam_infos.append(torch.tensor(
                    [cc[0].item(), cc[1].item(), cc[2].item(), s],
                    device=device, dtype=torch.float,
                ))
                dist = torch.sqrt(torch.sum((points - cc) ** 2, dim=1))
                dist_max = _safe_quantile(dist, self.dist_ratio)
                dist_min = _safe_quantile(dist, 1.0 - self.dist_ratio)
                all_dist.append(dist_min * s)
                all_dist.append(dist_max * s)
        self.cam_infos = torch.stack(cam_infos, dim=0)

        all_dist = torch.stack(all_dist)
        dist_max = _safe_quantile(all_dist, self.dist_ratio)
        dist_min = _safe_quantile(all_dist, 1.0 - self.dist_ratio)
        self.standard_dist = dist_max.detach().clone()

        if self.levels == -1:
            span = (dist_max / dist_min.clamp(min=1e-8)).item()
            self.levels = max(1, int(round(math.log2(span) / math.log2(self.fork))) + 1)

    def set_bbox_and_voxel_size(self, bbox_min: torch.Tensor, bbox_max: torch.Tensor):
        device = bbox_min.device
        self.bbox_min = bbox_min.detach().float().to(device)
        self.bbox_max = bbox_max.detach().float().to(device)

        extent = (self.bbox_max - self.bbox_min)
        box_d = extent.max() * self.extend
        self.init_pos = (self.bbox_min - 0.5 * extent * (self.extend - 1.0)).contiguous()

        if self.base_layer < 0:
            default_voxel_size = max(0.02, box_d.item() * 0.001)
            self.base_layer = int(round(math.log2(box_d.item() / default_voxel_size))) \
                - (self.levels // 2) + 1
        self.voxel_size = box_d / (float(self.fork) ** self.base_layer)


    def octree_sample(self, points: torch.Tensor, colors: torch.Tensor):
        assert self.voxel_size is not None, "call set_bbox_and_voxel_size() first"
        device = points.device
        init_pos = self.init_pos.to(device)

        pos_list = []
        lvl_list = []
        col_list = []
        for cur_level in range(self.levels):
            cur_size = self.voxel_size / (float(self.fork) ** cur_level)
            cells = torch.round((points - init_pos) / cur_size)
            unique_cells, inverse = torch.unique(cells, dim=0, return_inverse=True)
            new_positions = unique_cells * cur_size + init_pos + self.padding * cur_size
            new_levels = torch.full(
                (new_positions.shape[0],), cur_level, dtype=torch.int32, device=device
            )
            n_unique = unique_cells.shape[0]
            color_sum = torch.zeros(n_unique, colors.shape[1], device=device, dtype=colors.dtype)
            color_sum.index_add_(0, inverse, colors)
            count = torch.bincount(inverse, minlength=n_unique).clamp(min=1).unsqueeze(1).float()
            new_colors = color_sum / count

            pos_list.append(new_positions)
            lvl_list.append(new_levels)
            col_list.append(new_colors)

        positions = torch.cat(pos_list, dim=0)
        levels = torch.cat(lvl_list, dim=0)
        out_colors = torch.cat(col_list, dim=0)
        return positions, levels, out_colors


    def weed_out(self, positions: torch.Tensor, levels: torch.Tensor):
        assert self.cam_infos is not None, "call set_level() first to populate cameras"
        device = positions.device
        visible_count = torch.zeros(positions.shape[0], device=device, dtype=torch.float)
        for cam in self.cam_infos:
            cam_center = cam[:3]
            scale = cam[3]
            dist = torch.sqrt(torch.sum((positions - cam_center) ** 2, dim=1)) * scale
            pred_level = torch.log2(self.standard_dist / dist.clamp(min=1e-8)) / math.log2(self.fork)
            int_level = self.map_to_int_level(pred_level, self.levels - 1)
            visible_count += (levels <= int_level).float()
        visible_count = visible_count / max(1, len(self.cam_infos))
        mean_visible = visible_count.mean()
        mask = visible_count > self.visible_threshold
        return positions[mask], levels[mask], mean_visible, mask

    def auto_calibrate_and_weed(self, positions: torch.Tensor, levels: torch.Tensor):
        if self.visible_threshold < 0:
            _, _, mean_visible, _ = self.weed_out(positions, levels)
            self.visible_threshold = float(mean_visible.item())
        return self.weed_out(positions, levels)


    def predict_level(self, xyz: torch.Tensor, cam_center: torch.Tensor) -> torch.Tensor:
        dist = torch.sqrt(torch.sum((xyz - cam_center.unsqueeze(0)) ** 2, dim=1)).clamp(min=1e-8)
        return torch.log2(self.standard_dist / dist) / math.log2(self.fork)

    def map_to_int_level(self, pred_level: torch.Tensor, cur_level: int) -> torch.Tensor:
        if self.dist2level == "floor":
            int_level = torch.floor(pred_level).int()
        elif self.dist2level == "ceil":
            int_level = torch.ceil(pred_level).int()
        elif self.dist2level in ("round", "progressive"):
            int_level = torch.round(pred_level).int()
        else:
            int_level = torch.round(pred_level).int()
        return torch.clamp(int_level, min=0, max=int(cur_level))


    def set_coarse_interval(self, coarse_iter: int, coarse_factor: float = 1.5, init_level: int = -1):
        if init_level < 0:
            self.init_level = max(0, self.levels // 2)
        else:
            self.init_level = min(int(init_level), self.levels - 1)
        self._active_level = self.init_level
        self.coarse_intervals = []

        num_level = self.levels - 1 - self.init_level
        if num_level <= 0:
            self._active_level = self.levels - 1
            return

        q = 1.0 / float(coarse_factor)
        denom = 1.0 - q ** num_level
        a1 = coarse_iter * (1.0 - q) / max(denom, 1e-12)
        cumulative = 0.0
        for i in range(num_level):
            cumulative += a1 * (q ** i)
            self.coarse_intervals.append(cumulative)

    def get_active_level(self, iteration: int) -> int:
        if not self.coarse_intervals:
            return max(0, self.levels - 1)
        unlocked = 0
        for threshold in self.coarse_intervals:
            if iteration >= threshold:
                unlocked += 1
            else:
                break
        return min(self.init_level + unlocked, self.levels - 1)


    def compute_lod_mask(
        self,
        camera_center: torch.Tensor,
        xyz: torch.Tensor,
        lod_levels: torch.Tensor,
        iteration: int = -1,
        resolution_scale: float = 1.0,
    ) -> torch.Tensor:
        active_level = self.get_active_level(iteration)
        if self.standard_dist is None:
            return torch.ones(xyz.shape[0], dtype=torch.bool, device=xyz.device)

        if self.dist2level == "progressive":
            return lod_levels.long() <= active_level

        dist = torch.sqrt(torch.sum((xyz - camera_center.unsqueeze(0)) ** 2, dim=1)).clamp(min=1e-8)
        dist = dist * float(resolution_scale)
        pred_level = torch.log2(self.standard_dist / dist) / math.log2(self.fork)
        int_level = self.map_to_int_level(pred_level, active_level)
        return lod_levels.long() <= int_level.long()


    def state_dict(self) -> dict:
        return {
            "levels": self.levels,
            "fork": self.fork,
            "base_layer": self.base_layer,
            "extend": self.extend,
            "visible_threshold": self.visible_threshold,
            "dist_ratio": self.dist_ratio,
            "dist2level": self.dist2level,
            "padding": self.padding,
            "standard_dist": None if self.standard_dist is None else self.standard_dist.detach().cpu(),
            "voxel_size": None if self.voxel_size is None else (
                self.voxel_size.detach().cpu() if torch.is_tensor(self.voxel_size)
                else torch.tensor(float(self.voxel_size))
            ),
            "init_pos": None if self.init_pos is None else self.init_pos.detach().cpu(),
            "bbox_min": None if self.bbox_min is None else self.bbox_min.detach().cpu(),
            "bbox_max": None if self.bbox_max is None else self.bbox_max.detach().cpu(),
            "_active_level": self._active_level,
            "init_level": self.init_level,
            "coarse_intervals": self.coarse_intervals,
        }

    @classmethod
    def from_state_dict(cls, d: dict) -> "OctreeLOD":
        obj = cls(
            levels=d.get("levels", -1),
            fork=d.get("fork", 2),
            base_layer=d.get("base_layer", -1),
            extend=d.get("extend", 1.1),
            visible_threshold=d.get("visible_threshold", 0.9),
            dist_ratio=d.get("dist_ratio", 0.999),
            dist2level=d.get("dist2level", "round"),
            padding=d.get("padding", 0.0),
        )

        def _to_cuda(t):
            return None if t is None else t.float().cuda()

        obj.standard_dist = _to_cuda(d.get("standard_dist"))
        vs = d.get("voxel_size")
        obj.voxel_size = _to_cuda(vs) if vs is not None else None
        obj.init_pos = _to_cuda(d.get("init_pos"))
        obj.bbox_min = _to_cuda(d.get("bbox_min"))
        obj.bbox_max = _to_cuda(d.get("bbox_max"))
        obj._active_level = d.get("_active_level", max(0, obj.levels - 1))
        obj.init_level = d.get("init_level", 0)
        obj.coarse_intervals = d.get("coarse_intervals", [])
        return obj
