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

from argparse import ArgumentParser, Namespace
import sys
import os
import utils.general_utils as utils

class GroupParams:
    pass


class ParamGroup:
    def __init__(self, parser: ArgumentParser, name: str, fill_none=False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None
            if shorthand:
                if t == bool:
                    group.add_argument(
                        "--" + key, ("-" + key[0:1]), default=value, action="store_true"
                    )
                else:
                    group.add_argument(
                        "--" + key, ("-" + key[0:1]), default=value, type=t
                    )
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                elif t == list:
                    type_to_use = int
                    if value is not None and len(value) > 0:
                        type_to_use = type(value[0])
                    group.add_argument(
                        "--" + key, default=value, nargs="+", type=type_to_use
                    )
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group


class AuxiliaryParams(ParamGroup):
    def __init__(self, parser, sentinel=False):
        self.debug_from = -1
        self.detect_anomaly = False
        self.test_iterations = [10_000, 20_000, 30_000, 40_000, 50_000, 100_000, 150_000, 200_000, 250_000]
        self.save_iterations = [10_000, 20_000, 30_000, 40_000, 50_000, 100_000, 150_000, 200_000, 250_000]
        self.quiet = False
        self.checkpoint_iterations = []
        self.start_checkpoint = ""
        self.auto_start_checkpoint = False
        self.log_folder = "/tmp/gaussian_splatting"
        self.log_interval = 250
        self.llffhold = 83
        self.backend = "default"
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        return g


class ModelParams(ParamGroup):
    def __init__(self, parser, sentinel=False):

        self.multi_view_num = 8
        self.multi_view_max_angle = 30
        self.multi_view_min_dis = 0.01
        self.multi_view_max_dis = 1.5

        self.sh_degree = 3
        self._source_path = ""
        self._model_path = "/tmp/gaussian_splatting"
        self._images = "images"
        self._white_background = False
        self.eval = False
        self.resolution_scales = [1.0]
        self._resolution = 1
        self.data_device = "cuda"
        self.ds = 1
        self.ratio = 1
        self.undistorted = False 

        self.load_iteration = None

        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g


class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.compute_cov3D_python = False
        self.convert_SHs_python = False
        self.debug = False
        super().__init__(parser, "Pipeline Parameters")


class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 100_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = self.iterations//2


        self.offset_lr_init = 0.01
        self.offset_lr_final = 0.0001
        self.offset_lr_delay_mult = 0.01
        self.offset_lr_max_steps = self.iterations//2

        self.feature_lr = 0.0075
        self.opacity_lr = 0.02
        self.scaling_lr = 0.007
        self.rotation_lr = 0.002

        self.update_interval = 100
        self.update_until = 30000


        self.lr_scale_loss = 1.0
        self.lr_scale_pos_and_scale = 1.0
        self.rotation_lr = 0.001
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.densification_interval = 1000
        self.opacity_reset_interval = 3000
        self.scale_reset_factor = 0.0
        self.densify_from_iter = 2000
        self.densify_until_iter = self.iterations//2
        self.densify_grad_threshold = 0.0002
        self.densify_memory_limit_percentage = 0.9
        self.densify_memory_start_percentage = 0.7
        self.disable_auto_densification = False
        self.opacity_reset_until_iter = -1
        self.random_background = False
        self.min_opacity = 0.005
        self.lr_scale_mode = "sqrt"


        self.wo_image_weight = False
        self.scale_loss_from_iter = 0
        self.default_voxel_size = 0.0001

        self.use_octree_lod = True
        self.octree_levels = -1
        self.octree_fork = 2
        self.octree_base_layer = -1
        self.octree_extend = 1.1
        self.octree_visible_threshold = 0.9
        self.octree_dist_ratio = 0.999
        self.octree_dist2level = "round"
        self.octree_coarse_factor = 1.5
        self.octree_init_level = -1
        self.octree_min_filter_ratio = 0.20
        self.octree_diag_interval = 500

        self.use_ms_culling = True
        self.ms_culling_in_render = True
        self.simp_iteration1 = 16_000
        self.simp_iteration2 = 50_000
        self.imp_metric = "outdoor"
        self.sampling_factor = 0.8
        self.count_vis_prune = False
        self.disable_phi_weighting = False
        self.disable_simp1_prune = False
        self.disable_simp2_prune = False
        self.view_normalized_importance = False
        self.simp2_cdf_thres = 0.99
        self.spawn_gate = ""
        self.spawn_gate_k = 0.4
        self.spawn_gate_start = 16000

        self.use_topk_densify = False
        self.birth_rate = 0.05
        self.birth_schedule = "cosine"
        self.topk_score = "grad"

        self.use_error_sampling = False
        self.residual_ema = 0.9
        self.sampling_alpha_final = 1.0
        self.sampling_alpha_warmup_iters = 2000
        self.sampling_weight_cap = 4.0
        self.residual_metric = "l1"

        super().__init__(parser, "Optimization Parameters")


class DistributionParams(ParamGroup):
    def __init__(self, parser):
        self.image_distribution = True
        self.image_distribution_mode = "final"
        self.heuristic_decay = 0.0
        self.no_heuristics_update = False
        self.border_divpos_coeff = 1.0
        self.adjust_strategy_warmp_iterations = -1
        self.save_strategy_history = False

        self.gaussians_distribution = True
        self.redistribute_anchor_mode = "random_redistribute"
        self.redistribute_anchors_frequency = (
            10
        )
        self.redistribute_gaussians_threshold = (
            1.1
        )
        self.sync_grad_mode = "dense"
        self.grad_normalization_mode = "none"

        self.bsz = 1
        self.distributed_dataset_storage = True
        self.distributed_save = False
        self.local_sampling = False
        self.preload_dataset_to_gpu = (
            False
        )
        self.preload_dataset_to_gpu_threshold = (
            3
        )
        self.multiprocesses_image_loading = True
        self.num_train_cameras = -1
        self.num_test_cameras = -1


        super().__init__(parser, "Distribution Parameters")


class BenchmarkParams(ParamGroup):
    def __init__(self, parser):
        self.enable_timer = False
        self.end2end_time = False
        self.check_gpu_memory = False
        self.check_cpu_memory = False
        self.log_memory_summary = False

        super().__init__(parser, "Benchmark Parameters")


class DebugParams(ParamGroup):
    def __init__(self, parser):
        self.stop_update_param = (
            False
        )
        self.time_image_loading = False

        self.nsys_profile = False
        self.drop_initial_3dgs_p = 0.0
        self.drop_duplicate_gaussians_coeff = 1.0

        super().__init__(parser, "Debug Parameters")


def get_combined_args(parser: ArgumentParser, auto_find_cfg_args_path=False):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        if auto_find_cfg_args_path:
            if hasattr(args_cmdline, "load_ply_path"):
                path = args_cmdline.load_ply_path
                while not os.path.exists(
                    os.path.join(path, "cfg_args")
                ) and os.path.exists(path):
                    path = os.path.join(path, "..")
                cfgfilepath = os.path.join(path, "cfg_args")
        else:
            cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k, v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)


def print_all_args(args, log_file):
    log_file.write("arguments:\n")
    log_file.write("-" * 30 + "\n")
    for arg in vars(args):
        log_file.write("{}: {}\n".format(arg, getattr(args, arg)))
    log_file.write("-" * 30 + "\n\n")
    log_file.write(
        "world_size: "
        + str(utils.WORLD_SIZE)
        + " rank: "
        + str(utils.GLOBAL_RANK)
        + "; bsz: "
        + str(args.bsz)
        + "\n"
    )


def find_latest_checkpoint(log_folder):
    checkpoint_folder = os.path.join(log_folder, "checkpoints")
    if os.path.exists(checkpoint_folder):
        all_sub_folders = os.listdir(checkpoint_folder)
        if len(all_sub_folders) > 0:
            all_sub_folders.sort(key=lambda x: int(x), reverse=True)
            return os.path.join(checkpoint_folder, all_sub_folders[0])
    return ""


def init_args(args):

    if args.opacity_reset_until_iter == -1:
        args.opacity_reset_until_iter = args.densify_until_iter + args.bsz

    args.log_folder = args.model_path

    if args.auto_start_checkpoint:
        args.start_checkpoint = find_latest_checkpoint(args.log_folder)

    if utils.DEFAULT_GROUP.size() == 1:
        args.gaussians_distribution = False
        args.image_distribution = False
        args.image_distribution_mode = ""
        args.distributed_dataset_storage = False
        args.distributed_save = False
        args.local_sampling = False

    if args.preload_dataset_to_gpu:
        args.distributed_dataset_storage = False
        args.local_sampling = False

    if args.local_sampling:
        assert args.distributed_dataset_storage, "local_sampling works only when distributed_dataset_storage==True"

    if not args.gaussians_distribution:
        args.distributed_save = False

    args.test_iterations.sort()
    args.save_iterations.sort()
    if len(args.save_iterations) > 0 and args.iterations not in args.save_iterations:
        args.save_iterations.append(args.iterations)
    args.checkpoint_iterations.sort()

    utils.set_args(args)
