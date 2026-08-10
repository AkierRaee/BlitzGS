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
import torch
import sys
import json
from utils.general_utils import safe_state, init_distributed
import utils.general_utils as utils
from argparse import ArgumentParser
from arguments import (
    AuxiliaryParams,
    ModelParams,
    PipelineParams,
    OptimizationParams,
    DistributionParams,
    BenchmarkParams,
    DebugParams,
    print_all_args,
    init_args,
)
import train_internal
import debugpy
if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    ap = AuxiliaryParams(parser)
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    dist_p = DistributionParams(parser)
    bench_p = BenchmarkParams(parser)
    debug_p = DebugParams(parser)
    args = parser.parse_args(sys.argv[1:])

    rank = int(os.environ.get("LOCAL_RANK", 0))

    init_distributed(args)

    init_args(args)

    args = utils.get_args()

    if utils.GLOBAL_RANK == 0:
        os.makedirs(args.log_folder, exist_ok=True)
        os.makedirs(args.model_path, exist_ok=True)
    if utils.WORLD_SIZE > 1:
        torch.distributed.barrier(
            group=utils.DEFAULT_GROUP
        )
    if utils.GLOBAL_RANK == 0:
        with open(args.log_folder + "/args.json", "w") as f:
            json.dump(vars(args), f)

    safe_state(args.quiet)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    log_file = open(
        args.log_folder
        + "/python_ws="
        + str(utils.WORLD_SIZE)
        + "_rk="
        + str(utils.GLOBAL_RANK)
        + ".log",
        "a" if args.auto_start_checkpoint else "w",
    )
    utils.set_log_file(log_file)
    print_all_args(args, log_file)

    train_internal.training(
        lp.extract(args), op.extract(args), pp.extract(args), args, log_file,
    )

    if utils.WORLD_SIZE > 1:
        torch.distributed.barrier(group=utils.DEFAULT_GROUP)
    utils.print_rank_0("\nTraining complete.")
