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

from pathlib import Path
import os
from PIL import Image
import torch
import torch.multiprocessing as tmp
import torchvision.transforms.functional as tf
from utils.loss_utils import ssim
from utils.general_utils import set_args
from lpipsPyTorch import lpips
import json
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser
from torchvision import transforms

def _worker_metrics(args_tuple):
    """Compute SSIM/PSNR/LPIPS for a subset of images on one GPU."""
    device_id, fnames, renders_dir, gt_dir = args_tuple
    device = torch.device(f"cuda:{device_id}")
    torch.cuda.set_device(device)

    results = []
    for fname in fnames:
        render = Image.open(Path(renders_dir) / fname)
        gt    = Image.open(Path(gt_dir)    / fname)
        resize_transform = transforms.Resize(
            (int(render.height / 1.2), int(render.width / 1.2)),
            interpolation=transforms.InterpolationMode.BILINEAR,
        )
        render = resize_transform(render)
        gt     = resize_transform(gt)

        r = tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].to(device)
        g = tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].to(device)

        s = float(ssim(r, g))
        p = float(psnr(r, g))
        l = float(lpips(r, g, net_type="vgg"))
        results.append((fname, s, p, l))
    return results


def readImages(renders_dir, gt_dir):
    print("Reading images from", renders_dir)
    renders = []
    gts = []
    image_names = []

    for fname in os.listdir(renders_dir):
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)
        resize_transform = transforms.Resize(
            (int(render.height / 1.2), int(render.width / 1.2)),
            interpolation=transforms.InterpolationMode.BILINEAR
        )
        render = resize_transform(render)
        gt = resize_transform(gt)

        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :])
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :])
        image_names.append(fname)
    return renders, gts, image_names


def evaluate(model_paths, mode, num_gpus=1):

    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    print("")
    txt_path = './metrix.txt'
    for scene_dir in model_paths:
        print("Scene:", scene_dir)
        full_dict[scene_dir] = {}
        per_view_dict[scene_dir] = {}
        full_dict_polytopeonly[scene_dir] = {}
        per_view_dict_polytopeonly[scene_dir] = {}

        test_dir = Path(scene_dir) / mode

        for method in os.listdir(test_dir):
            print("Method:", method)

            full_dict[scene_dir][method] = {}
            per_view_dict[scene_dir][method] = {}
            full_dict_polytopeonly[scene_dir][method] = {}
            per_view_dict_polytopeonly[scene_dir][method] = {}

            method_dir = test_dir / method
            gt_dir      = method_dir / "gt"
            renders_dir = method_dir / "renders"

            fnames = sorted(os.listdir(renders_dir))
            print(f"Number of images: {len(fnames)}  |  GPUs: {num_gpus}")

            if not fnames:
                print("  [SKIP] no renders found, skipping metrics")
                continue

            if num_gpus > 1:
                chunks = [fnames[i::num_gpus] for i in range(num_gpus)]
                worker_args = [
                    (i, chunk, str(renders_dir), str(gt_dir))
                    for i, chunk in enumerate(chunks)
                    if chunk
                ]

                ctx = tmp.get_context("spawn")
                chunk_results = []
                with ctx.Pool(len(worker_args)) as pool:
                    with tqdm(total=len(fnames), desc="Metric evaluation progress") as pbar:
                        for result in pool.imap_unordered(_worker_metrics, worker_args):
                            chunk_results.append(result)
                            pbar.update(len(result))

                flat = [item for chunk in chunk_results for item in chunk]
                flat.sort(key=lambda x: x[0])

                image_names = [r[0] for r in flat]
                ssims  = [torch.tensor(r[1]) for r in flat]
                psnrs  = [torch.tensor(r[2]) for r in flat]
                lpipss = [torch.tensor(r[3]) for r in flat]

                with open(txt_path, "a") as f:
                    for name, s, p, l in zip(image_names, ssims, psnrs, lpipss):
                        f.write(f"ssim:{float(s)},psnr:{float(p)},lpips:{float(l)}img:{name}\n")

            else:
                device = torch.device("cuda:0")
                renders, gts, image_names = readImages(renders_dir, gt_dir)
                ssims  = []
                psnrs  = []
                lpipss = []

                for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
                    r = renders[idx].to(device)
                    g = gts[idx].to(device)
                    ssims.append(ssim(r, g))
                    psnrs.append(psnr(r, g))
                    lpipss.append(lpips(r, g, net_type="vgg"))
                    with open(txt_path, "a") as f:
                        f.write(
                            f"ssim:{float(ssims[idx])},psnr:{float(psnrs[idx])},lpips:{float(lpipss[idx])}img:{image_names[idx]}\n"
                        )

            print("  SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
            print("  PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
            print("  LPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))
            print("")

            full_dict[scene_dir][method].update(
                {
                    "SSIM":  torch.tensor(ssims).mean().item(),
                    "PSNR":  torch.tensor(psnrs).mean().item(),
                    "LPIPS": torch.tensor(lpipss).mean().item(),
                }
            )
            per_view_dict[scene_dir][method].update(
                {
                    "SSIM":  {name: float(s) for s, name in zip(ssims,  image_names)},
                    "PSNR":  {name: float(p) for p, name in zip(psnrs,  image_names)},
                    "LPIPS": {name: float(l) for l, name in zip(lpipss, image_names)},
                }
            )

            with open(scene_dir + f"/results_{mode}.json", "w") as fp:
                json.dump(full_dict[scene_dir], fp, indent=True)
            with open(scene_dir + f"/per_view_{mode}.json", "w") as fp:
                json.dump(per_view_dict[scene_dir], fp, indent=True)


if __name__ == "__main__":
    tmp.set_start_method("spawn", force=True)

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument("--model_paths", "-m", required=True, nargs="+", type=str, default=[])
    parser.add_argument(
        "--mode", type=str, choices=["train", "test"], default="test",
        help="train or test",
    )
    parser.add_argument(
        "--num_gpus", type=int, default=1,
        help="Number of GPUs to use for parallel metric computation",
    )
    args = parser.parse_args()

    set_args(args)
    evaluate(args.model_paths, args.mode, num_gpus=args.num_gpus)
